"""Kimi Delta Attention with Kimi K3's smooth lower-bounded decay."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import K3MiniPlusPlusPlusConfig as K3MiniConfig
from layers import RMSNorm

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    FLA_AVAILABLE = True
except Exception:
    chunk_kda = fused_recurrent_kda = None
    FLA_AVAILABLE = False


@dataclass
class KDACache:
    recurrent_state: torch.Tensor | None = None
    q_conv_state: torch.Tensor | None = None
    k_conv_state: torch.Tensor | None = None
    v_conv_state: torch.Tensor | None = None


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None,
                segments: torch.Tensor | None = None):
        b, _, c = x.shape
        pad = self.kernel_size - 1
        if segments is not None:
            if state is not None:
                raise ValueError(
                    "segment-aware convolution has no cached-window form: the "
                    "cache would have to carry the segment id of each cached "
                    "position for the mask to stay correct"
                )
            from segments import masked_lagged_sum
            return masked_lagged_sum(x, self.weight, segments), x[:, -pad:].transpose(1, 2)
        xt = x.transpose(1, 2)
        left = xt.new_zeros(b, c, pad) if state is None else state.to(xt.dtype)
        full = torch.cat([left, xt], dim=-1)
        y = F.conv1d(full, self.weight, groups=c)
        return y.transpose(1, 2), full[..., -pad:]


def lower_bounded_log_decay(
    raw: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """g_min * sigmoid(exp(A_log) * (raw + bias)), exactly smooth and bounded."""
    h, d = raw.shape[-2:]
    scale = a_log.float().exp().view(*([1] * (raw.ndim - 2)), h, 1)
    bias = dt_bias.float().view(*([1] * (raw.ndim - 2)), h, d)
    return lower_bound * torch.sigmoid(scale * (raw.float() + bias))


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    full_mask: torch.Tensor | None = None,
    starts: torch.Tensor | None = None,
):
    """Float32 decay-first delta-rule reference.

    `starts` marks the first position of each document; the state is cleared
    there, which is what `cu_seqlens` makes the Triton kernel do.  Without it the
    previous document's state is the new document's initial condition.
    """
    b, t, h, d = q.shape
    state = q.new_zeros(b, h, d, v.shape[-1], dtype=torch.float32)
    if initial_state is not None:
        state = state + initial_state.float()
    out = q.new_zeros(b, t, h, v.shape[-1], dtype=torch.float32)
    q32, k32, v32 = q.float(), k.float(), v.float()
    for pos in range(t):
        if starts is not None:
            state = state * (~starts[:, pos]).to(state.dtype).view(b, 1, 1, 1)
        state = state * log_decay[:, pos].exp().unsqueeze(-1)
        key = k32[:, pos]
        predicted = (key.unsqueeze(-1) * state).sum(-2)
        error = v32[:, pos] - predicted
        update = (
            beta[:, pos].float().unsqueeze(-1) * key
        ).unsqueeze(-1) * error.unsqueeze(-2)
        active = 1.0 if full_mask is None else full_mask[:, pos].float().view(b, 1, 1, 1)
        state = state + update * active
        value = (q32[:, pos].unsqueeze(-1) * state).sum(-2)
        out[:, pos] = value * (active if isinstance(active, float) else active.squeeze(-1))
    return out.to(v.dtype), state


class KDAttention(nn.Module):
    def __init__(self, config: K3MiniConfig):
        super().__init__()
        cfg = config.kda
        self.cfg = cfg
        self.hidden_size = config.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.inner = cfg.num_heads * cfg.head_dim
        impl = config.kda_impl
        if impl == "auto":
            impl = "fla" if FLA_AVAILABLE else "reference"
        if impl == "fla" and not FLA_AVAILABLE:
            raise RuntimeError("kda_impl='fla' requires fla-core")
        self.impl = impl

        self.q_proj = nn.Linear(self.hidden_size, self.inner, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.inner, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.inner, bias=False)
        self.q_conv = CausalDepthwiseConv1d(self.inner, cfg.short_conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(self.inner, cfg.short_conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(self.inner, cfg.short_conv_kernel_size)

        self.f_a_proj = nn.Linear(self.hidden_size, cfg.decay_rank, bias=False)
        self.f_b_proj = nn.Linear(cfg.decay_rank, self.inner, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads).uniform_(*cfg.a_log_init)))
        self.dt_bias = nn.Parameter(self._retention_bias(cfg))
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.output_gate = nn.Linear(self.hidden_size, self.inner, bias=False)
        self.output_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.output_proj = nn.Linear(self.inner, self.hidden_size, bias=False)

    def _retention_bias(self, cfg) -> torch.Tensor:
        """Solve `dt_bias` so a fresh layer starts at `cfg.init_retention`/token.

        The decay is `g = lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))` and
        the state is multiplied by `exp(g)` each step.  With `dt_bias = 0` and
        `raw ~ 0` that is `exp(-5 * 0.5) = 0.082`, so an untrained layer discards
        92% of its recurrent state every token -- it cannot carry information far
        enough to learn that carrying it was useful.  Inverting the expression at
        `raw = 0` for the wanted retention gives a bias that starts the layer near
        the identity in time and lets training introduce forgetting.
        """
        retention = float(cfg.init_retention)
        if not 0.0 < retention < 1.0:
            raise ValueError("kda.init_retention must lie strictly between 0 and 1")
        # sigmoid(z) = ln(retention) / lower_bound, then z = logit(that)
        target = math.log(retention) / cfg.gate_lower_bound
        if not 0.0 < target < 1.0:
            raise ValueError(
                "kda.init_retention is unreachable for this gate_lower_bound"
            )
        z = math.log(target / (1.0 - target))
        # exp(A_log) is per head; dt_bias is per channel, so broadcast the head
        # scale across that head's channels.
        scale = self.A_log.detach().exp().view(self.num_heads, 1)
        bias = (z / scale).expand(self.num_heads, self.head_dim)
        return bias.reshape(-1).contiguous()

    def _project(self, x: torch.Tensor, cache: KDACache | None,
                 q_src=None, k_src=None, v_src=None, segments=None):
        """MUDD supplies a separate source per stream; otherwise all three are x."""
        q_src = x if q_src is None else q_src
        k_src = x if k_src is None else k_src
        v_src = x if v_src is None else v_src
        q, qs = self.q_conv(self.q_proj(q_src),
                            None if cache is None else cache.q_conv_state, segments)
        k, ks = self.k_conv(self.k_proj(k_src),
                            None if cache is None else cache.k_conv_state, segments)
        v, vs = self.v_conv(self.v_proj(v_src),
                            None if cache is None else cache.v_conv_state, segments)
        shape = (*x.shape[:2], self.num_heads, self.head_dim)
        return (
            F.silu(q).view(shape),
            F.silu(k).view(shape),
            F.silu(v).view(shape),
            (qs, ks, vs),
        )

    def _raw_decay(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.f_b_proj(self.f_a_proj(x))
        return raw.view(*x.shape[:2], self.num_heads, self.head_dim)

    def forward(
        self, x: torch.Tensor | None = None, cache: KDACache | None = None,
        use_cache: bool = False, full_mask: torch.Tensor | None = None,
        q_input=None, k_input=None, v_input=None,
        segments: torch.Tensor | None = None,
        seq_offsets: torch.Tensor | None = None,
    ):
        # The gates, decay and output gate read the *query* stream: they describe
        # what this position is asking for, which is the same role Q plays.
        if x is None:
            x = q_input
        q, k, v, conv_states = self._project(
            x, cache, q_input, k_input, v_input, segments
        )
        raw_decay = self._raw_decay(x)
        beta_logits = self.beta_proj(x).float()
        initial = None if cache is None else cache.recurrent_state

        # DECAY_ONLY is expressed as beta = 0, not as a mask on the state update.
        # The two are algebraically identical -- with beta = 0 the delta-rule term
        # vanishes and the recurrence reduces to S_t = Diag(alpha_t) S_{t-1}, which
        # is exactly DECAY_ONLY -- but beta is a *kernel input*, so the masked path
        # keeps running on the FLA Triton kernels.  Masking the update outside the
        # kernel instead forces the pure-Python sequential reference: measured
        # 80.6x slower per KDA layer on an RTX 3090 (5.3 ms -> 424.7 ms fwd+bwd at
        # B=4, T=1024), about +14 h on a 100M-token run.  It also keeps both arms
        # on the same kernel, so the forced-Fixed inclusion guarantee holds on GPU
        # and not only on CPU.
        beta = beta_logits.sigmoid()
        if full_mask is not None:
            beta = beta * full_mask.to(beta.dtype).unsqueeze(-1)

        if self.impl == "reference":
            qn = F.normalize(q.float(), dim=-1, eps=self.cfg.l2_eps).to(q.dtype)
            kn = F.normalize(k.float(), dim=-1, eps=self.cfg.l2_eps).to(k.dtype)
            decay = lower_bounded_log_decay(
                raw_decay, self.A_log, self.dt_bias, self.cfg.gate_lower_bound
            )
            starts = None
            if segments is not None:
                starts = torch.zeros_like(segments, dtype=torch.bool)
                starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
                starts[:, 0] = True
            out, state = recurrent_kda(qn, kn, v, decay, beta, initial, starts=starts)
        elif seq_offsets is not None:
            # Variable-length form: the kernel keeps one recurrent state per
            # segment, so a document never inherits the previous one's state.  It
            # requires the batch flattened to a single row, which is why the
            # offsets index the flattened axis.
            rows = x.shape[0]
            flat = [t.reshape(1, -1, *t.shape[2:]) for t in (q, k, v, raw_decay, beta)]
            out, state = chunk_kda(
                q=flat[0], k=flat[1], v=flat[2], g=flat[3], beta=flat[4],
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=initial, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False, safe_gate=True,
                lower_bound=self.cfg.gate_lower_bound, scale=1.0,
                cu_seqlens=seq_offsets,
            )
            out = out.reshape(rows, -1, *out.shape[2:])
        elif x.shape[1] == 1:
            out, state = fused_recurrent_kda(
                q=q, k=k, v=v, g=raw_decay, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=initial, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False,
                lower_bound=self.cfg.gate_lower_bound, scale=1.0,
            )
        else:
            out, state = chunk_kda(
                q=q, k=k, v=v, g=raw_decay, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=initial, output_final_state=True,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=False, safe_gate=True,
                lower_bound=self.cfg.gate_lower_bound, scale=1.0,
            )

        # DECAY_ONLY emits no output; the state still decayed above.
        if full_mask is not None:
            out = out * full_mask.to(out.dtype).view(*full_mask.shape, 1, 1)

        gate = torch.sigmoid(self.output_gate(x)).view_as(out)
        out = self.output_norm(out) * gate
        y = self.output_proj(out.reshape(*x.shape[:2], self.inner))
        new_cache = KDACache(state, *conv_states) if use_cache else None
        return y, new_cache


__all__ = [
    "KDAttention", "KDACache", "CausalDepthwiseConv1d",
    "lower_bounded_log_decay", "recurrent_kda", "FLA_AVAILABLE",
]

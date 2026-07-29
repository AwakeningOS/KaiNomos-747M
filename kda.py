"""Kimi Delta Attention with Kimi K3's smooth lower-bounded decay."""

from __future__ import annotations

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

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        b, _, c = x.shape
        xt = x.transpose(1, 2)
        pad = self.kernel_size - 1
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
):
    """Float32 decay-first delta-rule reference."""
    b, t, h, d = q.shape
    state = q.new_zeros(b, h, d, v.shape[-1], dtype=torch.float32)
    if initial_state is not None:
        state = state + initial_state.float()
    out = q.new_zeros(b, t, h, v.shape[-1], dtype=torch.float32)
    q32, k32, v32 = q.float(), k.float(), v.float()
    for pos in range(t):
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
        self.dt_bias = nn.Parameter(torch.zeros(self.inner))
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.output_gate = nn.Linear(self.hidden_size, self.inner, bias=False)
        self.output_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.output_proj = nn.Linear(self.inner, self.hidden_size, bias=False)

    def _project(self, x: torch.Tensor, cache: KDACache | None,
                 q_src=None, k_src=None, v_src=None):
        """MUDD supplies a separate source per stream; otherwise all three are x."""
        q_src = x if q_src is None else q_src
        k_src = x if k_src is None else k_src
        v_src = x if v_src is None else v_src
        q, qs = self.q_conv(self.q_proj(q_src), None if cache is None else cache.q_conv_state)
        k, ks = self.k_conv(self.k_proj(k_src), None if cache is None else cache.k_conv_state)
        v, vs = self.v_conv(self.v_proj(v_src), None if cache is None else cache.v_conv_state)
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
    ):
        # The gates, decay and output gate read the *query* stream: they describe
        # what this position is asking for, which is the same role Q plays.
        if x is None:
            x = q_input
        q, k, v, conv_states = self._project(x, cache, q_input, k_input, v_input)
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
            out, state = recurrent_kda(qn, kn, v, decay, beta, initial)
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

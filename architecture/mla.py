"""NoPE Gated Multi-head Latent Attention with a compressed KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from config import KaiNomosConfig
from layers import RMSNorm
from torch import nn


@dataclass
class MLACache:
    latent: torch.Tensor | None = None
    key_inv_rms: torch.Tensor | None = None
    segment_ids: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.latent is None else self.latent.shape[1]


class GatedMLA(nn.Module):
    def __init__(self, config: KaiNomosConfig):
        super().__init__()
        cfg = config.mla
        self.cfg = cfg
        self.hidden_size = config.hidden_size
        self.num_heads = cfg.num_heads
        self.scale = 1.0 / math.sqrt(cfg.q_head_dim)

        self.q_a_proj = nn.Linear(config.hidden_size, cfg.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(cfg.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(cfg.q_lora_rank, cfg.num_heads * cfg.q_head_dim, bias=False)

        self.kv_a_proj = nn.Linear(config.hidden_size, cfg.kv_lora_rank, bias=False)
        self.kv_a_norm = RMSNorm(cfg.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            cfg.kv_lora_rank,
            cfg.num_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )
        self.output_gate = nn.Linear(config.hidden_size, cfg.num_heads * cfg.v_head_dim, bias=False)
        self.output_proj = nn.Linear(cfg.num_heads * cfg.v_head_dim, config.hidden_size, bias=False)
        # The algebraically equivalent latent-only decode path remains available
        # for benchmarking, but RTX 3090 A/B medians did not beat the explicit
        # PyTorch path.  Keep the measured winner as the production default.
        self.absorbed_decode_enabled = False

        # QK normalisation over the whole head, applied after projection.
        #
        # One norm across all `q_head_dim` channels rather than separate norms for
        # the content and shared parts: those two are concatenated into a single
        # vector and scored with a single dot product, so there is no
        # implementation boundary between them to normalise across.  (QK-Normed
        # MLA, arXiv 2606.16310, normalises its content and RoPE parts separately
        # because they are cached differently, and states that normalising the
        # concatenated head is equally valid algebraically.  This model is NoPE:
        # the 16 shared channels are not a RoPE path.)
        #
        # This is the explicit post-projection form, which is what that paper's
        # absorbed formulation is proved equivalent to.  It is the right choice
        # here because `expand_kv` already materialises every key from the latent
        # on each call, so there is no latent-only scoring path to preserve.  If an
        # absorbed decode path is ever written, this must become the paper's
        # scheme: fold the static key-side weight into the query projection and
        # cache one inverse-RMS scalar per token and KV group.
        self.q_norm = RMSNorm(cfg.q_head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.q_head_dim, config.rms_norm_eps)

    def project_q(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        return q.view(b, t, self.num_heads, self.cfg.q_head_dim).transpose(1, 2)

    def project_latent(self, x: torch.Tensor):
        return self.kv_a_norm(self.kv_a_proj(x))

    def expand_kv(self, latent: torch.Tensor):
        b, t, _ = latent.shape
        kv = self.kv_b_proj(latent)
        kv = kv.view(
            b, t, self.num_heads, self.cfg.qk_nope_head_dim + self.cfg.v_head_dim
        )
        content, value = kv.split(
            [self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], dim=-1
        )
        return content.transpose(1, 2), value.transpose(1, 2)

    def split_kv_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-head key/value up-projection weights."""
        width = self.cfg.qk_nope_head_dim + self.cfg.v_head_dim
        weight = self.kv_b_proj.weight.view(
            self.num_heads, width, self.cfg.kv_lora_rank
        )
        return weight.split(
            [self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], dim=1
        )

    def project_key(self, latent: torch.Tensor) -> torch.Tensor:
        key_weight, _ = self.split_kv_weights()
        b, t, _ = latent.shape
        key = F.linear(latent, key_weight.flatten(0, 1))
        return key.view(
            b, t, self.num_heads, self.cfg.qk_nope_head_dim
        ).transpose(1, 2)

    def key_inverse_rms(self, key: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(
            key.float().square().mean(-1, keepdim=True) + self.k_norm.eps
        )

    def absorbed_decode(
        self,
        x: torch.Tensor,
        current_latent: torch.Tensor,
        cache: MLACache,
    ) -> tuple[torch.Tensor, MLACache]:
        """Decode one token without expanding historical keys and values."""
        latent = torch.cat([cache.latent, current_latent], dim=1)
        key_weight, value_weight = self.split_kv_weights()

        current_key = self.project_key(current_latent)
        current_inverse_rms = self.key_inverse_rms(current_key).transpose(1, 2)
        cached_inverse_rms = cache.key_inv_rms
        if cached_inverse_rms is None:
            cached_key = self.project_key(cache.latent)
            cached_inverse_rms = self.key_inverse_rms(cached_key).transpose(1, 2)
        key_inverse_rms = torch.cat(
            [cached_inverse_rms, current_inverse_rms], dim=1
        )

        query = self.q_norm(self.project_q(x)).float()
        absorbed_query = torch.matmul(
            query * self.k_norm.weight.float(),
            key_weight.float(),
        )
        latent32 = latent.float().unsqueeze(1)
        score = torch.matmul(
            absorbed_query, latent32.transpose(-1, -2)
        )
        score = score * key_inverse_rms.squeeze(-1).transpose(1, 2).unsqueeze(2)
        probability = torch.softmax(score * self.scale, dim=-1)
        latent_context = torch.matmul(probability, latent32)
        out = torch.matmul(
            latent_context, value_weight.float().transpose(-1, -2)
        )
        out = out.transpose(1, 2).reshape(
            x.shape[0], 1, self.num_heads * self.cfg.v_head_dim
        )
        gate = torch.sigmoid(self.output_gate(x).float())
        y = self.output_proj((out * gate).to(x.dtype))
        return y, MLACache(
            latent=latent,
            key_inv_rms=key_inverse_rms,
            segment_ids=None,
        )

    def forward(
        self, x: torch.Tensor, cache: MLACache | None = None,
        use_cache: bool = False,
        segments: torch.Tensor | None = None,
    ):
        b, t, _ = x.shape
        current_latent = self.project_latent(x)
        if (
            cache is not None
            and cache.latent is not None
            and t == 1
            and segments is None
            and self.absorbed_decode_enabled
        ):
            return self.absorbed_decode(x, current_latent, cache)

        latent = current_latent
        current_segments = segments
        past = 0 if cache is None else cache.seq_len
        if cache is not None and cache.latent is not None:
            latent = torch.cat([cache.latent, latent], dim=1)
            if cache.segment_ids is not None and segments is not None:
                current_segments = torch.cat([cache.segment_ids, segments], dim=1)
        q = self.project_q(x)
        k, v = self.expand_kv(latent)
        key_inv_rms = self.key_inverse_rms(k)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if current_segments is not None:
            from segments import document_mask
            if past == 0:
                mask = document_mask(current_segments)
            else:
                query_segments = current_segments[:, -t:]
                same = query_segments[:, :, None] == current_segments[:, None, :]
                q_pos = torch.arange(past, past + t, device=x.device).view(1, t, 1)
                k_pos = torch.arange(current_segments.shape[1], device=x.device).view(1, 1, -1)
                mask = (same & (k_pos <= q_pos)).unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=mask, scale=self.scale
            )
        elif past == 0 and t == k.shape[2]:
            out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=True, scale=self.scale
            )
        else:
            q_pos = torch.arange(past, past + t, device=x.device).unsqueeze(-1)
            k_pos = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
            out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), attn_mask=k_pos <= q_pos, scale=self.scale
            )
        out = out.transpose(1, 2).reshape(b, t, self.num_heads * self.cfg.v_head_dim)
        gate = torch.sigmoid(self.output_gate(x).float())
        out = (out.float() * gate).to(x.dtype)
        y = self.output_proj(out)
        return y, MLACache(
            latent=latent,
            key_inv_rms=key_inv_rms.transpose(1, 2),
            segment_ids=current_segments,
        ) if use_cache else None


__all__ = ["GatedMLA", "MLACache"]

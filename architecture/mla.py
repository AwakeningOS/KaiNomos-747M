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

    def forward(
        self, x: torch.Tensor, cache: MLACache | None = None,
        use_cache: bool = False,
        segments: torch.Tensor | None = None,
    ):
        b, t, _ = x.shape
        latent = self.project_latent(x)
        current_segments = segments
        past = 0 if cache is None else cache.seq_len
        if cache is not None and cache.latent is not None:
            latent = torch.cat([cache.latent, latent], dim=1)
            if cache.segment_ids is not None and segments is not None:
                current_segments = torch.cat([cache.segment_ids, segments], dim=1)
        q = self.project_q(x)
        k, v = self.expand_kv(latent)
        key_inv_rms = torch.rsqrt(
            k.float().square().mean(-1, keepdim=True) + self.k_norm.eps
        )
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

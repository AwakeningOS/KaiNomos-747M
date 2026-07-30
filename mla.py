"""NoPE Gated Multi-head Latent Attention with a compressed KV cache."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KaiNomosConfig
from layers import RMSNorm


@dataclass
class MLACache:
    latent_kv: torch.Tensor | None = None
    shared_key: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.latent_kv is None else self.latent_kv.shape[1]


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

        self.kv_a_proj = nn.Linear(
            config.hidden_size, cfg.kv_lora_rank + cfg.qk_shared_head_dim, bias=False
        )
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
        latent, shared = self.kv_a_proj(x).split(
            [self.cfg.kv_lora_rank, self.cfg.qk_shared_head_dim], dim=-1
        )
        return latent, shared

    def expand_kv(self, latent: torch.Tensor, shared: torch.Tensor):
        b, t, _ = latent.shape
        kv = self.kv_b_proj(self.kv_a_norm(latent))
        kv = kv.view(
            b, t, self.num_heads, self.cfg.qk_nope_head_dim + self.cfg.v_head_dim
        )
        content, value = kv.split(
            [self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], dim=-1
        )
        shared = shared.unsqueeze(2).expand(b, t, self.num_heads, -1)
        key = torch.cat([content, shared], dim=-1)
        return key.transpose(1, 2), value.transpose(1, 2)

    def forward(
        self, x: torch.Tensor | None = None, cache: MLACache | None = None,
        use_cache: bool = False,
        q_input=None, k_input=None, v_input=None,
        segments: torch.Tensor | None = None,
    ):
        # MLA compresses K and V jointly into one latent, so the KV stream is
        # taken from `k_input`; splitting it further would need two latents and
        # would change the mechanism rather than its inputs.
        if x is None:
            x = q_input
        if v_input is not None:
            raise ValueError(
                "GatedMLA has a single KV input: K and V share one compressed "
                "latent, so pass the mixed KV stream as k_input.  Accepting a "
                "separate v_input silently discarded it and left the caller's "
                "V mixing coefficients with no gradient."
            )
        kv_src = x if k_input is None else k_input
        b, t, _ = x.shape
        latent, shared = self.project_latent(kv_src)
        past = 0 if cache is None else cache.seq_len
        if cache is not None and cache.latent_kv is not None:
            latent = torch.cat([cache.latent_kv, latent], dim=1)
            shared = torch.cat([cache.shared_key, shared], dim=1)
        q = self.project_q(x if q_input is None else q_input)
        k, v = self.expand_kv(latent, shared)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if segments is not None:
            if cache is not None:
                raise NotImplementedError(
                    "document masking with a KV cache needs the segment id of "
                    "every cached position; only the cache-free path is supported"
                )
            from segments import document_mask
            out = F.scaled_dot_product_attention(
                q.float(), k.float(), v.float(),
                attn_mask=document_mask(segments), scale=self.scale,
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
        return y, MLACache(latent, shared) if use_cache else None


__all__ = ["GatedMLA", "MLACache"]

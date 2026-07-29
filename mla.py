"""NoPE Gated Multi-head Latent Attention with a compressed KV cache."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import K3MiniPlusPlusPlusConfig as K3MiniConfig
from layers import RMSNorm


@dataclass
class MLACache:
    latent_kv: torch.Tensor | None = None
    shared_key: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return 0 if self.latent_kv is None else self.latent_kv.shape[1]


class GatedMLA(nn.Module):
    def __init__(self, config: K3MiniConfig):
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
        use_cache: bool = False, read_mask: torch.Tensor | None = None,
        q_input=None, k_input=None, v_input=None,
    ):
        # MLA compresses K and V jointly into one latent, so the KV stream is
        # taken from `k_input`; splitting it further would need two latents and
        # would change the mechanism rather than its inputs.
        if x is None:
            x = q_input
        kv_src = x if k_input is None else k_input
        b, t, _ = x.shape
        latent, shared = self.project_latent(kv_src)
        past = 0 if cache is None else cache.seq_len
        if cache is not None and cache.latent_kv is not None:
            latent = torch.cat([cache.latent_kv, latent], dim=1)
            shared = torch.cat([cache.shared_key, shared], dim=1)
        q = self.project_q(x if q_input is None else q_input)
        k, v = self.expand_kv(latent, shared)

        if past == 0 and t == k.shape[2]:
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
        if read_mask is not None:
            y = y * read_mask.to(y.dtype).unsqueeze(-1)
        return y, MLACache(latent, shared) if use_cache else None


__all__ = ["GatedMLA", "MLACache"]

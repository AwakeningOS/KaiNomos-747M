"""KaiNomos-750M: KDA/MLA hybrid with pure additive Delta stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from config import KaiNomosConfig
from delta_block import DeltaRouter, DeltaState, visible_sources
from kda import KDACache, KDAttention
from layers import RMSNorm, SiTUMLP
from mla import GatedMLA, MLACache
from mtp import MTPHead, mtp_loss, mtp_slices
from segments import (
    cu_seqlens,
    document_starts,
    mask_targets_at_boundaries,
    segment_ids,
)
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class LayerCache:
    kda: KDACache | None = None
    mla: MLACache | None = None


@dataclass
class ModelCache:
    layers: list[LayerCache]
    position: int = 0
    current_segment_id: int = 0


@dataclass
class KaiNomosOutput:
    logits: torch.Tensor | None
    loss: torch.Tensor | None = None
    ntp_loss: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    route_stats: list[dict] = field(default_factory=list)
    cache: ModelCache | None = None


class DecoderLayer(nn.Module):
    def __init__(self, config: KaiNomosConfig, kind: str):
        super().__init__()
        self.kind = kind
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = KDAttention(config) if kind == "KDA" else GatedMLA(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SiTUMLP(config.hidden_size, config.dense_intermediate_size)
        self.delta_attn = DeltaRouter(config.hidden_size, config.rms_norm_eps)
        self.delta_ffn = DeltaRouter(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        attn_sources: tuple[torch.Tensor, ...],
        ffn_source_factory,
        *,
        segments: torch.Tensor | None,
        seq_offsets: torch.Tensor | None,
        cache: LayerCache | None,
        use_cache: bool,
        route_enabled: bool,
        return_stats: bool,
    ):
        residual = hidden
        routed, attn_stats = self.delta_attn(
            hidden, attn_sources if route_enabled else (), return_stats=return_stats
        )
        if self.kind == "KDA":
            attn_out, next_cache = self.attn(
                self.attn_norm(routed),
                cache=None if cache is None else cache.kda,
                use_cache=use_cache,
                segments=segments,
                seq_offsets=seq_offsets,
            )
            layer_cache = LayerCache(kda=next_cache)
        else:
            attn_out, next_cache = self.attn(
                self.attn_norm(routed),
                cache=None if cache is None else cache.mla,
                use_cache=use_cache,
                segments=segments,
            )
            layer_cache = LayerCache(mla=next_cache)
        # The routed input is read-only context.  The untouched main residual is
        # always the base of the state update.
        hidden = residual + attn_out
        ffn_sources = ffn_source_factory(hidden)
        residual = hidden
        routed, ffn_stats = self.delta_ffn(
            hidden, ffn_sources if route_enabled else (), return_stats=return_stats
        )
        ffn_out = self.ffn(self.ffn_norm(routed))
        hidden = residual + ffn_out
        stats = None
        if return_stats:
            stats = {
                "attention": attn_stats.detached(),
                "ffn": ffn_stats.detached(),
            }
        return hidden, layer_cache, stats


class DecoderStage(nn.Module):
    def __init__(self, config: KaiNomosConfig, stage_index: int):
        super().__init__()
        self.stage_index = stage_index
        start = stage_index * 4
        self.layers = nn.ModuleList(
            DecoderLayer(config, config.layer_pattern[start + index])
            for index in range(4)
        )
        self.route_enabled = config.depth_routing == "delta_block"

    def forward(
        self,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
        completed: tuple[torch.Tensor, ...],
        *,
        segments: torch.Tensor | None = None,
        seq_offsets: torch.Tensor | None = None,
        caches: list[LayerCache] | None = None,
        use_cache: bool = False,
        return_stats: bool = False,
    ):
        stage_input = hidden
        state = DeltaState(embedding=embedding, completed=completed)
        next_caches: list[LayerCache] = []
        stats: list[dict] = []
        for local_index, layer in enumerate(self.layers):
            first_attention = self.stage_index == 0 and local_index == 0
            attn_sources = visible_sources(
                state, hidden, stage_input,
                embedding_visible=not first_attention,
                include_partial=local_index != 0,
            )
            completed_names = [
                f"stage_delta_{index}" for index in range(len(state.completed))
            ]
            attn_source_names = (
                ([] if first_attention else ["embedding"])
                + completed_names
                + ([] if local_index == 0 else ["current_stage_partial"])
            )
            ffn_source_names = [
                "embedding", *completed_names, "current_stage_partial",
            ]
            if not self.route_enabled:
                attn_source_names = []
                ffn_source_names = []

            def ffn_sources(current):
                return visible_sources(
                    state, current, stage_input,
                    embedding_visible=True,
                    include_partial=True,
                )

            hidden, layer_cache, layer_stats = layer(
                hidden,
                attn_sources,
                ffn_sources,
                segments=segments,
                seq_offsets=seq_offsets,
                cache=None if caches is None else caches[local_index],
                use_cache=use_cache,
                route_enabled=self.route_enabled,
                return_stats=return_stats,
            )
            next_caches.append(layer_cache)
            if layer_stats is not None:
                layer_stats["attention"]["source_order"] = attn_source_names
                layer_stats["ffn"]["source_order"] = ffn_source_names
                layer_stats.update({
                    "stage": self.stage_index,
                    "local_layer": local_index,
                    "kind": layer.kind,
                })
                stats.append(layer_stats)
        stage_delta = hidden - stage_input
        return hidden, stage_delta, next_caches, stats


class KaiNomosModel(nn.Module):
    def __init__(self, config: KaiNomosConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.stages = nn.ModuleList(
            DecoderStage(config, index) for index in range(config.delta.num_blocks)
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    @property
    def layers(self):
        return [layer for stage in self.stages for layer in stage.layers]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        respect_documents: bool = True,
        cache: ModelCache | None = None,
        use_cache: bool = False,
        return_route_stats: bool = False,
    ):
        segments = offsets = None
        if respect_documents:
            segments = segment_ids(input_ids, self.config.eod_token_id)
            offsets = cu_seqlens(document_starts(input_ids, self.config.eod_token_id))
        embedding = self.embed_tokens(input_ids)
        hidden = embedding
        completed: tuple[torch.Tensor, ...] = ()
        next_caches: list[LayerCache] = []
        route_stats: list[dict] = []
        layer_offset = 0
        for stage in self.stages:
            stage_caches = None
            if cache is not None:
                stage_caches = cache.layers[layer_offset:layer_offset + 4]
            if self.gradient_checkpointing and self.training and not use_cache \
                    and not return_route_stats:
                def stage_fn(h, e, *prior, module=stage):
                    out, delta, _, _ = module(
                        h, e, tuple(prior), segments=segments,
                        seq_offsets=offsets, use_cache=False,
                    )
                    return out, delta
                hidden, stage_delta = checkpoint(
                    stage_fn, hidden, embedding, *completed, use_reentrant=False
                )
                stage_next_caches = []
                stage_stats = []
            else:
                hidden, stage_delta, stage_next_caches, stage_stats = stage(
                    hidden, embedding, completed,
                    segments=segments, seq_offsets=offsets,
                    caches=stage_caches, use_cache=use_cache,
                    return_stats=return_route_stats,
                )
            completed = (*completed, stage_delta)
            next_caches.extend(stage_next_caches)
            route_stats.extend(stage_stats)
            layer_offset += 4
        next_cache = None
        if use_cache:
            next_cache = ModelCache(
                layers=next_caches,
                position=(0 if cache is None else cache.position) + input_ids.shape[1],
                current_segment_id=(0 if cache is None else cache.current_segment_id),
            )
        return self.final_norm(hidden), route_stats, next_cache


class KaiNomosForCausalLM(nn.Module):
    def __init__(self, config: KaiNomosConfig | None = None):
        super().__init__()
        self.config = config or KaiNomosConfig()
        self.model = KaiNomosModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._initialize_module(self.model)
        self.mtp = None
        if self.config.mtp.enabled:
            # MTP presence must not perturb backbone initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0x4F4D4547)
                self.mtp = MTPHead(self.config)
                self._initialize_module(self.mtp)

    def _initialize_module(self, root: nn.Module) -> None:
        for module in root.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, DeltaRouter):
                module.reset_parameters()
        for module in root.modules():
            if isinstance(module, KDAttention):
                module.reset_kda_parameters()

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing = True

    def _chunked_ntp_loss(
        self, hidden: torch.Tensor, targets: torch.Tensor, chunk_tokens: int = 32
    ) -> torch.Tensor:
        total = hidden.new_zeros((), dtype=torch.float32)
        valid = (targets != -100).sum().clamp_min(1)
        for start in range(0, hidden.shape[1], chunk_tokens):
            stop = min(start + chunk_tokens, hidden.shape[1])
            logits = self.lm_head(hidden[:, start:stop])
            total = total + F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets[:, start:stop].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        return total / valid

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        use_mtp: bool | None = None,
        respect_documents: bool = True,
        cache: ModelCache | None = None,
        use_cache: bool = False,
        return_route_stats: bool = False,
    ) -> KaiNomosOutput:
        hidden, route_stats, next_cache = self.model(
            input_ids,
            respect_documents=respect_documents,
            cache=cache,
            use_cache=use_cache,
            return_route_stats=return_route_stats,
        )
        memory_efficient = self.model.gradient_checkpointing and self.training
        logits = None if memory_efficient else self.lm_head(hidden)
        loss = ntp = mtp_value = None
        mtp_logits = None
        if labels is not None:
            targets = mask_targets_at_boundaries(
                labels[:, 1:], input_ids[:, :-1], self.config.eod_token_id
            )
            if memory_efficient:
                ntp = checkpoint(
                    lambda value: self._chunked_ntp_loss(
                        value[:, :-1], targets
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                ntp = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                    targets.reshape(-1), ignore_index=-100,
                )
            loss = ntp
            mtp_enabled = self.mtp is not None if use_mtp is None else use_mtp
            if mtp_enabled and self.mtp is None:
                raise ValueError("MTP was requested but is absent from this config")
            if mtp_enabled and input_ids.shape[1] >= 3:
                h_slice, token_slice, target_slice = mtp_slices(input_ids)
                segments = (
                    segment_ids(input_ids, self.config.eod_token_id)
                    if respect_documents else None
                )
                fused = self.mtp(
                    hidden[:, h_slice],
                    self.model.embed_tokens(input_ids[:, token_slice]),
                    segments=None if segments is None else segments[:, h_slice],
                )
                mtp_hidden = self.model.final_norm(fused)
                mtp_targets = labels[:, target_slice]
                if respect_documents:
                    crosses = (
                        (input_ids[:, h_slice] == self.config.eod_token_id)
                        | (input_ids[:, token_slice] == self.config.eod_token_id)
                    )
                    mtp_targets = mtp_targets.masked_fill(crosses, -100)
                if memory_efficient:
                    mtp_value = checkpoint(
                        lambda value: self._chunked_ntp_loss(value, mtp_targets),
                        mtp_hidden,
                        use_reentrant=False,
                    )
                else:
                    mtp_logits = self.lm_head(mtp_hidden)
                    mtp_value = mtp_loss(mtp_logits, mtp_targets)
                loss = loss + self.config.mtp.loss_weight * mtp_value
        return KaiNomosOutput(
            logits=logits,
            loss=loss,
            ntp_loss=ntp,
            mtp_logits=mtp_logits,
            mtp_loss=mtp_value,
            route_stats=route_stats,
            cache=next_cache,
        )

    def parameter_report(self) -> dict:
        total = sum(parameter.numel() for parameter in self.parameters())
        mtp_only = sum(
            parameter.numel() for name, parameter in self.named_parameters()
            if name.startswith("mtp.")
        )
        state_keys = tuple(self.state_dict())
        return {
            "total_params": total,
            "inference_backbone_params": total - mtp_only,
            "mtp_only_params": mtp_only,
            "mudd_params": sum(
                parameter.numel() for name, parameter in self.named_parameters()
                if "mudd" in name.lower()
            ),
            "mudd_state_keys": sum("mudd" in key.lower() for key in state_keys),
            "delta_router_queries": sum(
                name.endswith(("delta_attn.query", "delta_ffn.query"))
                for name, _ in self.named_parameters()
            ),
        }


__all__ = [
    "DecoderLayer", "DecoderStage", "KaiNomosForCausalLM",
    "KaiNomosModel", "KaiNomosOutput", "LayerCache", "ModelCache",
]

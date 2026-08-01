"""Runtime-only KaiNomos speed candidates.

This module deliberately leaves ``architecture/`` untouched.  A candidate is
applied before the canonical trainer constructs the model, so the durable
step-610 checkpoint remains loadable and every optimization can be benchmarked
or rejected independently.
"""

from __future__ import annotations

import functools
import importlib
import types
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OptimizationOptions:
    lm_loss: str = "chunked"
    lm_chunk_tokens: int = 32
    compile_mode: str = "off"
    checkpoint_policy: str = "full"
    mla_attention: str = "math_fp32"
    mla_gate: str = "eager-fp32"
    kda_final_state: str = "canonical"
    kda_disable_recompute: bool = False
    rms_norm: str = "canonical"
    delta_score: str = "canonical"

    def validate(self) -> None:
        if self.lm_loss != "chunked":
            raise ValueError(f"unknown LM loss backend: {self.lm_loss}")
        if self.lm_chunk_tokens < 1:
            raise ValueError("LM chunk size must be positive")
        if self.compile_mode not in {
            "off",
            "regional-default",
            "regional-max-autotune",
            "pointwise-default",
        }:
            raise ValueError(f"unknown compile mode: {self.compile_mode}")
        if self.checkpoint_policy not in {
            "full",
            "selective-matmul",
            "skip-last-stage",
            "skip-last-2-stages",
            "skip-last-3-stages",
            "skip-last-4-stages",
        }:
            raise ValueError(f"unknown checkpoint policy: {self.checkpoint_policy}")
        if self.mla_attention not in {"math_fp32", "varlen_flash_bf16"}:
            raise ValueError(f"unknown MLA attention backend: {self.mla_attention}")
        if self.mla_gate not in {"eager-fp32", "compiled-fp32"}:
            raise ValueError(f"unknown MLA output-gate backend: {self.mla_gate}")
        if self.mla_gate != "eager-fp32" and self.mla_attention != "varlen_flash_bf16":
            raise ValueError("compiled MLA gate requires varlen Flash MLA")
        if self.kda_final_state not in {"canonical", "training-off"}:
            raise ValueError(f"unknown KDA final-state policy: {self.kda_final_state}")
        if self.rms_norm not in {"canonical", "fla-bf16", "fla-bf16-all"}:
            raise ValueError(f"unknown RMSNorm backend: {self.rms_norm}")
        if self.delta_score not in {"canonical", "fla-rms-linear"}:
            raise ValueError(f"unknown Delta score backend: {self.delta_score}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _patch_lm_loss(model_module, options: OptimizationOptions) -> None:
    model_class = model_module.KaiNomosForCausalLM
    original = model_class._chunked_ntp_loss

    def configured_chunked_loss(self, hidden, targets, chunk_tokens=32):
        del chunk_tokens
        return original(self, hidden, targets, chunk_tokens=options.lm_chunk_tokens)

    model_class._chunked_ntp_loss = configured_chunked_loss


def _patch_selective_checkpoint(model_module) -> None:
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    original_checkpoint = model_module.checkpoint
    aten = torch.ops.aten
    save_ops = {
        aten.mm.default,
        aten.bmm.default,
        aten.addmm.default,
    }

    def policy(_ctx, op, *args, **kwargs):
        del args, kwargs
        if op in save_ops:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    context_fn = functools.partial(create_selective_checkpoint_contexts, policy)

    def selective_checkpoint(function, *args, **kwargs):
        kwargs.setdefault("context_fn", context_fn)
        return original_checkpoint(function, *args, **kwargs)

    model_module.checkpoint = selective_checkpoint


def _patch_mla_varlen(mla_module, gate_backend: str) -> None:
    from torch.nn.attention.varlen import varlen_attn

    original_forward = mla_module.GatedMLA.forward

    if gate_backend == "compiled-fp32":

        @torch.compile(fullgraph=True, dynamic=False)
        def output_gate_product(out, gate_logits):
            return (out.float() * torch.sigmoid(gate_logits.float())).to(out.dtype)

    else:

        def output_gate_product(out, gate_logits):
            return (out.float() * torch.sigmoid(gate_logits.float())).to(out.dtype)

    def varlen_forward(
        self,
        x,
        cache=None,
        use_cache=False,
        segments=None,
    ):
        # Generation/cache behavior remains on the canonical implementation.
        if cache is not None or use_cache or segments is None or not x.is_cuda:
            return original_forward(
                self, x, cache=cache, use_cache=use_cache, segments=segments
            )

        batch, length, _ = x.shape
        latent = self.project_latent(x)
        q = self.project_q(x)
        k, v = self.expand_kv(latent)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Each row begins a segment.  EOD boundaries begin further segments.
        starts = torch.zeros_like(segments, dtype=torch.bool)
        starts[:, 0] = True
        starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
        offsets = torch.cat(
            [
                starts.reshape(-1).nonzero().flatten(),
                starts.new_tensor([batch * length], dtype=torch.int64),
            ]
        ).to(torch.int32)

        q_flat = q.transpose(1, 2).reshape(-1, self.num_heads, self.cfg.q_head_dim)
        k_flat = k.transpose(1, 2).reshape(-1, self.num_heads, self.cfg.q_head_dim)
        v_flat = v.transpose(1, 2).reshape(-1, self.num_heads, self.cfg.v_head_dim)
        target_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else x.dtype
        )
        q_flat = q_flat.to(target_dtype)
        k_flat = k_flat.to(target_dtype)
        v_flat = v_flat.to(target_dtype)
        if q_flat.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                "varlen Flash MLA requires FP16/BF16 autocast inputs, got "
                f"{q_flat.dtype}"
            )
        out = varlen_attn(
            q_flat,
            k_flat,
            v_flat,
            offsets,
            offsets,
            length,
            length,
            scale=self.scale,
            window_size=(-1, 0),
        )
        out = out.reshape(batch, length, self.num_heads * self.cfg.v_head_dim)
        out = output_gate_product(out, self.output_gate(x))
        return self.output_proj(out), None

    mla_module.GatedMLA.forward = varlen_forward


def _patch_kda_chunk_runtime(kda_module, options: OptimizationOptions) -> None:
    """Configure FLA KDA training-only kernel switches.

    The canonical forward always asks the chunk kernel to materialize a final
    recurrent state, even though training discards it when ``use_cache=False``.
    The runtime wrapper can suppress that output while gradients are enabled,
    yet leaves no-grad generation/cache calls untouched.  ``disable_recompute``
    is an independent FLA speed/memory tradeoff and defaults to the canonical
    behavior.
    """
    if (
        options.kda_final_state == "canonical"
        and not options.kda_disable_recompute
    ):
        return
    if kda_module.chunk_kda is None:
        raise RuntimeError("KDA runtime candidates require fla-core")

    original_chunk_kda = kda_module.chunk_kda

    @functools.wraps(original_chunk_kda)
    def configured_chunk_kda(*args, **kwargs):
        if options.kda_final_state == "training-off" and torch.is_grad_enabled():
            kwargs["output_final_state"] = False
        if options.kda_disable_recompute and torch.is_grad_enabled():
            kwargs["disable_recompute"] = True
        return original_chunk_kda(*args, **kwargs)

    kda_module.chunk_kda = configured_chunk_kda


def _patch_fla_rms_norm(model_module, *, include_delta_source: bool) -> None:
    """Use FLA's fused CUDA RMSNorm without changing checkpoint parameters.

    The Delta router's source normalization remains canonical because its FP32
    output directly defines routing scores.  Every other RMSNorm feeds an
    autocast BF16 consumer, so the fused kernel avoids materializing a transient
    FP32 activation while keeping the same weight object and state-dict keys.
    CPU execution deliberately stays on the canonical implementation.
    """
    layers_module = importlib.import_module("layers")
    from fla.modules.layernorm import rms_norm as fla_rms_norm

    canonical_forward = layers_module.RMSNorm.forward

    def fused_forward(self, x):
        if not x.is_cuda:
            return canonical_forward(self, x)
        return fla_rms_norm(x, self.weight, None, eps=self.eps)

    model_class = model_module.KaiNomosForCausalLM
    original_init = model_class.__init__

    def configured_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for name, module in self.named_modules():
            allowed = include_delta_source or not name.endswith("source_norm")
            if isinstance(module, layers_module.RMSNorm) and allowed:
                module.forward = types.MethodType(fused_forward, module)

    model_class.__init__ = configured_init


def _patch_delta_score(delta_module) -> None:
    """Fuse each Delta source RMSNorm and scalar query projection on CUDA."""
    from fla.modules.layernorm import rms_norm_linear

    original_forward = delta_module.DeltaRouter.forward

    def fused_forward(self, hidden, sources, *, return_stats=False):
        if not hidden.is_cuda:
            return original_forward(
                self,
                hidden,
                sources,
                return_stats=return_stats,
            )
        if not sources:
            if not return_stats:
                return hidden, None
            zero = hidden.new_zeros((), dtype=torch.float32)
            return hidden, delta_module.RouteStats(
                source_count=0,
                entropy_mean=zero,
                max_weight_mean=zero,
                source_weight_mean=hidden.new_empty((0,), dtype=torch.float32),
                query_rms=self.query.float().square().mean().sqrt(),
                added_rms=zero,
                residual_rms=hidden.float().square().mean().sqrt(),
            )
        scores = torch.stack(
            [
                rms_norm_linear(
                    source,
                    self.source_norm.weight,
                    None,
                    self.query.unsqueeze(0),
                    None,
                    eps=self.source_norm.eps,
                ).squeeze(-1).float()
                for source in sources
            ],
            dim=-1,
        )
        weights = scores.softmax(-1)
        added = torch.zeros_like(hidden)
        for index, source in enumerate(sources):
            added = added + weights[..., index, None].to(source.dtype) * source
        routed = hidden + added
        if not return_stats:
            return routed, None
        probability = weights.float()
        entropy = -(
            probability * probability.clamp_min(1e-30).log()
        ).sum(-1)
        return routed, delta_module.RouteStats(
            source_count=len(sources),
            entropy_mean=entropy.mean(),
            max_weight_mean=probability.max(-1).values.mean(),
            source_weight_mean=probability.mean((0, 1)),
            query_rms=self.query.float().square().mean().sqrt(),
            added_rms=added.float().square().mean().sqrt(),
            residual_rms=hidden.float().square().mean().sqrt(),
        )

    delta_module.DeltaRouter.forward = fused_forward


def _patch_skip_last_stage_checkpoint(model_module, skip_count: int) -> None:
    model_class = model_module.KaiNomosModel
    original_forward = model_class.forward

    def stage_selective_forward(
        self,
        input_ids,
        *,
        respect_documents=True,
        cache=None,
        use_cache=False,
        return_route_stats=False,
    ):
        active = (
            self.gradient_checkpointing
            and self.training
            and not use_cache
            and not return_route_stats
        )
        if not active:
            return original_forward(
                self,
                input_ids,
                respect_documents=respect_documents,
                cache=cache,
                use_cache=use_cache,
                return_route_stats=return_route_stats,
            )

        segments = offsets = None
        if respect_documents:
            segments = model_module.segment_ids(input_ids, self.config.eod_token_id)
            offsets = model_module.cu_seqlens(
                model_module.document_starts(input_ids, self.config.eod_token_id)
            )
        embedding = self.embed_tokens(input_ids)
        hidden = embedding
        completed = ()
        first_direct_index = len(self.stages) - skip_count
        for index, stage in enumerate(self.stages):
            if index >= first_direct_index:
                hidden, stage_delta, _, _ = stage(
                    hidden,
                    embedding,
                    completed,
                    segments=segments,
                    seq_offsets=offsets,
                    use_cache=False,
                )
            else:

                def stage_fn(h, e, *prior, module=stage):
                    out, delta, _, _ = module(
                        h,
                        e,
                        tuple(prior),
                        segments=segments,
                        seq_offsets=offsets,
                        use_cache=False,
                    )
                    return out, delta

                hidden, stage_delta = model_module.checkpoint(
                    stage_fn,
                    hidden,
                    embedding,
                    *completed,
                    use_reentrant=False,
                )
            completed = (*completed, stage_delta)
        return self.final_norm(hidden), [], None

    model_class.forward = stage_selective_forward


def _patch_pointwise_compile() -> None:
    layers_module = importlib.import_module("layers")

    @torch.compile(fullgraph=True, dynamic=False)
    def fused_situ_product(gate, up):
        return layers_module.situ_gate(gate) * layers_module.softcap_up(up)

    def compiled_mlp_forward(self, x):
        combined = fused_situ_product(self.gate_proj(x), self.up_proj(x))
        return self.down_proj(combined)

    layers_module.SiTUMLP.forward = compiled_mlp_forward


def _patch_regional_compile(canonical_train, mode: str) -> None:
    model_class = canonical_train.KaiNomosForCausalLM
    original_init = model_class.__init__

    def compiled_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        compile_mode = None if mode == "regional-default" else "max-autotune"
        for stage in self.model.stages:
            stage.compile(fullgraph=False, dynamic=False, mode=compile_mode)

    model_class.__init__ = compiled_init


def apply_runtime_optimizations(
    canonical_train,
    options: OptimizationOptions,
) -> dict[str, Any]:
    """Apply one isolated candidate before model construction."""
    options.validate()
    model_module = importlib.import_module("model")
    mla_module = importlib.import_module("mla")
    kda_module = importlib.import_module("kda")
    delta_module = importlib.import_module("delta_block")

    _patch_lm_loss(model_module, options)
    if options.checkpoint_policy == "selective-matmul":
        _patch_selective_checkpoint(model_module)
    elif options.checkpoint_policy.startswith("skip-last"):
        counts = {
            "skip-last-stage": 1,
            "skip-last-2-stages": 2,
            "skip-last-3-stages": 3,
            "skip-last-4-stages": 4,
        }
        _patch_skip_last_stage_checkpoint(
            model_module, counts[options.checkpoint_policy]
        )
    if options.mla_attention == "varlen_flash_bf16":
        _patch_mla_varlen(mla_module, options.mla_gate)
    _patch_kda_chunk_runtime(kda_module, options)
    if options.rms_norm != "canonical":
        _patch_fla_rms_norm(
            model_module,
            include_delta_source=options.rms_norm == "fla-bf16-all",
        )
    if options.delta_score == "fla-rms-linear":
        _patch_delta_score(delta_module)
    if options.compile_mode == "pointwise-default":
        _patch_pointwise_compile()
    elif options.compile_mode != "off":
        _patch_regional_compile(canonical_train, options.compile_mode)
    return options.to_dict()


__all__ = ["OptimizationOptions", "apply_runtime_optimizations"]

"""Bounded CUDA parity gates for runtime-only optimization candidates.

This script never opens a training checkpoint or writes a production run.  It
uses production tensor dimensions with synthetic inputs and emits one JSON
record suitable for an audit trail.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(ARCHITECTURE), str(SCRIPTS)]

from kainomos_optimization_runtime import (
    OptimizationOptions,
    apply_runtime_optimizations,
)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0)
    )


def finite(*values: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def validate_mla() -> dict:
    config_module = importlib.import_module("config")
    mla_module = importlib.reload(importlib.import_module("mla"))
    train_module = importlib.reload(importlib.import_module("train"))
    config = config_module.KaiNomosConfig()
    torch.manual_seed(101)
    reference = mla_module.GatedMLA(config).cuda().train()
    candidate = mla_module.GatedMLA(config).cuda().train()
    candidate.load_state_dict(reference.state_dict())
    original_forward = mla_module.GatedMLA.forward
    segments = torch.ones(2, 256, dtype=torch.long, device="cuda")
    segments[:, 73:] += 1
    segments[:, 181:] += 1
    values = torch.randn(
        2, 256, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )

    ref_values = values.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_output, _ = original_forward(reference, ref_values, segments=segments)
        ref_loss = ref_output.float().square().mean()
    ref_loss.backward()

    apply_runtime_optimizations(
        train_module,
        OptimizationOptions(mla_attention="varlen_flash_bf16"),
    )
    cand_values = values.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cand_output, _ = candidate(cand_values, segments=segments)
        cand_loss = cand_output.float().square().mean()
    cand_loss.backward()

    output_cosine = cosine(ref_output, cand_output)
    input_grad_cosine = cosine(ref_values.grad, cand_values.grad)
    loss_relative_error = abs(float(ref_loss - cand_loss)) / max(
        abs(float(ref_loss)), 1e-30
    )
    passed = (
        finite(ref_output, cand_output, ref_values.grad, cand_values.grad)
        and output_cosine >= 0.999
        and input_grad_cosine >= 0.995
        and loss_relative_error <= 0.01
    )
    return {
        "gate": "mla_varlen_flash_bf16",
        "passed": passed,
        "output_cosine": output_cosine,
        "input_grad_cosine": input_grad_cosine,
        "loss_relative_error": loss_relative_error,
        "reference_loss": float(ref_loss),
        "candidate_loss": float(cand_loss),
    }


def _run_kda_chunk(*, output_final_state: bool, disable_recompute: bool) -> dict:
    from fla.ops.kda import chunk_kda

    torch.manual_seed(107)
    batch, length, heads, width = 1, 256, 10, 128
    bases = {
        "q": torch.randn(
            batch, length, heads, width, device="cuda", dtype=torch.bfloat16
        ),
        "k": torch.randn(
            batch, length, heads, width, device="cuda", dtype=torch.bfloat16
        ),
        "v": torch.randn(
            batch, length, heads, width, device="cuda", dtype=torch.bfloat16
        ),
        "g": torch.randn(
            batch, length, heads, width, device="cuda", dtype=torch.bfloat16
        ),
        "beta_logits": torch.randn(
            batch, length, heads, device="cuda", dtype=torch.float32
        ),
        "A_log": torch.zeros(heads, device="cuda", dtype=torch.float32),
        "dt_bias": torch.randn(
            heads * width, device="cuda", dtype=torch.float32
        ),
    }
    values = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in bases.items()
    }
    beta = values["beta_logits"].sigmoid()
    offsets = torch.tensor(
        [0, 64, 128, 192, 256], device="cuda", dtype=torch.int32
    )
    torch.cuda.reset_peak_memory_stats()
    output, state = chunk_kda(
        q=values["q"],
        k=values["k"],
        v=values["v"],
        g=values["g"],
        beta=beta,
        A_log=values["A_log"],
        dt_bias=values["dt_bias"],
        initial_state=None,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        safe_gate=True,
        lower_bound=-5.0,
        scale=1.0,
        cu_seqlens=offsets,
        disable_recompute=disable_recompute,
    )
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    return {
        "output": output.detach(),
        "state_is_none": state is None,
        "loss": loss.detach(),
        "grads": {
            name: value.grad.detach()
            for name, value in values.items()
            if value.grad is not None
        },
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def validate_kda(*, candidate: str) -> dict:
    if candidate == "final-state":
        reference = _run_kda_chunk(
            output_final_state=True,
            disable_recompute=False,
        )
        actual = _run_kda_chunk(
            output_final_state=False,
            disable_recompute=False,
        )
        state_condition = not reference["state_is_none"] and actual["state_is_none"]
    elif candidate == "disable-recompute":
        reference = _run_kda_chunk(
            output_final_state=False,
            disable_recompute=False,
        )
        actual = _run_kda_chunk(
            output_final_state=False,
            disable_recompute=True,
        )
        state_condition = reference["state_is_none"] and actual["state_is_none"]
    else:
        raise ValueError(candidate)

    shared_grad_names = sorted(set(reference["grads"]) & set(actual["grads"]))
    reference_grads = torch.cat(
        [reference["grads"][name].float().flatten() for name in shared_grad_names]
    )
    actual_grads = torch.cat(
        [actual["grads"][name].float().flatten() for name in shared_grad_names]
    )
    output_cosine = cosine(reference["output"], actual["output"])
    gradient_cosine = cosine(reference_grads, actual_grads)
    loss_relative_error = abs(float(reference["loss"] - actual["loss"])) / max(
        abs(float(reference["loss"])), 1e-30
    )
    passed = (
        state_condition
        and finite(
            reference["output"],
            actual["output"],
            reference_grads,
            actual_grads,
        )
        and output_cosine >= 0.999999
        and gradient_cosine >= 0.99999
        and loss_relative_error <= 1e-6
    )
    return {
        "gate": f"kda_{candidate.replace('-', '_')}",
        "passed": passed,
        "output_cosine": output_cosine,
        "gradient_cosine": gradient_cosine,
        "loss_relative_error": loss_relative_error,
        "reference_loss": float(reference["loss"]),
        "candidate_loss": float(actual["loss"]),
        "state_condition": state_condition,
        "reference_peak_reserved_gib": reference["peak_reserved_gib"],
        "candidate_peak_reserved_gib": actual["peak_reserved_gib"],
        "gradient_tensors": shared_grad_names,
    }


def validate_rms_norm(
    *,
    include_delta_source: bool,
    fused_delta_score: bool = False,
) -> dict:
    config_module = importlib.import_module("config")
    model_module = importlib.reload(importlib.import_module("model"))
    train_module = importlib.reload(importlib.import_module("train"))
    config = config_module.KaiNomosConfig.tiny()
    torch.manual_seed(109)
    reference = model_module.KaiNomosForCausalLM(config).cuda().train()
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if name.endswith(".query"):
                parameter.normal_(mean=0.0, std=0.05)
            elif name.endswith("source_norm.weight"):
                parameter.uniform_(0.8, 1.2)
    apply_runtime_optimizations(
        train_module,
        OptimizationOptions(
            rms_norm="fla-bf16-all" if include_delta_source else "fla-bf16",
            delta_score="fla-rms-linear" if fused_delta_score else "canonical",
        ),
    )
    candidate = model_module.KaiNomosForCausalLM(config).cuda().train()
    candidate.load_state_dict(reference.state_dict())
    ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_output = reference(ids, labels=ids)
        reference_loss = reference_output.loss
    reference_loss.backward()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        candidate_output = candidate(ids, labels=ids)
        candidate_loss = candidate_output.loss
    candidate_loss.backward()

    reference_grads = torch.cat(
        [
            parameter.grad.float().flatten()
            for parameter in reference.parameters()
            if parameter.grad is not None
        ]
    )
    candidate_grads = torch.cat(
        [
            parameter.grad.float().flatten()
            for parameter in candidate.parameters()
            if parameter.grad is not None
        ]
    )
    gradient_cosine = cosine(reference_grads, candidate_grads)
    loss_relative_error = abs(
        float((reference_loss - candidate_loss).detach())
    ) / max(
        abs(float(reference_loss.detach())), 1e-30
    )
    passed = (
        finite(reference_loss, candidate_loss, reference_grads, candidate_grads)
        and gradient_cosine >= 0.999
        and loss_relative_error <= 0.001
    )
    return {
        "gate": (
            "fla_delta_rms_norm_linear_score"
            if fused_delta_score
            else (
                "fla_bf16_all_rms_norm"
                if include_delta_source
                else "fla_bf16_rms_norm"
            )
        ),
        "passed": passed,
        "gradient_cosine": gradient_cosine,
        "loss_relative_error": loss_relative_error,
        "reference_loss": float(reference_loss.detach()),
        "candidate_loss": float(candidate_loss.detach()),
    }


def validate_mla_gate() -> dict:
    torch.manual_seed(113)
    shape = (2, 256, 1_280)
    base_output = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    base_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    reference_output = base_output.detach().clone().requires_grad_(True)
    reference_gate = base_gate.detach().clone().requires_grad_(True)
    expected = (
        reference_output.float() * torch.sigmoid(reference_gate.float())
    ).to(torch.bfloat16)
    reference_loss = expected.float().square().mean()
    reference_loss.backward()

    @torch.compile(fullgraph=True, dynamic=False)
    def compiled_gate_product(output, gate):
        return (output.float() * torch.sigmoid(gate.float())).to(output.dtype)

    candidate_output = base_output.detach().clone().requires_grad_(True)
    candidate_gate = base_gate.detach().clone().requires_grad_(True)
    actual = compiled_gate_product(candidate_output, candidate_gate)
    candidate_loss = actual.float().square().mean()
    candidate_loss.backward()

    output_cosine = cosine(expected, actual)
    output_gradient_cosine = cosine(reference_output.grad, candidate_output.grad)
    gate_gradient_cosine = cosine(reference_gate.grad, candidate_gate.grad)
    loss_relative_error = abs(
        float((reference_loss - candidate_loss).detach())
    ) / max(abs(float(reference_loss.detach())), 1e-30)
    passed = (
        finite(
            expected,
            actual,
            reference_output.grad,
            candidate_output.grad,
            reference_gate.grad,
            candidate_gate.grad,
        )
        and output_cosine >= 0.999999
        and output_gradient_cosine >= 0.99999
        and gate_gradient_cosine >= 0.99999
        and loss_relative_error <= 1e-6
    )
    return {
        "gate": "compiled_mla_fp32_gate_product",
        "passed": passed,
        "output_cosine": output_cosine,
        "output_gradient_cosine": output_gradient_cosine,
        "gate_gradient_cosine": gate_gradient_cosine,
        "loss_relative_error": loss_relative_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        choices=(
            "mla",
            "kda-final-state",
            "kda-disable-recompute",
            "rms-norm",
            "rms-norm-all",
            "mla-gate",
            "delta-score",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.gate == "mla":
        result = validate_mla()
    elif args.gate == "kda-final-state":
        result = validate_kda(candidate="final-state")
    elif args.gate == "kda-disable-recompute":
        result = validate_kda(candidate="disable-recompute")
    elif args.gate == "rms-norm":
        result = validate_rms_norm(include_delta_source=False)
    elif args.gate == "rms-norm-all":
        result = validate_rms_norm(include_delta_source=True)
    elif args.gate == "delta-score":
        result = validate_rms_norm(
            include_delta_source=True,
            fused_delta_score=True,
        )
    else:
        result = validate_mla_gate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

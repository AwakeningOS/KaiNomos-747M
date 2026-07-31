#!/usr/bin/env python
"""Compare completed KaiNomos-750M residual arms under the frozen contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MATCHING_METADATA = (
    "architecture_id",
    "source_sha256",
    "tokenizer_sha256",
    "data_manifest_sha256",
    "optimizer_contract",
    "data_order_contract",
    "manifest_adapter_id",
    "initial_parameter_sha256",
)
SCREEN_TOKENS = 67_108_864


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm(run_dir: Path) -> dict:
    summary = _read(run_dir / "run_summary.json")
    validations = sorted(run_dir.glob("validation_final_step*.json"))
    if summary.get("status") != "complete" or not validations:
        raise ValueError(f"arm is incomplete: {run_dir}")
    validation = _read(validations[-1])
    if validation.get("scope") != "full":
        raise ValueError(f"arm lacks final full validation: {run_dir}")
    if validation.get("tokens_done") != summary.get("tokens_done"):
        raise ValueError(f"arm summary/validation token mismatch: {run_dir}")
    if validation.get("weighted_nll") is None:
        raise ValueError(f"arm has no weighted NLL: {run_dir}")
    return {
        "run_dir": str(run_dir.resolve()),
        "summary": summary,
        "validation": validation,
    }


def compare(baseline_dir: Path, delta_dir: Path) -> dict:
    baseline = _arm(baseline_dir)
    delta = _arm(delta_dir)
    baseline_train = baseline["summary"].get("train_config", {})
    delta_train = delta["summary"].get("train_config", {})
    if baseline_train.get("depth_routing") != "none":
        raise ValueError("baseline arm must use depth_routing=none")
    if delta_train.get("depth_routing") != "delta_block":
        raise ValueError("delta arm must use depth_routing=delta_block")
    if baseline_train.get("mtp") != "off" or delta_train.get("mtp") != "off":
        raise ValueError("architecture screen requires MTP off in both arms")
    for key in (
        "seed", "schedule_tokens", "sequence_length", "micro_batch",
        "grad_accum", "lr", "min_lr", "warmup_steps", "weight_decay",
    ):
        if baseline_train.get(key) != delta_train.get(key):
            raise ValueError(f"A/B training contract mismatch: {key}")
    left_meta = baseline["summary"]["metadata"]
    right_meta = delta["summary"]["metadata"]
    for key in MATCHING_METADATA:
        if left_meta.get(key) != right_meta.get(key):
            raise ValueError(f"A/B contract mismatch: {key}")
    left_tokens = int(baseline["summary"]["tokens_done"])
    right_tokens = int(delta["summary"]["tokens_done"])
    if left_tokens != right_tokens:
        raise ValueError("A/B token budgets differ")
    if left_tokens < SCREEN_TOKENS:
        raise ValueError(
            f"architecture screen requires at least {SCREEN_TOKENS:,} tokens per arm"
        )
    left_nll = float(baseline["validation"]["weighted_nll"])
    right_nll = float(delta["validation"]["weighted_nll"])
    winner = "inconclusive" if left_nll == right_nll else (
        "delta_block" if right_nll < left_nll else "normal_residual"
    )
    return {
        "contract": "kainomos_750m_architecture_screen_v1",
        "tokens_per_arm": left_tokens,
        "selection_metric": "final_full_weighted_heldout_ntp_nll",
        "winner": winner,
        "arms": {
            "normal_residual": baseline,
            "delta_block": delta,
        },
        "mtp_decision": "not_part_of_architecture_screen",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--delta-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_run, args.delta_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

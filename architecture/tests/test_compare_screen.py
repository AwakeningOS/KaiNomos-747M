from __future__ import annotations

import json

import pytest
from compare_screen import SCREEN_TOKENS, compare


def _write_arm(
    path, nll: float, *, routing: str, tokens: int = SCREEN_TOKENS,
    manifest="same",
):
    path.mkdir()
    metadata = {
        "architecture_id": "kainomos_750m_v1",
        "source_sha256": "source",
        "tokenizer_sha256": "tokenizer",
        "data_manifest_sha256": manifest,
        "optimizer_contract": "optimizer",
        "data_order_contract": "order",
        "manifest_adapter_id": "adapter",
        "initial_parameter_sha256": "initial",
    }
    train_config = {
        "depth_routing": routing, "mtp": "off", "seed": 11,
        "schedule_tokens": 32_551_993_344, "sequence_length": 1024,
        "micro_batch": 1, "grad_accum": 64, "lr": 3e-4,
        "min_lr": 3e-5, "warmup_steps": 4968, "weight_decay": 0.1,
    }
    (path / "run_summary.json").write_text(json.dumps({
        "status": "complete", "tokens_done": tokens, "metadata": metadata,
        "train_config": train_config,
    }))
    (path / "validation_final_step00001024.json").write_text(json.dumps({
        "scope": "full", "tokens_done": tokens, "weighted_nll": nll,
        "per_source_nll": {"local": nll},
    }))


def test_screen_selects_lower_final_full_heldout_nll(tmp_path):
    baseline, delta = tmp_path / "baseline", tmp_path / "delta"
    _write_arm(baseline, 3.2, routing="none")
    _write_arm(delta, 3.1, routing="delta_block")
    result = compare(baseline, delta)
    assert result["winner"] == "delta_block"
    assert result["tokens_per_arm"] == SCREEN_TOKENS
    assert result["mtp_decision"] == "not_part_of_architecture_screen"


def test_screen_rejects_contract_or_budget_mismatch(tmp_path):
    baseline, delta = tmp_path / "baseline", tmp_path / "delta"
    _write_arm(baseline, 3.2, routing="none")
    _write_arm(delta, 3.1, routing="delta_block", manifest="different")
    with pytest.raises(ValueError, match="data_manifest_sha256"):
        compare(baseline, delta)

    (delta / "run_summary.json").unlink()
    (delta / "validation_final_step00001024.json").unlink()
    _write_arm(
        delta / "nested", 3.1, routing="delta_block",
        tokens=SCREEN_TOKENS - 65_536,
    )
    with pytest.raises(ValueError, match="token budgets differ"):
        compare(baseline, delta / "nested")

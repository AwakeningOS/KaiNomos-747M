from __future__ import annotations

import json

import numpy as np
import torch
from config import KaiNomosConfig
from model import KaiNomosForCausalLM
from observe import (
    SCHEMA_VERSION,
    ValidationSource,
    collect_architecture_diagnostics,
    measure_fixed_validation,
    sample,
    summarize_optimizer_state,
)


def _tokens(path, seed: int, count: int = 97):
    generator = np.random.default_rng(seed)
    generator.integers(5, 96, size=count, dtype=np.uint16).tofile(path)


def test_fixed_validation_is_per_source_and_weighted(tmp_path):
    cfg = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(cfg).eval()
    japanese = tmp_path / "japanese.bin"
    code = tmp_path / "code.bin"
    _tokens(japanese, 1)
    _tokens(code, 2)
    result = measure_fixed_validation(
        model,
        cfg,
        {
            "japanese": ValidationSource((japanese,), 3.0),
            "code": ValidationSource((code,), 1.0),
        },
        max_tokens=64,
        batch=1,
        device="cpu",
    )
    assert set(result["sources"]) == {"japanese", "code"}
    ja = result["sources"]["japanese"]
    code_row = result["sources"]["code"]
    expected = (3 * ja["nll"] + code_row["nll"]) / 4
    assert abs(result["weighted"]["nll"] - expected) < 1e-8
    assert "correct_token_margin" in ja
    assert len(ja["files"][0]["sha256"]) == 64


def test_snapshot_diagnostics_cover_model_mechanisms_and_are_json_safe():
    cfg = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(cfg).eval()
    ids = torch.randint(5, cfg.vocab_size, (1, 8))
    diagnostics = collect_architecture_diagnostics(model, ids)

    assert len(diagnostics["delta_routes"]) == cfg.num_hidden_layers
    assert all(set(row) >= {"attention", "ffn", "stage", "local_layer"}
               for row in diagnostics["delta_routes"])
    assert len(diagnostics["kda"]) == 6
    assert len(diagnostics["mla"]) == 2
    for row in diagnostics["kda"]:
        assert set(row) == {
            "module", "retention", "beta", "state", "output_gate",
            "document_reset",
        }
        assert row["state"] is not None
        assert "p01" in row["retention"] and "p99" in row["retention"]
        assert "saturation_low" in row["output_gate"]
    for row in diagnostics["mla"]:
        assert set(row) == {
            "module", "q", "k", "attention_logits", "output_gate", "latent",
        }
    first_route = diagnostics["delta_routes"][0]
    assert first_route["attention"]["source_order"] == []
    assert first_route["ffn"]["source_order"] == [
        "embedding", "current_stage_partial",
    ]
    json.dumps({"schema_version": SCHEMA_VERSION, "architecture": diagnostics},
               allow_nan=False)


def test_kda_observation_proves_document_state_is_zeroed_at_boundary():
    cfg = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(cfg).eval()
    ids = torch.randint(5, cfg.vocab_size, (1, 8))
    ids[:, 3] = cfg.eod_token_id
    diagnostics = collect_architecture_diagnostics(model, ids)
    for row in diagnostics["kda"]:
        reset = row["document_reset"]
        assert reset["boundaries"] == 1
        assert reset["state_rms_after_reset"]["max"] == 0.0
        assert reset["state_rms_before_reset"]["max"] >= 0.0


def test_optimizer_summary_is_bounded_and_json_safe():
    state = {
        "param_groups": [{
            "params": [0, 1], "lr": 3e-4, "weight_decay": 0.1,
            "use_muon": True, "update_rms": 0.2,
        }],
        "state": {
            0: {"step": 7, "momentum_buffer": torch.tensor([3.0, 4.0])},
            1: {"step": 7, "momentum_buffer": torch.tensor([0.0, 0.0])},
        },
    }
    summary = summarize_optimizer_state(state)
    assert summary["groups"][0]["parameter_count"] == 2
    assert summary["step_min"] == summary["step_max"] == 7
    assert summary["tensors"]["momentum_buffer"]["tensor_count"] == 2
    json.dumps(summary, allow_nan=False)


class _TinyTokenizer:
    def encode(self, text):
        return [5, 6]

    def decode(self, ids):
        return " ".join(map(str, ids))


def test_sampling_uses_cache_stable_generation_api_on_cpu():
    cfg = KaiNomosConfig.tiny()
    model = KaiNomosForCausalLM(cfg).eval()
    rows = sample(model, _TinyTokenizer(), ["prompt"], 2, 0.0, 11, "cpu")
    assert rows[0]["prompt"] == "prompt"
    assert len(rows[0]["continuation"].split()) == 2

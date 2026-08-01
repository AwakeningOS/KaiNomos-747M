"""CPU/static gates for the runtime optimization laboratory."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = ROOT / "architecture"
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(ARCHITECTURE), str(SCRIPTS)]

from kainomos_optimization_runtime import (
    OptimizationOptions,
    apply_runtime_optimizations,
)
from run_kainomos_runtime_tuned import target_step_for_tokens


def test_options_reject_invalid_values():
    with pytest.raises(ValueError):
        OptimizationOptions(lm_chunk_tokens=0).validate()
    with pytest.raises(ValueError):
        OptimizationOptions(compile_mode="magic").validate()
    with pytest.raises(ValueError):
        OptimizationOptions(kda_final_state="always-off").validate()
    with pytest.raises(ValueError):
        OptimizationOptions(rms_norm="unknown").validate()
    with pytest.raises(ValueError):
        OptimizationOptions(mla_gate="compiled-fp32").validate()
    with pytest.raises(ValueError):
        OptimizationOptions(delta_score="unknown").validate()


def test_target_step_uses_ceiling_so_final_checkpoint_is_not_skipped():
    assert target_step_for_tokens(16_000_000_000, 65_536) == 244_141
    assert 244_141 * 65_536 == 16_000_024_576
    assert target_step_for_tokens(65_536, 65_536) == 1


def test_fla_rms_norm_candidate_keeps_cpu_model_and_gradients_canonical():
    model_module = importlib.import_module("model")
    train_module = importlib.import_module("train")
    config_module = importlib.import_module("config")
    delta_module = importlib.import_module("delta_block")
    config = config_module.KaiNomosConfig.tiny()
    torch.manual_seed(7)
    reference = model_module.KaiNomosForCausalLM(config)
    original_init = model_module.KaiNomosForCausalLM.__init__
    original_delta_forward = delta_module.DeltaRouter.forward
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(
                rms_norm="fla-bf16-all",
                delta_score="fla-rms-linear",
            ),
        )
        candidate = model_module.KaiNomosForCausalLM(config)
        candidate.load_state_dict(reference.state_dict())
        ids = torch.randint(0, config.vocab_size, (2, 12))
        expected = reference(ids, labels=ids).loss
        expected.backward()
        expected_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
            if parameter.grad is not None
        }
        actual = candidate(ids, labels=ids).loss
        actual.backward()
        torch.testing.assert_close(actual, expected)
        for name, parameter in candidate.named_parameters():
            if name in expected_grads:
                torch.testing.assert_close(parameter.grad, expected_grads[name])
    finally:
        model_module.KaiNomosForCausalLM.__init__ = original_init
        delta_module.DeltaRouter.forward = original_delta_forward


def test_kda_training_switches_are_wired_without_touching_no_grad_cache_calls():
    kda_module = importlib.import_module("kda")
    train_module = importlib.import_module("train")
    calls = []

    def fake_chunk_kda(*args, **kwargs):
        calls.append(dict(kwargs))
        return args, kwargs

    original_chunk_kda = kda_module.chunk_kda
    try:
        kda_module.chunk_kda = fake_chunk_kda
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(
                kda_final_state="training-off",
                kda_disable_recompute=True,
            ),
        )
        kda_module.chunk_kda(output_final_state=True)
        assert calls[-1]["output_final_state"] is False
        assert calls[-1]["disable_recompute"] is True

        with torch.no_grad():
            kda_module.chunk_kda(
                output_final_state=True,
                disable_recompute=False,
            )
        assert calls[-1]["output_final_state"] is True
        assert calls[-1]["disable_recompute"] is False
    finally:
        kda_module.chunk_kda = original_chunk_kda


def test_kda_training_switches_leave_reference_forward_and_gradients_unchanged():
    kda_module = importlib.import_module("kda")
    train_module = importlib.import_module("train")
    config_module = importlib.import_module("config")
    config = config_module.KaiNomosConfig.tiny()
    torch.manual_seed(11)
    reference = kda_module.KDAttention(config)
    candidate = kda_module.KDAttention(config)
    candidate.load_state_dict(reference.state_dict())
    values = torch.randn(2, 7, config.hidden_size)

    expected_values = values.detach().clone().requires_grad_(True)
    expected, _ = reference(expected_values)
    expected.sum().backward()
    expected_input_grad = expected_values.grad.detach().clone()
    expected_parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    original_chunk_kda = kda_module.chunk_kda
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(
                kda_final_state="training-off",
                kda_disable_recompute=True,
            ),
        )
        actual_values = values.detach().clone().requires_grad_(True)
        actual, _ = candidate(actual_values)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_values.grad, expected_input_grad)
        for name, parameter in candidate.named_parameters():
            if name in expected_parameter_grads:
                torch.testing.assert_close(
                    parameter.grad, expected_parameter_grads[name]
                )
    finally:
        kda_module.chunk_kda = original_chunk_kda


def test_chunk_override_matches_canonical_loss_and_gradients():
    model_module = importlib.import_module("model")
    train_module = importlib.import_module("train")
    config_module = importlib.import_module("config")
    torch.manual_seed(9)
    model = model_module.KaiNomosForCausalLM(config_module.KaiNomosConfig.tiny())
    hidden = torch.randn(2, 9, model.config.hidden_size, requires_grad=True)
    targets = torch.randint(0, model.config.vocab_size, (2, 9))
    expected = model._chunked_ntp_loss(hidden, targets, chunk_tokens=3)
    expected_grad = torch.autograd.grad(expected, hidden)[0]

    original_loss = model_module.KaiNomosForCausalLM._chunked_ntp_loss
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(lm_chunk_tokens=3),
        )
        actual_hidden = hidden.detach().clone().requires_grad_(True)
        actual = model._chunked_ntp_loss(actual_hidden, targets)
        actual_grad = torch.autograd.grad(actual, actual_hidden)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
    finally:
        model_module.KaiNomosForCausalLM._chunked_ntp_loss = original_loss


def test_varlen_candidate_keeps_cpu_path_canonical():
    mla_module = importlib.import_module("mla")
    train_module = importlib.import_module("train")
    config_module = importlib.import_module("config")
    config = config_module.KaiNomosConfig.tiny()
    torch.manual_seed(13)
    reference = mla_module.GatedMLA(config).eval()
    candidate = mla_module.GatedMLA(config).eval()
    candidate.load_state_dict(reference.state_dict())
    values = torch.randn(2, 7, config.hidden_size)
    segments = torch.tensor([[1, 1, 2, 2, 2, 3, 3], [1, 1, 1, 1, 2, 2, 2]])
    expected, _ = reference(values, segments=segments)
    original_forward = mla_module.GatedMLA.forward
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(mla_attention="varlen_flash_bf16"),
        )
        actual, _ = candidate(values, segments=segments)
        torch.testing.assert_close(actual, expected)
    finally:
        mla_module.GatedMLA.forward = original_forward


def test_selective_checkpoint_matches_full_checkpoint_on_cpu():
    model_module = importlib.import_module("model")
    train_module = importlib.import_module("train")
    config_module = importlib.import_module("config")
    config = config_module.KaiNomosConfig.tiny()
    torch.manual_seed(17)
    reference = model_module.KaiNomosForCausalLM(config)
    candidate = model_module.KaiNomosForCausalLM(config)
    candidate.load_state_dict(reference.state_dict())
    reference.gradient_checkpointing_enable()
    candidate.gradient_checkpointing_enable()
    ids = torch.randint(0, config.vocab_size, (2, 12))

    expected = reference(ids, labels=ids).loss
    expected.backward()
    expected_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }

    original_checkpoint = model_module.checkpoint
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(checkpoint_policy="selective-matmul"),
        )
        actual = candidate(ids, labels=ids).loss
        actual.backward()
        torch.testing.assert_close(actual, expected)
        for name, parameter in candidate.named_parameters():
            if name in expected_grads:
                torch.testing.assert_close(parameter.grad, expected_grads[name])
    finally:
        model_module.checkpoint = original_checkpoint


def test_skip_last_stage_checkpoint_matches_full_checkpoint_on_cpu():
    model_module = importlib.import_module("model")
    train_module = importlib.import_module("train")
    config_module = importlib.import_module("config")
    config = config_module.KaiNomosConfig.tiny()
    torch.manual_seed(19)
    reference = model_module.KaiNomosForCausalLM(config)
    candidate = model_module.KaiNomosForCausalLM(config)
    candidate.load_state_dict(reference.state_dict())
    reference.gradient_checkpointing_enable()
    candidate.gradient_checkpointing_enable()
    ids = torch.randint(0, config.vocab_size, (2, 12))

    expected = reference(ids, labels=ids).loss
    expected.backward()
    expected_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    original_forward = model_module.KaiNomosModel.forward
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(checkpoint_policy="skip-last-stage"),
        )
        actual = candidate(ids, labels=ids).loss
        actual.backward()
        torch.testing.assert_close(actual, expected)
        for name, parameter in candidate.named_parameters():
            if name in expected_grads:
                torch.testing.assert_close(parameter.grad, expected_grads[name])
    finally:
        model_module.KaiNomosModel.forward = original_forward


def test_pointwise_compile_candidate_preserves_situ_formula():
    layers_module = importlib.import_module("layers")
    train_module = importlib.import_module("train")
    torch.manual_seed(23)
    reference = layers_module.SiTUMLP(16, 24)
    candidate = layers_module.SiTUMLP(16, 24)
    candidate.load_state_dict(reference.state_dict())
    values = torch.randn(2, 7, 16)
    expected = reference(values)
    original_forward = layers_module.SiTUMLP.forward
    try:
        apply_runtime_optimizations(
            train_module,
            OptimizationOptions(compile_mode="pointwise-default"),
        )
        actual = candidate(values)
        torch.testing.assert_close(actual, expected)
    finally:
        layers_module.SiTUMLP.forward = original_forward

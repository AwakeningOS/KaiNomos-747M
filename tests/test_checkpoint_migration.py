"""Migration from the 82M checkpoint must preserve kind and start near-identity."""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from migrate_82m_to_110m import build_source_layer_map, migrate
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_new_layers_are_cloned_from_the_same_kind():
    source = ["KDA", "KDA", "KDA", "MLA"] * 3 + ["KDA"]      # 13 layers
    target = ["KDA", "KDA", "KDA", "MLA"] * 4                # 16 layers
    mapping = build_source_layer_map(source, target)

    assert len(mapping) == 16
    for target_index, source_index in mapping.items():
        assert source[source_index] == target[target_index], (
            f"layer {target_index} ({target[target_index]}) seeded from "
            f"{source_index} ({source[source_index]}) -- KDA and MLA are "
            "different operators and must never be copied across"
        )
    # the depths that already existed keep their own weights
    for i in range(13):
        if source[i] == target[i]:
            assert mapping[i] == i


def test_migration_copies_widens_and_damps():
    torch.manual_seed(29)
    cfg = Config.tiny()
    source_cfg = Config.tiny()
    source_cfg.num_hidden_layers = 4
    source_cfg.layer_pattern = ("KDA", "KDA", "KDA", "MLA")
    source_cfg.delta.num_blocks = 2
    source_cfg.joint_route.ffn_width_tiers = (12, 24, 48, 60)
    source_cfg.dense_intermediate_size = 48
    source_cfg.joint_route.fixed_ffn_width = 48
    source = Model(source_cfg)

    model, stats = migrate(source.state_dict(), list(source_cfg.layer_pattern), cfg)
    assert stats["whole"] > 0
    assert stats["widened"] > 0
    assert stats["cloned_layers"], "the added depths must be recorded"

    width = max(source_cfg.joint_route.ffn_width_tiers)
    for layer in model.model.layers:
        tail = layer.ffn.down_proj.weight[:, width:]
        assert tail.abs().max() > 0, "the widened band should exist"


def test_migrated_model_starts_with_identity_mechanisms():
    torch.manual_seed(31)
    cfg = Config.tiny()
    source_cfg = Config.tiny()
    source_cfg.num_hidden_layers = 4
    source_cfg.layer_pattern = ("KDA", "KDA", "KDA", "MLA")
    source_cfg.delta.num_blocks = 2
    source = Model(source_cfg)

    model, _ = migrate(source.state_dict(), list(source_cfg.layer_pattern), cfg)
    for layer in model.model.layers:
        assert layer.delta_attn.gate.item() == 0.0
        assert layer.delta_ffn.gate.item() == 0.0
        assert torch.equal(layer.mudd.fc2.weight, torch.zeros_like(layer.mudd.fc2.weight))
        assert torch.equal(layer.mudd.static_bias[:, -1], torch.ones(3))


def test_migrated_model_runs():
    torch.manual_seed(37)
    cfg = Config.tiny()
    source_cfg = Config.tiny()
    source_cfg.num_hidden_layers = 4
    source_cfg.layer_pattern = ("KDA", "KDA", "KDA", "MLA")
    source_cfg.delta.num_blocks = 2
    source = Model(source_cfg)
    model, _ = migrate(source.state_dict(), list(source_cfg.layer_pattern), cfg)

    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()

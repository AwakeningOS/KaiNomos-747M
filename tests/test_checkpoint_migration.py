"""Migration copies only semantically and shape-compatible 82M tensors."""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from migrate_82m_to_110m import build_source_layer_map, migrate
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_added_layers_are_not_mapped_to_source_layers():
    source = ["KDA", "KDA", "KDA", "MLA"] * 3 + ["KDA"]      # 13 layers
    target = ["KDA", "KDA", "KDA", "MLA"] * 4                # 16 layers
    mapping = build_source_layer_map(source, target)

    assert len(mapping) == 13
    for target_index, source_index in mapping.items():
        assert target_index == source_index
        assert source[source_index] == target[target_index], (
            f"layer {target_index} ({target[target_index]}) mapped from "
            f"{source_index} ({source[source_index]})"
        )
    assert all(i not in mapping for i in range(13, 16))


def test_migration_copies_exact_shapes_and_leaves_added_layers_fresh():
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

    torch.manual_seed(100)
    embedding = torch.randn(cfg.vocab_size, cfg.hidden_size)
    torch.manual_seed(101)
    fresh = Model(cfg)
    torch.manual_seed(101)
    model, stats = migrate(
        source.state_dict(), list(source_cfg.layer_pattern), cfg,
        new_embedding=embedding,
    )
    assert stats["copied"] > 0
    assert stats["shape_mismatch"], "different FFN shapes must be reported"
    assert stats["fresh_layers"] == [4, 5, 6, 7]

    # A shape-compatible existing tensor is copied.
    assert torch.equal(
        model.model.layers[0].attn.q_proj.weight,
        source.model.layers[0].attn.q_proj.weight,
    )
    # An added depth remains exactly at the target model's fresh init.
    assert torch.equal(
        model.model.layers[4].attn.q_proj.weight,
        fresh.model.layers[4].attn.q_proj.weight,
    )
    # The new table is used for both input and tied output weights.
    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_migrated_model_starts_with_identity_mechanisms():
    torch.manual_seed(31)
    cfg = Config.tiny()
    source_cfg = Config.tiny()
    source_cfg.num_hidden_layers = 4
    source_cfg.layer_pattern = ("KDA", "KDA", "KDA", "MLA")
    source_cfg.delta.num_blocks = 2
    source = Model(source_cfg)

    embedding = torch.randn(cfg.vocab_size, cfg.hidden_size)
    model, _ = migrate(
        source.state_dict(), list(source_cfg.layer_pattern), cfg,
        new_embedding=embedding,
    )
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
    embedding = torch.randn(cfg.vocab_size, cfg.hidden_size)
    model, _ = migrate(
        source.state_dict(), list(source_cfg.layer_pattern), cfg,
        new_embedding=embedding,
    )

    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()

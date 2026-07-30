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


def test_migration_recovers_the_narrower_ffn_as_a_nested_prefix():
    """A widened nested FFN must keep the 82M weight, not discard it.

    Requiring exact shape equality here is what threw away 45,416,448 trained
    parameters: the 110M FFN is the 82M FFN widened, and channel `i` denotes the
    same channel in both, so the old matrix belongs in the leading slice.
    """
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
    narrow = source_cfg.ffn_intermediate_size
    wide = cfg.ffn_intermediate_size
    assert narrow < wide, "this test needs the target FFN to be the wider one"

    torch.manual_seed(100)
    embedding = torch.randn(cfg.vocab_size, cfg.hidden_size)
    torch.manual_seed(101)
    fresh = Model(cfg)
    torch.manual_seed(101)
    model, stats = migrate(
        source.state_dict(), list(source_cfg.layer_pattern), cfg,
        new_embedding=embedding, booster_scale=0.1,
    )
    assert stats["copied"] > 0
    assert not stats["shape_mismatch"], (
        f"nothing may be dropped for shape any more: {stats['shape_mismatch']}"
    )
    assert stats["prefix_copied"], "the widened FFN must be prefix-copied"
    assert stats["fresh_layers"] == [4, 5, 6, 7]

    # A shape-compatible existing tensor is copied.
    assert torch.equal(
        model.model.layers[0].attn.q_proj.weight,
        source.model.layers[0].attn.q_proj.weight,
    )
    # The 82M channels survive exactly, in place.
    for name in ("gate_proj", "up_proj"):
        target_w = getattr(model.model.layers[0].ffn, name).weight
        source_w = getattr(source.model.layers[0].ffn, name).weight
        assert torch.equal(target_w[:narrow], source_w)
    down = model.model.layers[0].ffn.down_proj.weight
    assert torch.equal(down[:, :narrow], source.model.layers[0].ffn.down_proj.weight)

    # The reinvestment band the 82M model never had starts weak, not at full scale.
    band = down[:, narrow:]
    reference = fresh.model.layers[0].ffn.down_proj.weight[:, narrow:]
    assert torch.allclose(band, reference * 0.1)

    # An added depth is damped rather than left at full fresh scale.
    assert torch.allclose(
        model.model.layers[4].attn.output_proj.weight,
        fresh.model.layers[4].attn.output_proj.weight * 0.1,
    )
    # The new table is used for both input and tied output weights.
    assert torch.equal(model.model.embed_tokens.weight, embedding)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_final_global_attention_layer_keeps_its_role_at_the_end():
    """82M ends in an extra MLA; index alignment would throw it away.

    The 82M stack is `KDA,KDA,KDA,MLA` x3 plus a final MLA at index 12, where the
    110M stack has a KDA.  Aligning by index discards a fully trained attention
    layer even though the 110M stack ends in an MLA doing the same job.
    """
    source = ["KDA", "KDA", "KDA", "MLA"] * 3 + ["MLA"]       # 13 layers
    target = ["KDA", "KDA", "KDA", "MLA"] * 4                 # 16 layers
    mapping = build_source_layer_map(source, target)

    assert mapping[15] == 12, "the final MLA must carry over to the final MLA"
    assert 12 not in mapping, "110M layer 12 is a KDA and has no 82M counterpart"
    for target_index, source_index in mapping.items():
        assert source[source_index] == target[target_index]


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
        # Delta Block has no gate; its query is the zero-initialised one.
        assert torch.count_nonzero(layer.delta_attn.query) == 0
        assert torch.count_nonzero(layer.delta_ffn.query) == 0
        assert torch.equal(layer.mudd.fc2.weight, torch.zeros_like(layer.mudd.fc2.weight))
        streams = len(layer.streams)
        assert torch.equal(layer.mudd.static_bias[:, -1], torch.ones(streams))


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

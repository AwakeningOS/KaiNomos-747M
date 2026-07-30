"""Delta Block Attention Residuals (arXiv:2605.18855).

Two properties define the mechanism and both break silently:

* the sources are **deltas** -- the embedding, the completed block deltas, and the
  change the open block has made so far -- never the accumulated hidden state
* the routed value is **added** to the sublayer input, and the residual stream is
  updated as `h = old_h + sublayer_output`, never as
  `h = routed_input + sublayer_output`

The second is the one that looks right and is not: adding the routed delta into
the stream would carry it forward and double-count it at the next hop.

There is deliberately no gate.  The published query is zero-initialised, so at
init the softmax is uniform and the routed value is the *mean of the sources*, not
zero -- the mechanism does not start as the identity.  These tests assert the
published behaviour rather than an identity the paper does not claim.
"""

import torch

from config import K3MiniPlusPlusPlusConfig as Config
from delta_block import DeltaBank, DeltaRouter
from model import K3MiniPlusPlusPlusForCausalLM as Model


def test_bank_stores_deltas_and_the_embedding_not_hidden_states():
    embedding = torch.randn(2, 3, 8)
    bank = DeltaBank()
    bank.start(embedding)

    h = embedding + 1.0
    sources = bank.sources(h)
    assert len(sources) == 2                              # embedding + partial
    assert torch.equal(sources[0], embedding)
    assert torch.allclose(sources[1], h - embedding)

    bank.close_block(h)
    h2 = h + 2.0
    sources = bank.sources(h2)
    assert len(sources) == 3                              # embedding, delta_0, partial
    assert torch.equal(sources[0], embedding)
    assert torch.allclose(sources[1], h - embedding)      # the block's change
    assert torch.allclose(sources[2], h2 - h)             # change since the boundary
    for source in sources:                                # never the state itself
        assert not torch.allclose(source, h2)


def test_partial_delta_is_zero_immediately_after_a_boundary():
    embedding = torch.randn(1, 2, 8)
    bank = DeltaBank()
    bank.start(embedding)
    h = embedding + 3.0
    bank.close_block(h)
    assert torch.count_nonzero(bank.sources(h)[-1]) == 0


def test_router_adds_and_never_replaces():
    torch.manual_seed(0)
    router = DeltaRouter(8).double()
    torch.nn.init.normal_(router.query, std=0.5)          # a trained-ish query
    hidden = torch.randn(2, 4, 8, dtype=torch.float64)
    sources = [torch.randn(2, 4, 8, dtype=torch.float64) for _ in range(3)]

    pooled = router(hidden, sources) - hidden
    stacked = torch.stack(sources, dim=-2)
    weights = torch.einsum(
        "btsh,h->bts", router.norm(stacked), router.query
    ).softmax(-1)
    assert torch.allclose(pooled, torch.einsum("bts,btsh->bth", weights, stacked),
                          atol=1e-12)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, dtype=torch.float64))


def test_zero_initialised_query_gives_the_mean_of_the_sources():
    """The published init is uniform attention, not the identity."""
    router = DeltaRouter(8).double()
    assert torch.count_nonzero(router.query) == 0
    hidden = torch.zeros(1, 1, 8, dtype=torch.float64)
    sources = [torch.full((1, 1, 8), float(v), dtype=torch.float64) for v in (1, 2, 3)]
    assert torch.allclose(router(hidden, sources),
                          torch.full((1, 1, 8), 2.0, dtype=torch.float64))


def test_no_gate_no_low_rank_key_and_no_controller_hook():
    router = DeltaRouter(8)
    names = dict(router.named_parameters())
    assert set(names) == {"norm.weight", "query"}, sorted(names)
    assert names["query"].shape == (8,)          # w_l in R^d, full width
    try:
        router(torch.randn(1, 1, 8), [torch.randn(1, 1, 8)],
               tier=torch.ones(1, 1, 4))
    except ValueError as error:
        assert "not under controller control" in str(error)
    else:
        raise AssertionError("a controller tier must be rejected")


def test_the_residual_stream_is_the_embedding_plus_every_block_delta():
    """Structural check that sublayer outputs land on `h` and nowhere else.

    If any layer did `h = routed_input + output` the routed deltas would enter the
    stream and this sum would stop reconstructing it.
    """
    cfg = Config.tiny()
    cfg.joint_route.enabled = False
    cfg.kda_impl = "reference"
    torch.manual_seed(5)
    model = Model(cfg).double().eval()
    inner = model.model
    ids = torch.randint(0, cfg.vocab_size, (2, 12))

    banked = []
    original = DeltaBank.close_block

    def spy(self, hidden):
        original(self, hidden)
        banked.append(self.completed[-1].clone())

    DeltaBank.close_block = spy
    try:
        with torch.no_grad():
            hidden, *_ = inner(ids, respect_documents=False)
            embedding = inner.embed_tokens(ids)
    finally:
        DeltaBank.close_block = original

    assert len(banked) == cfg.delta.num_blocks, len(banked)
    reconstructed = inner.final_norm(embedding + sum(banked))
    assert torch.allclose(reconstructed, hidden, atol=1e-10), \
        float((reconstructed - hidden).abs().max())


def test_depth_queries_receive_gradient_in_both_kda_and_mla_layers():
    cfg = Config.tiny()
    cfg.joint_route.enabled = False
    cfg.kda_impl = "reference"
    torch.manual_seed(7)
    model = Model(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model(ids, labels=ids).loss.backward()

    per_kind: dict[str, list[float]] = {}
    for index, layer in enumerate(model.model.layers):
        for position in ("delta_attn", "delta_ffn"):
            grad = getattr(layer, position).query.grad
            assert grad is not None, (index, position)
            assert torch.isfinite(grad).all(), (index, position)
            per_kind.setdefault(layer.kind, []).append(float(grad.abs().sum()))

    assert set(per_kind) == {"KDA", "MLA"}
    for kind, sums in per_kind.items():
        assert any(value > 0 for value in sums), kind

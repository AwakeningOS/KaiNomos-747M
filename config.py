"""KaiNomos-110M configuration.

Everything the 82M model already fixed is inherited unchanged -- tokenizer,
vocabulary, d_model, head counts, KDA/MLA internal dimensions, NoPE, RMSNorm
eps, SiTU, weight tying, optimizer.  What this file adds is the 16-layer body,
the widened nested FFN, MUDD-QKV, the Projected Low-Rank Delta Block and MTP-1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

LAYER_PATTERN = (
    "KDA", "KDA", "KDA", "MLA",
    "KDA", "KDA", "KDA", "MLA",
    "KDA", "KDA", "KDA", "MLA",
    "KDA", "KDA", "KDA", "MLA",
)


@dataclass
class KDAConfig:
    num_heads: int = 24         # 24 * 64 == hidden_size 1536
    head_dim: int = 64
    short_conv_kernel_size: int = 4
    # Must stay at the 82M value.  Narrowing this to 32 made `f_a_proj` and
    # `f_b_proj` shape-mismatch on migration, so all nine transferable KDA layers
    # silently threw away their trained decay projections and restarted from a
    # fresh low-rank map -- 12.4M parameters of learned forgetting behaviour, for
    # 426k parameters of saving.
    decay_rank: int = 112
    gate_lower_bound: float = -5.0
    l2_eps: float = 1e-6
    a_log_init: tuple[float, float] = (1.0, 16.0)
    # Retention at initialisation for a freshly created KDA layer.  dt_bias = 0
    # gives decay = -5*sigmoid(0) = -2.5, i.e. exp(-2.5) = 0.082: a new layer
    # forgets 92% of its state every token before it has learned anything.  The
    # bias is solved from this target instead, so a fresh layer starts by
    # remembering and learns to forget.
    init_retention: float = 0.9


@dataclass
class MLAConfig:
    num_heads: int = 24         # 24 * v_head_dim 64 == hidden_size 1536
    q_lora_rank: int = 224
    kv_lora_rank: int = 224
    qk_nope_head_dim: int = 48
    qk_shared_head_dim: int = 16
    v_head_dim: int = 64

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_shared_head_dim


@dataclass
class MuDDConfig:
    """Multiway dynamic dense connections for Q, K and V only.

    Residual-direction mixing is deliberately absent: the Delta Block owns that
    direction, and running both would give two mechanisms write access to the
    same path with no way to attribute a result to either.
    """

    enabled: bool = True
    hidden: int = 64            # width of the per-layer coefficient MLP
    # Stream sets are per operator.  MLA compresses K and V into a single latent
    # and reads only one KV input, so giving it a separate V stream created
    # coefficients that no gradient ever reached.
    streams: tuple[str, ...] = ("q", "k", "v")
    mla_streams: tuple[str, ...] = ("q", "kv")

    def streams_for(self, kind: str) -> tuple[str, ...]:
        return self.mla_streams if kind == "MLA" else self.streams


@dataclass
class DeltaConfig:
    """Projected Low-Rank Delta Block.

    Values stay at full width so nothing is lost when a delta is re-added; only
    the routing key is projected down, which is where the cost actually sits.
    """

    enabled: bool = True
    # "block": sources are the completed 4-layer block deltas plus the open
    # block's partial delta -- ~L/B sources.  "sublayer": every sublayer output is
    # its own source -- 2L sources.  The paper reports sublayer slightly ahead on
    # perplexity (36.83 vs 37.08 at 220M) and Block 1.24x faster with less memory.
    granularity: str = "block"
    num_blocks: int = 4         # 16 layers / 4 = one delta per 4-layer block
    key_rank: int = 64
    tiers: tuple[int, ...] = (0, 1, 2, -1)   # -1 == ALL available sources


@dataclass
class MTPConfig:
    """One extra future token, predicted through a dedicated KDA block."""

    enabled: bool = True
    extra_tokens: int = 1
    loss_weight: float = 0.30
    ffn_width: int = 6144       # matches the backbone FFN width


@dataclass
class JointRouteConfig:
    # Off by default.  Measured on an RTX 3090 at d_model 512, grouped per-width
    # FFN dispatch ran *slower* than the dense path it was meant to beat (6 tiers
    # averaging width 1941: 6.50 ms, against 6.07 ms for a dense 2816 and 3.88 ms
    # for a dense 1920).  Splitting 6,144 tokens six ways leaves GEMMs too small
    # for the hardware, so the FLOPs the router saved on paper cost more wall
    # clock than they returned.  The machinery is kept -- it is correct, and the
    # arithmetic changes in its favour at larger widths -- but nothing deployed
    # uses it, and the controller is not built when this is False.
    enabled: bool = False
    force_fixed: bool = False
    controller_hidden: int = 64
    chunk_size: int = 64
    # A single tier: the model is dense.  A tier the policy never selects gets
    # exactly zero gradient -- the earlier run's log shows `all/1792:2176` and
    # every band above it at 0.0 from the first step to the last -- so a ladder
    # is only worth its storage if routing actually visits it.
    ffn_width_tiers: tuple[int, ...] = (6144,)
    fixed_ffn_width: int = 6144
    temperature: float = 1.0
    init_policy_bias: float = 4.0
    price: float = 0.0
    budget_ratio: float = 1.0

    @property
    def fixed_ffn_index(self) -> int:
        return self.ffn_width_tiers.index(self.fixed_ffn_width)


@dataclass
class K3MiniPlusPlusPlusConfig:
    # KaiNomos-110M has its own 32,768-piece SentencePiece tokenizer. The
    # source 82M model keeps its original 16,384-piece tokenizer unchanged.
    vocab_size: int = 32768
    # 896 with 14 heads of 64, not 512 with 8.  The 82M checkpoint that 512 was
    # chosen to inherit from turned out to be worth almost nothing -- 100,007,936
    # tokens on the predecessor's 16,384-piece English BPE, which encoded Japanese
    # at 2.56 tokens per character -- so there was no reason left to keep the
    # narrow width.  Measured on an RTX 3090, wide and shallow also runs faster
    # per parameter than deep and thin: h640/L16 at 155.8M reached 10,075 tok/s
    # where h512/L24 at 143.2M managed 8,449.
    hidden_size: int = 1536
    num_hidden_layers: int = 16
    layer_pattern: tuple[str, ...] = LAYER_PATTERN
    dense_intermediate_size: int = 6144
    # `<|eod|>` in the project tokenizer.  build_pool.py terminates every
    # document with it and it appears nowhere else, so it is what document
    # boundaries are derived from at load time.
    eod_token_id: int = 4
    context_length_train: int = 1024
    rms_norm_eps: float = 1e-6
    attn_res_block_size: int = 4
    initializer_range: float = 0.02
    # The 110M vocabulary is tied. During migration only the old input
    # embedding is reused; the old tokenizer-specific LM head is not copied.
    tie_word_embeddings: bool = True
    # "auto" uses the FLA Triton kernels when available, "reference" the
    # sequential PyTorch path used by the CPU tests
    kda_impl: str = "auto"

    kda: KDAConfig = field(default_factory=KDAConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    mudd: MuDDConfig = field(default_factory=MuDDConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    joint_route: JointRouteConfig = field(default_factory=JointRouteConfig)

    def __post_init__(self) -> None:
        if len(self.layer_pattern) != self.num_hidden_layers:
            raise ValueError("layer_pattern length must equal num_hidden_layers")
        if tuple(self.layer_pattern[:4]) != ("KDA", "KDA", "KDA", "MLA"):
            raise ValueError("core pattern must repeat KDA,KDA,KDA,MLA")
        tiers = self.joint_route.ffn_width_tiers
        if tuple(sorted(set(tiers))) != tiers:
            raise ValueError("FFN width tiers must increase uniquely")
        if self.joint_route.fixed_ffn_width != self.dense_intermediate_size:
            raise ValueError("fixed_ffn_width must equal dense_intermediate_size")
        if self.num_hidden_layers % self.delta.num_blocks != 0:
            raise ValueError("layers must divide evenly into delta blocks")
        if self.kda.num_heads * self.kda.head_dim != self.hidden_size:
            raise ValueError("KDA heads * head_dim must equal hidden_size")
        if self.mla.num_heads * self.mla.v_head_dim != self.hidden_size:
            raise ValueError("MLA heads * v_head_dim must equal hidden_size")

    @property
    def ffn_intermediate_size(self) -> int:
        """Width the FFN is built at: the supernet maximum when routing is on."""
        if not self.joint_route.enabled:
            return self.dense_intermediate_size
        return max(self.joint_route.ffn_width_tiers)

    @property
    def block_size(self) -> int:
        return self.num_hidden_layers // self.delta.num_blocks

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "K3MiniPlusPlusPlusConfig":
        data = dict(data)
        for key, klass in (("kda", KDAConfig), ("mla", MLAConfig), ("mudd", MuDDConfig),
                           ("delta", DeltaConfig), ("mtp", MTPConfig),
                           ("joint_route", JointRouteConfig)):
            if isinstance(data.get(key), dict):
                data[key] = klass(**data[key])
        for key in ("layer_pattern",):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def tiny(cls) -> "K3MiniPlusPlusPlusConfig":
        """A CPU-sized model with the same structure, for the test suite."""
        return cls(
            vocab_size=97,
            hidden_size=32,
            num_hidden_layers=8,
            layer_pattern=("KDA", "KDA", "KDA", "MLA", "KDA", "KDA", "KDA", "MLA"),
            dense_intermediate_size=48,
            context_length_train=32,
            kda_impl="reference",
            kda=KDAConfig(num_heads=4, head_dim=8, decay_rank=4),
            mla=MLAConfig(num_heads=4, q_lora_rank=8, kv_lora_rank=8,
                          qk_nope_head_dim=6, qk_shared_head_dim=2, v_head_dim=8),
            mudd=MuDDConfig(hidden=8),
            delta=DeltaConfig(num_blocks=2, key_rank=8),
            mtp=MTPConfig(ffn_width=48),
            # Routing is *on* here even though the deployed model is dense: this
            # config is what the routing tests run against, and the machinery has
            # to stay covered for as long as it stays in the tree.
            joint_route=JointRouteConfig(
                enabled=True,
                controller_hidden=16,
                ffn_width_tiers=(12, 24, 48, 60, 72, 84),
                fixed_ffn_width=48,
            ),
        )


__all__ = [
    "K3MiniPlusPlusPlusConfig", "KDAConfig", "MLAConfig", "MuDDConfig",
    "DeltaConfig", "MTPConfig", "JointRouteConfig", "LAYER_PATTERN",
]
# Public name.  The `K3MiniPlusPlusPlusConfig` / `K3MiniConfig` spellings remain
# as aliases so the ported organ modules keep importing cleanly; they are being
# retired gradually rather than in one rename that could break a running job.
KaiNomosConfig = K3MiniPlusPlusPlusConfig
K3MiniConfig = K3MiniPlusPlusPlusConfig

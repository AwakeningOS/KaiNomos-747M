"""KaiNomos-747M production configuration."""

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
    # Measured production rank for hidden size 1,536.
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
    """Delta Block Attention Residuals configuration."""

    # "block": sources are the completed 4-layer block deltas plus the open
    # block's partial delta -- ~L/B sources.  "sublayer": every sublayer output is
    # its own source -- 2L sources.  The paper reports sublayer slightly ahead on
    # perplexity (36.83 vs 37.08 at 220M) and Block 1.24x faster with less memory.
    granularity: str = "block"
    num_blocks: int = 4         # 16 layers / 4 = one delta per 4-layer block


@dataclass
class MTPConfig:
    """One extra future token, predicted through a dedicated KDA block."""

    enabled: bool = True
    extra_tokens: int = 1
    loss_weight: float = 0.30
    ffn_width: int = 6144       # matches the backbone FFN width


@dataclass
class KaiNomosConfig:
    # Frozen DoubleDragon-DataMix-v2 SentencePiece Unigram vocabulary.
    vocab_size: int = 49152
    # Measured on an RTX 3090: this dense configuration retains operational VRAM
    # headroom at micro-batch 1 with the production Muon optimizer.
    hidden_size: int = 1536
    num_hidden_layers: int = 16
    layer_pattern: tuple[str, ...] = LAYER_PATTERN
    dense_intermediate_size: int = 6144
    # `<|eod|>` in the project tokenizer.  The DataMix-v2 packer terminates every
    # document with it and it appears nowhere else, so it is what document
    # boundaries are derived from at load time.
    eod_token_id: int = 4
    context_length_train: int = 1024
    rms_norm_eps: float = 1e-6
    attn_res_block_size: int = 4
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    # "auto" uses the FLA Triton kernels when available, "reference" the
    # sequential PyTorch path used by the CPU tests
    kda_impl: str = "auto"

    kda: KDAConfig = field(default_factory=KDAConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    mudd: MuDDConfig = field(default_factory=MuDDConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)

    def __post_init__(self) -> None:
        if len(self.layer_pattern) != self.num_hidden_layers:
            raise ValueError("layer_pattern length must equal num_hidden_layers")
        if tuple(self.layer_pattern[:4]) != ("KDA", "KDA", "KDA", "MLA"):
            raise ValueError("core pattern must repeat KDA,KDA,KDA,MLA")
        if self.num_hidden_layers % self.delta.num_blocks != 0:
            raise ValueError("layers must divide evenly into delta blocks")
        if self.kda.num_heads * self.kda.head_dim != self.hidden_size:
            raise ValueError("KDA heads * head_dim must equal hidden_size")
        if self.mla.num_heads * self.mla.v_head_dim != self.hidden_size:
            raise ValueError("MLA heads * v_head_dim must equal hidden_size")

    @property
    def ffn_intermediate_size(self) -> int:
        return self.dense_intermediate_size

    @property
    def block_size(self) -> int:
        return self.num_hidden_layers // self.delta.num_blocks

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KaiNomosConfig":
        data = dict(data)
        for key, klass in (("kda", KDAConfig), ("mla", MLAConfig), ("mudd", MuDDConfig),
                           ("delta", DeltaConfig), ("mtp", MTPConfig)):
            if isinstance(data.get(key), dict):
                data[key] = klass(**data[key])
        for key in ("layer_pattern",):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @classmethod
    def tiny(cls) -> "KaiNomosConfig":
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
            delta=DeltaConfig(num_blocks=2),
            mtp=MTPConfig(ffn_width=48),
        )


__all__ = [
    "KaiNomosConfig", "KDAConfig", "MLAConfig", "MuDDConfig",
    "DeltaConfig", "MTPConfig", "LAYER_PATTERN",
]

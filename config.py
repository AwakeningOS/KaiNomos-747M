"""K3Mini+++-110M configuration.

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
    num_heads: int = 8
    head_dim: int = 64
    short_conv_kernel_size: int = 4
    decay_rank: int = 32
    gate_lower_bound: float = -5.0
    l2_eps: float = 1e-6
    a_log_init: tuple[float, float] = (1.0, 16.0)


@dataclass
class MLAConfig:
    num_heads: int = 8
    q_lora_rank: int = 128
    kv_lora_rank: int = 128
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
    streams: tuple[str, ...] = ("q", "k", "v")


@dataclass
class DeltaConfig:
    """Projected Low-Rank Delta Block.

    Values stay at full width so nothing is lost when a delta is re-added; only
    the routing key is projected down, which is where the cost actually sits.
    """

    enabled: bool = True
    num_blocks: int = 4         # 16 layers / 4 = one delta per 4-layer block
    key_rank: int = 64
    tiers: tuple[int, ...] = (0, 1, 2, -1)   # -1 == ALL available sources


@dataclass
class MTPConfig:
    """One extra future token, predicted through a dedicated KDA block."""

    enabled: bool = True
    extra_tokens: int = 1
    loss_weight: float = 0.30
    ffn_width: int = 1792       # the only knob used to land inside 109-112M


@dataclass
class JointRouteConfig:
    enabled: bool = True
    force_fixed: bool = False
    controller_hidden: int = 64
    chunk_size: int = 64
    # Core 1024 | +384 | +384 | +384 | +256 | +384
    ffn_width_tiers: tuple[int, ...] = (1024, 1408, 1792, 2176, 2432, 2816)
    fixed_ffn_width: int = 1792
    temperature: float = 1.0
    init_policy_bias: float = 4.0
    price: float = 0.0
    budget_ratio: float = 1.0

    @property
    def fixed_ffn_index(self) -> int:
        return self.ffn_width_tiers.index(self.fixed_ffn_width)


@dataclass
class K3MiniPlusPlusPlusConfig:
    vocab_size: int = 16384
    hidden_size: int = 512
    num_hidden_layers: int = 16
    layer_pattern: tuple[str, ...] = LAYER_PATTERN
    dense_intermediate_size: int = 1792     # the Base policy width
    context_length_train: int = 1024
    rms_norm_eps: float = 1e-6
    attn_res_block_size: int = 4
    initializer_range: float = 0.02
    # The 82M model keeps a separate LM head; weight tying is on the list of
    # things this version must not change.
    tie_word_embeddings: bool = False
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
            joint_route=JointRouteConfig(
                controller_hidden=16,
                ffn_width_tiers=(12, 24, 48, 60, 72, 84),
                fixed_ffn_width=48,
            ),
        )


__all__ = [
    "K3MiniPlusPlusPlusConfig", "KDAConfig", "MLAConfig", "MuDDConfig",
    "DeltaConfig", "MTPConfig", "JointRouteConfig", "LAYER_PATTERN",
]
K3MiniConfig = K3MiniPlusPlusPlusConfig  # backwards-compatible alias for the ported organ modules

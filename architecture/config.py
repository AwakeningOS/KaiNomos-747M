"""Frozen KaiNomos-750M candidate configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

ARCHITECTURE_ID = "kainomos_750m_v1"
LAYER_PATTERN = ("KDA", "KDA", "KDA", "MLA") * 6


@dataclass
class KDAConfig:
    num_heads: int = 10
    head_dim: int = 128
    value_head_dim: int = 128
    short_conv_kernel_size: int = 4
    decay_rank: int = 128
    gate_lower_bound: float = -5.0
    l2_eps: float = 1e-6
    dt_init_min: float = 1e-3
    dt_init_max: float = 1e-1
    full_rank_output_gate: bool = True


@dataclass
class MLAConfig:
    num_heads: int = 10
    q_lora_rank: int = 256
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 128
    qk_shared_head_dim: int = 0
    v_head_dim: int = 128
    qk_rmsnorm: bool = True
    nope: bool = True
    full_rank_output_gate: bool = True
    latent_only_cache: bool = True
    cache_inverse_rms: bool = True

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim


@dataclass
class DeltaConfig:
    enabled: bool = True
    granularity: str = "block"
    num_blocks: int = 6
    layers_per_block: int = 4
    routers_per_layer: int = 2
    query_init: str = "zeros"
    additive: bool = True
    preserve_main_residual: bool = True


@dataclass
class MTPConfig:
    supported: bool = True
    enabled: bool = False
    extra_tokens: int = 1
    loss_weight: float = 0.1
    ffn_width: int = 5120


@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_update_rms: float = 0.2
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class KaiNomosConfig:
    architecture_id: str = ARCHITECTURE_ID
    vocab_size: int = 49_152
    hidden_size: int = 1_280
    num_hidden_layers: int = 24
    layer_pattern: tuple[str, ...] = LAYER_PATTERN
    dense_intermediate_size: int = 5_120
    eod_token_id: int = 4
    context_length_train: int = 1_024
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    kda_impl: str = "auto"
    depth_routing: str = "delta_block"
    kda: KDAConfig = field(default_factory=KDAConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    delta: DeltaConfig = field(default_factory=DeltaConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if self.architecture_id != ARCHITECTURE_ID:
            raise ValueError(f"architecture_id must be {ARCHITECTURE_ID}")
        if len(self.layer_pattern) != self.num_hidden_layers:
            raise ValueError("layer_pattern length must equal num_hidden_layers")
        if any(tuple(self.layer_pattern[i:i + 4]) != ("KDA", "KDA", "KDA", "MLA")
               for i in range(0, self.num_hidden_layers, 4)):
            raise ValueError("every stage must be KDA,KDA,KDA,MLA")
        if self.layer_pattern[-1] != "MLA":
            raise ValueError("the final layer must be MLA")
        if self.kda.num_heads * self.kda.head_dim != self.hidden_size:
            raise ValueError("KDA heads * head_dim must equal hidden_size")
        if self.kda.value_head_dim != self.kda.head_dim:
            raise ValueError("KaiNomos-750M v1 requires equal KDA key/value dimensions")
        if self.mla.num_heads * self.mla.v_head_dim != self.hidden_size:
            raise ValueError("MLA heads * value dim must equal hidden_size")
        if self.mla.qk_shared_head_dim != 0 or not self.mla.nope:
            raise ValueError("KaiNomos-750M v1 MLA is strictly NoPE with shared dim 0")
        if self.num_hidden_layers % 4 or self.delta.layers_per_block != 4:
            raise ValueError("KaiNomos-750M v1 uses four-layer Delta stages")
        if self.delta.num_blocks != self.num_hidden_layers // 4:
            raise ValueError("delta block count must match the layer count")
        if self.depth_routing not in {"none", "delta_block"}:
            raise ValueError("depth_routing must be none or delta_block")
        if not self.tie_word_embeddings:
            raise ValueError("KaiNomos-750M v1 requires tied token embeddings")

    @property
    def ffn_intermediate_size(self) -> int:
        return self.dense_intermediate_size

    @property
    def block_size(self) -> int:
        return 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> KaiNomosConfig:
        values = dict(values)
        nested = {
            "kda": KDAConfig,
            "mla": MLAConfig,
            "delta": DeltaConfig,
            "mtp": MTPConfig,
            "optimizer": OptimizerConfig,
        }
        for name, kind in nested.items():
            if isinstance(values.get(name), dict):
                values[name] = kind(**values[name])
        if "layer_pattern" in values:
            values["layer_pattern"] = tuple(values["layer_pattern"])
        return cls(**values)

    @classmethod
    def tiny(cls, *, depth_routing: str = "delta_block", mtp: bool = False):
        return cls(
            vocab_size=97,
            hidden_size=32,
            num_hidden_layers=8,
            layer_pattern=("KDA", "KDA", "KDA", "MLA") * 2,
            dense_intermediate_size=48,
            context_length_train=32,
            kda_impl="reference",
            depth_routing=depth_routing,
            kda=KDAConfig(num_heads=4, head_dim=8, value_head_dim=8,
                          decay_rank=4),
            mla=MLAConfig(num_heads=4, q_lora_rank=8, kv_lora_rank=8,
                          qk_nope_head_dim=8, v_head_dim=8),
            delta=DeltaConfig(num_blocks=2),
            mtp=MTPConfig(enabled=mtp, ffn_width=48),
        )


__all__ = [
    "ARCHITECTURE_ID", "LAYER_PATTERN", "DeltaConfig", "KDAConfig",
    "KaiNomosConfig", "MLAConfig", "MTPConfig", "OptimizerConfig",
]

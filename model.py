"""KaiNomos-110M.

Per-layer order (spec section 6), identical in every one of the 16 layers:

    1. Delta routing before attention
    2. MUDD-QKV builds separate Q/K/V inputs from every visible depth state
    3. KDA or MLA
    4. residual add onto the *main* stream h
    5. Delta routing before the FFN
    6. nested FFN at the routed width
    7. residual add onto h, then h joins the depth bank
    8. at a block boundary, the block's delta is banked

The routed inputs `h_attn_input` / `h_ffn_input` feed the sublayers but are never
themselves the residual base: sublayer outputs always land on `h`.  Routing the
residual base would let the delta mechanism silently replace the stream instead
of contributing to it, and a gate of zero would no longer be the identity.

No layer is ever skipped.  JointRoute varies the capacity used inside a layer,
so even an easy token keeps updating its representation through all 16 layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from cache import K3MiniCache
from config import K3MiniPlusPlusPlusConfig
from delta_block import DeltaBank, DeltaKeyProjection, DeltaRouter
from joint_router import JointRouteController, RouteDecision, RouteState
from kda import KDAttention
from layers import RMSNorm, SiTUMLP
from mla import GatedMLA as MLAttention
from mtp import MTPHead, mtp_loss, mtp_slices
from mudd_qkv import MuDDQKV


@dataclass
class K3MiniOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    ntp_loss: torch.Tensor | None = None
    mtp_logits: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    cache: K3MiniCache | None = None
    joint_decisions: list = field(default_factory=list)
    expected_cost: torch.Tensor | None = None
    cost_target: float | None = None
    full_policy_cost: float | None = None


class DecoderLayer(nn.Module):
    def __init__(self, config: K3MiniPlusPlusPlusConfig, index: int):
        super().__init__()
        self.config = config
        self.index = index
        self.kind = config.layer_pattern[index]
        eps = config.rms_norm_eps
        h = config.hidden_size

        # depth sources visible to this layer: embedding + every earlier output
        self.num_sources = index + 1
        self.mudd = MuDDQKV(h, self.num_sources, config.mudd.hidden, eps,
                            config.mudd.streams) if config.mudd.enabled else None

        self.attn = KDAttention(config) if self.kind == "KDA" else MLAttention(config)
        self.ffn_norm = RMSNorm(h, eps)
        self.ffn = SiTUMLP(h, config.ffn_intermediate_size)

        self.delta_attn = DeltaRouter(h, config.delta.key_rank, eps)
        self.delta_ffn = DeltaRouter(h, config.delta.key_rank, eps)


class K3MiniPlusPlusPlus(nn.Module):
    def __init__(self, config: K3MiniPlusPlusPlusConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.delta_keys = DeltaKeyProjection(
            config.hidden_size, config.delta.key_rank,
            config.delta.num_blocks + 1, config.rms_norm_eps,
        )
        self.controller = JointRouteController(config)
        self.apply(self._init_weights)
        self.controller.reset_to_fixed_policy()
        for layer in self.layers:
            if layer.mudd is not None:
                layer.mudd.reset_to_identity()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        route_state: RouteState | None = None,
        return_route_stats: bool = False,
    ):
        from cost_model import OrganCosts

        cfg = self.config
        h = self.embed_tokens(input_ids)
        depth_states = [h]
        bank = DeltaBank()
        bank.start_block(h)

        costs = OrganCosts(cfg, input_ids.shape[1])
        if route_state is None:
            route_state = RouteState(
                price=cfg.joint_route.price,
                hard=not self.training,
                force_fixed=cfg.joint_route.force_fixed,
            )
        decisions: list[RouteDecision] = []
        memory: dict = {}

        for index, layer in enumerate(self.layers):
            # both sublayer positions see the same set: completed block deltas
            # plus the delta accumulated so far inside the current block
            n_sources = index // cfg.block_size + 1
            decision = self.controller(
                h, index, layer.kind, n_sources, n_sources,
                costs, route_state, 0, memory,
            )
            decisions.append(decision)

            values = bank.sources(h)
            keys = self.delta_keys(values) if values else []

            # 1. delta routing before attention
            h_attn_input = layer.delta_attn(
                h, keys, values, decision.attn_res_tier, cfg.delta.tiers
            )

            # 2. MUDD-QKV over the depth states this layer can see
            sources = depth_states[:-1] + [h_attn_input]
            if layer.mudd is not None:
                streams = layer.mudd(sources)
                q_in, k_in, v_in = streams["q"], streams["k"], streams["v"]
            else:
                q_in = k_in = v_in = h_attn_input

            # 3. attention; 4. residual onto the main stream, never onto the input
            kwargs = {"q_input": q_in, "k_input": k_in, "v_input": v_in}
            if layer.kind == "KDA":
                kwargs["full_mask"] = decision.kda_full
            else:
                kwargs["read_mask"] = decision.mla_read
            attn_out, _ = layer.attn(**kwargs)
            h = h + attn_out

            # 5. delta routing before the FFN, from the updated stream
            values = bank.sources(h)
            keys = self.delta_keys(values) if values else []
            h_ffn_input = layer.delta_ffn(
                h, keys, values, decision.attn_res_tier, cfg.delta.tiers
            )

            # 6-7. nested FFN at the routed width, residual onto h
            h = h + layer.ffn(
                layer.ffn_norm(h_ffn_input),
                channel_mask=decision.ffn_channel_mask,
                uniform_width=decision.ffn_uniform_width,
            )
            depth_states.append(h)

            # 8. bank the block delta at a block boundary
            if (index + 1) % cfg.block_size == 0:
                bank.close_block(h)

        hidden = self.final_norm(h)
        variable = torch.stack([d.expected_variable_cost for d in decisions]).sum()
        expected = costs.fixed_share + variable
        stats = {"depth_states": depth_states} if return_route_stats else {}
        return hidden, decisions, expected, costs, stats


class K3MiniPlusPlusPlusForCausalLM(nn.Module):
    def __init__(self, config: K3MiniPlusPlusPlusConfig):
        super().__init__()
        self.config = config
        self.model = K3MiniPlusPlusPlus(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.mtp = MTPHead(config) if config.mtp.enabled else None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        route_state: RouteState | None = None,
        use_mtp: bool = True,
    ) -> K3MiniOutput:
        hidden, decisions, expected, costs, _ = self.model(input_ids, route_state)
        logits = self.lm_head(hidden)

        loss = ntp = mtp_l = None
        mtp_logits = None
        if labels is not None:
            ntp = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            loss = ntp
            if self.mtp is not None and use_mtp and input_ids.shape[1] >= 3:
                h_sl, e_sl, t_sl = mtp_slices(input_ids)
                fused = self.mtp(hidden[:, h_sl],
                                 self.model.embed_tokens(input_ids[:, e_sl]))
                mtp_logits = self.lm_head(self.model.final_norm(fused))
                mtp_l = mtp_loss(mtp_logits, labels[:, t_sl])
                loss = loss + self.config.mtp.loss_weight * mtp_l

        return K3MiniOutput(
            logits=logits, loss=loss, ntp_loss=ntp,
            mtp_logits=mtp_logits, mtp_loss=mtp_l,
            joint_decisions=decisions, expected_cost=expected,
            cost_target=costs.budget_target, full_policy_cost=costs.full_policy_cost,
        )

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        mtp_only = sum(p.numel() for n, p in self.named_parameters() if n.startswith("mtp."))
        controller = sum(p.numel() for n, p in self.named_parameters() if "controller" in n)
        mudd = sum(p.numel() for n, p in self.named_parameters() if ".mudd." in n)
        delta = sum(p.numel() for n, p in self.named_parameters()
                    if "delta_" in n or n.startswith("model.delta_keys"))
        return {
            "total_params": total,
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "inference_backbone_params": total - mtp_only,
            "mtp_only_params": mtp_only,
            "controller_params": controller,
            "mudd_params": mudd,
            "delta_params": delta,
        }


# Public names; the K3Mini* spellings stay as aliases during the migration.
KaiNomosForCausalLM = K3MiniPlusPlusPlusForCausalLM
KaiNomosModel = K3MiniPlusPlusPlus
KaiNomosOutput = K3MiniOutput

__all__ = [
    "KaiNomosForCausalLM", "KaiNomosModel", "KaiNomosOutput",
    "K3MiniPlusPlusPlusForCausalLM", "K3MiniPlusPlusPlus", "K3MiniOutput",
]

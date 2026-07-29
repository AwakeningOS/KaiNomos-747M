"""Shared controller for discrete, budget-priced execution capacity."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import K3MiniPlusPlusPlusConfig as K3MiniConfig
from layers import RMSNorm

ORGAN_MODES = {
    "K": ("DECAY_ONLY", "FULL"),
    "M": ("BYPASS", "GLOBAL_READ"),
    # F and R are sized from the config: the FFN tier list is what decides how
    # many width modes exist, and it now extends above the Base width.
    "F": None,
    "R": None,
}


def organ_mode_counts(config) -> dict[str, int]:
    return {
        "K": 2,
        "M": 2,
        "F": len(config.joint_route.ffn_width_tiers),
        "R": len(config.delta.tiers),
    }


@dataclass
class RouteState:
    price: float = 0.0
    hard: bool = False
    force_fixed: bool = False
    static_modes: dict[str, int] | None = None
    static_patterns: dict[str, tuple[int, ...]] | None = None
    # Per-layer {organ: one-hot} replacing the sampled decision.  Feeding a run's
    # own `hard_modes` back reproduces it exactly; feeding a permuted copy is the
    # same-cost input-dependence control.
    route_override: list[dict[str, torch.Tensor]] | None = None


@dataclass
class RouteDecision:
    kda_full: torch.Tensor | None = None
    mla_read: torch.Tensor | None = None
    ffn_channel_mask: torch.Tensor | None = None
    # set when every token in the batch chose the same FFN tier, which lets the
    # FFN slice instead of mask and keeps forced-Fixed bit-exact against Base
    ffn_uniform_width: int | None = None
    attn_res_tier: torch.Tensor | None = None
    probs: dict[str, torch.Tensor] = field(default_factory=dict)
    hard_modes: dict[str, torch.Tensor] = field(default_factory=dict)
    expected_variable_cost: torch.Tensor | None = None   # executed modes (ST)
    soft_variable_cost: torch.Tensor | None = None       # soft average, diagnostics only
    # Per-mode unit costs, kept so the batch price can be re-solved from the
    # recorded probabilities without re-running the backbone.
    unit_costs: dict[str, torch.Tensor] = field(default_factory=dict)


class JointRouteController(nn.Module):
    def __init__(self, config: K3MiniConfig):
        super().__init__()
        self.config = config
        cfg = config.joint_route
        width = cfg.controller_hidden
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Two extra scalar inputs: the token's normalised position in the
        # sequence and the normalised sequence length.  Position matters because
        # what is cheap to predict early is not cheap later, and the length tells
        # the controller how expensive an MLA read currently is.
        self.fc1 = nn.Linear(config.hidden_size + 2, width, bias=False)
        self.fc2 = nn.Linear(width, width, bias=False)
        self.layer_embedding = nn.Embedding(config.num_hidden_layers, width)
        self.organ_norm = nn.ModuleDict(
            {name: RMSNorm(width, config.rms_norm_eps) for name in ORGAN_MODES}
        )
        self.head = nn.ModuleDict(
            {
                name: nn.Linear(width, count)
                for name, count in organ_mode_counts(config).items()
            }
        )
        masks = torch.zeros(len(cfg.ffn_width_tiers), config.ffn_intermediate_size)
        for row, count in enumerate(cfg.ffn_width_tiers):
            masks[row, :count] = 1
        self.register_buffer("ffn_width_masks", masks, persistent=False)
        self.reset_to_fixed_policy()

    def reset_to_fixed_policy(self) -> None:
        fixed = {"K": 1, "M": 1,
                 "F": self.config.joint_route.fixed_ffn_index,
                 "R": len(self.config.delta.tiers) - 1}
        with torch.no_grad():
            for head in self.head.values():
                head.bias.zero_()
            for organ, index in fixed.items():
                self.head[organ].bias[index] = self.config.joint_route.init_policy_bias

    def trunk(self, hidden: torch.Tensor, layer_index: int, offset: int = 0) -> torch.Tensor:
        emb = self.layer_embedding.weight[layer_index].view(1, 1, -1)
        b, t, _ = hidden.shape
        span = float(self.config.context_length_train)
        pos = (torch.arange(offset, offset + t, device=hidden.device, dtype=hidden.dtype)
               / span).view(1, t, 1).expand(b, t, 1)
        length = torch.full_like(pos, (offset + t) / span)
        features = torch.cat([self.norm(hidden), pos, length], dim=-1)
        return F.silu(self.fc2(F.silu(self.fc1(features) + emb)))

    def _choose(
        self, logits: torch.Tensor, organ: str, costs: torch.Tensor, state: RouteState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adjusted = logits.float() - state.price * costs.float()
        probs = adjusted.softmax(-1)
        fixed = {"K": 1, "M": 1,
                 "F": self.config.joint_route.fixed_ffn_index,
                 "R": len(self.config.delta.tiers) - 1}
        if state.force_fixed:
            index = torch.full(adjusted.shape[:-1] + (1,), fixed[organ], device=adjusted.device)
            onehot = torch.zeros_like(adjusted).scatter_(-1, index, 1)
            probs = onehot
        elif state.static_modes is not None and organ in state.static_modes:
            selected = int(state.static_modes[organ])
            if not 0 <= selected < adjusted.shape[-1]:
                raise ValueError(f"static mode index out of range for organ {organ}")
            index = torch.full(
                adjusted.shape[:-1] + (1,), selected, device=adjusted.device
            )
            onehot = torch.zeros_like(adjusted).scatter_(-1, index, 1)
            probs = onehot
        elif state.hard or not self.training:
            onehot = torch.zeros_like(adjusted).scatter_(
                -1, adjusted.argmax(-1, keepdim=True), 1
            )
        else:
            onehot = F.gumbel_softmax(
                adjusted, tau=self.config.joint_route.temperature, hard=True, dim=-1
            )
        return onehot, probs

    def _pattern(
        self, state: RouteState, organ: str, layer_index: int
    ) -> tuple[int, ...] | None:
        if state.static_patterns is None:
            return None
        return state.static_patterns.get(
            f"{organ}@{layer_index}", state.static_patterns.get(organ)
        )

    @staticmethod
    def _pattern_onehot(
        logits: torch.Tensor, pattern: tuple[int, ...], position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not pattern:
            raise ValueError("static route pattern must not be empty")
        selected = int(pattern[position % len(pattern)])
        if not 0 <= selected < logits.shape[-1]:
            raise ValueError("static route pattern contains an invalid mode index")
        index = torch.full(
            logits.shape[:-1] + (1,), selected, device=logits.device
        )
        onehot = torch.zeros_like(logits, dtype=torch.float32).scatter_(
            -1, index, 1
        )
        return onehot, onehot

    def _token_select(
        self, z: torch.Tensor, organ: str, costs: torch.Tensor, state: RouteState,
        layer_index: int, offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head[organ](self.organ_norm[organ](z))
        pattern = self._pattern(state, organ, layer_index)
        if pattern is None:
            return self._choose(logits, organ, costs, state)
        onehots, probs = [], []
        for local in range(z.shape[1]):
            onehot, prob = self._pattern_onehot(
                logits[:, local:local + 1], pattern, offset + local
            )
            onehots.append(onehot)
            probs.append(prob)
        return torch.cat(onehots, 1), torch.cat(probs, 1)

    def _chunk_select(
        self, z: torch.Tensor, organ: str, costs: torch.Tensor, state: RouteState,
        layer_index: int, offset: int, memory: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal chunk decisions: only each global chunk's first token is read."""
        b, t, _ = z.shape
        onehots, probabilities = [], []
        cached = memory.get((layer_index, organ))
        pattern = self._pattern(state, organ, layer_index)
        for local in range(t):
            global_pos = offset + local
            if global_pos % self.config.joint_route.chunk_size == 0 or cached is None:
                logits = self.head[organ](self.organ_norm[organ](z[:, local:local + 1]))
                cached = (
                    self._choose(logits, organ, costs, state)
                    if pattern is None
                    else self._pattern_onehot(
                        logits, pattern,
                        global_pos // self.config.joint_route.chunk_size,
                    )
                )
                memory[(layer_index, organ)] = cached
            onehot, prob = cached
            onehots.append(onehot)
            probabilities.append(prob)
        return torch.cat(onehots, 1), torch.cat(probabilities, 1)

    def forward(
        self, hidden: torch.Tensor, layer_index: int, kind: str,
        pre_past_blocks: int, post_past_blocks: int, costs, state: RouteState,
        offset: int, memory: dict,
    ) -> RouteDecision:
        z = self.trunk(hidden, layer_index, offset)
        decision = RouteDecision()
        charged: list[torch.Tensor] = []
        soft_charged: list[torch.Tensor] = []
        override = None
        if state.route_override is not None and layer_index < len(state.route_override):
            override = state.route_override[layer_index]

        def use(organ_key, onehot):
            """Substitute an externally supplied decision, if one was given."""
            if override is not None and organ_key in override:
                return override[organ_key].to(onehot.dtype)
            return onehot

        def charge(organ_key, onehot, probs, unit):
            """Charge the mode that is actually executed, not the soft average.

            `onehot` carries straight-through gradients, so the forward value is
            the FLOPs really spent while the router still receives the soft
            gradient.  Charging `probs` instead lets the executed cost drift away
            from the constrained cost -- in the predecessor experiment that gap
            reached 1.3% and made the matched-compute comparison void.
            """
            decision.unit_costs[organ_key] = unit.detach()
            charged.append((onehot.to(unit.dtype) * unit).sum(-1).mean())
            soft_charged.append((probs * unit).sum(-1).mean().detach())
        if kind == "KDA":
            onehot, probs = self._chunk_select(
                z, "K", costs.kda.to(z.device), state, layer_index, offset, memory
            )
            onehot = use("K", onehot)
            decision.kda_full = onehot[..., 1]
            organ = "K"
            unit = costs.kda.to(z.device)
        else:
            unit = costs.mla.to(z.device)
            onehot, probs = self._chunk_select(
                z, "M", unit, state, layer_index, offset, memory
            )
            onehot = use("M", onehot)
            decision.mla_read = onehot[..., 1]
            organ = "M"
        decision.probs[organ] = probs
        decision.hard_modes[organ] = onehot
        charge(organ, onehot, probs, unit)

        unit = costs.ffn.to(z.device)
        onehot, probs = self._token_select(
            z, "F", unit, state, layer_index, offset
        )
        onehot = use("F", onehot)
        decision.ffn_channel_mask = (
            onehot.to(z.dtype) @ self.ffn_width_masks.to(z.dtype)
        )
        selected = onehot.argmax(-1)
        if selected.numel() and bool((selected == selected.reshape(-1)[0]).all()):
            tiers = self.config.joint_route.ffn_width_tiers
            decision.ffn_uniform_width = int(tiers[int(selected.reshape(-1)[0])])
        decision.probs["F"] = probs
        decision.hard_modes["F"] = onehot
        charge("F", onehot, probs, unit)

        unit = costs.attn_res(pre_past_blocks, post_past_blocks).to(z.device)
        onehot, probs = self._token_select(
            z, "R", unit, state, layer_index, offset
        )
        onehot = use("R", onehot)
        decision.attn_res_tier = onehot.to(z.dtype)
        decision.probs["R"] = probs
        decision.hard_modes["R"] = onehot
        charge("R", onehot, probs, unit)
        decision.expected_variable_cost = torch.stack(charged).sum()
        decision.soft_variable_cost = torch.stack(soft_charged).sum()
        return decision


__all__ = [
    "JointRouteController", "RouteDecision", "RouteState",
    "ORGAN_MODES", "organ_mode_counts",
]

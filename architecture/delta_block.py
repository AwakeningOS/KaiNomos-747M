"""Pure functional Delta Block routing for KaiNomos-750M."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from layers import RMSNorm
from torch import nn


@dataclass(frozen=True)
class DeltaState:
    embedding: torch.Tensor
    completed: tuple[torch.Tensor, ...] = ()


@dataclass
class RouteStats:
    source_count: int
    entropy_mean: torch.Tensor
    max_weight_mean: torch.Tensor
    source_weight_mean: torch.Tensor
    query_rms: torch.Tensor
    added_rms: torch.Tensor
    residual_rms: torch.Tensor

    def detached(self) -> dict:
        return {
            "source_count": self.source_count,
            "entropy_mean": float(self.entropy_mean.detach()),
            "max_weight_mean": float(self.max_weight_mean.detach()),
            "source_weight_mean": self.source_weight_mean.detach().float().cpu().tolist(),
            "query_rms": float(self.query_rms.detach()),
            "added_rms": float(self.added_rms.detach()),
            "residual_rms": float(self.residual_rms.detach()),
            "added_to_residual_rms": float(
                self.added_rms.detach() / self.residual_rms.detach().clamp_min(1e-30)
            ),
        }


def visible_sources(
    state: DeltaState,
    hidden: torch.Tensor,
    stage_start: torch.Tensor,
    *,
    embedding_visible: bool,
    include_partial: bool,
) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = []
    if embedding_visible:
        values.append(state.embedding)
    values.extend(state.completed)
    if include_partial:
        values.append(hidden - stage_start)
    return tuple(values)


class DeltaRouter(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.source_norm = RMSNorm(hidden_size, eps)
        self.query = nn.Parameter(torch.zeros(hidden_size))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.query.zero_()
            self.source_norm.weight.fill_(1)

    def forward(
        self,
        hidden: torch.Tensor,
        sources: tuple[torch.Tensor, ...],
        *,
        return_stats: bool = False,
    ) -> tuple[torch.Tensor, RouteStats | None]:
        if not sources:
            if not return_stats:
                return hidden, None
            zero = hidden.new_zeros((), dtype=torch.float32)
            return hidden, RouteStats(
                source_count=0,
                entropy_mean=zero,
                max_weight_mean=zero,
                source_weight_mean=hidden.new_empty((0,), dtype=torch.float32),
                query_rms=self.query.float().square().mean().sqrt(),
                added_rms=zero,
                residual_rms=hidden.float().square().mean().sqrt(),
            )
        scores = torch.stack(
            [
                torch.einsum(
                    "bth,h->bt",
                    self.source_norm(source).float(),
                    self.query.float(),
                )
                for source in sources
            ],
            dim=-1,
        )
        weights = scores.softmax(-1)
        added = torch.zeros_like(hidden)
        for index, source in enumerate(sources):
            added = added + weights[..., index, None].to(source.dtype) * source
        routed = hidden + added
        if not return_stats:
            return routed, None
        probability = weights.float()
        entropy = -(
            probability * probability.clamp_min(1e-30).log()
        ).sum(-1)
        return routed, RouteStats(
            source_count=len(sources),
            entropy_mean=entropy.mean(),
            max_weight_mean=probability.max(-1).values.mean(),
            source_weight_mean=probability.mean((0, 1)),
            query_rms=self.query.float().square().mean().sqrt(),
            added_rms=added.float().square().mean().sqrt(),
            residual_rms=hidden.float().square().mean().sqrt(),
        )


def close_stage(state: DeltaState, stage_input: torch.Tensor,
                stage_output: torch.Tensor) -> DeltaState:
    return DeltaState(
        embedding=state.embedding,
        completed=(*state.completed, stage_output - stage_input),
    )


__all__ = [
    "DeltaRouter", "DeltaState", "RouteStats", "close_stage",
    "visible_sources",
]

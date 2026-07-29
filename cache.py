"""Per-layer recurrent and compressed caches."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class K3MiniCache:
    layers: list[object | None]
    seen_tokens: int = 0
    route_chunks: dict[tuple[int, str], object] = field(default_factory=dict)

    @classmethod
    def empty(cls, num_layers: int) -> "K3MiniCache":
        return cls([None] * num_layers)

    def get(self, index: int):
        return self.layers[index]

    def set(self, index: int, value) -> None:
        self.layers[index] = value

    def advance(self, count: int) -> None:
        self.seen_tokens += int(count)

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adaptive_chunking.chunkers import BaseChunker

ChunkerFactory = Callable[..., "BaseChunker"]


class ChunkerRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ChunkerFactory] = {}

    def register(self, name: str, factory: ChunkerFactory) -> None:
        if not name:
            raise ValueError("chunker name must be non-empty")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> BaseChunker:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            available = ", ".join(self.names())
            raise ValueError(f"unknown chunking strategy '{name}'. Available: {available}") from exc
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def create_many(self, names: list[str] | None = None) -> list[BaseChunker]:
        selected = self.names() if names is None else names
        return [self.create(name) for name in selected]


registry = ChunkerRegistry()

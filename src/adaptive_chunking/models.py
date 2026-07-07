from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    document_id: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            text=str(data["text"]),
            index=int(data["index"]),
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MetricScore:
    name: str
    value: float
    weight: float
    explanation: str

    @property
    def weighted_value(self) -> float:
        return self.value * self.weight

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    strategy_name: str
    chunks: list[Chunk]
    metrics: list[MetricScore]

    @property
    def score(self) -> float:
        total_weight = sum(metric.weight for metric in self.metrics)
        if total_weight == 0:
            return 0.0
        return sum(metric.weighted_value for metric in self.metrics) / total_weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "score": self.score,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


@dataclass(frozen=True)
class ChunkingResult:
    document_id: str
    strategy_name: str
    chunks: list[Chunk]
    score: float
    metrics: list[MetricScore]
    candidates: list[CandidateResult]

    def to_dict(self, *, include_candidates: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "strategy_name": self.strategy_name,
            "score": self.score,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "metrics": [metric.to_dict() for metric in self.metrics],
        }
        if include_candidates:
            payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload

    def to_json(self, *, include_candidates: bool = False, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(include_candidates=include_candidates), indent=indent)


@dataclass(frozen=True)
class ChunkingConfig:
    strategies: list[str] | None = None
    target_min_tokens: int = 120
    target_max_tokens: int = 320
    max_chunks: int | None = None

    def __post_init__(self) -> None:
        if self.strategies is not None and not self.strategies:
            raise ValueError("strategies must be omitted or contain at least one strategy")
        if self.target_min_tokens <= 0:
            raise ValueError("target_min_tokens must be positive")
        if self.target_max_tokens < self.target_min_tokens:
            raise ValueError("target_max_tokens must be greater than or equal to target_min_tokens")
        if self.max_chunks is not None and self.max_chunks <= 0:
            raise ValueError("max_chunks must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

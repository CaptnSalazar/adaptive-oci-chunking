from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    text: str
    document_id: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass(frozen=True)
class MetricScore:
    name: str
    value: float
    weight: float
    explanation: str

    @property
    def weighted_value(self) -> float:
        return self.value * self.weight


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


@dataclass(frozen=True)
class ChunkingResult:
    document_id: str
    strategy_name: str
    chunks: list[Chunk]
    score: float
    metrics: list[MetricScore]
    candidates: list[CandidateResult]


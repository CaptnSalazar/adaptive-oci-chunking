from __future__ import annotations

from adaptive_chunking.chunkers import BaseChunker, default_chunkers
from adaptive_chunking.metrics import IntrinsicMetricEvaluator, MetricConfig
from adaptive_chunking.models import CandidateResult, ChunkingConfig
from adaptive_chunking.registry import registry


class AdaptiveSelector:
    def __init__(
        self,
        chunkers: list[BaseChunker] | None = None,
        evaluator: IntrinsicMetricEvaluator | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.chunkers = chunkers or (
            registry.create_many(self.config.strategies)
            if self.config.strategies is not None
            else default_chunkers()
        )
        self.evaluator = evaluator or IntrinsicMetricEvaluator(
            MetricConfig(
                target_min_tokens=self.config.target_min_tokens,
                target_max_tokens=self.config.target_max_tokens,
            )
        )

    def rank(self, text: str) -> list[CandidateResult]:
        candidates: list[CandidateResult] = []
        for chunker in self.chunkers:
            chunks = chunker.split(text)
            if self.config.max_chunks is not None:
                chunks = chunks[: self.config.max_chunks]
            metrics = self.evaluator.evaluate(text, chunks)
            candidates.append(CandidateResult(chunker.name, chunks, metrics))
        return sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                len(candidate.chunks),
                candidate.strategy_name,
            ),
        )

    def select(self, text: str) -> CandidateResult:
        ranked = self.rank(text)
        if not ranked:
            raise ValueError("no chunking candidates were produced")
        return ranked[0]

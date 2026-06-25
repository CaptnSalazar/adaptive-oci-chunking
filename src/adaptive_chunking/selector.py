from __future__ import annotations

from adaptive_chunking.chunkers import BaseChunker, default_chunkers
from adaptive_chunking.metrics import IntrinsicMetricEvaluator
from adaptive_chunking.models import CandidateResult


class AdaptiveSelector:
    def __init__(
        self,
        chunkers: list[BaseChunker] | None = None,
        evaluator: IntrinsicMetricEvaluator | None = None,
    ) -> None:
        self.chunkers = chunkers or default_chunkers()
        self.evaluator = evaluator or IntrinsicMetricEvaluator()

    def rank(self, text: str) -> list[CandidateResult]:
        candidates: list[CandidateResult] = []
        for chunker in self.chunkers:
            chunks = chunker.split(text)
            metrics = self.evaluator.evaluate(text, chunks)
            candidates.append(CandidateResult(chunker.name, chunks, metrics))
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def select(self, text: str) -> CandidateResult:
        ranked = self.rank(text)
        if not ranked:
            raise ValueError("no chunking candidates were produced")
        return ranked[0]


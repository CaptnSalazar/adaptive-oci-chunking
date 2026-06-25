from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from adaptive_chunking.models import Chunk, MetricScore
from adaptive_chunking.text import (
    cosine_bow,
    find_block_boundaries,
    find_references,
    lexical_entropy,
    token_count,
    words,
)


@dataclass(frozen=True)
class MetricWeights:
    references_completeness: float = 1.0
    intrachunk_cohesion: float = 1.2
    document_contextual_coherence: float = 1.0
    block_integrity: float = 0.8
    size_compliance: float = 1.0
    coverage: float = 1.0
    overlap_control: float = 0.7
    boundary_quality: float = 0.9
    semantic_drift: float = 0.9
    information_density: float = 0.6
    redundancy: float = 0.5


@dataclass(frozen=True)
class MetricConfig:
    target_min_tokens: int = 120
    target_max_tokens: int = 320
    reference_window_chars: int = 120
    boundary_tolerance_chars: int = 24
    max_overlap_ratio: float = 0.20
    weights: MetricWeights = MetricWeights()


class IntrinsicMetricEvaluator:
    def __init__(self, config: MetricConfig | None = None) -> None:
        self.config = config or MetricConfig()

    def evaluate(self, text: str, chunks: list[Chunk]) -> list[MetricScore]:
        weights = self.config.weights
        return [
            MetricScore(
                name="references_completeness",
                value=self.references_completeness(text, chunks),
                weight=weights.references_completeness,
                explanation="References should not be separated from nearby context.",
            ),
            MetricScore(
                name="intrachunk_cohesion",
                value=self.intrachunk_cohesion(chunks),
                weight=weights.intrachunk_cohesion,
                explanation="Sentences inside a chunk should remain topically related.",
            ),
            MetricScore(
                name="document_contextual_coherence",
                value=self.document_contextual_coherence(chunks),
                weight=weights.document_contextual_coherence,
                explanation=(
                    "Adjacent chunks should preserve document flow without over-fragmenting."
                ),
            ),
            MetricScore(
                name="block_integrity",
                value=self.block_integrity(text, chunks),
                weight=weights.block_integrity,
                explanation="Chunk boundaries should align with paragraphs or headings.",
            ),
            MetricScore(
                name="size_compliance",
                value=self.size_compliance(chunks),
                weight=weights.size_compliance,
                explanation="Chunks should stay within the configured token range.",
            ),
            MetricScore(
                name="coverage",
                value=self.coverage(text, chunks),
                weight=weights.coverage,
                explanation=(
                    "Chunk spans should cover the source document without dropping content."
                ),
            ),
            MetricScore(
                name="overlap_control",
                value=self.overlap_control(chunks),
                weight=weights.overlap_control,
                explanation=(
                    "Overlap should be intentional, bounded, and not duplicate too much text."
                ),
            ),
            MetricScore(
                name="boundary_quality",
                value=self.boundary_quality(text, chunks),
                weight=weights.boundary_quality,
                explanation="Chunk boundaries should avoid cutting through words or sentences.",
            ),
            MetricScore(
                name="semantic_drift",
                value=self.semantic_drift(chunks),
                weight=weights.semantic_drift,
                explanation="Adjacent chunks should not swing abruptly between unrelated topics.",
            ),
            MetricScore(
                name="information_density",
                value=self.information_density(chunks),
                weight=weights.information_density,
                explanation=(
                    "Chunks should contain meaningful lexical variety rather than boilerplate."
                ),
            ),
            MetricScore(
                name="redundancy",
                value=self.redundancy(chunks),
                weight=weights.redundancy,
                explanation="Candidate chunks should avoid repeated near-duplicate content.",
            ),
        ]

    def references_completeness(self, text: str, chunks: list[Chunk]) -> float:
        references = find_references(text)
        if not references:
            return 1.0
        complete = 0
        for start, end in references:
            window_start = max(0, start - self.config.reference_window_chars)
            window_end = min(len(text), end + self.config.reference_window_chars)
            if any(
                chunk.start_char <= window_start and chunk.end_char >= window_end
                for chunk in chunks
            ):
                complete += 1
        return _clamp(complete / len(references))

    def intrachunk_cohesion(self, chunks: list[Chunk]) -> float:
        scores: list[float] = []
        for chunk in chunks:
            midpoint = len(chunk.text) // 2
            left = chunk.text[:midpoint]
            right = chunk.text[midpoint:]
            if token_count(chunk.text) < 12:
                continue
            scores.append(cosine_bow(left, right))
        return _clamp(mean(scores)) if scores else 1.0

    def document_contextual_coherence(self, chunks: list[Chunk]) -> float:
        if len(chunks) <= 1:
            return 1.0
        scores = [
            cosine_bow(left.text, right.text)
            for left, right in zip(chunks, chunks[1:], strict=False)
        ]
        return _clamp(mean(scores)) if scores else 1.0

    def block_integrity(self, text: str, chunks: list[Chunk]) -> float:
        boundaries = find_block_boundaries(text)
        if not chunks:
            return 0.0
        aligned = 0
        for chunk in chunks:
            start_ok = _near_boundary(
                chunk.start_char,
                boundaries,
                self.config.boundary_tolerance_chars,
            )
            end_ok = _near_boundary(
                chunk.end_char,
                boundaries,
                self.config.boundary_tolerance_chars,
            )
            aligned += int(start_ok and end_ok)
        return _clamp(aligned / len(chunks))

    def size_compliance(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        scores: list[float] = []
        for chunk in chunks:
            count = token_count(chunk.text)
            if self.config.target_min_tokens <= count <= self.config.target_max_tokens:
                scores.append(1.0)
            elif count < self.config.target_min_tokens:
                scores.append(max(0.0, count / self.config.target_min_tokens))
            else:
                scores.append(max(0.0, self.config.target_max_tokens / count))
        return _clamp(mean(scores))

    def coverage(self, text: str, chunks: list[Chunk]) -> float:
        if not text.strip():
            return 1.0
        covered = [False] * len(text)
        for chunk in chunks:
            for index in range(max(0, chunk.start_char), min(len(text), chunk.end_char)):
                if not text[index].isspace():
                    covered[index] = True
        non_space = sum(not character.isspace() for character in text)
        if non_space == 0:
            return 1.0
        return _clamp(sum(covered) / non_space)

    def overlap_control(self, chunks: list[Chunk]) -> float:
        if len(chunks) <= 1:
            return 1.0
        overlap_chars = 0
        total_chars = sum(max(0, chunk.end_char - chunk.start_char) for chunk in chunks)
        for left, right in zip(chunks, chunks[1:], strict=False):
            overlap_chars += max(0, left.end_char - right.start_char)
        if total_chars == 0:
            return 0.0
        ratio = overlap_chars / total_chars
        if ratio <= self.config.max_overlap_ratio:
            return 1.0
        return _clamp(1.0 - (ratio - self.config.max_overlap_ratio))

    def boundary_quality(self, text: str, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        scores: list[float] = []
        for chunk in chunks:
            start_score = _character_boundary_score(text, chunk.start_char, left=False)
            end_score = _character_boundary_score(text, chunk.end_char, left=True)
            scores.append((start_score + end_score) / 2)
        return _clamp(mean(scores))

    def semantic_drift(self, chunks: list[Chunk]) -> float:
        if len(chunks) <= 1:
            return 1.0
        similarities = [
            cosine_bow(left.text, right.text)
            for left, right in zip(chunks, chunks[1:], strict=False)
        ]
        if not similarities:
            return 1.0
        avg_similarity = mean(similarities)
        variance = mean((score - avg_similarity) ** 2 for score in similarities)
        return _clamp(1.0 - variance)

    def information_density(self, chunks: list[Chunk]) -> float:
        if not chunks:
            return 0.0
        scores: list[float] = []
        for chunk in chunks:
            terms = words(chunk.text)
            if not terms:
                scores.append(0.0)
                continue
            unique_ratio = len(set(terms)) / len(terms)
            scores.append((lexical_entropy(chunk.text) + unique_ratio) / 2)
        return _clamp(mean(scores))

    def redundancy(self, chunks: list[Chunk]) -> float:
        if len(chunks) <= 1:
            return 1.0
        similarities: list[float] = []
        for index, chunk in enumerate(chunks):
            for other in chunks[index + 1 :]:
                similarities.append(cosine_bow(chunk.text, other.text))
        if not similarities:
            return 1.0
        return _clamp(1.0 - mean(similarities))


def _near_boundary(position: int, boundaries: set[int], tolerance: int) -> bool:
    return any(abs(position - boundary) <= tolerance for boundary in boundaries)


def _character_boundary_score(text: str, position: int, *, left: bool) -> float:
    if position <= 0 or position >= len(text):
        return 1.0
    character = text[position - 1] if left else text[position]
    neighbor = text[position] if left else text[position - 1]
    if character.isspace() or neighbor.isspace():
        return 1.0
    if character in ".!?;:,)]}" or neighbor in "([{":
        return 0.9
    if character.isalnum() and neighbor.isalnum():
        return 0.2
    return 0.6


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

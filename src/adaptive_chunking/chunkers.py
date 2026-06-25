from __future__ import annotations

import re
from abc import ABC, abstractmethod

from adaptive_chunking.models import Chunk
from adaptive_chunking.text import cosine_bow, normalize_space, sentences


class BaseChunker(ABC):
    name: str

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        raise NotImplementedError

    def _build_chunks(self, spans: list[tuple[int, int]], text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, (start, end) in enumerate(spans):
            chunk_text = normalize_space(text[start:end])
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, index=len(chunks), start_char=start, end_char=end))
        return chunks


class FixedWindowChunker(BaseChunker):
    name = "fixed-window"

    def __init__(self, chunk_size: int = 1200, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[Chunk]:
        spans: list[tuple[int, int]] = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            spans.append((start, end))
            if end == len(text):
                break
        return self._build_chunks(spans, text)


class SingleChunker(BaseChunker):
    name = "single"

    def split(self, text: str) -> list[Chunk]:
        return self._build_chunks([(0, len(text))], text)


class DelimiterChunker(BaseChunker):
    name = "delimiter"

    def __init__(
        self,
        delimiter: str = "\n---\n",
        keep_delimiter: bool = False,
        max_size: int = 1800,
    ) -> None:
        self.delimiter = delimiter
        self.keep_delimiter = keep_delimiter
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        if not self.delimiter or self.delimiter not in text:
            return self.fallback.split(text)
        spans: list[tuple[int, int]] = []
        cursor = 0
        while cursor < len(text):
            split_at = text.find(self.delimiter, cursor)
            if split_at < 0:
                spans.append((cursor, len(text)))
                break
            end = split_at + len(self.delimiter) if self.keep_delimiter else split_at
            spans.append((cursor, end))
            cursor = split_at + len(self.delimiter)
        return self._split_oversized_spans(spans, text)

    def _split_oversized_spans(self, spans: list[tuple[int, int]], text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for start, end in spans:
            if end - start <= self.max_size:
                chunks.extend(self._build_chunks([(start, end)], text))
                continue
            for chunk in self.fallback.split(text[start:end]):
                chunks.append(
                    Chunk(chunk.text, len(chunks), start + chunk.start_char, start + chunk.end_char)
                )
        return chunks


class PageChunker(DelimiterChunker):
    name = "page"

    def __init__(self, page_delimiter: str = "\f", max_size: int = 2200) -> None:
        super().__init__(delimiter=page_delimiter, keep_delimiter=False, max_size=max_size)


class RecursiveChunker(BaseChunker):
    name = "recursive"

    def __init__(self, chunk_size: int = 1200, separators: tuple[str, ...] | None = None) -> None:
        self.chunk_size = chunk_size
        self.separators = separators or ("\n\n", "\n", ". ", " ")

    def split(self, text: str) -> list[Chunk]:
        spans = self._split_span(text, 0, len(text), 0)
        return self._build_chunks(spans, text)

    def _split_span(self, text: str, start: int, end: int, separator_index: int) -> list[tuple[int, int]]:
        if end - start <= self.chunk_size:
            return [(start, end)]
        if separator_index >= len(self.separators):
            return [
                (cursor, min(cursor + self.chunk_size, end))
                for cursor in range(start, end, self.chunk_size)
            ]

        separator = self.separators[separator_index]
        pieces: list[tuple[int, int]] = []
        cursor = start
        while cursor < end:
            limit = min(cursor + self.chunk_size, end)
            split_at = text.rfind(separator, cursor, limit)
            if split_at <= cursor:
                pieces.extend(self._split_span(text, cursor, limit, separator_index + 1))
                cursor = limit
            else:
                split_end = split_at + len(separator)
                pieces.append((cursor, split_end))
                cursor = split_end
        return pieces


class SplitThenMergeChunker(BaseChunker):
    name = "split-then-merge"

    def __init__(self, min_size: int = 600, max_size: int = 1400) -> None:
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.min_size = min_size
        self.max_size = max_size

    def split(self, text: str) -> list[Chunk]:
        raw_spans = _paragraph_spans(text)
        if not raw_spans:
            return []
        merged: list[tuple[int, int]] = []
        start, end = raw_spans[0]
        for next_start, next_end in raw_spans[1:]:
            proposed_size = next_end - start
            if proposed_size <= self.max_size or end - start < self.min_size:
                end = next_end
            else:
                merged.append((start, end))
                start, end = next_start, next_end
        merged.append((start, end))
        return self._build_chunks(merged, text)


class SectionAwareChunker(BaseChunker):
    name = "section-aware"

    def __init__(self, min_size: int = 500, max_size: int = 1800) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = SplitThenMergeChunker(min_size=min_size, max_size=max_size)
        self.heading_pattern = re.compile(
            r"(?m)^(?:#{1,6}\s+.+|\d+(?:\.\d+)*\s+[A-Z].+|[A-Z][A-Za-z0-9 ,:;&()/-]{3,80})$"
        )

    def split(self, text: str) -> list[Chunk]:
        starts = sorted({0, len(text), *(match.start() for match in self.heading_pattern.finditer(text))})
        spans = [(starts[index], starts[index + 1]) for index in range(len(starts) - 1)]
        merged: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        for start, end in spans:
            if not text[start:end].strip():
                continue
            if current_start is None:
                current_start, current_end = start, end
                continue
            assert current_end is not None
            proposed = end - current_start
            if proposed <= self.max_size or current_end - current_start < self.min_size:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        if current_start is not None and current_end is not None:
            merged.append((current_start, current_end))
        chunks: list[Chunk] = []
        for start, end in merged:
            if end - start <= self.max_size:
                chunks.extend(self._build_chunks([(start, end)], text))
            else:
                for chunk in self.fallback.split(text[start:end]):
                    chunks.append(
                        Chunk(chunk.text, len(chunks), start + chunk.start_char, start + chunk.end_char)
                    )
        return chunks


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(
        self,
        max_size: int = 1400,
        min_size: int = 350,
        similarity_threshold: float = 0.10,
    ) -> None:
        self.max_size = max_size
        self.min_size = min_size
        self.similarity_threshold = similarity_threshold

    def split(self, text: str) -> list[Chunk]:
        sentence_spans = _sentence_spans(text)
        if not sentence_spans:
            return []
        chunks: list[tuple[int, int]] = []
        start, end = sentence_spans[0]
        previous_text = text[start:end]
        for next_start, next_end in sentence_spans[1:]:
            next_text = text[next_start:next_end]
            proposed_size = next_end - start
            similarity = cosine_bow(previous_text, next_text)
            should_break = (
                end - start >= self.min_size
                and (proposed_size > self.max_size or similarity < self.similarity_threshold)
            )
            if should_break:
                chunks.append((start, end))
                start, end = next_start, next_end
            else:
                end = next_end
            previous_text = next_text
        chunks.append((start, end))
        return self._build_chunks(chunks, text)


class RegexSectionChunker(BaseChunker):
    name = "regex-section"

    def __init__(self, max_size: int = 1800, heading_pattern: str | None = None) -> None:
        self.max_size = max_size
        self.heading_pattern = re.compile(
            heading_pattern or r"(?m)^(?:#{1,6}\s+.+|\d+(?:\.\d+)*\s+[A-Z].+)$"
        )
        self.fallback = RecursiveChunker(chunk_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        starts = [match.start() for match in self.heading_pattern.finditer(text)]
        if not starts:
            return self.fallback.split(text)
        starts = sorted(set([0, *starts, len(text)]))
        spans = [(starts[index], starts[index + 1]) for index in range(len(starts) - 1)]
        chunks: list[Chunk] = []
        for start, end in spans:
            section = text[start:end]
            if len(section) <= self.max_size:
                chunks.extend(self._build_chunks([(start, end)], text))
            else:
                for chunk in self.fallback.split(section):
                    chunks.append(
                        Chunk(
                            text=chunk.text,
                            index=len(chunks),
                            start_char=start + chunk.start_char,
                            end_char=start + chunk.end_char,
                        )
                    )
        return [Chunk(c.text, i, c.start_char, c.end_char, c.metadata) for i, c in enumerate(chunks)]


def default_chunkers() -> list[BaseChunker]:
    return [
        SingleChunker(),
        FixedWindowChunker(),
        RecursiveChunker(),
        SplitThenMergeChunker(),
        SectionAwareChunker(),
        DelimiterChunker(),
        PageChunker(),
        SemanticChunker(),
        RegexSectionChunker(),
    ]


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n", text):
        end = match.end()
        if text[cursor:end].strip():
            spans.append((cursor, end))
        cursor = end
    if text[cursor:].strip():
        spans.append((cursor, len(text)))
    return spans or ([(0, len(text))] if text.strip() else [])


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences(text):
        start = text.find(sentence, cursor)
        if start < 0:
            continue
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans

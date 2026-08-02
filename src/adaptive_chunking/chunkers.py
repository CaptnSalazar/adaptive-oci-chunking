from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from typing import Any

from adaptive_chunking.models import Chunk
from adaptive_chunking.text import cosine_bow, normalize_space


class BaseChunker(ABC):
    name: str

    @abstractmethod
    def split(self, text: str) -> list[Chunk]:
        raise NotImplementedError

    def _build_chunks(
        self,
        spans: list[tuple[int, int]],
        text: str,
        start_index: int = 0,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for start, end in spans:
            start, end = _trim_span(text, start, end)
            chunk_text = normalize_space(text[start:end])
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=start_index + len(chunks),
                        start_char=start,
                        end_char=end,
                    )
                )
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
        if not delimiter:
            raise ValueError("delimiter must be non-empty")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
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
                chunks.extend(self._build_chunks([(start, end)], text, start_index=len(chunks)))
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

    def split(self, text: str) -> list[Chunk]:
        if not self.delimiter or self.delimiter not in text:
            return [
                Chunk(
                    text=chunk.text,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={**chunk.metadata, "page_index": 0},
                )
                for chunk in self.fallback.split(text)
            ]

        chunks: list[Chunk] = []
        cursor = 0
        page_index = 0
        while cursor < len(text):
            split_at = text.find(self.delimiter, cursor)
            end = len(text) if split_at < 0 else split_at
            for chunk in self._split_page(text, cursor, end):
                chunks.append(
                    Chunk(
                        text=chunk.text,
                        index=len(chunks),
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={**chunk.metadata, "page_index": page_index},
                    )
                )
            if split_at < 0:
                break
            cursor = split_at + len(self.delimiter)
            page_index += 1
        return chunks

    def _split_page(self, text: str, start: int, end: int) -> list[Chunk]:
        if end - start <= self.max_size:
            return self._build_chunks([(start, end)], text)
        return [
            Chunk(
                chunk.text,
                chunk.index,
                start + chunk.start_char,
                start + chunk.end_char,
                chunk.metadata,
            )
            for chunk in self.fallback.split(text[start:end])
        ]


class PageIndexChunker(BaseChunker):
    name = "page-index"

    def __init__(self, page_delimiter: str = "\f", max_size: int = 1800) -> None:
        if not page_delimiter:
            raise ValueError("page_delimiter must be non-empty")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.page_delimiter = page_delimiter
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size)
        self.heading_pattern = re.compile(
            r"(?m)^(#{1,6})\s+(.+)$|^(\d+(?:\.\d+)*)\s+([A-Z].+)$"
        )

    def split(self, text: str) -> list[Chunk]:
        page_spans = self._page_spans(text)
        chunks: list[Chunk] = []
        for page_index, (page_start, page_end) in enumerate(page_spans):
            page_text = text[page_start:page_end]
            section_spans = self._section_spans(page_text, text)
            for section_index, (section_start, section_end) in enumerate(section_spans):
                absolute_start = page_start + section_start
                absolute_end = page_start + section_end
                metadata = {
                    **_section_metadata(text, absolute_start, len(chunks)),
                    "page_index": page_index,
                    "page_start_char": page_start,
                    "page_end_char": page_end,
                    "page_section_index": section_index,
                    "heading_path": self._heading_path(page_text, section_start, text),
                }
                section_text = text[absolute_start:absolute_end]
                section_chunks = (
                    self._build_chunks(
                        [(absolute_start, absolute_end)],
                        text,
                        start_index=len(chunks),
                    )
                    if len(section_text) <= self.max_size
                    else self.fallback.split(section_text)
                )
                for chunk in section_chunks:
                    start_char = (
                        chunk.start_char
                        if len(section_text) <= self.max_size
                        else absolute_start + chunk.start_char
                    )
                    end_char = (
                        chunk.end_char
                        if len(section_text) <= self.max_size
                        else absolute_start + chunk.end_char
                    )
                    chunks.append(
                        Chunk(
                            text=chunk.text,
                            index=len(chunks),
                            start_char=start_char,
                            end_char=end_char,
                            metadata={**chunk.metadata, **metadata},
                        )
                    )
        return chunks

    def _page_spans(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        spans: list[tuple[int, int]] = []
        cursor = 0
        while cursor <= len(text):
            split_at = text.find(self.page_delimiter, cursor)
            if split_at < 0:
                spans.append((cursor, len(text)))
                break
            spans.append((cursor, split_at))
            cursor = split_at + len(self.page_delimiter)
        return [span for span in spans if text[span[0] : span[1]].strip()]

    def _section_spans(self, page_text: str, source_text: str) -> list[tuple[int, int]]:
        starts = sorted(
            {
                0,
                len(page_text),
                *(
                    match.start()
                    for match in self.heading_pattern.finditer(page_text)
                    if not _is_ignored_heading(_heading_title(match), source_text)
                ),
            }
        )
        return [
            (starts[index], starts[index + 1])
            for index in range(len(starts) - 1)
            if page_text[starts[index] : starts[index + 1]].strip()
        ]

    def _heading_path(self, page_text: str, section_start: int, source_text: str) -> list[str]:
        path: list[str] = []
        for match in self.heading_pattern.finditer(page_text):
            if match.start() > section_start:
                break
            markdown_level, markdown_title, numbered_level, numbered_title = match.groups()
            title = _heading_title(match)
            if _is_ignored_heading(title, source_text):
                continue
            level = len(markdown_level) if markdown_level else numbered_level.count(".") + 1
            path = path[: level - 1]
            path.append(title.strip())
        return path


class RecursiveChunker(BaseChunker):
    name = "recursive"

    def __init__(self, chunk_size: int = 1200, separators: tuple[str, ...] | None = None) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.separators = separators or ("\n\n", "\n", ". ", " ")

    def split(self, text: str) -> list[Chunk]:
        spans = self._split_span(text, 0, len(text), 0)
        return self._build_chunks(spans, text)

    def _split_span(
        self,
        text: str,
        start: int,
        end: int,
        separator_index: int,
    ) -> list[tuple[int, int]]:
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
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = SplitThenMergeChunker(min_size=min_size, max_size=max_size)
        self.oversize_fallback = RecursiveChunker(chunk_size=max_size)
        self.heading_pattern = re.compile(
            r"(?m)^(?:#{1,6}\s+.+|\d+(?:\.\d+)*\s+[A-Z].+|[A-Z][A-Za-z0-9 ,:;&()/-]{3,80})$"
        )

    def split(self, text: str) -> list[Chunk]:
        starts = sorted(
            {
                0,
                len(text),
                *(
                    match.start()
                    for match in self.heading_pattern.finditer(text)
                    if not _is_ignored_heading(_heading_title(match), text)
                ),
            }
        )
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
        for section_index, (start, end) in enumerate(merged):
            metadata = _section_metadata(text, start, section_index)
            if end - start <= self.max_size:
                for chunk in self._build_chunks([(start, end)], text, start_index=len(chunks)):
                    chunks.append(
                        Chunk(
                            chunk.text,
                            chunk.index,
                            chunk.start_char,
                            chunk.end_char,
                            metadata={**chunk.metadata, **metadata},
                        )
                    )
            else:
                fallback_chunks = self.fallback.split(text[start:end])
                if len(fallback_chunks) == 1 and fallback_chunks[0].size > self.max_size:
                    fallback_chunks = self.oversize_fallback.split(text[start:end])
                for chunk in fallback_chunks:
                    chunks.append(
                        Chunk(
                            chunk.text,
                            len(chunks),
                            start + chunk.start_char,
                            start + chunk.end_char,
                            metadata={**chunk.metadata, **metadata},
                        )
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
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        self.max_size = max_size
        self.min_size = min_size
        self.similarity_threshold = similarity_threshold

    def split(self, text: str) -> list[Chunk]:
        semantic_spans = _semantic_unit_spans(text)
        if not semantic_spans:
            return []
        chunks: list[tuple[int, int]] = []
        start, end = semantic_spans[0]
        previous_text = text[start:end]
        for next_start, next_end in semantic_spans[1:]:
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
        if max_size <= 0:
            raise ValueError("max_size must be positive")
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
        for section_index, (start, end) in enumerate(spans):
            section = text[start:end]
            metadata = _section_metadata(text, start, section_index)
            if len(section) <= self.max_size:
                for chunk in self._build_chunks([(start, end)], text, start_index=len(chunks)):
                    chunks.append(
                        Chunk(
                            chunk.text,
                            chunk.index,
                            chunk.start_char,
                            chunk.end_char,
                            metadata={**chunk.metadata, **metadata},
                        )
                    )
            else:
                for chunk in self.fallback.split(section):
                    chunks.append(
                        Chunk(
                            text=chunk.text,
                            index=len(chunks),
                            start_char=start + chunk.start_char,
                            end_char=start + chunk.end_char,
                            metadata={**chunk.metadata, **metadata},
                        )
                    )
        return [
            Chunk(c.text, i, c.start_char, c.end_char, c.metadata)
            for i, c in enumerate(chunks)
        ]


class TokenWindowChunker(BaseChunker):
    name = "token-window"

    def __init__(self, chunk_tokens: int = 240, overlap_tokens: int = 24) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    def split(self, text: str) -> list[Chunk]:
        tokens = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        if not tokens:
            return []
        spans: list[tuple[int, int]] = []
        step = self.chunk_tokens - self.overlap_tokens
        for token_start in range(0, len(tokens), step):
            token_end = min(token_start + self.chunk_tokens, len(tokens))
            spans.append((tokens[token_start][0], tokens[token_end - 1][1]))
            if token_end == len(tokens):
                break
        return self._build_chunks(spans, text)


class SentenceChunker(BaseChunker):
    name = "sentence"

    def __init__(self, min_size: int = 400, max_size: int = 1400) -> None:
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        spans = _semantic_unit_spans(text)
        if not spans:
            return []
        merged = _merge_spans(spans, self.min_size, self.max_size)
        return _split_oversized(self, merged, text, self.max_size)


class ParagraphChunker(BaseChunker):
    name = "paragraph"

    def __init__(self, min_size: int = 500, max_size: int = 1600) -> None:
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        spans = _paragraph_spans(text)
        if not spans:
            return []
        merged = _merge_spans(spans, self.min_size, self.max_size)
        return _split_oversized(self, merged, text, self.max_size)


class MarkdownChunker(BaseChunker):
    name = "markdown"

    def __init__(self, min_size: int = 500, max_size: int = 1800) -> None:
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.min_size = min_size
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size, separators=("\n\n", "\n", " "))
        self.heading_pattern = re.compile(r"(?m)^#{1,6}\s+.+$")

    def split(self, text: str) -> list[Chunk]:
        spans = _markdown_block_spans(text)
        if not spans:
            return []
        heading_starts = {
            match.start()
            for match in self.heading_pattern.finditer(text)
            if not _is_ignored_heading(_heading_title(match), text)
        }
        section_spans: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        for start, end in spans:
            if current_start is None or start in heading_starts:
                if current_start is not None and current_end is not None:
                    section_spans.append((current_start, current_end))
                current_start, current_end = start, end
            else:
                current_end = end
        if current_start is not None and current_end is not None:
            section_spans.append((current_start, current_end))
        merged = _merge_spans(section_spans, self.min_size, self.max_size)
        chunks = _split_oversized(self, merged, text, self.max_size)
        return [
            Chunk(
                chunk.text,
                index,
                chunk.start_char,
                chunk.end_char,
                {**chunk.metadata, **_section_metadata(text, chunk.start_char, index)},
            )
            for index, chunk in enumerate(chunks)
        ]


class HtmlChunker(BaseChunker):
    name = "html"

    def __init__(self, min_size: int = 500, max_size: int = 1800) -> None:
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.text_chunker = ParagraphChunker(min_size=min_size, max_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        parser = _HTMLTextExtractor()
        parser.feed(text)
        extracted = parser.text()
        chunks = self.text_chunker.split(extracted)
        return [
            Chunk(
                chunk.text,
                chunk.index,
                0,
                len(text),
                {**chunk.metadata, "source_format": "html"},
            )
            for chunk in chunks
        ]


class JsonChunker(BaseChunker):
    name = "json"

    def __init__(self, max_size: int = 1800) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return self.fallback.split(text)
        records = list(_json_records(data))
        if not records:
            return []
        chunks: list[Chunk] = []
        for path, value in records:
            serialized = json.dumps(value, ensure_ascii=False, indent=2)
            if len(serialized) <= self.max_size:
                chunks.append(
                    Chunk(serialized, len(chunks), 0, len(text), {"json_path": path})
                )
                continue
            for chunk in self.fallback.split(serialized):
                chunks.append(
                    Chunk(
                        chunk.text,
                        len(chunks),
                        0,
                        len(text),
                        {"json_path": path, "source_format": "json"},
                    )
                )
        return chunks


class CodeChunker(BaseChunker):
    name = "code"

    def __init__(self, max_size: int = 1800) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.fallback = RecursiveChunker(chunk_size=max_size, separators=("\n\n", "\n", " "))
        self.pattern = re.compile(
            r"(?m)^(?:class|def|async def|function|export function|const\s+\w+\s*=|"
            r"public\s+class|private\s+|public\s+|protected\s+)"
        )

    def split(self, text: str) -> list[Chunk]:
        starts = sorted({0, len(text), *(match.start() for match in self.pattern.finditer(text))})
        spans = [
            (starts[index], starts[index + 1])
            for index in range(len(starts) - 1)
            if text[starts[index] : starts[index + 1]].strip()
        ]
        if not spans:
            return []
        return _split_oversized(self, spans, text, self.max_size)


class HybridChunker(BaseChunker):
    name = "hybrid"

    def __init__(self, min_size: int = 500, max_size: int = 1800) -> None:
        if min_size <= 0 or max_size <= min_size:
            raise ValueError("expected 0 < min_size < max_size")
        self.markdown = MarkdownChunker(min_size=min_size, max_size=max_size)
        self.section = SectionAwareChunker(min_size=min_size, max_size=max_size)
        self.paragraph = ParagraphChunker(min_size=min_size, max_size=max_size)

    def split(self, text: str) -> list[Chunk]:
        if re.search(r"(?m)^#{1,6}\s+.+$", text) or "```" in text:
            return self.markdown.split(text)
        if re.search(r"(?m)^(?:\d+(?:\.\d+)*\s+[A-Z].+|[A-Z][A-Za-z0-9 ,:;&()/-]{3,80})$", text):
            return self.section.split(text)
        return self.paragraph.split(text)


def default_chunkers() -> list[BaseChunker]:
    return [
        SingleChunker(),
        FixedWindowChunker(),
        TokenWindowChunker(),
        RecursiveChunker(),
        SentenceChunker(),
        ParagraphChunker(),
        SplitThenMergeChunker(),
        SectionAwareChunker(),
        MarkdownChunker(),
        DelimiterChunker(),
        PageChunker(),
        PageIndexChunker(),
        SemanticChunker(),
        RegexSectionChunker(),
        CodeChunker(),
        HybridChunker(),
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


def _semantic_unit_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|(?=\n)|$)", text):
        start, end = _trim_span(text, match.start(), match.end())
        if start < end:
            spans.append((start, end))
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _merge_spans(
    spans: list[tuple[int, int]],
    min_size: int,
    max_size: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    start, end = spans[0]
    for next_start, next_end in spans[1:]:
        proposed_size = next_end - start
        if proposed_size <= max_size or end - start < min_size:
            end = next_end
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def _split_oversized(
    chunker: BaseChunker,
    spans: list[tuple[int, int]],
    text: str,
    max_size: int,
) -> list[Chunk]:
    fallback = RecursiveChunker(chunk_size=max_size)
    chunks: list[Chunk] = []
    for start, end in spans:
        if end - start <= max_size:
            chunks.extend(chunker._build_chunks([(start, end)], text, start_index=len(chunks)))
            continue
        for chunk in fallback.split(text[start:end]):
            chunks.append(
                Chunk(chunk.text, len(chunks), start + chunk.start_char, start + chunk.end_char)
            )
    return chunks


def _markdown_block_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lines = text.splitlines(keepends=True)
    cursor = 0
    block_start = 0
    in_fence = False
    for line in lines:
        line_start = cursor
        cursor += len(line)
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        if not in_fence and not stripped:
            if text[block_start:line_start].strip():
                spans.append((block_start, line_start))
            block_start = cursor
    if text[block_start:].strip():
        spans.append((block_start, len(text)))
    return spans


class _HTMLTextExtractor(HTMLParser):
    block_tags = {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.block_tags:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.block_tags:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return normalize_space(" ".join(self.parts).replace("\n\n", "\n\n"))


def _json_records(data: Any, path: str = "$") -> list[tuple[str, Any]]:
    if isinstance(data, list):
        return [(f"{path}[{index}]", value) for index, value in enumerate(data)]
    if isinstance(data, dict):
        nested = [
            (f"{path}.{key}", value)
            for key, value in data.items()
            if isinstance(value, (dict, list))
        ]
        return nested or [(path, data)]
    return [(path, data)]


def _section_metadata(text: str, start: int, section_index: int) -> dict[str, object]:
    heading = text[start:].splitlines()[0].strip() if text[start:].strip() else ""
    title = re.sub(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*\s+)", "", heading).strip() or "document"
    if _is_ignored_heading(title, text):
        title = "document"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"
    return {
        "section_index": section_index,
        "section_title": title,
        "section_instance_id": f"section-{section_index}-{slug}",
    }


def add_section_paths(text: str, chunks: list[Chunk]) -> list[Chunk]:
    """Attach the active structural path to every chunk.

    Repeated page furniture such as headers and footers is deliberately excluded
    from the heading stream, so it cannot become a retrieval section.
    """
    headings = _section_heading_events(text)
    enriched: list[Chunk] = []
    for chunk in chunks:
        path: list[str] = []
        for position, level, title in headings:
            if position > chunk.start_char:
                break
            path = path[: level - 1]
            path.append(title)
        enriched.append(
            Chunk(
                text=chunk.text,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={**chunk.metadata, "section_path": path},
            )
        )
    return enriched


def _section_heading_events(text: str) -> list[tuple[int, int, str]]:
    pattern = re.compile(
        r"(?m)^(?:(?P<markdown>#{1,6})\s+(?P<markdown_title>.+)|"
        r"(?P<number>\d+(?:\.\d+)*)\s+(?P<number_title>[A-Z].+)|"
        r"(?P<title>[A-Z][A-Za-z0-9 ,:;&()/-]{3,80}))$"
    )
    events: list[tuple[int, int, str]] = []
    for match in pattern.finditer(text):
        title = _heading_title(match)
        if _is_ignored_heading(title, text):
            continue
        markdown_level = match.groupdict().get("markdown")
        number = match.groupdict().get("number")
        level = len(markdown_level) if markdown_level else (number.count(".") + 1 if number else 1)
        events.append((match.start(), level, title))
    return events


def _heading_title(match: re.Match[str]) -> str:
    groups = match.groupdict()
    title = groups.get("markdown_title") or groups.get("number_title") or groups.get("title")
    if title is None:
        title = match.group()
    return re.sub(r"^(?:#{1,6}\s+|\d+(?:\.\d+)*\s+)", "", title).strip()


def _is_ignored_heading(title: str, source_text: str) -> bool:
    """Return true for page furniture, not meaningful document section headings."""
    normalized = re.sub(r"\s+", " ", title).strip().casefold()
    if not normalized:
        return True
    if re.fullmatch(
        r"(?:header|footer|page\s*\d+(?:\s*(?:of|/)\s*\d+)?|"
        r"confidential|internal use only|copyright(?:\s+\d{4})?)",
        normalized,
    ):
        return True
    return False


def _register_default_chunkers() -> None:
    from adaptive_chunking.registry import registry

    for chunker_type in (
        SingleChunker,
        FixedWindowChunker,
        TokenWindowChunker,
        RecursiveChunker,
        SentenceChunker,
        ParagraphChunker,
        SplitThenMergeChunker,
        SectionAwareChunker,
        MarkdownChunker,
        DelimiterChunker,
        PageChunker,
        PageIndexChunker,
        SemanticChunker,
        RegexSectionChunker,
        HtmlChunker,
        JsonChunker,
        CodeChunker,
        HybridChunker,
    ):
        registry.register(chunker_type.name, chunker_type)


_register_default_chunkers()

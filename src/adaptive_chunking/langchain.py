from __future__ import annotations

from typing import Any

from adaptive_chunking.pipeline import AdaptiveChunker

try:
    from langchain_text_splitters import TextSplitter
except ImportError:  # pragma: no cover
    TextSplitter = None  # type: ignore[assignment]

_BaseTextSplitter = TextSplitter if TextSplitter is not None else object


class LangChainAdaptiveTextSplitter(_BaseTextSplitter):  # type: ignore[misc,valid-type]
    """LangChain TextSplitter backed by AdaptiveChunker."""

    def __init__(
        self,
        chunker: AdaptiveChunker | None = None,
        keep_separator: bool = False,
        **kwargs: Any,
    ) -> None:
        if TextSplitter is None:  # pragma: no cover
            raise RuntimeError(
                "Install LangChain support with `pip install -e .[langchain]`."
            )
        super().__init__(keep_separator=keep_separator, **kwargs)
        self.chunker = chunker or AdaptiveChunker()

    def split_text(self, text: str) -> list[str]:
        result = self.chunker.chunk(text)
        return [chunk.text for chunk in result.chunks]

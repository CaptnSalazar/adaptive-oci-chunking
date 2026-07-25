from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from adaptive_chunking.pipeline import AdaptiveChunker

try:
    from langchain_core.documents import Document as LangChainDocument
    from langchain_text_splitters import TextSplitter
except ImportError:  # pragma: no cover
    LangChainDocument = None  # type: ignore[assignment]
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
                "Install LangChain support with "
                "`pip install 'adaptive-oci-chunking[langchain]'`."
            )
        super().__init__(keep_separator=keep_separator, **kwargs)
        self.chunker = chunker or AdaptiveChunker()

    def split_text(self, text: str) -> list[str]:
        result = self.chunker.chunk(text)
        return [chunk.text for chunk in result.chunks]

    def create_documents(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[Any, Any]] | None = None,
    ) -> list[Any]:
        """Create LangChain documents without discarding adaptive chunk metadata."""
        if LangChainDocument is None:  # pragma: no cover
            raise RuntimeError(
                "Install LangChain support with "
                "`pip install 'adaptive-oci-chunking[langchain]'`."
            )

        metadata_items = metadatas if metadatas is not None else [{} for _ in texts]
        if len(metadata_items) != len(texts):
            raise ValueError("metadatas must contain one item for every text")

        documents: list[Any] = []
        for index, (text, metadata) in enumerate(zip(texts, metadata_items, strict=True)):
            documents.extend(
                self._chunk_to_documents(
                    text,
                    dict(metadata),
                    fallback_document_id=f"document-{index}",
                )
            )
        return documents

    def split_documents(self, documents: Iterable[Any]) -> list[Any]:
        """Split LangChain documents and preserve source and chunk provenance."""
        output: list[Any] = []
        for index, document in enumerate(documents):
            output.extend(
                self._chunk_to_documents(
                    document.page_content,
                    dict(document.metadata or {}),
                    fallback_document_id=f"document-{index}",
                )
            )
        return output

    def transform_documents(self, documents: Sequence[Any], **_: Any) -> list[Any]:
        """Support LangChain's document-transformer convention."""
        return self.split_documents(documents)

    def _chunk_to_documents(
        self,
        text: str,
        source_metadata: dict[str, Any],
        *,
        fallback_document_id: str,
    ) -> list[Any]:
        document_id = str(source_metadata.get("document_id", fallback_document_id))
        result = self.chunker.chunk(text, document_id=document_id)
        return [
            LangChainDocument(
                page_content=chunk.text,
                metadata={
                    **source_metadata,
                    **chunk.metadata,
                    "document_id": document_id,
                    "chunk_index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "strategy_name": result.strategy_name,
                    "adaptive_score": result.score,
                },
            )
            for chunk in result.chunks
        ]

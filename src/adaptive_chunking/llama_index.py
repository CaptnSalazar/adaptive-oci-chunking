from __future__ import annotations

from typing import Any

from adaptive_chunking.models import Chunk, ChunkingResult
from adaptive_chunking.pipeline import AdaptiveChunker


def chunks_to_llama_nodes(
    chunks: list[Chunk],
    *,
    document_id: str = "document",
    extra_metadata: dict[str, Any] | None = None,
) -> list[Any]:
    try:
        from llama_index.core.schema import TextNode
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install LlamaIndex support with `pip install -e .[llama-index]`."
        ) from exc

    metadata = extra_metadata or {}
    return [
        TextNode(
            text=chunk.text,
            id_=f"{document_id}:{chunk.index}",
            metadata={
                **metadata,
                **chunk.metadata,
                "document_id": document_id,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            },
        )
        for chunk in chunks
    ]


def result_to_llama_nodes(result: ChunkingResult) -> list[Any]:
    return chunks_to_llama_nodes(
        result.chunks,
        document_id=result.document_id,
        extra_metadata={"strategy_name": result.strategy_name, "adaptive_score": result.score},
    )


class LlamaIndexAdaptiveParser:
    """Small adapter with the same practical behavior as a LlamaIndex node parser."""

    def __init__(self, chunker: AdaptiveChunker | None = None) -> None:
        self.chunker = chunker or AdaptiveChunker()

    def get_nodes_from_documents(self, documents: list[Any], **_: Any) -> list[Any]:
        nodes: list[Any] = []
        for document_index, document in enumerate(documents):
            text = getattr(document, "text", None) or getattr(document, "get_content", lambda: "")()
            metadata = dict(getattr(document, "metadata", {}) or {})
            document_id = str(
                metadata.get("document_id") or getattr(document, "id_", document_index)
            )
            result = self.chunker.chunk(text, document_id=document_id)
            nodes.extend(
                chunks_to_llama_nodes(
                    result.chunks,
                    document_id=document_id,
                    extra_metadata={
                        **metadata,
                        "strategy_name": result.strategy_name,
                        "adaptive_score": result.score,
                    },
                )
            )
        return nodes

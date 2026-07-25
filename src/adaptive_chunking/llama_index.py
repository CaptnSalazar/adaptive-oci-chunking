from __future__ import annotations

from typing import Any

from adaptive_chunking.models import Chunk, ChunkingResult
from adaptive_chunking.pipeline import AdaptiveChunker

try:
    from llama_index.core.node_parser import NodeParser
    from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, TextNode
    from pydantic import Field
except ImportError:  # pragma: no cover
    Field = None  # type: ignore[assignment]
    NodeParser = None  # type: ignore[assignment]
    BaseNode = Any  # type: ignore[assignment,misc]
    MetadataMode = None  # type: ignore[assignment]
    NodeRelationship = None  # type: ignore[assignment]
    TextNode = None  # type: ignore[assignment]


def chunks_to_llama_nodes(
    chunks: list[Chunk],
    *,
    document_id: str = "document",
    extra_metadata: dict[str, Any] | None = None,
) -> list[Any]:
    if TextNode is None:  # pragma: no cover
        raise RuntimeError(
            "Install LlamaIndex support with "
            "`pip install 'adaptive-oci-chunking[llama-index]'`."
        )

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


if NodeParser is not None:

    class LlamaIndexAdaptiveParser(NodeParser):
        """A native LlamaIndex NodeParser backed by :class:`AdaptiveChunker`.

        The parser preserves document metadata, chunk offsets, and adaptive-selection
        diagnostics while creating source and neighbouring-node relationships through
        LlamaIndex's standard ``NodeParser`` lifecycle.
        """

        chunker: AdaptiveChunker = Field(default_factory=AdaptiveChunker, exclude=True)

        def _parse_nodes(
            self,
            nodes: list[BaseNode],
            show_progress: bool = False,
            **_: Any,
        ) -> list[BaseNode]:
            parsed_nodes: list[BaseNode] = []
            for source_node in nodes:
                document_id = str(source_node.id_)
                text = source_node.get_content(metadata_mode=MetadataMode.NONE)
                result = self.chunker.chunk(text, document_id=document_id)
                for chunk in result.chunks:
                    parsed_nodes.append(
                        TextNode(
                            text=chunk.text,
                            id_=self.id_func(chunk.index, source_node),
                            metadata={
                                **chunk.metadata,
                                "document_id": document_id,
                                "chunk_index": chunk.index,
                                "start_char": chunk.start_char,
                                "end_char": chunk.end_char,
                                "strategy_name": result.strategy_name,
                                "adaptive_score": result.score,
                            },
                            start_char_idx=chunk.start_char,
                            end_char_idx=chunk.end_char,
                            relationships={
                                NodeRelationship.SOURCE: source_node.as_related_node_info()
                            },
                        )
                    )
            return parsed_nodes

else:

    class LlamaIndexAdaptiveParser:
        """Placeholder that explains how to enable the optional dependency."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "Install LlamaIndex support with "
                "`pip install 'adaptive-oci-chunking[llama-index]'`."
            )

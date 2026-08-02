from __future__ import annotations

from adaptive_chunking.models import ChunkingConfig, ChunkingResult, Document
from adaptive_chunking.selector import AdaptiveSelector


class AdaptiveChunker:
    def __init__(
        self,
        selector: AdaptiveSelector | None = None,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.selector = selector or AdaptiveSelector(config=self.config)

    def chunk(self, text: str, document_id: str = "document") -> ChunkingResult:
        return self.chunk_document(Document(text=text, document_id=document_id))

    def chunk_file(self, path: str, document_id: str | None = None) -> ChunkingResult:
        """Load a supported text file or PDF and chunk its extracted text."""
        from adaptive_chunking.io import load_document

        document = load_document(path)
        if document_id is not None:
            document = Document(
                text=document.text,
                document_id=document_id,
                metadata=document.metadata,
            )
        return self.chunk_document(document)

    def chunk_document(self, document: Document) -> ChunkingResult:
        candidates = self.selector.rank(document.text)
        if not candidates:
            raise ValueError("no chunking candidates were produced")
        best = candidates[0]
        return ChunkingResult(
            document_id=document.document_id,
            strategy_name=best.strategy_name,
            chunks=best.chunks,
            score=best.score,
            metrics=best.metrics,
            candidates=candidates,
        )

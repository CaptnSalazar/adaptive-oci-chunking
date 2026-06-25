from __future__ import annotations

from adaptive_chunking.models import ChunkingResult, Document
from adaptive_chunking.selector import AdaptiveSelector


class AdaptiveChunker:
    def __init__(self, selector: AdaptiveSelector | None = None) -> None:
        self.selector = selector or AdaptiveSelector()

    def chunk(self, text: str, document_id: str = "document") -> ChunkingResult:
        document = Document(text=text, document_id=document_id)
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


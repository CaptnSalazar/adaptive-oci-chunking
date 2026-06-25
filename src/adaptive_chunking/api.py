from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive_chunking.pipeline import AdaptiveChunker

try:
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API support with `pip install -e .[api]`.") from exc


class ChunkRequest(BaseModel):
    text: str = Field(min_length=1)
    document_id: str = "document"


class ChunkResponse(BaseModel):
    document_id: str
    strategy_name: str
    score: float
    chunks: list[dict]
    metrics: list[dict]


app = FastAPI(title="Adaptive OCI Chunking")
chunker = AdaptiveChunker()


@app.post("/chunk", response_model=ChunkResponse)
def chunk(request: ChunkRequest) -> dict:
    result = chunker.chunk(request.text, document_id=request.document_id)
    return {
        "document_id": result.document_id,
        "strategy_name": result.strategy_name,
        "score": result.score,
        "chunks": [chunk.__dict__ for chunk in result.chunks],
        "metrics": [metric.__dict__ for metric in result.metrics],
    }


from __future__ import annotations

from pydantic import BaseModel, Field

from adaptive_chunking.models import ChunkingConfig
from adaptive_chunking.pipeline import AdaptiveChunker

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Install API support with `pip install 'adaptive-oci-chunking[api]'`."
    ) from exc


class ChunkRequest(BaseModel):
    text: str = Field(min_length=1)
    document_id: str = "document"
    strategies: list[str] | None = None
    include_candidates: bool = False


class ChunkResponse(BaseModel):
    document_id: str
    strategy_name: str
    score: float
    chunks: list[dict]
    metrics: list[dict]
    candidates: list[dict] | None = None


app = FastAPI(title="Adaptive OCI Chunking")


@app.post("/chunk", response_model=ChunkResponse)
def chunk(request: ChunkRequest) -> dict:
    try:
        chunker = AdaptiveChunker(config=ChunkingConfig(strategies=request.strategies))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = chunker.chunk(request.text, document_id=request.document_id)
    return result.to_dict(include_candidates=request.include_candidates)

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_chunking import AdaptiveChunker  # noqa: E402


def build_vector_records(path: Path) -> list[dict[str, Any]]:
    """Create vector-store-ready records from an adaptively chunked document.

    The records use a common shape accepted by many vector database SDKs:
    an id, text payload, and metadata. Add embeddings with your preferred
    embedding model before upserting.
    """
    result = AdaptiveChunker().chunk_file(str(path))
    return [
        {
            "id": f"{result.document_id}:{chunk.index}",
            "text": chunk.text,
            "metadata": {
                **chunk.metadata,
                "document_id": result.document_id,
                "chunk_index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "strategy_name": result.strategy_name,
                "adaptive_score": result.score,
            },
        }
        for chunk in result.chunks
    ]


def to_chroma(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert records to the argument shape used by Chroma collection.add/upsert."""
    return {
        "ids": [record["id"] for record in records],
        "documents": [record["text"] for record in records],
        "metadatas": [record["metadata"] for record in records],
    }


def to_qdrant_points(
    records: list[dict[str, Any]], vectors: list[list[float]]
) -> list[dict[str, Any]]:
    """Convert records and precomputed vectors to Qdrant point dictionaries."""
    return [
        {
            "id": record["id"],
            "vector": vector,
            "payload": {
                "text": record["text"],
                **record["metadata"],
            },
        }
        for record, vector in zip(records, vectors, strict=True)
    ]


def to_pinecone_vectors(
    records: list[dict[str, Any]], vectors: list[list[float]]
) -> list[dict[str, Any]]:
    """Convert records and precomputed vectors to Pinecone upsert dictionaries."""
    return [
        {
            "id": record["id"],
            "values": vector,
            "metadata": {
                "text": record["text"],
                **record["metadata"],
            },
        }
        for record, vector in zip(records, vectors, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vector-store-ready records from adaptive chunks."
    )
    parser.add_argument("path", type=Path, help="Text, Markdown, reStructuredText, or PDF file.")
    args = parser.parse_args()

    records = build_vector_records(args.path)
    chroma_payload = to_chroma(records)

    print(f"Built {len(records)} records.")
    print("First record:")
    print(records[0] if records else "<none>")
    print("\nChroma payload keys:")
    print({key: len(value) for key, value in chroma_payload.items()})


if __name__ == "__main__":
    main()

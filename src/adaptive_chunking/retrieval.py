from __future__ import annotations

from collections.abc import Iterable

from adaptive_chunking.models import Chunk


def expand_section_instances(
    chunks: Iterable[Chunk],
    fetched_chunks: Iterable[Chunk],
) -> list[Chunk]:
    fetched = list(fetched_chunks)
    section_instance_ids = {
        chunk.metadata.get("section_instance_id")
        for chunk in fetched
        if chunk.metadata.get("section_instance_id") is not None
    }
    if not section_instance_ids:
        return sorted(fetched, key=lambda chunk: chunk.index)

    expanded = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("section_instance_id") in section_instance_ids
    ]
    return sorted(expanded, key=lambda chunk: chunk.index)

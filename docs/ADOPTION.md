# Adoption checklist

Use this checklist before promoting a release to the RAG community.

## Positioning

- Lead with “Adaptive Chunking for RAG,” not OCI.
- Explain the core promise in one sentence: automatically choose the best splitter per document and show why.
- Keep OCI framed as an optional adapter for teams that need it.

## Proof

- Run `python benchmarks/compare_chunkers.py <your-docs> --format markdown`.
- Add at least one benchmark table to the README or release post.
- For serious claims, include retrieval-level evaluation: recall@k, answer correctness, latency, and token cost.

## Examples

- Keep `examples/vector_store_ingestion.py` working as the canonical vector-store recipe.
- Add provider-specific examples as demand appears: Chroma, Qdrant, Pinecone, Weaviate, FAISS.
- Keep LangChain and LlamaIndex examples current with their public APIs.

## Launch assets

- Record a short demo using `demos/streamlit_app.py`.
- Publish a post showing one messy document where a default splitter fails and adaptive chunking preserves useful context.
- Submit to relevant lists and communities: RAG, LangChain, LlamaIndex, vector DB, and document AI communities.

## Release hygiene

- Confirm `pyproject.toml` version matches the PyPI release.
- Run `ruff check .`, `pytest`, and `python -m compileall src tests examples benchmarks demos`.
- Tag the release and include benchmark/demo notes in the changelog or GitHub release.

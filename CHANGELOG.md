# Changelog

## 0.3.0 - Unreleased

### Added

- Repositioned project documentation around cloud-neutral adaptive chunking for RAG.
- Added `benchmarks/compare_chunkers.py` for comparing adaptive selection with baseline strategies.
- Added `examples/benchmark_policy.md` as a more realistic structured benchmark document.
- Added `examples/vector_store_ingestion.py` with Chroma, Qdrant, and Pinecone-style payload helpers.
- Added `demos/streamlit_app.py` and a `demo` optional extra for local visual inspection.
- Added `docs/ADOPTION.md` with positioning, proof, example, launch, and release guidance.

### Changed

- Candidate ranking now preserves configured chunker order as the final tie-breaker instead of sorting ties alphabetically.

### Release notes

- The package name remains `adaptive-oci-chunking`; OCI support is still optional.
- PyPI currently has `0.2.0` as the latest published release, so `0.3.0` is ready to publish after final maintainer approval.

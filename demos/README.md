# Demo playground

The Streamlit playground gives new users a quick way to inspect what adaptive chunking is doing before they wire it into a RAG system.

```bash
pip install "adaptive-oci-chunking[api]" streamlit
streamlit run demos/streamlit_app.py
```

Use it to compare:

- the selected strategy;
- candidate rankings;
- per-metric explanations;
- final chunk text and metadata.

This demo is intentionally local-only. It does not call an embedding model or vector database.

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_chunking import AdaptiveChunker, ChunkingConfig  # noqa: E402

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - optional demo dependency
    raise RuntimeError("Install Streamlit with `pip install streamlit`.") from exc


st.set_page_config(page_title="Adaptive Chunking Playground", layout="wide")
st.title("Adaptive Chunking Playground")
st.caption("Inspect selected chunks, candidate rankings, and metric explanations locally.")

uploaded_file = st.file_uploader("Upload a text, Markdown, reStructuredText, or PDF file")
manual_text = st.text_area(
    "Or paste text",
    height=220,
    placeholder="# Handbook\n\nPaste a document here to inspect chunking behavior.",
)
strategy_options = [
    "code",
    "delimiter",
    "fixed-window",
    "html",
    "json",
    "markdown",
    "page",
    "page-index",
    "paragraph",
    "recursive",
    "regex-section",
    "section-aware",
    "semantic",
    "sentence",
    "single",
    "split-then-merge",
    "token-window",
]
strategies = st.multiselect(
    "Limit candidate strategies",
    strategy_options,
    default=[],
    help="Leave empty to evaluate all built-in strategies.",
)

if st.button("Chunk document", type="primary"):
    config = ChunkingConfig(strategies=strategies or None)
    chunker = AdaptiveChunker(config=config)

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / f"uploaded{suffix}"
            temp_path.write_bytes(uploaded_file.getvalue())
            result = chunker.chunk_file(str(temp_path), document_id=Path(uploaded_file.name).stem)
    elif manual_text.strip():
        result = chunker.chunk(manual_text, document_id="pasted-text")
    else:
        st.warning("Upload a file or paste text first.")
        st.stop()

    st.subheader("Selected result")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Strategy", result.strategy_name)
    col_b.metric("Score", f"{result.score:.3f}")
    col_c.metric("Chunks", len(result.chunks))

    st.subheader("Candidate ranking")
    st.dataframe(
        [
            {
                "rank": index + 1,
                "strategy": candidate.strategy_name,
                "score": round(candidate.score, 3),
                "chunks": len(candidate.chunks),
            }
            for index, candidate in enumerate(result.candidates)
        ],
        use_container_width=True,
    )

    st.subheader("Selected metrics")
    st.dataframe(
        [
            {
                "metric": metric.name,
                "value": round(metric.value, 3),
                "weight": metric.weight,
                "explanation": metric.explanation,
            }
            for metric in result.metrics
        ],
        use_container_width=True,
    )

    st.subheader("Chunks")
    for chunk in result.chunks:
        with st.expander(f"Chunk {chunk.index} - chars {chunk.start_char}:{chunk.end_char}"):
            st.write(chunk.metadata)
            st.code(chunk.text)

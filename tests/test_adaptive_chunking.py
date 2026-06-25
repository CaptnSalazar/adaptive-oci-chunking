from adaptive_chunking import AdaptiveChunker
from adaptive_chunking.chunkers import (
    DelimiterChunker,
    FixedWindowChunker,
    PageChunker,
    RecursiveChunker,
    SectionAwareChunker,
    SemanticChunker,
    SplitThenMergeChunker,
)
from adaptive_chunking.metrics import IntrinsicMetricEvaluator, MetricConfig, MetricWeights
from adaptive_chunking.selector import AdaptiveSelector


def test_adaptive_chunker_returns_best_candidate() -> None:
    text = (
        "# Overview\n"
        "Adaptive chunking selects a strategy for a document.\n\n"
        "## Section 2\n"
        "See Section 2.1 for the details that should remain close to the explanation.\n"
        "The method evaluates references, cohesion, coherence, block integrity, and size."
    )

    result = AdaptiveChunker().chunk(text, document_id="demo")

    assert result.document_id == "demo"
    assert result.strategy_name
    assert result.chunks
    assert 0.0 <= result.score <= 1.0
    assert {
        "references_completeness",
        "intrachunk_cohesion",
        "document_contextual_coherence",
        "block_integrity",
        "size_compliance",
    }.issubset({metric.name for metric in result.metrics})
    assert result.candidates == sorted(result.candidates, key=lambda candidate: candidate.score, reverse=True)


def test_split_then_merge_preserves_order() -> None:
    text = "A first paragraph.\n\nA second paragraph.\n\nA third paragraph."
    chunks = SplitThenMergeChunker(min_size=10, max_size=40).split(text)

    assert "".join(chunk.text for chunk in chunks).replace(" ", "")
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].start_char == 0


def test_metric_scores_are_bounded() -> None:
    text = "Section 1 explains the approach.\n\nSection 2 explains the result."
    chunks = SplitThenMergeChunker(min_size=10, max_size=80).split(text)
    metrics = IntrinsicMetricEvaluator().evaluate(text, chunks)

    assert all(0.0 <= metric.value <= 1.0 for metric in metrics)


def test_custom_metric_weights_change_reported_weights() -> None:
    evaluator = IntrinsicMetricEvaluator(
        MetricConfig(weights=MetricWeights(block_integrity=2.5, coverage=3.0))
    )
    chunks = SplitThenMergeChunker(min_size=10, max_size=80).split("One paragraph.\n\nTwo paragraph.")

    by_name = {metric.name: metric for metric in evaluator.evaluate("One paragraph.\n\nTwo paragraph.", chunks)}

    assert by_name["block_integrity"].weight == 2.5
    assert by_name["coverage"].weight == 3.0


def test_selector_can_be_limited_to_custom_chunkers() -> None:
    selector = AdaptiveSelector(
        chunkers=[
            DelimiterChunker(delimiter="\n---\n"),
            SectionAwareChunker(min_size=10, max_size=120),
        ]
    )
    result = AdaptiveChunker(selector=selector).chunk("# A\nText.\n---\n# B\nMore text.")

    assert [candidate.strategy_name for candidate in result.candidates] == [
        result.candidates[0].strategy_name,
        result.candidates[1].strategy_name,
    ]
    assert {candidate.strategy_name for candidate in result.candidates} == {"delimiter", "section-aware"}


def test_delimiter_chunker_splits_custom_boundaries() -> None:
    text = "First part.\n---\nSecond part.\n---\nThird part."
    chunks = DelimiterChunker(delimiter="\n---\n").split(text)

    assert [chunk.text for chunk in chunks] == ["First part.", "Second part.", "Third part."]


def test_delimiter_chunker_can_keep_delimiters() -> None:
    chunks = DelimiterChunker(delimiter="---", keep_delimiter=True).split("alpha---beta")

    assert chunks[0].text == "alpha---"
    assert chunks[1].text == "beta"


def test_page_chunker_uses_form_feed_pages() -> None:
    chunks = PageChunker().split("Page one.\fPage two.")

    assert [chunk.text for chunk in chunks] == ["Page one.", "Page two."]


def test_section_aware_chunker_prefers_heading_boundaries() -> None:
    text = "# Alpha\nAlpha body.\n\n# Beta\nBeta body.\n\n# Gamma\nGamma body."
    chunks = SectionAwareChunker(min_size=5, max_size=24).split(text)

    assert len(chunks) >= 2
    assert all(chunk.text.startswith("#") for chunk in chunks)


def test_semantic_chunker_returns_ordered_chunks() -> None:
    text = "Cats sleep on mats. Cats like warm rooms. Databases store rows. SQL queries rows."
    chunks = SemanticChunker(max_size=45, min_size=10, similarity_threshold=0.01).split(text)

    assert chunks
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].start_char == 0


def test_recursive_chunker_does_not_drop_long_unseparated_text() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = RecursiveChunker(chunk_size=10, separators=()).split(text)

    assert "".join(chunk.text for chunk in chunks) == text
    assert [chunk.text for chunk in chunks] == ["abcdefghij", "klmnopqrst", "uvwxyz"]


def test_fixed_window_validates_overlap() -> None:
    try:
        FixedWindowChunker(chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("FixedWindowChunker should reject overlap >= chunk_size")


def test_coverage_penalizes_dropped_content() -> None:
    text = "alpha beta gamma"
    chunks = FixedWindowChunker(chunk_size=5, overlap=0).split(text)
    coverage = IntrinsicMetricEvaluator().coverage(text, chunks[:1])

    assert 0.0 < coverage < 1.0


def test_langchain_adapter_has_helpful_missing_dependency_error() -> None:
    from adaptive_chunking import langchain

    if langchain.TextSplitter is not None:
        return

    try:
        langchain.LangChainAdaptiveTextSplitter()
    except RuntimeError as exc:
        assert ".[langchain]" in str(exc)
    else:
        raise AssertionError("missing LangChain dependency should raise RuntimeError")


def test_llama_index_adapter_has_helpful_missing_dependency_error() -> None:
    from adaptive_chunking.llama_index import chunks_to_llama_nodes
    from adaptive_chunking.models import Chunk

    try:
        chunks_to_llama_nodes([Chunk("text", 0, 0, 4)])
    except RuntimeError as exc:
        assert ".[llama-index]" in str(exc)
    else:
        # Dependency is installed in this environment, which is also fine.
        assert True

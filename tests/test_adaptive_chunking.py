from adaptive_chunking import AdaptiveChunker
from adaptive_chunking.chunkers import (
    CodeChunker,
    DelimiterChunker,
    FixedWindowChunker,
    HtmlChunker,
    JsonChunker,
    MarkdownChunker,
    PageChunker,
    PageIndexChunker,
    ParagraphChunker,
    RecursiveChunker,
    SectionAwareChunker,
    SemanticChunker,
    SentenceChunker,
    SplitThenMergeChunker,
    TokenWindowChunker,
    default_chunkers,
)
from adaptive_chunking.metrics import IntrinsicMetricEvaluator, MetricConfig, MetricWeights
from adaptive_chunking.models import Chunk, ChunkingConfig, Document
from adaptive_chunking.registry import registry
from adaptive_chunking.retrieval import expand_section_instances
from adaptive_chunking.selector import AdaptiveSelector
from adaptive_chunking.text import normalize_space


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
    assert result.candidates == sorted(
        result.candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )


def test_split_then_merge_preserves_order() -> None:
    text = "A first paragraph.\n\nA second paragraph.\n\nA third paragraph."
    chunks = SplitThenMergeChunker(min_size=10, max_size=40).split(text)

    assert [chunk.text for chunk in chunks] == [
        "A first paragraph.",
        "A second paragraph.\n\nA third paragraph.",
    ]
    assert [text[chunk.start_char : chunk.end_char].strip() for chunk in chunks] == [
        "A first paragraph.",
        "A second paragraph.\n\nA third paragraph.",
    ]
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
    text = "One paragraph.\n\nTwo paragraph."
    chunks = SplitThenMergeChunker(min_size=10, max_size=80).split(text)

    by_name = {metric.name: metric for metric in evaluator.evaluate(text, chunks)}

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
    assert {candidate.strategy_name for candidate in result.candidates} == {
        "delimiter",
        "section-aware",
    }


def test_delimiter_chunker_splits_custom_boundaries() -> None:
    text = "First part.\n---\nSecond part.\n---\nThird part."
    chunks = DelimiterChunker(delimiter="\n---\n").split(text)

    assert [chunk.text for chunk in chunks] == ["First part.", "Second part.", "Third part."]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_delimiter_chunker_can_keep_delimiters() -> None:
    chunks = DelimiterChunker(delimiter="---", keep_delimiter=True).split("alpha---beta")

    assert chunks[0].text == "alpha---"
    assert chunks[1].text == "beta"


def test_page_chunker_uses_form_feed_pages() -> None:
    chunks = PageChunker().split("Page one.\fPage two.")

    assert [chunk.text for chunk in chunks] == ["Page one.", "Page two."]
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.metadata["page_index"] for chunk in chunks] == [0, 1]


def test_page_chunker_preserves_page_index_for_oversized_pages() -> None:
    text = "alpha beta gamma delta\fsecond page"
    chunks = PageChunker(max_size=12).split(text)

    assert [chunk.metadata["page_index"] for chunk in chunks] == [0, 0, 1]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_page_index_chunker_chunks_pages_hierarchically() -> None:
    text = (
        "# Overview\n"
        "Intro.\n\n"
        "## Scope\n"
        "Scope details.\f"
        "# Overview\n"
        "Second page intro.\n\n"
        "## Scope\n"
        "Second scope details."
    )
    chunks = PageIndexChunker(max_size=80).split(text)

    assert [chunk.text for chunk in chunks] == [
        "# Overview\nIntro.",
        "## Scope\nScope details.",
        "# Overview\nSecond page intro.",
        "## Scope\nSecond scope details.",
    ]
    assert [chunk.metadata["page_index"] for chunk in chunks] == [0, 0, 1, 1]
    assert chunks[1].metadata["heading_path"] == ["Overview", "Scope"]
    assert chunks[3].metadata["heading_path"] == ["Overview", "Scope"]
    assert chunks[1].metadata["section_instance_id"] != chunks[3].metadata[
        "section_instance_id"
    ]


def test_page_index_chunker_validates_settings() -> None:
    try:
        PageIndexChunker(page_delimiter="")
    except ValueError as exc:
        assert "page_delimiter" in str(exc)
    else:
        raise AssertionError("PageIndexChunker should reject an empty delimiter")

    try:
        PageIndexChunker(max_size=0)
    except ValueError as exc:
        assert "max_size" in str(exc)
    else:
        raise AssertionError("PageIndexChunker should reject a non-positive max_size")


def test_section_aware_chunker_prefers_heading_boundaries() -> None:
    text = "# Alpha\nAlpha body.\n\n# Beta\nBeta body.\n\n# Gamma\nGamma body."
    chunks = SectionAwareChunker(min_size=5, max_size=24).split(text)

    assert len(chunks) >= 2
    assert all(chunk.text.startswith("#") for chunk in chunks)


def test_section_chunks_have_unique_instance_ids_for_repeated_headings() -> None:
    text = "# Overview\nFirst body.\n\n# Details\nMiddle body.\n\n# Overview\nSecond body."
    chunks = SectionAwareChunker(min_size=5, max_size=30).split(text)

    overview_chunks = [chunk for chunk in chunks if chunk.metadata["section_title"] == "Overview"]

    assert len(overview_chunks) == 2
    assert overview_chunks[0].metadata["section_instance_id"] != overview_chunks[1].metadata[
        "section_instance_id"
    ]


def test_section_chunking_ignores_headers_and_footers_as_section_boundaries() -> None:
    text = (
        "# Header\nCompany handbook\n\n# Access\nAccess policy details.\n\n"
        "# Footer\nInternal use only."
    )
    chunks = SectionAwareChunker(min_size=5, max_size=24).split(text)

    assert all(chunk.metadata["section_title"] not in {"Header", "Footer"} for chunk in chunks)


def test_adaptive_results_attach_section_path_to_every_chunk() -> None:
    text = "# Header\nCompany handbook\n\n# Access\nAccess policy details.\n\n## MFA\nRequired."
    result = AdaptiveChunker(
        selector=AdaptiveSelector(chunkers=[MarkdownChunker(min_size=5, max_size=30)])
    ).chunk(text)

    assert all("section_path" in chunk.metadata for chunk in result.chunks)
    assert all("Header" not in chunk.metadata["section_path"] for chunk in result.chunks)
    assert any(chunk.metadata["section_path"] == ["Access", "MFA"] for chunk in result.chunks)


def test_expand_section_instances_returns_all_chunks_under_fetched_section() -> None:
    text = "# Overview\n" + ("alpha " * 20) + "\n\n# Other\nbeta"
    chunks = SectionAwareChunker(min_size=5, max_size=40).split(text)
    fetched = [chunks[1]]

    expanded = expand_section_instances(chunks, fetched)

    assert len(expanded) > 1
    assert {chunk.metadata["section_instance_id"] for chunk in expanded} == {
        fetched[0].metadata["section_instance_id"]
    }


def test_expand_section_instances_keeps_generator_hits_without_section_metadata() -> None:
    chunks = FixedWindowChunker(chunk_size=5, overlap=0).split("alpha beta")

    expanded = expand_section_instances(chunks, (chunk for chunk in chunks[:1]))

    assert expanded == chunks[:1]


def test_semantic_chunker_returns_ordered_chunks() -> None:
    text = "Cats sleep on mats. Cats like warm rooms. Databases store rows. SQL queries rows."
    chunks = SemanticChunker(max_size=45, min_size=10, similarity_threshold=0.01).split(text)

    assert chunks
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].start_char == 0


def test_semantic_chunker_preserves_non_sentence_fragments() -> None:
    text = "# Alpha\nCats sleep.\n\n# Beta\n- Dogs bark\nTrailing fragment"
    chunks = SemanticChunker(max_size=24, min_size=3, similarity_threshold=0.2).split(text)

    assert chunks
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert "# Alpha" in [chunk.text for chunk in chunks]
    assert "# Beta" in [chunk.text for chunk in chunks]
    assert "- Dogs bark" in [chunk.text for chunk in chunks]
    assert "Trailing fragment" in [chunk.text for chunk in chunks]


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


def test_default_chunkers_return_valid_indexes_and_offsets() -> None:
    samples = [
        "# Alpha\nBody text.\n\n# Beta\nMore body.",
        "one\n---\ntwo\n---\nthree",
        "Page one.\fPage two.",
        "alpha beta gamma delta epsilon",
        "   ",
        "x" * 40,
    ]

    for text in samples:
        for chunker in default_chunkers():
            chunks = chunker.split(text)

            assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
            assert all(chunk.text for chunk in chunks)
            for chunk in chunks:
                assert 0 <= chunk.start_char <= chunk.end_char <= len(text)
                if chunk.metadata.get("source_format") in {"html", "json"}:
                    continue
                if "json_path" in chunk.metadata:
                    continue
                assert normalize_space(text[chunk.start_char : chunk.end_char]) == chunk.text


def test_token_window_chunker_splits_by_words_with_overlap() -> None:
    chunks = TokenWindowChunker(chunk_tokens=3, overlap_tokens=1).split("one two three four five")

    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]


def test_sentence_and_paragraph_chunkers_preserve_boundaries() -> None:
    sentence_chunks = SentenceChunker(min_size=5, max_size=24).split(
        "First sentence. Second sentence. Third sentence."
    )
    paragraph_chunks = ParagraphChunker(min_size=5, max_size=40).split(
        "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    )

    assert all(chunk.text.endswith(".") for chunk in sentence_chunks)
    assert "".join(chunk.text.replace("\n\n", "") for chunk in paragraph_chunks)
    assert all(chunk.text.endswith(".") for chunk in paragraph_chunks)


def test_markdown_chunker_preserves_fenced_code_block() -> None:
    text = "# Demo\nIntro.\n\n```python\nprint('hello')\n```\n\n## Next\nBody."
    chunks = MarkdownChunker(min_size=5, max_size=80).split(text)

    assert any("```python\nprint('hello')\n```" in chunk.text for chunk in chunks)
    assert all("section_title" in chunk.metadata for chunk in chunks)


def test_html_json_and_code_chunkers_return_structured_metadata() -> None:
    html_chunks = HtmlChunker(min_size=5, max_size=80).split("<h1>Title</h1><p>Body text.</p>")
    json_chunks = JsonChunker(max_size=80).split('{"items": [{"id": 1}, {"id": 2}]}')
    code_chunks = CodeChunker(max_size=80).split("def alpha():\n    pass\n\ndef beta():\n    pass")

    assert html_chunks[0].metadata["source_format"] == "html"
    assert [chunk.metadata["json_path"] for chunk in json_chunks] == ["$.items"]
    assert [chunk.text.splitlines()[0] for chunk in code_chunks] == ["def alpha():", "def beta():"]


def test_registry_and_config_select_named_strategies() -> None:
    assert "markdown" in registry.names()

    result = AdaptiveChunker(
        config=ChunkingConfig(strategies=["paragraph", "sentence"], max_chunks=1)
    ).chunk("Alpha sentence. Beta sentence.\n\nSecond paragraph.")

    assert {candidate.strategy_name for candidate in result.candidates} == {
        "paragraph",
        "sentence",
    }
    assert all(len(candidate.chunks) <= 1 for candidate in result.candidates)


def test_chunking_config_rejects_empty_strategy_list() -> None:
    try:
        ChunkingConfig(strategies=[])
    except ValueError as exc:
        assert "strategies" in str(exc)
    else:
        raise AssertionError("empty strategies should be rejected")


def test_result_serialization_and_document_api() -> None:
    result = AdaptiveChunker(config=ChunkingConfig(strategies=["single"])).chunk_document(
        Document(text="hello world", document_id="doc-1")
    )
    payload = result.to_dict(include_candidates=True)

    assert payload["document_id"] == "doc-1"
    assert payload["candidates"][0]["strategy_name"] == "single"
    assert Chunk.from_dict(payload["chunks"][0]) == result.chunks[0]


def test_langchain_adapter_has_helpful_missing_dependency_error() -> None:
    from adaptive_chunking import langchain

    if langchain.TextSplitter is not None:
        return

    try:
        langchain.LangChainAdaptiveTextSplitter()
    except RuntimeError as exc:
        assert "adaptive-oci-chunking[langchain]" in str(exc)
    else:
        raise AssertionError("missing LangChain dependency should raise RuntimeError")


def test_llama_index_adapter_has_helpful_missing_dependency_error() -> None:
    from adaptive_chunking.llama_index import chunks_to_llama_nodes
    from adaptive_chunking.models import Chunk

    try:
        chunks_to_llama_nodes([Chunk("text", 0, 0, 4)])
    except RuntimeError as exc:
        assert "adaptive-oci-chunking[llama-index]" in str(exc)
    else:
        # Dependency is installed in this environment, which is also fine.
        assert True


def test_langchain_adapter_preserves_source_and_adaptive_metadata() -> None:
    from adaptive_chunking.langchain import LangChainAdaptiveTextSplitter, TextSplitter

    if TextSplitter is None:
        return

    from langchain_core.documents import Document as LangChainDocument

    splitter = LangChainAdaptiveTextSplitter(
        chunker=AdaptiveChunker(config=ChunkingConfig(strategies=["markdown"]))
    )
    documents = splitter.split_documents(
        [LangChainDocument(page_content="# Title\nBody", metadata={"source": "guide.md"})]
    )

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "guide.md"
    assert documents[0].metadata["strategy_name"] == "markdown"
    assert documents[0].metadata["start_char"] == 0
    assert documents[0].metadata["end_char"] == len("# Title\nBody")


def test_langchain_adapter_rejects_mismatched_metadata() -> None:
    from adaptive_chunking.langchain import LangChainAdaptiveTextSplitter, TextSplitter

    if TextSplitter is None:
        return

    splitter = LangChainAdaptiveTextSplitter()
    try:
        splitter.create_documents(["first", "second"], metadatas=[])
    except ValueError as exc:
        assert "metadatas" in str(exc)
    else:
        raise AssertionError("mismatched metadata should raise ValueError")


def test_langchain_adapter_can_chunk_a_file(tmp_path) -> None:
    from adaptive_chunking.langchain import LangChainAdaptiveTextSplitter, TextSplitter

    if TextSplitter is None:
        return

    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nUseful details.", encoding="utf-8")
    documents = LangChainAdaptiveTextSplitter().split_file(source)

    assert documents
    assert documents[0].metadata["source_format"] == "md"
    assert documents[0].metadata["document_id"] == "guide"


def test_llama_index_parser_is_a_native_node_parser_with_relationships() -> None:
    from adaptive_chunking.llama_index import LlamaIndexAdaptiveParser, NodeParser

    if NodeParser is None:
        return

    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.schema import NodeRelationship

    parser = LlamaIndexAdaptiveParser(
        chunker=AdaptiveChunker(config=ChunkingConfig(strategies=["markdown"]))
    )
    source = LlamaDocument(text="# Title\nBody", metadata={"source": "guide.md"})
    nodes = parser.get_nodes_from_documents([source])

    assert isinstance(parser, NodeParser)
    assert len(nodes) == 1
    assert nodes[0].metadata["source"] == "guide.md"
    assert nodes[0].metadata["strategy_name"] == "markdown"
    assert nodes[0].source_node is not None
    assert nodes[0].relationships[NodeRelationship.SOURCE].node_id == source.id_


def test_llama_index_parser_links_neighbouring_nodes() -> None:
    from adaptive_chunking.llama_index import LlamaIndexAdaptiveParser, NodeParser

    if NodeParser is None:
        return

    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.core.schema import NodeRelationship

    selector = AdaptiveSelector(chunkers=[FixedWindowChunker(chunk_size=5, overlap=0)])
    parser = LlamaIndexAdaptiveParser(chunker=AdaptiveChunker(selector=selector))
    nodes = parser.get_nodes_from_documents([LlamaDocument(text="abcdefghij")])

    assert len(nodes) == 2
    assert nodes[0].relationships[NodeRelationship.NEXT].node_id == nodes[1].node_id
    assert nodes[1].relationships[NodeRelationship.PREVIOUS].node_id == nodes[0].node_id

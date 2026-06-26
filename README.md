<div align="center">

# Adaptive OCI Chunking

**Adaptive chunking toolkit for RAG with OCI, LangChain, and LlamaIndex support**

[![CI](https://github.com/CaptnSalazar/adaptive-oci-chunking/actions/workflows/ci.yml/badge.svg)](https://github.com/CaptnSalazar/adaptive-oci-chunking/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.25333-b31b1b.svg)](https://arxiv.org/abs/2603.25333)

</div>

Adaptive OCI Chunking is an extensible Python implementation for document-aware chunk selection in Retrieval-Augmented Generation (RAG). It is inspired by Ekimetrics' `adaptive-chunking` repository and the paper _Adaptive Chunking: Optimizing Chunking-Method Selection for RAG_.

The package evaluates several chunking strategies for each document, scores them with intrinsic metrics, and selects the best candidate before indexing or generation. Oracle Cloud Infrastructure (OCI) integrations are optional: the core chunking engine runs locally, while OCI Object Storage and Generative AI can be enabled when needed.

## Architecture

![Adaptive OCI Chunking architecture](Architecture.png)

## What is Adaptive Chunking?

No single chunking method works best for every document in a RAG pipeline. Adaptive chunking treats chunking as a selection problem: try multiple splitting strategies, score each result with intrinsic quality metrics, and choose the best candidate for the document at hand.

This repo builds on that idea as a practical toolkit. It keeps the core dependency-light, adds extra production-oriented metrics, and includes optional adapters for OCI, LangChain, and LlamaIndex.

## Features

- Candidate chunkers:
  - single-document
  - fixed window with overlap
  - recursive split
  - split-then-merge
  - section-aware
  - delimiter-aware
  - page-aware
  - semantic lexical drift
  - regex-guided section splitting
- Metric-guided selection using paper-aligned intrinsic metrics:
  - References Completeness (RC)
  - Intrachunk Cohesion (ICC)
  - Document Contextual Coherence (DCC)
  - Block Integrity (BI)
  - Size Compliance (SC)
- Additional practical metrics:
  - source coverage
  - overlap control
  - boundary quality
  - semantic drift
  - information density
  - redundancy
- Weighted strategy selection with explainable per-metric scores.
- LangChain `TextSplitter` adapter.
- LlamaIndex node conversion and parser-style adapter.
- CLI for local text/Markdown files.
- Optional OCI Object Storage loader and OCI Generative AI embedding adapter.
- Small, dependency-light core for local document chunking workflows.

## Contributing

Contributions are welcome for new chunkers, metrics, examples, integrations, benchmarks, documentation, and bug fixes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, PR expectations, and guidance for adding chunkers or metrics.

Maintained by [Yash Shukla](https://www.linkedin.com/in/yashtechi/), focused on AI, cloud, and RAG systems.

## Install

```bash
pip install -e ".[dev]"
```

With OCI support:

```bash
pip install -e ".[oci]"
```

With the API server:

```bash
pip install -e ".[api]"
```

With framework integrations:

```bash
pip install -e ".[langchain,llama-index]"
```

## Quick Start

```bash
adaptive-chunk chunk examples/sample.md --json
```

Python usage:

```python
from adaptive_chunking import AdaptiveChunker

text = "## Introduction\nAdaptive chunking chooses a splitter per document.\n\n## Details\n..."
chunker = AdaptiveChunker()
result = chunker.chunk(text, document_id="demo")

print(result.strategy_name)
for chunk in result.chunks:
    print(chunk.text)
```

## Examples

Runnable examples live in `examples/`:

- `basic_adaptive_chunking.py`: end-to-end adaptive selection with metric output.
- `custom_selector.py`: custom chunker list and metric weights.
- `langchain_integration.py`: LangChain `TextSplitter` usage.
- `llama_index_integration.py`: LlamaIndex `TextNode` conversion.
- `oci_object_storage.py`: loading source text from OCI Object Storage.

## Chunker Options

```python
from adaptive_chunking.chunkers import (
    DelimiterChunker,
    PageChunker,
    SectionAwareChunker,
    SemanticChunker,
)
from adaptive_chunking.selector import AdaptiveSelector
from adaptive_chunking import AdaptiveChunker

selector = AdaptiveSelector(
    chunkers=[
        SectionAwareChunker(max_size=1800),
        DelimiterChunker(delimiter="\n---\n"),
        PageChunker(page_delimiter="\f"),
        SemanticChunker(max_size=1400, similarity_threshold=0.08),
    ]
)

result = AdaptiveChunker(selector=selector).chunk(text)
```

## Metrics

The selector ranks every candidate by a weighted average of intrinsic scores. The first five metrics follow the paper's evaluation dimensions; the additional metrics make the implementation more practical for production RAG systems where dropped text, excessive overlap, and duplicated chunks are common failure modes.

Weights can be tuned:

```python
from adaptive_chunking.metrics import IntrinsicMetricEvaluator, MetricConfig, MetricWeights
from adaptive_chunking.selector import AdaptiveSelector

weights = MetricWeights(
    block_integrity=1.4,
    coverage=1.5,
    redundancy=0.8,
)
evaluator = IntrinsicMetricEvaluator(MetricConfig(weights=weights))
selector = AdaptiveSelector(evaluator=evaluator)
```

## Adaptive Scoring

For each document, the selector runs every candidate chunker and evaluates the chunks it produces. Each candidate receives a normalized weighted score:

```text
score(candidate) = sum(metric_value_i * metric_weight_i) / sum(metric_weight_i)
```

Where:

- `metric_value_i` is the metric score for a candidate, normalized from `0.0` to `1.0`.
- `metric_weight_i` controls how important that metric is for selection.
- Higher scores are better.
- Candidates are ranked from highest score to lowest score.

For example, a domain that cares about preserving source text and section boundaries might emphasize `coverage` and `block_integrity`:

| Metric | Value | Weight | Weighted value |
|--------|------:|-------:|---------------:|
| coverage | 1.00 | 1.50 | 1.50 |
| block_integrity | 0.90 | 1.40 | 1.26 |
| redundancy | 0.80 | 0.80 | 0.64 |

```text
score = (1.50 + 1.26 + 0.64) / (1.50 + 1.40 + 0.80)
      = 3.40 / 3.70
      = 0.919
```

You can inspect every candidate, not just the winner:

```python
from adaptive_chunking import AdaptiveChunker

result = AdaptiveChunker().chunk(text, document_id="demo")

for candidate in result.candidates:
    print(candidate.strategy_name, round(candidate.score, 3), len(candidate.chunks))
    for metric in candidate.metrics:
        print(" ", metric.name, metric.value, "weight=", metric.weight)
```

This makes the selection process explainable: if a chunker loses, you can see whether it dropped content, produced excessive overlap, cut through structure, or failed a size constraint.

## LangChain

```python
from adaptive_chunking.langchain import LangChainAdaptiveTextSplitter

splitter = LangChainAdaptiveTextSplitter()
documents = splitter.create_documents([text])
```

## LlamaIndex

```python
from adaptive_chunking import AdaptiveChunker
from adaptive_chunking.llama_index import result_to_llama_nodes

result = AdaptiveChunker().chunk(text, document_id="policy")
nodes = result_to_llama_nodes(result)
```

## OCI Usage

Copy `.env.example` and set the values for your tenancy and compartment. The core library does not require OCI credentials unless you instantiate an OCI adapter.

```python
from adaptive_chunking.oci import OCIObjectStorageTextLoader

loader = OCIObjectStorageTextLoader(
    namespace="my-namespace",
    bucket_name="documents",
)
text = loader.load_text("policies/example.md")
```

## API Server

```bash
uvicorn adaptive_chunking.api:app --reload
```

Then post:

```bash
curl -X POST http://127.0.0.1:8000/chunk \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"# Title\nBody text\", \"document_id\":\"demo\"}"
```

## Project Layout

```text
src/adaptive_chunking/
  chunkers.py      # candidate splitting strategies
  metrics.py       # intrinsic metric implementations
  selector.py      # weighted adaptive strategy selection
  pipeline.py      # high-level AdaptiveChunker
  langchain.py     # optional LangChain TextSplitter adapter
  llama_index.py   # optional LlamaIndex node helpers
  oci.py           # optional OCI adapters
  api.py           # optional FastAPI app
  cli.py           # command line interface
tests/
examples/
```

## Notes

This repo is designed as a clean, extensible foundation rather than a verbatim copy of the reference implementation. The metric implementations are practical approximations intended for engineering use and experimentation. Production RAG deployments should calibrate weights, chunk sizes, and embedding models against their document domains.

## References

- Ekimetrics reference implementation: [ekimetrics/adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking)
- Paper: [Adaptive Chunking: Optimizing Chunking-Method Selection for RAG](https://arxiv.org/abs/2603.25333)

## Citation

If this project helps your work, please cite the original adaptive chunking paper:

```bibtex
@inproceedings{demoura2026adaptive,
    title={Adaptive Chunking: Optimizing Chunking-Method Selection for RAG},
    author={de Moura Junior, Paulo Roberto and Lelong, Jean and Blangero, Annabelle},
    booktitle={Proceedings of the 15th Language Resources and Evaluation Conference (LREC 2026)},
    year={2026},
    url={https://arxiv.org/abs/2603.25333},
}
```

## License

This project is licensed under the [MIT License](LICENSE).

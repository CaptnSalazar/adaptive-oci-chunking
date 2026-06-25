# Adaptive OCI Chunking

Adaptive OCI Chunking is an extensible Python implementation for document-aware chunk selection in Retrieval-Augmented Generation (RAG). It is inspired by Ekimetrics' `adaptive-chunking` repository and the paper _Adaptive Chunking: Optimizing Chunking-Method Selection for RAG_.

The package evaluates several chunking strategies for each document, scores them with intrinsic metrics, and selects the best candidate before indexing or generation. Oracle Cloud Infrastructure (OCI) integrations are optional: the core chunking engine runs locally, while OCI Object Storage and Generative AI can be enabled when needed.

## Architecture

![Adaptive OCI Chunking architecture](Archtecture.png)

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
- Small, dependency-light core suitable for a public GitHub repo.

## Contributing

This repo is intended to be public and community-maintained. Contributions are welcome for new chunkers, metrics, examples, integrations, benchmarks, documentation, and bug fixes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, PR expectations, and guidance for adding chunkers or metrics.

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

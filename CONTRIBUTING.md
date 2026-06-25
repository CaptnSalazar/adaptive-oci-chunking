# Contributing

Thanks for helping improve Adaptive OCI Chunking. This project is intended to be useful for practitioners building RAG systems, researchers testing chunking strategies, and maintainers who want a clean place to compare document-splitting ideas.

Maintainer: [Yash Shukla](https://www.linkedin.com/in/yashtechi/), focused on AI, cloud, and RAG systems.

## Ways to Contribute

- Add new chunkers for specific document structures or domains.
- Improve intrinsic metrics or add new evaluation dimensions.
- Add examples for LangChain, LlamaIndex, OCI, or other RAG workflows.
- Improve tests, documentation, type hints, and packaging.
- Report bugs with small reproducible examples.
- Share benchmark results from real document collections.

## Development Setup

Clone the repo and install it in editable mode:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you only want optional integration support:

```bash
pip install -e ".[langchain,llama-index,oci,api]"
```

## Running Checks

```bash
ruff check .
pytest
python -m compileall src tests examples
```

If you add an example, make sure it either runs without optional credentials or clearly documents the required environment variables.

## Adding a Chunker

Chunkers live in `src/adaptive_chunking/chunkers.py` and implement `BaseChunker`.

A good chunker should:

- Preserve source order.
- Return non-empty `Chunk` objects with stable `start_char` and `end_char` spans.
- Avoid silently dropping text.
- Fall back gracefully when its preferred structure is not present.
- Include focused tests in `tests/test_adaptive_chunking.py`.
- Be added to `default_chunkers()` only if it is broadly useful.

## Adding a Metric

Metrics live in `src/adaptive_chunking/metrics.py`.

A good metric should:

- Return a bounded score from `0.0` to `1.0`.
- Be explainable from document and chunk structure alone.
- Have a default weight in `MetricWeights`.
- Include an explanation string in `IntrinsicMetricEvaluator.evaluate`.
- Include tests for normal and edge cases.

## Pull Request Checklist

Before opening a PR:

- Run `ruff check .`.
- Run `pytest`.
- Run `python -m compileall src tests examples`.
- Update README or examples when behavior changes.
- Add or update tests for code changes.
- Keep changes focused on one concern where possible.

## Design Principles

- The core package should stay dependency-light.
- Optional integrations should import their heavy dependencies only when used.
- Chunking behavior should be inspectable and explainable.
- Metrics should help users understand tradeoffs, not hide them behind a black box.
- OCI support should remain optional.

## Reporting Issues

Please include:

- Python version.
- Installation command.
- Minimal input text or document shape.
- Expected chunking behavior.
- Actual chunking behavior.
- Any traceback or metric output.

For private or sensitive documents, replace content with a synthetic example that preserves the relevant structure.

# Benchmarks

Use these scripts to answer the practical adoption question:

> Does adaptive chunking produce better chunks for my documents than the splitter I already use?

The first benchmark is intentionally dependency-light. It compares adaptive selection with named baseline strategies using the library's intrinsic metrics: coverage, boundary quality, size compliance, cohesion, redundancy, and related diagnostics.

For a fair RAG-ingestion comparison, the adaptive row selects among the same baseline strategies listed in `--strategies`. That keeps the benchmark focused on “which splitter should index this document?” instead of letting a no-split baseline win on small files.

```bash
python benchmarks/compare_chunkers.py examples/benchmark_policy.md
python benchmarks/compare_chunkers.py examples --format markdown
```

Suggested next layer for production teams:

1. create a small golden set of real questions and expected supporting passages;
2. index chunks from each strategy into the same vector store;
3. measure recall@k, answer correctness, latency, and token cost;
4. tune `MetricWeights`, chunk sizes, and allowed strategies for the domain.

This keeps the benchmark honest: chunking quality is domain-sensitive, so the most persuasive result is usually measured on the documents your RAG system actually serves.

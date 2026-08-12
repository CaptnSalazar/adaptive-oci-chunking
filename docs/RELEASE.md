# Release checklist

Use this before publishing a new package version.

## Local checks

```bash
ruff check .
pytest
python -m compileall src tests examples benchmarks demos
python benchmarks/compare_chunkers.py examples/benchmark_policy.md --format markdown
python examples/vector_store_ingestion.py examples/sample.md
python -m build
python -m twine check dist/*
```

If `python -m build` or `python -m twine` is not available, install release tooling first:

```bash
python -m pip install build twine
```

## Version and metadata

- Confirm `pyproject.toml` has the intended release version.
- Confirm `src/adaptive_chunking/__init__.py` fallback version matches `pyproject.toml`.
- Confirm `CHANGELOG.md` has release notes for the version.
- Confirm README examples mention any new public files or extras.

## Publish

```bash
python -m twine upload dist/*
```

After publish:

- verify the PyPI page shows the new version;
- create a GitHub release/tag;
- include the benchmark output and demo commands in the release notes.

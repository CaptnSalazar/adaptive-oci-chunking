from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_chunking import AdaptiveChunker, ChunkingConfig, load_document  # noqa: E402

DEFAULT_BASELINES = [
    "fixed-window",
    "recursive",
    "markdown",
    "section-aware",
    "token-window",
]


def discover_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    supported = {".txt", ".md", ".markdown", ".rst", ".pdf"}
    return sorted(child for child in path.rglob("*") if child.suffix.lower() in supported)


def summarize_candidate(
    path: Path,
    strategy: str | None,
    *,
    adaptive_strategies: list[str] | None = None,
) -> dict[str, object]:
    document = load_document(path)
    config = ChunkingConfig(strategies=[strategy] if strategy else adaptive_strategies)
    result = AdaptiveChunker(config=config).chunk_document(document)
    metrics = {metric.name: metric.value for metric in result.metrics}
    return {
        "document": str(path),
        "mode": strategy or "adaptive",
        "selected_strategy": result.strategy_name,
        "score": result.score,
        "chunks": len(result.chunks),
        "avg_chars": sum(chunk.size for chunk in result.chunks) / max(len(result.chunks), 1),
        "coverage": metrics.get("coverage", 0.0),
        "boundary_quality": metrics.get("boundary_quality", 0.0),
        "size_compliance": metrics.get("size_compliance", 0.0),
        "redundancy": metrics.get("redundancy", 0.0),
    }


def print_markdown(rows: list[dict[str, object]]) -> None:
    columns = [
        "document",
        "mode",
        "selected_strategy",
        "score",
        "chunks",
        "avg_chars",
        "coverage",
        "boundary_quality",
        "size_compliance",
        "redundancy",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def print_csv(rows: list[dict[str, object]]) -> None:
    columns = [
        "document",
        "mode",
        "selected_strategy",
        "score",
        "chunks",
        "avg_chars",
        "coverage",
        "boundary_quality",
        "size_compliance",
        "redundancy",
    ]
    print(",".join(columns))
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value).replace(",", " "))
        print(",".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare adaptive chunking against named baseline strategies."
    )
    parser.add_argument("path", type=Path, help="A supported document file or folder.")
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=DEFAULT_BASELINES,
        help="Baseline strategies to compare against adaptive selection.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "csv"],
        default="markdown",
        help="Output format.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for input_path in discover_inputs(args.path):
        rows.append(summarize_candidate(input_path, None, adaptive_strategies=args.strategies))
        for strategy in args.strategies:
            try:
                rows.append(summarize_candidate(input_path, strategy))
            except ValueError as exc:
                rows.append(
                    {
                        "document": str(input_path),
                        "mode": strategy,
                        "selected_strategy": "error",
                        "score": 0.0,
                        "chunks": 0,
                        "avg_chars": 0.0,
                        "coverage": 0.0,
                        "boundary_quality": 0.0,
                        "size_compliance": 0.0,
                        "redundancy": 0.0,
                        "error": str(exc),
                    }
                )

    if args.format == "csv":
        print_csv(rows)
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from adaptive_chunking.io import load_text_file
from adaptive_chunking.models import ChunkingConfig
from adaptive_chunking.pipeline import AdaptiveChunker
from adaptive_chunking.registry import registry

app = typer.Typer(help="Adaptive document chunking for RAG.")
console = Console()


@app.callback()
def main() -> None:
    """Adaptive document chunking for RAG."""


@app.command()
def chunk(
    path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Text or Markdown file to chunk."),
    ],
    document_id: Annotated[
        str | None,
        typer.Option(help="Stable document identifier."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
    strategy: Annotated[
        list[str] | None,
        typer.Option("--strategy", help="Chunking strategy to evaluate. Repeatable."),
    ] = None,
    include_candidates: Annotated[
        bool,
        typer.Option("--include-candidates", help="Include every candidate in JSON output."),
    ] = False,
) -> None:
    text = load_text_file(path)
    config = ChunkingConfig(strategies=strategy)
    result = AdaptiveChunker(config=config).chunk(text, document_id=document_id or path.stem)
    if json_output:
        console.print(result.to_json(include_candidates=include_candidates, indent=2))
        return

    console.print(f"[bold]Strategy:[/bold] {result.strategy_name}")
    console.print(f"[bold]Score:[/bold] {result.score:.3f}")
    table = Table("Metric", "Value", "Weight")
    for metric in result.metrics:
        table.add_row(metric.name, f"{metric.value:.3f}", f"{metric.weight:.2f}")
    console.print(table)
    for chunk_item in result.chunks:
        console.rule(f"Chunk {chunk_item.index}")
        console.print(chunk_item.text)


@app.command()
def strategies() -> None:
    """List available chunking strategies."""
    for name in registry.names():
        console.print(name)


if __name__ == "__main__":
    app()

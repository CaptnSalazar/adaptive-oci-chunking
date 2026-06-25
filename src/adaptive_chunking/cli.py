from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from adaptive_chunking.io import load_text_file
from adaptive_chunking.pipeline import AdaptiveChunker

app = typer.Typer(help="Adaptive document chunking for RAG.")
console = Console()


@app.command()
def chunk(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Text or Markdown file to chunk."),
    document_id: str | None = typer.Option(None, help="Stable document identifier."),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
) -> None:
    text = load_text_file(path)
    result = AdaptiveChunker().chunk(text, document_id=document_id or path.stem)
    if json_output:
        console.print(
            json.dumps(
                {
                    "document_id": result.document_id,
                    "strategy_name": result.strategy_name,
                    "score": result.score,
                    "chunks": [chunk.__dict__ for chunk in result.chunks],
                    "metrics": [metric.__dict__ for metric in result.metrics],
                },
                indent=2,
            )
        )
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


if __name__ == "__main__":
    app()


from pathlib import Path
from types import ModuleType

from typer.testing import CliRunner

from adaptive_chunking import AdaptiveChunker, __version__, load_document
from adaptive_chunking.cli import app
from adaptive_chunking.io import load_pdf_file

runner = CliRunner()


def test_installed_package_generates_chunks() -> None:
    text = (
        "# Demo\n"
        "Adaptive chunking chooses a chunking strategy for each document.\n\n"
        "## Usage\n"
        "The selected strategy returns ordered chunks with source offsets and scores.\n"
    )

    result = AdaptiveChunker().chunk(text, document_id="pip-install-demo")

    assert result.document_id == "pip-install-demo"
    assert result.strategy_name
    assert result.chunks
    assert all(chunk.text.strip() for chunk in result.chunks)
    assert [chunk.index for chunk in result.chunks] == list(range(len(result.chunks)))
    assert 0.0 <= result.score <= 1.0


def test_package_exposes_version() -> None:
    assert __version__ != "0.0.0"


def test_cli_chunk_subcommand_generates_json(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text(
        "# Demo\nAdaptive chunking creates chunks for retrieval.\n\n## Details\n"
        "The CLI should emit JSON for automation.\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["chunk", str(sample), "--json"])

    assert result.exit_code == 0
    assert '"document_id": "sample"' in result.stdout
    assert '"chunks":' in result.stdout
    assert '"strategy_name":' in result.stdout


def test_cli_lists_strategies() -> None:
    result = runner.invoke(app, ["strategies"])

    assert result.exit_code == 0
    assert "markdown" in result.stdout
    assert "token-window" in result.stdout


def test_cli_can_limit_strategy(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    result = runner.invoke(app, ["chunk", str(sample), "--strategy", "single", "--json"])

    assert result.exit_code == 0
    assert '"strategy_name": "single"' in result.stdout


def test_file_api_loads_and_chunks_text_files(tmp_path: Path) -> None:
    sample = tmp_path / "guide.md"
    sample.write_text("# Guide\n\nChunk this document.", encoding="utf-8")

    document = load_document(sample)
    result = AdaptiveChunker().chunk_file(str(sample))

    assert document.document_id == "guide"
    assert document.metadata["source_format"] == "md"
    assert result.document_id == "guide"
    assert result.chunks


def test_pdf_loader_preserves_page_boundaries(monkeypatch, tmp_path: Path) -> None:
    class Page:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, _: str) -> None:
            self.pages = [Page("First page"), Page("Second page")]

    fake_pypdf = ModuleType("pypdf")
    fake_pypdf.PdfReader = Reader  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake_pypdf)
    document = load_pdf_file(tmp_path / "handbook.pdf")

    assert document.text == "First page\fSecond page"
    assert document.document_id == "handbook"
    assert document.metadata["page_count"] == 2


def test_api_accepts_strategy_config_when_fastapi_installed() -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return

    from adaptive_chunking.api import app

    client = TestClient(app)
    response = client.post(
        "/chunk",
        json={"text": "# Demo\nBody text.", "strategies": ["single"]},
    )

    assert response.status_code == 200
    assert response.json()["strategy_name"] == "single"


def test_api_rejects_empty_strategy_list_when_fastapi_installed() -> None:
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return

    from adaptive_chunking.api import app

    client = TestClient(app)
    response = client.post("/chunk", json={"text": "Body text.", "strategies": []})

    assert response.status_code == 400

from __future__ import annotations

from pathlib import Path

from adaptive_chunking.models import Document

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
SUPPORTED_DOCUMENT_SUFFIXES = SUPPORTED_TEXT_SUFFIXES | {".pdf"}


def load_text_file(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
        raise ValueError(f"unsupported file type: {file_path.suffix}")
    return file_path.read_text(encoding="utf-8")


def discover_text_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(
        child
        for child in root.rglob("*")
        if child.is_file() and child.suffix.lower() in SUPPORTED_TEXT_SUFFIXES
    )


def load_pdf_file(path: str | Path) -> Document:
    """Extract PDF text, retaining pages as form-feed characters for page-aware chunking."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install PDF support with `pip install 'adaptive-oci-chunking[pdf]'`."
        ) from exc

    file_path = Path(path)
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\f".join(pages)
    if not text.strip():
        raise ValueError(
            "no extractable text found in PDF; run OCR first for scanned or image-only PDFs"
        )
    return Document(
        text=text,
        document_id=file_path.stem,
        metadata={
            "source": str(file_path),
            "source_format": "pdf",
            "page_count": len(pages),
        },
    )


def load_document(path: str | Path) -> Document:
    """Load a supported text file or PDF into the library's ``Document`` model."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(file_path)
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return Document(
            text=load_text_file(file_path),
            document_id=file_path.stem,
            metadata={"source": str(file_path), "source_format": suffix.lstrip(".")},
        )
    raise ValueError(f"unsupported file type: {file_path.suffix}")

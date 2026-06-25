from __future__ import annotations

from pathlib import Path


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


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


from importlib.metadata import PackageNotFoundError, version

from adaptive_chunking.models import Chunk, ChunkingResult, Document
from adaptive_chunking.pipeline import AdaptiveChunker

try:
    __version__ = version("adaptive-oci-chunking")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.2"

__all__ = ["__version__", "AdaptiveChunker", "Chunk", "ChunkingResult", "Document"]

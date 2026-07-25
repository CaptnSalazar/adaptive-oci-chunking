from importlib.metadata import PackageNotFoundError, version

from adaptive_chunking.models import Chunk, ChunkingConfig, ChunkingResult, Document
from adaptive_chunking.pipeline import AdaptiveChunker
from adaptive_chunking.registry import registry
from adaptive_chunking.retrieval import expand_section_instances

try:
    __version__ = version("adaptive-oci-chunking")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.2.1"

__all__ = [
    "__version__",
    "AdaptiveChunker",
    "Chunk",
    "ChunkingConfig",
    "ChunkingResult",
    "Document",
    "expand_section_instances",
    "registry",
]

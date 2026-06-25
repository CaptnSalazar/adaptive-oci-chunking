import os

from adaptive_chunking import AdaptiveChunker
from adaptive_chunking.oci import OCIObjectStorageTextLoader


def main() -> None:
    loader = OCIObjectStorageTextLoader(
        namespace=os.environ["OCI_OBJECT_STORAGE_NAMESPACE"],
        bucket_name=os.environ["OCI_OBJECT_STORAGE_BUCKET"],
    )
    text = loader.load_text("documents/policy.md")
    result = AdaptiveChunker().chunk(text, document_id="oci-policy")

    print(f"Selected strategy: {result.strategy_name}")
    print(f"Chunk count: {len(result.chunks)}")


if __name__ == "__main__":
    main()


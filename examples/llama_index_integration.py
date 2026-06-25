from adaptive_chunking import AdaptiveChunker
from adaptive_chunking.llama_index import result_to_llama_nodes


TEXT = """
# Knowledge Base Article

Adaptive chunking can produce LlamaIndex TextNode objects for indexing.
"""


def main() -> None:
    result = AdaptiveChunker().chunk(TEXT, document_id="kb-article")
    nodes = result_to_llama_nodes(result)

    for node in nodes:
        print(node.id_)
        print(node.text)
        print(node.metadata)


if __name__ == "__main__":
    main()


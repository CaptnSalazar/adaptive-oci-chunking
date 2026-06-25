from adaptive_chunking.langchain import LangChainAdaptiveTextSplitter

TEXT = """
# Runbook

Restart the worker after changing queue settings.

## Alerts

Investigate sustained retry spikes before scaling the service.
"""


def main() -> None:
    splitter = LangChainAdaptiveTextSplitter()
    documents = splitter.create_documents([TEXT], metadatas=[{"source": "runbook"}])

    for document in documents:
        print(document.page_content)
        print(document.metadata)


if __name__ == "__main__":
    main()


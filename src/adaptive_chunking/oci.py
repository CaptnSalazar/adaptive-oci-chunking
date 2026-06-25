from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def _load_oci() -> Any:
    try:
        import oci  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Install OCI support with `pip install -e .[oci]`.") from exc
    return oci


@dataclass(frozen=True)
class OCISettings:
    config_file: str = os.getenv("OCI_CONFIG_FILE", "~/.oci/config")
    profile: str = os.getenv("OCI_PROFILE", "DEFAULT")
    compartment_id: str | None = os.getenv("OCI_COMPARTMENT_ID")
    genai_endpoint: str | None = os.getenv("OCI_GENAI_ENDPOINT")


class OCIObjectStorageTextLoader:
    def __init__(
        self,
        namespace: str,
        bucket_name: str,
        settings: OCISettings | None = None,
    ) -> None:
        oci = _load_oci()
        self.namespace = namespace
        self.bucket_name = bucket_name
        self.settings = settings or OCISettings()
        config = oci.config.from_file(self.settings.config_file, self.settings.profile)
        self.client = oci.object_storage.ObjectStorageClient(config)

    def load_text(self, object_name: str, encoding: str = "utf-8") -> str:
        response = self.client.get_object(self.namespace, self.bucket_name, object_name)
        return response.data.content.decode(encoding)


class OCIGenAIEmbeddingClient:
    def __init__(
        self,
        model_id: str | None = None,
        settings: OCISettings | None = None,
    ) -> None:
        oci = _load_oci()
        self.settings = settings or OCISettings()
        self.model_id = model_id or os.getenv(
            "OCI_GENAI_EMBEDDING_MODEL",
            "cohere.embed-english-v3.0",
        )
        if not self.settings.compartment_id:
            raise ValueError("OCI_COMPARTMENT_ID is required for OCI Generative AI embeddings")
        config = oci.config.from_file(self.settings.config_file, self.settings.profile)
        kwargs = (
            {"service_endpoint": self.settings.genai_endpoint}
            if self.settings.genai_endpoint
            else {}
        )
        self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(config, **kwargs)
        self.models = oci.generative_ai_inference.models

    def embed(self, texts: list[str]) -> list[list[float]]:
        details = self.models.EmbedTextDetails(
            inputs=texts,
            serving_mode=self.models.OnDemandServingMode(model_id=self.model_id),
            compartment_id=self.settings.compartment_id,
        )
        response = self.client.embed_text(details)
        return [list(vector) for vector in response.data.embeddings]

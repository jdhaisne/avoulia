from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


def _normalize_azure_endpoint(url: str) -> str:
    """Retourne l'URL de base Azure (sans chemin /openai/... ni query string)."""
    if not url:
        return url
    url = url.split("?")[0].rstrip("/")
    for prefix in (
        "/openai/deployments/",
        "/openai/responses",
        "/openai/embeddings",
        "/embeddings",
        "/chat/completions",
    ):
        if prefix in url:
            url = url.split(prefix)[0]
    return url.rstrip("/")


class Settings(BaseSettings):
    """
    Configuration de l'application.
    Toutes les valeurs sont surchargées par backend/.env (voir noms ENV ci-dessous).
    """

    # API
    api_title: str = "RAG Chatbot API"
    api_prefix: str = "/api/v1"

    # OpenAI direct (si Azure non utilisé) — OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL
    openai_api_key: str = ""
    openai_embedding_model: str = "l"
    openai_chat_model: str = ""

    # Azure OpenAI / Azure AI Foundry — chargé depuis backend/.env
    # AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (obligatoires pour Foundry)
    azure_openai_api_key: str = ""
    azure_openai_api_key_chat: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_endpoint_chat: str = ""
    azure_openai_api_version: str = ""
    azure_openai_api_version_chat: str = ""
    # Chat Foundry : gpt-5-chat. Embedding : text-embedding-3-small
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = ""
    azure_openai_deployment_name: str = ""

    # Chroma — CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "documents"

    # RAG — CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieve: int = 50
    # USE_RAG : si True, le chat utilise les documents indexés (Chroma) pour répondre ; sinon chat simple (LLM seul)
    use_rag: bool = True

    # DEBUG (env DEBUG) : True → niveau logging DEBUG pour le package app (ex. app.haystack_rag).
    debug: bool = True

    @property
    def use_azure_openai(self) -> bool:
        return bool(self.azure_openai_api_key and self.azure_openai_endpoint)

    @property
    def azure_endpoint_normalized(self) -> str:
        return _normalize_azure_endpoint(self.azure_openai_endpoint)

    @property
    def azure_endpoint_normalized_chat(self) -> str:
        return _normalize_azure_endpoint(self.azure_openai_endpoint_chat)
    @property
    def azure_chat_deployment(self) -> str:
        """Nom du déploiement chat : AZURE_OPENAI_CHAT_DEPLOYMENT prioritaire, sinon AZURE_OPENAI_DEPLOYMENT_NAME."""
        return self.azure_openai_chat_deployment or self.azure_openai_deployment_name

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()

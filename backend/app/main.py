"""
Point d'entrée FastAPI : montage des routes chat + documents sous /api/v1.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import chat

settings = get_settings()
app = FastAPI(title=settings.api_title)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=settings.api_prefix)


@app.on_event("startup")
def log_config():
    """Affiche l'endpoint et le mode RAG au démarrage."""
    ep = settings.azure_endpoint_normalized or "(OpenAI direct)"
    ep_chat = settings.azure_endpoint_normalized_chat or "(OpenAI direct)"
    print(f"[Config] Endpoint: {ep}")
    print(f"[Config] Chat Endpoint: {ep_chat}")
    print(f"[Config] Chat deployment: {settings.azure_chat_deployment} (si 404 DeploymentNotFound, vérifier ce nom dans le portail Foundry)")
    print(f"[Config] USE_RAG={settings.use_rag}")


@app.get("/health")
def health():
    return {"status": "ok"}

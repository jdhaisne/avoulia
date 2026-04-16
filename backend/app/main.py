"""
Point d'entrée FastAPI : montage des routes chat + documents sous /api/v1.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import chat

settings = get_settings()


def _configure_app_logging() -> None:
    """
    Si Settings.debug est True : niveau DEBUG pour `app` et sous-loggers, avec handler dédié.
    (Sinon les DEBUG seraient filtrés par le logger racine souvent à INFO sous uvicorn.)
    """
    level = logging.DEBUG if settings.debug else logging.INFO

    # Uvicorn configure souvent le logging racine ; on force le niveau voulu ici.
    root = logging.getLogger()
    root.setLevel(level)

    pkg = logging.getLogger("app")
    pkg.setLevel(level)
    if settings.debug:
        if not pkg.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
            pkg.addHandler(handler)
        for handler in pkg.handlers:
            handler.setLevel(logging.DEBUG)
        pkg.propagate = False
    else:
        pkg.handlers.clear()
        pkg.propagate = True

    # Harmonise aussi les loggers uvicorn pour voir les traces DEBUG au runtime.
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)


_configure_app_logging()

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
    print(f"[Config] DEBUG={settings.debug} (loggers app.* → {'DEBUG' if settings.debug else 'INFO'})")


@app.get("/health")
def health():
    return {"status": "ok"}

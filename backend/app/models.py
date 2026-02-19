from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    """Un message dans la conversation."""

    role: str = Field(..., description="'user' ou 'assistant'")
    content: str = Field(..., description="Contenu du message")


class ChatRequest(BaseModel):
    """Requête envoyée au endpoint chat."""

    message: str = Field(..., min_length=1, description="Message de l'utilisateur")
    history: list[ChatMessage] = Field(default_factory=list, description="Historique de la conversation")


class ChatResponse(BaseModel):
    """Réponse du chatbot."""

    answer: str = Field(..., description="Réponse générée")
    sources: list[str] = Field(default_factory=list, description="Extraits de documents utilisés")


class IngestResponse(BaseModel):
    """Réponse après ingestion de documents."""

    success: bool = True
    message: str = "Documents ingérés avec succès"
    count: int = 0
    ids: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Info sur un document indexé (pour listing)."""

    id: str
    metadata: dict = Field(default_factory=dict)

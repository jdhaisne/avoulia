from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    """Un message dans la conversation."""

    role: str = Field(..., description="'user' ou 'assistant'")
    content: str = Field(..., description="Contenu du message")


class SuggestedCase(BaseModel):
    """Un cas suggéré (id + contenu), pour last_suggested_cases et suggested_cases."""

    id: str = Field(..., description="Identifiant du cas")
    content: str = Field(..., description="Contenu du cas (complet pour le détail)")


class ChatRequest(BaseModel):
    """Requête envoyée au endpoint chat."""

    message: str = Field(..., min_length=1, description="Message de l'utilisateur")
    history: list[ChatMessage] = Field(default_factory=list, description="Historique de la conversation")
    last_suggested_cases: list[SuggestedCase] | None = Field(
        default=None,
        description="Liste des cas proposés au tour précédent (pour détail / « ok vas-y »).",
    )
    pending_action: str | None = Field(
        default=None,
        description="Action en attente (ex. expand_details). À renvoyer quand l'utilisateur dit « ok / vas-y ».",
    )
    pending_use_case_id: str | None = Field(
        default=None,
        description="Id du cas concerné par l'action en attente (renvoyer avec pending_action).",
    )


class ChatResponse(BaseModel):
    """Réponse du chatbot."""

    answer: str = Field(..., description="Réponse générée")
    sources: list[str] = Field(default_factory=list, description="Extraits de documents utilisés")
    suggested_case_ids: list[str] = Field(default_factory=list, description="Ids des cas suggérés (ordre de la liste)")
    suggested_cases: list[SuggestedCase] | None = Field(
        default=None,
        description="Cas suggérés (id + contenu complet). À renvoyer dans last_suggested_cases.",
    )
    pending_action: str | None = Field(
        default=None,
        description="Si l'assistant propose un détail (« Souhaitez-vous le détail du 2ème ? »), à stocker et renvoyer avec le prochain « ok ».",
    )
    pending_use_case_id: str | None = Field(default=None, description="Id du cas proposé pour le détail.")
    pending_case_index: int | None = Field(default=None, description="Numéro 1-based du cas proposé pour le détail.")


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

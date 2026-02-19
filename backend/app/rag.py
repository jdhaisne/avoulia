"""
Service chat : LLM seul (sans RAG pour le moment).
Utilise LangChain et OpenAI ou Azure OpenAI selon la config.
"""

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

from app.config import get_settings


def get_embeddings():
    settings = get_settings()
    if settings.use_azure_openai:
        return AzureOpenAIEmbeddings(
            azure_endpoint=settings.azure_endpoint_normalized,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_openai_embedding_deployment,
        )
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key or None,
    )


def get_vector_store():
    """Retourne le store Chroma (persistant)."""
    settings = get_settings()
    return Chroma(
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
        collection_name=settings.chroma_collection_name,
    )


def get_llm():
    settings = get_settings()
    if settings.use_azure_openai:
        # gpt-5.2-chat n'accepte que temperature=1 (obligatoire)
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_endpoint_normalized,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.azure_chat_deployment,
            temperature=1,

        )
    return ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key or None,
        temperature=1,
    )


def _history_to_messages(history: list[dict]) -> list[BaseMessage]:
    """Convertit l'historique [{role, content}] en messages LangChain."""
    messages: list[BaseMessage] = []
    for m in history:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


CLARIFICATION_SYSTEM = """Tu es un assistant utile. Règle : si le cas du client est vague (demande imprécise, contexte ou objectif peu clair), pose 1 question ou 2 maximum pour préciser le cas du client. Si le cas est déjà clair, réponds directement. Tu réponds en français."""

SYSTEM_ANSWER_AFTER_3 = """Tu es un assistant utile. C'est au moins la 3e demande de l'utilisateur : tente de répondre avec les éléments qu'il a déjà fournis, ne pose plus de questions de clarification. Tu réponds en français."""


def chat_simple(message: str, history: list[dict]) -> str:
    """
    Chat simple : envoie le message + historique au LLM, sans RAG.
    L'agent pose 1 ou 2 questions si le cas est vague, puis tente de répondre au 3e échange.
    """
    llm = get_llm()
    # À partir du 3e échange (4 messages = 2 allers-retours), forcer une réponse
    system = SYSTEM_ANSWER_AFTER_3 if len(history) >= 4 else CLARIFICATION_SYSTEM
    messages: list[BaseMessage] = [SystemMessage(content=system)]
    messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=message))
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def chat_simple_stream(message: str, history: list[dict]):
    """Génère des chunks de réponse (streaming) pour le chat simple."""
    llm = get_llm()
    system = SYSTEM_ANSWER_AFTER_3 if len(history) >= 4 else CLARIFICATION_SYSTEM
    messages: list[BaseMessage] = [SystemMessage(content=system)]
    messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=message))
    for chunk in llm.stream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


def stream_prompt(prompt_text: str):
    """Génère des chunks à partir d'un prompt unique (pour RAG en streaming)."""
    llm = get_llm()
    for chunk in llm.stream([HumanMessage(content=prompt_text)]):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content

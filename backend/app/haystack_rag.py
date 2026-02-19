"""
RAG avec Haystack + Chroma, via Azure AI Foundry.
- Embeddings : Foundry (Azure OpenAI) -> stockage dans Chroma.
- Chat : modèle gpt-5-chat sur Foundry (avec ou sans RAG).
"""

from pathlib import Path

import chromadb
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

from app.config import get_settings


def _secret(key: str) -> Secret:
    """Retourne toujours un Secret (jamais None) pour éviter 'NoneType' has no attribute 'resolve_value'."""
    return Secret.from_token((key or "").strip())


def get_document_store():
    """Store Chroma persistant (Haystack), partagé indexation et RAG."""
    s = get_settings()
    return ChromaDocumentStore(
        persist_path=s.chroma_persist_dir,
        collection_name=s.chroma_collection_name,
    )


def _drop_chroma_collection() -> None:
    """Supprime la collection Chroma (fichiers sous persist_path) pour permettre une recréation avec une nouvelle dimension d'embedding."""
    s = get_settings()
    path = Path(s.chroma_persist_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    try:
        client.delete_collection(name=s.chroma_collection_name)
    except Exception:
        pass


def _get_text_embedder():
    """Embedder pour la requête (Foundry / Azure)."""
    s = get_settings()
    if s.use_azure_openai:
        from haystack.components.embedders import AzureOpenAITextEmbedder
        return AzureOpenAITextEmbedder(
            azure_endpoint=s.azure_endpoint_normalized,
            api_key=_secret(s.azure_openai_api_key),
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_openai_embedding_deployment,
        )
    from haystack.components.embedders import OpenAITextEmbedder
    return OpenAITextEmbedder(
        api_key=_secret(s.openai_api_key),
        model=s.openai_embedding_model,
    )


def _get_document_embedder():
    """Embedder pour les documents (indexation Foundry / Azure)."""
    s = get_settings()
    if s.use_azure_openai:
        from haystack.components.embedders import AzureOpenAIDocumentEmbedder
        return AzureOpenAIDocumentEmbedder(
            azure_endpoint=s.azure_endpoint_normalized,
            api_key=_secret(s.azure_openai_api_key),
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_openai_embedding_deployment,
        )
    from haystack.components.embedders import OpenAIDocumentEmbedder
    return OpenAIDocumentEmbedder(
        api_key=_secret(s.openai_api_key),
        model=s.openai_embedding_model,
    )


def _get_generator():
    """Générateur chat Foundry (gpt-5-chat)."""
    s = get_settings()
    if s.use_azure_openai:
        from haystack.components.generators import AzureOpenAIGenerator
        return AzureOpenAIGenerator(
            azure_endpoint=s.azure_endpoint_normalized,
            api_key=_secret(s.azure_openai_api_key),
            api_version=s.azure_openai_api_version,
            azure_deployment=s.azure_chat_deployment,
        )
    from haystack.components.generators import OpenAIGenerator
    return OpenAIGenerator(
        api_key=_secret(s.openai_api_key),
        model=s.openai_chat_model,
    )


RAG_PROMPT = """Tu es un assistant qui aide l'utilisateur en t'appuyant sur les extraits fournis. Tu réponds toujours en français.
tu t'adresses à des dirigeants ou responsables non techniques.
Aucun nom de produit, d’éditeur ou de technologie ne doit être mentionné.
Chaque recommandation débouche sur un premier pas faisable en 48h.
Toute recommandation doit pouvoir être expliquée à partir des colonnes de la base de données.
L’agent propose d’abord une short list (3–5 cas), puis détaille uniquement à la demande.
Aucun score global n’est utilisé ou affiché. Le classement est déterministe et explicable.
Colonnes utilisables par l’agent (et seulement celles-ci)
- use_case_id
- Cas d’utilisation
- Description du cas d’utilisation
- metier_principal_norm
- primary_value
- execution_mode_pme
- effort_3lvl
- complexity_3lvl
- time_to_value_3lvl
- data_sensitivity
- guardrails_pme
- first_step_48h
- expected_gain_proxy
- data_prerequisites

Étape 1 — Comprendre l’intention utilisateur
L’agent identifie, via au maximum 4 questions : - le métier ou l’équipe cible →
metier_principal_norm - l’objectif principal → primary_value - la préférence de mise
en œuvre → execution_mode_pme - l’horizon de résultats attendu →
time_to_value_3lvl

Étape 2 — Filtrage de la base
Filtres stricts : - metier_principal_norm - primary_value
Filtres d’ajustement : - execution_mode_pme - time_to_value_3lvl - data_sensitivity

Étape 3 — Classement (sans score)
Le classement repose sur des règles simples et explicables :
1. Correspondance exacte métier + objectif

2. Délai compatible (time_to_value_3lvl)
3. Effort acceptable (effort_3lvl)
4. Complexité maîtrisable (complexity_3lvl)
5. Niveau de vigilance données compatible (data_sensitivity)
L’agent explique toujours pourquoi un cas est prioritaire.

Étape 4 — Sélection
1. 3 à 5 cas d’usage maximum
2. Jamais moins de 3 sauf contrainte forte de la base
3. Priorité à la pertinence, pas à l’exhaustivité

Tu es un agent conversationnel expert de l’adoption de l’IA générative pour
les PME françaises.
Ta mission est d’aider une PME à identifier, prioriser et comprendre des cas
d’usage concrets de l’IA générative, de manière simple, responsable et
actionnable.
Tu t’adresses à des dirigeants ou responsables non techniques.
Tu ne cites jamais de noms de produits, d’éditeurs ou de technologies.
Tu ne donnes jamais de ROI financier chiffré ni d’estimation budgétaire.
Tu poses au maximum 4 questions pour comprendre le contexte.
Tu proposes toujours entre 3 et 5 cas d’usage maximum, classés par
pertinence.
Tu expliques systématiquement pourquoi tu proposes ces cas d’usage.
Tu ne donnes le détail complet d’un cas que si l’utilisateur le demande.

Chaque cas détaillé doit inclure :
- l’impact concret (primary_value),
- le gain attendu (expected_gain_proxy),
- le mode, l’effort, la complexité et le délai,
- la vigilance données (data_sensitivity, guardrails_pme),
- un premier pas faisable en 48h,
- les prérequis simples.
Tu n’utilises aucun score global.
Tu es neutre, pédagogique et orienté action.

RÈGLE PRINCIPALE : Si le cas du client est vague (demande imprécise, contexte ou objectif peu clair), pose 1 question ou 2 maximum pour préciser le cas du client avant de répondre. Pas plus de 2 questions. Si le cas est déjà clair ou que les extraits permettent de répondre, réponds directement en t'appuyant sur les extraits.
{% if hint %}
{{ hint }}
{% endif %}

Extraits :
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Demande de l'utilisateur : {{ query }}

Réponse (réponse directe OU 1 à 2 questions de clarification, selon le cas) :"""


def build_rag_prompt_only_pipeline():
    """Pipeline sans générateur : embedder -> retriever -> prompt_builder (pour récupérer le prompt)."""
    s = get_settings()
    store = get_document_store()
    embedder = _get_text_embedder()
    retriever = ChromaEmbeddingRetriever(document_store=store, top_k=s.top_k_retrieve)
    prompt_builder = PromptBuilder(template=RAG_PROMPT)
    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    return pipeline


def build_rag_pipeline():
    """Pipeline RAG : embedder (Foundry) -> retriever (Chroma) -> prompt -> generator (gpt-5-chat)."""
    s = get_settings()
    store = get_document_store()
    embedder = _get_text_embedder()
    retriever = ChromaEmbeddingRetriever(document_store=store, top_k=s.top_k_retrieve)
    prompt_builder = PromptBuilder(template=RAG_PROMPT)
    generator = _get_generator()

    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    return pipeline


def _get_rag_hint(history: list[dict]) -> str:
    if len(history) >= 4:
        return "Important : c'est au moins la 3e demande de l'utilisateur. Tente de répondre avec les extraits et les informations déjà fournies ; ne pose plus de questions de clarification."
    return ""


def get_rag_prompt_and_sources(question: str, history: list[dict]) -> tuple[str, list[str]]:
    """Retourne le prompt RAG (sans génération) et les sources, pour streaming côté appelant."""
    hint = _get_rag_hint(history)
    pipeline = build_rag_prompt_only_pipeline()
    result = pipeline.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"query": question, "hint": hint},
        }
    )
    docs = result.get("retriever", {}).get("documents") or []
    sources = [d.content[:400] + "..." if len(d.content) > 400 else d.content for d in docs]
    out = result.get("prompt_builder", {}) or {}
    prompt_text = out.get("prompt") or ""
    if isinstance(prompt_text, list):
        prompt_text = prompt_text[0] if prompt_text else ""
    return (prompt_text or "Aucun contexte."), sources


def query_rag_haystack(question: str, history: list[dict]) -> tuple[str, list[str]]:
    """Interroge le RAG : recherche Chroma + génération via gpt-5-chat (Foundry)."""
    hint = _get_rag_hint(history)
    pipeline = build_rag_pipeline()
    result = pipeline.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"query": question, "hint": hint},
        }
    )
    docs = result.get("retriever", {}).get("documents", [])
    sources = [d.content[:400] + "..." if len(d.content) > 400 else d.content for d in docs]
    replies = result.get("generator", {}).get("replies", [])
    answer = replies[0] if replies else "Aucune réponse générée."
    return answer, sources


def clear_all_documents() -> None:
    """Supprime tous les documents et la collection Chroma. La prochaine indexation recréera la collection avec la dimension d'embedding actuelle."""
    try:
        store = get_document_store()
        store.delete_all_documents(recreate_index=False)
    except Exception:
        pass
    _drop_chroma_collection()


def index_documents_haystack(documents: list[Document]) -> int:
    """Indexe des documents dans Chroma : embedding via Foundry puis écriture."""
    if not documents:
        return 0
    embedder = _get_document_embedder()
    store = get_document_store()
    embedded = embedder.run(documents=documents)
    docs_with_embeddings = embedded.get("documents", documents)
    try:
        return store.write_documents(docs_with_embeddings)
    except Exception as e:
        err_msg = str(e).lower()
        # Collection créée avec une autre dimension (ex. 384 vs 1536) : on supprime et on réessaie
        if "dimension" in err_msg or "384" in err_msg or "1536" in err_msg:
            _drop_chroma_collection()
            store = get_document_store()
            return store.write_documents(docs_with_embeddings)
        raise

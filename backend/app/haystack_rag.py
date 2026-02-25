"""
RAG avec Haystack + Chroma, via Azure AI Foundry.
- Embeddings : Foundry (Azure OpenAI) -> stockage dans Chroma.
- Chat : modèle gpt-5-chat sur Foundry (avec ou sans RAG).
"""

import re
from pathlib import Path

import chromadb
from jinja2 import Template
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


RAG_PROMPT = """
Tu es un agent conversationnel spécialisé dans l’identification de cas d’usage d’IA générative pour dirigeants et responsables de PME françaises.

Tu fonctionnes exclusivement selon un parcours guidé structuré.
L’utilisateur ne commence jamais en langage libre.
Tu poses des questions fermées successives.
Tu n’interprètes jamais librement les réponses.
Tu ne modifies jamais le domaine ou l’intention sans validation explicite.
tu t'adresses à des dirigeants ou responsables non techniques.
Aucun nom de produit, d’éditeur ou de technologie ne doit être mentionné.

Aucun score global n’est utilisé ou affiché. Le classement est déterministe et explicable.
Colonnes utilisables par l’agent (et seulement celles-ci)
- cas_utilisation
- domaine
- intention
- micro_theme
- description_cas_utilisation
- declencheurs_typiques
- questions_qualification
- secteur
- mode_execution
- effort
- sensibilite_donnees
- guardrails
- prerequis_donnees
- premiere_action_48h
- rag_text_auto
- domaine_label

Tu commences toujours par afficher EXACTEMENT le message suivant :

"Bonjour, je vais vous aider à identifier des cas d’usage concrets de l’IA adaptés à votre organisation. Pour commencer, je vais vous poser quelques questions simples afin de cibler précisément votre priorité."

Étape 1 — Comprendre l’intention utilisateur

Q1 — Domaine

Tu poses EXACTEMENT la question suivante :

"Dans quel domaine souhaitez-vous agir en priorité ?"

Tu proposes EXACTEMENT les choix suivants :

1. Direction & décisions stratégiques
2. Organisation & efficacité interne
3. RH & gestion des équipes
4. Développement commercial
5. Marketing & visibilité
6. Service & relation client
7. Finances & rentabilité
8. Outils, systèmes & données
9. Obligations & gestion des risques
10. Achats & relations fournisseurs
11. Stocks & logistique
12. Production & opérations
13. Chantiers & activités terrain
14. Innovation & nouveaux projets

Règles :
- L’utilisateur doit choisir un seul domaine.
- Tu n’expliques pas les domaines.
- Si la réponse ne correspond pas exactement à un choix proposé, tu redemandes de choisir parmi la liste.

Q1.5 — Secteur (optionnel)

Tu poses EXACTEMENT :

"Dans quel secteur évoluez-vous ? (facultatif)"

Choix possibles :

- Industrie
- Commerce de proximité
- Restauration
- BTP / Construction
- Transport & Logistique
- Autre / Non précisé

Règles :
- Tu acceptes une absence de réponse.
- Tu n’infères jamais un secteur.
- Si la réponse ne correspond pas à la liste, tu redemandes un choix valide.

Q2 — Objectif principal

Tu poses EXACTEMENT :

"Quel est votre objectif principal dans ce domaine ?"

Règles :
- Tu proposes uniquement les intentions correspondant au domaine sélectionné.
- Tu n’inventes jamais d’intention hors domaine.
- Tu ne reformules pas les intentions.
- Si la réponse ne correspond pas à la liste fournie, tu redemandes un choix valide.

Q2.5 — Précision (si nécessaire uniquement)

Si le backend indique que l’intention couvre plusieurs micro-thèmes, tu poses EXACTEMENT :

"Pouvez-vous préciser le type de sujet concerné ?"

Règles :
- Tu ne poses cette question que si nécessaire.
- Tu proposes uniquement les micro-thèmes fournis par le backend.
- Tu n’inventes jamais de micro-thème.

Q3 — Problème concret

Tu poses EXACTEMENT :

"Pouvez-vous décrire en une ou deux phrases le problème concret que vous rencontrez actuellement ?"

Règles :
- Réponse courte attendue.
- Tu n’analyses pas le problème.
- Tu ne changes jamais de domaine.
- Tu ne modifies jamais le filtrage.
- Tu ne reclasses rien.

-------------------------------------
PHASE 1 BIS — FALLBACK INCOHÉRENCE DOMAINE
-------------------------------------

Si le backend fournit un domaine suggéré en cas d’incohérence potentielle, tu affiches EXACTEMENT :

"Votre situation semble également concerner le domaine suivant :

[Nom du domaine suggéré]

Souhaitez-vous explorer également ce domaine ?"

Règles :
- Tu ne changes jamais automatiquement de domaine.
- Tu attends la décision explicite de l’utilisateur.

-------------------------------------
PHASE 2 — PRÉSENTATION DES CAS
-------------------------------------

Les cas fournis sont déjà :
- filtrés par domaine
- filtrés par intention
- éventuellement filtrés par micro-thème
- éventuellement priorisés par secteur
- sélectionnés de manière déterministe

Tu ne modifies jamais cet ordre.
Tu ne reclasses jamais.
Tu ne scores rien.
Tu ne supprimes rien sauf si plus de 5 cas sont fournis.

Tu présentes :
- Minimum 3 cas
- Maximum 5 cas
- Un seul use_case_id par bloc
- Aucun mélange

-------------------------------------
FORMAT OBLIGATOIRE POUR CHAQUE CAS
-------------------------------------

🔹 Nom du cas

Pourquoi c’est pertinent pour vous :
(1 à 2 phrases contextualisées par rapport au problème exprimé.)

Ce que cela permet concrètement :
(Description claire et opérationnelle, sans jargon technique.)

Première étape simple :
(Reformulation claire de la première action proposée.)

-------------------------------------
RÈGLES STRICTES
-------------------------------------

Tu ne :
- promets jamais de ROI chiffré
- recommandes jamais un outil spécifique
- mentionnes jamais le système interne
- expliques jamais le mécanisme de filtrage
- modifies jamais la sélection fournie
- ajoutes jamais un sixième cas
- inventes jamais un cas
- interprètes jamais la taxonomie

Ton ton est :
- clair
- structuré
- professionnel
- accessible à un dirigeant de PME
- sans jargon IA
- sans discours marketing

Objectif final :
Aider un dirigeant à comprendre ses options,
décider par quoi commencer,
et avancer concrètement.

Tu dois tenir compte de l'historique de la conversation : métier, objectifs, contraintes et réponses déjà données par l'utilisateur. Ne redemande pas ce qu'il a déjà dit. Enchaîne de façon cohérente.
{% if hint %}
{{ hint }}
{% endif %}
{% if conversation_history %}
Historique récent de la conversation (utilise-le pour garder le contexte) :
{{ conversation_history }}

{% endif %}
Extraits (ordre fixe : Cas 1 = point 1, Cas 2 = point 2, etc. Présente toujours ta liste dans cet ordre) :
{% for doc in documents %}
--- Cas {{ loop.index }} ---
{{ doc.content }}

{% endfor %}

RÈGLE : Ne propose et ne détaille que les cas listés ci-dessus (Cas 1 à Cas {{ documents|length }}). Si l'utilisateur demande « le point 2 », c'est toujours le Cas 2 ci-dessus. Ne confonds jamais les numéros.

Demande actuelle de l'utilisateur : {{ query }}

Réponse (réponse directe OU 1 à 2 questions de clarification, en tenant compte de l'historique) :"""


# Prompt pour détailler un seul cas (liste déjà connue, pas de re-retrieval).
DETAIL_PROMPT = """Tu es un assistant expert qui détaille un cas d'usage de l'IA générative pour une PME.
Tu t'adresses à des dirigeants ou responsables non techniques. Tu réponds toujours en français.
Tu ne cites jamais de noms de produits, d'éditeurs ou de technologies.

L'utilisateur a demandé à détailler un cas précis que tu avais proposé. Tu dois décrire UNIQUEMENT ce cas-ci, en t'appuyant exclusivement sur les données fournies ci-dessous. Ne décris aucun autre cas. Extrais et présente les informations suivantes lorsqu'elles sont présentes dans les données :

- l'impact concret (primary_value),
- le gain attendu (expected_gain_proxy),
- le mode, l'effort, la complexité et le délai,
- la vigilance données (data_sensitivity, guardrails_pme),
- un premier pas faisable en 48h,
- les prérequis simples.

Données complètes du cas à détailler :
{{ case_content }}

Réponse (détail de ce cas uniquement, d'après les données ci-dessus) :"""


def _build_rag_prompt_from_docs(
    query: str,
    hint: str,
    conversation_history: str,
    documents: list,
) -> str:
    """
    Construit le prompt RAG à partir de la même liste de documents utilisée pour
    suggested_case_ids / full_contents, pour garantir que l'ordre (Cas 1, Cas 2, …)
    est identique entre le prompt envoyé au LLM et la liste affichée.
    """
    template = Template(RAG_PROMPT)
    return template.render(
        query=query or "",
        hint=hint or "",
        conversation_history=conversation_history or "",
        documents=documents or [],
    )


def _is_detail_request(message: str) -> bool:
    """
    Détecte si le message demande à détailler UN point précis de la liste déjà proposée.
    On ne doit pas déclencher pour une quantité (ex. « je veux 3 cas », « donne-moi 2 idées »).
    """
    if not message or len(message.strip()) < 2:
        return False
    msg = message.strip().lower()
    # Verbes / formulations qui indiquent « détaille ce point » ou « donne le détail de »
    detail_verbs = [
        "détaille", "détailler", "detaille", "détaillant", "détails", "detail",
        "développe", "developpe", "précise", "precise",
        "plus d'info", "plus d info", "en savoir plus",
        "parle-moi du", "parle moi du", "explique le", "explique la",
        "dis-moi plus", "dis moi plus", "décris le", "decris le", "décris la", "decris la",
        "donne le détail", "donne les détails", "donne-moi le détail", "donne moi le détail",
        "veux le détail", "voudrais le détail", "je veux le détail", "je voudrais le détail",
        "le détail du", "les détails du", "détail du point", "détails du point",
    ]
    # Référence à un rang précis (« le 2ème », « point 3 ») — pas une quantité
    rank_refs = [
        "le premier", "le 1er", "le deuxieme", "le 2ème", "le 2eme", "le 2e",
        "le troisieme", "le 3ème", "le 3eme", "le 4ème", "le 5ème",
        "point 1", "point 2", "point 3", "point 4", "point 5",
        "numéro 1", "numero 1", "numéro 2", "numero 2", "lequel sur", "celui sur", "celle sur",
    ]
    # « point N » ou « le Nème » dans le message = demande de détail ciblée
    if re.search(r"point\s*[1-5]\b", msg) or re.search(r"(?:le\s+)?[1-5]\s*(?:er|ème|e|eme)\b", msg):
        return True
    has_verb = any(v in msg for v in detail_verbs)
    has_rank = any(r in msg for r in rank_refs)
    # Verbe + chiffre (ex. « détaille le 2 », « développe le 3 »)
    verb_then_num = re.search(
        r"\b(?:détaille|détailler|développe|précise|explique|décris)\b.*\b(?:le\s+)?([1-5])(?:er|ème|e|eme)?\b",
        msg,
    )
    # Ne jamais déclencher sur un chiffre seul ou une quantité (ex. « 3 cas », « 2 idées »)
    if re.search(r"\b[1-5]\s+(?:cas|idées|propositions|suggestions|exemples)\b", msg):
        return False
    if re.search(r"(?:veux|voudrais|donne|avoir)\s+[1-5]\s", msg):
        return False
    return has_verb or has_rank or bool(verb_then_num)


def _has_explicit_point_number(message: str) -> bool:
    """
    True si le message contient une référence numérique explicite (point 2, 2ème, le 3, etc.).
    Dans ce cas on ne doit détailler QUE si on a last_suggested_cases (même ordre que la liste affichée).
    """
    if not message or len(message.strip()) < 2:
        return False
    msg = message.strip().lower()
    if re.search(r"point\s*[1-5]\b", msg):
        return True
    if re.search(r"(?:le\s+)?[1-5]\s*(?:er|ème|e|eme)\b", msg):
        return True
    if re.search(r"(?:premier|1er|deuxième|2ème|troisième|3ème|quatrième|4ème|cinquième|5ème)", msg):
        return True
    if re.search(r"(?:le|numero|numéro)\s*[1-5]\b", msg):
        return True
    return False


def _is_affirmation(message: str) -> bool:
    """Détecte si le message est une affirmation courte (ok, vas-y, oui, etc.) pour exécuter l'action en attente."""
    if not message or len(message.strip()) > 80:
        return False
    msg = message.strip().lower()
    affirmations = [
        "ok", "okay", "vas-y", "vas y", "oui", "ouais", "d'accord", "d accord",
        "go", "allez", "oui vas-y", "ok vas-y", "c'est parti", "oui s'il te plaît",
        "je veux le détail", "oui je veux", "je le souhaite", "oui allez-y",
    ]
    if msg in ("ok", "oui", "go", "vas-y", "vas y", "ouais", "d'accord", "allez"):
        return True
    return any(a in msg for a in affirmations)


def _get_last_assistant_message(history: list[dict]) -> str | None:
    """Retourne le contenu du dernier message assistant dans l'historique."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "assistant":
            content = history[i].get("content") or ""
            if content.strip():
                return content.strip()
    return None


def _parse_offer_detail_from_text(text: str) -> int | None:
    """
    Si le texte de l'assistant propose le détail d'un cas (ex. « Souhaitez-vous le détail du 1er ? »),
    retourne l'index 1-based du cas proposé. On ancre la zone sur la QUESTION (souhaitez-vous / voulez-vous)
    pour ne pas prendre un « 2ème » ou « détail du 2 » venant d'une phrase plus haut dans le message.
    """
    if not text or len(text) < 10:
        return None
    msg = text.strip().lower()
    # Privilégier la phrase-question (où se trouve le bon numéro), pas une mention antérieure de "détail du"
    question_markers = [
        "souhaitez-vous le détail", "souhaitez vous le détail",
        "voulez-vous le détail", "voulez vous le détail",
        "veux-tu le détail", "veux tu le détail",
    ]
    offer_start = -1
    for p in question_markers:
        i = msg.find(p)
        if i >= 0:
            offer_start = i
            break
    if offer_start < 0:
        fallback = ["détail du ", "détail de la ", "que je détaille le", "que je détaille la"]
        for p in fallback:
            i = msg.find(p)
            if i >= 0:
                offer_start = i
                break
    if offer_start < 0:
        return None
    zone = msg[offer_start : offer_start + 80]
    # Numéro qui suit directement « détail du » dans la zone (= celui proposé)
    right_after = re.search(
        r"détail\s+(?:du|de\s+la)\s+(?:le\s+)?(premier|1er|1ère|deuxième|2ème|2eme|troisième|3ème|3eme|quatrième|4ème|cinquième|5ème|\d)\s*(?:er|ème|e|eme)?\b",
        zone,
        re.IGNORECASE,
    )
    if right_after:
        word = right_after.group(1).lower()
        ord_map = {"premier": 1, "1er": 1, "1ère": 1, "deuxième": 2, "2ème": 2, "2eme": 2, "troisième": 3, "3ème": 3, "3eme": 3, "quatrième": 4, "4ème": 4, "cinquième": 5, "5ème": 5}
        if word in ord_map:
            return ord_map[word]
        if word.isdigit():
            n = int(word)
            if 1 <= n <= 5:
                return n
    ordinals_1based = [
        ("premier", 1), ("1er", 1), ("1ère", 1), ("1e ", 1),
        ("deuxième", 2), ("2ème", 2), ("2eme", 2), ("2e ", 2),
        ("troisième", 3), ("3ème", 3), ("3eme", 3),
        ("quatrième", 4), ("4ème", 4), ("cinquième", 5), ("5ème", 5),
    ]
    for phrase, idx in ordinals_1based:
        if phrase in zone:
            return idx
    m = re.search(r"(?:cas|point|numéro?)\s*(\d)", zone, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 5:
            return n
    m = re.search(r"\b(?:le\s+)?(\d)\s*(?:er|ème|e|eme)?\b", zone, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 5:
            return n
    return None


def _get_previous_user_message(history: list[dict]) -> str | None:
    """Retourne le dernier message utilisateur dans l'historique (pour refaire une recherche)."""
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            content = history[i].get("content") or ""
            if content.strip():
                return content.strip()
    return None


def _retrieve_docs(query: str, top_k: int | None = None) -> list:
    """Lance une recherche RAG et retourne la liste de documents (sans génération)."""
    s = get_settings()
    k = top_k or s.top_k_retrieve
    store = get_document_store()
    embedder = _get_text_embedder()
    retriever = ChromaEmbeddingRetriever(document_store=store, top_k=k)
    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    result = pipeline.run({"embedder": {"text": query}})
    return result.get("retriever", {}).get("documents") or []


def _resolve_detail_selection(
    message: str, last_suggested_cases: list[dict]
) -> int | None:
    """
    Détermine quel cas de la liste précédente l'utilisateur demande à détailler.
    Retourne l'index 0-based ou None si ambigu.
    """
    if not last_suggested_cases:
        return None
    msg = message.strip().lower()
    n = len(last_suggested_cases)

    # Résolution par numéro : index 0-based (point 1 -> 0, point 2 -> 1, etc.)
    # Priorité aux formes explicites "point N" pour éviter toute ambiguïté
    point_num = re.search(r"point\s*(\d+)", msg, re.IGNORECASE)
    if point_num:
        idx = int(point_num.group(1)) - 1
        if 0 <= idx < n:
            return idx
        return None  # numéro explicite hors plage : ne pas détailler un cas au hasard
    ordinals = {
        "premier": 0, "1er": 0, "1e ": 0, "1e": 0, "1 ": 0, " 1": 0,
        "deuxième": 1, "2ème": 1, "2eme": 1, "2e ": 1, "2e": 1, " 2": 1, "le 2": 1,
        "troisième": 2, "3ème": 2, "3eme": 2, "3e ": 2, "3e": 2, "le 3": 2,
        "quatrième": 3, "4ème": 3, "4eme": 3, "4e": 3, "le 4": 3,
        "cinquième": 4, "5ème": 4, "5eme": 4, "5e": 4, "le 5": 4,
    }
    for phrase, idx in ordinals.items():
        if phrase in msg:
            if idx < n:
                return idx
            return None  # ordinal explicite hors plage
    # Regex : "le 2", "numéro 2"
    num_match = re.search(r"(?:le|numero|numéro)\s*(\d+)", msg, re.IGNORECASE)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < n:
            return idx
        return None
    # Un chiffre seul en début ou après "le/la"
    simple_num = re.search(r"\b(?:le\s+)?(\d)\s*(?:er|ème|e|eme)?\b", msg)
    if simple_num:
        idx = int(simple_num.group(1)) - 1
        if 0 <= idx < n:
            return idx
        return None

    # Résolution par thème uniquement s'il n'y a pas de numéro explicite (ex. « celui sur la synthèse »)
    msg_words = set(re.findall(r"\w{3,}", msg)) - {"détaille", "detailler", "detail", "plus", "info", "savoir", "premier", "deuxieme", "trois", "quatre", "cinq", "point", "numero", "lequel", "celui", "celle", "sur", "cas", "usage"}
    if not msg_words:
        return None
    best_idx = None
    best_score = 0
    for i, item in enumerate(last_suggested_cases):
        content = (item.get("content") or "").lower()
        score = sum(1 for w in msg_words if w in content)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score > 0 else None


def build_rag_retrieval_only_pipeline():
    """Pipeline retrieval seul : embedder -> retriever. Pour construire le prompt nous-mêmes depuis les mêmes docs."""
    s = get_settings()
    store = get_document_store()
    embedder = _get_text_embedder()
    retriever = ChromaEmbeddingRetriever(document_store=store, top_k=s.top_k_retrieve)
    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", retriever)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    return pipeline


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


def _run_detail_pipeline(case_content: str) -> str:
    """Génère une réponse de détail pour un seul cas (pas de retrieval)."""
    prompt_str = DETAIL_PROMPT.replace("{{ case_content }}", case_content)
    print("[LLM PROMPT (détail)]", "-" * 40)
    print(prompt_str)
    print("-" * 40)
    prompt_builder = PromptBuilder(template=DETAIL_PROMPT)
    generator = _get_generator()
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    result = pipeline.run({"prompt_builder": {"case_content": case_content}})
    replies = result.get("generator", {}).get("replies", [])
    return replies[0] if replies else "Aucune réponse générée."


def _format_conversation_history(history: list[dict], max_messages: int = 20) -> str:
    """
    Formate l'historique pour l'injection dans le prompt RAG (derniers échanges).
    Limite à max_messages pour ne pas dépasser la fenêtre de contexte.
    """
    if not history:
        return ""
    recent = history[-max_messages:] if len(history) > max_messages else history
    lines = []
    for m in recent:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "Utilisateur" if role == "user" else "Assistant"
        lines.append(f"{label} : {content}")
    return "\n".join(lines)


def _get_rag_hint(history: list[dict]) -> str:
    if len(history) >= 4:
        return "Important : c'est au moins la 3e demande de l'utilisateur. Tente de répondre avec les extraits et les informations déjà fournies ; ne pose plus de questions de clarification."
    return ""


def get_rag_prompt_and_sources(
    question: str,
    history: list[dict],
    last_suggested_cases: list[dict] | None = None,
    pending_action: str | None = None,
    pending_use_case_id: str | None = None,
) -> tuple[str, list[str], list[str], list[str]]:
    """
    Retourne (prompt, sources, suggested_case_ids, full_contents).
    Gère « ok / vas-y » + pending_action expand_details : retourne le prompt de détail pour le cas en attente.
    """
    # Affirmation + action en attente → prompt de détail pour ce cas
    if _is_affirmation(question) and pending_action == "expand_details" and pending_use_case_id and last_suggested_cases:
        for case in last_suggested_cases:
            if (case.get("id") or "") == pending_use_case_id:
                content = case.get("content") or ""
                if len(content.strip()) > 20:
                    prompt = DETAIL_PROMPT.replace("{{ case_content }}", content)
                    sources = [content[:400] + "..." if len(content) > 400 else content]
                    ids = [c.get("id", "") for c in last_suggested_cases]
                    full_contents = [c.get("content", "") for c in last_suggested_cases]
                    return prompt, sources, ids, full_contents
    # Affirmation « ok » sans pending_* : uniquement si on a last_suggested_cases (ordre fiable)
    if _is_affirmation(question) and last_suggested_cases:
        last_assistant = _get_last_assistant_message(history)
        if last_assistant:
            case_index_1based = _parse_offer_detail_from_text(last_assistant)
            if case_index_1based is not None and 1 <= case_index_1based <= len(last_suggested_cases):
                content = (last_suggested_cases[case_index_1based - 1].get("content") or "").strip()
                if len(content) > 50:
                    prompt = DETAIL_PROMPT.replace("{{ case_content }}", content)
                    sources = [content[:400] + "..." if len(content) > 400 else content]
                    ids = [c.get("id", "") for c in last_suggested_cases]
                    full_contents = [c.get("content", "") for c in last_suggested_cases]
                    return prompt, sources, ids, full_contents

    if last_suggested_cases and _is_detail_request(question):
        idx = _resolve_detail_selection(question, last_suggested_cases)
        if idx is not None:
            case = last_suggested_cases[idx]
            content = case.get("content") or ""
            prompt = DETAIL_PROMPT.replace("{{ case_content }}", content)
            sources = [content[:400] + "..." if len(content) > 400 else content]
            ids = [c.get("id", "") for c in last_suggested_cases]
            full_contents = [c.get("content", "") for c in last_suggested_cases]
            return prompt, sources, ids, full_contents

    hint = _get_rag_hint(history)
    conversation_history = _format_conversation_history(history)
    pipeline = build_rag_prompt_only_pipeline()
    result = pipeline.run(
        {
            "embedder": {"text": question},
            "prompt_builder": {"query": question, "hint": hint, "conversation_history": conversation_history},
        }
    )
    docs = result.get("retriever", {}).get("documents") or []
    # Liste plate : même ordre que pour l'affichage (Cas 1 = docs[0], etc.)
    if docs and isinstance(docs[0], list):
        docs = [d for sub in docs for d in sub]
    sources = [d.content[:400] + "..." if len(d.content) > 400 else d.content for d in docs]
    suggested_case_ids = [getattr(d, "id", None) or str(i) for i, d in enumerate(docs)]
    full_contents = [d.content for d in docs]
    # Prompt construit à partir des MÊMES docs → ordre identique à la liste affichée
    prompt_text = _build_rag_prompt_from_docs(question, hint, conversation_history, docs)
    return (prompt_text or "Aucun contexte."), sources, suggested_case_ids, full_contents


def _try_detail_flow(
    question: str,
    history: list[dict],
    last_suggested_cases: list[dict] | None,
) -> tuple[str, list[str], list[str], list[str]] | None:
    """
    Si la question est une demande de détail et qu'on peut déterminer quel cas (avec
    last_suggested_cases ou en refaisant une recherche avec le dernier message user),
    retourne (answer, sources, suggested_case_ids, full_contents). Sinon None.
    """
    if not _is_detail_request(question):
        return None

    # 1) Utiliser last_suggested_cases si fourni et avec du contenu (seule source fiable pour l'ordre)
    if last_suggested_cases:
        idx = _resolve_detail_selection(question, last_suggested_cases)
        if idx is not None:
            case = last_suggested_cases[idx]
            content = case.get("content") or ""
            if len(content.strip()) > 50:  # contenu utilisable
                answer = _run_detail_pipeline(content)
                sources = [content[:400] + "..." if len(content) > 400 else content]
                ids = [c.get("id", "") for c in last_suggested_cases]
                full_contents = [c.get("content", "") for c in last_suggested_cases]
                return answer, sources, ids, full_contents

    # 2) Référence explicite (point 2, 2ème…) SANS liste : ne pas deviner avec une autre recherche.
    #    L'ordre des docs récupérés ne correspond pas à la liste affichée → on éviterait la confusion.
    if _has_explicit_point_number(question):
        return None

    # 3) Fallback uniquement pour demande par thème (ex. « détaille celui sur la synthèse »)
    previous = _get_previous_user_message(history)
    if not previous:
        return None
    docs = _retrieve_docs(previous)
    if not docs:
        return None
    cases_from_docs = [
        {"id": getattr(d, "id", None) or str(i), "content": d.content}
        for i, d in enumerate(docs)
    ]
    idx = _resolve_detail_selection(question, cases_from_docs)
    if idx is None:
        return None
    content = docs[idx].content
    answer = _run_detail_pipeline(content)
    sources = [content[:400] + "..." if len(content) > 400 else content]
    ids = [getattr(d, "id", None) or str(i) for i, d in enumerate(docs)]
    full_contents = [d.content for d in docs]
    return answer, sources, ids, full_contents


def _execute_pending_expand_details(
    pending_use_case_id: str,
    last_suggested_cases: list[dict],
) -> tuple[str, list[str], list[str], list[str]] | None:
    """Exécute l'action expand_details pour le cas donné. Retourne (answer, sources, ids, full_contents) ou None."""
    for case in last_suggested_cases:
        if (case.get("id") or "") == pending_use_case_id:
            content = case.get("content") or ""
            if len(content.strip()) < 20:
                return None
            answer = _run_detail_pipeline(content)
            sources = [content[:400] + "..." if len(content) > 400 else content]
            ids = [c.get("id", "") for c in last_suggested_cases]
            full_contents = [c.get("content", "") for c in last_suggested_cases]
            return answer, sources, ids, full_contents
    return None


def query_rag_haystack(
    question: str,
    history: list[dict],
    last_suggested_cases: list[dict] | None = None,
    pending_action: str | None = None,
    pending_use_case_id: str | None = None,
) -> tuple[str, list[str], list[str], list[str], str | None, str | None, int | None]:
    """
    Interroge le RAG. Retourne (answer, sources, suggested_case_ids, full_contents, pending_action, pending_use_case_id, pending_case_index).
    Si l'utilisateur dit « ok / vas-y / oui » et qu'une action est en attente (ex. expand_details),
    on l'exécute au lieu de relancer une recommandation.
    Quand la réponse de l'assistant propose un détail (« Souhaitez-vous le détail du 2ème ? »),
    on renvoie pending_action=expand_details et pending_use_case_id pour que le client les renvoie au prochain « ok ».
    """
    # 1) Affirmation + action en attente fournie par le client → exécuter l'action
    if _is_affirmation(question) and pending_action == "expand_details" and pending_use_case_id and last_suggested_cases:
        result = _execute_pending_expand_details(pending_use_case_id, last_suggested_cases)
        if result is not None:
            a, s, i, f = result
            return a, s, i, f, None, None, None

    # 2) Affirmation sans pending_* : inférer depuis le dernier message assistant (offre de détail)
    #    Uniquement si on a last_suggested_cases (ordre garanti). Sinon on ne devine pas.
    if _is_affirmation(question) and last_suggested_cases:
        last_assistant = _get_last_assistant_message(history)
        if last_assistant:
            case_index_1based = _parse_offer_detail_from_text(last_assistant)
            if case_index_1based is not None and 1 <= case_index_1based <= len(last_suggested_cases):
                case = last_suggested_cases[case_index_1based - 1]
                content = case.get("content") or ""
                if len(content.strip()) > 50:
                    answer = _run_detail_pipeline(content)
                    sources = [content[:400] + "..." if len(content) > 400 else content]
                    ids = [c.get("id", "") for c in last_suggested_cases]
                    full_contents = [c.get("content", "") for c in last_suggested_cases]
                    return answer, sources, ids, full_contents, None, None, None

    # 3) Demande explicite de détail (« détaille le 2 »)
    detail_result = _try_detail_flow(question, history, last_suggested_cases)
    if detail_result is not None:
        a, s, i, f = detail_result
        return a, s, i, f, None, None, None

    # 4) Flux RAG normal (liste de cas) : retrieval → prompt construit depuis les MÊMES docs → generator
    hint = _get_rag_hint(history)
    conversation_history = _format_conversation_history(history)
    retrieval_pipeline = build_rag_retrieval_only_pipeline()
    result = retrieval_pipeline.run({"embedder": {"text": question}})
    docs = result.get("retriever", {}).get("documents") or []
    if docs and isinstance(docs[0], list):
        docs = [d for sub in docs for d in sub]
    sources = [d.content[:400] + "..." if len(d.content) > 400 else d.content for d in docs]
    suggested_case_ids = [getattr(d, "id", None) or str(i) for i, d in enumerate(docs)]
    full_contents = [d.content for d in docs]
    # Prompt construit à partir des MÊMES docs → ordre Cas 1, 2, 3 = liste affichée
    prompt_text = _build_rag_prompt_from_docs(question, hint, conversation_history, docs)
    print("[LLM PROMPT (RAG)]", "-" * 40)
    print(prompt_text)
    print("-" * 40)
    generator = _get_generator()
    gen_result = generator.run(prompt=prompt_text)
    replies = gen_result.get("replies", [])
    answer = replies[0] if replies else "Aucune réponse générée."
    if hasattr(answer, "content"):
        answer = answer.content

    # Détecter si l'assistant propose un détail → renvoyer pending_* pour le prochain « ok »
    pending_case_index = _parse_offer_detail_from_text(answer)
    if pending_case_index is not None and 1 <= pending_case_index <= len(suggested_case_ids):
        pending_uid = suggested_case_ids[pending_case_index - 1]
        return answer, sources, suggested_case_ids, full_contents, "expand_details", pending_uid, pending_case_index
    return answer, sources, suggested_case_ids, full_contents, None, None, None


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

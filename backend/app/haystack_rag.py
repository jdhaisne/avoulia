"""
RAG avec Haystack + Chroma, via Azure AI Foundry.
- Embeddings : Foundry (Azure OpenAI) -> stockage dans Chroma.
- Chat : modèle gpt-5-chat sur Foundry (avec ou sans RAG).
"""

import logging
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
from app.rag_constants import (
    CASE_EXTRA_FIELD_ALIASES,
    CASE_EXTRA_KEYS,
    CHOIX_Q1_TO_DOMAINE_CODE,
    CHROMA_DOMAIN_META_FILTER_FIELDS,
    DOMAINES_SANS_SECTEURS,
    DOMAINE_META_KEYS,
    INTENTIONS_PAR_DOMAINE,
    INTENTION_META_KEYS,
    Q1_DOMAINS_LIST,
    Q3_TRIGGERS_DISPLAY_LIMIT,
    SECTEURS_PAR_DOMAINE,
    TRIGGER_META_KEYS,
    WELCOME_MESSAGE,
)

logger = logging.getLogger(__name__)


def get_q15_choices(domaine_code: str) -> list[str] | None:
    """Retourne la liste des choix secteur pour Q1.5, ou None. Q1.5 est posée uniquement si le domaine est présent dans SECTEURS_PAR_DOMAINE."""
    if domaine_code not in SECTEURS_PAR_DOMAINE:
        return None
    secteurs = SECTEURS_PAR_DOMAINE[domaine_code]
    if not secteurs:
        return None
    return secteurs + ["Autre / Non spécifique"]


def _get_domaine_label(domaine_code: str) -> str:
    """Retourne le libellé du domaine pour l'affichage (Q1), à partir de CHOIX_Q1_TO_DOMAINE_CODE et Q1_DOMAINS_LIST."""
    for choix, code in CHOIX_Q1_TO_DOMAINE_CODE.items():
        if code == domaine_code:
            return Q1_DOMAINS_LIST[choix - 1]
    return domaine_code or "—"


def _get_trigger_from_meta(meta: dict) -> str:
    """Extrait la valeur trigger/situation depuis les meta."""
    for key in TRIGGER_META_KEYS:
        val = (meta.get(key) or "").strip()
        if val:
            return val
    return ""


def _doc_matches_domain(doc, label: str, domaine_code: str) -> bool:
    """True si les meta du document correspondent au domaine (label ou code)."""
    meta = getattr(doc, "meta", None) or {}
    label_lower = (label or "").strip().lower()
    code_lower = (domaine_code or "").strip().lower()
    for key in DOMAINE_META_KEYS:
        val = (meta.get(key) or "").strip()
        if not val:
            continue
        if val.lower() == label_lower or val.lower() == code_lower:
            return True
    return False


def _get_intention_from_meta(meta: dict) -> str:
    """Extrait la valeur intention depuis les meta (plusieurs noms de colonnes possibles)."""

    for key in INTENTION_META_KEYS:
        val = (meta.get(key) or "").strip()
        if val:
            return val
    return ""


def _meta_first_nonempty(meta: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        raw = meta.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return ""


def _case_extra_fields_from_meta(meta: dict | None) -> dict[str, str | None]:
    meta = meta or {}
    out: dict[str, str | None] = {}
    for canonical, aliases in CASE_EXTRA_FIELD_ALIASES.items():
        val = _meta_first_nonempty(meta, aliases)
        out[canonical] = val if val else None
    return out


def _case_extras_from_case_dict(case: dict) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key in CASE_EXTRA_KEYS:
        raw = case.get(key)
        if raw is None:
            out[key] = None
            continue
        s = str(raw).strip()
        out[key] = s if s else None
    return out


def _doc_to_case_dict(doc, index: int) -> dict:
    meta = getattr(doc, "meta", None) or {}
    extras = _case_extra_fields_from_meta(meta)
    return {
        "id": str(getattr(doc, "id", None) or index),
        "content": getattr(doc, "content", None) or "",
        **extras,
    }


def _format_case_extra_block(case: dict) -> str:
    """Bloc texte des champs structurés pour injection dans les prompts (détail / contexte modèle)."""
    labels = {
        "effort": "Niveau d'effort (effort)",
        "prerequis_donnees": "Prérequis données (prerequis_donnees)",
        "guardrails": "Guardrails",
        "questions_qualification": "Questions de qualification",
        "sensibilite_donnees": "Sensibilité des données (contexte pour le point de vigilance)",
    }
    lines: list[str] = []
    for key in CASE_EXTRA_KEYS:
        raw = case.get(key)
        if raw is None:
            continue
        val = str(raw).strip()
        if not val:
            continue
        lines.append(f"- {labels[key]} : {val}")
    if not lines:
        return ""
    return "Données structurées du cas (fichier source) :\n" + "\n".join(lines)


def _append_structured_case_fields_to_content(content: str, case: dict) -> str:
    block = _format_case_extra_block(case)
    if not block:
        return content
    return content.rstrip() + "\n\n" + block


def _build_metadata_or_filter(meta_keys: tuple[str, ...], values: list[str | None]) -> dict | None:
    """Construit un filtre OR multi-champs pour les metadata Chroma."""
    normalized_values: list[str] = []
    for value in values:
        candidate = (value or "").strip()
        if candidate and candidate not in normalized_values:
            normalized_values.append(candidate)

    if not normalized_values:
        return None

    conditions = [
        {"field": f"meta.{meta_key}", "operator": "==", "value": value}
        for meta_key in meta_keys
        for value in normalized_values
    ]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"operator": "OR", "conditions": conditions}


SECTEUR_META_KEYS = (
    "secteur",
    "Secteur",
    "sector",
    "Sector",
    "secteur_activite",
    "secteur_activité",
)
MULTI_SECTOR_VALUES = ("multi-sectoriel", "multisectoriel", "multi sectoriel")


def _normalize_metadata_value(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def _is_multisector_label(value: str) -> bool:
    normalized = _normalize_metadata_value(value)
    if not normalized:
        return False
    return any(token in normalized for token in MULTI_SECTOR_VALUES)


def _doc_matches_sector(doc, selected_sector: str, *, include_multisector: bool = False) -> bool:
    """True si le document correspond au secteur choisi (ou multi-sectoriel autorisé)."""
    if not selected_sector:
        return False
    expected = _normalize_metadata_value(selected_sector)
    meta = getattr(doc, "meta", None) or {}
    for key in SECTEUR_META_KEYS:
        raw = str(meta.get(key) or "").strip()
        if not raw:
            continue
        normalized = _normalize_metadata_value(raw)
        if normalized == expected:
            return True
        # Tolère des valeurs concaténées (ex: "BTP / Industrie").
        tokens = [_normalize_metadata_value(t) for t in re.split(r"[,;/|]", raw)]
        if expected in tokens:
            return True
        if include_multisector and _is_multisector_label(raw):
            return True
    return False


def _build_retrieval_filters(
    domaine_code: str | None = None,
    intention_code: str | None = None,
    selected_sector: str | None = None,
    include_multisector: bool = False,
) -> dict | None:
    """Construit les filtres metadata appliqués avant le retrieval vectoriel."""
    conditions: list[dict] = []

    domaine_label = _get_domaine_label(domaine_code) if domaine_code else None
    domain_filter = _build_metadata_or_filter(DOMAINE_META_KEYS, [domaine_label, domaine_code])
    if domain_filter:
        conditions.append(domain_filter)

    intention_label = (
        _get_intention_label_from_code(domaine_code, intention_code, secteur_choisi=selected_sector)
        if domaine_code and intention_code
        else None
    )
    intention_filter = _build_metadata_or_filter(INTENTION_META_KEYS, [intention_label])
    if intention_filter:
        conditions.append(intention_filter)

    sector_values: list[str | None] = [selected_sector]
    if include_multisector:
        sector_values.extend(MULTI_SECTOR_VALUES)
    sector_filter = _build_metadata_or_filter(SECTEUR_META_KEYS, sector_values)
    if sector_filter:
        conditions.append(sector_filter)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"operator": "AND", "conditions": conditions}


def _fetch_documents_for_domaine(domaine_code: str, *, top_k_fallback: int = 150) -> list:
    """
    Documents Chroma dont les métadonnées correspondent au domaine (libellé Q1 ou code interne).
    Stratégie : filter_documents sur plusieurs champs meta, puis fallback retrieval + filtre Python.
    """
    label = _get_domaine_label(domaine_code) if domaine_code else None
    docs: list = []
    if not domaine_code and not label:
        return docs
    try:
        store = get_document_store()
        for field in CHROMA_DOMAIN_META_FILTER_FIELDS:
            try:
                if label:
                    docs = store.filter_documents(
                        filters={"field": field, "operator": "==", "value": label}
                    )
                if not docs and domaine_code:
                    docs = store.filter_documents(
                        filters={"field": field, "operator": "==", "value": domaine_code}
                    )
                if docs:
                    break
            except Exception:
                continue
        if not docs and (label or domaine_code):
            query = label or domaine_code.replace("_", " ")
            try:
                all_candidates = _retrieve_docs(query, top_k=top_k_fallback)
                if all_candidates and isinstance(all_candidates[0], list):
                    all_candidates = [d for sub in all_candidates for d in sub]
            except Exception:
                all_candidates = []
            for d in all_candidates:
                if _doc_matches_domain(d, label, domaine_code):
                    docs.append(d)
    except Exception:
        pass
    return docs


def _get_intentions_from_store(domaine_code: str) -> list[str]:
    """
    Récupère les intentions distinctes depuis Chroma pour ce domaine.
    Essaie filter_documents avec plusieurs noms de champs meta, puis fallback par retrieval + filtre en Python.
    """
    docs = _fetch_documents_for_domaine(domaine_code)
    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        meta = getattr(d, "meta", None) or {}
        intention = _get_intention_from_meta(meta)
        if intention and intention not in seen:
            seen.add(intention)
            out.append(intention)
    return sorted(out)


def _get_triggers_from_store(domaine_code: str, intention: str | None = None) -> list[str]:
    """
    Récupère les triggers (exemples de situations) distincts depuis Chroma pour ce domaine,
    optionnellement filtrés par intention. Même logique que _get_intentions_from_store pour les docs.
    """
    docs = _fetch_documents_for_domaine(domaine_code)

    if intention:
        intention_norm = intention.strip().lower()
        filtered = []
        for d in docs:
            meta = getattr(d, "meta", None) or {}
            doc_int = _get_intention_from_meta(meta)
            if doc_int and doc_int.strip().lower() == intention_norm:
                filtered.append(d)
        docs = filtered if filtered else docs

    seen: set[str] = set()
    out: list[str] = []
    for d in docs:
        meta = getattr(d, "meta", None) or {}
        trigger = _get_trigger_from_meta(meta)
        if trigger and trigger not in seen:
            seen.add(trigger)
            out.append(trigger)
    return sorted(out)


def _get_doc_sector_score(doc, secteur_choisi: str | None) -> int:
    """Score secteur pour un doc: 3=secteur exact, 2=multi-sectoriel, 1=autre."""
    if not secteur_choisi:
        return 1
    selected_norm = _normalize_metadata_value(secteur_choisi)
    for key in SECTEUR_META_KEYS:
        raw = str((getattr(doc, "meta", None) or {}).get(key) or "").strip()
        if not raw:
            continue
        normalized = _normalize_metadata_value(raw)
        if normalized == selected_norm:
            return 3
        tokens = [_normalize_metadata_value(t) for t in re.split(r"[,;/|]", raw)]
        if selected_norm in tokens:
            return 3
        if _is_multisector_label(raw):
            return 2
    return 1


def build_pool(
    domaine_code: str,
    intention: str | None = None,
    secteur_choisi: str | None = None,
    top_k: int | None = None,
) -> list[str]:
    """
    Construit le pool Q3 trié:
    - score 3 si secteur doc == secteur utilisateur
    - score 2 si secteur doc == Multi-sectoriel
    - score 1 sinon
    Restitue les triggers triés par score DESC (puis alpha), limités à top_k si fourni.
    """
    docs = _fetch_documents_for_domaine(domaine_code)

    if intention:
        intention_norm = _normalize_metadata_value(intention)
        docs = [
            d
            for d in docs
            if _normalize_metadata_value(_get_intention_from_meta(getattr(d, "meta", None) or {}))
            == intention_norm
        ] or docs

    # Conserver le meilleur score par trigger.
    trigger_scores: dict[str, int] = {}
    for doc in docs:
        trigger = _get_trigger_from_meta(getattr(doc, "meta", None) or {})
        if not trigger:
            continue
        score = _get_doc_sector_score(doc, secteur_choisi)
        prev = trigger_scores.get(trigger)
        if prev is None or score > prev:
            trigger_scores[trigger] = score

    ranked = sorted(trigger_scores.items(), key=lambda item: (-item[1], item[0].lower()))
    triggers = [trigger for trigger, _ in ranked]
    if top_k is not None and top_k > 0:
        return triggers[:top_k]
    return triggers


def get_q2_choices(
    domaine_code: str,
    secteur_choisi: str | None = None,
) -> list[str] | dict[str, str | bool]:
    """
    Pseudo-code :
    intentions = set(c[intention] for c in base if c[domaine] == domaine)
    if secteur_choisi and secteur_choisi != 'Autre':
        intentions = [i for i in intentions
            if any(c[secteur] in (secteur_choisi, 'Multi-sectoriel')
                for c in base
                if c[domaine] == domaine and c[intention] == i)]
    if not intentions:
        return {'fallback': True, 'message': '...'}
    """
    def _doc_has_multisector_case(candidate_doc) -> bool:
        """True si le doc porte au moins un champ secteur contenant 'Multi-sectoriel'."""
        meta = getattr(candidate_doc, "meta", None) or {}
        for key in SECTEUR_META_KEYS:
            raw = str(meta.get(key) or "").strip()
            if raw and _is_multisector_label(raw):
                return True
        return False

    docs = _fetch_documents_for_domaine(domaine_code)

    intentions_set: set[str] = set()
    for doc in docs:
        intention = _get_intention_from_meta(getattr(doc, "meta", None) or {})
        if intention:
            intentions_set.add(intention)

    if not intentions_set:
        return {"fallback": True, "message": "Aucune intention disponible. Explorer un autre domaine ?"}

    # Pas de filtre si secteur Q1.5 non fourni.
    if not secteur_choisi:
        return sorted(intentions_set)

    secteur_norm = _normalize_metadata_value(secteur_choisi)
    filter_for_autre = secteur_norm.startswith("autre")

    filtered: list[str] = []
    for intention in sorted(intentions_set):
        intention_norm = _normalize_metadata_value(intention)

        has_sector_case = any(
            _normalize_metadata_value(_get_intention_from_meta(getattr(candidate_doc, "meta", None) or {})) == intention_norm
            and (
                _doc_has_multisector_case(candidate_doc)
                if filter_for_autre
                else _doc_matches_sector(candidate_doc, secteur_choisi, include_multisector=True)
            )
            for candidate_doc in docs
        )
        if has_sector_case:
            filtered.append(intention)

    if not filtered:
        return {"fallback": True, "message": "Aucune intention disponible. Explorer un autre domaine ?"}

    return filtered


def _get_q2_choices_list(domaine_code: str, secteur_choisi: str | None = None) -> list[str]:
    """Retourne toujours une liste de choix Q2, même si get_q2_choices est en mode fallback."""
    result = get_q2_choices(domaine_code, secteur_choisi=secteur_choisi)
    if isinstance(result, list):
        return result
    return []


def get_q3_triggers(
    domaine_code: str,
    intention: str | None = None,
    secteur_choisi: str | None = None,
    top_k: int | None = None,
) -> list[str]:
    """Retourne la liste des triggers (exemples de situations) pour Q3 pour ce domaine, optionnellement pour cette intention."""
    return build_pool(domaine_code, intention, secteur_choisi=secteur_choisi, top_k=top_k)


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
            azure_endpoint=s.azure_endpoint_normalized_chat,
            api_key=_secret(s.azure_openai_api_key_chat),
            api_version=s.azure_openai_api_version_chat,
            azure_deployment=s.azure_chat_deployment,
        )

    from haystack.components.generators import OpenAIGenerator
    return OpenAIGenerator(
        api_key=_secret(s.openai_api_key),
        model=s.openai_chat_model,
    )

# Prompt unique RAG : parcours guidé (Q1 → Q1.5 conditionnel → Q2 → Q2.5 conditionnel → Q3 → Phase 2).
# Les listes dynamiques (secteurs Q1.5, intentions Q2, etc.) sont injectées via le hint.
RAG_PROMPT = """
Tu es un agent conversationnel spécialisé dans l'identification
de cas d'usage d'IA générative pour dirigeants et responsables
de PME françaises.

Tu fonctionnes exclusivement selon un parcours guidé structuré.
L'utilisateur ne commence jamais en langage libre.
Tu poses des questions fermées successives.
Tu n'interprètes jamais librement les réponses.
Tu ne modifies jamais le domaine ou l'intention sans validation
explicite.

SÉQUENCE OBLIGATOIRE — NE JAMAIS DÉROGER :
Avant de présenter des cas, tu DOIS avoir reçu
une réponse à CHAQUE étape dans cet ordre :
ÉTAPE 1 — Q1 (domaine) → obligatoire
ÉTAPE 2 — Q1.5 (secteur) → obligatoire si secteurs
ÉTAPE 3 — Q2 (intention) → obligatoire, FORMAT LISTE
ÉTAPE 4 — Q2.5 → si déclenché
ÉTAPE 5 — Q3 (problème) → obligatoire
Tu ne présentes JAMAIS de cas avant 5 étapes complètes.

-------------------------------------
INTRODUCTION
-------------------------------------

Tu commences toujours par afficher EXACTEMENT le message suivant :

"Bonjour, je vais vous aider à identifier des cas d'usage concrets
de l'IA adaptés à votre organisation. Pour commencer, je vais vous
poser quelques questions simples afin de cibler précisément votre
priorité."

-------------------------------------
PHASE 1 — QUESTIONNEMENT GUIDÉ
-------------------------------------

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
- L'utilisateur doit choisir un seul domaine.
- Tu n'expliques pas les domaines.
- Si la réponse ne correspond pas exactement à un choix proposé,
  tu redemandes de choisir parmi la liste.

-------------------------------------
Q1.5 — Secteur (conditionnel)
-------------------------------------

Cette question est posée SI ET SEULEMENT SI 
le domaine est dans SECTEURS_PAR_DOMAINE et que la liste de secteurs est non vide.
SECTEURS_PAR_DOMAINE est :{
"ressources_humaines": [
        "BTP", "Industrie", "Services & artisanat",
        "Hôtellerie & tourisme",
    ],
    "organisation_coordination": [
        "Commerce & retail", "Industrie",
        "Santé & médico-social", "Agroalimentaire",
        "Transport & logistique", "Restauration",
        "Cabinet & conseil",
    ],
    "conformite_risque": [
        "BTP", "Industrie", "Agroalimentaire",
        "Santé & médico-social", "Transport & logistique",
        "Cabinet & conseil",
    ],
    "finance_pilotage": [
        "BTP", "Commerce & retail", "Industrie",
        "Santé & médico-social", "Agroalimentaire",
        "Cabinet & conseil", "Restauration",
        "Services & artisanat", "Hôtellerie & tourisme",
        "Énergie & télécoms",
    ],
    "production": [
        "Industrie", "Agroalimentaire", "BTP",
        "Restauration", "Services & artisanat",
        "Cabinet & conseil", "Transport & logistique",
    ],
    "relation_client": [
        "Commerce & retail", "Hôtellerie & tourisme",
        "Industrie", "BTP", "Santé & médico-social",
        "Agroalimentaire", "Restauration",
        "Cabinet & conseil", "Services & artisanat",
    ],
    "marketing_visibilite": [
        "Commerce & retail", "Restauration",
        "Hôtellerie & tourisme", "Industrie",
        "Transport & logistique", "Services & artisanat",
    ],
    "activites_terrain": [
        "BTP", "Services & artisanat",
        "Santé & médico-social", "Commerce & retail",
        "Industrie", "Transport & logistique",
        "Agroalimentaire",
    ],
    "ventes_developpement": [
        "Commerce & retail", "Industrie", "BTP",
        "Restauration", "Services & artisanat",
    ],
    "logistique_stocks": [
        "Transport & logistique", "Industrie",
        "Commerce & retail", "Agroalimentaire",
        "Restauration",
    ],
    "achats_fournisseurs": [
        "Industrie", "BTP", "Commerce & retail",
        "Transport & logistique", "Restauration",
    ]
}

Sinon tu passes directement à Q2.

Si déclenchée, tu poses EXACTEMENT :

"Pour mieux cibler mes recommandations, pouvez-vous me
dire dans quel secteur vous opérez ? Répondez avec le numéro du choix. (optionnel)"
et tu fournis la liste des secteurs possibles numérotée (1..N) pour le domaine donné.

Règle stricte :
- si l'utilisateur répond avec un numéro hors plage (ex: 9 alors qu'il n'y a que 5 choix),
  cette réponse est invalide.
- dans ce cas, tu ne passes JAMAIS à Q2 : tu répètes Q1.5 et redonnes la liste numérotée.

-------------------------------------
Q2 — Objectif principal
-------------------------------------

Tu poses Q2 UNIQUEMENT sous cette forme exacte :
« Quel est votre objectif principal dans ce domaine ?
1. [intention_1]
2. [intention_2] ... »

Tu n’utilises JAMAIS une formulation ouverte pour Q2.
Si aucune intention n’est disponible, tu réponds :
« Je n’ai pas pu charger les objectifs. Reformulez. »

Règles :
- Tu proposes uniquement les intentions correspondant au domaine
  sélectionné.
- Tu n'inventes jamais d'intention hors domaine.
- Tu ne reformules pas les intentions.
- Si la réponse ne correspond pas à la liste fournie, tu
  redemandes un choix valide.
- Si l'utilisateur répond avec un numéro hors plage, tu ne passes
  pas à Q3 : tu répètes Q2 et redonnes la liste numérotée.

-------------------------------------
Q2.5 — Précision du sujet (conditionnel)
-------------------------------------

Cette question est posée SI ET SEULEMENT SI le backend
détecte 4 micro-thèmes ou plus dans le pool filtré après Q2.

Si déclenchée, tu poses EXACTEMENT :

"Pour affiner, quel aspect vous concerne le plus ?"

Règles :
- Tu proposes UNIQUEMENT les micro-thèmes fournis par le backend.
- Tu n'inventes jamais de micro-thème.
- Tu n'affiches jamais plus de 6 choix.
- Si le backend ne déclenche pas Q2.5, tu passes directement
  à Q3 sans mentionner les micro-thèmes.

-------------------------------------
Q3 — Problème concret (texte libre guidé)
-------------------------------------
  exemples fournis par le backend :
{{ q3_triggers_affichage }}

Si le backend fournit une liste d'exemples de situations
(triggers), tu poses EXACTEMENT :

"Pouvez-vous décrire le problème concret que vous
rencontrez actuellement ?"

"Voici quelques situations fréquentes dans votre cas
pour vous aider à formuler :"

Tu affiches les exemples fournis par le backend reformulés en phrase courtes et en français sous
forme de liste simple (tirets), par exemple :
  - marge en baisse sans explication claire
  - stocks d'invendus en fin de saison
  - prix fixés à l'intuition
  - concurrence agressive sur les prix

Puis tu ajoutes :
"Décrivez votre situation en une ou deux phrases."

Règles :
- L'utilisateur répond TOUJOURS en texte libre.
- Les exemples sont une aide à la formulation, pas des
  choix à sélectionner.
- Les exemples sont limités à 6 exemples.
- Tu ne proposes AUCUN mécanisme de coche ou de clic.
- Tu ne reformules jamais les exemples fournis.
- Tu ne changes jamais le domaine, l'intention ou le
  micro-thème en fonction de la réponse Q3.
- Q3 sert uniquement à contextualiser et à alimenter
  le retrieval vectoriel.
- Si le backend ne fournit pas d'exemples (pool trop
  petit), tu poses la question sans exemples :
  "Quel problème concret rencontrez-vous actuellement ?"




-------------------------------------
PHASE 1 BIS — FALLBACK INCOHÉRENCE DOMAINE
-------------------------------------

Si le backend fournit un domaine suggéré en cas d'incohérence
potentielle, tu affiches EXACTEMENT :

"Votre situation semble également concerner le domaine suivant :

[Nom du domaine suggéré]

Souhaitez-vous explorer également ce domaine ?"

Règles :
- Tu ne changes jamais automatiquement de domaine.
- Tu attends la décision explicite de l'utilisateur.

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
PHASE 3 — CLOTURE
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
- continue a détailler les cas si demandé par l'utilisateur
- sinon, tu termines la conversation
- tu ne proposes jamais de nouveaux cas
- tu ne proposes jamais de nouveaux micro-thèmes
- tu ne proposes jamais de nouveaux secteurs
- tu ne proposes jamais de nouveaux domaines
- tu ne proposes jamais de nouveaux intentions
- tu ne proposes jamais de nouveaux micro-thèmes

FORMAT OBLIGATOIRE POUR CHAQUE CAS — NIVEAU 1 (APERÇU)
(présentation initiale, 3 à 5 cas)
 [Numéro]. Nom du cas
Pourquoi c’est pertinent pour vous :
(1 à 2 phrases contextualisées par rapport au problème Q3.)
Ce que cela permet concrètement :
(Description claire et opérationnelle, sans jargon technique.)
---
Après les 3–5 cas, tu ajoutes EXACTEMENT :
« Souhaitez-vous approfondir l’un de ces cas ?
Indiquez son numéro pour obtenir le détail complet. »
Règles Niveau 1 :
- Tu ne montres PAS l’effort, les prérequis, la première
étape, les guardrails, ni les questions de qualification.
- Tu gardes chaque cas court (5–6 lignes max).

- L’objectif est de permettre un scan rapide.

FORMAT APPROFONDI — NIVEAU 2 (SUR DEMANDE)
 Nom du cas
Pourquoi c’est pertinent pour vous :
(1 à 2 phrases contextualisées par rapport au problème Q3.)
Ce que cela permet concrètement :
(Description claire et opérationnelle, sans jargon technique.)
Niveau d’effort : [Faible / Moyen / Élevé]
Ce qu’il vous faut pour démarrer :
(Reformulation claire de prerequis_donnees.)
Première étape simple :
(Reformulation claire de premiere_action_48h.)
⚠️ Point de vigilance :
(Reformulation accessible de guardrails. Omis si vide.)
 Auto-diagnostic rapide
Avant de vous lancer, posez-vous ces questions :
• [question_qualification_1 reformulée]
• [question_qualification_2 reformulée]
• [question_qualification_3 reformulée]
Si vous répondez « non » à au moins 2 de ces questions,
ce cas est particulièrement pertinent pour vous.
Règles Niveau 2 :
- Tu affiches TOUTES les questions_qualification du cas.
- Tu reformules pour un dirigeant non technique.
- Tu n’inventes jamais de question hors du backend.
- Phrase « au moins 2 » : seulement si ≥ 3 questions.
- Si 2 questions : « Ces questions vous aideront à
évaluer si ce cas répond à votre situation. »

-------------------------------------
RÈGLES STRICTES
-------------------------------------

Tu ne :
- promets jamais de ROI chiffré
- recommandes jamais un outil spécifique
- mentionnes jamais le système interne
- expliques jamais le mécanisme de filtrage
- mentionnes jamais les exemples de situations comme provenant
  d'une base
- modifies jamais la sélection fournie
- ajoutes jamais un sixième cas
- inventes jamais un cas
- interprètes jamais la taxonomie
- inventes jamais de question d’auto-diagnostic hors de celles fournies par le backend

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
{% if user_choices_summary %}
Résumé des choix utilisateur :
{{ user_choices_summary }}

{% endif %}
{% if identified_cases_summary %}
Cas identifiés (3 à 5) :
{{ identified_cases_summary }}

{% endif %}
{% if cases_extra_context %}
{{ cases_extra_context }}

{% endif %}
{% if detail_prompt_context %}
Consignes de détail à appliquer si l'utilisateur demande de détailler un cas :
{{ detail_prompt_context }}

{% endif %}
{% if conversation_history %}
Historique récent de la conversation (utilise-le pour garder le contexte) :
{{ conversation_history }}

{% endif %}


RÈGLE : Ne propose et ne détaille que les cas listés ci-dessus (ordre 1 à {{ documents|length }}). Si l'utilisateur demande « le point 2 » ou « le 2ème », c'est toujours le 2e cas de ta liste ci-dessus. Ne confonds jamais les numéros. Tes numéros visibles dans la réponse (1., 2., 3., etc.) doivent rester alignés avec cet ordre.

Demande actuelle de l'utilisateur : {{ query }}

Réponse (réponse directe OU 1 à 2 questions de clarification, en tenant compte de l'historique):"""


# Prompt pour détailler un seul cas (liste déjà connue, pas de re-retrieval).
DETAIL_PROMPT = """Tu es un assistant expert qui détaille un cas d'usage de l'IA générative pour une PME.
Tu t'adresses à des dirigeants ou responsables non techniques. Tu réponds toujours en français.
Tu ne cites jamais de noms de produits, d'éditeurs ou de technologies.

L'utilisateur a demandé à détailler un cas précis que tu avais proposé. Tu dois décrire UNIQUEMENT ce cas-ci, en t'appuyant exclusivement sur les données fournies ci-dessous. Ne décris aucun autre cas. Extrais et présente les informations suivantes lorsqu'elles sont présentes dans les données :

- l'impact concret (primary_value),
- le gain attendu (expected_gain_proxy),
- le mode, l'effort, la complexité et le délai,
- la vigilance données : croise guardrails / points de vigilance avec la sensibilité des données (sensibilite_donnees) lorsque fournie,
- un premier pas faisable en 48h,
- les prérequis simples.
- les questions de qualification listées (questions_qualification), sans en inventer d'autres
- après le détail, proposer uniquement de revenir à la liste ou de terminer, sans inventer de nouveaux services

Données complètes du cas à détailler :
{{ case_content }}

Réponse (détail de ce cas uniquement, d'après les données ci-dessus) :"""


def _build_rag_prompt_from_docs(
    query: str,
    hint: str,
    conversation_history: str,
    documents: list,
    history: list[dict],
    secteur_choices_affichage: str = "",
    intention_choices_affichage: str = "",
    q3_triggers_affichage: str = "",
    selected_domain_code: str | None = None,
    selected_sector: str | None = None,
    selected_intention: str | None = None,
) -> str:
    """
    Construit le prompt RAG avec le prompt unique. Les listes dynamiques (secteurs Q1.5,
    intentions Q2, triggers Q3) sont injectées dans le hint selon l'étape du parcours.
    """
    docs = documents or []
    history_list = history or []
    phase_hint = hint or ""

    if not docs:
        phase_hint = (phase_hint + "\n\n" if phase_hint else "") + (
            "Aucun extrait de cas fourni pour l'instant : tu es en phase questionnement guidé. "
            "Pose UNIQUEMENT la prochaine question selon l'étape (Q1, Q1.5 si liste fournie, Q2, Q3). "
            "Ne présente aucun cas, ne propose aucune liste de cas."
        )

    # Injecter les listes fournies par le backend pour Q1.5, Q2 et Q3
    domaine_code = selected_domain_code or _get_domaine_code_from_history(history_list)
    q1_5_choices = get_q15_choices(domaine_code) if domaine_code else None

    if domaine_code and q1_5_choices and not selected_sector and secteur_choices_affichage:
        phase_hint = (phase_hint + "\n\n" if phase_hint else "") + (
            "Liste des secteurs à proposer par le backend pour Q1.5 (affiche cette liste telle quelle) :\n"
            + secteur_choices_affichage
        )
    if domaine_code and intention_choices_affichage and not selected_intention:
        # Q2 : sans dépendance au nombre de messages
        # - domaines sans Q1.5: directement après Q1
        # - domaines avec Q1.5: seulement après secteur validé
        if not q1_5_choices or selected_sector:
            phase_hint = (phase_hint + "\n\n" if phase_hint else "") + (
                "Liste des intentions à proposer par le backend pour Q2 (affiche cette liste telle quelle) :\n"
                + intention_choices_affichage
            )
    # Q3 triggers : seulement quand domaine + intention sont validés
    if domaine_code and selected_intention and q3_triggers_affichage:
        phase_hint = (phase_hint + "\n\n" if phase_hint else "") + (
            "exemples fournis par le backend :\n"
            + q3_triggers_affichage
        )

    domain_label = _get_domaine_label(domaine_code) if domaine_code else "non sélectionné"
    intention_label = _get_intention_label_from_code(domaine_code, selected_intention) if domaine_code else None
    user_choices_summary = "\n".join(
        [
            f"- Domaine: {domain_label}",
            f"- Secteur: {selected_sector or 'non sélectionné'}",
            f"- Intention: {intention_label or 'non sélectionnée'}",
        ]
    )
    identified_cases_summary = ""
    if docs:
        selected_cases = docs[:5]
        if len(selected_cases) < 3 and len(docs) >= 3:
            selected_cases = docs[:3]
        lines = []
        for i, d in enumerate(selected_cases, start=1):
            content = (getattr(d, "content", "") or "").strip()
            short = content[:240] + "..." if len(content) > 240 else content
            lines.append(f"{i}. {short}")
        identified_cases_summary = "\n".join(lines)
    cases_extra_context = ""
    if docs:
        selected_for_context = docs[:5]
        if len(selected_for_context) < 3 and len(docs) >= 3:
            selected_for_context = docs[:3]
        blocks: list[str] = []
        for i, d in enumerate(selected_for_context, start=1):
            ex = _case_extra_fields_from_meta(getattr(d, "meta", None) or {})
            blk = _format_case_extra_block(ex)
            if blk:
                blocks.append(f"Cas {i} :\n{blk}")
        if blocks:
            cases_extra_context = (
                "Champs structurés par cas (fichier source ; niveau 1 : ne pas les afficher dans l'aperçu ; "
                "niveau 2 : t'en servir pour effort, prérequis, guardrails, questions de qualification, sensibilité données) :\n\n"
                + "\n\n".join(blocks)
            )
    detail_prompt_context = ""
    if _is_detail_request(query) or _has_explicit_point_number(query):
        detail_prompt_context = DETAIL_PROMPT

    template = Template(RAG_PROMPT)
    return template.render(
        query=query or "",
        hint=phase_hint,
        user_choices_summary=user_choices_summary,
        identified_cases_summary=identified_cases_summary,
        cases_extra_context=cases_extra_context,
        detail_prompt_context=detail_prompt_context,
        conversation_history=conversation_history or "",
        documents=docs,
        q3_triggers_affichage=q3_triggers_affichage or "",
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


def _extract_explicit_point_number(message: str) -> int | None:
    """
    Extrait un numéro explicite de cas/point (1-based) depuis le message.
    Retourne None si aucun numéro explicite n'est présent.
    """
    if not message or len(message.strip()) < 2:
        return None
    msg = message.strip().lower()

    m = re.search(r"\b(?:point|cas|numéro|numero)\s*([1-9]\d?)\b", msg, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"\ble\s+([1-9]\d?)\s*(?:er|ème|e|eme)?\b", msg, re.IGNORECASE)
    if m:
        return int(m.group(1))

    ord_map = {
        "premier": 1, "1er": 1, "1ère": 1, "1ere": 1,
        "deuxième": 2, "2ème": 2, "2eme": 2, "2e": 2,
        "troisième": 3, "3ème": 3, "3eme": 3, "3e": 3,
        "quatrième": 4, "4ème": 4, "4eme": 4, "4e": 4,
        "cinquième": 5, "5ème": 5, "5eme": 5, "5e": 5,
    }
    for token, idx in ord_map.items():
        if re.search(r"\b" + re.escape(token) + r"\b", msg):
            return idx
    return None


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


def _retrieve_docs(query: str, top_k: int | None = None, filters: dict | None = None) -> list:
    """Lance une recherche RAG et retourne la liste de documents (sans génération)."""
    s = get_settings()
    k = top_k or s.top_k_retrieve
    store = get_document_store()
    embedder = _get_text_embedder()
    retriever = ChromaEmbeddingRetriever(document_store=store, filters=filters, top_k=k)
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

    explicit = _extract_explicit_point_number(message)
    if explicit is not None:
        idx = explicit - 1
        if 0 <= idx < n:
            return idx
        return None

    # Un seul cas proposé : « détaille » sans numéro = ce cas-là
    if n == 1:
        return 0

    # Résolution par numéro : uniquement 1 à 5 (Cas 1 à 5), avec regex pour éviter faux positifs
    # "point N" ou "numéro N" ou "le N" avec N = 1..5
    point_num = re.search(r"\bpoint\s*([1-5])\b", msg, re.IGNORECASE)
    if point_num:
        idx = int(point_num.group(1)) - 1
        if idx < n:
            return idx
        return None
    num_match = re.search(r"\b(?:le|numero|numéro)\s*([1-5])\s*(?:er|ème|e|eme)?\b", msg, re.IGNORECASE)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if idx < n:
            return idx
        return None
    # Ordinals en mots (sans " 1", " 2" etc. qui matchent dans "le 10")
    ordinals = {
        "premier": 0, "1er": 0, "1ère": 0, "1ere": 0,
        "deuxième": 1, "2ème": 1, "2eme": 1, "2e": 1,
        "troisième": 2, "3ème": 2, "3eme": 2, "3e": 2,
        "quatrième": 3, "4ème": 3, "4eme": 3, "4e": 3,
        "cinquième": 4, "5ème": 4, "5eme": 4, "5e": 4,
    }
    for phrase, idx in ordinals.items():
        if phrase in msg and re.search(r"\b" + re.escape(phrase) + r"\b", msg):
            if idx < n:
                return idx
            return None

    # Résolution par thème : seulement si un seul cas se détache (pas d'égalité)
    msg_words = set(re.findall(r"\w{3,}", msg)) - {
        "détaille", "detailler", "detail", "plus", "info", "savoir",
        "premier", "deuxieme", "trois", "quatre", "cinq", "point", "numero",
        "lequel", "celui", "celle", "sur", "cas", "usage",
    }
    if not msg_words:
        return None
    best_idx = None
    best_score = 0
    second_best_score = 0
    for i, item in enumerate(last_suggested_cases):
        content = (item.get("content") or "").lower()
        score = sum(1 for w in msg_words if w in content)
        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_idx = i
        elif score > second_best_score:
            second_best_score = score
    # Éviter de détailler le mauvais cas : ex æquo ou score trop faible = None
    if best_score == 0 or best_score == second_best_score:
        return None
    return best_idx


def build_rag_retrieval_only_pipeline(filters: dict | None = None):
    """Pipeline retrieval seul : embedder -> retriever. Pour construire le prompt nous-mêmes depuis les mêmes docs."""
    s = get_settings()
    store = get_document_store()
    embedder = _get_text_embedder()
    retriever = ChromaEmbeddingRetriever(document_store=store, filters=filters, top_k=s.top_k_retrieve)
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


def _build_detail_input_with_recap(
    case_index_1based: int,
    cases: list[dict],
    base_content: str,
) -> str:
    """
    Construit un texte pour le détail qui rappelle d'abord la numérotation des cas
    tels qu'ils ont été présentés (Cas 1, Cas 2, etc.), puis précise quel cas
    doit être détaillé, puis insère le contenu complet du cas ciblé.

    On ne ré-affiche PAS le texte des autres cas ici pour éviter toute confusion
    avec la formulation visible par l'utilisateur (qui peut être différente du
    contenu brut en base).
    """
    parts: list[str] = []
    parts.append("Rappel : les cas ont été présentés à l'utilisateur sous la forme d'une liste numérotée (Cas 1, Cas 2, Cas 3, ...), dans cet ordre exact.")
    parts.append("Ne réinterprète pas la liste, considère uniquement que :")
    parts.append(f"- Cas 1 est le premier cas affiché, Cas 2 le deuxième, etc.")
    parts.append(f"Le cas à détailler est : Cas {case_index_1based}.")
    parts.append("")
    parts.append(base_content)
    return "\n".join(parts)


def _run_detail_pipeline(case_content: str) -> str:
    """Génère une réponse de détail pour un seul cas (pas de retrieval)."""
    logger.debug(
        "prompt_detail case_content_len=%s preview=%r",
        len(case_content or ""),
        (case_content or "")[:500],
    )
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


def _parse_domaine_from_message(content: str | int | None) -> str | None:
    """
    Extrait un code domaine depuis un message utilisateur (réponse Q1).
    Utilise CHOIX_Q1_TO_DOMAINE_CODE : nombre 1-14 (entier ou dans le texte) ou libellé Q1_DOMAINS_LIST.
    """
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None
    text_norm = " ".join(text.lower().split())
    # 0) Chaîne = un seul entier 1-14 (ex. "3" ou front envoie 3)
    try:
        n = int(text)
        if 1 <= n <= 14:
            return CHOIX_Q1_TO_DOMAINE_CODE.get(n)
    except (ValueError, TypeError):
        pass
    # 1) Nombre 1-14 en début (ex. "3", "3.", " 12 ")
    num_match = re.match(r"^\s*(\d{1,2})\s*([\.\)\s,]|$)", text)
    if num_match:
        try:
            choix = int(num_match.group(1))
            if 1 <= choix <= 14:
                return CHOIX_Q1_TO_DOMAINE_CODE.get(choix)
        except (ValueError, TypeError):
            pass
    # 2) Nombre 1-14 ailleurs dans le message (ex. "je choisis 3")
    any_num = re.search(r"\b(1[0-4]|[1-9])\b", text)
    if any_num:
        try:
            choix = int(any_num.group(1))
            if 1 <= choix <= 14:
                return CHOIX_Q1_TO_DOMAINE_CODE.get(choix)
        except (ValueError, TypeError):
            pass
    # 3) Libellé Q1 (Q1_DOMAINS_LIST)
    text_lower = text.lower()
    for choix in range(1, 15):
        label = Q1_DOMAINS_LIST[choix - 1]
        if not label:
            continue
        label_norm = " ".join(label.lower().split())
        if text_lower == label.lower() or label.lower() in text_lower:
            return CHOIX_Q1_TO_DOMAINE_CODE.get(choix)
        # Tolérance pour saisies partielles (ex. "btp" -> "Construction ... BTP")
        # On évite les très petites chaînes pour limiter les faux positifs.
        if len(text_norm) >= 3 and text_norm in label_norm:
            return CHOIX_Q1_TO_DOMAINE_CODE.get(choix)
    return None


def _get_domaine_code_from_history(history: list[dict]) -> str | None:
    """
    Retourne le code domaine choisi par l'utilisateur (réponse Q1).
    Utilise la DERNIÈRE réponse utilisateur qui indique un domaine (nombre 1-14 ou libellé),
    pour prendre en compte une correction ou un clic sur un libellé (ex. "Ressources humaines & recrutement").
    """
    last_domaine = None
    for m in history:
        if (m.get("role") or "").strip().lower() != "user":
            continue
        raw = m.get("content")
        content = str(raw).strip() if raw is not None else ""
        code = _parse_domaine_from_message(content)
        if code:
            last_domaine = code
    return last_domaine


def _parse_sector_from_message(text: str, choices: list[str]) -> str | None:
    """Si text correspond à un choix secteur (numéro 1..N ou libellé), retourne le libellé du secteur, sinon None."""
    if not choices:
        return None
    text = (text or "").strip()
    if not text:
        return None
    # Numéro 1..N
    try:
        n = int(text)
        if 1 <= n <= len(choices):
            return choices[n - 1]
    except (ValueError, TypeError):
        pass
    num_match = re.match(r"^\s*(\d+)\s*([\.\)\s,]|$)", text)
    if num_match:
        try:
            n = int(num_match.group(1))
            if 1 <= n <= len(choices):
                return choices[n - 1]
        except (ValueError, TypeError):
            pass
    # Libellé du secteur : égalité ou contenu (insensible à la casse, espaces normalisés)
    text_norm = " ".join(text.lower().split())
    for s in choices:
        if not s:
            continue
        s_norm = " ".join(s.lower().split())
        if text_norm == s_norm or s_norm in text_norm or text_norm in s_norm:
            return s
    return None


def _parse_intention_from_message(text: str, choices: list[str]) -> str | None:
    """Si text correspond à un choix d'intention (numéro 1..N ou libellé), retourne le libellé, sinon None."""
    if not choices or not (text or "").strip():
        return None
    text = (text or "").strip()
    try:
        n = int(text)
        if 1 <= n <= len(choices):
            return choices[n - 1]
    except (ValueError, TypeError):
        pass
    num_match = re.match(r"^\s*(\d+)\s*([\.\)\s,]|$)", text)
    if num_match:
        try:
            n = int(num_match.group(1))
            if 1 <= n <= len(choices):
                return choices[n - 1]
        except (ValueError, TypeError):
            pass
    text_norm = " ".join(text.lower().split())
    for c in choices:
        if not c:
            continue
        c_norm = " ".join(c.lower().split())
        if text_norm == c_norm or c_norm in text_norm or text_norm in c_norm:
            return c
    return None


def _parse_intention_code_from_message(text: str, choices: list[str]) -> str | None:
    """Retourne le code d'intention (index 1..N en string) depuis une réponse Q2, sinon None."""
    parsed = _parse_intention_from_message(text, choices)
    if not parsed:
        return None
    try:
        return str(choices.index(parsed) + 1)
    except ValueError:
        return None


def _get_intention_label_from_code(
    domaine_code: str, intention_code: str | None, secteur_choisi: str | None = None
) -> str | None:
    """Traduit un code d'intention (1..N) en libellé Q2 pour un domaine donné."""
    if not domaine_code or not intention_code:
        return None
    choices = _get_q2_choices_list(domaine_code, secteur_choisi=secteur_choisi)
    if not choices:
        return None
    try:
        idx = int(str(intention_code).strip())
    except (ValueError, TypeError):
        return None
    if 1 <= idx <= len(choices):
        return choices[idx - 1]
    return None


def _resolve_selection_state(
    question: str,
    selected_domain_code: str | None,
    selected_sector: str | None,
    selected_intention: str | None,
) -> tuple[str | None, str | None, str | None]:
    """
    Met à jour domaine/secteur/intention à partir du message courant et de l'état client.
    Règles:
    - changement de domaine -> reset secteur + intention
    - changement de secteur -> reset intention
    - intention stockée en code (string 1..N)
    """
    current_domain = selected_domain_code
    current_sector = selected_sector
    current_intention = selected_intention

    question_text = (question or "").strip()
    parsed_domain = _parse_domaine_from_message(question)
    # Eviter de changer de domaine sur un nombre ambigu (ex. "2" en Q2).
    # On accepte le changement si aucun domaine n'est encore sélectionné, ou si le message
    # est explicite (non-numérique) et correspond à un domaine.
    is_plain_integer = bool(re.fullmatch(r"\d+", question_text))
    if parsed_domain and parsed_domain != current_domain:
        if current_domain is None or not is_plain_integer:
            return parsed_domain, None, None

    if not current_domain:
        return current_domain, current_sector, current_intention

    sector_choices = get_q15_choices(current_domain)
    # Le secteur ne doit être détecté que tant qu'il n'est pas déjà choisi.
    if sector_choices and not current_sector:
        parsed_sector = _parse_sector_from_message(question, sector_choices)
        if parsed_sector and parsed_sector != current_sector:
            return current_domain, parsed_sector, None

    intention_choices = _get_q2_choices_list(current_domain, secteur_choisi=current_sector)
    if intention_choices:
        parsed_intention_code = _parse_intention_code_from_message(question, intention_choices)
        if parsed_intention_code:
            current_intention = parsed_intention_code

    return current_domain, current_sector, current_intention


def _derive_selection_state_from_history(
    history: list[dict],
    selected_domain_code: str | None = None,
    selected_sector: str | None = None,
    selected_intention: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Rejoue l'identification domaine/secteur/intention sur tous les messages,
    en se basant sur le type de question posé par l'assistant (Q1/Q1.5/Q2),
    sans dépendre du nombre de tours.
    """
    def _detect_expected_step_from_assistant(text: str) -> str | None:
        t = " ".join((text or "").lower().split())
        if not t:
            return None
        if "dans quel domaine" in t or "domaine souhaitez-vous" in t:
            return "domain"
        if "q1.5" in t or "secteur" in t:
            return "sector"
        if "q2" in t or "objectif principal" in t or "intentions" in t:
            return "intention"
        return None

    current_domain = selected_domain_code
    current_sector = selected_sector
    current_intention = selected_intention
    expected_step: str | None = None
    for msg in history:
        role = (msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            detected = _detect_expected_step_from_assistant(content)
            if detected:
                expected_step = detected
            continue
        if role != "user":
            continue

        # Domaine explicite : toujours prioritaire. Evite de relire ce message comme secteur/intention.
        parsed_domain = _parse_domaine_from_message(content)
        is_plain_integer = bool(re.fullmatch(r"\d+", content))
        if parsed_domain and parsed_domain != current_domain:
            if current_domain is None or not is_plain_integer or expected_step == "domain":
                current_domain = parsed_domain
                current_sector = None
                current_intention = None
                expected_step = "sector" if get_q15_choices(current_domain) else "intention"
                continue

        if not current_domain:
            # Tant qu'aucun domaine n'est validé, on ignore secteur/intention.
            continue

        if expected_step == "sector":
            sector_choices = get_q15_choices(current_domain)
            parsed_sector = _parse_sector_from_message(content, sector_choices) if sector_choices else None
            if parsed_sector:
                if parsed_sector != current_sector:
                    current_sector = parsed_sector
                    current_intention = None
                expected_step = "intention"
            continue

        if expected_step == "intention":
            intention_choices = _get_q2_choices_list(current_domain, secteur_choisi=current_sector)
            parsed_intention_code = _parse_intention_code_from_message(content, intention_choices) if intention_choices else None
            if parsed_intention_code:
                current_intention = parsed_intention_code
                expected_step = None
            continue

        # Fallback défensif (si aucun step détecté dans les messages assistant)
        if not current_sector:
            sector_choices = get_q15_choices(current_domain)
            parsed_sector = _parse_sector_from_message(content, sector_choices) if sector_choices else None
            if parsed_sector:
                current_sector = parsed_sector
                current_intention = None
                continue
        intention_choices = _get_q2_choices_list(current_domain, secteur_choisi=current_sector)
        parsed_intention_code = _parse_intention_code_from_message(content, intention_choices) if intention_choices else None
        if parsed_intention_code:
            current_intention = parsed_intention_code
    return current_domain, current_sector, current_intention


def _get_user_replies_ordered(history: list[dict], current_message: str | None = None) -> list[str]:
    """Retourne la liste des réponses utilisateur dans l'ordre (contenu uniquement). current_message est ajouté en dernier si fourni (même si identique au précédent, c'est un tour de parole différent)."""
    replies = []
    for m in history:
        if (m.get("role") or "").strip().lower() != "user":
            continue
        raw = m.get("content")
        text = str(raw).strip() if raw is not None else ""
        if text:
            replies.append(text)
    if current_message is not None:
        msg = str(current_message).strip()
        if msg:
            replies.append(msg)
    return replies


def _get_intention_from_history(
    history: list[dict],
    selected_domain_code: str | None,
    current_message: str | None = None,
) -> str | None:
    """
    Retourne l'intention (libellé) détectée depuis les messages utilisateur, ou None.
    Le domaine est fourni explicitement (selected_domain_code), sans redéduction depuis l'historique.
    """
    if not selected_domain_code:
        return None
    sector = _get_sector_from_history(history, current_message=current_message)
    choices = _get_q2_choices_list(selected_domain_code, secteur_choisi=sector)
    if not choices:
        return None
    history_with_current = history + ([{"role": "user", "content": current_message}] if current_message else [])
    _, _, intention_code = _derive_selection_state_from_history(
        history_with_current,
        selected_domain_code=selected_domain_code,
        selected_sector=None,
        selected_intention=None,
    )
    return _get_intention_label_from_code(selected_domain_code, intention_code)


def _get_sector_from_history(history: list[dict], current_message: str | None = None) -> str | None:
    """
    Retourne le secteur choisi en Q1.5 par l'utilisateur, ou None.
    Déduction basée sur le flux conversationnel (questions assistant + réponses user),
    sans dépendre du nombre de messages.
    """
    history_with_current = history + ([{"role": "user", "content": current_message}] if current_message else [])
    _, sector, _ = _derive_selection_state_from_history(
        history_with_current,
        selected_domain_code=None,
        selected_sector=None,
        selected_intention=None,
    )
    return sector


def _get_secteur_choices_affichage(history: list[dict], domaine_code: str | None = None) -> str:
    """
    Retourne la liste des secteurs pour Q1.5 formatée pour affichage.
    Si domaine_code est fourni (ex. stocké après Q1), il est réutilisé ; sinon déduit de l'historique.
    """
    if domaine_code is None:
        if len(history) < 2:
            return ""
        domaine_code = _get_domaine_code_from_history(history)
    if not domaine_code:
        return ""
    choices = get_q15_choices(domaine_code)
    if not choices:
        return ""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(choices, start=1))


def _get_intention_choices_affichage(
    history: list[dict], domaine_code: str | None = None, secteur_choisi: str | None = None
) -> str:
    """
    Retourne la liste des intentions (Q2) pour le domaine choisi, formatée pour affichage.
    Si domaine_code est fourni (ex. stocké après Q1), il est réutilisé ; sinon déduit de l'historique.
    """
    if domaine_code is None:
        if len(history) < 2:
            return ""
        domaine_code = _get_domaine_code_from_history(history)
    if not domaine_code:
        return ""
    resolved_sector = secteur_choisi if secteur_choisi is not None else _get_sector_from_history(history)
    choices = _get_q2_choices_list(domaine_code, secteur_choisi=resolved_sector)
    if not choices:
        return ""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(choices, start=1))


def _get_q3_triggers_affichage(
    history: list[dict],
    domaine_code: str | None = None,
    selected_intention: str | None = None,
    selected_sector: str | None = None,
) -> str:
    """
    Retourne la liste des triggers (exemples de situations) pour Q3, formatée pour affichage (tirets).
    Utilisée quand le parcours est à l'étape Q3 (domaine + intention déjà choisis). Les triggers
    viennent du pool Chroma filtré par domaine et intention.
    """
    if not domaine_code:
        return ""
    if not selected_intention:
        return ""
    intention = _get_intention_label_from_code(
        domaine_code, selected_intention, secteur_choisi=selected_sector
    )
    triggers = get_q3_triggers(
        domaine_code,
        intention,
        secteur_choisi=selected_sector,
        top_k=Q3_TRIGGERS_DISPLAY_LIMIT,
    )
    if not triggers:
        return ""
    return "\n".join(f"- {t}" for t in triggers)


def _get_rag_hint(history: list[dict]) -> str:
    if len(history) >= 4:
        return "Important : c'est au moins la 3e demande de l'utilisateur. Tente de répondre avec les extraits et les informations déjà fournies ; ne pose plus de questions de clarification."
    # Après bienvenue + première réponse : obligatoirement Q1 (14 domaines fixes)
    if len(history) == 2 and not _get_domaine_code_from_history(history):
        domains_line = " ; ".join(f"{i}. {d}" for i, d in enumerate(Q1_DOMAINS_LIST, start=1))
        return (
            "C'est la première question après le message de bienvenue. Tu DOIS poser UNIQUEMENT la question Q1 (domaine) : "
            "« Dans quel domaine souhaitez-vous agir en priorité ? » "
            "Puis afficher EXACTEMENT les 14 choix suivants (numérotés 1 à 14) : "
            f"{domains_line}. "
            "Ne demande jamais les priorités, l'objectif ou le secteur avant le domaine. "
            "Ne pose pas Q1.5 (secteur) ni Q2 (objectif) tant que l'utilisateur n'a pas choisi un domaine (un nombre entre 1 et 14)."
        )
    # Q1.5 ou Q2 : rappel dans le hint (les listes sont dans le prompt)
    if len(history) >= 2:
        if _get_secteur_choices_affichage(history):
            return (
                "Si tu poses la question Q1.5 (secteur), tu DOIS afficher dans ta réponse "
                "la liste des secteurs fournie ci-dessous. "
                "Ne dis jamais « choisissez parmi la liste » sans afficher la liste."
            )
        intention_affichage = _get_intention_choices_affichage(history)
        if intention_affichage:
            return (
                "Si tu poses la question Q2 (objectif principal), tu DOIS afficher dans ta réponse "
                "la liste des intentions fournie ci-dessous. "
                "Ne dis jamais « choisissez parmi la liste » ou « intentions proposées » sans afficher la liste."
            )
        if _get_domaine_code_from_history(history):
            return (
                "Si tu poses Q2 (objectif principal) et qu'aucune liste d'intentions n'est fournie ci-dessous : "
                "demande à l'utilisateur de décrire son objectif en une phrase. "
                "Ne dis jamais « la liste va s'afficher » ou « veuillez patienter »."
            )
    return ""


def _should_inject_rag_documents(
    selected_domain_code: str | None,
    selected_sector: str | None,
    selected_intention: str | None,
) -> bool:
    """
    True uniquement quand le parcours guidé est suffisamment complété
    pour identifier des cas:
    - domaine validé
    - intention validée
    - secteur validé si Q1.5 existe pour ce domaine
    """
    if not selected_domain_code or not selected_intention:
        return False
    q15_choices = get_q15_choices(selected_domain_code)
    if q15_choices and not selected_sector:
        return False
    return True


def _resolve_current_selection_state(
    history: list[dict],
    question: str,
    selected_domain_code: str | None,
    selected_sector: str | None,
    selected_intention: str | None,
) -> tuple[list[dict], str | None, str | None, str | None]:
    """Résout domaine/secteur/intention courants en priorisant ce qui est détecté dans l'historique."""
    history_with_current = history + [{"role": "user", "content": question}]
    derived_domain, derived_sector, derived_intention = _derive_selection_state_from_history(
        history_with_current,
        selected_domain_code=None,
        selected_sector=None,
        selected_intention=None,
    )
    current_domain = derived_domain or selected_domain_code
    current_sector = derived_sector or selected_sector
    current_intention = derived_intention or selected_intention
    return history_with_current, current_domain, current_sector, current_intention


def _retrieve_docs_for_question(
    question: str,
    selected_domain_code: str | None = None,
    selected_intention: str | None = None,
    selected_sector: str | None = None,
) -> list:
    """Récupère les documents pertinents via fallback progressif des filtres."""

    def _run_retrieval(filters: dict | None) -> list:
        retrieval_pipeline = build_rag_retrieval_only_pipeline(filters=filters)
        result = retrieval_pipeline.run({"embedder": {"text": question}})
        docs = result.get("retriever", {}).get("documents") or []
        if docs and isinstance(docs[0], list):
            docs = [d for sub in docs for d in sub]
        return docs

    min_docs = 3
    if not selected_sector:
        return _run_retrieval(
            _build_retrieval_filters(
                domaine_code=selected_domain_code,
                intention_code=selected_intention,
            )
        )

    # Étape 1: domaine + intention + secteur (AND)
    docs = _run_retrieval(
        _build_retrieval_filters(
            domaine_code=selected_domain_code,
            intention_code=selected_intention,
            selected_sector=selected_sector,
        )
    )
    if len(docs) >= min_docs:
        return docs

    # Étape 2: relâcher intention, garder domaine + secteur
    docs = _run_retrieval(
        _build_retrieval_filters(
            domaine_code=selected_domain_code,
            selected_sector=selected_sector,
        )
    )
    if len(docs) >= min_docs:
        return docs

    # Étape 3: domaine + intention + (secteur OR multi-sectoriel)
    docs = _run_retrieval(
        _build_retrieval_filters(
            domaine_code=selected_domain_code,
            intention_code=selected_intention,
            selected_sector=selected_sector,
            include_multisector=True,
        )
    )
    if len(docs) >= min_docs:
        return docs

    # Étape 4: retrieval sans filtre secteur, puis post-filtrage Python sur secteur.
    broad_docs = _run_retrieval(
        _build_retrieval_filters(
            domaine_code=selected_domain_code,
            intention_code=selected_intention,
        )
    )
    post_filtered = [
        doc for doc in broad_docs if _doc_matches_sector(doc, selected_sector, include_multisector=True)
    ]
    return post_filtered or broad_docs


def _docs_to_payload(docs: list) -> tuple[list[str], list[str], list[str], list[dict[str, str | None]]]:
    """Transforme les docs en (sources, suggested_case_ids, full_contents, case_extras)."""
    case_dicts = [_doc_to_case_dict(d, i) for i, d in enumerate(docs)]
    sources = [d.content[:400] + "..." if len(d.content) > 400 else d.content for d in docs]
    suggested_case_ids = [c["id"] for c in case_dicts]
    full_contents = [c["content"] for c in case_dicts]
    case_extras = [_case_extras_from_case_dict(c) for c in case_dicts]
    return sources, suggested_case_ids, full_contents, case_extras


def _build_detail_prompt_payload_from_cases(
    case_index: int,
    cases: list[dict],
    with_recap: bool = False,
) -> tuple[str, list[str], list[str], list[str], list[dict[str, str | None]]] | None:
    """Construit le payload de détail côté prompt (stream)."""
    if not (0 <= case_index < len(cases)):
        return None
    case_row = cases[case_index]
    content = (case_row.get("content") or "").strip()
    if not content:
        return None
    content_for_model = _append_structured_case_fields_to_content(content, case_row)
    case_content = (
        _build_detail_input_with_recap(case_index + 1, cases, content_for_model) if with_recap else content_for_model
    )
    prompt = DETAIL_PROMPT.replace("{{ case_content }}", case_content)
    sources = [content[:400] + "..." if len(content) > 400 else content]
    ids = [c.get("id", "") for c in cases]
    full_contents = [c.get("content", "") for c in cases]
    case_extras = [_case_extras_from_case_dict(c) for c in cases]
    return prompt, sources, ids, full_contents, case_extras


def _build_detail_answer_payload_from_cases(
    case_index: int,
    cases: list[dict],
    with_recap: bool = True,
) -> tuple[str, list[str], list[str], list[str], list[dict[str, str | None]]] | None:
    """Construit le payload de détail côté answer (non-stream)."""
    if not (0 <= case_index < len(cases)):
        return None
    case_row = cases[case_index]
    content = (case_row.get("content") or "").strip()
    if len(content) < 20:
        return None
    content_for_model = _append_structured_case_fields_to_content(content, case_row)
    case_content = (
        _build_detail_input_with_recap(case_index + 1, cases, content_for_model) if with_recap else content_for_model
    )
    answer = _run_detail_pipeline(case_content)
    sources = [content[:400] + "..." if len(content) > 400 else content]
    ids = [c.get("id", "") for c in cases]
    full_contents = [c.get("content", "") for c in cases]
    case_extras = [_case_extras_from_case_dict(c) for c in cases]
    return answer, sources, ids, full_contents, case_extras


def get_rag_prompt_and_sources(
    question: str,
    history: list[dict],
    last_suggested_cases: list[dict] | None = None,
    pending_action: str | None = None,
    pending_use_case_id: str | None = None,
    selected_domain_code: str | None = None,
    selected_sector: str | None = None,
    selected_intention: str | None = None,
) -> tuple[str, list[str], list[str], list[str], list[dict[str, str | None]], str | None, str | None, str | None]:
    """
    Retourne (prompt, sources, suggested_case_ids, full_contents, case_extras, selected_domain_code, selected_sector, selected_intention).
    case_extras : métadonnées structurées par cas (effort, prérequis, etc.), même ordre que les ids.
    Utilise les sélections explicites du client et les met à jour avec le message courant.
    Les trois derniers éléments sont domaine/secteur/intention après ce message (à stocker côté client).
    """
    # Référence numérique explicite ("point 2", "2ème", etc.) :
    # forcer le flux détail avec la liste fiable renvoyée au tour précédent.
    if last_suggested_cases and _has_explicit_point_number(question):
        idx = _resolve_detail_selection(question, last_suggested_cases)
        if idx is not None:
            payload = _build_detail_prompt_payload_from_cases(idx, last_suggested_cases)
            if payload is not None:
                prompt, sources, ids, full_contents, case_extras = payload
                return (
                    prompt,
                    sources,
                    ids,
                    full_contents,
                    case_extras,
                    selected_domain_code,
                    selected_sector,
                    selected_intention,
                )

    # Affirmation + action en attente → prompt de détail pour ce cas (pas de mise à jour domaine/secteur)
    if _is_affirmation(question) and pending_action == "expand_details" and pending_use_case_id and last_suggested_cases:
        for i, case in enumerate(last_suggested_cases):
            if (case.get("id") or "") == pending_use_case_id:
                payload = _build_detail_prompt_payload_from_cases(i, last_suggested_cases)
                if payload is not None:
                    prompt, sources, ids, full_contents, case_extras = payload
                    return (
                        prompt,
                        sources,
                        ids,
                        full_contents,
                        case_extras,
                        selected_domain_code,
                        selected_sector,
                        selected_intention,
                    )
    # Affirmation « ok » sans pending_* : uniquement si on a last_suggested_cases (ordre fiable)
    if _is_affirmation(question) and last_suggested_cases:
        last_assistant = _get_last_assistant_message(history)
        if last_assistant:
            case_index_1based = _parse_offer_detail_from_text(last_assistant)
            if case_index_1based is not None and 1 <= case_index_1based <= len(last_suggested_cases):
                payload = _build_detail_prompt_payload_from_cases(case_index_1based - 1, last_suggested_cases)
                if payload is not None:
                    prompt, sources, ids, full_contents, case_extras = payload
                    return (
                        prompt,
                        sources,
                        ids,
                        full_contents,
                        case_extras,
                        selected_domain_code,
                        selected_sector,
                        selected_intention,
                    )

    if last_suggested_cases and (_is_detail_request(question) or _has_explicit_point_number(question)):
        idx = _resolve_detail_selection(question, last_suggested_cases)
        if idx is not None:
            payload = _build_detail_prompt_payload_from_cases(idx, last_suggested_cases)
            if payload is not None:
                prompt, sources, ids, full_contents, case_extras = payload
                return (
                    prompt,
                    sources,
                    ids,
                    full_contents,
                    case_extras,
                    selected_domain_code,
                    selected_sector,
                    selected_intention,
                )

    # Demande de détail en stream sans last_suggested_cases:
    # fallback par thème (même logique que _try_detail_flow en non-stream).
    if _is_detail_request(question) and not last_suggested_cases:
        # Référence explicite ("point 2") sans liste précédente fiable: ne pas deviner.
        if _has_explicit_point_number(question):
            pass
        else:
            previous = _get_previous_user_message(history)
            if previous:
                previous_domain, _, previous_intention = _derive_selection_state_from_history(
                    history,
                    selected_domain_code=None,
                    selected_sector=None,
                    selected_intention=None,
                )
                docs = _retrieve_docs(
                    previous,
                    filters=_build_retrieval_filters(previous_domain, previous_intention),
                )
                if docs:
                    cases_from_docs = [_doc_to_case_dict(d, i) for i, d in enumerate(docs)]
                    idx = _resolve_detail_selection(question, cases_from_docs)
                    if idx is not None:
                        payload = _build_detail_prompt_payload_from_cases(idx, cases_from_docs, with_recap=True)
                        if payload is not None:
                            prompt, sources, ids, full_contents, case_extras = payload
                            return (
                                prompt,
                                sources,
                                ids,
                                full_contents,
                                case_extras,
                                selected_domain_code,
                                selected_sector,
                                selected_intention,
                            )

    history_with_current, current_domain, current_sector, current_intention = _resolve_current_selection_state(
        history,
        question,
        selected_domain_code,
        selected_sector,
        selected_intention,
    )
    hint = _get_rag_hint(history_with_current)
    conversation_history = _format_conversation_history(history)
    docs = []
    if _should_inject_rag_documents(current_domain, current_sector, current_intention):
        docs = _retrieve_docs_for_question(
            question,
            selected_domain_code=current_domain,
            selected_intention=current_intention,
            selected_sector=current_sector,
        )
    sources, suggested_case_ids, full_contents, case_extras = _docs_to_payload(docs)
    secteur_affichage = _get_secteur_choices_affichage(history_with_current, domaine_code=current_domain)
    intention_affichage = _get_intention_choices_affichage(history_with_current, domaine_code=current_domain)
    q3_triggers_affichage = _get_q3_triggers_affichage(
        history_with_current,
        domaine_code=current_domain,
        selected_intention=current_intention,
        selected_sector=current_sector,
    )

    logger.debug(
        "get_rag_prompt_and_sources selection domain=%r sector=%r intention=%r",
        current_domain,
        current_sector,
        current_intention,
    )
    prompt_text = _build_rag_prompt_from_docs(
        question,
        hint,
        conversation_history,
        docs,
        history_with_current,
        secteur_choices_affichage=secteur_affichage,
        intention_choices_affichage=intention_affichage,
        q3_triggers_affichage=q3_triggers_affichage,
        selected_domain_code=current_domain,
        selected_sector=current_sector,
        selected_intention=current_intention,
    )
    _pt = prompt_text or "Aucun contexte."
    logger.debug("get_rag_prompt_and_sources prompt_len=%s preview=%r", len(_pt), _pt[:1200])
    return (
        (prompt_text or "Aucun contexte."),
        sources,
        suggested_case_ids,
        full_contents,
        case_extras,
        current_domain,
        current_sector,
        current_intention,
    )


def _try_detail_flow(
    question: str,
    history: list[dict],
    last_suggested_cases: list[dict] | None,
) -> tuple[str, list[str], list[str], list[str], list[dict[str, str | None]]] | None:
    """
    Si la question est une demande de détail et qu'on peut déterminer quel cas (avec
    last_suggested_cases ou en refaisant une recherche avec le dernier message user),
    retourne (answer, sources, suggested_case_ids, full_contents, case_extras). Sinon None.
    """
    if not (_is_detail_request(question) or _has_explicit_point_number(question)):
        return None

    # 1) Utiliser last_suggested_cases si fourni et avec du contenu (seule source fiable pour l'ordre)
    if last_suggested_cases:
        idx = _resolve_detail_selection(question, last_suggested_cases)
        if idx is not None:
            payload = _build_detail_answer_payload_from_cases(idx, last_suggested_cases, with_recap=True)
            if payload is not None:
                return payload

    # 2) Référence explicite (point 2, 2ème…) SANS liste : ne pas deviner avec une autre recherche.
    #    L'ordre des docs récupérés ne correspond pas à la liste affichée → on éviterait la confusion.
    if _has_explicit_point_number(question):
        return None

    # 3) Fallback uniquement pour demande par thème (ex. « détaille celui sur la synthèse »)
    previous = _get_previous_user_message(history)
    if not previous:
        return None
    current_domain, _, current_intention = _derive_selection_state_from_history(
        history,
        selected_domain_code=None,
        selected_sector=None,
        selected_intention=None,
    )
    docs = _retrieve_docs(
        previous,
        filters=_build_retrieval_filters(current_domain, current_intention),
    )
    if not docs:
        return None
    cases_from_docs = [_doc_to_case_dict(d, i) for i, d in enumerate(docs)]
    idx = _resolve_detail_selection(question, cases_from_docs)
    if idx is None:
        return None
    payload = _build_detail_answer_payload_from_cases(idx, cases_from_docs, with_recap=True)
    if payload is None:
        return None
    return payload


def _execute_pending_expand_details(
    pending_use_case_id: str,
    last_suggested_cases: list[dict],
) -> tuple[str, list[str], list[str], list[str], list[dict[str, str | None]]] | None:
    """Exécute l'action expand_details pour le cas donné. Retourne (answer, sources, ids, full_contents, case_extras) ou None."""
    for idx, case in enumerate(last_suggested_cases):
        if (case.get("id") or "") == pending_use_case_id:
            return _build_detail_answer_payload_from_cases(idx, last_suggested_cases, with_recap=True)
    return None


def query_rag_haystack(
    question: str,
    history: list[dict],
    last_suggested_cases: list[dict] | None = None,
    pending_action: str | None = None,
    pending_use_case_id: str | None = None,
    selected_domain_code: str | None = None,
    selected_sector: str | None = None,
    selected_intention: str | None = None,
) -> tuple[
    str,
    list[str],
    list[str],
    list[str],
    list[dict[str, str | None]],
    str | None,
    str | None,
    int | None,
    str | None,
    str | None,
    str | None,
]:
    """
    Interroge le RAG. Retourne (answer, sources, suggested_case_ids, full_contents, case_extras, pending_action, pending_use_case_id, pending_case_index, selected_domain_code, selected_sector, selected_intention).
    selected_domain_code/selected_sector/selected_intention : état explicite fourni par le client.
    Les trois derniers retours sont domaine/secteur/intention après ce message (à stocker côté client).
    """
    # 1) Affirmation + action en attente fournie par le client → exécuter l'action
    if _is_affirmation(question) and pending_action == "expand_details" and pending_use_case_id and last_suggested_cases:
        result = _execute_pending_expand_details(pending_use_case_id, last_suggested_cases)
        if result is not None:
            a, s, i, f, x = result
            return a, s, i, f, x, None, None, None, selected_domain_code, selected_sector, selected_intention

    # 2) Affirmation sans pending_* : inférer depuis le dernier message assistant (offre de détail)
    if _is_affirmation(question) and last_suggested_cases:
        last_assistant = _get_last_assistant_message(history)
        if last_assistant:
            case_index_1based = _parse_offer_detail_from_text(last_assistant)
            if case_index_1based is not None and 1 <= case_index_1based <= len(last_suggested_cases):
                payload = _build_detail_answer_payload_from_cases(case_index_1based - 1, last_suggested_cases, with_recap=True)
                if payload is not None:
                    answer, sources, ids, full_contents, case_extras = payload
                    return (
                        answer,
                        sources,
                        ids,
                        full_contents,
                        case_extras,
                        None,
                        None,
                        None,
                        selected_domain_code,
                        selected_sector,
                        selected_intention,
                    )

    # 3) Demande explicite de détail (« détaille le 2 »)
    detail_result = _try_detail_flow(question, history, last_suggested_cases)
    if detail_result is not None:
        a, s, i, f, x = detail_result
        return a, s, i, f, x, None, None, None, selected_domain_code, selected_sector, selected_intention

    # 4) Flux RAG normal : utiliser l'état explicite client, mis à jour par le message courant
    history_with_current, current_domain, current_sector, current_intention = _resolve_current_selection_state(
        history,
        question,
        selected_domain_code,
        selected_sector,
        selected_intention,
    )
    hint = _get_rag_hint(history_with_current)
    conversation_history = _format_conversation_history(history)
    docs = []
    if _should_inject_rag_documents(current_domain, current_sector, current_intention):
        docs = _retrieve_docs_for_question(
            question,
            selected_domain_code=current_domain,
            selected_intention=current_intention,
            selected_sector=current_sector,
        )
    sources, suggested_case_ids, full_contents, case_extras = _docs_to_payload(docs)
    secteur_affichage = _get_secteur_choices_affichage(history_with_current, domaine_code=current_domain)
    intention_affichage = _get_intention_choices_affichage(history_with_current, domaine_code=current_domain)
    q3_triggers_affichage = _get_q3_triggers_affichage(
        history_with_current,
        domaine_code=current_domain,
        selected_intention=current_intention,
        selected_sector=current_sector,
    )
    logger.debug("query_rag_haystack q3_triggers_affichage=%r domain=%r", q3_triggers_affichage, current_domain)
    prompt_text = _build_rag_prompt_from_docs(
        question,
        hint,
        conversation_history,
        docs,
        history_with_current,
        secteur_choices_affichage=secteur_affichage,
        intention_choices_affichage=intention_affichage,
        q3_triggers_affichage=q3_triggers_affichage,
        selected_domain_code=current_domain,
        selected_sector=current_sector,
        selected_intention=current_intention,
    )
    _pt = prompt_text or "Aucun contexte."
    logger.debug("query_rag_haystack prompt_len=%s preview=%r", len(_pt), _pt[:1200])
    generator = _get_generator()
    gen_result = generator.run(prompt=prompt_text)
    replies = gen_result.get("replies", [])
    answer = replies[0] if replies else "Aucune réponse générée."
    if hasattr(answer, "content"):
        answer = answer.content

    pending_case_index = _parse_offer_detail_from_text(answer)
    if pending_case_index is not None and 1 <= pending_case_index <= len(suggested_case_ids):
        pending_uid = suggested_case_ids[pending_case_index - 1]
        return (
            answer,
            sources,
            suggested_case_ids,
            full_contents,
            case_extras,
            "expand_details",
            pending_uid,
            pending_case_index,
            current_domain,
            current_sector,
            current_intention,
        )
    return (
        answer,
        sources,
        suggested_case_ids,
        full_contents,
        case_extras,
        None,
        None,
        None,
        current_domain,
        current_sector,
        current_intention,
    )


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

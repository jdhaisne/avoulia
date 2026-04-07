"""
Constantes partagées du parcours RAG / taxonomie / métadonnées Chroma.
"""

# Premier message envoyé par l'agent au chargement du chat (exposé via GET /chat/welcome).
WELCOME_MESSAGE = (
    "Bonjour, je vais vous aider à identifier des cas d'usage concrets "
    "de l'IA adaptés à votre organisation. Pour commencer, je vais vous "
    "poser quelques questions simples afin de cibler précisément votre priorité."
)

# Domaines sans liste de secteurs (référence) : absents de SECTEURS_PAR_DOMAINE, Q1.5 est ignorée pour eux.
DOMAINES_SANS_SECTEURS = ["direction_strategie", "innovation_rnd", "it_systemes_donnees"]

# Secteurs proposés pour Q1.5 selon le domaine choisi (choix 1–14).
SECTEURS_PAR_DOMAINE = {
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
    ],
}

# Correspondance choix Q1 (1–14) → code domaine. Source unique pour la question Q1.
CHOIX_Q1_TO_DOMAINE_CODE = {
    1: "direction_strategie",
    2: "organisation_coordination",
    3: "ressources_humaines",
    4: "ventes_developpement",
    5: "marketing_visibilite",
    6: "relation_client",
    7: "finance_pilotage",
    8: "it_systemes_donnees",
    9: "conformite_risque",
    10: "achats_fournisseurs",
    11: "logistique_stocks",
    12: "production",
    13: "activites_terrain",
    14: "innovation_rnd",
}

# Libellés des 14 choix Q1 (ordre 1 à 14), pour affichage et matching. Dérivé de CHOIX_Q1_TO_DOMAINE_CODE.
Q1_DOMAINS_LIST = [
    "Direction & décisions stratégiques",
    "Organisation & efficacité interne",
    "RH & gestion des équipes",
    "Développement commercial",
    "Marketing & visibilité",
    "Service & relation client",
    "Finances & rentabilité",
    "Outils, systèmes & données",
    "Obligations & gestion des risques",
    "Achats & relations fournisseurs",
    "Stocks & logistique",
    "Production & opérations",
    "Chantiers & activités terrain",
    "Innovation & nouveaux projets",
]

# Intentions (objectifs) par domaine pour Q2. Si vide, on tente de les charger depuis Chroma (meta "intention" / "domaine_label").
INTENTIONS_PAR_DOMAINE: dict[str, list[str]] = {code: [] for code in CHOIX_Q1_TO_DOMAINE_CODE.values()}

# Noms de colonnes possibles dans l'index (XLSX) pour domaine et intention (selon en-têtes du fichier).
DOMAINE_META_KEYS = ("domaine_label", "domaine_label_fr", "Domaine", "domaine")
INTENTION_META_KEYS = ("intention", "Intention", "objectif", "Objectif")
# Clés meta pour les triggers (exemples de situations) Q3 — colonnes XLSX possibles.
TRIGGER_META_KEYS = (
    "trigger",
    "Trigger",
    "situation",
    "Situation",
    "situation_trigger",
    "exemple_situation",
    "declencheurs_typiques",
)

# Champs Chroma/Haystack pour filtrer par domaine (préfixe meta. pour filter_documents).
CHROMA_DOMAIN_META_FILTER_FIELDS = ("meta.domaine_label", "meta.domaine", "meta.Domaine")

# Champs structurés supplémentaires par cas (XLSX : en-têtes de colonnes possibles, voir ingest).
CASE_EXTRA_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "effort": ("effort", "Effort", "niveau_effort", "Niveau d'effort"),
    "prerequis_donnees": (
        "prerequis_donnees",
        "prérequis_données",
        "Prerequis_donnees",
        "prerequis",
        "Prérequis données",
    ),
    "guardrails": ("guardrails", "Guardrails", "guardrails_pme"),
    "questions_qualification": (
        "questions_qualification",
        "Questions qualification",
        "questions_de_qualification",
        "Questions de qualification",
    ),
    "sensibilite_donnees": (
        "sensibilite_donnees",
        "sensibilité_données",
        "Sensibilite_donnees",
        "data_sensitivity",
        "sensibilite",
        "Sensibilité données",
    ),
}
CASE_EXTRA_KEYS: tuple[str, ...] = tuple(CASE_EXTRA_FIELD_ALIASES.keys())

# Limite d’exemples de situations affichés pour Q3.
Q3_TRIGGERS_DISPLAY_LIMIT = 12

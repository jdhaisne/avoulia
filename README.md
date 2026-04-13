## Objectif
Chatbot open source pour aider à explorer et structurer des cas d’usage de
l’IA.
## Transparence &amp; limites
- Système IA : réponses potentiellement inexactes.
- Ne saisissez pas de données personnelles ou confidentielles.
- Ne constitue pas un conseil professionnel.
## Données &amp; confidentialité
- Pas de compte requis.
- Pas de conservation de l’historique (voir PRIVACY.md).
- Messages envoyés à Azure OpenAI (voir FAQ).
## Licence
[MIT ou Apache-2.0]
## Marques
Les noms et logos de Microsoft et Simplon ne sont pas concédés par la
licence open source.

# Chatbot RAG – FastAPI + Vue

Application de recherche de documents par question/réponse (RAG) : backend FastAPI, **Chroma** (stockage vectoriel) + **Haystack** (pipelines RAG), LLM OpenAI ou Azure, frontend Vue 3.

## Prérequis

- Python 3.11+
- Node.js 18+
- Clé API OpenAI

## Installation

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Éditer .env et mettre OPENAI_API_KEY=sk-...
```

### Frontend

```bash
cd frontend
npm install
```

## Lancement

1. **Démarrer l’API** (depuis `backend/`) :

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Démarrer le frontend** (depuis `frontend/`) :

```bash
npm run dev
```

3. Ouvrir http://localhost:5173. Le proxy Vite envoie les appels `/api` vers le backend (port 8000).

## Déploiement Docker

À la racine du projet :

```bash
# Créer backend/.env à partir de .env.example et le remplir (clés Azure/OpenAI, etc.)
cp backend/.env.example backend/.env

# Build et lancement
docker compose up -d

# Logs
docker compose logs -f
```

- **Frontend** : http://localhost:8080. Nginx sert l’app et proxyfie `/api` vers le backend.
- **Backend** : http://localhost:8000 (API, `/health`, `/docs`). Chat en streaming : `POST /api/v1/chat/stream`.
- Les données Chroma sont persistées dans le volume `backend_data`. Nginx désactive le buffering sur `/api/` pour le SSE.

Arrêt : `docker compose down`. Suppression des données : `docker compose down -v`.

## Indexation des documents

L’indexation se fait **uniquement via le script** (même machine et même base Chroma que l’API). Il n’y a plus d’upload ni de page « Documents » dans l’interface.

- **En local** (depuis `backend/`) :
  ```bash
  python -m app.scripts.index_documents [--clear] /chemin/vers/fichier.pdf /chemin/vers/dossier
  ```
  Types supportés : **PDF**, **TXT**, **MD**, **XLSX**. Option `--clear` pour vider l’index avant d’indexer.

- **Au lancement du conteneur** : par défaut, le `docker-compose` monte le dossier `./documents` en lecture et définit `INDEX_PATH=/app/documents`. Au démarrage du conteneur backend, les fichiers (PDF, TXT, MD, XLSX) présents dans `./documents` sont indexés une fois, puis l’API démarre.  
  Déposez vos fichiers dans le dossier `documents/` à la racine du projet avant de lancer `docker compose up -d`. Pour utiliser un autre dossier ou désactiver l’indexation au démarrage, modifiez ou supprimez `INDEX_PATH` et le volume associé dans `docker-compose.yml`.

## Utilisation

1. **Indexer les documents** : utiliser le script ci-dessus (une fois au déploiement ou au lancement du conteneur).
2. **Poser des questions** : taper une question et envoyer. Si `USE_RAG=true` dans le `.env`, le RAG Haystack interroge Chroma et génère une réponse avec sources ; sinon le chat est simple (LLM seul).

## Structure du projet

```
avoulia_v2/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI, CORS, routes
│   │   ├── config.py        # Settings (env)
│   │   ├── models.py        # Schémas Pydantic
│   │   ├── rag.py           # Chat simple (LangChain + Azure/OpenAI)
│   │   ├── haystack_rag.py  # RAG Haystack (Chroma + embedders + generator)
│   │   ├── routes/          # chat
│   │   ├── scripts/         # index_documents (CLI)
│   │   └── services/        # ingestion (PDF, TXT, MD)
│   ├── data/chroma/         # Base vectorielle (créée au premier run)
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── api/chat.ts      # Client API (chat, ingest)
│   │   ├── views/ChatView.vue
│   │   └── ...
│   └── vite.config.ts       # Proxy /api -> :8000
└── README.md
```

## API (backend)

- `POST /api/v1/chat` – Envoi d’un message, retourne la réponse RAG + sources.
- `POST /api/v1/chat/stream` – Même chose en streaming (SSE).
- `GET /health` – Santé du serveur.
- `GET /docs` – Documentation Swagger.

L’indexation des documents se fait uniquement via le script `app.scripts.index_documents` (voir section « Indexation des documents »).

## Configuration (backend)

Variables d’environnement (fichier `.env`) :

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | URL de base Azure (ex. `https://xxx.cognitiveservices.azure.com`) |
| `AZURE_OPENAI_API_KEY` | Clé API Azure OpenAI |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Déploiement chat (défaut: gpt-4o-mini) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Déploiement embeddings (défaut: text-embedding-3-small) |
| `OPENAI_API_KEY` | Clé OpenAI (si pas Azure) |
| `OPENAI_EMBEDDING_MODEL` | Modèle d’embeddings (défaut: text-embedding-3-small) |
| `OPENAI_CHAT_MODEL` | Modèle de chat (défaut: gpt-4o-mini) |
| `CHROMA_PERSIST_DIR` | Dossier Chroma (défaut: ./data/chroma) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Découpage des documents |
| `TOP_K_RETRIEVE` | Nombre d’extraits récupérés pour chaque question |

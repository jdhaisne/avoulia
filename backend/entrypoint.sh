#!/bin/sh
# Si INDEX_PATH est défini, indexer une fois au démarrage (même base Chroma que l'API).
if [ -n "$INDEX_PATH" ]; then
  echo "[Entrypoint] Indexation des documents : $INDEX_PATH"
  python -m app.scripts.index_documents --clear $INDEX_PATH
  echo "[Entrypoint] Indexation terminée."
fi
exec "$@"

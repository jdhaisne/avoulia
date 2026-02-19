"""
Indexation des documents dans Chroma (même base que l'API).
À lancer avec des chemins en arguments (fichiers ou dossiers).
Exemple : python -m app.scripts.index_documents --clear /data/docs /data/fic.pdf
"""

import argparse
import sys
from pathlib import Path

# Permet d'importer app depuis la racine backend
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.haystack_rag import clear_all_documents
from app.services.ingest import ingest_file

SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".xlsx")


def collect_files(paths: list[str]) -> list[Path]:
    """Collecte tous les fichiers supportés à partir de chemins (fichiers ou dossiers)."""
    collected: list[Path] = []
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            print(f"[skip] N'existe pas : {path}")
            continue
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                collected.append(path)
            else:
                print(f"[skip] Type non supporté : {path}")
        else:
            for ext in SUPPORTED_SUFFIXES:
                collected.extend(path.rglob(f"*{ext}"))
    return sorted(set(collected))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Indexe des documents (PDF, TXT, MD, XLSX) dans Chroma (même base que l'API)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Fichiers ou dossiers à indexer",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Vider l'index Chroma avant d'indexer",
    )
    args = parser.parse_args()

    if args.clear:
        print("Vidage de l'index Chroma…")
        try:
            clear_all_documents()
            print("Index vidé.")
        except Exception as e:
            print(f"Erreur lors du vidage : {e}", file=sys.stderr)
            sys.exit(1)

    files = collect_files(args.paths)
    if not files:
        print("Aucun fichier à indexer.")
        return

    print(f"Indexation de {len(files)} fichier(s)…")
    total_chunks = 0
    errors: list[str] = []
    for f in files:
        try:
            ids = ingest_file(str(f))
            n = len(ids)
            total_chunks += n
            print(f"  OK {f.name} → {n} chunk(s)")
        except Exception as e:
            msg = f"{f}: {e}"
            errors.append(msg)
            print(f"  ERREUR {f.name}: {e}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} erreur(s).", file=sys.stderr)
        sys.exit(1)
    print(f"\nIndexation terminée : {total_chunks} chunk(s) au total.")


if __name__ == "__main__":
    main()

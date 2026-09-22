# domains/financial_services/data/ingest_reference_corpus.py
# One-time (idempotent -- see core/rag/reference_corpus.py) ingestion of
# Financial Services' static reference corpus (FinCEN CVC guidance
# FIN-2019-G001) into the vector DB, so chat retrieval can draw on it
# alongside whatever wallet/exchange statement a user actually uploads.
# Run this once per fresh vector DB (a new local ./vector_store) -- not per user upload.
#
# Usage (from the prism/ repo root -- must run as a module, not a bare
# script path, so `core`/`domains` resolve on sys.path):
#   python -m domains.financial_services.data.ingest_reference_corpus

from pathlib import Path

from core.rag.reference_corpus import ingest_reference_corpus
from domains.financial_services.agents.document_loader_agent import DocumentLoaderAgent

REFERENCE_FILE = Path(__file__).parent.parent / "reference_corpus" / "fincen_cvc_guidance_excerpt.txt"


def main():
    loader = DocumentLoaderAgent()
    result = ingest_reference_corpus(loader, str(REFERENCE_FILE))
    print(f"Ingested {result['chunks_stored']} chunks from {REFERENCE_FILE.name} "
          f"into the '{loader.settings.COLLECTION_NAME}' collection.")


if __name__ == "__main__":
    main()

# core/rag/reference_corpus.py
# Ingests each domain's static reference/policy document -- once, by the
# maintainer, not per user upload -- into the same vector collection a
# domain's own DocumentLoaderAgent already writes to, under one fixed
# source id. BaseRetrieverAgent.retrieve()'s `reference_source` param
# then makes it compete fairly (same HyDE/RRF/rerank pipeline) alongside
# whatever document a user actually uploaded, rather than a separate
# retrieval pass bolted on after the fact.
#
# Reuses BaseDocumentLoaderAgent.load() as-is: a reference document is
# ingested exactly like a user's own upload, just under a fixed filename
# instead of a per-upload one, and it's typically curated .txt/.md
# (regulatory/policy text doesn't need PDF's layout fidelity) rather than
# a PDF, but any format load() accepts works here too.

REFERENCE_SOURCE_ID = "__reference_corpus__"


def ingest_reference_corpus(loader, file_path: str) -> dict:
    """Call once per domain (e.g. at deploy time or the first time this
    module runs against a fresh vector DB) -- re-running it is safe and
    idempotent, since load() already deletes-then-reinserts any existing
    chunks for the same source id (see _store_chroma)."""
    return loader.load(file_path, filename=REFERENCE_SOURCE_ID)

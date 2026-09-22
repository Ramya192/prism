# Tests get their own throwaway vector store, never the developer's real
# ./vector_store. Without this, every run re-ingests the same fixed sample
# documents into the real store, which (a) pollutes it with test debris and
# (b) now collides with the duplicate-upload guard
# (core/rag/base_document_loader_agent.py's find_duplicate) on the second run.
#
# CHROMA_PATH must be set before any domain settings module is imported
# (each reads it once, at import), which is why this runs at conftest load
# time rather than inside a fixture. load_dotenv() never overrides an
# already-set variable, so this wins over .env.

import atexit
import importlib
import os
import shutil
import tempfile

import pytest

_STORE = tempfile.mkdtemp(prefix="prism_test_store_")
os.environ["CHROMA_PATH"] = _STORE
atexit.register(shutil.rmtree, _STORE, ignore_errors=True)

_REFERENCE_CORPUS_INGESTS = [
    "domains.banking.documents.data.ingest_reference_corpus",
    "domains.insurance.data.ingest_reference_corpus",
    "domains.financial_services.data.ingest_reference_corpus",
    "domains.payroll_hr.data.ingest_reference_corpus",
]


@pytest.fixture(scope="session", autouse=True)
def reference_corpora():
    """The fresh store starts empty; tests that chat against a domain's
    reference corpus (IRS/CMS/FinCEN/SHRM) need it ingested first, exactly
    as a real deployment does once per new vector store."""
    for module in _REFERENCE_CORPUS_INGESTS:
        importlib.import_module(module).main()


@pytest.fixture(scope="module", autouse=True)
def fresh_documents_per_module():
    """Test modules ingest the same fixed sample files (e.g. CLM001_claim in
    both the insurance and cross-document suites). The duplicate-upload guard
    rightly refuses a second ingest of identical content, so each module
    starts with only the reference corpora present in the throwaway store."""
    import chromadb
    from core.rag.reference_corpus import REFERENCE_SOURCE_ID

    client = chromadb.PersistentClient(path=_STORE)
    for collection in client.list_collections():
        collection.delete(where={"source": {"$ne": REFERENCE_SOURCE_ID}})
    yield

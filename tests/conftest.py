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
from dotenv import load_dotenv

# Every domain settings module calls load_dotenv() too, but only on first
# import -- which for most test files happens well after this conftest
# module (and its HAS_OPENAI_KEY check below) has already run. Without
# this, a real key sitting in .env is invisible here and every
# "integration" test gets skipped even when the suite could actually run.
# load_dotenv() never overrides an already-set variable, so this is safe
# to call before CHROMA_PATH is forced below.
load_dotenv()

_STORE = tempfile.mkdtemp(prefix="prism_test_store_")
os.environ["CHROMA_PATH"] = _STORE
# The public-site daily scoring budget (core/usage_budget.py) would otherwise
# be shared, process-wide, across every test that ingests a CSV.
os.environ.setdefault("PRISM_DAILY_SCORED_ROWS_BUDGET", "0")
atexit.register(shutil.rmtree, _STORE, ignore_errors=True)

_REFERENCE_CORPUS_INGESTS = [
    "domains.banking.documents.data.ingest_reference_corpus",
    "domains.insurance.data.ingest_reference_corpus",
    "domains.financial_services.data.ingest_reference_corpus",
    "domains.payroll_hr.data.ingest_reference_corpus",
]

# Values .env.example ships as fill-in-the-blank placeholders, or that a
# careless copy-paste leaves behind -- treated the same as "not set" so a
# copied-but-unfilled .env doesn't look configured.
_PLACEHOLDER_KEY_VALUES = {
    "", "your-openai-api-key-here", "your-cohere-api-key-here",
    "changeme", "placeholder", "xxx",
}


def _is_real_key(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in _PLACEHOLDER_KEY_VALUES


HAS_OPENAI_KEY = _is_real_key(os.environ.get("OPENAI_API_KEY"))

if not os.environ.get("OPENAI_API_KEY"):
    # langchain's ChatOpenAI/OpenAIEmbeddings require *a* key string to
    # construct, even for the many non-"integration" tests that build a
    # pipeline/agent object but never call .invoke()/.embed*() on it (see
    # e.g. test_conversational_rag.py's and test_domain_pipelines.py's
    # docstrings). A syntactically-plausible placeholder lets those run on
    # a totally fresh clone with no .env at all; it's deliberately not in
    # _PLACEHOLDER_KEY_VALUES's realness check, since HAS_OPENAI_KEY above
    # is already computed from the pre-injection value.
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-not-a-real-key"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: makes real OpenAI (and optionally Cohere) API calls; "
        "auto-skipped unless a real OPENAI_API_KEY is configured.",
    )


def pytest_collection_modifyitems(config, items):
    if HAS_OPENAI_KEY:
        return
    skip_no_key = pytest.mark.skip(
        reason="requires a real OPENAI_API_KEY (embeddings/LLM calls) -- set one in .env to run this test"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_no_key)


@pytest.fixture(scope="session")
def reference_corpora():
    """The fresh store starts empty; tests that chat against a domain's
    reference corpus (IRS/CMS/FinCEN/SHRM) need it ingested first, exactly
    as a real deployment does once per new vector store. Not autouse --
    only the real end-to-end pipeline tests that actually chat against
    reference-corpus content request this (via `pytestmark = [...,
    pytest.mark.usefixtures("reference_corpora")]`), so the rest of the
    suite (drift detectors, scorers, config loader, etc.) never needs an
    OpenAI key just to ingest documents it never reads."""
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

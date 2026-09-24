# Cross-batch duplicate-upload guard (core/rag/base_document_loader_agent.py's
# find_duplicate + core/unified_pipeline.py's ingest). Offline: the loader's
# embedder is stubbed, so no API key or network is needed.

import pytest

from core.rag.base_document_loader_agent import BaseDocumentLoaderAgent, DuplicateDocumentError
from core.unified_pipeline import UnifiedDomainPipeline


class _Settings:
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 0
    EMBEDDING_MODEL = "unused"
    COLLECTION_NAME = "dup_test"


class _FakeEmbedder:
    def embed_documents(self, chunks):
        return [[0.1, 0.2, 0.3] for _ in chunks]


@pytest.fixture
def loader(tmp_path, monkeypatch):
    class Settings(_Settings):
        CHROMA_PATH = str(tmp_path / "chroma")

    monkeypatch.setattr("core.rag.base_document_loader_agent.OpenAIEmbeddings", lambda **_: _FakeEmbedder())
    return BaseDocumentLoaderAgent(Settings)


class _Pipeline(UnifiedDomainPipeline):
    def __init__(self, loader):
        self.loader = loader

    def score_record(self, record, context=None):
        return {"predicted": "LEGITIMATE", "reason": ""}


def _csv(path, body="a,b\n1,2\n3,4\n"):
    path.write_text(body)
    return str(path)


def test_same_content_second_ingest_raises(loader, tmp_path):
    pipeline = _Pipeline(loader)
    first = _csv(tmp_path / "one.csv")
    assert pipeline.ingest(first, filename="one")["chunks_stored"] == 2
    with pytest.raises(DuplicateDocumentError) as exc:
        pipeline.ingest(first, filename="one")
    assert exc.value.existing == "one"


def test_renamed_copy_is_still_a_duplicate(loader, tmp_path):
    pipeline = _Pipeline(loader)
    pipeline.ingest(_csv(tmp_path / "one.csv"), filename="one")
    with pytest.raises(DuplicateDocumentError) as exc:
        pipeline.ingest(_csv(tmp_path / "copy.csv"), filename="copy")
    assert exc.value.existing == "one"


def test_different_content_is_not_a_duplicate(loader, tmp_path):
    pipeline = _Pipeline(loader)
    pipeline.ingest(_csv(tmp_path / "one.csv"), filename="one")
    other = _csv(tmp_path / "two.csv", "a,b\n9,9\n")
    assert pipeline.ingest(other, filename="two")["document"] == "two"


def test_duplicate_is_rejected_before_anything_is_stored(loader, tmp_path):
    pipeline = _Pipeline(loader)
    path = _csv(tmp_path / "one.csv")
    pipeline.ingest(path, filename="one")
    before = loader.collection.count()
    with pytest.raises(DuplicateDocumentError):
        pipeline.ingest(path, filename="renamed")
    assert loader.collection.count() == before


def test_legacy_ownerless_upload_is_invisible_to_owner_check_and_purgeable(loader, tmp_path):
    from core.rag.purge_legacy_uploads import find_legacy_chunks
    from core.rag.reference_corpus import REFERENCE_SOURCE_ID

    path = _csv(tmp_path / "old.csv")
    loader.load(path, filename="old")                       # pre-owner upload: no owner tag
    loader.load(path, filename="mine", owner="abc12345")    # a visitor's own upload
    loader.load(path, filename=REFERENCE_SOURCE_ID)         # reference corpus: ownerless by design

    # An owner-scoped check finds only that owner's copy, never the legacy one.
    assert loader.find_duplicate(path, "abc12345") == "mine@abc12345"

    # Only the legacy upload is a purge candidate.
    legacy = find_legacy_chunks(loader.collection)
    assert list(legacy) == ["old"]
    loader.collection.delete(ids=legacy["old"])
    assert find_legacy_chunks(loader.collection) == {}
    assert loader.find_duplicate(path, "abc12345") == "mine@abc12345"

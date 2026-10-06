# Visitor-upload expiry (core/rag/purge_old_uploads.py) and the upload-time stamp
# the loader writes. Offline: chromadb in a temp dir, embedder stubbed.

import chromadb
import pytest

from core.rag.base_document_loader_agent import BaseDocumentLoaderAgent
from core.rag.purge_old_uploads import select_expired
from core.rag.reference_corpus import REFERENCE_SOURCE_ID

HOUR = 3600


def _collection(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_or_create_collection("purge_test")


def _add(col, source, n, **meta):
    col.add(
        documents=[f"{source} chunk {i}" for i in range(n)],
        embeddings=[[0.1, 0.2, 0.3]] * n,
        ids=[f"{source}_c{i}" for i in range(n)],
        metadatas=[{"source": source, "chunk_index": i, **meta} for i in range(n)],
    )


def test_only_old_owned_uploads_are_selected(tmp_path):
    col = _collection(tmp_path)
    now = 1_000_000.0
    _add(col, REFERENCE_SOURCE_ID, 3)                                        # reference corpus: no owner
    _add(col, "old.csv@abc", 2, owner="abc", uploaded_at=now - 30 * HOUR)    # 30 h old
    _add(col, "fresh.csv@def", 2, owner="def", uploaded_at=now - 1 * HOUR)   # 1 h old
    expired, skipped = select_expired(col, cutoff=now - 24 * HOUR)
    assert set(expired) == {"old.csv@abc"}
    assert len(expired["old.csv@abc"]) == 2
    assert skipped == 0


def test_reference_corpus_is_never_selected_even_if_stamped(tmp_path):
    col = _collection(tmp_path)
    now = 1_000_000.0
    _add(col, REFERENCE_SOURCE_ID, 2, owner="x", uploaded_at=now - 99 * HOUR)
    expired, _ = select_expired(col, cutoff=now, include_untimed=True)
    assert expired == {}


def test_untimed_owned_uploads_need_the_flag(tmp_path):
    col = _collection(tmp_path)
    _add(col, "legacy.pdf@old", 2, owner="old")                              # owner, no uploaded_at
    expired, skipped = select_expired(col, cutoff=10**9)
    assert expired == {} and skipped == 2
    expired, skipped = select_expired(col, cutoff=10**9, include_untimed=True)
    assert set(expired) == {"legacy.pdf@old"} and skipped == 0


def test_ownerless_uploads_are_left_to_the_legacy_purge(tmp_path):
    col = _collection(tmp_path)
    _add(col, "plain.csv", 2)                                                # no owner at all
    expired, skipped = select_expired(col, cutoff=10**9, include_untimed=True)
    assert expired == {} and skipped == 0


class _FakeEmbedder:
    def embed_documents(self, chunks):
        return [[0.1, 0.2, 0.3] for _ in chunks]


@pytest.fixture
def loader(tmp_path, monkeypatch):
    class Settings:
        CHUNK_SIZE = 200
        CHUNK_OVERLAP = 0
        EMBEDDING_MODEL = "unused"
        COLLECTION_NAME = "stamp_test"
        CHROMA_PATH = str(tmp_path / "chroma2")

    monkeypatch.setattr("core.rag.base_document_loader_agent.OpenAIEmbeddings", lambda **_: _FakeEmbedder())
    return BaseDocumentLoaderAgent(Settings)


def test_loader_stamps_owned_uploads_only(loader, tmp_path):
    f = tmp_path / "doc.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    owned = loader.load(str(f), filename="doc", owner="sess1")
    metas = loader.collection.get(where={"source": owned["document"]}, include=["metadatas"])["metadatas"]
    assert all(isinstance(m.get("uploaded_at"), int) and m["uploaded_at"] > 1_700_000_000 for m in metas)

    g = tmp_path / "other.csv"
    g.write_text("c,d\n5,6\n")
    plain = loader.load(str(g), filename="other")                            # no owner
    metas = loader.collection.get(where={"source": plain["document"]}, include=["metadatas"])["metadatas"]
    assert all("uploaded_at" not in m for m in metas)

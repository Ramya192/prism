# core/rag/purge_legacy_uploads.py
# One-off cleanup for documents uploaded before per-visitor naming existed.
#
# Those uploads carry no `owner` tag, so the owner-scoped duplicate check
# (BaseDocumentLoaderAgent.find_duplicate) deliberately never matches them --
# matching would let one visitor's upload block, or reveal the filename of,
# another's -- and no session's chat can reach them either. They still sit in
# the shared vector store as orphans. Ownership can't be reconstructed after
# the fact, so the only real fix is to delete them (a visitor just re-uploads).
#
# The one legitimate ownerless document is each domain's reference corpus
# (REFERENCE_SOURCE_ID), which is always kept.
#
#   python -m core.rag.purge_legacy_uploads            # dry run: list only
#   python -m core.rag.purge_legacy_uploads --apply    # actually delete
#
# (In Docker: `docker compose exec prism python -m core.rag.purge_legacy_uploads`.)

from __future__ import annotations

import sys

from core.config_loader import ConfigLoader
from core.rag.reference_corpus import REFERENCE_SOURCE_ID


def find_legacy_chunks(collection) -> dict[str, list[str]]:
    """{source name: [chunk ids]} for every chunk with no owner tag, excluding
    the reference corpus."""
    found = collection.get(include=["metadatas"])
    legacy: dict[str, list[str]] = {}
    for chunk_id, meta in zip(found["ids"], found["metadatas"]):
        meta = meta or {}
        source = meta.get("source", "")
        if meta.get("owner") or source == REFERENCE_SOURCE_ID:
            continue
        legacy.setdefault(source, []).append(chunk_id)
    return legacy


def purge(apply: bool = False) -> int:
    """Returns the number of legacy chunks found (deleted only if `apply`)."""
    from chromadb import PersistentClient

    # Several domains can share one Chroma path; open each (path, collection) once.
    targets = {}
    for domain in ConfigLoader().list_domains():
        vdb = domain.extra.get("vector_db") or {}
        if vdb.get("collection") and vdb.get("chroma_path"):
            targets[(vdb["chroma_path"], vdb["collection"])] = domain.id

    total = 0
    for (path, name), domain_id in sorted(targets.items(), key=lambda kv: kv[1]):
        collection = PersistentClient(path=path).get_or_create_collection(name=name)
        legacy = find_legacy_chunks(collection)
        n = sum(len(ids) for ids in legacy.values())
        total += n
        print(f"[{domain_id}] {name}: {len(legacy)} ownerless document(s), {n} chunk(s)")
        for source, ids in sorted(legacy.items()):
            print(f"    {source}  ({len(ids)} chunks)")
            if apply:
                collection.delete(ids=ids)

    print(f"{'Deleted' if apply else 'Would delete'} {total} chunk(s)."
          + ("" if apply else "  Re-run with --apply to delete."))
    return total


if __name__ == "__main__":
    purge(apply="--apply" in sys.argv)

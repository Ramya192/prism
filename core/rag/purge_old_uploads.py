# core/rag/purge_old_uploads.py
# Expires visitor uploads by age, so documents uploaded to the public demo do
# not pile up in the vector store indefinitely.
#
# A visitor upload is a chunk with an `owner` tag; ingest stamps it with
# `uploaded_at` (epoch seconds). Each domain's reference corpus has no owner
# and is never touched. Uploads made before the stamp existed have an owner but
# no age: they are only deleted when --include-untimed is passed.
#
#   python -m core.rag.purge_old_uploads                         # dry run, 24 h
#   python -m core.rag.purge_old_uploads --hours 6 --apply       # delete older than 6 h
#   python -m core.rag.purge_old_uploads --include-untimed --apply
#
# On the server, from cron (see deploy/README.md):
#   docker compose exec -T prism python -m core.rag.purge_old_uploads --hours 24 --apply

from __future__ import annotations

import sys
import time

from core.rag.purge_legacy_uploads import collection_targets
from core.rag.reference_corpus import REFERENCE_SOURCE_ID


def select_expired(collection, cutoff: float, include_untimed: bool = False) -> tuple[dict[str, list[str]], int]:
    """({source: [chunk ids]} to delete, number of untimed owned chunks skipped).
    Only chunks with an owner tag are candidates; the reference corpus never is."""
    found = collection.get(include=["metadatas"])
    expired: dict[str, list[str]] = {}
    skipped_untimed = 0
    for chunk_id, meta in zip(found["ids"], found["metadatas"]):
        meta = meta or {}
        source = meta.get("source", "")
        if not meta.get("owner") or source == REFERENCE_SOURCE_ID:
            continue
        stamp = meta.get("uploaded_at")
        if stamp is None:
            if include_untimed:
                expired.setdefault(source, []).append(chunk_id)
            else:
                skipped_untimed += 1
        elif stamp < cutoff:
            expired.setdefault(source, []).append(chunk_id)
    return expired, skipped_untimed


def purge(hours: float = 24.0, apply: bool = False, include_untimed: bool = False, now: float | None = None) -> int:
    """Returns the number of chunks selected (deleted only if `apply`)."""
    from chromadb import PersistentClient

    cutoff = (now if now is not None else time.time()) - hours * 3600
    total = 0
    for (path, name), domain_id in sorted(collection_targets().items(), key=lambda kv: kv[1]):
        collection = PersistentClient(path=path).get_or_create_collection(name=name)
        expired, skipped = select_expired(collection, cutoff, include_untimed)
        n = sum(len(ids) for ids in expired.values())
        total += n
        print(f"[{domain_id}] {name}: {len(expired)} expired document(s), {n} chunk(s)"
              + (f"; {skipped} untimed chunk(s) kept (use --include-untimed)" if skipped else ""))
        for source, ids in sorted(expired.items()):
            print(f"    {source}  ({len(ids)} chunks)")
            if apply:
                collection.delete(ids=ids)

    print(f"{'Deleted' if apply else 'Would delete'} {total} chunk(s) older than {hours:g} h."
          + ("" if apply else "  Re-run with --apply to delete."))
    return total


def _hours_arg(argv: list[str]) -> float:
    if "--hours" in argv:
        return float(argv[argv.index("--hours") + 1])
    return 24.0


if __name__ == "__main__":
    purge(hours=_hours_arg(sys.argv), apply="--apply" in sys.argv, include_untimed="--include-untimed" in sys.argv)

# core/rag/seed_history.py
# Shared by every document-shaped domain's Pipeline.ingest(): turns the
# auto-scan capability's own query + verdict into the first two
# conversation turns, so a caller seeding chat history with this gets a
# natural "here's what was automatically checked, and what it found"
# opening instead of a blank slate the user has to re-ask for. A
# follow-up like "why was this flagged?" then has the actual verdict
# already in context rather than the model re-deriving one from a fresh,
# disconnected retrieval.
#
# Field names differ slightly per domain (banking documents: anomaly_flag/
# anomaly_reason; payroll: flag/flag_reason) -- handled generically here
# rather than duplicated per domain.


def build_seed_history(auto_query: str, scan: dict, document: str | None = None) -> list[dict]:
    if scan.get("status") != "valid":
        return []
    data = scan["data"]
    flagged = data.get("anomaly_flag") or data.get("flag")
    reason = data.get("anomaly_reason") or data.get("flag_reason")
    if flagged:
        verdict_text = f"⚠ Flagged: {reason}"
    elif reason:
        # Nothing was scored (e.g. not this domain's kind of document): say so,
        # rather than implying a fraud check ran and came back clean.
        verdict_text = f"ℹ {reason}"
    else:
        verdict_text = "No anomalies found — this looks consistent."
    # With several documents loaded, a bare verdict turn can't say which
    # file it belongs to, so a follow-up got anchored to the wrong one.
    label = f"[{document}] " if document else ""
    return [
        {"speaker": "user", "text": f"{label}{auto_query}"},
        {"speaker": "assistant", "text": f"{label}{data.get('answer', '')} {verdict_text}".strip()},
    ]

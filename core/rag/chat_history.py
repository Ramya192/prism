# core/rag/chat_history.py
# How every domain's ReasoningAgent shows earlier chat turns to the model.
# It was four identical copies (banking, insurance, financial_services,
# payroll); one copy now, so a change to it can't reach only some domains.

FOLLOW_UP_ONLY_NOTE = (
    "(Use these earlier turns only to resolve a follow-up such as \"why was that flagged?\". "
    "If the new question is about a different topic, ignore them and answer the new question. "
    "Each earlier verdict is tagged with the document it belongs to; a turn that says no "
    "anomalies were found means nothing was flagged in that document -- say so rather than "
    "inventing flags, and when more than one document is loaded name which document you are "
    "talking about.)"
)


def render_history(history: list[dict] | None) -> str:
    """Renders prior conversation turns for the prompt -- including, when the
    caller seeds it, the auto-scan verdict as the first turn, so a follow-up
    like "why was this flagged?" can reference the verdict already reached
    instead of the LLM re-deriving one from scratch. Each turn:
    {"speaker": "user"|"assistant", "text": str}.

    The closing note matters: the seeded verdict is always turn one, and
    without it a question on a NEW topic (e.g. a regulation question) was
    answered from that verdict instead of from the retrieved context."""
    if not history:
        return ""
    lines = ["Previous conversation on this same document (most recent last):"]
    for turn in history:
        speaker = "You" if turn.get("speaker") == "assistant" else "User"
        lines.append(f"{speaker}: {turn.get('text', '')}")
    return "\n".join(lines) + "\n" + FOLLOW_UP_ONLY_NOTE + "\n"

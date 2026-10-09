# ui/shared.py — renderers used by every domain's (C) workspace, Banking
# included. render_chat_panel's transactions/total_amount block is the
# one Banking-specific bit, folded in as a guarded `.get()` rather than a
# separate drawer implementation (see that function's own docstring).
# Split out of streamlit_app.py, which had grown to 1483 lines.

import html
import logging
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

from core.rag.base_document_loader_agent import DuplicateDocumentError, display_name
from core.rag.reference_corpus import REFERENCE_SOURCE_ID
from ui.guard import allow, describe_ingest_error, get_owner
from ui.uploads import demo_file_uploader

logger = logging.getLogger(__name__)

# Up to this many flagged records are listed inline (a bank statement or a
# claim, typically). More than that -- a CSV with hundreds of rows -- switches
# to a summary: nobody reads 300 near-identical reasons, so the screen leads
# with counts and the few records most worth a look, and keeps the full list
# in a collapsed, searchable, downloadable table.
FLAGGED_INLINE_LIMIT = 5
GROUPS_SHOWN = 4      # distinct rule / LLM reasons spelled out in the summary
EXAMPLE_ROWS = 3      # example row numbers quoted per group

_UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")


def md_safe(text) -> str:
    """Escapes `$` in model-written text before it goes through Streamlit's
    markdown. Two dollar amounts in one sentence ("over $500 ... $5.00") are
    otherwise parsed as inline LaTeX: the text between them turns into a
    math span and the dollar signs vanish."""
    return _UNESCAPED_DOLLAR.sub(r"\\$", str(text))


def render_fraud_verdicts(verdicts: list[dict] | None, summary: dict | None) -> None:
    """UnifiedDomainPipeline.ingest()'s result shape (core/unified_pipeline.py):
    every extracted record gets its own real verdict through the exact
    same rules->tiered ML->LLM engine the CSV fraud path uses, not one
    whole-document LLM-only flag. A domain whose document genuinely
    carries no fraud-schema fields (e.g. an HR handbook) gets an honest
    empty list here, not a forced verdict — see
    core/unified_pipeline.py's document-type gate for the real bug this
    distinction exists to prevent."""
    if verdicts is None:
        return
    if summary and summary.get("budget_limited"):
        st.warning(
            "This demo's daily scoring budget is used up, so some or all records were not checked "
            "for fraud. Chat is still available; scoring resumes within 24 hours."
        )
    if not verdicts:
        if not (summary and summary.get("budget_limited")):
            st.info("No recognizable fraud-schema records could be extracted from this document — chat is still available below.")
        hint = (summary or {}).get("domain_hint")
        if hint:
            st.warning(
                f"This looks like {_a_an(hint['name'])} **{hint['name']}** document, not one this workspace can fraud-check. "
                f"Use “← change domain” and pick {hint['name']} to get a real fraud verdict."
            )
        return
    if summary:
        st.caption(f"Scored {summary['total_scored']} record(s) — {summary['flagged']} flagged, {summary['clean']} clean.")
        if summary.get("truncated") and not summary.get("budget_limited"):
            st.warning(
                f"Only the first {summary['total_scored']:,} of {summary['rows_in_file']:,} rows in this file "
                "were scored — the rest were not checked (scoring ceiling for interactive uploads)."
            )
    numbered = len(verdicts) > 1   # a row number only helps when there is more than one record
    flagged = []
    for i, v in enumerate(verdicts, 1):
        if v["predicted"] == "FRAUD":
            reason = v.get("reason", "")
            parsed = v.get("parsed") or {}
            tier = parsed.get("decided_by") or _classify_detection_tier(reason, parsed.get("action", ""))
            flagged.append((i, tier, reason))

    if not flagged:
        st.success("✓ Nothing flagged — everything extracted looks consistent")
    elif len(flagged) <= FLAGGED_INLINE_LIMIT:
        st.error(f"⚠ {len(flagged)} record(s) flagged")
        for i, tier, reason in flagged:
            st.caption(md_safe(f"— {f'#{i} ' if numbered else ''}[{tier}] {reason}"))
    else:
        _render_flagged_summary(flagged, len(verdicts))

    # Banking Tier 1 only (the one schema this was measured as a rare,
    # meaningful signal rather than noise — see drift_detector.py) —
    # independent of the verdict above, so shown for flagged AND clean
    # records alike; a `.get()` no-op for every other domain, which never
    # produces this key at all.
    caveat_records = [(i, (v.get("parsed") or {})["caveat"]) for i, v in enumerate(verdicts, 1)
                      if (v.get("parsed") or {}).get("caveat")]
    if caveat_records:
        # The note is the same sentence for every record: say it once, with
        # which records it applies to, instead of repeating it N times.
        by_text: dict[str, list[int]] = {}
        for i, text in caveat_records:
            by_text.setdefault(text, []).append(i)
        with st.expander(f"🔍 {len(caveat_records):,} record(s) also flagged as statistically unusual"):
            for text, rows in by_text.items():
                st.caption(md_safe(text[:1].upper() + text[1:]))
                if numbered:
                    shown = ", ".join(f"#{i}" for i in rows[:20])
                    more = f" … and {len(rows) - 20:,} more" if len(rows) > 20 else ""
                    st.caption(f"Records: {shown}{more}")


def _priority(tier: str, reason: str) -> float:
    """How much a flagged record deserves a human's first look. A rule hit is
    a deterministic impossibility (top), an ML hit ranks by its own fraud
    probability, and an LLM-only hit (a judgement call) comes last."""
    if tier == "Rule Engine":
        return 101.0
    if tier == "ML Detector":
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", reason)
        return float(m.group(1)) if m else 50.0
    return 40.0


def _rows(ids: list[int]) -> str:
    shown = ", ".join(f"#{i}" for i in ids[:EXAMPLE_ROWS])
    return shown + (" …" if len(ids) > EXAMPLE_ROWS else "")


def _layer_summary(flagged: list[tuple[int, str, str]]) -> list[str]:
    """One line per finding, not per record: identical rule reasons collapse to
    a count, the ML layer to a probability range, and only a few LLM judgement
    calls are quoted. Markdown-safe."""
    lines = []
    rules = [f for f in flagged if f[1] == "Rule Engine"]
    by_reason: dict[str, list[int]] = {}
    for i, _, reason in rules:
        by_reason.setdefault(reason, []).append(i)
    for reason, ids in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))[:GROUPS_SHOWN]:
        noun = "record" if len(ids) == 1 else "records"
        lines.append(f"📏 **Rule Engine** — {reason} · {len(ids):,} {noun} (e.g. {_rows(ids)})")
    if len(by_reason) > GROUPS_SHOWN:
        lines.append(f"📏 **Rule Engine** — {len(by_reason) - GROUPS_SHOWN} other rule(s) in the full list")

    ml = [f for f in flagged if f[1] == "ML Detector"]
    if ml:
        ranked = sorted(ml, key=lambda f: _priority(f[1], f[2]), reverse=True)
        probs = [_priority(f[1], f[2]) for f in ml]
        lines.append(
            f"🧠 **ML Detector** — {len(ml):,} records, fraud probability {min(probs):.0f}–{max(probs):.0f}% "
            f"(highest: {_rows([f[0] for f in ranked])})"
        )

    llm = [f for f in flagged if f[1] == "LLM Reasoning"]
    for i, _, reason in llm[:GROUPS_SHOWN]:
        lines.append(f"💬 **LLM Reasoning** — #{i}: {reason}")
    if len(llm) > GROUPS_SHOWN:
        lines.append(f"💬 **LLM Reasoning** — {len(llm) - GROUPS_SHOWN} more in the full list")
    return [md_safe(line) for line in lines]


def _render_flagged_summary(flagged: list[tuple[int, str, str]], total: int) -> None:
    """Many flagged records: counts, one line per finding, and the full list
    collapsed into a searchable/sortable table with a CSV download."""
    n = len(flagged)
    st.error(f"⚠ {n:,} of {total:,} records flagged ({n / total:.0%})")

    by_layer = Counter(tier for _, tier, _ in flagged)
    cols = st.columns(len(by_layer))
    for col, (tier, count) in zip(cols, by_layer.most_common()):
        col.metric(tier, f"{count:,}")

    st.markdown("**What was flagged**")
    for line in _layer_summary(flagged):
        st.markdown(line)

    frame = pd.DataFrame(flagged, columns=["Row", "Layer", "Reason"])
    with st.expander(f"All {n:,} flagged records (search and sort)"):
        # A dataframe renders cells as plain text, so no markdown escaping is needed.
        st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇ Download flagged records (CSV)", frame.to_csv(index=False).encode("utf-8"),
        file_name="flagged_records.csv", mime="text/csv",
        key=f"dl_flagged_{n}_{total}_{flagged[0][0]}",
    )


def _classify_detection_tier(reason: str, action: str) -> str:
    """Every domain's rule_based_filter()/ml_filter() writes its reason
    through the same three conventions (see e.g.
    domains/insurance/agents/healthcare_detector_agent.py): an ML-tier
    reason always contains the literal "ML model" plus a probability;
    a rule-tier reason is a short, deterministic template (never a
    probability, never the FLAG action — rules only fire on genuine
    impossibilities, which are always a clean BLOCK, or don't fire at
    all); anything else is the LLM's own free-form sentence. This reads
    that convention back out instead of guessing, so the breakdown
    shown to the user names the layer that actually decided."""
    if not reason:
        return "LLM Reasoning"
    if "ML model" in reason:
        return "ML Detector"
    if action in {"BLOCK", "APPROVE"} and "%" not in reason and len(reason) < 160:
        return "Rule Engine"
    return "LLM Reasoning"


def render_explainability(flagged: bool, action: str, reason: str, decided_by: str | None = None) -> None:
    """"WHY WAS THIS FLAGGED?" — names which of Rules/ML/LLM actually
    produced this verdict's reason, instead of presenting the pipeline
    as one opaque black box. `decided_by` is the layer the detector
    recorded; without it (domains that don't yet) the tier is inferred."""
    if not reason:
        return
    tier = decided_by or _classify_detection_tier(reason, action)
    icon = {"Rule Engine": "📏", "ML Detector": "🧠", "LLM Reasoning": "💬"}.get(tier, "💬")
    title = "WHY WAS THIS FLAGGED?" if flagged else "WHY THIS LOOKS CLEAN"
    st.markdown(
        f"<div class='explain-panel'><div class='explain-title'>{title}</div>"
        f"<div class='explain-row'><span class='explain-tier'>{icon} {tier}</span>"
        f"<span class='explain-reason'>{html.escape(str(reason)).replace('$', '&#36;')}</span></div></div>",
        unsafe_allow_html=True,
    )


UNVERIFIABLE_REASON_PREFIX = "No positive gross pay"   # payroll_detector_agent's _rule_check_no_pay


def _a_an(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def render_verdict_result(result: dict) -> None:
    """Same {"predicted", "parsed": {...}} shape all 4 domains'
    score_record() returns for a single CSV/manual-entry record (see
    each domain's pipeline.py) — one renderer, reused everywhere a
    single record gets scored rather than a whole document's many."""
    if result is None:   # rate-limited -- guarded_score() already said why
        return
    predicted = result["predicted"]
    parsed = result["parsed"]
    reason = parsed.get("reason") or result.get("reason", "")
    risk_level = parsed.get("risk_level", "—")
    action = parsed.get("action", "—")
    flagged = predicted == "FRAUD"

    with st.container(key="analysis_card"):
        st.markdown("<div class='analysis-title'>PRISM Analysis</div>", unsafe_allow_html=True)
        if flagged and reason.startswith(UNVERIFIABLE_REASON_PREFIX):
            st.warning("⚠ UNABLE TO VERIFY — nothing to score")
        elif flagged:
            st.error(f"🚨 FRAUD — Risk: {risk_level}")
        else:
            st.success(f"✅ LEGITIMATE — Risk: {risk_level}")
        st.markdown(
            "<div class='pipeline-checklist'><span>✓ Domain Agent</span>"
            "<span>✓ Configuration Loaded</span><span>✓ Detection Engine</span>"
            "<span>✓ Explanation</span></div>",
            unsafe_allow_html=True,
        )
        render_explainability(flagged, action, reason, parsed.get("decided_by"))
    with st.expander("Detector report"):
        st.text(f"Risk Level : {risk_level}")
        st.text(f"Action     : {action}")
        st.text(f"Reason     : {reason}")


def render_pipeline_flow(domain_config) -> None:
    """The actual agent routing this domain's config wires up, so the
    architecture shown is real rather than a mockup: every domain has a
    Fraud Agent (rules → ML → LLM); a Document Agent (RAG) only shows up
    if this domain's config actually declares a document_qa capability,
    so a domain without one isn't shown a branch it doesn't have."""
    has_doc_qa = any(c.id == "document_qa" for c in domain_config.capabilities)
    branches = "<div class='flow-branch'>Fraud Agent<div class='flow-sub'>Rules → ML → LLM</div></div>"
    if has_doc_qa:
        branches += "<div class='flow-branch'>Document Agent<div class='flow-sub'>RAG retrieval</div></div>"
    st.markdown(
        "<div class='arch-flow'>"
        "<div class='flow-node flow-root'>PRISM</div>"
        "<div class='flow-arrow'>↓</div>"
        "<div class='flow-node'>Domain Agent</div>"
        "<div class='flow-arrow'>↓</div>"
        f"<div class='flow-detected'>{domain_config.name} detected</div>"
        "<div class='flow-arrow'>↓</div>"
        f"<div class='flow-branches'>{branches}</div>"
        "<div class='flow-arrow'>↓</div>"
        "<div class='flow-node flow-result'>Explainable Result</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_config_panel(domain_config) -> None:
    """"PRISM CONFIGURATION" — a live read of configs/<id>.yaml via
    DomainConfig, not a mockup. Rows only appear when the underlying
    YAML actually declares that section, so a domain with no
    thresholds/reference corpus is shown as such rather than a
    fabricated value."""
    extra = domain_config.extra
    rows = [("Domain", domain_config.name), ("Detection", "Rules → ML → LLM")]
    if domain_config.capabilities:
        rows.append(("Agents", " + ".join(c.label for c in domain_config.capabilities)))
    data_cfg = extra.get("data", {})
    ref_corpus = data_cfg.get("reference_corpus")
    # An explicit label reads better than a title-cased filename ("Irs Pub15T
    # Excerpt"); the filename is only the fallback for a config without one.
    kb_label = data_cfg.get("reference_corpus_label") or (
        Path(ref_corpus).stem.replace("_", " ").title() if ref_corpus else "Document only"
    )
    rows.append(("Knowledge Base", kb_label))
    model = extra.get("llm", {}).get("openai_model") or extra.get("model", {}).get("name")
    if model:
        rows.append(("LLM Model", model))
    thresholds = extra.get("thresholds")
    if thresholds:
        rows.append(("Thresholds", ", ".join(f"{k}={v}" for k, v in thresholds.items())))
    rows.append(("Human Review", "Enabled — confirm before run"))

    rows_html = "".join(
        f"<div class='config-row'><span class='config-key'>{k}</span>"
        f"<span class='config-val'>{v}</span></div>"
        for k, v in rows
    )
    st.markdown(
        f"<div class='config-panel'><div class='config-panel-title'>Prism Configuration</div>{rows_html}</div>",
        unsafe_allow_html=True,
    )


def render_citations(chunks: list[dict] | None, doc_name: str | None) -> None:
    """D — small source chips under a chat answer. Every domain's
    retrieve() can now merge the uploaded document's own chunks with the
    domain's static reference corpus in one fused ranking (see
    core/rag/base_retriever_agent.py's reference_source param); this is
    what makes which source actually backed an answer checkable at a
    glance instead of taken on faith."""
    if not chunks:
        return
    labels, seen = [], set()
    for c in chunks:
        src = c.get("source")
        if not src or src in seen:
            continue
        seen.add(src)
        if src == REFERENCE_SOURCE_ID:
            labels.append("📚 reference corpus")
        else:
            labels.append(f"📄 {display_name(src)}")
    if labels:
        st.caption("Sources: " + " · ".join(labels))


def render_banner(banner_id: str, level: str, title: str, body: str) -> None:
    """A dismissible top-of-workspace notice — 'info' (blue) or 'warn'
    (yellow), styled after PayNexus's banner pattern. Dismissal is
    per-session (session_state), not permanent, same as PayNexus's own
    dismissible notices."""
    dismiss_key = f"banner_dismissed_{banner_id}"
    if st.session_state.get(dismiss_key):
        return
    with st.container(key=f"banner_{level}"):
        col_text, col_close = st.columns([20, 1])
        with col_text:
            st.markdown(
                f"<div class='banner-title'>{title}</div><div class='banner-body'>{body}</div>",
                unsafe_allow_html=True,
            )
        with col_close:
            if st.button("✕", key=f"{dismiss_key}_btn"):
                st.session_state[dismiss_key] = True
                st.rerun()


def render_floating_chat(domain_id: str, render_drawer_content) -> None:
    """A floating chat bubble (bottom-right, fixed position) that opens a
    floating drawer above it — the PayNexus-style floating-widget pattern
    (see that project's ChatWidget.tsx/App.tsx). Only one workspace is
    ever on screen at a time, so a single fixed key ("chat_fab"/
    "chat_drawer") is safe to reuse across domains. `render_drawer_content`
    is a zero-arg callable — every domain, Banking included, passes the
    same render_chat_panel(); this stays generic rather than assuming
    render_chat_panel specifically, since that's the caller's choice, not
    this function's.

    PayNexus doesn't actually make the drawer "push" the page via real
    flex/grid layout either — the panel is `position: fixed` (out of flow)
    and a separate padding-right is reserved on the main content area to
    match, a fake push via reserved space, not a real one. Same trick
    here: reserve padding-right on Streamlit's `.block-container` (the
    main content area — st.sidebar is a structurally separate element, so
    it's untouched, matching PayNexus's own nav not shifting either)."""
    chat_key = f"{domain_id}_chat_open"
    max_key = f"{domain_id}_chat_maximized"
    is_open = st.session_state.get(chat_key, False)
    is_maximized = st.session_state.get(max_key, False)

    if is_open:
        st.markdown(
            "<style>@media (min-width: 1200px) { .block-container { padding-right: 480px !important; "
            "transition: padding-right 0.2s ease; } }</style>",
            unsafe_allow_html=True,
        )
        if is_maximized:
            # PayNexus's own maximized mode (ChatWidget.tsx's `inset-4`/
            # `sm:inset-8`) -- near-fullscreen instead of the small corner
            # panel. Left offset clears the sidebar (~336px default width)
            # rather than covering it, same as PayNexus's nav staying put.
            st.markdown(
                "<style>.st-key-chat_drawer { top: 90px !important; left: 360px !important; "
                "right: 20px !important; bottom: 20px !important; width: auto !important; "
                "max-width: none !important; }</style>",
                unsafe_allow_html=True,
            )

    # Same pattern as PayNexus's ChatWidget: a launcher bubble when closed;
    # when open, a header bar (title + maximize/restore + minimize) over the
    # conversation. Minimizing also resets the maximized size.
    if not is_open:
        with st.container(key="chat_fab"):
            if st.button("💬", key="chat_fab_btn"):
                st.session_state[chat_key] = True
                st.rerun()
        return

    with st.container(key="chat_drawer"):
        with st.container(key="chat_header"):
            title_col, max_col, min_col = st.columns([6, 1, 1], vertical_alignment="center")
            with title_col:
                st.markdown("<div class='chat-drawer-title'>💬 Prism Assistant</div>", unsafe_allow_html=True)
            with max_col:
                if st.button("", key=f"{domain_id}_chat_max_btn",
                             icon=":material/close_fullscreen:" if is_maximized else ":material/open_in_full:",
                             help="Restore" if is_maximized else "Maximize"):
                    st.session_state[max_key] = not is_maximized
                    st.rerun()
            with min_col:
                if st.button("", key=f"{domain_id}_chat_min_btn", icon=":material/remove:", help="Minimize"):
                    st.session_state[chat_key] = False
                    st.session_state[max_key] = False
                    st.rerun()
        with st.container(key="chat_body"):
            render_drawer_content()


# Kept short on purpose: each of these was run against a real sample document
# and answered correctly on repeated tries. Don't add one without re-testing it
# (a question the model answers vaguely, or differently each time, makes the
# suggestion look broken).
GENERIC_QUESTIONS = [
    "Summarize this document in a few sentences.",
]

SUGGESTED_QUESTIONS = {
    "banking": [
        "Summarize this statement in a few sentences.",
        "Which transactions look the most suspicious, and why?",
        "What does Regulation E say about unauthorized transfers?",
    ],
    "payroll_hr": [
        "Which deductions are taken from this payslip?",
        "Summarize this payslip in a few sentences.",
        "What is the net pay?",
    ],
    "financial_services": [
        "Summarize the account activity in this document.",
        "Which transactions are the largest?",
    ],
    "insurance": [
        "Summarize this claim in a few sentences.",
        "What is the claimed amount?",
    ],
}


_CHAT_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_CHAT_JOBS: dict[str, dict] = {}   # "<owner>:<domain>" -> {"query", "future"}


@st.fragment(run_every=1)
def _chat_job_waiting(job_key: str) -> None:
    """Polls the background answer once a second. A fragment reruns on its
    own, so the page (and the chat's scroll position) isn't redrawn each
    tick -- a full-script `st.rerun()` poll made the whole UI blink."""
    job = _CHAT_JOBS.get(job_key)
    if job is None or job["future"].done():
        st.rerun()   # one full rerun to render the finished answer
    st.markdown("⏳ _Running retrieve → reason → validate..._")


def render_chat_panel(domain_id: str, pipeline, doc_key: str, history_key: str) -> None:
    """The docked chat panel's contents — one implementation shared by
    every domain, Banking included. The transactions/total_amount block
    below only ever fires for Banking (the only domain whose reasoning
    agent's response schema has those fields, see
    domains/banking/documents/agents/reasoning_agent.py) — a no-op
    `.get()` for every other domain, not a Banking-specific branch."""
    doc = st.session_state.get(doc_key)
    history = st.session_state.get(history_key, [])
    # Layout, top to bottom: conversation (past turns, then the question being
    # answered and its answer), then suggestions, then the input box. The
    # conversation is a container created first and filled later in this
    # function, so a new turn lands ABOVE the input instead of below it.
    convo = st.container()
    with convo:
        for turn in history:
            with st.chat_message("user" if turn["speaker"] == "user" else "assistant"):
                st.markdown(md_safe(turn["text"]))

    pending_key = f"{domain_id}_chat_pending"
    if doc:
        with st.expander("💡 Suggested questions", expanded=not history):
            for i, q in enumerate(SUGGESTED_QUESTIONS.get(domain_id, GENERIC_QUESTIONS)):
                if st.button(q, key=f"{domain_id}_suggest_{i}", use_container_width=True):
                    st.session_state[pending_key] = q
                    st.rerun()

    query = st.chat_input("Ask about this document...", key=f"{domain_id}_chat_input")
    query = query or st.session_state.pop(pending_key, None)

    with convo:
        # The answer is computed on a background thread, not in this script run:
        # Streamlit aborts a running script whenever any widget is clicked (e.g.
        # maximize), which used to throw the question away mid-answer. The job
        # lives in a module-level dict, so a rerun just finds it and keeps waiting.
        job_key = f"{get_owner()}:{domain_id}"
        job = _CHAT_JOBS.get(job_key)
        if query:
            if not doc:
                st.warning("Ingest a document first.")
                return
            if job is not None:
                st.info("Still working on your previous question — one at a time.")
            elif allow("chat"):
                job = _CHAT_JOBS[job_key] = {
                    "query": query,
                    "future": _CHAT_EXECUTOR.submit(
                        pipeline.run, query, source_document=doc, history=list(history),
                        skip_scope_check=query in SUGGESTED_QUESTIONS.get(domain_id, GENERIC_QUESTIONS),
                    ),
                }
        if job is None:
            return

        query = job["query"]
        with st.chat_message("user"):
            st.markdown(md_safe(query))
        if not job["future"].done():
            _chat_job_waiting(job_key)   # refreshes only this status line, not the whole page
            return
        _CHAT_JOBS.pop(job_key, None)
        try:
            result = job["future"].result()
        except Exception:
            logger.exception("chat answer failed for %s", domain_id)
            st.error("Something went wrong answering that — please try again.")
            return
        if result["status"] != "valid":
            st.error(f"Validation errors: {result['errors']}")
            return
        data = result["data"]
        history.append({"speaker": "user", "text": query})
        history.append({"speaker": "assistant", "text": data.get("answer", "No answer returned.")})
        st.session_state[history_key] = history
        with st.chat_message("assistant"):
            st.markdown(md_safe(data.get("answer", "No answer returned.")))
            if str(data.get("source_document", "")).lower() != "none":   # "none" = off-topic question, nothing backed it
                render_citations(result.get("chunks"), doc)
            flagged = data.get("flag") or data.get("anomaly_flag")
            reason = data.get("flag_reason") or data.get("anomaly_reason")
            if flagged:
                st.warning(md_safe(f"⚠ {reason}"))
            if data.get("transactions"):   # empty for a regulation/policy answer -- no transaction card then
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Transactions", len(data["transactions"]))
                c2.metric("Total", f"${data['total_amount']:,.2f}" if data.get("total_amount") is not None else "—")
                c3.metric("Confidence", f"{data.get('confidence', 0) * 100:.0f}%")
                c4.metric("Source", data.get("source_document", "—"))
                st.dataframe(pd.DataFrame(data["transactions"]), use_container_width=True, hide_index=True)
            evaluation = result.get("evaluation")
            if evaluation:
                with st.expander("RAG evaluation"):
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Context Precision", evaluation["context_precision"] if evaluation["context_precision"] is not None else "N/A")
                    e2.metric("Faithfulness", evaluation["faithfulness"])
                    e3.metric("Answer Relevancy", evaluation["answer_relevancy"])


def guarded_score(pipeline, record: dict, context: str | None = None) -> dict | None:
    """pipeline.score_record() behind the per-session/daily rate limit
    (ui/guard.py) -- a manual-entry record can reach the LLM tier, which
    costs money. Returns None (after showing why) if over a limit."""
    if not allow("score"):
        return None
    return pipeline.score_record(record, context) if context is not None else pipeline.score_record(record)


@st.cache_resource
def _domain_classifier():
    from core.config_loader import ConfigLoader
    from core.domain_classifier_agent import DomainClassifierAgent
    return DomainClassifierAgent(ConfigLoader())


def _other_domain_hint(domain_id: str, pipeline, result: dict) -> dict | None:
    """For a document that got no fraud verdict: does its wording clearly
    belong to a different domain? Returns {"id","name"} of that domain, or
    None (also on any error -- this is only a courtesy hint)."""
    try:
        head = pipeline._document_head(result)
        if not head:
            return None
        guess = _domain_classifier().classify(head).best_guess
        if guess and guess.domain_id != domain_id and guess.score >= 0.05:
            return {"id": guess.domain_id, "name": guess.domain_name}
    except Exception:
        logger.warning("domain hint skipped", exc_info=True)
    return None


def attach_domain_hint(domain_id: str, pipeline, result: dict) -> None:
    """If an ingested document got no fraud verdict and clearly belongs to
    another domain, record that in result["fraud_summary"]["domain_hint"] so
    render_fraud_verdicts can point the user at the right workspace. Shared by
    the workspace upload and the landing page's confirm-and-ingest."""
    summary = result.get("fraud_summary")
    if summary is not None and not result.get("fraud_verdicts"):
        hint = _other_domain_hint(domain_id, pipeline, result)
        if hint:
            summary["domain_hint"] = hint


def render_document_upload(domain_id: str, pipeline, statements_dir: Path, doc_key: str, verdicts_key: str,
                            history_key: str, upload_caption: str) -> None:
    """Any-format ingestion (CSV/PDF/DOCX/TXT, core/rag/base_document_loader_agent.py)
    always runs both a fraud verdict AND RAG ingestion on the same
    upload, never gated by file type. One implementation shared by every
    domain's own "Upload X" tab.

    Accepts multiple files at once; each is ingested and scored
    independently (its own fraud verdict, tracked per filename in
    `verdicts_key`), and all ingested documents become jointly chattable
    via the same cross-document retrieval BaseRetrieverAgent already
    supports (source_document as a list — see
    docs/UNIFIED_INGESTION_VISION.md's cross-document retrieval section).
    `doc_key` holds the accumulated list of ingested filenames."""
    st.caption(upload_caption)
    available = sorted(p.stem for p in statements_dir.glob("*") if p.suffix.lower() in {".pdf", ".docx"}) if statements_dir.exists() else []
    uploaded_files = demo_file_uploader(
        "Upload document(s)", type=["csv", "pdf", "docx", "txt"], key=f"{domain_id}_upload",
        label_visibility="collapsed", accept_multiple_files=True,
    )

    docs: list[str] = st.session_state.setdefault(doc_key, [])
    per_doc: dict[str, dict] = st.session_state.setdefault(verdicts_key, {})

    def _ingest(path: str, filename: str):
        if not allow("ingest"):
            return
        bar = st.progress(0.0, text=f"Ingesting {filename}...")

        def _on_progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Scoring {filename}: {done:,} of {total:,} records")

        with st.spinner(f"Ingesting {filename}..."):
            try:
                result = pipeline.ingest(path, filename=filename, owner=get_owner(), progress=_on_progress)
            except DuplicateDocumentError as e:
                bar.empty()
                st.error(describe_ingest_error(e))
                return
            except Exception as e:  # never show a raw traceback for a bad/unsupported file
                bar.empty()
                logger.exception("ingest failed for %s", filename)
                st.error(f"{filename}: {describe_ingest_error(e)}")
                return
        bar.empty()
        stored_name = result["document"]
        if stored_name not in docs:
            docs.append(stored_name)
        attach_domain_hint(domain_id, pipeline, result)
        per_doc[stored_name] = {"verdicts": result.get("fraud_verdicts"), "summary": result.get("fraud_summary")}
        # Every document's verdict turns join the chat (each tagged with its
        # filename), appended so an already-started conversation is kept.
        st.session_state[history_key] = [
            *st.session_state.get(history_key, []), *result.get("seed_history", []),
        ]
        st.success(f"Stored {result['chunks_stored']} chunks for {display_name(stored_name)}")

    if available:
        demo = st.selectbox("...or pick a demo document", options=available, key=f"{domain_id}_demo_pick")
        if st.button("⊕ Ingest demo document", key=f"{domain_id}_ingest_demo"):
            demo_path = next(p for p in statements_dir.glob("*") if p.stem == demo)
            _ingest(str(demo_path), demo_path.name)

    if uploaded_files:
        label = f"⊕ Ingest {len(uploaded_files)} uploaded files" if len(uploaded_files) > 1 else "⊕ Ingest uploaded file"
        if st.button(label, key=f"{domain_id}_ingest_upload"):
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    _ingest(tmp_path, Path(uploaded_file.name).name)
                finally:
                    os.remove(tmp_path)

    if docs:
        st.divider()
        st.markdown("**Active document(s):** " + ", ".join(f"`{display_name(d)}`" for d in docs))
        for d in docs:
            info = per_doc.get(d, {})
            with st.expander(f"Fraud verdict — {display_name(d)}", expanded=len(docs) == 1):
                render_fraud_verdicts(info.get("verdicts"), info.get("summary"))


def render_workspace(domain_id: str, pipeline, upload_tab_label: str, statements_dir: Path,
                      upload_caption: str, manual_entry_fn) -> None:
    """The (C) workspace itself: 2 input tabs (full width) and a floating
    chat bubble/drawer that overlays the page rather than splitting it
    into columns. manual_entry_fn(pipeline) renders that domain's own
    manual-entry form and calls pipeline.score_record(). No header of its
    own -- the shared step-2 workspace header (+ WORKSPACE_SUBTITLES
    line) above already names the domain."""
    doc_key, verdicts_key = f"{domain_id}_doc", f"{domain_id}_verdicts"
    history_key = f"{domain_id}_history"

    tab1, tab2 = st.tabs([upload_tab_label, "✏️ Manual Entry"])
    with tab1:
        render_document_upload(domain_id, pipeline, statements_dir, doc_key, verdicts_key,
                                history_key, upload_caption)
    with tab2:
        manual_entry_fn(pipeline)

    render_floating_chat(domain_id, lambda: render_chat_panel(domain_id, pipeline, doc_key, history_key))

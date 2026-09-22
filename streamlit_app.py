# streamlit_app.py — Prism's single entry point.
# Flow: a command bar (paste/upload, or jump straight to a domain) ->
# DomainClassifierAgent guesses the domain -> a human confirms or
# overrides -> AgentOrchestrator hands back that domain's Pipeline ->
# that domain's workspace takes over. The confirm step is deliberate,
# not a formality — no pipeline ever runs on the classifier's guess
# alone, and the quick-pick chips below skip the guess, never the
# confirm (they set confirmed_domain directly, same trust level as
# picking from the dropdown after a guess).
#
# Every domain's workspace follows the same shape: tabs for how you
# provide data (upload / manual entry / image input, depending on the
# domain), a verdict card for the auto fraud check, and a floating chat
# bubble (not a tab) that opens a drawer for follow-up questions rather
# than uploading or ingesting anything itself. Every chat answer carries
# citation chips showing whether it drew on the uploaded document, the
# domain's static reference corpus, or both.

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator
from core.domain_classifier_agent import DomainClassifierAgent
from core.rag.base_document_loader_agent import DuplicateDocumentError
from core.rag.reference_corpus import REFERENCE_SOURCE_ID
from domains.banking.fraud.agents.detector_agent import FraudDetectorAgent
from domains.banking.fraud.pipeline import FraudPipeline

st.set_page_config(page_title="Prism", page_icon="◈", layout="wide")

DOMAIN_QUICK_PICKS = [
    ("banking", "🏦 Banking"),
    ("payroll_hr", "🧾 Payroll & HR"),
    ("financial_services", "💱 Financial Services"),
    ("insurance", "🩺 Insurance"),
]

# One line per domain, shown once under the shared step-2 workspace
# header (see the STEP 2 block below). Unnumbered on purpose: the landing
# page is never itself labeled "1 ·" anywhere, so a "2 ·" prefix here
# would read as an orphaned number rather than part of a visible flow.
WORKSPACE_SUBTITLES = {
    "banking": (
        "Transaction fraud detection — upload a document, enter one manually, or scan a "
        "receipt/cheque image — plus 💬 chat with anything you've ingested."
    ),
    "insurance": "Claim fraud detection + document chat — one pipeline, grounded in CMS Ch.26.",
    "financial_services": "Wallet/exchange fraud detection + document chat — one pipeline, grounded in FinCEN CVC guidance.",
    "payroll_hr": "Payroll/job-posting fraud detection + document chat — one pipeline, grounded in IRS Pub 15-T.",
}

# Small, already-curated demo files (the same ones each workspace's own
# "pick a demo document" selectbox reads server-side) offered here too, on
# the landing page's "...or try a sample file" picker — for an evaluator
# on the deployed app who has no bank statement/claim/payslip of their
# own, this stays inside the detect->confirm->run story (classify ->
# confirm/override -> auto-ingested) instead of a separate download step.
# (key, dropdown label, source path, filename used once ingested, its
# actual known-correct domain id — used to default the batch review
# table's override to the right answer even when filename-only
# classification guesses wrong; see render_batch_upload_section)
SAMPLE_FILES = [
    ("banking_pdf", "🏦 Bank statement (PDF)",
     "domains/banking/documents/data/statements/AC00003_statement.pdf", "AC00003_statement.pdf", "banking"),
    ("banking_csv", "🏦 Transactions (CSV)",
     "domains/banking/fraud/data/transactions_balanced.csv", "transactions_balanced.csv", "banking"),
    ("insurance_pdf", "🩺 Insurance claim (PDF)",
     "domains/insurance/data/statements/CLM001_claim.pdf", "CLM001_claim.pdf", "insurance"),
    ("finserv_pdf", "💱 Wallet statement (PDF)",
     "domains/financial_services/data/statements/WALLET001_statement.pdf", "WALLET001_statement.pdf", "financial_services"),
    ("payroll_pdf", "🧾 Payslip (PDF)",
     "domains/payroll_hr/payroll/data/statements/EMP001_payslip.pdf", "EMP001_payslip.pdf", "payroll_hr"),
]


# ══════════════════════════════════════════════════════════════════════
# SHARED RENDERERS — used by every domain's (C) workspace, Banking
# included. render_chat_panel's transactions/total_amount block is the
# one Banking-specific bit, folded in as a guarded `.get()` rather than a
# separate drawer implementation (see that function's own docstring).
# ══════════════════════════════════════════════════════════════════════


def render_fraud_verdicts(verdicts: list[dict] | None, summary: dict | None) -> None:
    """UnifiedDomainPipeline.ingest()'s result shape (core/unified_pipeline.py):
    every extracted record gets its own real verdict through the exact
    same rules->tiered ML->LLM engine the CSV fraud path uses, not one
    whole-document LLM-only flag. A domain whose document genuinely
    carries no fraud-schema fields (e.g. an HR handbook) gets an honest
    empty list here, not a forced verdict — see
    domains/payroll_hr/pipeline.py's _looks_like_payslip() for the real
    bug this distinction exists to prevent."""
    if verdicts is None:
        return
    if not verdicts:
        st.info("No recognizable fraud-schema records could be extracted from this document — chat is still available below.")
        return
    if summary:
        st.caption(f"Scored {summary['total_scored']} record(s) — {summary['flagged']} flagged, {summary['clean']} clean.")
    flagged = [v for v in verdicts if v["predicted"] == "FRAUD"]
    if flagged:
        st.error(f"⚠ {len(flagged)} record(s) flagged")
        for v in flagged:
            reason = v.get("reason", "")
            parsed = v.get("parsed") or {}
            tier = _classify_detection_tier(reason, parsed.get("action", ""))
            st.caption(f"— [{tier}] {reason}")
    else:
        st.success("✓ Nothing flagged — everything extracted looks consistent")

    # Banking Tier 1 only (the one schema this was measured as a rare,
    # meaningful signal rather than noise — see drift_detector.py) —
    # independent of the verdict above, so shown for flagged AND clean
    # records alike; a `.get()` no-op for every other domain, which never
    # produces this key at all.
    caveat_records = [v for v in verdicts if (v.get("parsed") or {}).get("caveat")]
    if caveat_records:
        with st.expander(
            f"🔍 {len(caveat_records)} record(s) also flagged as statistically unusual "
            "(unsupervised check, independent of the verdict above)"
        ):
            for v in caveat_records:
                st.caption((v.get("parsed") or {})["caveat"])


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


def render_explainability(flagged: bool, action: str, reason: str) -> None:
    """"WHY WAS THIS FLAGGED?" — names which of Rules/ML/LLM actually
    produced this verdict's reason, instead of presenting the pipeline
    as one opaque black box."""
    if not reason:
        return
    tier = _classify_detection_tier(reason, action)
    icon = {"Rule Engine": "📏", "ML Detector": "🧠", "LLM Reasoning": "💬"}.get(tier, "💬")
    title = "WHY WAS THIS FLAGGED?" if flagged else "WHY THIS LOOKS CLEAN"
    st.markdown(
        f"<div class='explain-panel'><div class='explain-title'>{title}</div>"
        f"<div class='explain-row'><span class='explain-tier'>{icon} {tier}</span>"
        f"<span class='explain-reason'>{reason}</span></div></div>",
        unsafe_allow_html=True,
    )


def render_verdict_result(result: dict) -> None:
    """Same {"predicted", "parsed": {...}} shape all 4 domains'
    score_record() returns for a single CSV/manual-entry record (see
    each domain's pipeline.py) — one renderer, reused everywhere a
    single record gets scored rather than a whole document's many."""
    predicted = result["predicted"]
    parsed = result["parsed"]
    reason = parsed.get("reason") or result.get("reason", "")
    risk_level = parsed.get("risk_level", "—")
    action = parsed.get("action", "—")
    flagged = predicted == "FRAUD"

    with st.container(key="analysis_card"):
        st.markdown("<div class='analysis-title'>PRISM Analysis</div>", unsafe_allow_html=True)
        if flagged:
            st.error(f"🚨 FRAUD — Risk: {risk_level}")
        else:
            st.success(f"✅ LEGITIMATE — Risk: {risk_level}")
        st.markdown(
            "<div class='pipeline-checklist'><span>✓ Domain Agent</span>"
            "<span>✓ Configuration Loaded</span><span>✓ Detection Engine</span>"
            "<span>✓ Explanation</span></div>",
            unsafe_allow_html=True,
        )
        render_explainability(flagged, action, reason)
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
    ref_corpus = extra.get("data", {}).get("reference_corpus")
    rows.append((
        "Knowledge Base",
        Path(ref_corpus).stem.replace("_", " ").title() if ref_corpus else "Document only",
    ))
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
            labels.append(f"📄 {src}")
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
            "<style>.block-container { padding-right: 480px !important; "
            "transition: padding-right 0.2s ease; }</style>",
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

    with st.container(key="chat_fab"):
        if st.button("✕" if is_open else "💬", key="chat_fab_btn"):
            st.session_state[chat_key] = not is_open
            st.rerun()

    if is_open:
        with st.container(key="chat_drawer"):
            title_col, max_col = st.columns([5, 1])
            with title_col:
                st.markdown("<div class='chat-drawer-title'>💬 Chat</div>", unsafe_allow_html=True)
            with max_col:
                with st.container(key="chat_max_btn_wrap"):
                    max_label = "🗗" if is_maximized else "⛶"
                    if st.button(max_label, key=f"{domain_id}_chat_max_btn",
                                 help="Restore" if is_maximized else "Maximize"):
                        st.session_state[max_key] = not is_maximized
                        st.rerun()
            render_drawer_content()


def render_chat_panel(domain_id: str, pipeline, doc_key: str, history_key: str) -> None:
    """The docked chat panel's contents — one implementation shared by
    every domain, Banking included. The transactions/total_amount block
    below only ever fires for Banking (the only domain whose reasoning
    agent's response schema has those fields, see
    domains/banking/documents/agents/reasoning_agent.py) — a no-op
    `.get()` for every other domain, not a Banking-specific branch."""
    doc = st.session_state.get(doc_key)
    history = st.session_state.get(history_key, [])
    for turn in history:
        with st.chat_message("user" if turn["speaker"] == "user" else "assistant"):
            st.markdown(turn["text"])

    query = st.chat_input("Ask about this document...", key=f"{domain_id}_chat_input")
    if not query:
        return
    if not doc:
        st.warning("Ingest a document first.")
        return
    with st.chat_message("user"):
        st.markdown(query)
    with st.spinner("Running retrieve → reason → validate..."):
        result = pipeline.run(query, source_document=doc, history=history)
    if result["status"] != "valid":
        st.error(f"Validation errors: {result['errors']}")
        return
    data = result["data"]
    history.append({"speaker": "user", "text": query})
    history.append({"speaker": "assistant", "text": data.get("answer", "No answer returned.")})
    st.session_state[history_key] = history
    with st.chat_message("assistant"):
        st.markdown(data.get("answer", "No answer returned."))
        render_citations(result.get("chunks"), doc)
        flagged = data.get("flag") or data.get("anomaly_flag")
        reason = data.get("flag_reason") or data.get("anomaly_reason")
        if flagged:
            st.warning(f"⚠ {reason}")
        if data.get("transactions") is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions", len(data.get("transactions", [])))
            c2.metric("Total", f"${data['total_amount']:,.2f}" if data.get("total_amount") is not None else "—")
            c3.metric("Confidence", f"{data.get('confidence', 0) * 100:.0f}%")
            c4.metric("Source", data.get("source_document", "—"))
            if data["transactions"]:
                st.dataframe(pd.DataFrame(data["transactions"]), use_container_width=True, hide_index=True)
        evaluation = result.get("evaluation")
        if evaluation:
            with st.expander("RAG evaluation"):
                e1, e2, e3 = st.columns(3)
                e1.metric("Context Precision", evaluation["context_precision"] if evaluation["context_precision"] is not None else "N/A")
                e2.metric("Faithfulness", evaluation["faithfulness"])
                e3.metric("Answer Relevancy", evaluation["answer_relevancy"])


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
    uploaded_files = st.file_uploader(
        "Upload document(s)", type=["csv", "pdf", "docx", "txt"], key=f"{domain_id}_upload",
        label_visibility="collapsed", accept_multiple_files=True,
    )

    docs: list[str] = st.session_state.setdefault(doc_key, [])
    per_doc: dict[str, dict] = st.session_state.setdefault(verdicts_key, {})

    def _ingest(path: str, filename: str):
        with st.spinner(f"Ingesting {filename}..."):
            try:
                result = pipeline.ingest(path, filename=filename)
            except DuplicateDocumentError as e:
                st.error(f"Duplicate upload — {e}")
                return
        stored_name = result["document"]
        if stored_name not in docs:
            docs.append(stored_name)
        per_doc[stored_name] = {"verdicts": result.get("fraud_verdicts"), "summary": result.get("fraud_summary")}
        if not st.session_state.get(history_key):
            # Only the first document in a batch seeds the chat — later
            # ones would otherwise clobber an already-started conversation.
            st.session_state[history_key] = result.get("seed_history", [])
        st.success(f"Stored {result['chunks_stored']} chunks for {stored_name}")

    if available:
        demo = st.selectbox("...or pick a demo document", options=available, key=f"{domain_id}_demo_pick")
        if st.button("⊕ Ingest demo document", key=f"{domain_id}_ingest_demo"):
            demo_path = next(p for p in statements_dir.glob("*") if p.stem == demo)
            _ingest(str(demo_path), demo)

    if uploaded_files:
        label = f"⊕ Ingest {len(uploaded_files)} uploaded files" if len(uploaded_files) > 1 else "⊕ Ingest uploaded file"
        if st.button(label, key=f"{domain_id}_ingest_upload"):
            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    _ingest(tmp_path, Path(uploaded_file.name).stem)
                finally:
                    os.remove(tmp_path)

    if docs:
        st.divider()
        st.markdown("**Active document(s):** " + ", ".join(f"`{d}`" for d in docs))
        for d in docs:
            info = per_doc.get(d, {})
            with st.expander(f"Fraud verdict — {d}", expanded=len(docs) == 1):
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


def render_batch_results_view(config_loader, orchestrator) -> None:
    """Step 2 for the batch-upload flow (top-level batch upload across
    *different* domains at once): every staged file was already ingested+scored in its own
    confirmed domain (see the batch-upload section on the landing page,
    below); this just groups the results by domain and lets the user drop
    into that domain's normal workspace to keep chatting, reusing the
    exact same per-domain session-state contract render_document_upload's
    _ingest() already writes (doc_key list + verdicts_key dict +
    history_key) rather than building a second, parallel results UI."""
    st.subheader("Batch results")
    st.caption("Each file below was ingested and scored in its own confirmed domain — open a workspace to keep chatting about any of them.")
    change_col, reset_col = st.columns([1, 1])
    with change_col:
        if st.button("← back to upload"):
            del st.session_state["batch_results"]
            st.session_state.pop("batch_errors", None)
            st.rerun()
    with reset_col:
        if st.button("↺ Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    st.divider()

    errors = st.session_state.get("batch_errors") or []
    if errors:
        with st.expander(f"⚠ {len(errors)} file(s) failed", expanded=True):
            for e in errors:
                st.error(e)

    results = st.session_state.get("batch_results") or {}
    if not results:
        st.info("No files were successfully ingested.")
        return

    for domain_id, bucket in results.items():
        domain_config = config_loader.get_domain(domain_id)
        with st.container(border=True):
            st.markdown(f"#### {domain_config.name}")
            st.caption(f"{len(bucket['docs'])} document(s) ingested")
            for stored_name in bucket["docs"]:
                info = bucket["verdicts"].get(stored_name, {})
                with st.expander(f"Fraud verdict — {stored_name}", expanded=len(bucket["docs"]) == 1):
                    render_fraud_verdicts(info.get("verdicts"), info.get("summary"))
            if st.button(f"💬 Open {domain_config.name} workspace to chat", key=f"open_batch_{domain_id}"):
                st.session_state[f"{domain_id}_doc"] = list(bucket["docs"])
                st.session_state[f"{domain_id}_verdicts"] = dict(bucket["verdicts"])
                st.session_state[f"{domain_id}_history"] = bucket["seed_history"] or []
                st.session_state["confirmed_domain"] = domain_id
                st.rerun()


def render_domain_quick_picks(config_loader, key_prefix: str) -> None:
    """"Already know the domain? Jump straight in" — skips classification
    (and, for the batch tab, skips batch mode entirely) and goes straight
    to that domain's own workspace, same trust level as picking from the
    dropdown after a guess. Shared by both the single-file and batch-upload
    tabs (see streamlit_app.py's module docstring) so this shortcut isn't
    single-file-only; `key_prefix` keeps each tab's button/container keys
    distinct since both tabs render in the same script run."""
    st.markdown(
        "<p style='text-align:center;color:#626977;font-size:12.5px;margin-top:18px;'>"
        "Already know the domain? Jump straight in:</p>",
        unsafe_allow_html=True,
    )
    runnable_ids = {d.id for d in config_loader.list_domains(runnable_only=True)}
    with st.container(key=f"{key_prefix}_chip_row"):
        chip_cols = st.columns(4)
        for col, (dom_id, label) in zip(chip_cols, DOMAIN_QUICK_PICKS):
            with col:
                if st.button(label, key=f"{key_prefix}_jump_{dom_id}", use_container_width=True, disabled=dom_id not in runnable_ids):
                    st.session_state["confirmed_domain"] = dom_id
                    st.rerun()


def _stage_batch_item(classifier, name: str, suffix: str, raw: bytes, known_domain_id: str | None = None) -> dict:
    """Classify one batch file's bytes into a staged-review row — shared by
    both the manual multi-uploader and the "try sample files" shortcut
    below, so a sample is staged/reviewed/overridden through the exact
    same path a real upload is, not a shortcut that skips the guess.
    `known_domain_id` is only ever set for a curated sample (we know its
    real domain because we picked it) — it defaults the override dropdown
    to the right answer without hiding what the classifier itself actually
    guessed (still shown as guess_id/guess_score), since filename-only
    classification on a PDF is honestly weak and 3-of-4 samples guessing
    "Banking" isn't a bug, just not a great first click for a one-button
    demo."""
    if suffix in {".pdf", ".docx"}:
        # Same "classify by filename for now" honesty as the single-file
        # flow above — full text is only read once a domain is confirmed
        # and the file is actually ingested.
        classify_text = f"document {name}"
    else:
        try:
            classify_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            classify_text = ""
    result = classifier.classify(classify_text)
    guess_id = result.best_guess.domain_id if result.best_guess else None
    return {
        "name": name, "suffix": suffix, "bytes": raw,
        "guess_id": guess_id,
        "guess_score": result.best_guess.score if result.best_guess else 0.0,
        "default_id": known_domain_id or guess_id,
        "is_known_sample": known_domain_id is not None,
    }


# One curated sample per domain (a subset of SAMPLE_FILES — skips
# banking_csv so the batch demo is "one file per domain", not two for
# banking) — this is what "✨ Try with sample files" below stages.
BATCH_SAMPLE_KEYS = ("banking_pdf", "insurance_pdf", "finserv_pdf", "payroll_pdf")


def render_batch_upload_section(classifier, orchestrator, config_loader) -> None:
    """The landing page's OTHER upload path — multiple files at once,
    each independently classified and (after human confirm/override, same
    trust rule as the single-file flow above) routed to its own domain's
    pipeline. Two-step, like the single-file flow: "Classify batch" stages
    guesses without running anything, "Confirm and run batch" is the
    actual human sign-off. Results are grouped by domain and handed to
    render_batch_results_view()."""
    with st.container():
        st.caption(
            "Drop files from different domains together — each gets its own domain guess, "
            "and nothing is ingested or scored until you confirm (or override) every guess below."
        )
        batch_files = st.file_uploader(
            "Upload multiple files", type=["csv", "pdf", "docx", "txt"],
            accept_multiple_files=True, key="batch_uploader",
        )
        if batch_files and st.button("Classify batch →", key="batch_classify_btn"):
            st.session_state["batch_staged"] = [
                _stage_batch_item(classifier, f.name, Path(f.name).suffix.lower(), f.getvalue())
                for f in batch_files
            ]

        st.caption("New here? Skip the upload — try one sample file per domain:")
        if st.button("✨ Try with sample files (one per domain)", key="batch_sample_btn"):
            samples_by_key = {
                key: (path_str, download_name, domain_id)
                for key, _, path_str, download_name, domain_id in SAMPLE_FILES
            }
            staged = []
            for key in BATCH_SAMPLE_KEYS:
                path_str, download_name, domain_id = samples_by_key[key]
                path = Path(path_str)
                if not path.exists():
                    continue
                staged.append(_stage_batch_item(
                    classifier, download_name, Path(download_name).suffix.lower(), path.read_bytes(),
                    known_domain_id=domain_id,
                ))
            st.session_state["batch_staged"] = staged

        render_domain_quick_picks(config_loader, key_prefix="batch")

        staged = st.session_state.get("batch_staged")
        if staged:
            st.markdown("**Confirm or override each file's domain before anything runs:**")
            runnable_list = [d.id for d in config_loader.list_domains(runnable_only=True)]
            chosen_ids = []
            for i, item in enumerate(staged):
                name_col, domain_col, score_col = st.columns([2, 2, 1])
                with name_col:
                    st.markdown(f"`{item['name']}`")
                with domain_col:
                    default_id = item.get("default_id") or item["guess_id"]
                    default_idx = runnable_list.index(default_id) if default_id in runnable_list else 0
                    chosen_ids.append(st.selectbox(
                        f"Domain for {item['name']}", options=runnable_list, index=default_idx,
                        format_func=lambda i: config_loader.get_domain(i).name,
                        key=f"batch_domain_{i}", label_visibility="collapsed",
                    ))
                with score_col:
                    if item.get("is_known_sample"):
                        # Not a classifier guess at all — this is one of
                        # the curated "try with sample files" picks, whose
                        # real domain we already know because we chose it
                        # (see BATCH_SAMPLE_KEYS) — say so plainly instead
                        # of leaving it looking like an ordinary, possibly
                        # wrong, filename-only guess.
                        st.caption("✓ known sample")
                    else:
                        # "guess_score" is a keyword-overlap fraction, not
                        # a calibrated probability -- "match %" says what
                        # it actually measures without overclaiming the
                        # precision "confidence" would imply.
                        st.caption(f"{item['guess_score']:.0%} match" if item["guess_id"] else "no match")

            if st.button("✅ Confirm and run batch", type="primary", key="batch_confirm_btn"):
                results: dict[str, dict] = {}
                errors: list[str] = []
                # Real per-file progress, not one silent spinner for the
                # whole batch — each document is a real LLM extraction +
                # RAG embed (slower still on a local Ollama backend, see
                # the "Running on local Ollama" banner), so a batch of
                # several files can genuinely take minutes; without this,
                # that looks indistinguishable from being stuck.
                with st.status(f"Ingesting {len(staged)} file(s) across domains...", expanded=True) as status:
                    for i, (item, domain_id) in enumerate(zip(staged, chosen_ids), start=1):
                        status.update(label=f"Ingesting file {i}/{len(staged)}: {item['name']} → {domain_id}...")
                        try:
                            pipeline = orchestrator.get_pipeline(domain_id)
                        except Exception as e:
                            errors.append(f"{item['name']}: could not load the '{domain_id}' pipeline — {e}")
                            st.write(f"✗ {item['name']} — could not load the '{domain_id}' pipeline")
                            continue
                        suffix = item["suffix"] or ".pdf"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(item["bytes"])
                            tmp_path = tmp.name
                        try:
                            ingest_result = pipeline.ingest(tmp_path, filename=Path(item["name"]).stem)
                        except DuplicateDocumentError as e:
                            errors.append(f"{item['name']}: duplicate upload — {e}")
                            st.write(f"✗ {item['name']} — duplicate, already ingested as '{e.existing}'")
                            continue
                        except Exception as e:
                            errors.append(f"{item['name']}: ingest failed — {e}")
                            st.write(f"✗ {item['name']} — ingest failed: {e}")
                            continue
                        finally:
                            os.remove(tmp_path)
                        st.write(f"✓ Ingested {item['name']} ({ingest_result['chunks_stored']} chunks)")
                        bucket = results.setdefault(domain_id, {"docs": [], "verdicts": {}, "seed_history": None})
                        stored_name = ingest_result["document"]
                        bucket["docs"].append(stored_name)
                        bucket["verdicts"][stored_name] = {
                            "verdicts": ingest_result.get("fraud_verdicts"),
                            "summary": ingest_result.get("fraud_summary"),
                        }
                        if bucket["seed_history"] is None:
                            # Only the first document per domain seeds that
                            # domain's chat — same rule render_document_upload's
                            # own _ingest() already follows for a single-domain
                            # multi-file upload, applied here per domain bucket.
                            bucket["seed_history"] = ingest_result.get("seed_history", [])
                    status.update(
                        label=f"Done — {len(staged) - len(errors)}/{len(staged)} file(s) ingested.",
                        state="error" if errors else "complete",
                    )
                st.session_state["batch_results"] = results
                st.session_state["batch_errors"] = errors
                del st.session_state["batch_staged"]
                st.rerun()


# ══════════════════════════════════════════════════════════════════════
# PER-DOMAIN MANUAL-ENTRY FORMS
# ══════════════════════════════════════════════════════════════════════


def insurance_manual_entry(pipeline) -> None:
    c1, c2 = st.columns(2)
    with c1:
        claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 500.0, 1.0, key="insurance_claim_amt")
        approved_amount = st.number_input("Approved Amount ($)", 0.0, 100000.0, 400.0, 1.0, key="insurance_approved_amt")
        service_date = st.date_input("Service Date", key="insurance_service_date")
        claim_date = st.date_input("Claim Date", key="insurance_claim_date")
    with c2:
        patient_age = st.number_input("Patient Age", 0, 120, 45, key="insurance_age")
        patient_gender = st.selectbox("Patient Gender", ["Male", "Female"], key="insurance_gender")
        insurance_type = st.selectbox("Insurance Type", ["Medicare", "Medicaid", "Private"], key="insurance_itype")
        visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "Emergency"], key="insurance_vtype")

    if st.button("🔍 Analyse This Claim", type="primary", key="insurance_manual_run"):
        record = {
            "Claim_Amount": claim_amount,
            "Approved_Amount": approved_amount,
            "Days_Between_Service_and_Claim": (claim_date - service_date).days,
            "Patient_Age": patient_age,
            "Patient_Gender": patient_gender,
            "Insurance_Type": insurance_type,
            "Visit_Type": visit_type,
        }
        with st.spinner("Running through hybrid detection pipeline..."):
            result = pipeline.score_record(record)
        st.divider()
        render_verdict_result(result)


def financial_services_manual_entry(pipeline) -> None:
    """Same record shape _map_extraction_to_records() produces from a
    document (see domains/financial_services/pipeline.py) -- the
    realistic, human-fillable fields, not either fraud tier's own
    REQUIRED_COLS (sender_wallet_age_days / round_trip_score /
    counterparty_reuse_ratio), which a document or a manual form can't
    honestly carry either. This always resolves to Tier 3, same
    documented honest limit as the document path."""
    c1, c2 = st.columns(2)
    with c1:
        amount_usd = st.number_input("Amount (USD)", 0.0, 1000000.0, 500.0, 1.0, key="finserv_amount")
        transaction_type = st.selectbox("Transaction Type", ["Send", "Receive", "Swap", "Bridge", "Trade"], key="finserv_type")
        blockchain = st.text_input("Blockchain", value="Ethereum", key="finserv_chain")
    with c2:
        counterparty = st.text_input("Counterparty (address or exchange name)", key="finserv_counterparty")
        is_cross_chain = st.checkbox("Cross-chain transfer", key="finserv_crosschain")

    if st.button("🔍 Analyse This Transaction", type="primary", key="finserv_manual_run"):
        record = {
            "transaction_amount_usd": amount_usd,
            "transaction_type": transaction_type,
            "blockchain": blockchain,
            "counterparty": counterparty,
            "is_cross_chain": is_cross_chain,
        }
        with st.spinner("Running through hybrid detection pipeline..."):
            result = pipeline.score_record(record)
        st.divider()
        render_verdict_result(result)


def payroll_hr_manual_entry(pipeline) -> None:
    """GROSS/Deduction/Net_Pay -- Tier 2's shape, the same one
    _map_extraction_to_records() prefers for a payslip document (see
    that method's docstring for why: it checks the actual compliance
    question a payslip represents, gross minus deductions equals net)."""
    c1, c2, c3 = st.columns(3)
    with c1:
        gross = st.number_input("Gross Pay ($)", 0.0, 100000.0, 3000.0, 1.0, key="payroll_gross")
    with c2:
        deduction = st.number_input("Total Deductions ($)", 0.0, 100000.0, 600.0, 1.0, key="payroll_deduction")
    with c3:
        net_pay = st.number_input("Net Pay ($)", 0.0, 100000.0, 2400.0, 1.0, key="payroll_net")

    if st.button("🔍 Analyse This Payslip", type="primary", key="payroll_manual_run"):
        record = {"GROSS": gross, "Deduction": deduction, "Net_Pay": net_pay}
        with st.spinner("Running through hybrid detection pipeline..."):
            result = pipeline.score_record(record)
        st.divider()
        render_verdict_result(result)


@st.cache_resource
def load_core():
    loader = ConfigLoader()
    orchestrator = AgentOrchestrator(loader)
    classifier = DomainClassifierAgent(loader)
    return loader, orchestrator, classifier


config_loader, orchestrator, classifier = load_core()

# ══════════════════════════════════════════════════════════════════════
# Top navbar — brand mark top-left, author top-right. Rendered once,
# above both step 1 (landing) and step 2 (workspace), so the branding
# doesn't reset/duplicate when a domain is confirmed.
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    .block-container { padding-top: 3.6rem !important; }

    .st-key-top_navbar {
        display: flex !important; flex-direction: row !important; justify-content: space-between !important;
        align-items: center !important; padding: 4px 0 16px; margin-bottom: 8px;
        border-bottom: 1px solid #e3e6ec;
    }
    .st-key-top_navbar > .stElementContainer { width: auto !important; flex: 0 0 auto !important; }
    .st-key-top_navbar .navbar-logo { display: flex; align-items: center; gap: 12px; }
    .st-key-top_navbar .navbar-mark {
        width: 40px; height: 40px; border-radius: 10px; background: #2f5fdb; color: #fff;
        display: flex; align-items: center; justify-content: center;
        line-height: 1; flex-shrink: 0;
    }
    .st-key-top_navbar .navbar-wordmark {
        font-size: 27px; font-weight: 800; letter-spacing: 0.04em; color: #2f5fdb; text-transform: uppercase;
    }
    .st-key-top_navbar .navbar-author { font-size: 16px; font-weight: 500; color: #626977; }

    /* Dismissible workspace notice banners (PayNexus-style) */
    .st-key-banner_info, .st-key-banner_warn {
        border-radius: 10px; padding: 12px 16px; margin-bottom: 14px;
    }
    .st-key-banner_info { background: #eef4fc; border: 1px solid #cfe0f7; }
    .st-key-banner_warn { background: #fff8e6; border: 1px solid #f3dfa0; }
    .banner-title { font-weight: 700; font-size: 13.5px; color: #191d24; }
    .banner-body { font-size: 12.5px; color: #626977; margin-top: 3px; }
    .st-key-banner_info .stButton>button, .st-key-banner_warn .stButton>button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        color: #8993a6 !important; padding: 0 !important; font-size: 14px !important;
    }

    /* Floating chat bubble + drawer (replaces the old docked side panel) */
    .st-key-chat_fab {
        position: fixed !important; bottom: 28px; right: 28px; z-index: 9999;
        width: 56px !important; flex: 0 0 auto !important;
    }
    .st-key-chat_fab .stButton>button {
        width: 56px; height: 56px; border-radius: 50%; background: #2f5fdb !important; color: #fff !important;
        font-size: 22px; box-shadow: 0 4px 14px rgba(20,24,32,.25); border: none !important;
    }
    .st-key-chat_drawer {
        position: fixed !important; top: 132px; bottom: 28px; right: 28px; z-index: 9998;
        width: 440px !important; max-width: calc(100vw - 40px); overflow-y: auto;
        background: #ffffff; border: 1px solid #e3e6ec; border-radius: 16px;
        box-shadow: 0 8px 30px rgba(20,24,32,.18); padding: 16px;
    }
    .chat-drawer-title { font-weight: 700; font-size: 14px; margin-bottom: 8px; padding-top: 4px; }
    .st-key-chat_max_btn_wrap .stButton>button {
        border: none !important; background: transparent !important; box-shadow: none !important;
        color: #8993a6 !important; padding: 0 !important; font-size: 16px !important;
    }
    .st-key-chat_max_btn_wrap .stButton>button:hover { color: #2f5fdb !important; }

    /* "What happens next" strip */
    .next-steps {
        display: flex; gap: 18px; flex-wrap: wrap; justify-content: center;
        font-size: 11.5px; color: #8993a6; margin-top: 16px; font-family: 'IBM Plex Mono', ui-monospace, monospace;
    }
    .next-steps span { white-space: nowrap; }

    /* Architecture flow diagram */
    .arch-flow { text-align: center; padding: 18px 10px; }
    .arch-flow .flow-node {
        display: inline-block; border: 1px solid #e3e6ec; border-radius: 10px;
        background: #f6f7f9; padding: 6px 16px; font-size: 12.5px; font-weight: 600; color: #191d24;
    }
    .arch-flow .flow-root { background: #2f5fdb; color: #fff; border-color: #2f5fdb; }
    .arch-flow .flow-result { background: #191d24; color: #fff; border-color: #191d24; }
    .arch-flow .flow-arrow { color: #b7bec9; font-size: 13px; margin: 4px 0; }
    .arch-flow .flow-detected { font-size: 12px; color: #626977; margin: 2px 0; }
    .arch-flow .flow-branches { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; }
    .arch-flow .flow-branch {
        border: 1px solid #cfe0f7; background: #eef4fc; border-radius: 10px;
        padding: 8px 14px; font-size: 12px; font-weight: 600; color: #191d24;
    }
    .arch-flow .flow-sub { font-size: 10.5px; font-weight: 500; color: #626977; }

    /* Config-driven panel — real values pulled from configs/<id>.yaml */
    .config-panel {
        border: 1px solid #e3e6ec; border-radius: 12px; background: #f6f7f9;
        padding: 14px 18px; height: 100%;
    }
    .config-panel-title {
        font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11px; font-weight: 700;
        letter-spacing: .06em; color: #8993a6; margin-bottom: 10px; text-transform: uppercase;
    }
    .config-row { display: flex; justify-content: space-between; gap: 12px; padding: 4px 0; font-size: 12.5px; border-bottom: 1px dashed #e3e6ec; }
    .config-row:last-child { border-bottom: none; }
    .config-key { color: #626977; }
    .config-val { color: #191d24; font-weight: 600; text-align: right; }

    /* Result "wow" card */
    .st-key-analysis_card {
        border: 1px solid #e3e6ec; border-radius: 14px; padding: 16px 18px; margin-top: 8px;
        background: linear-gradient(180deg, #fbfcfe 0%, #ffffff 100%);
    }
    .analysis-title {
        font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11px; font-weight: 700;
        letter-spacing: .06em; color: #8993a6; margin-bottom: 8px; text-transform: uppercase;
    }
    .pipeline-checklist {
        display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: #2f8a3e;
        margin: 8px 0 4px; font-family: 'IBM Plex Mono', ui-monospace, monospace;
    }

    /* Explainability breakdown */
    .explain-panel { margin-top: 10px; border-top: 1px solid #e3e6ec; padding-top: 10px; }
    .explain-title { font-size: 11px; font-weight: 700; color: #8993a6; letter-spacing: .04em; margin-bottom: 6px; }
    .explain-row { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
    .explain-tier {
        font-size: 11.5px; font-weight: 700; color: #2f5fdb; background: #eef4fc;
        border-radius: 6px; padding: 2px 8px; white-space: nowrap;
    }
    .explain-reason { font-size: 12.5px; color: #191d24; }

    .stButton>button { transition: all .15s ease; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRISM_ICON_SVG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<line x1="0" y1="13" x2="8.5" y2="13" stroke="#ffffff" stroke-width="1.7" stroke-linecap="round"/>
<polygon points="12,3 21,19 3,19" fill="#ffffff" fill-opacity="0.95"/>
<line x1="14.3" y1="13.6" x2="23.5" y2="7.5" stroke="#ff5c5c" stroke-width="1.4" stroke-linecap="round"/>
<line x1="14.9" y1="15.1" x2="24" y2="12.2" stroke="#ffb703" stroke-width="1.4" stroke-linecap="round"/>
<line x1="15.1" y1="16.7" x2="24" y2="17" stroke="#2ec4b6" stroke-width="1.4" stroke-linecap="round"/>
<line x1="14.6" y1="18.1" x2="22.5" y2="21.5" stroke="#3a86ff" stroke-width="1.4" stroke-linecap="round"/>
</svg>"""

with st.container(key="top_navbar"):
    st.markdown(
        f"<div class='navbar-logo'><div class='navbar-mark'>{PRISM_ICON_SVG}</div>"
        "<div class='navbar-wordmark'>Prism</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='navbar-author'>Ramya A</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Command bar: a centered icon mark, one search-bar-style
# input, and pill quick-pick chips — no sidebar domain list (every
# runnable domain is already one tap away via the chips below, and a
# permanent "working"/"coming soon" panel added nothing a chip doesn't
# already convey via being enabled/disabled).
# ══════════════════════════════════════════════════════════════════════

if "confirmed_domain" not in st.session_state and "batch_results" not in st.session_state:
    st.markdown(
        """
        <style>
        .st-key-cmd_input_box input {
            border-radius: 20px !important; text-align: center;
        }
        .st-key-cmd_input_box [data-testid="stFileUploaderDropzone"] { border-radius: 12px; }
        div[class*="st-key-"][class*="_chip_row"] .stButton>button {
            border-radius: 20px !important; border: 1px solid #e3e6ec !important; background: #ffffff !important;
            color: #191d24 !important; font-size: 12.5px !important; font-weight: 500 !important;
            padding: 7px 16px !important; box-shadow: none !important; white-space: nowrap !important;
        }
        div[class*="st-key-"][class*="_chip_row"] .stButton>button:hover { border-color: #2f5fdb !important; color: #2f5fdb !important; }
        /* Forcing all 4 chips to dead-equal width (min-width:0 + 100%
           button width) kept clipping "Financial Services" differently
           depending on the viewer's actual font metrics/viewport --
           fragile across browsers/OSes. Sized-to-content instead: each
           column shrinks to fit its own button (flex: 0 0 auto), the row
           centers them with a gap, so no button's text can ever be
           narrower than what it actually needs regardless of label
           length or where it renders. */
        div[class*="st-key-"][class*="_chip_row"] [data-testid="stHorizontalBlock"] {
            display: flex !important; flex-wrap: wrap !important; justify-content: center !important; gap: 10px !important;
        }
        div[class*="st-key-"][class*="_chip_row"] [data-testid="stColumn"] {
            flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
        }
        .st-key-cmd_input_box .stButton>button {
            border-radius: 20px !important; padding: 7px 20px !important;
        }
        .st-key-landing_hero { text-align: left !important; padding-top: 0; }
        .st-key-landing_hero .hero-tagline {
            font-size: 28px; font-weight: 700; color: #191d24; line-height: 1.25; max-width: 640px; margin: 0;
            text-align: left !important;
        }
        .st-key-landing_hero .hero-desc {
            font-size: 15.5px; font-weight: 500; color: #3d4451; max-width: 100%; margin: 12px 0 0; line-height: 1.6;
            text-align: left !important;
        }
        .st-key-landing_hero .hero-badges {
            display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-start; margin-top: 18px;
        }
        .st-key-landing_hero .hero-badge {
            font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11px; color: #626977;
            background: #f6f7f9; border: 1px solid #e3e6ec; border-radius: 20px; padding: 5px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="landing_hero"):
        st.markdown(
            "<div class='hero-tagline'>Intelligence that adapts to the domain.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-desc'>Config-driven multi-agent platform for fraud detection "
            "and document intelligence — Banking, Payroll &amp; HR, Financial Services, and Insurance.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-badges'>"
            "<span class='hero-badge'>Rules → ML → LLM</span>"
            "<span class='hero-badge'>RAG</span>"
            "<span class='hero-badge'>Multi-Agent</span>"
            "<span class='hero-badge'>Config-driven</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)

    # Short, parallel tab labels -- the fuller explanation each tab used to
    # carry in its own label ("...multiple files, multiple domains") now
    # lives as that tab's own intro caption instead, standard tab-UX
    # practice (a tab strip names the mode, doesn't explain it).
    tab_single, tab_batch = st.tabs(["📄 Quick Upload", "📦 Batch Upload"])

    with tab_single:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            with st.container(key="cmd_input_box"):
                text_input = st.text_input(
                    "Paste text",
                    placeholder="e.g. a claim description, a transaction note, a payslip line, a job posting…",
                    label_visibility="collapsed",
                )
                st.caption(
                    "Paste any snippet of real content — not a filename — and I'll match its wording "
                    "against each domain's own vocabulary to guess where it belongs."
                )
                uploaded = st.file_uploader(
                    "...or drop a file", type=["csv", "pdf", "docx", "txt"], label_visibility="collapsed",
                )
                upload_text = ""
                staged_bytes, staged_name = None, None
                if uploaded is not None:
                    staged_bytes, staged_name = uploaded.getvalue(), uploaded.name
                    if uploaded.name.lower().endswith((".pdf", ".docx")):
                        st.caption(f"{uploaded.name} — classifying by filename for now; full text is read once you confirm the domain and it's ingested.")
                        upload_text = f"document {uploaded.name}"
                    else:
                        try:
                            upload_text = staged_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            upload_text = ""
                else:
                    # No file of your own? Try one of ours — same small curated
                    # demo files each workspace's own "pick a demo document"
                    # reads server-side, offered here so a recruiter can stay
                    # inside the detect->confirm->run story instead of bouncing
                    # out to a download and back.
                    st.caption("New here? Try a sample file below — no upload needed:")
                    sample_labels = ["— none —"] + [label for _, label, _, _, _ in SAMPLE_FILES]
                    sample_choice = st.selectbox(
                        "...or try a sample file", options=sample_labels,
                        label_visibility="collapsed", key="landing_sample_pick",
                    )
                    if sample_choice != "— none —":
                        _, _, path_str, download_name, _ = next(s for s in SAMPLE_FILES if s[1] == sample_choice)
                        path = Path(path_str)
                        if path.exists():
                            staged_bytes, staged_name = path.read_bytes(), download_name
                            if download_name.lower().endswith((".pdf", ".docx")):
                                upload_text = f"document {download_name}"
                            else:
                                upload_text = staged_bytes.decode("utf-8", errors="ignore")

                _, btn_col, _ = st.columns([1, 1, 1])
                with btn_col:
                    classify_clicked = st.button("Analyze with PRISM →", type="primary", use_container_width=True)

            st.markdown(
                "<div class='next-steps'><span>① Detect domain</span><span>② Load configuration</span>"
                "<span>③ Run agents</span><span>④ Explain result</span></div>",
                unsafe_allow_html=True,
            )

            render_domain_quick_picks(config_loader, key_prefix="single")

            if classify_clicked:
                combined = f"{text_input}\n{upload_text}".strip()
                if not combined:
                    st.warning("Paste some text, upload a file, or pick a sample first.")
                else:
                    with st.status("Running PRISM analysis...", expanded=True) as status:
                        st.write("✓ File received" if staged_bytes is not None else "✓ Text received")
                        if staged_bytes is not None:
                            st.write("✓ Extracting content")
                        status.update(label="Identifying domain...")
                        result = classifier.classify(combined)
                        guess_name = result.best_guess.domain_name if result.best_guess else "no confident match"
                        st.write(f"✓ Domain identified — {guess_name}")
                        status.update(label="Analysis ready", state="complete")
                    st.session_state["classification_result"] = result
                    # Unified for a real upload OR a picked sample -- either way
                    # this is what "Confirm and continue" below actually
                    # ingests, so confirming doesn't dead-end into "now go
                    # re-upload the same file in the workspace."
                    st.session_state["staged_file"] = (
                        {"bytes": staged_bytes, "name": staged_name} if staged_bytes is not None else None
                    )
                    st.session_state["staged_text"] = text_input

            result = st.session_state.get("classification_result")
            if result and result.best_guess:
                st.divider()
                st.markdown("**Guess**")
                st.success(
                    f"Best guess: **{result.best_guess.domain_name}** "
                    f"(`{result.best_guess.domain_id}`) — {result.best_guess.score:.0%} match, method: {result.method}"
                )
                if result.llm_reasoning:
                    st.caption(f"LLM tiebreak reasoning: {result.llm_reasoning}")

                with st.expander("All domain scores"):
                    for g in result.all_scores:
                        st.markdown(f"- `{g.domain_id}` — {g.score:.0%} match ({', '.join(g.matched_keywords) or 'no keyword matches'})")

                guessed_config = config_loader.get_domain(result.best_guess.domain_id)
                flow_col, config_col = st.columns([3, 2], gap="medium")
                with flow_col:
                    render_pipeline_flow(guessed_config)
                with config_col:
                    render_config_panel(guessed_config)

                runnable_list = [d.id for d in config_loader.list_domains(runnable_only=True)]
                default_idx = (
                    runnable_list.index(result.best_guess.domain_id)
                    if result.best_guess.domain_id in runnable_list
                    else 0
                )
                st.markdown("**Confirm or override before anything runs:**")
                chosen = st.selectbox(
                    "Domain to run",
                    options=runnable_list,
                    index=default_idx,
                    format_func=lambda i: config_loader.get_domain(i).name,
                )
                if st.button("✅ Confirm and continue", type="primary"):
                    staged = st.session_state.get("staged_file")
                    if staged:
                        with st.spinner(f"Ingesting {staged['name']}..."):
                            try:
                                pipeline = orchestrator.get_pipeline(chosen)
                                suffix = Path(staged["name"]).suffix or ".pdf"
                                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                                    tmp.write(staged["bytes"])
                                    tmp_path = tmp.name
                                try:
                                    ingest_result = pipeline.ingest(tmp_path, filename=Path(staged["name"]).stem)
                                finally:
                                    os.remove(tmp_path)
                            except Exception as e:
                                st.session_state["staged_ingest_error"] = str(e)
                            else:
                                stored_name = ingest_result["document"]
                                st.session_state[f"{chosen}_doc"] = [stored_name]
                                st.session_state[f"{chosen}_verdicts"] = {
                                    stored_name: {
                                        "verdicts": ingest_result.get("fraud_verdicts"),
                                        "summary": ingest_result.get("fraud_summary"),
                                    }
                                }
                                st.session_state[f"{chosen}_history"] = ingest_result.get("seed_history", [])
                        del st.session_state["staged_file"]
                    st.session_state["confirmed_domain"] = chosen
                    st.rerun()

    with tab_batch:
        render_batch_upload_section(classifier, orchestrator, config_loader)

# ══════════════════════════════════════════════════════════════════════
# STEP 2b — batch-upload results, grouped by domain (see
# render_batch_upload_section above for how a file lands here)
# ══════════════════════════════════════════════════════════════════════

elif "batch_results" in st.session_state and "confirmed_domain" not in st.session_state:
    render_batch_results_view(config_loader, orchestrator)

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Domain-specific UI, only after human confirmation
# ══════════════════════════════════════════════════════════════════════

else:
    domain_id = st.session_state["confirmed_domain"]
    domain_config = config_loader.get_domain(domain_id)

    # Configuration panel + LLM-backend banner live in the sidebar, not
    # the main body, so the workspace's own tabs render immediately below
    # the header without scrolling. The sidebar is a structurally
    # separate element from Streamlit's own .block-container, so it's
    # also unaffected by render_floating_chat's padding-right reserve
    # when a chat drawer is open.
    with st.sidebar:
        render_config_panel(domain_config)
        _llm_provider = os.getenv("LLM_PROVIDER", "ollama" if os.getenv("ENV", "local") == "local" else "openai")
        if _llm_provider == "ollama":
            render_banner(
                "llm_backend", "info", "Running on local Ollama",
                f"Model: `{os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}` — document extraction and chat "
                "may be slower here than on the hosted OpenAI backend.",
            )
        elif not os.getenv("OPENAI_API_KEY"):
            render_banner(
                "llm_backend", "warn", "OPENAI_API_KEY is not set",
                f"LLM_PROVIDER is `{_llm_provider}` but no API key is configured — LLM-backed steps "
                "(extraction, chat, classification tiebreak) will fail until one is set.",
            )

    st.subheader(domain_config.name)
    st.caption(f"Running as `{domain_id}` — confirmed by you a moment ago.")
    if domain_id in WORKSPACE_SUBTITLES:
        st.caption(WORKSPACE_SUBTITLES[domain_id])
    change_col, reset_col = st.columns([1, 1])
    with change_col:
        if st.button("← change domain"):
            del st.session_state["confirmed_domain"]
            st.rerun()
    with reset_col:
        if st.button("↺ Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.get("staged_ingest_error"):
        st.error(f"Couldn't auto-ingest the file you staged in step 1: {st.session_state['staged_ingest_error']}")
        del st.session_state["staged_ingest_error"]

    st.divider()

    pipeline = orchestrator.get_pipeline(domain_id)

    # ── banking (fraud detection) UI ──────────────────────────────────
    if domain_id == "banking":
        tab1, tab2, tab3 = st.tabs(["📄 Upload Document", "✏️ Manual Entry", "🖼 Image Input"])

        demo_csv = Path("domains/banking/fraud/data/transactions_balanced.csv")
        df_full = pd.read_csv(demo_csv) if demo_csv.exists() else None
        context = (
            FraudPipeline.default_context(df_full["Amount"].mean(), df_full["Amount"].max())
            if df_full is not None
            else None
        )

        def show_pipeline_results(result: dict):
            predicted = result["predicted"]
            parsed = result["parsed"]
            flagged = predicted == "FRAUD"

            with st.container(key="analysis_card"):
                st.markdown("<div class='analysis-title'>PRISM Analysis</div>", unsafe_allow_html=True)
                if flagged:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")
                st.markdown(
                    "<div class='pipeline-checklist'><span>✓ Domain Agent</span>"
                    "<span>✓ Configuration Loaded</span><span>✓ Detection Engine</span>"
                    "<span>✓ Explanation</span></div>",
                    unsafe_allow_html=True,
                )
                render_explainability(flagged, parsed.get("action", ""), parsed.get("reason", ""))

            with st.expander("🔍 Detector Agent Report", expanded=True):
                st.text(f"Risk Level : {parsed['risk_level']}")
                st.text(f"Action     : {parsed['action']}")
                st.text(f"Reason     : {parsed['reason']}")

            if parsed.get("caveat"):
                # Tier 1 only, independent of the verdict above — see
                # drift_detector.py for why only this schema's measured
                # false-positive rate (4.5%) made this a meaningful signal
                # rather than noise.
                st.warning(f"🔍 {parsed['caveat']}")

            if predicted == "FRAUD":
                with st.expander("🔬 Analyst Agent Investigation"):
                    st.text(result["investigation"])
                with st.expander("🗺 Routing Decision"):
                    st.text(result["routing"])
                if result["alert"]:
                    with st.expander("🚨 Formal Bank Alert", expanded=True):
                        st.code(result["alert"])

        with tab1:
            # One upload feature, not two -- a CSV, PDF, or DOCX dropped
            # here always gets both a fraud verdict AND RAG chat-readiness
            # through the exact same UnifiedDomainPipeline.ingest() every
            # other domain's own "Upload X" tab already uses (loader.load()
            # dispatches by extension; a CSV's rows become both scoreable
            # records and chattable chunks, same as a PDF/DOCX's extracted
            # ones -- see core/rag/base_document_loader_agent.py).
            render_document_upload(
                "banking", pipeline, Path("domains/banking/documents/data/statements"),
                "banking_doc", "banking_verdicts", "banking_history",
                "Upload a bank statement or a transaction CSV — it's run through the same "
                "rules→ML→LLM fraud engine either way, and becomes chattable via the 💬 button below.",
            )

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                amount = st.number_input("Transaction Amount ($)", 0.0, 100000.0, 250.0, 0.01)
                hour = st.slider("Hour of transaction (0=midnight, 23=11pm)", 0, 23, 14)
            with col2:
                st.info(f"Amount: ${amount:.2f} · Hour: {hour}:00")
            if st.button("🔍 Analyse This Transaction", type="primary"):
                transaction = {"Amount": amount, "hour": hour, "Time": 50000}
                with st.spinner("Running through hybrid detection pipeline..."):
                    result = pipeline.score_record(transaction, context)
                st.divider()
                show_pipeline_results(result)

        with tab3:
            st.caption("Upload a JPG or PNG image — GPT-4o Vision extracts the transaction amount automatically.")
            uploaded_image = st.file_uploader("Upload receipt or cheque image", type=["jpg", "jpeg", "png", "webp"], key="fraud_img")
            if uploaded_image is not None:
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
                if st.button("🔍 Extract & Analyse Transaction", type="primary"):
                    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
                    ext = uploaded_image.name.split(".")[-1].lower()
                    with st.spinner("Extracting transaction details via GPT-4o Vision..."):
                        try:
                            transaction = pipeline.extract_from_image(uploaded_image.read(), mime_map.get(ext, "image/jpeg"))
                        except Exception as e:
                            st.error(f"Could not extract transaction from image: {e}")
                            st.stop()
                    with st.spinner("Running through hybrid detection pipeline..."):
                        result = pipeline.score_record(transaction, context)
                    show_pipeline_results(result)

        render_floating_chat("banking", lambda: render_chat_panel("banking", pipeline, "banking_doc", "banking_history"))

    # ── insurance (C workspace) ────────────────────────────────────────
    elif domain_id == "insurance":
        render_workspace(
            domain_id="insurance", pipeline=pipeline,
            upload_tab_label="📄 Upload Claim",
            statements_dir=Path("domains/insurance/data/statements"),
            upload_caption="Upload a claim or EOB (PDF/DOCX).",
            manual_entry_fn=insurance_manual_entry,
        )

    # ── financial services (C workspace) ────────────────────────────────
    elif domain_id == "financial_services":
        render_workspace(
            domain_id="financial_services", pipeline=pipeline,
            upload_tab_label="📄 Upload Statement",
            statements_dir=Path("domains/financial_services/data/statements"),
            upload_caption="Upload a wallet or exchange activity statement (PDF/DOCX).",
            manual_entry_fn=financial_services_manual_entry,
        )

    # ── payroll & hr (C workspace + HDBSCAN companion) ──────────────────
    elif domain_id == "payroll_hr":
        render_workspace(
            domain_id="payroll_hr", pipeline=pipeline,
            upload_tab_label="📄 Upload Document",
            statements_dir=Path("domains/payroll_hr/payroll/data/statements"),
            upload_caption=(
                "Upload a payslip, payroll register, or general HR document (e.g. an employee "
                "handbook, PDF/DOCX) — Prism detects which it is and only runs the fraud check "
                "when it's actually a payslip."
            ),
            manual_entry_fn=payroll_hr_manual_entry,
        )

        st.divider()
        with st.expander("🔍 Fraud Pattern Analysis (unsupervised, HDBSCAN)", expanded=False):
            st.markdown(
                "This runs **after** classification, on records payroll's CSV-register fraud "
                "detector already flagged — a different question from \"is this fraud?\": "
                "*are there distinct fraud sub-types, and does the model's flagged population "
                "actually have real structure, or is it one undifferentiated blob?*\n\n"
                "HDBSCAN clusters the flagged records using only the ratio of Base/Overtime/"
                "Other pay to total pay — **no fraud label is used at any point.** It blindly "
                "recovers exactly the 3 injected fraud patterns as clean, near-equal, "
                "zero-noise clusters — validating that the detector's flagged population isn't "
                "arbitrary."
            )
            if st.button("▶ Run cluster analysis", key="run_hdbscan"):
                from domains.payroll_hr.payroll.tools.fraud_pattern_clusterer import FraudPatternClusterer

                with st.spinner("Clustering flagged records..."):
                    tier1_train = pd.read_csv("domains/payroll_hr/payroll/data/tier_ml_train.csv")
                    flagged = tier1_train[tier1_train["Is_Flagged"] == 1]
                    clusterer = FraudPatternClusterer()
                    clustered = clusterer.cluster(flagged)
                    summary = clusterer.summarize(clustered)

                noise_pct = (clustered["cluster"] == -1).mean()
                st.metric("Clusters found", (summary["cluster"] != -1).sum())
                st.metric("Unclustered (ambiguous) records", f"{noise_pct:.1%}")

                display = summary[summary["cluster"] != -1].copy()
                display["Base %"] = (display["base_pct"] * 100).round(1)
                display["Overtime %"] = (display["ot_pct"] * 100).round(1)
                display["Other %"] = (display["other_pct"] * 100).round(1)
                st.dataframe(
                    display[["cluster", "size", "Base %", "Overtime %", "Other %"]],
                    use_container_width=True, hide_index=True,
                )

                st.scatter_chart(
                    clustered, x="ot_pct", y="other_pct", color="cluster", size=20,
                )
                st.caption(
                    "Each point is one flagged record, positioned by its overtime-pay and "
                    "other-pay share of total compensation. Three visually distinct groups "
                    "correspond to the three injected fraud patterns (base-only, overtime-heavy, "
                    "other-heavy) — discovered without ever telling the algorithm which pattern "
                    "each record came from."
                )

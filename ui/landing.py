# ui/landing.py — STEP 1: a centered icon mark, one search-bar-style
# input, and pill quick-pick chips — no sidebar domain list (every
# runnable domain is already one tap away via the chips below, and a
# permanent "working"/"coming soon" panel added nothing a chip doesn't
# already convey via being enabled/disabled). Split out of
# streamlit_app.py, which had grown to 1483 lines.

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from ui.constants import SAMPLE_FILES
from ui.guard import allow, describe_ingest_error, get_owner
from ui.shared import md_safe, render_pipeline_flow, render_config_panel
from ui.batch_upload import render_domain_quick_picks, render_batch_upload_section

logger = logging.getLogger(__name__)

LANDING_CSS = """
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
"""


def render_landing(config_loader, orchestrator, classifier) -> None:
    """The whole step-1 command bar: hero copy, the Quick Upload /
    Batch Upload tabs, classification, and the confirm-and-ingest step
    that hands off into a domain's own step-2 workspace."""
    st.markdown(LANDING_CSS, unsafe_allow_html=True)

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
                    st.caption(md_safe(f"LLM tiebreak reasoning: {result.llm_reasoning}"))

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
                                    if not allow("ingest"):
                                        raise RuntimeError("upload rate limit reached")
                                    ingest_result = pipeline.ingest(
                                        tmp_path, filename=Path(staged["name"]).name, owner=get_owner(),
                                    )
                                finally:
                                    os.remove(tmp_path)
                            except Exception as e:
                                logger.exception("landing ingest failed for %s", staged["name"])
                                st.session_state["staged_ingest_error"] = describe_ingest_error(e)
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

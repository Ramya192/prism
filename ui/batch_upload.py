# ui/batch_upload.py — the landing page's "batch upload across different
# domains" path, plus the quick-pick chips it shares with the single-file
# flow. Split out of streamlit_app.py, which had grown to 1483 lines.

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from core.rag.base_document_loader_agent import DuplicateDocumentError, display_name
from ui.guard import allow, describe_ingest_error, get_owner

from ui.constants import DOMAIN_QUICK_PICKS, SAMPLE_FILES
from ui.shared import render_fraud_verdicts

logger = logging.getLogger(__name__)


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
                with st.expander(f"Fraud verdict — {display_name(stored_name)}", expanded=len(bucket["docs"]) == 1):
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
    tabs (see ui/landing.py) so this shortcut isn't single-file-only;
    `key_prefix` keeps each tab's button/container keys distinct since both
    tabs render in the same script run."""
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
                        if not allow("ingest"):
                            errors.append(f"{item['name']}: skipped — upload rate limit reached")
                            continue
                        suffix = item["suffix"] or ".pdf"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(item["bytes"])
                            tmp_path = tmp.name
                        try:
                            ingest_result = pipeline.ingest(tmp_path, filename=Path(item["name"]).name, owner=get_owner())
                        except DuplicateDocumentError as e:
                            errors.append(f"{item['name']}: {describe_ingest_error(e)}")
                            st.write(f"✗ {item['name']} — duplicate, already ingested as '{display_name(e.existing)}'")
                            continue
                        except Exception as e:
                            logger.exception("batch ingest failed for %s", item["name"])
                            errors.append(f"{item['name']}: {describe_ingest_error(e)}")
                            st.write(f"✗ {item['name']} — {describe_ingest_error(e)}")
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

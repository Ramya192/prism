# streamlit_app.py — Prism's single entry point.
# Flow: upload/input -> DomainClassifierAgent guesses the domain -> a human
# confirms or overrides -> AgentOrchestrator hands back that domain's
# Pipeline -> that domain's own UI section (ported from the two old app.py
# files) takes over. The confirm step is deliberate, not a formality — no
# pipeline ever runs on the classifier's guess alone.

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator
from core.domain_classifier_agent import DomainClassifierAgent
from domains.bfsi_fraud.agents.detector_agent import FraudDetectorAgent
from domains.bfsi_fraud.pipeline import FraudPipeline

st.set_page_config(page_title="Prism", page_icon="◈", layout="wide")


@st.cache_resource
def load_core():
    loader = ConfigLoader()
    orchestrator = AgentOrchestrator(loader)
    classifier = DomainClassifierAgent(loader)
    return loader, orchestrator, classifier


config_loader, orchestrator, classifier = load_core()

st.title("◈ Prism")
st.caption("Domain-agnostic multi-agent intelligence platform — the domain is a YAML config, not hardcoded.")
st.divider()

with st.sidebar:
    st.header("Domains")
    for d in config_loader.list_domains():
        badge = "🟢 working" if d.is_runnable else "🚧 coming soon"
        st.markdown(f"**{d.name}**  \n`{d.id}` · {badge}")
    st.divider()
    if st.button("↺ Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Input + domain classification + human confirmation
# ══════════════════════════════════════════════════════════════════════

if "confirmed_domain" not in st.session_state:
    st.subheader("1 · Describe or upload what you're working with")
    st.caption(
        "Paste a snippet of the document/data, or upload a file — Prism guesses the "
        "domain, then you confirm before anything runs."
    )

    col1, col2 = st.columns(2)
    with col1:
        text_input = st.text_area(
            "Paste a snippet (transaction details, statement text, etc.)",
            height=150,
            placeholder="e.g. 'Suspicious $500 card transaction at 2am, chargeback flagged' "
            "or 'Bank Statement — Account ACC001, Closing Balance $1,200'",
        )
    with col2:
        uploaded = st.file_uploader("...or upload a file", type=["csv", "pdf", "txt"])
        upload_text = ""
        if uploaded is not None:
            if uploaded.type == "application/pdf":
                st.info("PDF detected — classifying by filename + declared type for now; full text is read after you confirm the domain and it's ingested.")
                upload_text = f"bank statement pdf document {uploaded.name}"
            else:
                raw = uploaded.getvalue()
                try:
                    upload_text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    upload_text = ""

    classify_clicked = st.button("🔎 Guess domain", type="primary")

    if classify_clicked:
        combined = f"{text_input}\n{upload_text}".strip()
        if not combined:
            st.warning("Paste some text or upload a file first.")
        else:
            result = classifier.classify(combined)
            st.session_state["classification_result"] = result
            st.session_state["staged_upload"] = uploaded
            st.session_state["staged_text"] = text_input

    result = st.session_state.get("classification_result")
    if result and result.best_guess:
        st.divider()
        st.markdown("**Guess**")
        st.success(
            f"Best guess: **{result.best_guess.domain_name}** "
            f"(`{result.best_guess.domain_id}`) — score {result.best_guess.score:.2f}, method: {result.method}"
        )
        if result.llm_reasoning:
            st.caption(f"LLM tiebreak reasoning: {result.llm_reasoning}")

        with st.expander("All domain scores"):
            for g in result.all_scores:
                st.markdown(f"- `{g.domain_id}` — {g.score:.2f} ({', '.join(g.matched_keywords) or 'no keyword matches'})")

        runnable_ids = [d.id for d in config_loader.list_domains(runnable_only=True)]
        default_idx = (
            runnable_ids.index(result.best_guess.domain_id)
            if result.best_guess.domain_id in runnable_ids
            else 0
        )
        st.markdown("**Confirm or override before anything runs:**")
        chosen = st.selectbox(
            "Domain to run",
            options=runnable_ids,
            index=default_idx,
            format_func=lambda i: config_loader.get_domain(i).name,
        )
        if st.button("✅ Confirm and continue", type="primary"):
            st.session_state["confirmed_domain"] = chosen
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Domain-specific UI, only after human confirmation
# ══════════════════════════════════════════════════════════════════════

else:
    domain_id = st.session_state["confirmed_domain"]
    domain_config = config_loader.get_domain(domain_id)
    st.subheader(f"2 · {domain_config.name}")
    st.caption(f"Running as `{domain_id}` — confirmed by you in step 1.")
    if st.button("← change domain"):
        del st.session_state["confirmed_domain"]
        st.rerun()
    st.divider()

    pipeline = orchestrator.get_pipeline(domain_id)

    # ── bfsi_fraud UI ────────────────────────────────────────────────
    if domain_id == "bfsi_fraud":
        tab1, tab2, tab3 = st.tabs(["📂 Upload CSV", "✏️ Manual Entry", "🖼 Image Input"])

        demo_csv = Path("domains/bfsi_fraud/data/transactions_balanced.csv")
        df_full = pd.read_csv(demo_csv) if demo_csv.exists() else None
        context = (
            FraudPipeline.default_context(df_full["Amount"].mean(), df_full["Amount"].max())
            if df_full is not None
            else None
        )

        def show_pipeline_results(result: dict):
            predicted = result["predicted"]
            parsed = result["parsed"]
            if predicted == "FRAUD":
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

            with st.expander("🔍 Detector Agent Report", expanded=True):
                st.text(f"Risk Level : {parsed['risk_level']}")
                st.text(f"Action     : {parsed['action']}")
                st.text(f"Reason     : {parsed['reason']}")

            if predicted == "FRAUD":
                with st.expander("🔬 Analyst Agent Investigation"):
                    st.text(result["investigation"])
                with st.expander("🗺 Routing Decision"):
                    st.text(result["routing"])
                if result["alert"]:
                    with st.expander("🚨 Formal Bank Alert", expanded=True):
                        st.code(result["alert"])

        with tab1:
            st.caption("Upload a CSV with columns: Amount, hour, Time (V1–V28 optional but recommended)")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="fraud_csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df)} transactions")
                max_txns = min(len(df), 20)
                num_txns = st.slider("Number of transactions to analyse", 1, max_txns, min(5, max_txns))
                if st.button("🔍 Analyse Transactions", type="primary"):
                    results = []
                    progress = st.progress(0)
                    for i, (_, row) in enumerate(df.head(num_txns).iterrows()):
                        transaction = row.to_dict()
                        progress.progress((i + 1) / num_txns)
                        result = pipeline.run(transaction, context)
                        actual = "FRAUD" if transaction.get("is_Fraud") == 1 else "Unknown"
                        results.append({
                            "Amount ($)": transaction.get("Amount", 0),
                            "Hour": int(transaction.get("hour", 0)),
                            "Predicted": result["predicted"],
                            "Actual": actual,
                            "Risk Level": result["parsed"]["risk_level"],
                            "Action": result["parsed"]["action"],
                        })
                    progress.empty()
                    st.dataframe(pd.DataFrame(results), use_container_width=True)

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
                    result = pipeline.run(transaction, context)
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
                        result = pipeline.run(transaction, context)
                    show_pipeline_results(result)

    # ── bfsi_documents UI ────────────────────────────────────────────
    elif domain_id == "bfsi_documents":
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown("**01 · Ingest a statement**")
            statements_dir = Path("domains/bfsi_documents/data/statements")
            available = sorted(p.stem for p in statements_dir.glob("*.pdf")) if statements_dir.exists() else []
            uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

            source = None
            if available:
                source = st.selectbox("...or pick a demo statement", options=available)
                if st.button("⊕ Ingest demo statement"):
                    pdf_path = statements_dir / f"{source}.pdf"
                    with st.spinner(f"Ingesting {source}..."):
                        result = pipeline.ingest(str(pdf_path), filename=source)
                    st.session_state["selected_doc"] = result["document"]
                    st.success(f"Stored {result['chunks_stored']} chunks for {result['document']}")

            if uploaded_pdf is not None and st.button("⊕ Ingest uploaded PDF"):
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    tmp_path = tmp.name
                try:
                    with st.spinner(f"Ingesting {uploaded_pdf.name}..."):
                        result = pipeline.ingest(tmp_path, filename=Path(uploaded_pdf.name).stem)
                    st.session_state["selected_doc"] = result["document"]
                    st.success(f"Stored {result['chunks_stored']} chunks for {result['document']}")
                finally:
                    os.remove(tmp_path)

            st.divider()
            st.markdown("**02 · Query**")
            query = st.text_area("Query", placeholder="What are the large withdrawals?", height=100)
            run_clicked = st.button("◈ Run Analysis", type="primary")

        with col_right:
            st.markdown("**03 · Results**")
            if run_clicked:
                doc = st.session_state.get("selected_doc")
                if not query.strip():
                    st.warning("Enter a query first.")
                elif not doc:
                    st.warning("Ingest a document first.")
                else:
                    with st.spinner("Running retrieve → reason → validate → evaluate..."):
                        result = pipeline.run(query, source_document=doc)
                    if result["status"] != "valid":
                        st.error(f"Validation errors: {result['errors']}")
                    else:
                        data = result["data"]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Transactions", len(data.get("transactions", [])))
                        c2.metric("Total", f"${data['total_amount']:,.2f}" if data.get("total_amount") is not None else "—")
                        c3.metric("Confidence", f"{data.get('confidence', 0) * 100:.0f}%")
                        c4.metric("Source", data.get("source_document", "—"))

                        st.info(data.get("answer", "No answer returned."))
                        if data.get("anomaly_flag"):
                            st.warning(f"⚠ Anomaly: {data.get('anomaly_reason', '')}")
                        else:
                            st.success("✓ No anomalies detected")

                        if data.get("transactions"):
                            st.dataframe(pd.DataFrame(data["transactions"]), use_container_width=True, hide_index=True)

                        evaluation = result.get("evaluation")
                        if evaluation:
                            st.markdown("**RAG Evaluation**")
                            e1, e2, e3 = st.columns(3)
                            e1.metric("Context Precision", evaluation["context_precision"] if evaluation["context_precision"] is not None else "N/A")
                            e2.metric("Faithfulness", evaluation["faithfulness"])
                            e3.metric("Answer Relevancy", evaluation["answer_relevancy"])

# ui/domains/banking.py — Banking's own step-2 workspace: 3 tabs (Upload
# Document / Manual Entry / Image Input), since Banking is the one domain
# with an image-input path (GPT-4o Vision) on top of the shared 2-tab
# render_workspace() every other domain uses. Split out of
# streamlit_app.py, which had grown to 1483 lines.

from pathlib import Path

import pandas as pd
import streamlit as st

from domains.banking.fraud.pipeline import FraudPipeline

from ui.guard import allow
from ui.shared import (
    guarded_score, md_safe, render_document_upload, render_chat_panel, render_floating_chat,
    render_explainability,
)


def render_banking_workspace(pipeline) -> None:
    tab1, tab2, tab3 = st.tabs(["📄 Upload Document", "✏️ Manual Entry", "🖼 Image Input"])

    demo_csv = Path("domains/banking/fraud/data/transactions_balanced.csv")
    df_full = pd.read_csv(demo_csv) if demo_csv.exists() else None
    context = (
        FraudPipeline.default_context(df_full["Amount"].mean(), df_full["Amount"].max())
        if df_full is not None
        else None
    )

    def show_pipeline_results(result: dict | None):
        if result is None:   # rate-limited -- guarded_score() already said why
            return
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
            st.warning(md_safe(f"🔍 {parsed['caveat']}"))

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
                result = guarded_score(pipeline, transaction, context)
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
                if not allow("score"):
                    st.stop()
                with st.spinner("Extracting transaction details via GPT-4o Vision..."):
                    try:
                        transaction = pipeline.extract_from_image(uploaded_image.read(), mime_map.get(ext, "image/jpeg"))
                    except Exception as e:
                        st.error(f"Could not extract transaction from image: {e}")
                        st.stop()
                with st.spinner("Running through hybrid detection pipeline..."):
                    result = guarded_score(pipeline, transaction, context)
                show_pipeline_results(result)

    render_floating_chat("banking", lambda: render_chat_panel("banking", pipeline, "banking_doc", "banking_history"))

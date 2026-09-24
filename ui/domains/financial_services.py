# ui/domains/financial_services.py — Financial Services' step-2 workspace
# (the shared 2-tab render_workspace()) + its manual-entry form. Split
# out of streamlit_app.py, which had grown to 1483 lines.

from pathlib import Path

import streamlit as st

from ui.shared import guarded_score, render_verdict_result, render_workspace


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
            result = guarded_score(pipeline, record)
        st.divider()
        render_verdict_result(result)


def render_financial_services_workspace(pipeline) -> None:
    render_workspace(
        domain_id="financial_services", pipeline=pipeline,
        upload_tab_label="📄 Upload Statement",
        statements_dir=Path("domains/financial_services/data/statements"),
        upload_caption="Upload a wallet or exchange activity statement (PDF/DOCX).",
        manual_entry_fn=financial_services_manual_entry,
    )

# ui/domains/insurance.py — Insurance's step-2 workspace (the shared
# 2-tab render_workspace()) + its manual-entry form. Split out of
# streamlit_app.py, which had grown to 1483 lines.

from pathlib import Path

import streamlit as st

from ui.shared import guarded_score, render_verdict_result, render_workspace


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
            result = guarded_score(pipeline, record)
        st.divider()
        render_verdict_result(result)


def render_insurance_workspace(pipeline) -> None:
    render_workspace(
        domain_id="insurance", pipeline=pipeline,
        upload_tab_label="📄 Upload Claim",
        statements_dir=Path("domains/insurance/data/statements"),
        upload_caption="Upload a claim or EOB (PDF/DOCX).",
        manual_entry_fn=insurance_manual_entry,
    )

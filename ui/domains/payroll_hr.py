# ui/domains/payroll_hr.py — Payroll & HR's step-2 workspace (the shared
# 2-tab render_workspace()) + its manual-entry form + the HDBSCAN fraud-
# pattern cluster analysis expander below it. Split out of
# streamlit_app.py, which had grown to 1483 lines.

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from ui.shared import guarded_score, render_verdict_result, render_workspace


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
            result = guarded_score(pipeline, record)
        st.divider()
        render_verdict_result(result)


def render_payroll_hr_workspace(pipeline) -> None:
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
                # Committed extract of tier_ml_train.csv's Is_Flagged==1 rows (the
                # full train CSV is gitignored, so it's absent on a fresh clone/EC2).
                flagged = pd.read_csv("domains/payroll_hr/payroll/data/tier_ml_flagged.csv")
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

            # Explicit axis domains: these shares are exactly 0 for whole clusters
            # (base-only sits at the origin), and the auto-scaled axes of
            # st.scatter_chart start above 0, which clipped every point out of view.
            # A stratified sample keeps the payload small (~50k rows otherwise).
            sample = clustered.groupby("cluster", group_keys=False).apply(
                lambda g: g.sample(min(len(g), 1000), random_state=0)
            )
            st.altair_chart(
                alt.Chart(sample).mark_circle(size=60, opacity=0.4).encode(
                    x=alt.X("ot_pct:Q", title="Overtime share of total pay",
                            scale=alt.Scale(domain=[-0.05, 1.0])),
                    y=alt.Y("other_pct:Q", title="Other-pay share of total pay",
                            scale=alt.Scale(domain=[-0.05, 1.0])),
                    color=alt.Color("cluster:N", title="cluster", scale=alt.Scale(scheme="category10")),
                ),
                use_container_width=True,
            )
            st.caption(
                "Each point is a flagged record (a sample of up to 1,000 per cluster is drawn), "
                "positioned by its overtime-pay and other-pay share of total compensation. "
                "Three distinct groups correspond to the three injected fraud patterns: "
                "base-only (all at the origin), overtime-heavy (along the x-axis) and "
                "other-heavy (along the y-axis) — discovered without ever telling the "
                "algorithm which pattern each record came from."
            )

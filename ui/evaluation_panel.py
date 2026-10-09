# ui/evaluation_panel.py -- a static "How well does it work?" panel.
#
# The numbers are the README's Evaluation table, measured by each domain's
# eval_harness.py against a temporal holdout at its real base rate, free
# deterministic layers only (rules + ML, no LLM calls). Static on purpose: the
# harnesses need the train/holdout CSVs, which are not in the deployed
# container. tests/test_evaluation_panel.py keeps this table and the README in
# sync, so edit both together (re-run the harness, then update here).

import pandas as pd
import streamlit as st

# (domain id, label, holdout rows, fraud base rate, F1 of the rules + ML layers)
EVAL_ROWS = [
    ("banking", "Banking Tier 1", 85443, "0.13%", 0.234),
    ("banking", "Banking Tier 2", 3000, "29.07%", 0.518),
    ("insurance", "Insurance Tier 1", 2697, "8.71%", 0.954),
    ("insurance", "Insurance Tier 2", 6030, "25.01%", 0.795),
    ("payroll_hr", "Payroll Tier 1", 44021, "50.11%", 0.993),
    ("payroll_hr", "Payroll Tier 2", 541, "50.46%", 0.966),
    ("payroll_hr", "HR job postings (single tier)", 5364, "4.77%", 0.774),
    ("financial_services", "Financial Services Tier 1", 4000, "7.25%", 0.493),
    ("financial_services", "Financial Services Tier 2", 100000, "5.95%", 0.470),
]

CAVEATS = (
    "Measured on a temporal holdout (train on the past, test on the future) at each dataset's "
    "real fraud rate, using only the free rules + ML layers. Rows they are unsure about go on to "
    "the LLM tier, which is evaluated separately on a small sample. A low F1 on a rare-fraud tier "
    "(Banking Tier 1: 0.13% fraud) is mostly a precision effect: recall there is 0.76, but so few "
    "rows are fraud that borderline legitimate ones dilute precision. Payroll's holdout is about "
    "50% fraud, so its F1 is not comparable with Banking's."
)


def render_evaluation_panel(domain_id: str | None = None) -> None:
    """An expander with the evaluation table: every domain on the landing page,
    just the open workspace's own tiers inside a workspace."""
    rows = [r for r in EVAL_ROWS if domain_id is None or r[0] == domain_id]
    if not rows:
        return
    with st.expander("📊 How well does it work?"):
        st.dataframe(
            pd.DataFrame(
                [{"Tier": label, "Holdout rows": f"{n:,}", "Fraud rate": rate, "F1": f1} for _, label, n, rate, f1 in rows]
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "F1": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0),
            },
        )
        st.caption(CAVEATS)

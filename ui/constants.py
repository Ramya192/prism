# ui/constants.py — cross-cutting constants shared across the landing
# page, batch upload, and the step-2 workspace header. Split out of
# streamlit_app.py (which had grown to 1483 lines) so each piece that
# reads these has a single import instead of relying on module-global
# state in the one entry-point script.

DOMAIN_QUICK_PICKS = [
    ("banking", "🏦 Banking"),
    ("payroll_hr", "🧾 Payroll & HR"),
    ("financial_services", "💱 Financial Services"),
    ("insurance", "🩺 Insurance"),
]

# One line per domain, shown once under the shared step-2 workspace
# header (see streamlit_app.py's STEP 2 block). Unnumbered on purpose: the
# landing page is never itself labeled "1 ·" anywhere, so a "2 ·" prefix
# here would read as an orphaned number rather than part of a visible flow.
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
# classification guesses wrong; see ui/batch_upload.py's _stage_batch_item)
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

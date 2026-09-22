# main.py — CLI smoke test for Prism.
# Exercises AgentOrchestrator end-to-end for all 4 domains (banking,
# payroll_hr, insurance, financial_services) against their small,
# committed demo data, driven through the config-driven orchestrator
# rather than importing agent classes directly.
#
# Usage (from the prism/ repo root):
#   python main.py
#
# Requires OPENAI_API_KEY (all domains) in .env — see .env.example.
# Banking's document chat also needs Ollama running locally
# (LLM_PROVIDER=ollama, the default) or LLM_PROVIDER=openai set in .env.

import sys
from pathlib import Path

import pandas as pd

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator
from domains.banking.fraud.pipeline import FraudPipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_domains(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("Prism — registered domains")
    print("=" * 60)
    for d in orchestrator.config_loader.list_domains():
        flag = "✅ working" if d.is_runnable else f"🚧 {d.status}"
        print(f"  {d.id:<16} {flag:<16} {d.name}")
    print()


def run_fraud_smoke_test(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("banking — smoke test (few demo transactions)")
    print("=" * 60)
    demo_csv = Path("domains/banking/fraud/data/transactions_balanced.csv")
    if not demo_csv.exists():
        print(f"  ⚠ {demo_csv} not found — skipping.")
        return

    df = pd.read_csv(demo_csv)
    sample = pd.concat(
        [df[df["is_Fraud"] == 1].head(2), df[df["is_Fraud"] == 0].head(2)]
    ).reset_index(drop=True)

    context = FraudPipeline.default_context(df["Amount"].mean(), df["Amount"].max())
    pipeline = orchestrator.get_pipeline("banking")

    for _, row in sample.iterrows():
        transaction = row.to_dict()
        actual = "FRAUD" if transaction["is_Fraud"] == 1 else "LEGITIMATE"
        result = pipeline.run(transaction, context)
        status = "✓" if result["predicted"] == actual else "✗"
        print(
            f"  {status} ${transaction['Amount']:>8.2f} hour={int(transaction['hour']):>2} "
            f"actual={actual:<11} predicted={result['predicted']}"
        )
    print()


def run_documents_smoke_test(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("banking — document chat smoke test (ingest + score + query one statement)")
    print("=" * 60)
    statements_dir = Path("domains/banking/documents/data/statements")
    pdfs = sorted(statements_dir.glob("*.pdf")) if statements_dir.exists() else []
    if not pdfs:
        print(f"  ⚠ No PDFs found in {statements_dir} — skipping.")
        return

    pdf_path = pdfs[0]
    pipeline = orchestrator.get_pipeline("banking")

    print(f"  Ingesting {pdf_path.name}...")
    ingest_result = pipeline.ingest(str(pdf_path), filename=pdf_path.stem)
    print(f"  ✓ Stored {ingest_result['chunks_stored']} chunks for {ingest_result['document']}")
    summary = ingest_result["fraud_summary"]
    print(f"  ✓ Scored {summary['total_scored']} transaction(s) through the real fraud engine "
          f"({summary['flagged']} flagged, {summary['clean']} clean)")

    query = "What is the closing balance?"
    print(f"  Querying: \"{query}\"")
    result = pipeline.run(query, source_document=pdf_path.stem)
    if result["status"] == "valid":
        print(f"  ✓ Answer: {result['data']['answer']}")
    else:
        print(f"  ✗ Validation errors: {result['errors']}")
    print()


def run_payroll_smoke_test(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("payroll_hr — document chat smoke test (ingest -> score -> follow-up chat)")
    print("=" * 60)
    statements_dir = Path("domains/payroll_hr/payroll/data/statements")
    pdfs = sorted(statements_dir.glob("*.pdf")) if statements_dir.exists() else []
    if not pdfs:
        print(f"  ⚠ No PDFs found in {statements_dir} — run "
              f"`python domains/payroll_hr/payroll/tools/generate_payroll_register.py` first. Skipping.")
        return

    pdf_path = pdfs[0]
    pipeline = orchestrator.get_pipeline("payroll_hr")

    print(f"  Ingesting {pdf_path.name}...")
    ingest_result = pipeline.ingest(str(pdf_path), filename=pdf_path.stem)
    print(f"  ✓ Stored {ingest_result['chunks_stored']} chunks for {ingest_result['document']}")

    verdicts = ingest_result["fraud_verdicts"]
    if verdicts:
        verdict = "⚠ FLAGGED" if verdicts[0]["predicted"] == "FRAUD" else "✓ clean"
        print(f"  ✓ Scored through the real fraud engine ({verdicts[0]['kind']}) — {verdict}")
    else:
        print(f"  ⚠ No recognizable fraud-schema fields extracted from this payslip")

    query = "What was the net pay?"
    print(f"  Follow-up query on the same ingested document: \"{query}\"")
    result = pipeline.run(query, source_document=pdf_path.stem)
    if result["status"] == "valid":
        print(f"  ✓ Answer: {result['data']['answer']}")
    else:
        print(f"  ✗ Validation errors: {result['errors']}")
    print()


def run_insurance_smoke_test(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("insurance — smoke test (one claim record, rules layer)")
    print("=" * 60)
    pipeline = orchestrator.get_pipeline("insurance")
    claim = {
        "Claim_Amount": 500.0, "Approved_Amount": 9999.0,
        "Days_Between_Service_and_Claim": 15,
    }
    result = pipeline.score_record(claim)
    status = "✓" if result["predicted"] == "FRAUD" else "✗"
    print(f"  {status} claim_amount=$500.00 approved=$9999.00 -> predicted={result['predicted']}")
    print()


def run_payroll_hr_smoke_test(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("payroll_hr — smoke test (content-routed: payroll record + job posting)")
    print("=" * 60)
    pipeline = orchestrator.get_pipeline("payroll_hr")

    payroll_record = {
        "BasePay": 5000.0, "OvertimePay": 200.0, "OtherPay": 0.0,
        "Stated_TotalPay": 9999.0,  # deliberately doesn't reconcile
    }
    result = pipeline.score_record(payroll_record)
    status = "✓" if result["kind"] == "payroll" and result["predicted"] == "FRAUD" else "✗"
    print(f"  {status} payroll-shaped record -> routed={result['kind']:<8} predicted={result['predicted']}")

    posting = {
        "title": "", "description": "x",
        "telecommuting": 0, "has_company_logo": 1, "has_questions": 0,
    }
    result = pipeline.score_record(posting)
    status = "✓" if result["kind"] == "hr" and result["predicted"] == "FRAUD" else "✗"
    print(f"  {status} posting-shaped record -> routed={result['kind']:<8} predicted={result['predicted']}")
    print()


def run_financial_services_smoke_test(orchestrator: AgentOrchestrator) -> None:
    print("=" * 60)
    print("financial_services — smoke test (tier1 + tier2 malformed records, rules layer)")
    print("=" * 60)
    pipeline = orchestrator.get_pipeline("financial_services")

    tier1_record = {
        "transaction_amount_usd": -100, "sender_wallet_age_days": 30, "is_cross_chain": 0,
    }
    result = pipeline.score_record(tier1_record)
    status = "✓" if result["predicted"] == "FRAUD" else "✗"
    print(f"  {status} tier1 (blockchain scam schema), negative amount -> predicted={result['predicted']}")

    tier2_record = {
        "notional_usd": -50, "round_trip_score": 0.5, "counterparty_reuse_ratio": 0.1,
    }
    result = pipeline.score_record(tier2_record)
    status = "✓" if result["predicted"] == "FRAUD" else "✗"
    print(f"  {status} tier2 (exchange/DEX schema), negative notional -> predicted={result['predicted']}")
    print()


def main():
    loader = ConfigLoader()
    orchestrator = AgentOrchestrator(loader)

    print_domains(orchestrator)

    try:
        run_fraud_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ banking smoke test failed: {e}\n")

    try:
        run_documents_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ banking document-chat smoke test failed: {e}\n")

    try:
        run_payroll_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ payroll smoke test failed: {e}\n")

    try:
        run_insurance_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ insurance smoke test failed: {e}\n")

    try:
        run_payroll_hr_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ payroll_hr smoke test failed: {e}\n")

    try:
        run_financial_services_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ financial_services smoke test failed: {e}\n")


if __name__ == "__main__":
    main()

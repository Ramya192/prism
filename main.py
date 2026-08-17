# main.py — CLI smoke test for Prism.
# Exercises AgentOrchestrator end-to-end for both working domains against
# their small, committed demo data — same spirit as the old fraud_detection
# main.py and document_intelligence_system main.py, just driven through the
# new config-driven orchestrator instead of importing agents directly.
#
# Usage (from the prism/ repo root):
#   python main.py
#
# Requires OPENAI_API_KEY (both domains) in .env — see .env.example.
# bfsi_documents also needs Ollama running locally (LLM_PROVIDER=ollama,
# the default) or LLM_PROVIDER=openai set in .env.

import sys
from pathlib import Path

import pandas as pd

from core.config_loader import ConfigLoader
from core.orchestrator import AgentOrchestrator
from domains.bfsi_fraud.pipeline import FraudPipeline

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
    print("bfsi_fraud — smoke test (few demo transactions)")
    print("=" * 60)
    demo_csv = Path("domains/bfsi_fraud/data/transactions_balanced.csv")
    if not demo_csv.exists():
        print(f"  ⚠ {demo_csv} not found — skipping.")
        return

    df = pd.read_csv(demo_csv)
    sample = pd.concat(
        [df[df["is_Fraud"] == 1].head(2), df[df["is_Fraud"] == 0].head(2)]
    ).reset_index(drop=True)

    context = FraudPipeline.default_context(df["Amount"].mean(), df["Amount"].max())
    pipeline = orchestrator.get_pipeline("bfsi_fraud")

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
    print("bfsi_documents — smoke test (ingest + query one statement)")
    print("=" * 60)
    statements_dir = Path("domains/bfsi_documents/data/statements")
    pdfs = sorted(statements_dir.glob("*.pdf")) if statements_dir.exists() else []
    if not pdfs:
        print(f"  ⚠ No PDFs found in {statements_dir} — skipping.")
        return

    pdf_path = pdfs[0]
    pipeline = orchestrator.get_pipeline("bfsi_documents")

    print(f"  Ingesting {pdf_path.name}...")
    ingest_result = pipeline.ingest(str(pdf_path), filename=pdf_path.stem)
    print(f"  ✓ Stored {ingest_result['chunks_stored']} chunks for {ingest_result['document']}")

    query = "What is the closing balance?"
    print(f"  Querying: \"{query}\"")
    result = pipeline.run(query, source_document=pdf_path.stem)
    if result["status"] == "valid":
        print(f"  ✓ Answer: {result['data']['answer']}")
    else:
        print(f"  ✗ Validation errors: {result['errors']}")
    print()


def main():
    loader = ConfigLoader()
    orchestrator = AgentOrchestrator(loader)

    print_domains(orchestrator)

    try:
        run_fraud_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ bfsi_fraud smoke test failed: {e}\n")

    try:
        run_documents_smoke_test(orchestrator)
    except Exception as e:
        print(f"  ✗ bfsi_documents smoke test failed: {e}\n")


if __name__ == "__main__":
    main()

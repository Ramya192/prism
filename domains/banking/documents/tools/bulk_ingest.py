"""
domains/banking/documents/tools/bulk_ingest.py
Ported from document_intelligence_system/tools/bulk_ingest.py — logic
unchanged, paths updated. Bulk-ingests statement PDFs directly through
DocumentLoaderAgent (in-process — no FastAPI hop needed for Prism's
Streamlit-only deployment, unlike the original which called a running
uvicorn API).

Usage (from the prism/ repo root):
    python domains/banking/documents/tools/bulk_ingest.py

Optional: ingest a specific range
    python domains/banking/documents/tools/bulk_ingest.py --start 50 --end 100
"""

import sys
import argparse
from pathlib import Path

# Console output uses emoji markers; on Windows the default console codepage
# (cp1252) can't encode them and print() raises UnicodeEncodeError. Force
# UTF-8 so this runs out of the box in a plain Windows terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

STATEMENTS_DIR = Path(__file__).parent.parent / "data" / "statements"


def main(start: int = 0, end: int = None):
    from domains.banking.documents.agents.document_loader_agent import DocumentLoaderAgent

    pdfs = sorted(STATEMENTS_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"❌ No PDFs found in {STATEMENTS_DIR}")
        return

    # Apply range slice if specified
    pdfs = pdfs[start:end]

    total = len(pdfs)
    print("=" * 60)
    print("Prism (banking documents) — Bulk Ingest Script")
    print("=" * 60)
    print(f"📁 Source:  {STATEMENTS_DIR}")
    print(f"📄 Files:   {total} PDFs")
    print("=" * 60)

    loader = DocumentLoaderAgent()
    success = 0
    failed = 0
    failed_files = []

    for i, pdf_path in enumerate(pdfs, 1):
        try:
            loader.load(str(pdf_path), filename=pdf_path.stem)
            print(f"  ✅ [{i:>3}/{total}] {pdf_path.name}")
            success += 1
        except Exception as e:
            print(f"  ❌ [{i:>3}/{total}] {pdf_path.name} — {e}")
            failed += 1
            failed_files.append(pdf_path.name)

    print("\n" + "=" * 60)
    print(f"✅ Ingested: {success}/{total}")
    if failed:
        print(f"❌ Failed:   {failed}")
        print("   Failed files:")
        for f in failed_files:
            print(f"   - {f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk ingest PDFs into Prism's banking domain")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()
    main(start=args.start, end=args.end)

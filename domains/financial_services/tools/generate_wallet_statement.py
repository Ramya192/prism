"""
domains/financial_services/tools/generate_wallet_statement.py
Modeled on domains/banking/documents/tools/generate_statement.py's
reportlab approach -- same visual/document-generation pattern. Lists
several transactions per statement (like a bank statement, not a single
artifact like Insurance's claim), exercising the "score every extracted
transaction independently" path in core/unified_pipeline.py.

Usage (from the prism/ repo root):
    python domains/financial_services/tools/generate_wallet_statement.py

Output:
    domains/financial_services/data/statements/WALLET001_statement.pdf
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

_HERE = Path(__file__).parent.parent  # domains/financial_services/
OUTPUT_DIR = _HERE / "data" / "statements"

WALLET_ADDRESS = "0x8f3a...c92e"

TRANSACTIONS = [
    {"date": "2026-02-01", "amount_usd": 250.00, "transaction_type": "Receive", "blockchain": "Ethereum", "counterparty": "0x1a2b...9f0e", "is_cross_chain": False},
    {"date": "2026-02-03", "amount_usd": 1200.00, "transaction_type": "Swap", "blockchain": "Ethereum", "counterparty": "Uniswap V3", "is_cross_chain": False},
    {"date": "2026-02-07", "amount_usd": 800.00, "transaction_type": "Bridge", "blockchain": "Ethereum -> Polygon", "counterparty": "Polygon Bridge", "is_cross_chain": True},
    {"date": "2026-02-10", "amount_usd": 45.00, "transaction_type": "Send", "blockchain": "Polygon", "counterparty": "0x7c4d...2b1a", "is_cross_chain": False},
    {"date": "2026-02-14", "amount_usd": 3000.00, "transaction_type": "Trade", "blockchain": "Ethereum", "counterparty": "Coinbase", "is_cross_chain": False},
]


def generate_pdf(output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=15 * mm, leftMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
    )

    navy = colors.HexColor("#0a1628")
    gold = colors.HexColor("#c9a84c")
    light_gray = colors.HexColor("#f5f5f5")
    mid_gray = colors.HexColor("#e0e0e0")
    dark_gray = colors.HexColor("#333333")

    title_style = ParagraphStyle("Title", fontSize=18, fontName="Helvetica-Bold", textColor=navy, alignment=TA_CENTER, spaceAfter=4)
    sub_style = ParagraphStyle("Sub", fontSize=8, fontName="Helvetica", textColor=colors.HexColor("#666666"), alignment=TA_CENTER)
    normal_style = ParagraphStyle("Normal2", fontSize=8, fontName="Helvetica", textColor=dark_gray)

    elements = [
        Spacer(1, 2 * mm),
        Paragraph("Wallet Activity Statement", title_style),
        Paragraph(f"Wallet: {WALLET_ADDRESS}", sub_style),
        Paragraph("Statement period: Feb 1 – Feb 14, 2026", sub_style),
        Spacer(1, 4 * mm),
        HRFlowable(width="100%", thickness=2, color=navy),
        Spacer(1, 4 * mm),
    ]

    rows = [["Date", "Type", "Amount (USD)", "Blockchain", "Counterparty"]]
    for t in TRANSACTIONS:
        rows.append([
            t["date"], t["transaction_type"], f"${t['amount_usd']:,.2f}",
            t["blockchain"], t["counterparty"],
        ])
    table = Table(rows, colWidths=[25 * mm, 22 * mm, 28 * mm, 45 * mm, 40 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), gold),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, mid_gray), ("BACKGROUND", (0, 1), (-1, -1), light_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements += [table, Spacer(1, 6 * mm)]

    total = sum(t["amount_usd"] for t in TRANSACTIONS)
    elements.append(Paragraph(f"Total activity: ${total:,.2f}", normal_style))

    elements.append(Spacer(1, 6 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=mid_gray))
    footer_style = ParagraphStyle("Footer", fontSize=6.5, fontName="Helvetica", textColor=colors.HexColor("#999999"), alignment=TA_CENTER)
    elements.append(Paragraph(f"Computer-generated activity statement for {WALLET_ADDRESS}.", footer_style))
    elements.append(Paragraph("Prism — Domain-Agnostic Multi-Agent Intelligence Platform · financial_services domain", footer_style))
    doc.build(elements)


def main():
    print("=" * 60)
    print("Prism (financial_services) — Wallet Statement Generator")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "WALLET001_statement.pdf"
    generate_pdf(output_path)
    print(f"   WALLET001_statement.pdf — {len(TRANSACTIONS)} transactions "
          f"(1 cross-chain bridge)")
    print(f"\n✅ Generated 1 wallet statement")
    print(f"📁 Saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

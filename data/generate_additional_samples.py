# data/generate_additional_samples.py
# Generates a SECOND, distinct set of sample documents for each domain --
# one PDF, one DOCX, and one CSV per domain -- deliberately different
# content (new account/claim/wallet/employee IDs, different transactions
# and figures) from the existing UI samples already wired into
# streamlit_app.py's SAMPLE_FILES (AC00003_statement.pdf, CLM001_claim.pdf,
# WALLET001_statement.pdf, EMP001_payslip.pdf, transactions_balanced.csv).
#
# Purpose: give someone testing the app outside the UI (or wanting a
# second, independent sample per domain/format) real files to drop into
# the upload widgets, without reusing exactly what the UI's own "try a
# sample" pickers already offer.
#
# The PDF/DOCX pair per domain is the same content in both formats (proves
# both loader paths -- core/rag/base_document_loader_agent.py dispatches
# by extension); the CSV is a separate, multi-row bulk file in that
# domain's real Tier 1 schema (see core/domain_classifier_agent.py's
# CSV_SCHEMA_HINTS), not just the PDF/DOCX content reformatted.
#
# Usage (from the prism/ repo root):
#     python data/generate_additional_samples.py                    # writes into data/<domain>/
#     python data/generate_additional_samples.py --out data/set2    # writes into data/set2/<domain>/ instead
#
# Output: <out>/<domain>/<id>_*.pdf, .docx, and a *_bulk.csv per domain.
# Deterministic (fixed random seeds per domain) -- rerunning with the same
# --out reproduces byte-for-byte the same files; a different --out just
# relocates the same content rather than generating new content, since
# nothing about the actual synthetic data depends on the output path.

import argparse
import csv
import random
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

from docx import Document

HERE = Path(__file__).parent

NAVY = colors.HexColor("#0a1628")
LIGHT_GRAY = colors.HexColor("#f5f5f5")
MID_GRAY = colors.HexColor("#e0e0e0")
DARK_GRAY = colors.HexColor("#333333")


def _write_pdf(output_path: Path, title: str, subtitle_lines: list[str],
               header_row: list[str], rows: list[list[str]], col_widths: list[float],
               footer_lines: list[str] | None = None) -> None:
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=15 * mm, leftMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
    )
    title_style = ParagraphStyle("Title", fontSize=17, fontName="Helvetica-Bold", textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
    sub_style = ParagraphStyle("Sub", fontSize=9, fontName="Helvetica", textColor=colors.HexColor("#666666"), alignment=TA_CENTER)
    normal_style = ParagraphStyle("Normal2", fontSize=9, fontName="Helvetica", textColor=DARK_GRAY)

    elements = [Spacer(1, 2 * mm), Paragraph(title, title_style)]
    for line in subtitle_lines:
        elements.append(Paragraph(line, sub_style))
    elements += [Spacer(1, 4 * mm), HRFlowable(width="100%", thickness=2, color=NAVY), Spacer(1, 4 * mm)]

    table = Table([header_row] + rows, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, MID_GRAY), ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GRAY),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(table)

    if footer_lines:
        elements.append(Spacer(1, 6 * mm))
        for line in footer_lines:
            elements.append(Paragraph(line, normal_style))

    doc.build(elements)


def _write_docx(output_path: Path, title: str, subtitle_lines: list[str],
                 header_row: list[str], rows: list[list[str]],
                 footer_lines: list[str] | None = None) -> None:
    d = Document()
    d.add_heading(title, level=1)
    for line in subtitle_lines:
        d.add_paragraph(line)

    table = d.add_table(rows=1, cols=len(header_row))
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(header_row):
        table.rows[0].cells[i].text = h
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    if footer_lines:
        d.add_paragraph("")
        for line in footer_lines:
            d.add_paragraph(line)

    d.save(str(output_path))


def _write_csv(output_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════
# Banking — new account AC00777 (existing UI sample is AC00003)
# ══════════════════════════════════════════════════════════════════════

def banking_samples(out_dir: Path) -> None:
    txns = [
        ["2026-03-02", "Salary credit - Initech LLC", "+4,200.00", "4,200.00"],
        ["2026-03-05", "ATM withdrawal", "-300.00", "3,900.00"],
        ["2026-03-09", "UPI to Grocery Mart", "-86.40", "3,813.60"],
        ["2026-03-15", "Zelle received - J. Rivera", "+150.00", "3,963.60"],
        ["2026-03-22", "Electric utility bill", "-92.15", "3,871.45"],
        ["2026-03-28", "Rent payment", "-1,450.00", "2,421.45"],
    ]
    header = ["Date", "Description", "Amount", "Balance"]
    subtitle = ["Account: AC00777 — First National Trust Bank", "Statement period: Mar 1 – Mar 31, 2026"]
    _write_pdf(out_dir / "AC00777_statement.pdf", "Bank Statement", subtitle, header, txns,
               [26 * mm, 68 * mm, 26 * mm, 26 * mm])
    _write_docx(out_dir / "AC00777_statement.docx", "Bank Statement", subtitle, header, txns)

    rng = random.Random(777)
    fieldnames = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "hour"]
    rows = []
    for _ in range(6):
        row = {f"V{j}": round(rng.uniform(-2.5, 2.5), 4) for j in range(1, 29)}
        row["Amount"] = round(rng.uniform(5, 900), 2)
        row["Time"] = rng.randint(0, 172792)
        row["hour"] = rng.randint(0, 23)
        rows.append(row)
    _write_csv(out_dir / "AC00777_transactions_bulk.csv", fieldnames, rows)


# ══════════════════════════════════════════════════════════════════════
# Insurance — new claim CLM900 (existing UI sample is CLM001)
# ══════════════════════════════════════════════════════════════════════

def insurance_samples(out_dir: Path) -> None:
    header = ["Field", "Value"]
    rows = [
        ["Claim Number", "CLM900"],
        ["Patient", "D. Okafor"],
        ["Provider Specialty", "Orthopedics"],
        ["Visit Type", "Outpatient"],
        ["Insurance Type", "Private"],
        ["Service Date", "2026-02-18"],
        ["Claim Date", "2026-02-24"],
        ["Claim Amount", "$3,450.00"],
        ["Approved Amount", "$3,100.00"],
        ["Number of Procedures", "2"],
    ]
    subtitle = ["Meridian Health Plan", "500 Claims Way, Hartford, CT 06103"]
    _write_pdf(out_dir / "CLM900_claim.pdf", "Explanation of Benefits", subtitle, header, rows,
               [55 * mm, 55 * mm])
    _write_docx(out_dir / "CLM900_claim.docx", "Explanation of Benefits", subtitle, header, rows)

    fieldnames = [
        "Claim_Amount", "Approved_Amount", "Days_Between_Service_and_Claim",
        "Provider_Specialty", "Visit_Type", "Insurance_Type", "Number_of_Procedures",
        "Number_of_Previous_Claims_Patient", "Number_of_Previous_Claims_Provider", "Claim_Submitted_Late",
    ]
    csv_rows = [
        {"Claim_Amount": 1200, "Approved_Amount": 1100, "Days_Between_Service_and_Claim": 12,
         "Provider_Specialty": "Cardiology", "Visit_Type": "Outpatient", "Insurance_Type": "Medicare",
         "Number_of_Procedures": 1, "Number_of_Previous_Claims_Patient": 2,
         "Number_of_Previous_Claims_Provider": 340, "Claim_Submitted_Late": 0},
        {"Claim_Amount": 8900, "Approved_Amount": 8900, "Days_Between_Service_and_Claim": 1,
         "Provider_Specialty": "Neurology", "Visit_Type": "Inpatient", "Insurance_Type": "Private",
         "Number_of_Procedures": 4, "Number_of_Previous_Claims_Patient": 0,
         "Number_of_Previous_Claims_Provider": 210, "Claim_Submitted_Late": 0},
        {"Claim_Amount": 340, "Approved_Amount": 300, "Days_Between_Service_and_Claim": 21,
         "Provider_Specialty": "Dermatology", "Visit_Type": "Outpatient", "Insurance_Type": "Medicaid",
         "Number_of_Procedures": 1, "Number_of_Previous_Claims_Patient": 5,
         "Number_of_Previous_Claims_Provider": 150, "Claim_Submitted_Late": 0},
        {"Claim_Amount": 5200, "Approved_Amount": 5200, "Days_Between_Service_and_Claim": 2,
         "Provider_Specialty": "Orthopedics", "Visit_Type": "Emergency", "Insurance_Type": "Private",
         "Number_of_Procedures": 3, "Number_of_Previous_Claims_Patient": 1,
         "Number_of_Previous_Claims_Provider": 400, "Claim_Submitted_Late": 1},
        {"Claim_Amount": 610, "Approved_Amount": 610, "Days_Between_Service_and_Claim": 9,
         "Provider_Specialty": "General Practice", "Visit_Type": "Outpatient", "Insurance_Type": "Medicare",
         "Number_of_Procedures": 1, "Number_of_Previous_Claims_Patient": 3,
         "Number_of_Previous_Claims_Provider": 520, "Claim_Submitted_Late": 0},
    ]
    _write_csv(out_dir / "CLM900_claims_bulk.csv", fieldnames, csv_rows)


# ══════════════════════════════════════════════════════════════════════
# Financial Services — new wallet WALLET900 (existing UI sample is WALLET001)
# ══════════════════════════════════════════════════════════════════════

def financial_services_samples(out_dir: Path) -> None:
    txns = [
        ["2026-04-01", "Receive", "$180.00", "Ethereum", "0x99aa...11bb"],
        ["2026-04-04", "Trade", "$2,400.00", "Ethereum", "Kraken"],
        ["2026-04-08", "Bridge", "$950.00", "Ethereum -> Arbitrum", "Arbitrum Bridge"],
        ["2026-04-11", "Send", "$60.00", "Arbitrum", "0x22cc...44dd"],
        ["2026-04-16", "Swap", "$1,750.00", "Ethereum", "Uniswap V3"],
        ["2026-04-20", "Receive", "$25.00", "Polygon", "0x88ee...77ff"],
    ]
    header = ["Date", "Type", "Amount (USD)", "Blockchain", "Counterparty"]
    subtitle = ["Wallet: 0x4d2f...a81c", "Statement period: Apr 1 – Apr 20, 2026"]
    _write_pdf(out_dir / "WALLET900_statement.pdf", "Wallet Activity Statement", subtitle, header, txns,
               [24 * mm, 20 * mm, 26 * mm, 42 * mm, 38 * mm])
    _write_docx(out_dir / "WALLET900_statement.docx", "Wallet Activity Statement", subtitle, header, txns)

    fieldnames = [
        "transaction_amount_usd", "sender_wallet_age_days", "is_cross_chain",
        "blockchain", "platform", "failed_txn_ratio_sender",
    ]
    rng = random.Random(900)
    types = ["Ethereum", "Polygon", "Arbitrum", "Ethereum -> Polygon"]
    platforms = ["Uniswap V3", "Coinbase", "Kraken", "Direct Transfer"]
    csv_rows = []
    for _ in range(6):
        csv_rows.append({
            "transaction_amount_usd": round(rng.uniform(20, 5000), 2),
            "sender_wallet_age_days": rng.randint(1, 900),
            "is_cross_chain": rng.choice([True, False]),
            "blockchain": rng.choice(types),
            "platform": rng.choice(platforms),
            "failed_txn_ratio_sender": round(rng.uniform(0, 0.4), 3),
        })
    _write_csv(out_dir / "WALLET900_transactions_bulk.csv", fieldnames, csv_rows)


# ══════════════════════════════════════════════════════════════════════
# Payroll & HR — new employee EMP900 (existing UI sample is EMP001)
# ══════════════════════════════════════════════════════════════════════

def payroll_hr_samples(out_dir: Path) -> None:
    header = ["Field", "Value"]
    rows = [
        ["Employee", "EMP900 — R. Whitfield"],
        ["Job Title", "Senior Analyst"],
        ["Pay Period", "2026-03-01 to 2026-03-31"],
        ["Gross Pay", "$5,400.00"],
        ["Total Deductions", "$1,080.00"],
        ["Net Pay", "$4,320.00"],
    ]
    subtitle = ["Initech LLC — Payroll Register", "Pay Date: 2026-04-01"]
    _write_pdf(out_dir / "EMP900_payslip.pdf", "Payslip", subtitle, header, rows, [55 * mm, 55 * mm])
    _write_docx(out_dir / "EMP900_payslip.docx", "Payslip", subtitle, header, rows)

    fieldnames = ["BasePay", "OvertimePay", "OtherPay", "Benefits", "Stated_TotalPay",
                  "Stated_TotalPayBenefits", "JobTitle"]
    csv_rows = [
        {"BasePay": 5000, "OvertimePay": 300, "OtherPay": 100, "Benefits": 450,
         "Stated_TotalPay": 5400, "Stated_TotalPayBenefits": 5850, "JobTitle": "Senior Analyst"},
        {"BasePay": 4200, "OvertimePay": 0, "OtherPay": 0, "Benefits": 380,
         "Stated_TotalPay": 4200, "Stated_TotalPayBenefits": 4580, "JobTitle": "Support Specialist"},
        {"BasePay": 6100, "OvertimePay": 750, "OtherPay": 200, "Benefits": 520,
         "Stated_TotalPay": 7050, "Stated_TotalPayBenefits": 7570, "JobTitle": "Operations Manager"},
        {"BasePay": 3800, "OvertimePay": 120, "OtherPay": 0, "Benefits": 300,
         "Stated_TotalPay": 3920, "Stated_TotalPayBenefits": 4220, "JobTitle": "Associate"},
        {"BasePay": 5600, "OvertimePay": 0, "OtherPay": 500, "Benefits": 470,
         "Stated_TotalPay": 6100, "Stated_TotalPayBenefits": 6570, "JobTitle": "Data Engineer"},
    ]
    _write_csv(out_dir / "EMP900_payroll_register_bulk.csv", fieldnames, csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(HERE),
        help="Output root (a <domain> subfolder is created under it for each of the 4 domains). Default: data/",
    )
    args = parser.parse_args()
    out_root = Path(args.out)

    for domain, fn in [
        ("banking", banking_samples),
        ("insurance", insurance_samples),
        ("financial_services", financial_services_samples),
        ("payroll_hr", payroll_hr_samples),
    ]:
        domain_dir = out_root / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        fn(domain_dir)

    print(f"Wrote new sample sets under {out_root}/<domain>/ for banking, insurance, financial_services, payroll_hr")


if __name__ == "__main__":
    main()

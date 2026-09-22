"""
domains/payroll_hr/payroll/tools/generate_payroll_register.py
Modeled on domains/banking/documents/tools/generate_statement.py's reportlab
approach — same visual/document-generation pattern, own synthetic data (no
source CSV exists for payroll anywhere in this codebase, so this generates
the payslip figures itself instead of reading one).

One demo employee's payslip is deliberately generated with a miscalculated
net pay (the health-insurance deduction line is computed but silently left
out of the stated net pay) — the same "intentional signal" pattern already
used for bfsi_fraud's rule-based demo data and bfsi_documents' M100
merchant, so domains/payroll_hr/payroll/agents/validator_agent.py's reconciliation
check has something real to catch.

Usage (from the prism/ repo root):
    python domains/payroll_hr/payroll/tools/generate_payroll_register.py

Output:
    domains/payroll_hr/payroll/data/statements/EMP001_payslip.pdf   (correct)
    domains/payroll_hr/payroll/data/statements/EMP002_payslip.pdf   (correct)
    domains/payroll_hr/payroll/data/statements/EMP003_payslip.pdf   (miscalculated net pay)
"""

from pathlib import Path
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

_HERE = Path(__file__).parent.parent  # domains/payroll_hr/payroll/
OUTPUT_DIR = _HERE / "data" / "statements"

COMPANY_NAME = "Meridian Manufacturing Co."
COMPANY_ADDRESS = "1200 Industrial Pkwy, Columbus, OH 43215"
COMPANY_PHONE = "1-614-555-0142"

FEDERAL_TAX_RATE = 0.12
STATE_TAX_RATE = 0.05
SOCIAL_SECURITY_RATE = 0.062
MEDICARE_RATE = 0.0145
K401_RATE = 0.05
HEALTH_DEDUCTION = 150.00


def compute_payslip(
    employee_id: str,
    name: str,
    regular_hours: float,
    hourly_rate: float,
    overtime_hours: float,
    break_reconciliation: bool = False,
) -> dict:
    regular_pay = regular_hours * hourly_rate
    overtime_pay = overtime_hours * hourly_rate * 1.5
    gross_pay = regular_pay + overtime_pay

    federal_tax = round(gross_pay * FEDERAL_TAX_RATE, 2)
    state_tax = round(gross_pay * STATE_TAX_RATE, 2)
    social_security = round(gross_pay * SOCIAL_SECURITY_RATE, 2)
    medicare = round(gross_pay * MEDICARE_RATE, 2)
    k401 = round(gross_pay * K401_RATE, 2)
    health = HEALTH_DEDUCTION

    deductions_total = federal_tax + state_tax + social_security + medicare + k401 + health
    correct_net_pay = round(gross_pay - deductions_total, 2)

    # The deliberate error: health deduction is calculated and *listed* on
    # the payslip, but silently left out of the stated net pay — a real,
    # common payroll-processing mistake. The PDF's own printed net pay is
    # wrong; the line items themselves are all correct and present, which
    # is exactly what a reconciliation check (rather than an LLM guessing
    # from vibes) is needed to catch.
    stated_net_pay = (
        round(correct_net_pay + health, 2) if break_reconciliation else correct_net_pay
    )

    return {
        "employee_id": employee_id,
        "name": name,
        "regular_hours": regular_hours,
        "hourly_rate": hourly_rate,
        "overtime_hours": overtime_hours,
        "regular_pay": regular_pay,
        "overtime_pay": overtime_pay,
        "gross_pay": gross_pay,
        "federal_tax": federal_tax,
        "state_tax": state_tax,
        "social_security": social_security,
        "medicare": medicare,
        "k401": k401,
        "health": health,
        "deductions_total": deductions_total,
        "net_pay": stated_net_pay,
        "correct_net_pay": correct_net_pay,
        "broken": break_reconciliation,
    }


def generate_pdf(payslip: dict, output_path: Path, pay_period: str, pay_date: str) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    navy = colors.HexColor("#0a1628")
    gold = colors.HexColor("#c9a84c")
    light_gray = colors.HexColor("#f5f5f5")
    mid_gray = colors.HexColor("#e0e0e0")
    dark_gray = colors.HexColor("#333333")
    green = colors.HexColor("#2e7d32")
    red = colors.HexColor("#c62828")

    title_style = ParagraphStyle(
        "Title", fontSize=18, fontName="Helvetica-Bold", textColor=navy,
        alignment=TA_CENTER, spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "Sub", fontSize=8, fontName="Helvetica", textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section", fontSize=9, fontName="Helvetica-Bold", textColor=navy,
        spaceBefore=8, spaceAfter=4,
    )
    normal_style = ParagraphStyle("Normal2", fontSize=8, fontName="Helvetica", textColor=dark_gray)
    bold_style = ParagraphStyle("Bold2", fontSize=8, fontName="Helvetica-Bold", textColor=dark_gray)

    elements = [
        Spacer(1, 2 * mm),
        Paragraph(COMPANY_NAME, title_style),
        Paragraph(COMPANY_ADDRESS, sub_style),
        Paragraph(COMPANY_PHONE, sub_style),
        Spacer(1, 4 * mm),
        HRFlowable(width="100%", thickness=2, color=navy),
        Spacer(1, 2 * mm),
    ]

    title_table = Table([["PAYROLL REGISTER — EARNINGS STATEMENT"]], colWidths=[180 * mm])
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), navy),
        ("TEXTCOLOR", (0, 0), (-1, -1), gold),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [title_table, Spacer(1, 4 * mm)]

    info_data = [
        [Paragraph("<b>Employee</b>", bold_style), Paragraph(payslip["name"], normal_style),
         Paragraph("<b>Employee ID</b>", bold_style), Paragraph(payslip["employee_id"], normal_style)],
        [Paragraph("<b>Pay Period</b>", bold_style), Paragraph(pay_period, normal_style),
         Paragraph("<b>Pay Date</b>", bold_style), Paragraph(pay_date, normal_style)],
    ]
    info_table = Table(info_data, colWidths=[35 * mm, 55 * mm, 35 * mm, 55 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_gray),
        ("GRID", (0, 0), (-1, -1), 0.5, mid_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [info_table, Spacer(1, 4 * mm)]

    elements.append(Paragraph("EARNINGS", section_style))
    earnings_rows = [
        ["Category", "Hours", "Rate", "Amount ($)"],
        ["Regular Hours", f"{payslip['regular_hours']:.1f}", f"${payslip['hourly_rate']:.2f}",
         f"${payslip['regular_pay']:,.2f}"],
        ["Overtime", f"{payslip['overtime_hours']:.1f}", f"${payslip['hourly_rate'] * 1.5:.2f}",
         f"${payslip['overtime_pay']:,.2f}"],
        ["Gross Pay", "", "", f"${payslip['gross_pay']:,.2f}"],
    ]
    earnings_table = Table(earnings_rows, colWidths=[60 * mm, 30 * mm, 30 * mm, 40 * mm])
    earnings_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), light_gray),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.3, mid_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements += [earnings_table, Spacer(1, 4 * mm)]

    elements.append(Paragraph("DEDUCTIONS", section_style))
    deduction_rows = [
        ["Category", "Amount ($)"],
        ["Federal Tax", f"${payslip['federal_tax']:,.2f}"],
        ["State Tax", f"${payslip['state_tax']:,.2f}"],
        ["Social Security", f"${payslip['social_security']:,.2f}"],
        ["Medicare", f"${payslip['medicare']:,.2f}"],
        ["Health Deduction", f"${payslip['health']:,.2f}"],
        ["401k", f"${payslip['k401']:,.2f}"],
        ["Total Deductions", f"${payslip['deductions_total']:,.2f}"],
    ]
    deduction_table = Table(deduction_rows, colWidths=[120 * mm, 40 * mm])
    ded_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, -1), (-1, -1), light_gray),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.3, mid_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (0, 1), (0, 6), red),
    ])
    deduction_table.setStyle(ded_style)
    elements += [deduction_table, Spacer(1, 4 * mm)]

    elements.append(Paragraph("SUMMARY", section_style))
    summary_rows = [
        ["Gross Pay", "Total Deductions", "Net Pay"],
        [f"${payslip['gross_pay']:,.2f}", f"${payslip['deductions_total']:,.2f}",
         f"${payslip['net_pay']:,.2f}"],
    ]
    summary_table = Table(summary_rows, colWidths=[60 * mm, 60 * mm, 60 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#e8f5e9")),
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#ffebee")),
        ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#e8f5e9") if not payslip["broken"] else colors.HexColor("#fff3e0")),
        ("TEXTCOLOR", (2, 1), (2, 1), green if not payslip["broken"] else colors.HexColor("#e65100")),
        ("GRID", (0, 0), (-1, -1), 0.5, mid_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [summary_table, Spacer(1, 6 * mm)]

    elements.append(HRFlowable(width="100%", thickness=0.5, color=mid_gray))
    elements.append(Spacer(1, 2 * mm))
    footer_style = ParagraphStyle(
        "Footer", fontSize=6.5, fontName="Helvetica", textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER,
    )
    elements.append(Paragraph(
        f"Computer-generated payroll register for employee {payslip['employee_id']}. "
        f"For queries, contact HR at {COMPANY_PHONE}.",
        footer_style,
    ))
    elements.append(Paragraph(
        "Prism — Domain-Agnostic Multi-Agent Intelligence Platform · payroll domain",
        footer_style,
    ))
    doc.build(elements)


DEMO_EMPLOYEES = [
    # (employee_id, name, regular_hours, hourly_rate, overtime_hours, break_reconciliation)
    ("EMP001", "Alicia Ferreira", 80.0, 34.00, 4.0, False),
    ("EMP002", "Marcus Whitfield", 80.0, 41.50, 0.0, False),
    ("EMP003", "Priya Subramanian", 80.0, 38.00, 6.0, True),  # deliberately broken
]


def main():
    print("=" * 60)
    print("Prism (payroll) — Payroll Register Generator")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")

    pay_period = "March 1, 2026 — March 14, 2026"
    pay_date = date(2026, 3, 20).strftime("%B %d, %Y")

    for employee_id, name, hours, rate, ot, broken in DEMO_EMPLOYEES:
        payslip = compute_payslip(employee_id, name, hours, rate, ot, break_reconciliation=broken)
        output_path = OUTPUT_DIR / f"{employee_id}_payslip.pdf"
        generate_pdf(payslip, output_path, pay_period, pay_date)
        flag = "⚠ deliberately miscalculated net pay" if broken else "✅ reconciles correctly"
        print(f"   {employee_id}_payslip.pdf — {name} ({flag})")

    print(f"\n✅ Generated {len(DEMO_EMPLOYEES)} payslips")
    print(f"📁 Saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

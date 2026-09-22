"""
domains/insurance/tools/generate_claim_document.py
Modeled on domains/payroll_hr/payroll/tools/generate_payroll_register.py's
reportlab approach -- same visual/document-generation pattern, synthetic
data shaped to exercise both the RAG extraction path (ReasoningAgent)
and, once mapped, the real Tier 1 fraud detector
(HealthcareDetectorAgent's rule_based_filter/ml_filter) -- not just
plausible-looking numbers.

Three demo claims, one deliberately rule-violating (Approved_Amount >
Claim_Amount, the exact case healthcare_detector_agent.py's
_rule_check_tier1 catches) -- same "intentional signal" pattern already
used for payroll's/banking's demo documents.

Usage (from the prism/ repo root):
    python domains/insurance/tools/generate_claim_document.py

Output:
    domains/insurance/data/statements/CLM001_claim.pdf   (clean)
    domains/insurance/data/statements/CLM002_claim.pdf   (clean)
    domains/insurance/data/statements/CLM003_claim.pdf   (approved > billed -- rule violation)
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

_HERE = Path(__file__).parent.parent  # domains/insurance/
OUTPUT_DIR = _HERE / "data" / "statements"

PAYER_NAME = "Meridian Health Plan"
PAYER_ADDRESS = "500 Claims Way, Hartford, CT 06103"
PAYER_PHONE = "1-860-555-0199"


def generate_pdf(claim: dict, output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=15 * mm, leftMargin=15 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
    )

    navy = colors.HexColor("#0a1628")
    gold = colors.HexColor("#c9a84c")
    light_gray = colors.HexColor("#f5f5f5")
    mid_gray = colors.HexColor("#e0e0e0")
    dark_gray = colors.HexColor("#333333")
    red = colors.HexColor("#c62828")

    title_style = ParagraphStyle("Title", fontSize=18, fontName="Helvetica-Bold", textColor=navy, alignment=TA_CENTER, spaceAfter=4)
    sub_style = ParagraphStyle("Sub", fontSize=8, fontName="Helvetica", textColor=colors.HexColor("#666666"), alignment=TA_CENTER)
    section_style = ParagraphStyle("Section", fontSize=9, fontName="Helvetica-Bold", textColor=navy, spaceBefore=8, spaceAfter=4)
    normal_style = ParagraphStyle("Normal2", fontSize=8, fontName="Helvetica", textColor=dark_gray)
    bold_style = ParagraphStyle("Bold2", fontSize=8, fontName="Helvetica-Bold", textColor=dark_gray)

    elements = [
        Spacer(1, 2 * mm),
        Paragraph(PAYER_NAME, title_style),
        Paragraph(PAYER_ADDRESS, sub_style),
        Paragraph(PAYER_PHONE, sub_style),
        Spacer(1, 4 * mm),
        HRFlowable(width="100%", thickness=2, color=navy),
        Spacer(1, 2 * mm),
    ]

    title_table = Table([["EXPLANATION OF BENEFITS — CLAIM SUMMARY"]], colWidths=[180 * mm])
    title_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), navy), ("TEXTCOLOR", (0, 0), (-1, -1), gold),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [title_table, Spacer(1, 4 * mm)]

    info_data = [
        [Paragraph("<b>Claim Number</b>", bold_style), Paragraph(claim["claim_id"], normal_style),
         Paragraph("<b>Patient</b>", bold_style), Paragraph(claim["patient_name"], normal_style)],
        [Paragraph("<b>Service Date</b>", bold_style), Paragraph(claim["service_date"], normal_style),
         Paragraph("<b>Claim Date</b>", bold_style), Paragraph(claim["claim_date"], normal_style)],
        [Paragraph("<b>Patient Age / Gender</b>", bold_style), Paragraph(f"{claim['patient_age']} / {claim['patient_gender']}", normal_style),
         Paragraph("<b>Insurance Type</b>", bold_style), Paragraph(claim["insurance_type"], normal_style)],
        [Paragraph("<b>Provider Specialty</b>", bold_style), Paragraph(claim["provider_specialty"], normal_style),
         Paragraph("<b>Visit Type</b>", bold_style), Paragraph(claim["visit_type"], normal_style)],
    ]
    info_table = Table(info_data, colWidths=[42 * mm, 48 * mm, 42 * mm, 48 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), light_gray), ("GRID", (0, 0), (-1, -1), 0.5, mid_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements += [info_table, Spacer(1, 4 * mm)]

    elements.append(Paragraph("CLAIM AMOUNTS", section_style))
    amount_rows = [
        ["Billed (Claim Amount)", "Approved Amount"],
        [f"${claim['claim_amount']:,.2f}", f"${claim['approved_amount']:,.2f}"],
    ]
    amount_table = Table(amount_rows, colWidths=[90 * mm, 90 * mm])
    approved_over_billed = claim["approved_amount"] > claim["claim_amount"]
    amount_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.5, mid_gray),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TEXTCOLOR", (1, 1), (1, 1), red if approved_over_billed else dark_gray),
    ]))
    elements += [amount_table, Spacer(1, 6 * mm)]

    elements.append(HRFlowable(width="100%", thickness=0.5, color=mid_gray))
    elements.append(Spacer(1, 2 * mm))
    footer_style = ParagraphStyle("Footer", fontSize=6.5, fontName="Helvetica", textColor=colors.HexColor("#999999"), alignment=TA_CENTER)
    elements.append(Paragraph(f"Computer-generated EOB for claim {claim['claim_id']}. For queries, contact {PAYER_PHONE}.", footer_style))
    elements.append(Paragraph("Prism — Domain-Agnostic Multi-Agent Intelligence Platform · insurance domain", footer_style))
    doc.build(elements)


DEMO_CLAIMS = [
    {
        "claim_id": "CLM001", "patient_name": "Dana Whitfield", "service_date": "2026-01-14",
        "claim_date": "2026-01-22", "patient_age": 55, "patient_gender": "Male",
        "insurance_type": "Medicaid", "provider_specialty": "Cardiology", "visit_type": "Outpatient",
        "claim_amount": 480.00, "approved_amount": 400.00,
    },
    {
        "claim_id": "CLM002", "patient_name": "Renee Okafor", "service_date": "2026-02-03",
        "claim_date": "2026-02-09", "patient_age": 34, "patient_gender": "Female",
        "insurance_type": "Private", "provider_specialty": "Dermatology", "visit_type": "Outpatient",
        "claim_amount": 220.00, "approved_amount": 180.00,
    },
    {
        # Deliberately rule-violating: approved amount exceeds the billed
        # claim amount -- exactly what healthcare_detector_agent.py's
        # _rule_check_tier1 flags, and what ValidatorAgent's own sanity
        # check flags independently on the RAG-extraction path.
        "claim_id": "CLM003", "patient_name": "Priya Subramanian", "service_date": "2026-01-05",
        "claim_date": "2026-01-11", "patient_age": 47, "patient_gender": "Female",
        "insurance_type": "Medicare", "provider_specialty": "Orthopedics", "visit_type": "Inpatient",
        "claim_amount": 500.00, "approved_amount": 9999.00,
    },
]


def main():
    print("=" * 60)
    print("Prism (insurance) — Claim/EOB Document Generator")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")

    for claim in DEMO_CLAIMS:
        output_path = OUTPUT_DIR / f"{claim['claim_id']}_claim.pdf"
        generate_pdf(claim, output_path)
        flag = "⚠ approved > billed (rule violation)" if claim["approved_amount"] > claim["claim_amount"] else "✅ clean"
        print(f"   {claim['claim_id']}_claim.pdf — {claim['patient_name']} ({flag})")

    print(f"\n✅ Generated {len(DEMO_CLAIMS)} claim documents")
    print(f"📁 Saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

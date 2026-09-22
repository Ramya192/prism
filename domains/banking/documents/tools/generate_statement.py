"""
domains/banking/documents/tools/generate_statement.py
Ported from document_intelligence_system/tools/generate_statement.py —
logic unchanged, paths updated to this domain's own data/ folder. This is
also where the credit/debit classification fix lives (see clean_transactions()
below) — configs/bfsi_documents.yaml's to-do note points here.

Reads Kaggle bank transaction CSV and generates realistic
bank statement PDFs — one per account (full year).

Usage (from the prism/ repo root):
    python domains/banking/documents/tools/generate_statement.py

Output:
    domains/banking/documents/data/statements/AC00001_statement.pdf
    domains/banking/documents/data/statements/AC00002_statement.pdf
    ...

Only accounts with >= 5 transactions are generated.
Bank names are distributed across 5 US banks (unequal split
to simulate real-world market share).
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
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
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ── Constants ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent  # domains/banking/documents/
CSV_PATH = _HERE / "data" / "raw" / "bank_transactions_data_2.csv"
OUTPUT_DIR = _HERE / "data" / "statements"
MIN_TRANSACTIONS = 5

# ── Bank Distribution (unequal — mirrors real-world market share) ─────────────
BANK_DISTRIBUTION = [
    (
        "JPMorgan Chase",
        "383 Madison Ave, New York, NY 10017",
        "1-800-935-9935",
        "www.jpmorganchase.com",
        80,
    ),
    (
        "Bank of America",
        "100 N Tryon St, Charlotte, NC 28255",
        "1-800-432-1000",
        "www.bankofamerica.com",
        70,
    ),
    (
        "Wells Fargo",
        "420 Montgomery St, San Francisco, CA 94104",
        "1-800-869-3557",
        "www.wellsfargo.com",
        60,
    ),
    (
        "Citibank",
        "388 Greenwich St, New York, NY 10013",
        "1-800-374-9700",
        "www.citibank.com",
        50,
    ),
    (
        "Goldman Sachs",
        "200 West St, New York, NY 10282",
        "1-800-323-5678",
        "www.goldmansachs.com",
        25,
    ),
]

# Merchant ID → readable description
MERCHANT_MAP = {
    "M001": "Amazon Online Purchase",
    "M002": "Walmart Superstore",
    "M003": "Target Store",
    "M004": "Best Buy Electronics",
    "M005": "Costco Wholesale",
    "M006": "Home Depot",
    "M007": "Kroger Grocery",
    "M008": "CVS Pharmacy",
    "M009": "Walgreens Pharmacy",
    "M010": "Starbucks Coffee",
    "M011": "McDonald's Restaurant",
    "M012": "Uber Ride",
    "M013": "Netflix Subscription",
    "M014": "Spotify Premium",
    "M015": "Apple Store",
    "M016": "Shell Gas Station",
    "M017": "Local Restaurant",
    "M018": "Marriott Hotel",
    "M019": "Delta Airlines",
    "M020": "State Farm Insurance",
    "M021": "Con Edison Utilities",
    "M022": "Rent Payment",
    "M023": "Planet Fitness",
    "M024": "Rite Aid Pharmacy",
    "M025": "Whole Foods Market",
    "M026": "Trader Joe's",
    "M027": "Bank Transfer",
    "M028": "PayPal Transfer",
    "M029": "Venmo Payment",
    "M030": "Zelle Transfer",
    "M031": "AT&T Mobile",
    "M032": "Verizon Wireless",
    "M033": "Comcast Internet",
    "M034": "Electric Bill",
    "M035": "Water Bill",
    "M036": "Chase Mortgage",
    "M037": "Sallie Mae Loan",
    "M038": "Student Loan Payment",
    "M039": "Car Insurance",
    "M040": "Auto Loan Payment",
    # M041-M049 — added in v2 (were missing, causing "Unknown Merchant" for valid txns)
    "M041": "Chase Bank Transfer",
    "M042": "Wells Fargo Transfer",
    "M043": "Unknown Merchant (M043)",  # intentionally unknown — fraud signal
    "M044": "Unknown Merchant (M044)",  # intentionally unknown — fraud signal
    "M045": "Square Payment",
    "M046": "Cash App Transfer",
    "M047": "Apple Pay Purchase",
    "M048": "Google Pay Purchase",
    "M049": "Samsung Pay Purchase",
    "M050": "Online Transfer",
    "M051": "Wire Transfer",
    "M052": "ATM Withdrawal",
    "M053": "Whole Foods Delivery",
    "M054": "DoorDash Food Delivery",
    "M055": "Uber Eats Delivery",
    "M056": "Grubhub Food Delivery",
    "M057": "Instacart Grocery",
    "M058": "Lyft Ride",
    "M059": "Airbnb Lodging",
    "M060": "Southwest Airlines",
    "M061": "United Airlines",
    "M062": "American Airlines",
    "M063": "Hilton Hotels",
    "M064": "Hyatt Hotels",
    "M065": "Booking.com",
    "M066": "Expedia Travel",
    "M067": "Apple iTunes",
    "M068": "Google Play Store",
    "M069": "Amazon Prime",
    "M070": "Disney Plus",
    "M071": "HBO Max",
    "M072": "Hulu Subscription",
    "M073": "YouTube Premium",
    "M074": "Adobe Creative Cloud",
    "M075": "Microsoft Office 365",
    "M076": "Dropbox Storage",
    "M077": "Zoom Subscription",
    "M078": "Slack Subscription",
    "M079": "LinkedIn Premium",
    "M080": "Chewy Pet Supplies",
    "M081": "PetSmart",
    "M082": "IKEA Furniture",
    "M083": "Wayfair Home Goods",
    "M084": "Macy's Department Store",
    "M085": "Nordstrom",
    "M086": "Gap Clothing",
    "M087": "H&M Fashion",
    "M088": "Nike Store",
    "M089": "Adidas Store",
    "M090": "Sephora Beauty",
    "M091": "Ulta Beauty",
    "M092": "GameStop",
    "M093": "Steam Gaming",
    "M094": "PlayStation Store",
    "M095": "Xbox Game Pass",
    "M096": "Chipotle Mexican Grill",
    "M097": "Panera Bread",
    "M098": "Subway Restaurant",
    "M099": "Pizza Hut",
    "M100": "Unknown Merchant (M100)",  # intentionally unknown — fraud signal
}

# Occupation → customer name prefix
OCCUPATION_NAME_MAP = {
    "Doctor": ["Dr. James Wilson", "Dr. Sarah Mitchell", "Dr. Robert Chen"],
    "Engineer": ["Alex Johnson", "Maria Garcia", "David Kim"],
    "Teacher": ["Emily Brown", "Michael Davis", "Linda Thompson"],
    "Student": ["Jake Martinez", "Ashley Lee", "Ryan Patel"],
    "Manager": ["Christopher White", "Jennifer Taylor", "Daniel Harris"],
    "Lawyer": ["Esq. Patricia Moore", "Esq. Kevin Anderson", "Esq. Nancy Martin"],
    "Accountant": ["Thomas Jackson", "Barbara Robinson", "Charles Lewis"],
    "Nurse": ["Susan Walker", "Joseph Hall", "Karen Allen"],
}


def assign_banks(sorted_accounts: list) -> dict:
    assignments = {}
    idx = 0
    for bank_name, bank_addr, bank_phone, bank_web, count in BANK_DISTRIBUTION:
        for account_id in sorted_accounts[idx : idx + count]:
            assignments[account_id] = (bank_name, bank_addr, bank_phone, bank_web)
        idx += count
    last_bank = BANK_DISTRIBUTION[-1]
    for account_id in sorted_accounts[idx:]:
        assignments[account_id] = (
            last_bank[0],
            last_bank[1],
            last_bank[2],
            last_bank[3],
        )
    return assignments


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Credit/debit classification fix: the source Kaggle data has ATM-channel
    # rows mislabeled TransactionType="Credit" (an ATM in this dataset is
    # always a withdrawal, i.e. a debit) — force the correct label before any
    # statement/ground-truth data is generated from this frame, so the fix
    # is baked into every downstream PDF and eval reference answer.
    atm_credit_mask = (df["Channel"] == "ATM") & (df["TransactionType"] == "Credit")
    fixed_atm = atm_credit_mask.sum()
    df.loc[atm_credit_mask, "TransactionType"] = "Debit"
    print(f"   🔧 Fixed {fixed_atm} ATM+Credit mismatches → Debit")
    df["is_suspicious"] = False
    df.loc[df["MerchantID"] == "M100", "is_suspicious"] = True
    suspicious_count = (df["MerchantID"] == "M100").sum()
    print(f"   🚨 Flagged {suspicious_count} M100 transactions as suspicious")
    return df


def get_customer_name(account_id: str, occupation: str) -> str:
    names = OCCUPATION_NAME_MAP.get(
        occupation, ["John Doe", "Jane Smith", "Bob Wilson"]
    )
    idx = int(account_id.replace("AC", "")) % len(names)
    return names[idx]


def get_description(row: pd.Series) -> str:
    merchant = MERCHANT_MAP.get(
        row["MerchantID"], f"Unknown Merchant ({row['MerchantID']})"
    )
    channel = row["Channel"]
    location = row["Location"]

    if row.get("is_suspicious", False):
        merchant = f"[SUSPICIOUS] {merchant}"

    if channel == "ATM":
        return f"ATM Withdrawal — {location}"
    elif channel == "Online":
        # Fix: avoid doubling "Online" if already in merchant name
        if "online" in merchant.lower():
            return merchant
        return f"{merchant} — Online"
    elif channel == "Branch":
        return f"{merchant} — Branch ({location})"
    else:
        return f"{merchant} — {channel}"


def format_amount(amount: float, txn_type: str) -> tuple:
    formatted = f"${amount:,.2f}"
    if txn_type == "Debit":
        return formatted, ""
    else:
        return "", formatted


def generate_pdf(
    account_id: str,
    transactions: pd.DataFrame,
    output_path: Path,
    bank_name: str,
    bank_address: str,
    bank_phone: str,
    bank_website: str,
):
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()
    elements = []

    navy = colors.HexColor("#0a1628")
    gold = colors.HexColor("#c9a84c")
    light_gray = colors.HexColor("#f5f5f5")
    mid_gray = colors.HexColor("#e0e0e0")
    dark_gray = colors.HexColor("#333333")
    green = colors.HexColor("#2e7d32")
    red = colors.HexColor("#c62828")

    bank_title_style = ParagraphStyle(
        "BankTitle",
        fontSize=20,
        fontName="Helvetica-Bold",
        textColor=navy,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    bank_sub_style = ParagraphStyle(
        "BankSub",
        fontSize=8,
        fontName="Helvetica",
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=1,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=navy,
        spaceBefore=8,
        spaceAfter=4,
    )
    normal_style = ParagraphStyle(
        "Normal2",
        fontSize=8,
        fontName="Helvetica",
        textColor=dark_gray,
        spaceAfter=2,
    )
    bold_style = ParagraphStyle(
        "Bold2",
        fontSize=8,
        fontName="Helvetica-Bold",
        textColor=dark_gray,
    )

    transactions = transactions.copy()
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    transactions = transactions.sort_values("TransactionDate")

    occupation = transactions["CustomerOccupation"].iloc[0]
    customer_name = get_customer_name(account_id, occupation)
    location = transactions["Location"].iloc[0]
    period_start = transactions["TransactionDate"].min().strftime("%B %d, %Y")
    period_end = transactions["TransactionDate"].max().strftime("%B %d, %Y")
    opening_balance = transactions["AccountBalance"].iloc[0]
    closing_balance = transactions["AccountBalance"].iloc[-1]
    total_debits = transactions[transactions["TransactionType"] == "Debit"][
        "TransactionAmount"
    ].sum()
    total_credits = transactions[transactions["TransactionType"] == "Credit"][
        "TransactionAmount"
    ].sum()

    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(bank_name, bank_title_style))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph(bank_address, bank_sub_style))
    elements.append(Spacer(1, 1 * mm))
    elements.append(Paragraph(f"{bank_phone}  |  {bank_website}", bank_sub_style))
    elements.append(Spacer(1, 4 * mm))
    elements.append(HRFlowable(width="100%", thickness=2, color=navy))
    elements.append(Spacer(1, 2 * mm))

    title_data = [["ACCOUNT STATEMENT"]]
    title_table = Table(title_data, colWidths=[180 * mm])
    title_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), navy),
                ("TEXTCOLOR", (0, 0), (-1, -1), gold),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(title_table)
    elements.append(Spacer(1, 4 * mm))

    info_data = [
        [
            Paragraph("<b>Account Holder</b>", bold_style),
            Paragraph(customer_name, normal_style),
            Paragraph("<b>Account Number</b>", bold_style),
            Paragraph(account_id, normal_style),
        ],
        [
            Paragraph("<b>Occupation</b>", bold_style),
            Paragraph(occupation, normal_style),
            Paragraph("<b>Statement Period</b>", bold_style),
            Paragraph(f"{period_start} — {period_end}", normal_style),
        ],
        [
            Paragraph("<b>Primary Location</b>", bold_style),
            Paragraph(location, normal_style),
            Paragraph("<b>Account Type</b>", bold_style),
            Paragraph("Checking Account", normal_style),
        ],
    ]
    info_table = Table(info_data, colWidths=[40 * mm, 50 * mm, 40 * mm, 50 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), light_gray),
                ("GRID", (0, 0), (-1, -1), 0.5, mid_gray),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(info_table)
    elements.append(Spacer(1, 4 * mm))

    elements.append(Paragraph("ACCOUNT SUMMARY", section_header_style))
    summary_data = [
        ["Opening Balance", "Total Debits", "Total Credits", "Closing Balance"],
        [
            f"${opening_balance:,.2f}",
            f"${total_debits:,.2f}",
            f"${total_credits:,.2f}",
            f"${closing_balance:,.2f}",
        ],
    ]
    summary_table = Table(summary_data, colWidths=[45 * mm, 45 * mm, 45 * mm, 45 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), navy),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (0, 1), light_gray),
                ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#ffebee")),
                ("BACKGROUND", (2, 1), (2, 1), colors.HexColor("#e8f5e9")),
                ("BACKGROUND", (3, 1), (3, 1), light_gray),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, mid_gray),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 4 * mm))

    elements.append(Paragraph("TRANSACTION DETAILS", section_header_style))
    txn_header = [
        "Date",
        "Description",
        "Channel",
        "Debit ($)",
        "Credit ($)",
        "Balance ($)",
    ]
    txn_rows = [txn_header]
    for _, row in transactions.iterrows():
        date_str = row["TransactionDate"].strftime("%m/%d/%Y")
        description = get_description(row)
        channel = row["Channel"]
        debit, credit = format_amount(row["TransactionAmount"], row["TransactionType"])
        balance = f"${row['AccountBalance']:,.2f}"
        txn_rows.append([date_str, description, channel, debit, credit, balance])

    txn_table = Table(
        txn_rows, colWidths=[22 * mm, 72 * mm, 18 * mm, 22 * mm, 22 * mm, 24 * mm]
    )
    txn_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 7.5),
            ("ALIGN", (3, 1), (5, -1), "RIGHT"),
            ("ALIGN", (0, 1), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light_gray]),
            ("GRID", (0, 0), (-1, -1), 0.3, mid_gray),
            ("LINEBELOW", (0, 0), (-1, 0), 1.5, gold),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]
    )
    for i, row in enumerate(transactions.itertuples(), start=1):
        if row.TransactionType == "Debit":
            txn_style.add("TEXTCOLOR", (3, i), (3, i), red)
        else:
            txn_style.add("TEXTCOLOR", (4, i), (4, i), green)
    txn_table.setStyle(txn_style)
    elements.append(txn_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(HRFlowable(width="100%", thickness=0.5, color=mid_gray))
    elements.append(Spacer(1, 2 * mm))
    footer_style = ParagraphStyle(
        "Footer",
        fontSize=6.5,
        fontName="Helvetica",
        textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER,
    )
    elements.append(
        Paragraph(
            f"This is a computer-generated statement for account {account_id}. "
            f"Generated on {datetime.now().strftime('%B %d, %Y')}. "
            f"For queries, contact {bank_phone} or visit {bank_website}.",
            footer_style,
        )
    )
    elements.append(
        Paragraph(
            "Prism — Domain-Agnostic Multi-Agent Intelligence Platform · bfsi_documents domain",
            footer_style,
        )
    )
    doc.build(elements)


def main():
    print("=" * 60)
    print("Prism (bfsi_documents) — Bank Statement Generator")
    print("Multi-bank: JPMorgan Chase / Bank of America /")
    print("            Wells Fargo / Citibank / Goldman Sachs")
    print("=" * 60)

    print(f"\n📂 Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   Total rows: {len(df):,}")
    print(f"\n🔍 Running data quality checks...")
    df = clean_transactions(df)
    print(f"   Unique accounts: {df['AccountID'].nunique()}")

    account_counts = df.groupby("AccountID").size()
    eligible = account_counts[account_counts >= MIN_TRANSACTIONS].index
    df_filtered = df[df["AccountID"].isin(eligible)]
    sorted_accounts = sorted(eligible)
    print(f"   Accounts with >= {MIN_TRANSACTIONS} transactions: {len(eligible)}")

    bank_assignments = assign_banks(sorted_accounts)
    print(f"\n🏦 Bank distribution:")
    for bank_name, _, _, _, count in BANK_DISTRIBUTION:
        print(f"   {bank_name}: {count} accounts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"\n🔄 Generating {len(eligible)} PDFs...\n")

    success = 0
    failed = 0

    for account_id in sorted_accounts:
        account_df = df_filtered[df_filtered["AccountID"] == account_id]
        output_path = OUTPUT_DIR / f"{account_id}_statement.pdf"
        bank_name, bank_addr, bank_phone, bank_web = bank_assignments[account_id]

        try:
            generate_pdf(
                account_id=account_id,
                transactions=account_df,
                output_path=output_path,
                bank_name=bank_name,
                bank_address=bank_addr,
                bank_phone=bank_phone,
                bank_website=bank_web,
            )
            txn_count = len(account_df)
            print(f"   ✅ {account_id}_statement.pdf ({bank_name} · {txn_count} txns)")
            success += 1
        except Exception as e:
            print(f"   ❌ {account_id}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"✅ Generated: {success} PDFs")
    if failed:
        print(f"❌ Failed:    {failed} PDFs")
    print(f"📁 Saved to:  {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Console output uses emoji markers; on Windows the default console
    # codepage (cp1252) can't encode them and print() raises UnicodeEncodeError.
    # Force UTF-8 so this runs out of the box in a plain Windows terminal.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

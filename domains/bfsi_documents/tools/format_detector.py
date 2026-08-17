"""
domains/bfsi_documents/tools/format_detector.py
Ported from document_intelligence_system/tools/format_detector.py — logic
unchanged. Detects which bank issued a statement based on keyword matching
in the extracted PDF text.

Supported banks:
    - JPMorgan Chase
    - Bank of America
    - Wells Fargo
    - Citibank
    - Goldman Sachs

Returns "Unknown" if no bank is detected.

Usage:
    from domains.bfsi_documents.tools.format_detector import FormatDetector
    detector = FormatDetector()
    bank = detector.detect(text)  # returns e.g. "JPMorgan Chase"
"""


class FormatDetector:
    """
    Rule-based bank format detector.
    Scans the first 500 characters of extracted PDF text
    for bank-specific keywords. Deterministic, zero LLM cost.
    """

    # Bank name → list of keyword variants to match (case-insensitive)
    BANK_KEYWORDS = {
        "JPMorgan Chase": [
            "jpmorgan chase",
            "jp morgan chase",
            "jpmorganchase",
            "jpmorgan",
        ],
        "Bank of America": [
            "bank of america",
            "bankofamerica",
        ],
        "Wells Fargo": [
            "wells fargo",
            "wellsfargo",
        ],
        "Citibank": [
            "citibank",
            "citi bank",
            "citigroup",
        ],
        "Goldman Sachs": [
            "goldman sachs",
            "goldmansachs",
            "goldman",
        ],
    }

    # How many characters from the start of the text to scan
    SCAN_WINDOW = 500

    def detect(self, text: str) -> str:
        """
        Detect bank name from extracted PDF text.

        Args:
            text: Raw text extracted from PDF (full document or first page)

        Returns:
            Bank name string, or "Unknown" if not detected.
        """
        if not text:
            return "Unknown"

        # Only scan the first SCAN_WINDOW characters — bank name is always in header
        scan_text = text[: self.SCAN_WINDOW].lower()

        for bank_name, keywords in self.BANK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in scan_text:
                    return bank_name

        return "Unknown"

    def detect_from_file(self, pdf_path: str) -> str:
        """
        Detect bank name directly from a PDF file path.
        Convenience method — extracts text then detects.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Bank name string, or "Unknown" if not detected.
        """
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                first_page_text = pdf.pages[0].extract_text() or ""
            return self.detect(first_page_text)
        except Exception as e:
            print(f"FormatDetector error reading {pdf_path}: {e}")
            return "Unknown"

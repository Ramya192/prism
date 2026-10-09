# Offline tests for ui/landing.py's _classification_text: PDF/DOCX are classified
# from their opening text, not the filename alone.
import io

import docx

from ui.landing import _classification_text


def _docx_bytes(*paragraphs: str) -> bytes:
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def test_docx_text_is_included_with_the_filename():
    text = _classification_text(_docx_bytes("Explanation of Benefits", "Billed amount $480.00"), "claim.docx")
    assert text.startswith("document claim.docx")
    assert "Explanation of Benefits" in text and "$480.00" in text


def test_unreadable_file_falls_back_to_the_filename():
    assert _classification_text(b"not a real pdf", "CLM001_claim.pdf") == "document CLM001_claim.pdf"


def test_opening_text_is_capped():
    text = _classification_text(_docx_bytes("x" * 10_000), "big.docx", max_chars=100)
    assert len(text) <= len("document big.docx ") + 100

# Offline: the in-app evaluation table must stay identical to the README's.
import re
from pathlib import Path

from ui.evaluation_panel import EVAL_ROWS

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def test_every_panel_row_appears_in_the_readme_table():
    for _, label, n, rate, f1 in EVAL_ROWS:
        readme_label = label.replace("HR job postings (single tier)", "HR (single tier)")
        row = f"| {readme_label} | {n:,} | {rate} | {f1:.3f} |"
        assert row in README, f"README is missing or differs: {row}"


def test_panel_covers_all_four_domains_and_nine_tiers():
    assert len(EVAL_ROWS) == 9
    assert {r[0] for r in EVAL_ROWS} == {"banking", "insurance", "payroll_hr", "financial_services"}

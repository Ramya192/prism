# ui/shared.py's verdict + chat text rendering. Offline: Streamlit's AppTest
# runs the renderer headlessly, no API key or server needed.

from streamlit.testing.v1 import AppTest

from ui.shared import FLAGGED_INLINE_LIMIT, _layer_summary, _priority, md_safe


def test_md_safe_escapes_dollars_once():
    assert md_safe("over $500 and $5.00") == r"over \$500 and \$5.00"
    assert md_safe(r"already \$ok") == r"already \$ok"          # never double-escapes
    assert md_safe("no money here") == "no money here"


def _verdicts(n_flagged, n_clean=0, caveat=False):
    def v(pred, reason):
        parsed = {"action": "FLAG", "reason": reason}
        if caveat:
            parsed["caveat"] = "looks statistically unusual"
        return {"predicted": pred, "reason": reason, "parsed": parsed}
    return ([v("FRAUD", f"ML model fraud probability {90 + i % 10}% - amount over $500 or under $5.00")
             for i in range(n_flagged)]
            + [v("LEGITIMATE", "ok") for _ in range(n_clean)])


def _run(verdicts):
    def app(verdicts, summary):
        from ui.shared import render_fraud_verdicts
        render_fraud_verdicts(verdicts, summary)
    summary = {"total_scored": len(verdicts), "flagged": sum(v["predicted"] == "FRAUD" for v in verdicts),
               "clean": sum(v["predicted"] != "FRAUD" for v in verdicts)}
    at = AppTest.from_function(app, args=(verdicts, summary))
    at.run(timeout=30)
    assert not at.exception
    return at


def test_many_flagged_records_show_a_summary_not_a_wall_of_text():
    at = _run(_verdicts(n_flagged=60, n_clean=40))
    # Headline + per-layer count, not 60 lines.
    assert any("60 of 100 records flagged (60%)" in e.value for e in at.error)
    assert [m.label for m in at.metric] == ["ML Detector"] and at.metric[0].value == "60"
    # One line per finding, dollars escaped; no per-record lines on screen.
    summary = [m.value for m in at.markdown if "ML Detector" in m.value and "records" in m.value]
    assert len(summary) == 1 and "60 records" in summary[0]
    assert not [c for c in at.caption if c.value.startswith("— #")]
    # Everything else lives in one collapsed, searchable table.
    assert any("All 60 flagged records" in e.label for e in at.expander)
    assert len(at.dataframe[0].value) == 60


def test_layer_summary_groups_identical_reasons_and_ranges_ml_probabilities():
    flagged = ([(i, "Rule Engine", "Micro-transaction at night") for i in range(1, 41)]
               + [(50, "ML Detector", "ML model probability 71%"), (51, "ML Detector", "ML model probability 99%")]
               + [(60 + k, "LLM Reasoning", f"over $500 case {k}") for k in range(6)])
    lines = _layer_summary(flagged)
    rule = [ln for ln in lines if "Rule Engine" in ln]
    assert len(rule) == 1 and "40 records" in rule[0] and "#1, #2, #3 …" in rule[0]
    ml = next(ln for ln in lines if "ML Detector" in ln)
    assert "2 records" in ml and "71–99%" in ml and "highest: #51, #50" in ml
    llm = [ln for ln in lines if "LLM Reasoning" in ln]
    assert len(llm) == 5 and "2 more in the full list" in llm[-1]      # 4 quoted + "more"
    assert r"over \$500" in llm[0]                                       # escaped


def test_priority_puts_rule_hits_first_then_higher_ml_probability():
    rows = [(1, "LLM Reasoning", "judgement call"), (2, "ML Detector", "ML model ... probability 71%"),
            (3, "ML Detector", "ML model ... probability 99%"), (4, "Rule Engine", "Zero-dollar")]
    ranked = sorted(rows, key=lambda r: _priority(r[1], r[2]), reverse=True)
    assert [r[0] for r in ranked] == [4, 3, 2, 1]


def test_few_flagged_records_are_listed_inline():
    at = _run(_verdicts(n_flagged=FLAGGED_INLINE_LIMIT, n_clean=10))
    inline = [c.value for c in at.caption if c.value.startswith("— #")]
    assert len(inline) == FLAGGED_INLINE_LIMIT
    assert not at.metric and not at.dataframe


def test_single_record_keeps_the_plain_format():
    at = _run(_verdicts(n_flagged=1))
    captions = [c.value for c in at.caption]
    assert any(c.startswith("— [ML Detector]") for c in captions)     # no "#1" for one record
    assert not any(c.startswith("By layer:") for c in captions)
    assert not at.expander and not at.metric


def test_identical_drift_notes_are_listed_once():
    at = _run(_verdicts(n_flagged=5, n_clean=5, caveat=True))
    labels = [e.label for e in at.expander]
    assert any("10 record(s) also flagged as statistically unusual" in label for label in labels)
    captions = [c.value for c in at.caption]
    assert captions.count("looks statistically unusual") == 1
    assert any(c.startswith("Records: #1, #2") for c in captions)

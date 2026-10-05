"""Offline tests for fixes from the 2026-10-05 local test report."""
import io
import re
from unittest.mock import MagicMock

from PIL import Image

from ui.domains.banking import _is_valid_image


def _upload(data: bytes):
    m = MagicMock()
    m.getvalue.return_value = data
    return m


def test_fake_png_rejected_real_png_accepted():
    assert not _is_valid_image(_upload(b"not an image"))
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    assert _is_valid_image(_upload(buf.getvalue()))


def test_alert_ids_are_not_llm_generated():
    from domains.banking.fraud.agents import alert_agent
    agent = alert_agent.AlertAgent.__new__(alert_agent.AlertAgent)
    agent.name, agent.alerts_generated = "a", 0
    agent.llm = MagicMock()
    agent.llm.invoke.return_value.content = "ok"
    prompts = []
    agent.llm.invoke.side_effect = lambda msgs: (prompts.append(msgs[1].content), MagicMock(content="ok"))[1]
    for _ in range(5):
        agent.generate_alert({"Amount": 5, "hour": 3}, "d", "a")
    ids = [re.search(r"ALERT ID: (\d{8})", p).group(1) for p in prompts]
    assert len(set(ids)) > 1


def _bare_pipeline(llm_reply):
    from core.unified_pipeline import UnifiedDomainPipeline
    p = UnifiedDomainPipeline.__new__(UnifiedDomainPipeline)
    p.reasoning = MagicMock()
    p.reasoning.llm.invoke.return_value.content = llm_reply
    p.retriever = MagicMock()
    return p


def test_out_of_scope_question_skips_retrieval_and_reasoning():
    p = _bare_pipeline("OUT")
    result = p.run("What is the capital of France?", source_document="doc", history=[])
    assert result["status"] == "valid"
    assert result["data"]["source_document"] == "none" and result["data"]["transactions"] == []
    assert result["chunks"] == []
    p.retriever.retrieve.assert_not_called()
    p.reasoning.reason.assert_not_called()


def test_scope_check_fails_open_on_error():
    p = _bare_pipeline("IN")
    p.reasoning.llm.invoke.side_effect = RuntimeError("api down")
    assert p._in_scope("anything", None) is True

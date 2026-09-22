# domains/payroll_hr/payroll/agents/reasoning_agent.py
# Modeled on domains/banking/documents/agents/reasoning_agent.py — same
# retrieve-grounded-answer shape, but payslip-shaped instead of
# transaction-shaped: line items are pay categories (Gross Pay, Federal
# Tax, 401k, Overtime, Net Pay, ...), not bank transactions, and the
# "anomaly" this domain cares about is a miscalculation, not a suspicious
# merchant. See validator_agent.py for the deterministic reconciliation
# check that actually verifies the arithmetic — this agent only extracts
# and gives its own (LLM) read of whether anything looks wrong.

import json
from pydantic import BaseModel, Field
from typing import Optional

from domains.payroll_hr.payroll.settings import Settings


class PayLineItem(BaseModel):
    category: str = Field(
        description="Gross Pay / Regular Hours / Overtime / Federal Tax / "
        "State Tax / Social Security / Medicare / Health Deduction / 401k / "
        "Net Pay / etc."
    )
    amount: float = Field(description="Amount in USD")
    description: Optional[str] = Field(default="")


class ReasoningOutput(BaseModel):
    answer: str = Field(description="Direct answer to user question")
    line_items: list[PayLineItem] = Field(default=[])
    gross_pay: Optional[float] = Field(default=None)
    net_pay: Optional[float] = Field(default=None)
    confidence: float = Field(description="0 to 1")
    source_document: str = Field(description="Source PDF filename")
    flag: bool = Field(default=False)
    flag_reason: Optional[str] = Field(default=None)


class ReasoningAgent:
    def __init__(self):
        # temperature=0 -- structured extraction feeding validator_agent.py's
        # deterministic reconciliation, not creative generation. This was
        # the actual root cause of a real, observed test flake: with
        # temperature unset (Ollama defaults to ~0.8), the local model's
        # line-item extraction for a genuinely correct payslip
        # non-deterministically dropped a deduction category, making the
        # reconciliation check see a mismatch that wasn't real.
        if Settings.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(
                model=Settings.OLLAMA_MODEL, base_url=Settings.OLLAMA_BASE_URL, temperature=0,
            )
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL, temperature=0)

    @staticmethod
    def _render_history(history: list[dict] | None) -> str:
        """Renders prior conversation turns for the prompt -- including,
        when the caller seeds it, the auto-scan verdict as the first
        turn, so a follow-up like "why was this flagged?" can reference
        the actual verdict already reached instead of the LLM re-deriving
        one from scratch. Each turn: {"speaker": "user"|"assistant", "text": str}."""
        if not history:
            return ""
        lines = ["Previous conversation on this same document (most recent last):"]
        for turn in history:
            speaker = "You" if turn.get("speaker") == "assistant" else "User"
            lines.append(f"{speaker}: {turn.get('text', '')}")
        return "\n".join(lines) + "\n"

    def _build_prompt(
        self, query: str, chunks: list[dict], source_document: str | list[str] = None,
        history: list[dict] | None = None,
    ) -> str:
        context = "\n\n".join(
            [
                f"[{i+1}] {chunk['text']}\nSource: {chunk['source']}"
                for i, chunk in enumerate(chunks)
            ]
        )
        # A list means cross-document retrieval (see
        # core/rag/base_retriever_agent.py's retrieve() docstring) -- see
        # domains/insurance/agents/reasoning_agent.py's identical handling
        # for the rationale.
        if isinstance(source_document, list):
            actual_source = source_document[0] if source_document else (
                chunks[0].get("source", "unknown") if chunks else "unknown"
            )
            source_instruction = (
                f'The "source_document" field should be the filename of whichever '
                f'of these documents most directly backs your answer, one of: '
                f'{", ".join(source_document)}.'
            )
        else:
            actual_source = source_document or (
                chunks[0].get("source", "unknown") if chunks else "unknown"
            )
            source_instruction = (
                f'The "source_document" field MUST be exactly "{actual_source}". '
                f'Do not change it or invent a different filename.'
            )
        history_block = self._render_history(history)
        return f"""You are a payroll compliance analyst.
{history_block}Use the context below to answer the question. If the question refers
back to something in the previous conversation (e.g. "why was that flagged?",
"what rule does that violate?"), use that conversation to understand what
"that" refers to, and answer consistently with any verdict already given —
do not contradict a flag/reason already stated unless the context clearly
shows it was wrong.
Flag ONLY if you detect a genuine inconsistency: gross pay minus deductions
that doesn't match the stated net pay, an overtime rate that isn't 1.5x the
regular rate, a missing required withholding line, or duplicate line items.
Do NOT flag normal, correctly-calculated payslip line items just because
the amounts are large.
All amounts are in USD. Always use the $ symbol. Never use ₹ or any other
currency symbol.

Context:
{context}

Question: {query}

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.
The "answer" field MUST be a plain text string — NOT a list or array.
The "answer" must mention every line item listed in the "line_items" array. Do not omit any.
Write the "answer" as a human-readable sentence. Use $ symbol for USD amounts.
IMPORTANT: The "answer" text and the "line_items" array must be consistent. Every item in the array must appear in the answer. Do not summarise or select a subset.
The "category" field must be one of: Gross Pay, Regular Hours, Overtime, Federal Tax, State Tax, Social Security, Medicare, Health Deduction, 401k, Net Pay. Never leave it empty.
The "line_items" array must ONLY contain items directly relevant to the question asked. If the question is about deductions, only include deductions. Do not include unrelated items.
The "description" field must be copied exactly from the context.
If multiple inconsistencies exist, list ALL of them in "flag_reason" separated by commas. Do not pick just one. {source_instruction} Only flag issues you can directly evidence from the numbers in context — do not invent reasons not supported by the data.
{{
    "answer": "A clear, readable sentence answering the question. Example: 'Gross pay was $6,500.00, with $1,200.00 in federal tax and $400.00 in 401k contributions, for a net pay of $4,900.00.'",
    "line_items": [
        {{"category": "Gross Pay/Overtime/Federal Tax/etc.", "amount": 0.0, "description": "details"}}
    ],
    "gross_pay": 0.0,
    "net_pay": 0.0,
    "confidence": 0.9,
    "source_document": "{actual_source}",
    "flag": false,
    "flag_reason": "Comma-separated list of all inconsistencies found, or null if none"
}}"""

    def _parse_ollama_response(self, raw: str) -> ReasoningOutput:
        raw = raw.strip()

        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        import re

        raw = re.sub(r",\s*([\]}])", r"\1", raw)
        data = json.loads(raw)

        if isinstance(data.get("answer"), list):
            items = data["answer"]
            if items and isinstance(items[0], dict):
                data["answer"] = " | ".join(
                    f"{t.get('category', '')} ${t.get('amount', '')}" for t in items
                )
            else:
                data["answer"] = " ".join(str(i) for i in items)

        if not data.get("answer") or not isinstance(data["answer"], str):
            data["answer"] = "No answer could be extracted."

        if not data.get("line_items") or not isinstance(data["line_items"], list):
            data["line_items"] = []

        for li in data["line_items"]:
            if li.get("description") is None:
                li["description"] = ""
            if li.get("category") is None:
                li["category"] = "UNKNOWN"

        conf = data.get("confidence") or 0.8
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except ValueError:
                conf = 0.8
        data["confidence"] = max(0.0, min(1.0, float(conf)))

        return ReasoningOutput(**data)

    def reason(
        self, query: str, chunks: list[dict], source_document: str | list[str] = None,
        history: list[dict] | None = None,
    ) -> ReasoningOutput:
        prompt = self._build_prompt(query, chunks, source_document, history)
        if Settings.LLM_PROVIDER == "ollama":
            output = self._parse_ollama_response(self.llm.invoke(prompt).content)
        else:
            structured_llm = self.llm.with_structured_output(ReasoningOutput)
            output = structured_llm.invoke(prompt)

        return output

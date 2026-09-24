# domains/banking/documents/agents/reasoning_agent.py
# Ported from document_intelligence_system/agents/reasoning_agent.py —
# logic unchanged, import paths moved under domains.banking.documents.

from core.rag.prompt_safety import UNTRUSTED_CONTEXT_NOTICE
import json
from pydantic import BaseModel, Field
from typing import Optional

from domains.banking.documents.settings import Settings


class Transaction(BaseModel):
    date: str = Field(description="Transaction date YYYY-MM-DD")
    amount: float = Field(description="Amount in USD")
    type: str = Field(description="ATM / UPI / NEFT etc.")
    description: Optional[str] = Field(default="")


class ReasoningOutput(BaseModel):
    answer: str = Field(description="Direct answer to user question")
    transactions: list[Transaction] = Field(default=[])
    total_amount: Optional[float] = Field(default=None)
    confidence: float = Field(description="0 to 1")
    source_document: str = Field(description="Source PDF filename")
    anomaly_flag: bool = Field(default=False)
    anomaly_reason: Optional[str] = Field(default=None)


class ReasoningAgent:
    def __init__(self):
        # temperature=0 -- this agent does structured extraction that
        # downstream validation checks arithmetically (see
        # validator_agent.py), not creative generation. Left unset before
        # (Ollama's default is ~0.8), which let a real payroll-domain
        # reconciliation test flake -- a correct payslip's LLM extraction
        # non-deterministically dropped a line item and tripped a false
        # mismatch. Fixed here too even though banking documents' own test
        # suite didn't happen to catch it, since it's the same root cause.
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
        when the caller seeds it, the anomaly-scan verdict as the first
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
        return f"""You are a banking document analyst.
{history_block}Use the context below to answer the question. If the question refers
back to something in the previous conversation (e.g. "why was that flagged?",
"which rule does that break?"), use that conversation to understand what
"that" refers to, and answer consistently with any verdict already given —
do not contradict a flag/reason already stated unless the context clearly
shows it was wrong.
Flag anomalies ONLY if you detect genuinely suspicious patterns such as: unknown merchants, duplicate transactions, unusually large amounts compared to the rest, late-night ATM withdrawals, or rapid successive transactions. Do NOT flag normal retail purchases (grocery, food delivery, online shopping) as anomalies purely based on amount size.
All amounts are in USD. Always use the $ symbol. Never use ₹ or any other currency symbol.

{UNTRUSTED_CONTEXT_NOTICE}

Context:
{context}

Question: {query}

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.
The "answer" field MUST be a plain text string — NOT a list or array.
Do not include closing balance or opening balance as transactions.
The "answer" must mention ALL transactions listed in the "transactions" array. Do not omit any.
Write the "answer" as a human-readable sentence. Use $ symbol for USD amounts and readable dates (e.g. '4th March' not '04-Mar-2024').
IMPORTANT: The "answer" text and the "transactions" array must be consistent. Every transaction in the array must appear in the answer. Do not summarise or select a subset.
The "type" field must be one of: Debit, Credit, ACH, Wire, ATM. Never leave it empty.
The "transactions" array must ONLY contain transactions that are directly relevant to the question asked. If the question is about suspicious transactions, only include suspicious ones. If the question is about large withdrawals, only include large withdrawals. Do not include unrelated transactions.
The "description" field must be copied exactly from the context. Do not append channel names like "Online" or "ATM" to the description.
If multiple anomalies exist, list ALL of them in "anomaly_reason" separated by commas. Do not pick just one. {source_instruction} Only flag anomalies you can directly evidence from the transactions — do not invent reasons not supported by the data.
{{
    "answer": "A clear, readable sentence answering the question. Example: 'There were 3 large withdrawals in March: $18,000 on 2nd March (Rent), $60,000 on 4th March (ATM), and $50,000 on 20th March (ATM).'",
    "transactions": [
        {{"date": "YYYY-MM-DD", "amount": 0.0, "type": "Debit/Credit/ACH/Wire/ATM", "description": "details"}}
    ],
    "total_amount": 0.0,
    "confidence": 0.9,
    "source_document": "{actual_source}",
    "anomaly_flag": false,
    "anomaly_reason": "Comma-separated list of all anomalies found, or null if none"
}}"""

    def _parse_ollama_response(self, raw: str) -> ReasoningOutput:
        raw = raw.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break

        # Extract JSON object bounds
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        import re

        # Fix trailing commas before ] or }
        raw = re.sub(r",\s*([\]}])", r"\1", raw)
        data = json.loads(raw)

        # Repair: if answer is a list, flatten to string
        if isinstance(data.get("answer"), list):
            items = data["answer"]
            # If list of transaction-like dicts, summarise them
            if items and isinstance(items[0], dict):
                data["answer"] = " | ".join(
                    f"{t.get('date', '')} {t.get('description', '')} ${t.get('amount', '')}"
                    for t in items
                )
            else:
                # List of strings
                data["answer"] = " ".join(str(i) for i in items)

        # Repair: ensure answer is always a non-empty string
        if not data.get("answer") or not isinstance(data["answer"], str):
            data["answer"] = "No answer could be extracted."

        # Repair: transactions must be a list
        if not data.get("transactions") or not isinstance(data["transactions"], list):
            data["transactions"] = []  # ← this line was missing entirely

        # Repair: fix None fields in transactions
        for t in data["transactions"]:
            if t.get("description") is None:
                t["description"] = ""
            if t.get("type") is None:
                t["type"] = "UNKNOWN"

        # Repair: recalculate total_amount from actual transactions
        data["total_amount"] = (
            sum(t["amount"] for t in data["transactions"])
            if data["transactions"]
            else 0.0
        )

        # Repair: confidence must be float 0–1
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

        # Always recalculate total_amount from actual transactions
        if output.transactions:
            output.total_amount = sum(t.amount for t in output.transactions)

        return output

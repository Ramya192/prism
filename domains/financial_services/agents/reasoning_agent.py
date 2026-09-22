# domains/financial_services/agents/reasoning_agent.py
# Modeled on domains/banking/documents/agents/reasoning_agent.py's
# extraction-schema shape -- a wallet/exchange statement, like a bank
# statement, lists MANY transactions, not one artifact (unlike
# Insurance's single claim or Payroll's single payslip).
#
# Fields deliberately do NOT try to force a match against Tier 1
# (transaction_amount_usd/sender_wallet_age_days/is_cross_chain) or
# Tier 2's (notional_usd/round_trip_score/counterparty_reuse_ratio)
# REQUIRED_COLS -- round_trip_score and counterparty_reuse_ratio are
# backend risk-engine outputs no real statement would ever print, and
# sender_wallet_age_days is an account-level attribute, not a per-
# transaction line item. What IS honestly extractable (amount, date,
# transaction type, blockchain, counterparty) still enriches Tier 3's
# LLM reasoning even though it can never trigger a Tier 1/2 ML match --
# see FintechDetectorAgent.build_prompt()'s tier3 branch, which renders
# every field a record actually has into the prompt.

import json
from pydantic import BaseModel, Field
from typing import Optional

from domains.financial_services.settings import Settings


class WalletTransaction(BaseModel):
    date: str = Field(description="Transaction date YYYY-MM-DD")
    amount_usd: float = Field(description="Amount in USD")
    transaction_type: str = Field(description="Send / Receive / Swap / Bridge / Trade")
    blockchain: Optional[str] = Field(default="")
    counterparty: Optional[str] = Field(default="", description="Wallet address or exchange name")
    is_cross_chain: bool = Field(default=False)


class ReasoningOutput(BaseModel):
    answer: str = Field(description="Direct answer to user question")
    transactions: list[WalletTransaction] = Field(default=[])
    total_amount: Optional[float] = Field(default=None)
    confidence: float = Field(description="0 to 1")
    source_document: str = Field(description="Source document filename")
    flag: bool = Field(default=False)
    flag_reason: Optional[str] = Field(default=None)


class ReasoningAgent:
    def __init__(self):
        # temperature=0 -- structured extraction, not creative generation.
        # Same root-cause fix as every other domain's reasoning agent.
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
        return f"""You are a fintech/digital-asset fraud analyst.
{history_block}Use the context below to answer the question. If the question refers
back to something in the previous conversation (e.g. "why was that flagged?",
"which one was cross-chain?"), use that conversation to understand what
"that" refers to, and answer consistently with any verdict already given —
do not contradict a flag/reason already stated unless the context clearly
shows it was wrong.
Extract every transaction you can find, exactly as stated — do not guess
or invent a value that isn't in the context. Flag ONLY if you detect a
genuine inconsistency directly evidenced in the text: a duplicate
transaction, a counterparty that appears with two different amounts for
the same timestamp, or an internally contradictory total.
All amounts are in USD. Always use the $ symbol. Never use another
currency symbol.

Context:
{context}

Question: {query}

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.
The "answer" field MUST be a plain text string — NOT a list or array.
The "answer" must mention every transaction listed in the "transactions" array.
Write the "answer" as a human-readable sentence.
The "transaction_type" field must be one of: Send, Receive, Swap, Bridge, Trade.
The "transactions" array must ONLY contain transactions directly relevant to the question asked.
{source_instruction}
Only flag issues you can directly evidence from the context — do not invent reasons not supported by the data.
{{
    "answer": "A clear, readable sentence answering the question.",
    "transactions": [
        {{"date": "YYYY-MM-DD", "amount_usd": 0.0, "transaction_type": "Send/Receive/Swap/Bridge/Trade", "blockchain": "e.g. Ethereum", "counterparty": "address or exchange name", "is_cross_chain": false}}
    ],
    "total_amount": 0.0,
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

        if not data.get("answer") or not isinstance(data["answer"], str):
            data["answer"] = "No answer could be extracted."

        if not data.get("transactions") or not isinstance(data["transactions"], list):
            data["transactions"] = []

        for t in data["transactions"]:
            if t.get("blockchain") is None:
                t["blockchain"] = ""
            if t.get("counterparty") is None:
                t["counterparty"] = ""
            if t.get("is_cross_chain") is None:
                t["is_cross_chain"] = False

        data["total_amount"] = (
            sum(t["amount_usd"] for t in data["transactions"])
            if data["transactions"]
            else 0.0
        )

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

        if output.transactions:
            output.total_amount = sum(t.amount_usd for t in output.transactions)

        return output

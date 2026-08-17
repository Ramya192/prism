# domains/bfsi_documents/agents/reasoning_agent.py
# Ported from document_intelligence_system/agents/reasoning_agent.py —
# logic unchanged, import paths moved under domains.bfsi_documents.

import json
from pydantic import BaseModel, Field
from typing import Optional

from domains.bfsi_documents.settings import Settings


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
        if Settings.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(
                model=Settings.OLLAMA_MODEL, base_url=Settings.OLLAMA_BASE_URL
            )
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL)

    def _build_prompt(
        self, query: str, chunks: list[dict], source_document: str = None
    ) -> str:
        context = "\n\n".join(
            [
                f"[{i+1}] {chunk['text']}\nSource: {chunk['source']}"
                for i, chunk in enumerate(chunks)
            ]
        )
        actual_source = source_document or (
            chunks[0].get("source", "unknown") if chunks else "unknown"
        )
        return f"""You are a BFSI document analyst.
Use the context below to answer the question.
Flag anomalies ONLY if you detect genuinely suspicious patterns such as: unknown merchants, duplicate transactions, unusually large amounts compared to the rest, late-night ATM withdrawals, or rapid successive transactions. Do NOT flag normal retail purchases (grocery, food delivery, online shopping) as anomalies purely based on amount size.
All amounts are in USD. Always use the $ symbol. Never use ₹ or any other currency symbol.

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
If multiple anomalies exist, list ALL of them in "anomaly_reason" separated by commas. Do not pick just one. The "source_document" field MUST be exactly "{actual_source}". Do not change it or invent a different filename. Only flag anomalies you can directly evidence from the transactions — do not invent reasons not supported by the data.
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
        self, query: str, chunks: list[dict], source_document: str = None
    ) -> ReasoningOutput:
        prompt = self._build_prompt(query, chunks, source_document)
        if Settings.LLM_PROVIDER == "ollama":
            output = self._parse_ollama_response(self.llm.invoke(prompt).content)
        else:
            structured_llm = self.llm.with_structured_output(ReasoningOutput)
            output = structured_llm.invoke(prompt)

        # Always recalculate total_amount from actual transactions
        if output.transactions:
            output.total_amount = sum(t.amount for t in output.transactions)

        return output

# domains/insurance/agents/reasoning_agent.py
# Modeled on domains/payroll_hr/payroll/agents/reasoning_agent.py's
# extraction-schema shape. Fields mirror tools/claim_scorer.py's
# REQUIRED_COLS and the subset of its FEATURE_COLS a real claim
# document/EOB can actually state -- Number_of_Claims_Per_Provider_Monthly,
# Length_of_Stay, Chronic_Condition_Flag, and Prior_Visits_12m are
# provider/patient-HISTORY aggregates, not something on a single claim
# form, so they're deliberately not extraction targets here; the scorer
# already defaults missing optional columns to 0 (see its .get(col, 0)
# pattern), an accepted approximation, not a new one introduced here.

import json
from pydantic import BaseModel, Field
from typing import Optional

from domains.insurance.settings import Settings


class ExtractedClaim(BaseModel):
    answer: str = Field(description="Direct answer to user question")
    Claim_Amount: Optional[float] = Field(default=None, description="Total billed amount in USD")
    Approved_Amount: Optional[float] = Field(default=None, description="Amount actually approved/paid in USD")
    Service_Date: Optional[str] = Field(default=None, description="Date of service, YYYY-MM-DD")
    Claim_Date: Optional[str] = Field(default=None, description="Date the claim was submitted, YYYY-MM-DD")
    Patient_Age: Optional[int] = Field(default=None)
    Patient_Gender: Optional[str] = Field(default=None, description="Male / Female")
    Insurance_Type: Optional[str] = Field(default=None, description="Medicare / Medicaid / Private / etc.")
    Provider_Specialty: Optional[str] = Field(default=None)
    Visit_Type: Optional[str] = Field(default=None, description="Inpatient / Outpatient / Emergency / etc.")
    confidence: float = Field(description="0 to 1")
    source_document: str = Field(description="Source document filename")
    flag: bool = Field(default=False)
    flag_reason: Optional[str] = Field(default=None)


class ReasoningAgent:
    def __init__(self):
        # temperature=0 -- structured extraction, not creative generation.
        # Same root-cause fix as payroll's/bfsi_documents' reasoning
        # agents (see their docstrings for the real reconciliation-flake
        # this fixed there).
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
        # core/rag/base_retriever_agent.py's retrieve() docstring) -- don't
        # force one exact filename onto every answer, since the chunks
        # backing it may genuinely come from more than one of them. A
        # single string (or None) reproduces the exact prior behavior.
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
        return f"""You are an insurance claims compliance analyst.
{history_block}Use the context below to answer the question. If the question refers
back to something in the previous conversation (e.g. "why was that flagged?",
"what rule does that violate?"), use that conversation to understand what
"that" refers to, and answer consistently with any verdict already given —
do not contradict a flag/reason already stated unless the context clearly
shows it was wrong.
Extract every claim field you can find, exactly as stated — do not guess or
invent a value that isn't in the context; leave a field null if it isn't
present. Flag ONLY if you detect a genuine inconsistency directly evidenced
in the text: an approved amount greater than the billed claim amount, a
claim date before the service date, or a diagnosis/procedure code
referenced that isn't defined anywhere in the document.
All amounts are in USD. Always use the $ symbol. Never use ₹ or any other
currency symbol.

Context:
{context}

Question: {query}

Respond ONLY with a valid JSON object. No markdown, no explanation, no extra text.
The "answer" field MUST be a plain text string — NOT a list or array.
Write the "answer" as a human-readable sentence.
Dates must be YYYY-MM-DD or null if not stated.
{source_instruction}
Only flag issues you can directly evidence from the context — do not invent reasons not supported by the data.
{{
    "answer": "A clear, readable sentence answering the question.",
    "Claim_Amount": 0.0,
    "Approved_Amount": 0.0,
    "Service_Date": "YYYY-MM-DD",
    "Claim_Date": "YYYY-MM-DD",
    "Patient_Age": 0,
    "Patient_Gender": "Male/Female",
    "Insurance_Type": "Medicare/Medicaid/Private",
    "Provider_Specialty": "e.g. Cardiology",
    "Visit_Type": "Inpatient/Outpatient/Emergency",
    "confidence": 0.9,
    "source_document": "{actual_source}",
    "flag": false,
    "flag_reason": "Comma-separated list of all inconsistencies found, or null if none"
}}"""

    def _parse_ollama_response(self, raw: str) -> ExtractedClaim:
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

        conf = data.get("confidence") or 0.8
        if isinstance(conf, str):
            try:
                conf = float(conf)
            except ValueError:
                conf = 0.8
        data["confidence"] = max(0.0, min(1.0, float(conf)))

        return ExtractedClaim(**data)

    def reason(
        self, query: str, chunks: list[dict], source_document: str | list[str] = None,
        history: list[dict] | None = None,
    ) -> ExtractedClaim:
        prompt = self._build_prompt(query, chunks, source_document, history)
        if Settings.LLM_PROVIDER == "ollama":
            output = self._parse_ollama_response(self.llm.invoke(prompt).content)
        else:
            structured_llm = self.llm.with_structured_output(ExtractedClaim)
            output = structured_llm.invoke(prompt)

        return output

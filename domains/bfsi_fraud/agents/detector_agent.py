# domains/bfsi_fraud/agents/detector_agent.py
# Ported from fraud_detection/agents/detector_agent.py — logic unchanged,
# only the settings/tools import paths moved under domains.bfsi_fraud.
#
# Hybrid DetectorAgent: Rule-based -> ML Scorer -> LLM (borderline only)
#
# Decision flow:
#   1. Rule-based pre-filter  -> catches obvious fraud instantly (no API call)
#   2. ML Scorer (RF)         -> fast-path for clear fraud/legit (no API call)
#   3. LLM                    -> only called for borderline cases (0.3 - 0.7)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.bfsi_fraud.settings import OPENAI_API_KEY, MODEL_NAME
from domains.bfsi_fraud.tools.ml_scorer import MLScorer


class FraudDetectorAgent:

    VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, name, data_path="domains/bfsi_fraud/data/transactions_balanced.csv"):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        print("  [DetectorAgent] Loading ML scorer...")
        self.scorer = MLScorer(data_path=data_path)

    # ── IMPROVEMENT 4: Robust response parser ───────────────────────────
    # Replaces fragile `"HIGH" in response` string search.
    # Extracts Risk Level, Action, and Reason from the LLM response safely.
    @staticmethod
    def parse_response(response: str) -> dict:
        """
        Parses structured LLM response into a dict with keys:
          risk_level : "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
          action     : "APPROVE" | "FLAG" | "BLOCK" | "UNKNOWN"
          reason     : str

        Handles variations in:
          - capitalisation  (high / High / HIGH)
          - extra whitespace or punctuation
          - missing fields (returns "UNKNOWN" safely)
          - extra text before/after the value
        """
        result = {"risk_level": "UNKNOWN", "action": "UNKNOWN", "reason": ""}

        for line in response.strip().splitlines():
            line = line.strip()
            lower = line.lower()

            if lower.startswith("risk level"):
                # Extract everything after the first colon
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    # Take only the first word in case of extra text
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    if first_word in FraudDetectorAgent.VALID_RISK_LEVELS:
                        result["risk_level"] = first_word
                    else:
                        result["risk_level"] = "UNKNOWN"

            elif lower.startswith("action"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    if first_word in {"APPROVE", "FLAG", "BLOCK"}:
                        result["action"] = first_word
                    else:
                        result["action"] = "FLAG"   # safe default

            elif lower.startswith("reason"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    result["reason"] = parts[1].strip()

        return result

    @staticmethod
    def is_fraud(parsed: dict) -> bool:
        """
        Returns True if the parsed response indicates fraud.
        MEDIUM and HIGH both count as fraud — safer for a fraud detection system.
        UNKNOWN defaults to True (fail safe — flag rather than miss fraud).
        """
        return parsed["risk_level"] in {"MEDIUM", "HIGH", "UNKNOWN"}

    # ── LAYER 1: Rule-based pre-filter ──────────────────────────────────
    # Static — doesn't touch self, so eval_harness.py can call this
    # directly against the full holdout set without paying for an
    # OpenAI client + MLScorer retrain just to reach the rule layer.
    @staticmethod
    def rule_based_filter(transaction):
        amount = transaction.get("Amount", 0)
        hour   = transaction.get("hour", 12)

        if amount == 0.0:
            return (
                "Risk Level: HIGH\n"
                "Reason: Zero-dollar transaction indicates a card verification attack.\n"
                "Action: BLOCK"
            )
        if 0 < amount <= 5.0 and 0 <= hour <= 5:
            return (
                "Risk Level: HIGH\n"
                "Reason: Micro-transaction during night hours — classic card-testing pattern.\n"
                "Action: BLOCK"
            )
        if amount > 1000 and 0 <= hour <= 5:
            return (
                "Risk Level: HIGH\n"
                "Reason: Large transaction amount during overnight hours is highly suspicious.\n"
                "Action: BLOCK"
            )
        return None

    # ── LAYER 2: ML fast-path ────────────────────────────────────────────
    def ml_filter(self, transaction):
        score = self.scorer.score(transaction)

        if score >= MLScorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model fraud probability {score:.0%} — strongly matches fraud patterns in training data.\n"
                f"Action: BLOCK"
            )
        if score <= MLScorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model fraud probability {score:.0%} — strongly matches legitimate transaction patterns.\n"
                f"Action: APPROVE"
            )
        return None

    # ── LAYER 3: LLM prompt (borderline cases only) ──────────────────────
    def build_prompt(self, transaction, context=None, ml_score=None):
        context_str = context if context else "No historical context available"

        ml_hint = ""
        if ml_score is not None:
            ml_hint = f"""
            ML Model Assessment:
            - Fraud probability score: {ml_score:.0%}
            - This score is BORDERLINE (between 30% and 70%) — human reasoning needed.
            - Weight this alongside the transaction features below.
            """

        return f"""
            You are a fraud detection expert at a bank.
            A machine learning model has flagged this transaction as borderline.
            Your job is to make the final call using your expert reasoning.

            Key fraud indicators:
            - Exactly $0.00 transactions ALWAYS indicate card verification attacks
            - Amounts between $0.01 and $5.00 at hours 0-5 are suspicious
            - Transactions between midnight and 5am (hour 0-5) are HIGH RISK
            - Amounts over $500 at unusual hours are suspicious
            - Legitimate transactions typically have amounts between $10-$500
            - Transactions at hour 23 (11pm) with amounts over $200 are suspicious

            Historical context:
            {context_str}
            {ml_hint}

            Transaction details:
            - Amount: ${transaction['Amount']}
            - Hour of day: {transaction['hour']} (0=midnight, 23=11pm)
            - Time since first transaction: {transaction['Time']} seconds

            Respond in EXACTLY this format (no extra text):
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """

    # ── Main entry point ─────────────────────────────────────────────────
    def analyse(self, transaction, context=None):
        self.cases_reviewed += 1

        # Layer 1: Rules
        rule_result = self.rule_based_filter(transaction)
        if rule_result:
            return rule_result

        # Layer 2: ML scorer
        ml_result = self.ml_filter(transaction)
        if ml_result:
            return ml_result

        # Layer 3: LLM — borderline only
        ml_score = self.scorer.score(transaction)
        try:
            messages = [
                SystemMessage(content="You are a bank fraud detection expert. Be concise and precise."),
                HumanMessage(content=self.build_prompt(transaction, context, ml_score))
            ]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            print(f"  [DetectorAgent ERROR] {e}")
            return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} transactions")

# domains/banking/fraud/agents/analyst_agent.py
# Ported from fraud_detection/agents/analyst_agent.py — logic unchanged.

import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.banking.fraud.settings import OPENAI_API_KEY, MODEL_NAME


class AnalystAgent:

    MAX_RETRIES = 2
    RETRY_DELAY = 3   # seconds between retries

    def __init__(self, name):
        self.name = name
        self.investigations = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )

    def investigate(self, transaction, detector_response):
        self.investigations += 1

        prompt = f"""
            You are a senior fraud analyst at a bank.
            The fraud detector has flagged this transaction as HIGH RISK.

            Your job is to provide a detailed investigation report.

            Flagged transaction:
            - Amount: ${transaction["Amount"]}
            - Hour of day: {transaction["hour"]} (0=midnight, 23=11pm)
            - Time in dataset: {transaction["Time"]} seconds

            Detector's initial assessment:
            {detector_response}

            Provide a structured investigation report with:
            1. Fraud Pattern: <type>
            2. Risk Factors: List exactly 3 risk factors present
            3. Recommended Action: BLOCK / ESCALATE / MONITOR
            4. Confidence: LOW / MEDIUM / HIGH
            5. Next Steps: one sentence on what bank should do

            Respond in exactly this format:
            Fraud Pattern: <type>
            Risk Factors: <factor1>, <factor2>, <factor3>
            Recommended Action: BLOCK / ESCALATE / MONITOR
            Confidence: LOW / MEDIUM / HIGH
            Next Steps: <one sentence>
        """

        messages = [
            SystemMessage(content="You are a senior bank fraud analyst. Be precise and professional."),
            HumanMessage(content=prompt)
        ]

        # ── Retry logic ────────────────────────────────────────────────
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.llm.invoke(messages)
                return response.content

            except Exception as e:
                print(f"  [AnalystAgent] Attempt {attempt}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    print(f"  [AnalystAgent] Retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    print(f"  [AnalystAgent] All retries exhausted.")
                    return (
                        "Fraud Pattern: Unknown\n"
                        "Risk Factors: Analysis unavailable\n"
                        "Recommended Action: ESCALATE\n"
                        "Confidence: LOW\n"
                        "Next Steps: Manual review required"
                    )

    def status(self):
        print(f"{self.name} has investigated {self.investigations} cases")

# domains/bfsi_fraud/agents/alert_agent.py
# Ported from fraud_detection/agents/alert_agent.py — logic unchanged.

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.bfsi_fraud.settings import OPENAI_API_KEY, MODEL_NAME
from datetime import datetime

class AlertAgent:

    def __init__(self,name):
        self.name = name
        self.alerts_generated = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )

    def generate_alert(self, transaction, detector_response, analyst_response):
        self.alerts_generated += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            prompt = f"""
                You are a bank alert system.
                Generate a  formal fraud alert notification.

                Timestamp: {timestamp}
                Transaction Amount: ${transaction["Amount"]}
                Hour of Transaction: {transaction["hour"]}

                Detector Assessment: {detector_response}

                Analyst Investigation: {analyst_response}

                Generate a formal bank alert exactly in this format:
                ALERT ID: <generate a randon 8- digit number>
                TIMESTAMP: {timestamp}
                SEVERITY: <CRITICAL/HIGH/MEDIUM>
                TRANSACTION AMOUNT: ${transaction["Amount"]}
                FRAUD TYPE: <one phrase>
                SUMMARY: <one sentence summary for the bank officer>
                IMMEDIATE ACTION: <exactly what the bank should do right now>
                CUSTOMER IMPACT: <how this affects the customer>
                """
            messages =[
                SystemMessage(content="You are formal bank alert system. Be precise and professional."),
                HumanMessage(content=prompt)
            ]

            response= self.llm.invoke(messages)
            return response.content

        except Exception as e:
            print(f"  [AlertAgent ERROR] {e}")
            return f"ALERT: Transaction of ${transaction['Amount']} flagged for manual review\nAction: ESCALATE TO HUMAN ANALYST"

    def status(self):
        print(f"{self.name} has generated {self.alerts_generated} alerts")

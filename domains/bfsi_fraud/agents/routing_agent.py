# domains/bfsi_fraud/agents/routing_agent.py
# Ported from fraud_detection/agents/routing_agent.py — logic unchanged.

import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.bfsi_fraud.settings import OPENAI_API_KEY, MODEL_NAME


class RoutingAgent:

    MAX_RETRIES = 2
    RETRY_DELAY = 3   # seconds between retries

    def __init__(self, name):
        self.name = name
        self.routes_made  = 0
        self.it_routes    = 0
        self.fraud_routes = 0
        self.support_routes = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )

    def route(self, transaction, analyst_response):
        self.routes_made += 1

        prompt = f"""
        You are a bank operations routing system.
        Based on the transaction details and analyst findings,
        route this case to the correct department.

        Transaction:
        - Amount: ${transaction['Amount']}
        - Hour: {transaction['hour']}

        Analyst findings:
        {analyst_response}

        Routing rules:
        - IT Department: network failures, transaction errors,
          system issues, failed transactions
        - Fraud Department: suspicious patterns, card verification
          attacks, unusual amounts, high risk flags
        - General Support: normal customer queries, legitimate
          large purchases, routine account activity

        Respond in exactly this format:
        Department: <IT/FRAUD/GENERAL_SUPPORT>
        Priority: <HIGH/MEDIUM/LOW>
        Reason: <one sentence>
        Action Required: <what the department should do>
        """

        messages = [
            SystemMessage(content="You are a bank routing system. Route cases to correct departments."),
            HumanMessage(content=prompt)
        ]

        # ── Retry logic ────────────────────────────────────────────────
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.llm.invoke(messages)
                result   = response.content

                # Track department counts
                first_line = result.split('\n')[0].upper()
                if "IT" in first_line:
                    self.it_routes += 1
                elif "FRAUD" in first_line:
                    self.fraud_routes += 1
                else:
                    self.support_routes += 1

                return result

            except Exception as e:
                print(f"  [RoutingAgent] Attempt {attempt}/{self.MAX_RETRIES} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    print(f"  [RoutingAgent] Retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    print(f"  [RoutingAgent] All retries exhausted.")
                    self.fraud_routes += 1
                    return (
                        "Department: FRAUD\n"
                        "Priority: MEDIUM\n"
                        "Reason: Routing failed — defaulting to Fraud team\n"
                        "Action Required: Manual review"
                    )

    def status(self):
        print(f"{self.name} made {self.routes_made} routing decisions")
        print(f"  → IT Department:      {self.it_routes}")
        print(f"  → Fraud Department:   {self.fraud_routes}")
        print(f"  → General Support:    {self.support_routes}")

# domains/banking/documents/agents/validator_agent.py
# Ported from document_intelligence_system/agents/validator_agent.py —
# logic unchanged, import path moved under domains.banking.documents.

from domains.banking.documents.agents.reasoning_agent import ReasoningOutput


class ValidatorAgent:

    def validate(self, output: ReasoningOutput) -> dict:
        errors = []

        if len(output.answer) == 0:
            errors.append("answer is empty")

        if not (0 <= output.confidence <= 1):
            errors.append("confidence out of range")

        if output.anomaly_flag and output.anomaly_reason is None:
            errors.append("anomaly_flag set but no reason given")

        if output.transactions and output.total_amount is None:
            errors.append("transactions present but total_amount missing")

        if errors:
            return {"status": "invalid", "errors": errors}
        return {"status": "valid", "data": output}

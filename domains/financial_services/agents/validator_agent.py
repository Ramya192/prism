# domains/financial_services/agents/validator_agent.py
# Same shape as Banking's validator_agent.py -- structural checks only.
# No deterministic reconciliation check exists here the way Payroll's
# does, because a wallet/exchange statement has no gross/net-style
# invariant to recompute -- each transaction is independent, unlike a
# payslip's earnings-minus-deductions relationship.

from domains.financial_services.agents.reasoning_agent import ReasoningOutput


class ValidatorAgent:

    def validate(self, output: ReasoningOutput) -> dict:
        errors = []

        if len(output.answer) == 0:
            errors.append("answer is empty")

        if not (0 <= output.confidence <= 1):
            errors.append("confidence out of range")

        if output.flag and output.flag_reason is None:
            errors.append("flag is set but no reason given")

        if output.transactions and output.total_amount is None:
            errors.append("transactions present but total_amount missing")

        if errors:
            return {"status": "invalid", "errors": errors}
        return {"status": "valid", "data": output}

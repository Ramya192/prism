# domains/payroll_hr/payroll/agents/validator_agent.py
# Modeled on domains/banking/documents/agents/validator_agent.py's shape
# checks, plus the actual differentiator for this domain: a deterministic
# reconciliation of gross pay minus deductions against the stated net pay.
# The reasoning agent's own "flag" is an LLM's read of the text; this check
# is arithmetic, not judgment — it catches a real miscalculation an LLM
# could miss (or hallucinate a false alarm about) by recomputing the number
# and comparing it directly.

from domains.payroll_hr.payroll.agents.reasoning_agent import ReasoningOutput

# Only these categories are actually subtracted from gross pay. "Gross Pay",
# "Regular Hours", "Overtime", and "Net Pay" are pay/earnings figures, not
# deductions — summing them in here would double-count against gross pay
# and produce a false mismatch.
_DEDUCTION_CATEGORIES = {
    "Federal Tax",
    "State Tax",
    "Social Security",
    "Medicare",
    "Health Deduction",
    "401k",
}

_RECONCILIATION_TOLERANCE = 0.01  # cents-level rounding slack


class ValidatorAgent:

    def validate(self, output: ReasoningOutput) -> dict:
        errors = []

        if len(output.answer) == 0:
            errors.append("answer is empty")

        if not (0 <= output.confidence <= 1):
            errors.append("confidence out of range")

        if output.flag and output.flag_reason is None:
            errors.append("flag is set but no reason given")

        # Deterministic reconciliation — only meaningful when the answer
        # actually carries a full breakdown (gross + net + at least one
        # deduction line item). A query scoped to one line item (e.g. "what
        # was my federal tax?") won't have enough data for this to mean
        # anything, and that's fine — it's not an error, just not checkable.
        deductions = [
            li for li in output.line_items if li.category in _DEDUCTION_CATEGORIES
        ]
        if output.gross_pay is not None and output.net_pay is not None and deductions:
            deductions_total = sum(li.amount for li in deductions)
            expected_net = output.gross_pay - deductions_total
            mismatch = abs(expected_net - output.net_pay) > _RECONCILIATION_TOLERANCE
            if mismatch:
                reason = (
                    f"Reconciliation mismatch: gross pay ${output.gross_pay:,.2f} "
                    f"minus deductions ${deductions_total:,.2f} = ${expected_net:,.2f}, "
                    f"but the stated net pay is ${output.net_pay:,.2f}."
                )
                output.flag = True
                output.flag_reason = (
                    f"{output.flag_reason}; {reason}" if output.flag_reason else reason
                )

        if errors:
            return {"status": "invalid", "errors": errors}
        return {"status": "valid", "data": output}

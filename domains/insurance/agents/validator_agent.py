# domains/insurance/agents/validator_agent.py
# Modeled on payroll's validator_agent.py shape: basic structural checks
# plus a deterministic sanity check the LLM's own "flag" judgment
# shouldn't be solely responsible for -- an approved amount exceeding the
# billed claim amount, or a claim submitted before the service date,
# are both arithmetically/logically impossible for a legitimate claim,
# not a judgment call.

from datetime import date

from domains.insurance.agents.reasoning_agent import ExtractedClaim


class ValidatorAgent:

    def validate(self, output: ExtractedClaim) -> dict:
        errors = []

        if len(output.answer) == 0:
            errors.append("answer is empty")

        if not (0 <= output.confidence <= 1):
            errors.append("confidence out of range")

        if output.flag and output.flag_reason is None:
            errors.append("flag is set but no reason given")

        extra_reasons = []

        if output.Claim_Amount is not None and output.Approved_Amount is not None:
            if output.Approved_Amount > output.Claim_Amount:
                extra_reasons.append(
                    f"Approved amount ${output.Approved_Amount:,.2f} exceeds the billed "
                    f"claim amount ${output.Claim_Amount:,.2f} — not possible for a "
                    "legitimate claim."
                )

        if output.Service_Date and output.Claim_Date:
            try:
                service = date.fromisoformat(output.Service_Date)
                claimed = date.fromisoformat(output.Claim_Date)
                if claimed < service:
                    extra_reasons.append(
                        f"Claim date {output.Claim_Date} is before the service date "
                        f"{output.Service_Date} — a claim cannot be submitted before "
                        "the service it bills for occurred."
                    )
            except ValueError:
                pass  # unparseable date -- not this validator's job to fix extraction

        if extra_reasons:
            output.flag = True
            combined = "; ".join(extra_reasons)
            output.flag_reason = f"{output.flag_reason}; {combined}" if output.flag_reason else combined

        if errors:
            return {"status": "invalid", "errors": errors}
        return {"status": "valid", "data": output}

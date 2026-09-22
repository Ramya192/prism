# domains/insurance/agents/healthcare_detector_agent.py
# HealthcareDetectorAgent — the claim-level counterpart to bfsi_fraud's
# FraudDetectorAgent and payroll's PayrollDetectorAgent, same rules ->
# tiered ML -> LLM shape.
#
# Tier 1 (nudratabbas' Claim_Amount/Approved_Amount/
# Days_Between_Service_and_Claim schema) and Tier 2 (tejalaveti2306's
# Policy_Number/Provider_Patient_Distance_Miles schema — see
# data/prepare_tier2_data.py's module docstring for the search story and
# its honest unrealistic-amount-scale caveat) are two genuinely different,
# independently-trained real datasets, same reasoning as bfsi_fraud/
# payroll's tiering.
#
#   Tier 1 — rules: approved amount exceeding claimed, or negative
#            amounts/days. ML (ClaimFraudScorer, LightGBM): AUC 0.998,
#            F1 0.932 -- see that file's module docstring for the
#            RandomForest/XGBoost/LightGBM benchmark this came from.
#            Top feature: Days_Between_Service_and_Claim (fraud submitted
#            much faster than legitimate claims).
#   Tier 2 — rules: claim filed after policy expiration, or service dated
#            after the claim itself. ML (ClaimGeneralizableScorer): AUC
#            0.871, F1 0.673-0.83 depending on threshold.
#   Tier 3 — neither schema recognized: straight to the LLM.
#
# Both tiers' rules checks are genuine business-logic impossibilities,
# not statistical signal — no precision/recall/F1 attached to either
# rules layer on purpose. Verified: neither fires on either tier's own
# clean training/test data (same honest situation as payroll Tier 2) —
# kept for correctness against malformed/adversarial real input anyway.

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.insurance.settings import Settings
from domains.insurance.tools.claim_scorer import ClaimFraudScorer
from domains.insurance.tools.generalizable_scorer import ClaimGeneralizableScorer

RULE_TOLERANCE = 0.01


class HealthcareDetectorAgent:

    VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, name, data_path="domains/insurance/data/train.csv"):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        print("  [HealthcareDetectorAgent] Loading ML scorer (tier 1: Claim_Amount/Approved_Amount schema)...")
        self.scorer = ClaimFraudScorer(data_path=data_path)
        print("  [HealthcareDetectorAgent] Loading ML scorer (tier 2: Policy_Number/distance schema)...")
        self.scorer_generalizable = ClaimGeneralizableScorer()

    @staticmethod
    def parse_response(response: str) -> dict:
        """Same robust-parsing contract as the other domains' detector agents."""
        result = {"risk_level": "UNKNOWN", "action": "UNKNOWN", "reason": ""}
        for line in response.strip().splitlines():
            line = line.strip()
            lower = line.lower()
            if lower.startswith("risk level"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    result["risk_level"] = first_word if first_word in HealthcareDetectorAgent.VALID_RISK_LEVELS else "UNKNOWN"
            elif lower.startswith("action"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    result["action"] = first_word if first_word in {"APPROVE", "FLAG", "BLOCK"} else "FLAG"
            elif lower.startswith("reason"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    result["reason"] = parts[1].strip()
        return result

    @staticmethod
    def is_fraud(parsed: dict) -> bool:
        return parsed["risk_level"] in {"MEDIUM", "HIGH", "UNKNOWN"}

    # ── Schema tiering ──────────────────────────────────────────────────────
    @staticmethod
    def detect_tier(claim: dict) -> str:
        """See the module docstring above for what each tier means."""
        if all(c in claim for c in ClaimFraudScorer.REQUIRED_COLS):
            return "tier1"
        if all(c in claim for c in ClaimGeneralizableScorer.REQUIRED_COLS):
            return "tier2"
        return "tier3"

    @staticmethod
    def has_recognizable_schema(claim: dict) -> bool:
        return HealthcareDetectorAgent.detect_tier(claim) != "tier3"

    # ── LAYER 1: Rules — genuine impossibilities, not statistical signal ──
    @staticmethod
    def _rule_check_tier1(claim: dict):
        claim_amt = claim.get("Claim_Amount")
        approved_amt = claim.get("Approved_Amount")
        days = claim.get("Days_Between_Service_and_Claim")

        if claim_amt is None:
            return None

        if claim_amt < 0 or (approved_amt is not None and approved_amt < 0):
            return (
                "Risk Level: HIGH\n"
                "Reason: Negative claim or approved amount — not a valid billing state.\n"
                "Action: BLOCK"
            )
        if approved_amt is not None and approved_amt > claim_amt + RULE_TOLERANCE:
            return (
                "Risk Level: HIGH\n"
                f"Reason: Approved amount ({approved_amt:.2f}) exceeds the claimed amount "
                f"({claim_amt:.2f}) — a payer cannot approve more than what was billed.\n"
                "Action: BLOCK"
            )
        if days is not None and days < 0:
            return (
                "Risk Level: HIGH\n"
                "Reason: Negative days-between-service-and-claim — claim submitted before service occurred.\n"
                "Action: BLOCK"
            )
        return None

    @staticmethod
    def _rule_check_tier2(claim: dict):
        claim_amt = claim.get("Claim_Amount")
        claim_date = claim.get("Claim_Date")
        service_date = claim.get("Service_Date")
        expiration_date = claim.get("Policy_Expiration_Date")

        if claim_amt is not None and claim_amt < 0:
            return (
                "Risk Level: HIGH\n"
                "Reason: Negative claim amount — not a valid billing state.\n"
                "Action: BLOCK"
            )
        if claim_date and service_date and str(service_date) > str(claim_date):
            return (
                "Risk Level: HIGH\n"
                "Reason: Service date is after the claim date — claim filed before service occurred.\n"
                "Action: BLOCK"
            )
        if claim_date and expiration_date and str(claim_date) > str(expiration_date):
            return (
                "Risk Level: HIGH\n"
                "Reason: Claim filed after the policy's expiration date.\n"
                "Action: BLOCK"
            )
        return None

    @staticmethod
    def rule_based_filter(claim: dict, tier: str = "tier1"):
        if tier == "tier1":
            return HealthcareDetectorAgent._rule_check_tier1(claim)
        if tier == "tier2":
            return HealthcareDetectorAgent._rule_check_tier2(claim)
        return None

    # ── LAYER 2: ML fraud scorer ──────────────────────────────────────────
    def ml_filter(self, claim: dict, tier: str = "tier1"):
        if tier == "tier1":
            scorer, desc = self.scorer, "how quickly this claim was submitted after service relative to typical claims"
        else:
            scorer, desc = self.scorer_generalizable, "this claim's overall risk pattern (procedures, provider distance, claim history)"
        score = scorer.score(claim)

        if score >= scorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model — {score:.0%} fraud probability, primarily driven by {desc}.\n"
                f"Action: BLOCK"
            )
        if score <= scorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model — {score:.0%} fraud probability; matches typical claim patterns.\n"
                f"Action: APPROVE"
            )
        return None

    # ── LAYER 3: LLM prompt ────────────────────────────────────────────────
    def build_prompt(self, claim: dict, context=None, ml_score=None, tier: str = "tier1"):
        context_str = context if context else "No historical context available"

        if tier == "tier3":
            fields = "\n".join(f"- {k}: {v}" for k, v in claim.items()) or "(no fields provided)"
            return f"""
                You are a healthcare claims fraud investigator.
                This input doesn't match a recognizable claim schema, so no
                rule-based or ML score could be computed. Use your own judgement.

                Historical context:
                {context_str}

                Claim details (exactly as provided):
                {fields}

                Respond in EXACTLY this format (no extra text):
                Risk Level: <LOW/MEDIUM/HIGH>
                Reason: <one sentence>
                Action: <APPROVE/FLAG/BLOCK>
            """

        if tier == "tier2":
            scorer = self.scorer_generalizable
            ml_hint = ""
            if ml_score is not None:
                ml_hint = f"""
                ML Model Assessment:
                - Fraud probability: {ml_score:.0%}
                - This score is BORDERLINE ({scorer.LEGIT_THRESHOLD:.0%}-{scorer.FRAUD_THRESHOLD:.0%}) — human reasoning needed.
                """
            return f"""
                You are a healthcare claims fraud investigator.
                An ML model has flagged this claim as borderline. Your job is the
                final call. Note: this dataset's claim amounts run unusually
                high relative to typical individual claims -- judge the amount
                relative to OTHER claims in this same population, not an
                absolute real-world dollar sense.

                Key indicators for this kind of data:
                - A claim far outside the patient's or provider's typical
                  claim history is worth scrutiny
                - A large distance between provider and patient can indicate
                  identity or referral irregularities
                - A high number of procedures for the stated admission/visit
                  type is worth scrutiny

                Historical context:
                {context_str}
                {ml_hint}

                Claim details:
                - Claim amount: ${claim.get('Claim_Amount', 'N/A')}
                - Number of procedures: {claim.get('Number_of_Procedures', 'N/A')}
                - Provider type: {claim.get('Provider_Type', 'N/A')}
                - Provider-patient distance (miles): {claim.get('Provider_Patient_Distance_Miles', 'N/A')}
                - Previous claims (this patient): {claim.get('Number_of_Previous_Claims_Patient', 'N/A')}
                - Previous claims (this provider): {claim.get('Number_of_Previous_Claims_Provider', 'N/A')}
                - Claim submitted late: {claim.get('Claim_Submitted_Late', 'N/A')}

                Respond in EXACTLY this format (no extra text):
                Risk Level: <LOW/MEDIUM/HIGH>
                Reason: <one sentence>
                Action: <APPROVE/FLAG/BLOCK>
            """

        ml_hint = ""
        if ml_score is not None:
            ml_hint = f"""
            ML Model Assessment:
            - Fraud probability: {ml_score:.0%}
            - This score is BORDERLINE ({self.scorer.LEGIT_THRESHOLD:.0%}-{self.scorer.FRAUD_THRESHOLD:.0%}) — human reasoning needed.
            """

        return f"""
            You are a healthcare claims fraud investigator.
            An ML model has flagged this claim as borderline. Your job is the
            final call.

            Key indicators for this kind of data:
            - Claims submitted very quickly after the date of service (within a
              few days) are, in this population, disproportionately fraudulent —
              legitimate claims typically take one to several weeks
            - A high number of claims per provider per month can indicate
              billing-mill patterns
            - Consider whether the claim amount is plausible for the stated
              procedure and visit type

            Historical context:
            {context_str}
            {ml_hint}

            Claim details:
            - Claim amount: ${claim.get('Claim_Amount', 'N/A')}
            - Approved amount: ${claim.get('Approved_Amount', 'N/A')}
            - Days between service and claim: {claim.get('Days_Between_Service_and_Claim', 'N/A')}
            - Provider specialty: {claim.get('Provider_Specialty', 'N/A')}
            - Visit type: {claim.get('Visit_Type', 'N/A')}
            - Insurance type: {claim.get('Insurance_Type', 'N/A')}
            - Claims per provider per month: {claim.get('Number_of_Claims_Per_Provider_Monthly', 'N/A')}

            Respond in EXACTLY this format (no extra text):
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """

    # ── Main entry point ───────────────────────────────────────────────────
    def analyse(self, claim: dict, context=None):
        self.cases_reviewed += 1
        tier = self.detect_tier(claim)

        ml_score = None
        if tier != "tier3":
            rule_result = self.rule_based_filter(claim, tier)
            if rule_result:
                return rule_result

            ml_result = self.ml_filter(claim, tier)
            if ml_result:
                return ml_result

            scorer = self.scorer if tier == "tier1" else self.scorer_generalizable
            ml_score = scorer.score(claim)

        try:
            messages = [
                SystemMessage(content="You are a healthcare claims fraud investigator. Be concise and precise."),
                HumanMessage(content=self.build_prompt(claim, context, ml_score, tier)),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"  [HealthcareDetectorAgent ERROR] {e}")
            return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} claims")

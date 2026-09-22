# domains/payroll_hr/payroll/agents/payroll_detector_agent.py
# PayrollDetectorAgent — the CSV payroll-register counterpart to
# bfsi_fraud's FraudDetectorAgent, same rules -> tiered ML -> LLM shape,
# for uploads of many employee rows at once (as opposed to this domain's
# existing single-payslip PDF/RAG pipeline in pipeline.py, which is
# untouched and stays a separate capability).
#
# Schema tiering (same reasoning as bfsi_fraud's detect_tier() — see that
# module's docstring): a payroll-register CSV can come in more than one
# real export convention, and each needs its own trained scorer:
#   Tier 1 — BasePay/OvertimePay/OtherPay/Stated_TotalPay (SF Salaries'
#            convention). Rules check the 3-way sum; ML (PayrollAnomalyScorer)
#            catches implausible splits. AUC 0.973, F1 0.899.
#   Tier 2 — GROSS/Deduction/Net_Pay (a different, simpler two-component
#            convention — kaggle/sample-employees-monthly-salary,
#            INR->USD converted, see data/prepare_tier2_data.py). Rules
#            check GROSS-Deduction==Net_Pay; ML (PayrollGeneralizableScorer)
#            catches implausible deduction RATIOS. AUC 0.982, F1 0.953.
#   Tier 3 — neither schema recognized: no rules or ML layer applies
#            (each tier's rules check needs that tier's own fields —
#            defaulting a missing field to 0 would produce a false
#            reconciliation-break verdict), straight to the LLM with
#            whatever fields the record actually has.
#
# Rules are exact arithmetic, not a statistical model — no precision/
# recall/F1 attached to either tier's rules check on purpose (a perfect
# score on an identity check isn't an ML result and reporting it like one
# would be misleading — see docs/DATA_CONVENTIONS.md and this domain's
# rejected first Tier 1 labeling attempt, where rules scored a literal
# 1.0/1.0/1.0 for exactly this reason).

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.payroll_hr.payroll.settings import Settings
from domains.payroll_hr.payroll.tools.anomaly_scorer import PayrollAnomalyScorer
from domains.payroll_hr.payroll.tools.generalizable_scorer import PayrollGeneralizableScorer

RECONCILIATION_TOLERANCE = 0.01  # cents-level rounding, not a detection threshold


class PayrollDetectorAgent:

    VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, name, data_path="domains/payroll_hr/payroll/data/tier_ml_train.csv"):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        print("  [PayrollDetectorAgent] Loading ML scorer (tier 1: BasePay/OvertimePay/OtherPay schema)...")
        self.scorer = PayrollAnomalyScorer(data_path=data_path)
        print("  [PayrollDetectorAgent] Loading ML scorer (tier 2: GROSS/Deduction/Net_Pay schema)...")
        self.scorer_generalizable = PayrollGeneralizableScorer()

    @staticmethod
    def parse_response(response: str) -> dict:
        """Same robust-parsing contract as FraudDetectorAgent.parse_response."""
        result = {"risk_level": "UNKNOWN", "action": "UNKNOWN", "reason": ""}
        for line in response.strip().splitlines():
            line = line.strip()
            lower = line.lower()
            if lower.startswith("risk level"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    result["risk_level"] = first_word if first_word in PayrollDetectorAgent.VALID_RISK_LEVELS else "UNKNOWN"
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
    def detect_tier(record: dict) -> str:
        """See the module docstring above for what each tier means."""
        if all(c in record for c in PayrollAnomalyScorer.REQUIRED_COLS):
            return "tier1"
        if all(c in record for c in PayrollGeneralizableScorer.REQUIRED_COLS):
            return "tier2"
        return "tier3"

    @staticmethod
    def has_recognizable_schema(record: dict) -> bool:
        return PayrollDetectorAgent.detect_tier(record) != "tier3"

    # ── LAYER 1: Rules — deterministic reconciliation ────────────────────
    # Each tier's own check, using ITS OWN field names -- deliberately not
    # a shared "canonical field" remap the way bfsi_fraud does, because
    # tier1 and tier2 aren't just different names for the same math (3-way
    # sum vs. 2-way difference), so there's no single canonical form to
    # remap onto here.
    @staticmethod
    def _rule_check_tier1(record: dict):
        base = record.get("BasePay", 0) or 0
        ot = record.get("OvertimePay", 0) or 0
        other = record.get("OtherPay", 0) or 0
        benefits = record.get("Benefits", 0) or 0
        stated_total = record.get("Stated_TotalPay")
        stated_total_benefits = record.get("Stated_TotalPayBenefits")

        if stated_total is None:
            return None

        computed_total = base + ot + other
        if abs(stated_total - computed_total) > RECONCILIATION_TOLERANCE:
            return (
                "Risk Level: HIGH\n"
                f"Reason: Reconciliation error — stated total {stated_total:.2f} doesn't match "
                f"BasePay+OvertimePay+OtherPay ({computed_total:.2f}).\n"
                "Action: BLOCK"
            )
        if stated_total_benefits is not None:
            computed_total_benefits = computed_total + benefits
            if abs(stated_total_benefits - computed_total_benefits) > RECONCILIATION_TOLERANCE:
                return (
                    "Risk Level: HIGH\n"
                    f"Reason: Reconciliation error — stated total-with-benefits "
                    f"{stated_total_benefits:.2f} doesn't match total+Benefits ({computed_total_benefits:.2f}).\n"
                    "Action: BLOCK"
                )
        return None

    @staticmethod
    def _rule_check_tier2(record: dict):
        gross = record.get("GROSS")
        deduction = record.get("Deduction", 0) or 0
        net = record.get("Net_Pay")

        if gross is None or net is None:
            return None

        computed_net = gross - deduction
        if abs(net - computed_net) > RECONCILIATION_TOLERANCE:
            return (
                "Risk Level: HIGH\n"
                f"Reason: Reconciliation error — stated net pay {net:.2f} doesn't match "
                f"GROSS-Deduction ({computed_net:.2f}).\n"
                "Action: BLOCK"
            )
        return None

    @staticmethod
    def rule_based_filter(record: dict, tier: str = "tier1"):
        if tier == "tier1":
            return PayrollDetectorAgent._rule_check_tier1(record)
        if tier == "tier2":
            return PayrollDetectorAgent._rule_check_tier2(record)
        return None

    # ── LAYER 2: ML anomaly scorer ────────────────────────────────────────
    def ml_filter(self, record: dict, tier: str = "tier1"):
        if tier == "tier1":
            scorer, desc = self.scorer, "compensation split (overtime/other vs. base pay)"
        else:
            scorer, desc = self.scorer_generalizable, "deduction-to-gross ratio"
        score = scorer.score(record)

        # Each scorer is calibrated against its own dataset's base rate --
        # read from the scorer instance, not a shared constant.
        if score >= scorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model — {score:.0%} probability this {desc} is implausible.\n"
                f"Action: BLOCK"
            )
        if score <= scorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model — {score:.0%} probability of an implausible {desc}; matches "
                "typical patterns.\n"
                f"Action: APPROVE"
            )
        return None

    # ── LAYER 3: LLM prompt ────────────────────────────────────────────────
    def build_prompt(self, record: dict, context=None, ml_score=None, tier: str = "tier1"):
        context_str = context if context else "No historical context available"

        if tier == "tier3":
            fields = "\n".join(f"- {k}: {v}" for k, v in record.items()) or "(no fields provided)"
            return f"""
                You are a payroll compliance expert.
                This input doesn't match a recognizable payroll-register schema
                (no clear BasePay/OvertimePay/OtherPay/total or GROSS/Deduction/
                Net_Pay breakdown), so no rule-based or ML score could be
                computed. Use your own judgement.

                Historical context:
                {context_str}

                Record details (exactly as provided):
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
                - Implausible-deduction-ratio probability: {ml_score:.0%}
                - This score is BORDERLINE ({scorer.LEGIT_THRESHOLD:.0%}-{scorer.FRAUD_THRESHOLD:.0%}) — human reasoning needed.
                """
            return f"""
                You are a payroll compliance expert.
                The reconciliation math checks out (GROSS minus Deduction
                correctly equals Net_Pay) and an ML model has flagged the
                DEDUCTION RATE as borderline. Your job is the final call.

                Key indicators for this kind of data:
                - A near-0% deduction rate is unusual for almost any income level
                - A deduction rate above ~70-80% leaves an implausibly small
                  take-home amount for routine compensation
                - Typical deduction rates for this population cluster around 15-35%

                Historical context:
                {context_str}
                {ml_hint}

                Record details:
                - Gross pay: ${record.get('GROSS', 'N/A')}
                - Deduction: ${record.get('Deduction', 'N/A')}
                - Net pay: ${record.get('Net_Pay', 'N/A')}
                - Designation: {record.get('Designation', 'N/A')}

                Respond in EXACTLY this format (no extra text):
                Risk Level: <LOW/MEDIUM/HIGH>
                Reason: <one sentence>
                Action: <APPROVE/FLAG/BLOCK>
            """

        ml_hint = ""
        if ml_score is not None:
            ml_hint = f"""
            ML Model Assessment:
            - Implausible-split probability: {ml_score:.0%}
            - This score is BORDERLINE ({self.scorer.LEGIT_THRESHOLD:.0%}-{self.scorer.FRAUD_THRESHOLD:.0%}) — human reasoning needed.
            """

        return f"""
            You are a payroll compliance expert.
            The reconciliation math checks out (total pay correctly equals the
            sum of its components) and an ML model has flagged the SPLIT among
            those components as borderline. Your job is the final call.

            Key indicators for this kind of data:
            - Overtime pay should typically be a fraction of base pay, not the
              majority of total compensation
            - A newly-appearing large "other pay" (bonus/misc) relative to base
              pay for a routine role is worth scrutiny
            - Consider whether the split is plausible for the stated job title

            Historical context:
            {context_str}
            {ml_hint}

            Record details:
            - Base pay: ${record.get('BasePay', 'N/A')}
            - Overtime pay: ${record.get('OvertimePay', 'N/A')}
            - Other pay: ${record.get('OtherPay', 'N/A')}
            - Benefits: ${record.get('Benefits', 'N/A')}
            - Stated total pay: ${record.get('Stated_TotalPay', 'N/A')}
            - Job title: {record.get('JobTitle', 'N/A')}

            Respond in EXACTLY this format (no extra text):
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """

    # ── Main entry point ───────────────────────────────────────────────────
    def analyse(self, record: dict, context=None):
        self.cases_reviewed += 1
        tier = self.detect_tier(record)

        ml_score = None
        if tier != "tier3":
            rule_result = self.rule_based_filter(record, tier)
            if rule_result:
                return rule_result

            ml_result = self.ml_filter(record, tier)
            if ml_result:
                return ml_result

            scorer = self.scorer if tier == "tier1" else self.scorer_generalizable
            ml_score = scorer.score(record)

        try:
            messages = [
                SystemMessage(content="You are a payroll compliance expert. Be concise and precise."),
                HumanMessage(content=self.build_prompt(record, context, ml_score, tier)),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"  [PayrollDetectorAgent ERROR] {e}")
            return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} records")

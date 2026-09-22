# domains/payroll_hr/hr/agents/hr_detector_agent.py
# HrDetectorAgent — the job-posting counterpart to bfsi_fraud's
# FraudDetectorAgent, payroll's PayrollDetectorAgent, and healthcare's
# HealthcareDetectorAgent, same rules -> ML -> LLM shape. Scores a single
# job posting: is THIS posting fraudulent (a recruitment scam), not
# whether the company that posted it is generally trustworthy -- see
# data/prepare_data.py's module docstring for why this dataset was
# chosen over the originally-planned employee-attendance one (same
# per-entity-vs-per-artifact conflict that got rohitrox rejected for
# healthcare).
#
# Only ONE tier here (no Tier 2 yet): a single real, well-studied
# academic dataset (EMSCAD) with a genuine per-posting label.
#
#   1. Rules — genuine malformed-input checks: a posting with no title or
#      no description at all isn't a posting to judge, it's missing data.
#      This is NOT a restatement of the ML signal (which uses structural
#      presence/absence of OTHER fields -- company profile, salary range,
#      etc. -- not title/description themselves) -- deliberately checking
#      a different, more basic thing than what the model looks at.
#   2. ML (PostingFraudScorer, XGBoost) — the real detection layer.
#      AUC 0.966, F1 0.774 -- see that file's module docstring for the
#      RandomForest/XGBoost/LightGBM benchmark this came from. Deliberately
#      excludes any raw salary amount (see data/prepare_data.py for the
#      currency risk this avoids) -- uses structural signals instead:
#      missing company profile/logo (the two strongest features), description
#      length, country, industry.
#   3. LLM — fallback for borderline ML scores and postings that don't
#      carry a recognizable schema.

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.payroll_hr.hr.settings import Settings
from domains.payroll_hr.hr.tools.posting_scorer import PostingFraudScorer


class HrDetectorAgent:

    VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, name, data_path="domains/payroll_hr/hr/data/train.csv"):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        print("  [HrDetectorAgent] Loading ML posting scorer...")
        self.scorer = PostingFraudScorer(data_path=data_path)

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
                    result["risk_level"] = first_word if first_word in HrDetectorAgent.VALID_RISK_LEVELS else "UNKNOWN"
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

    @staticmethod
    def has_recognizable_schema(posting: dict) -> bool:
        return all(c in posting for c in PostingFraudScorer.REQUIRED_COLS)

    # ── LAYER 1: Rules — genuine malformed-input checks ───────────────────
    @staticmethod
    def rule_based_filter(posting: dict):
        title = (posting.get("title") or "").strip()
        description = (posting.get("description") or "").strip()

        if not title:
            return (
                "Risk Level: HIGH\n"
                "Reason: Posting has no title at all — not a valid job posting to evaluate.\n"
                "Action: BLOCK"
            )
        if not description:
            return (
                "Risk Level: HIGH\n"
                "Reason: Posting has no description at all — not a valid job posting to evaluate.\n"
                "Action: BLOCK"
            )
        return None

    # ── LAYER 2: ML fraud scorer ──────────────────────────────────────────
    def ml_filter(self, posting: dict):
        score = self.scorer.score(posting)
        if score >= self.scorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model — {score:.0%} fraud probability, primarily driven by missing "
                "company information (profile/logo) typical of recruitment scams.\n"
                f"Action: BLOCK"
            )
        if score <= self.scorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model — {score:.0%} fraud probability; matches typical legitimate postings.\n"
                f"Action: APPROVE"
            )
        return None

    # ── LAYER 3: LLM prompt ────────────────────────────────────────────────
    def build_prompt(self, posting: dict, context=None, ml_score=None):
        context_str = context if context else "No historical context available"

        if not self.has_recognizable_schema(posting):
            fields = "\n".join(f"- {k}: {v}" for k, v in posting.items()) or "(no fields provided)"
            return f"""
                You are a recruitment-fraud investigator.
                This input doesn't match a recognizable job-posting schema, so
                no rule-based or ML score could be computed. Use your own
                judgement.

                Historical context:
                {context_str}

                Posting details (exactly as provided):
                {fields}

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
            You are a recruitment-fraud investigator.
            An ML model has flagged this job posting as borderline. Your job
            is the final call.

            Key indicators for this kind of data:
            - Missing company profile or company logo is a classic
              recruitment-scam signal
            - Vague or unusually short descriptions with no concrete
              responsibilities are worth scrutiny
            - A posting with no salary range, no requirements, and no
              benefits listed together is more suspicious than any one alone
            - Consider whether the role/industry/location combination makes
              sense together

            Historical context:
            {context_str}
            {ml_hint}

            Posting details:
            - Title: {posting.get('title', 'N/A')}
            - Has company profile: {posting.get('has_company_profile', posting.get('company_profile') is not None if 'company_profile' in posting else 'N/A')}
            - Has company logo: {posting.get('has_company_logo', 'N/A')}
            - Has salary range: {posting.get('has_salary_range', posting.get('salary_range') is not None if 'salary_range' in posting else 'N/A')}
            - Employment type: {posting.get('employment_type', 'N/A')}
            - Industry: {posting.get('industry', 'N/A')}
            - Location: {posting.get('location', 'N/A')}

            Respond in EXACTLY this format (no extra text):
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """

    # ── Main entry point ───────────────────────────────────────────────────
    def analyse(self, posting: dict, context=None):
        self.cases_reviewed += 1
        recognizable = self.has_recognizable_schema(posting)

        ml_score = None
        if recognizable:
            rule_result = self.rule_based_filter(posting)
            if rule_result:
                return rule_result

            ml_result = self.ml_filter(posting)
            if ml_result:
                return ml_result

            ml_score = self.scorer.score(posting)

        try:
            messages = [
                SystemMessage(content="You are a recruitment-fraud investigator. Be concise and precise."),
                HumanMessage(content=self.build_prompt(posting, context, ml_score)),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"  [HrDetectorAgent ERROR] {e}")
            return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} postings")

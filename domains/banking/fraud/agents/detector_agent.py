# domains/banking/fraud/agents/detector_agent.py
# Ported from fraud_detection/agents/detector_agent.py — logic unchanged,
# only the settings/tools import paths moved under domains.banking.fraud.
#
# Hybrid DetectorAgent: Rule-based -> ML Scorer -> LLM (borderline only)
#
# Decision flow:
#   1. Rule-based pre-filter  -> catches obvious fraud instantly (no API call)
#   2. ML Scorer (RF)         -> fast-path for clear fraud/legit (no API call)
#   3. LLM                    -> only called for borderline cases (0.3 - 0.7)
#
# Schema tiering (added so this domain accepts more than the exact Kaggle
# shape it was trained on): detect_tier() below picks which of the three
# layers above are even reachable for a given transaction dict —
#   Tier 1 — has V1-V28 (creditcard.csv's own PCA columns): full rules ->
#            ML (scorer, best accuracy) -> LLM, unchanged from before.
#   Tier 2 — has GeneralizableScorer.REQUIRED_COLS (Transaction_Amount,
#            Account_Age, Credit_Score -- a second, real-world-shaped
#            dataset's schema, NOT a reduced version of Tier 1's: an
#            Amount+hour-only reduction of creditcard.csv was tried first
#            and empirically failed, AUC 0.66 / precision 0.3% on the real
#            holdout -- see generalizable_scorer.py's module docstring):
#            rules (remapped to this schema's field names) -> ML
#            (scorer_generalizable) -> LLM.
#   Tier 3 — neither schema recognized: rules and ML both need a
#            recognizable amount field, so both are skipped entirely (a
#            rules check defaulting a missing amount to 0 would wrongly
#            trip "zero-dollar transaction"); goes straight to the LLM
#            with whatever fields the transaction actually has.
#
# Both tiers' training data is USD (Tier 1 converted from EUR in
# prepare_data.py; Tier 2's dataset has no currency field but implies USD
# via its Credit_Score's US FICO scale) -- kept consistent on purpose,
# since rule_based_filter()'s thresholds below are USD-labeled in their
# output text and would silently misjudge a different currency's scale.

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.banking.fraud.settings import OPENAI_API_KEY, MODEL_NAME
from domains.banking.fraud.tools.ml_scorer import MLScorer
from domains.banking.fraud.tools.generalizable_scorer import GeneralizableScorer
from domains.banking.fraud.tools.drift_detector import DriftDetector


class FraudDetectorAgent:

    VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(self, name, data_path="domains/banking/fraud/data/train.csv"):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        print("  [DetectorAgent] Loading ML scorer (tier 1: full trained schema)...")
        self.scorer = MLScorer(data_path=data_path)
        print("  [DetectorAgent] Loading ML scorer (tier 2: generalizable real-world schema)...")
        self.scorer_generalizable = GeneralizableScorer()
        print("  [DetectorAgent] Loading drift detector (unsupervised, tier 1 schema only)...")
        self.drift_detector = DriftDetector(data_path=data_path)

    # ── IMPROVEMENT 4: Robust response parser ───────────────────────────
    # Replaces fragile `"HIGH" in response` string search.
    # Extracts Risk Level, Action, and Reason from the LLM response safely.
    @staticmethod
    def parse_response(response: str) -> dict:
        """
        Parses structured LLM response into a dict with keys:
          risk_level : "LOW" | "MEDIUM" | "HIGH" | "UNKNOWN"
          action     : "APPROVE" | "FLAG" | "BLOCK" | "UNKNOWN"
          reason     : str
          caveat     : str | None

        Handles variations in:
          - capitalisation  (high / High / HIGH)
          - extra whitespace or punctuation
          - missing fields (returns "UNKNOWN" safely)
          - extra text before/after the value

        `caveat` captures the "Caveat: ..." line analyse() appends for
        the Tier 1 unsupervised drift/outlier check (see analyse()'s own
        comment on why only Tier 1) — a separate signal from
        risk_level/action/reason, independent of the fraud verdict.
        """
        result = {"risk_level": "UNKNOWN", "action": "UNKNOWN", "reason": "", "caveat": None}

        for line in response.strip().splitlines():
            line = line.strip()
            lower = line.lower()

            if lower.startswith("risk level"):
                # Extract everything after the first colon
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    # Take only the first word in case of extra text
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    if first_word in FraudDetectorAgent.VALID_RISK_LEVELS:
                        result["risk_level"] = first_word
                    else:
                        result["risk_level"] = "UNKNOWN"

            elif lower.startswith("action"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    value = parts[1].strip().upper()
                    first_word = value.split()[0].rstrip(".,;") if value else ""
                    if first_word in {"APPROVE", "FLAG", "BLOCK"}:
                        result["action"] = first_word
                    else:
                        result["action"] = "FLAG"   # safe default

            elif lower.startswith("reason"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    result["reason"] = parts[1].strip()

            elif lower.startswith("caveat"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    result["caveat"] = parts[1].strip()

        return result

    @staticmethod
    def is_fraud(parsed: dict) -> bool:
        """
        Returns True if the parsed response indicates fraud.
        MEDIUM and HIGH both count as fraud — safer for a fraud detection system.
        UNKNOWN defaults to True (fail safe — flag rather than miss fraud).
        """
        return parsed["risk_level"] in {"MEDIUM", "HIGH", "UNKNOWN"}

    # ── LAYER 1: Rule-based pre-filter ──────────────────────────────────
    # Static — doesn't touch self, so eval_harness.py can call this
    # directly against the full holdout set without paying for an
    # OpenAI client + MLScorer retrain just to reach the rule layer.
    @staticmethod
    def rule_based_filter(transaction):
        amount = transaction.get("Amount", 0)
        hour   = transaction.get("hour", 12)

        if amount == 0.0:
            return (
                "Risk Level: HIGH\n"
                "Reason: Zero-dollar transaction indicates a card verification attack.\n"
                "Action: BLOCK"
            )
        if 0 < amount <= 5.0 and 0 <= hour <= 5:
            return (
                "Risk Level: HIGH\n"
                "Reason: Micro-transaction during night hours — classic card-testing pattern.\n"
                "Action: BLOCK"
            )
        if amount > 1000 and 0 <= hour <= 5:
            return (
                "Risk Level: HIGH\n"
                "Reason: Large transaction amount during overnight hours is highly suspicious.\n"
                "Action: BLOCK"
            )
        return None

    # ── Schema tiering ───────────────────────────────────────────────────
    @staticmethod
    def detect_tier(transaction: dict) -> str:
        """See the module docstring above for what each tier means."""
        if all(f"V{i}" in transaction for i in range(1, 29)):
            return "tier1"
        if all(c in transaction for c in GeneralizableScorer.REQUIRED_COLS):
            return "tier2"
        return "tier3"

    @staticmethod
    def _normalize_for_rules(transaction: dict) -> dict | None:
        """rule_based_filter() expects canonical "Amount"/"hour" keys.
        Deliberately independent of detect_tier()'s ML-schema result:
        rules are cheap and apply to any recognizable amount field, even
        for a transaction that doesn't fully qualify for either ML tier
        (e.g. a bare Amount/hour manual entry with no V1-V28 and no
        Account_Age/Credit_Score — that's still worth a rules check, it
        just won't reach the ML layer). Tier 1 uses "Amount" already; Tier
        2's is named "Transaction_Amount" and gets remapped. Returns None
        (skip rules entirely) only when neither field is present at all —
        defaulting a truly-missing amount to 0 would wrongly trip the
        zero-dollar rule."""
        if "Amount" in transaction:
            return transaction
        if "Transaction_Amount" in transaction:
            return {"Amount": transaction["Transaction_Amount"], "hour": transaction.get("hour", 12)}
        return None

    # ── LAYER 2: ML fast-path ────────────────────────────────────────────
    def ml_filter(self, transaction, tier="tier1"):
        if tier == "tier1":
            scorer, label = self.scorer, "tier1: full trained schema"
        else:
            scorer, label = self.scorer_generalizable, "tier2: generalizable real-world schema"
        score = scorer.score(transaction)

        # Each scorer is calibrated against its own training data's base
        # rate (Tier 1: ~0.13%, Tier 2: ~28%) -- these are NOT the same
        # thresholds, deliberately read from the scorer instance itself
        # rather than a shared constant.
        if score >= scorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model ({label}) fraud probability {score:.0%} — strongly matches fraud patterns in training data.\n"
                f"Action: BLOCK"
            )
        if score <= scorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model ({label}) fraud probability {score:.0%} — strongly matches legitimate transaction patterns.\n"
                f"Action: APPROVE"
            )
        return None

    # ── LAYER 3: LLM prompt (borderline cases, and all of tier 3) ────────
    def build_prompt(self, transaction, context=None, ml_score=None, tier="tier1"):
        context_str = context if context else "No historical context available"

        if tier == "tier3":
            # No "Amount" at all -- rules and both ML scorers need one, so
            # neither ran. Nothing to weigh a score against; render
            # whatever fields this transaction actually has and let the
            # LLM reason from first principles instead of a fixed rubric
            # that assumes fields that aren't there.
            fields = "\n".join(f"- {k}: {v}" for k, v in transaction.items()) or "(no fields provided)"
            return f"""
                You are a fraud detection expert at a bank.
                This input doesn't match any known transaction schema (no
                recognizable amount field), so no rule-based or ML score
                could be computed. Use your own judgement about what's
                shown below.

                Historical context:
                {context_str}

                Transaction details (exactly as provided):
                {fields}

                Respond in EXACTLY this format (no extra text):
                Risk Level: <LOW/MEDIUM/HIGH>
                Reason: <one sentence>
                Action: <APPROVE/FLAG/BLOCK>
            """

        ml_hint = ""
        if ml_score is not None:
            fraud_t = self.scorer_generalizable.FRAUD_THRESHOLD if tier == "tier2" else MLScorer.FRAUD_THRESHOLD
            legit_t = self.scorer_generalizable.LEGIT_THRESHOLD if tier == "tier2" else MLScorer.LEGIT_THRESHOLD
            ml_hint = f"""
            ML Model Assessment:
            - Fraud probability score: {ml_score:.0%}
            - This score is BORDERLINE (between {legit_t:.0%} and {fraud_t:.0%}) — human reasoning needed.
            - Weight this alongside the transaction features below.
            """

        if tier == "tier2":
            # Different dataset, different schema, and a very different
            # Amount scale (median ~$5,000, not Tier 1's ~$25-30) -- Tier
            # 1's fixed dollar-amount rubric below would actively mislead
            # here, so this uses fields this dataset actually has instead
            # (Previous_Fraud and Credit_Score are its strongest signals —
            # see generalizable_scorer.py).
            return f"""
                You are a fraud detection expert at a bank.
                A machine learning model has flagged this transaction as borderline.
                Your job is to make the final call using your expert reasoning.

                Key fraud indicators for this kind of data:
                - A customer with prior fraud history (Previous_Fraud) committing
                  fraud again is far more likely than a first-time incident
                - A very low credit score combined with a large transaction amount
                  is a meaningful risk signal
                - A newly opened account (low Account_Age) transacting immediately
                  at unusual hours is suspicious
                - There is no fixed "normal" dollar amount for this dataset —
                  judge the amount relative to the customer's own profile, not
                  a universal threshold

                Historical context:
                {context_str}
                {ml_hint}

                Transaction details:
                - Amount: ${transaction.get('Transaction_Amount', 'N/A')}
                - Transaction type: {transaction.get('Transaction_Type', 'N/A')}
                - Device used: {transaction.get('Device_Used', 'N/A')}
                - Account age (days): {transaction.get('Account_Age', 'N/A')}
                - Credit score: {transaction.get('Credit_Score', 'N/A')}
                - Previous fraud on this customer: {transaction.get('Previous_Fraud', 'N/A')}
                - Hour of day: {transaction.get('hour', 'N/A')} (0=midnight, 23=11pm)

                Respond in EXACTLY this format (no extra text):
                Risk Level: <LOW/MEDIUM/HIGH>
                Reason: <one sentence>
                Action: <APPROVE/FLAG/BLOCK>
            """

        return f"""
            You are a fraud detection expert at a bank.
            A machine learning model has flagged this transaction as borderline.
            Your job is to make the final call using your expert reasoning.

            Key fraud indicators:
            - Exactly $0.00 transactions ALWAYS indicate card verification attacks
            - Amounts between $0.01 and $5.00 at hours 0-5 are suspicious
            - Transactions between midnight and 5am (hour 0-5) are HIGH RISK
            - Amounts over $500 at unusual hours are suspicious
            - Legitimate transactions typically have amounts between $10-$500
            - Transactions at hour 23 (11pm) with amounts over $200 are suspicious

            Historical context:
            {context_str}
            {ml_hint}

            Transaction details:
            - Amount: ${transaction['Amount']}
            - Hour of day: {transaction.get('hour', 'N/A')} (0=midnight, 23=11pm)
            - Time since first transaction: {transaction.get('Time', 'N/A')} seconds

            Respond in EXACTLY this format (no extra text):
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """

    # ── Main entry point ─────────────────────────────────────────────────
    def analyse(self, transaction, context=None):
        self.cases_reviewed += 1
        tier = self.detect_tier(transaction)
        result = None

        # Layer 1: Rules — deliberately independent of the ML tier (see
        # _normalize_for_rules): applies whenever a recognizable amount
        # field exists, even for a transaction that doesn't fully qualify
        # for either ML schema. None means skip rules rather than risk a
        # false zero-dollar match.
        rule_txn = self._normalize_for_rules(transaction)
        if rule_txn is not None:
            rule_result = self.rule_based_filter(rule_txn)
            if rule_result:
                result = rule_result

        # Layer 2: ML scorer — only for genuine tier1/tier2 schema matches
        ml_score = None
        if result is None and tier != "tier3":
            ml_result = self.ml_filter(transaction, tier)
            if ml_result:
                result = ml_result
            else:
                scorer = self.scorer if tier == "tier1" else self.scorer_generalizable
                ml_score = scorer.score(transaction)

        # Layer 3: LLM — borderline tier1/tier2 cases, and all of tier3
        if result is None:
            try:
                messages = [
                    SystemMessage(content="You are a bank fraud detection expert. Be concise and precise."),
                    HumanMessage(content=self.build_prompt(transaction, context, ml_score, tier))
                ]
                response = self.llm.invoke(messages)
                result = response.content
            except Exception as e:
                print(f"  [DetectorAgent ERROR] {e}")
                result = "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

        # Unsupervised drift/outlier caveat -- Tier 1 only. DriftDetector's
        # features are Tier 1's exact V1-V28+Amount+Time+hour schema;
        # running it on Tier 2/3 data would silently zero-fill missing
        # V1-V28 and produce a meaningless result. Measured tradeoff (see
        # drift_detector.py): Tier 1's legit-outlier rate is low enough
        # (4.5%) that this caveat is a rare, meaningful signal here.
        # Every other domain's own DriftDetector was measured the same way
        # and found too noisy to wire into a live caveat -- Payroll 10.4%,
        # Financial Services 13.9%/12.0% (Tier 1/2), HR 43.2%, Insurance
        # 51.1% (see each domain's tests/test_<domain>_drift_detector.py)
        # -- so this stays Banking-only, not an oversight. Appends a
        # caveat; never changes the verdict or parsed risk_level/action
        # fields above.
        if tier == "tier1" and self.drift_detector.is_outlier(transaction):
            result += (
                "\nCaveat: this transaction also looks statistically unusual relative to the "
                "training distribution (unsupervised check, independent of the verdict above) "
                "— worth extra scrutiny regardless of the risk level assigned."
            )

        return result

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} transactions")

# domains/financial_services/agents/fintech_detector_agent.py
# FintechDetectorAgent -- the non-bank crypto/digital-asset counterpart
# to Banking's FraudDetectorAgent, Payroll's PayrollDetectorAgent,
# Insurance's HealthcareDetectorAgent, and HR's HrDetectorAgent. Same
# rules -> tiered ML -> LLM shape.
#
# Schema tiering (same reasoning as every other domain's detect_tier() --
# see e.g. Banking's detector_agent.py module docstring):
#   Tier 1 -- per-blockchain-transaction scam detection (ScamScorer).
#            AUC 0.908, F1 0.425. See tools/scam_scorer.py and
#            data/prepare_data.py for the dataset search.
#   Tier 2 -- per-transaction exchange/DEX market-manipulation detection
#            (WashTradingScorer) -- a genuinely different fraud problem
#            (wash trading/pump-and-dump/ramping, not scam wallets) and
#            schema, NOT a reduced version of Tier 1's data. AUC 0.928,
#            F1 0.356. See tools/wash_trading_scorer.py and
#            data/prepare_tier2_data.py.
#   Tier 3 -- neither schema recognized: straight to the LLM with
#            whatever fields the record actually has.
#
# No fraud-heuristic rule survived vetting for EITHER tier -- checked
# empirically before assuming: Tier 1's best 2-condition combo
# (is_cross_chain + high failed_txn_ratio) reached ~49% scam rate on a
# tiny slice (n=245), nowhere near a rule's bar; Tier 2's best
# (round_trip_score>0.99) reached only 7.2%, barely above its 5.9% base
# rate. Layer 1 here only rejects malformed records for both tiers; the
# real detection work happens at the ML/LLM layers.
#
# Both tiers' training data is USD (transaction_amount_usd/notional_usd/
# gas_fee_usd) -- matches this project's standing convention. The
# earlier Tier 1 candidate this replaced (Central Asia Fintech,
# mobile-wallet transactions) was UZS and never converted -- benched,
# not deleted, in data/backup_wallet_raw/.

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from domains.financial_services.settings import Settings
from domains.financial_services.tools.scam_scorer import ScamScorer
from domains.financial_services.tools.wash_trading_scorer import WashTradingScorer


class FintechDetectorAgent:

    VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

    def __init__(
        self, name,
        data_path="domains/financial_services/data/train.csv",
        tier2_data_path="domains/financial_services/data/tier2_train.csv",
    ):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        print("  [FintechDetectorAgent] Loading ML scorer (tier 1: blockchain scam schema)...")
        self.scorer = ScamScorer(data_path=data_path)
        print("  [FintechDetectorAgent] Loading ML scorer (tier 2: exchange/DEX manipulation schema)...")
        self.scorer_tier2 = WashTradingScorer(data_path=tier2_data_path)

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
                    result["risk_level"] = first_word if first_word in FintechDetectorAgent.VALID_RISK_LEVELS else "UNKNOWN"
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

    # ── Schema tiering ──────────────────────────────────────────────────
    @staticmethod
    def detect_tier(record: dict) -> str:
        """See the module docstring above for what each tier means."""
        if all(c in record for c in ScamScorer.REQUIRED_COLS):
            return "tier1"
        if all(c in record for c in WashTradingScorer.REQUIRED_COLS):
            return "tier2"
        return "tier3"

    @staticmethod
    def has_recognizable_schema(record: dict) -> bool:
        return FintechDetectorAgent.detect_tier(record) != "tier3"

    # ── LAYER 1: Rules — malformed-input check only, not a fraud
    # heuristic (see module docstring for why no real one survived
    # vetting on either tier's data) ─────────────────────────────────────
    @staticmethod
    def rule_based_filter(record: dict, tier: str = "tier1"):
        amount_field = "transaction_amount_usd" if tier == "tier1" else "notional_usd"
        amount = record.get(amount_field)
        if amount is None:
            return (
                "Risk Level: HIGH\n"
                "Reason: Record has no transaction amount at all — not evaluable.\n"
                "Action: BLOCK"
            )
        if amount < 0:
            return (
                "Risk Level: HIGH\n"
                "Reason: Negative transaction amount — malformed record, not evaluable.\n"
                "Action: BLOCK"
            )
        return None

    # ── LAYER 2: ML scorer ────────────────────────────────────────────────
    def ml_filter(self, record: dict, tier: str = "tier1"):
        if tier == "tier1":
            scorer, label, noun = self.scorer, "tier1: blockchain scam schema", "scam"
        else:
            scorer, label, noun = self.scorer_tier2, "tier2: exchange/DEX manipulation schema", "manipulation"
        score = scorer.score(record)

        if score >= scorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model ({label}) {noun} probability {score:.0%} — no single dominant "
                "signal, several weak indicators combined.\n"
                f"Action: BLOCK"
            )
        if score <= scorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model ({label}) {noun} probability {score:.0%}; matches typical patterns.\n"
                f"Action: APPROVE"
            )
        return None

    # ── LAYER 3: LLM prompt ────────────────────────────────────────────────
    def build_prompt(self, record: dict, context=None, ml_score=None, tier: str = "tier1"):
        context_str = context if context else "No historical context available"

        if tier == "tier3":
            fields = "\n".join(f"- {k}: {v}" for k, v in record.items()) or "(no fields provided)"
            return f"""
                You are a fraud analyst at a non-bank fintech/digital-asset
                provider. This input doesn't match a recognizable
                transaction schema, so no rule-based or ML score could be
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

        ml_hint = ""
        if ml_score is not None:
            scorer = self.scorer if tier == "tier1" else self.scorer_tier2
            ml_hint = f"""
            ML Model Assessment:
            - Fraud probability: {ml_score:.0%}
            - This score is BORDERLINE ({scorer.LEGIT_THRESHOLD:.0%}-{scorer.FRAUD_THRESHOLD:.0%}) — human reasoning needed.
            - Note: this model's signal is real but diffuse — no single
              feature reliably separates fraud on its own; treat its score
              as one weak-to-moderate input, not a confident verdict.
            """

        if tier == "tier2":
            return f"""
                You are a market-surveillance analyst at a crypto exchange.
                An ML model has flagged this trade as borderline for wash
                trading / pump-and-dump / ramping. Your job is the final call.

                Key indicators for this kind of data (individually weak —
                look for several together):
                - Rapid round-trip trades between the same counterparties
                - Unusual price deviation from the volume-weighted average
                - A sudden burst of trades in a low-liquidity token
                - A newly created wallet with an outsized share of volume

                Historical context:
                {context_str}
                {ml_hint}

                Trade details:
                - Notional (USD): {record.get('notional_usd', 'N/A')}
                - Venue type: {record.get('venue_type', 'N/A')}
                - Token liquidity score: {record.get('token_liquidity_score', 'N/A')}
                - Round-trip score: {record.get('round_trip_score', 'N/A')}
                - Counterparty reuse ratio: {record.get('counterparty_reuse_ratio', 'N/A')}
                - Price deviation from VWAP (%): {record.get('price_deviation_vwap_pct', 'N/A')}

                Respond in EXACTLY this format (no extra text):
                Risk Level: <LOW/MEDIUM/HIGH>
                Reason: <one sentence>
                Action: <APPROVE/FLAG/BLOCK>
            """

        return f"""
            You are a fraud analyst at a non-bank fintech/digital-asset
            provider. An ML model has flagged this transaction as
            borderline. Your job is the final call.

            Key indicators for this kind of data (individually weak —
            look for several together):
            - A cross-chain bridge transfer (a well-known laundering vector)
            - A newly created sender wallet
            - A high recent failed-transaction ratio for the sender

            Historical context:
            {context_str}
            {ml_hint}

            Transaction details:
            - Amount (USD): {record.get('transaction_amount_usd', 'N/A')}
            - Cross-chain: {record.get('is_cross_chain', 'N/A')}
            - Sender wallet age (days): {record.get('sender_wallet_age_days', 'N/A')}
            - Failed txn ratio (sender): {record.get('failed_txn_ratio_sender', 'N/A')}
            - Blockchain: {record.get('blockchain', 'N/A')}
            - Platform: {record.get('platform', 'N/A')}

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

            scorer = self.scorer if tier == "tier1" else self.scorer_tier2
            ml_score = scorer.score(record)

        try:
            messages = [
                SystemMessage(content="You are a fintech fraud analyst. Be concise and precise."),
                HumanMessage(content=self.build_prompt(record, context, ml_score, tier)),
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"  [FintechDetectorAgent ERROR] {e}")
            return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} records")

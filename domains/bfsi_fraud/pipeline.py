# domains/bfsi_fraud/pipeline.py
# FraudPipeline — the AgentOrchestrator-facing adapter for this domain.
# Thin wrapper around the exact detector -> analyst -> router -> alert
# sequence that fraud_detection/app.py's run_pipeline() and main.py already
# used; this just returns a structured dict instead of printing/rendering,
# so both streamlit_app.py and main.py (and tests) can share one code path.

from __future__ import annotations

from core.config_loader import DomainConfig
from domains.bfsi_fraud.agents.detector_agent import FraudDetectorAgent
from domains.bfsi_fraud.agents.analyst_agent import AnalystAgent
from domains.bfsi_fraud.agents.alert_agent import AlertAgent
from domains.bfsi_fraud.agents.routing_agent import RoutingAgent
from domains.bfsi_fraud.tools.image_input import ImageInputTool

DEFAULT_DEMO_CSV = "domains/bfsi_fraud/data/transactions_balanced.csv"


class FraudPipeline:
    def __init__(self, config: DomainConfig):
        self.config = config
        data_path = config.extra.get("data", {}).get("demo_csv", DEFAULT_DEMO_CSV)

        self.detector = FraudDetectorAgent(name="DetectorAgent-1", data_path=data_path)
        self.analyst = AnalystAgent(name="AnalystAgent-1")
        self.alert = AlertAgent(name="AlertAgent-1")
        self.router = RoutingAgent(name="RoutingAgent-1")
        self.image_tool = ImageInputTool()

    def run(self, transaction: dict, context: str | None = None) -> dict:
        """Runs the full detector -> [analyst -> router -> alert] pipeline
        for one transaction. The investigation/routing/alert steps only
        fire when the detector actually calls it FRAUD — same branching as
        the original app.py/main.py."""
        response = self.detector.analyse(transaction, context)
        parsed = FraudDetectorAgent.parse_response(response)
        predicted = "FRAUD" if FraudDetectorAgent.is_fraud(parsed) else "LEGITIMATE"

        investigation = routing = alert_text = ""
        if predicted == "FRAUD":
            investigation = self.analyst.investigate(transaction, response)
            routing = self.router.route(transaction, investigation)
            if "BLOCK" in investigation:
                alert_text = self.alert.generate_alert(transaction, response, investigation)

        return {
            "predicted": predicted,
            "parsed": parsed,
            "detector_response": response,
            "investigation": investigation,
            "routing": routing,
            "alert": alert_text,
        }

    def extract_from_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        """Extracts a transaction dict from a receipt/cheque image via
        GPT-4o Vision — used by streamlit_app.py's image-input tab."""
        return self.image_tool.extract_from_bytes(image_bytes, mime_type)

    @staticmethod
    def default_context(avg_amount: float, max_amount: float) -> str:
        """Same free-text context string the original app.py built from the
        demo CSV, used as background for the detector's LLM layer."""
        return (
            f"Dataset average transaction amount: ${avg_amount:.2f}. "
            f"Maximum amount: ${max_amount:.2f}. "
            f"Typical legitimate transactions range from $5 to $500."
        )

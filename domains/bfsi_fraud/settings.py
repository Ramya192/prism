# domains/bfsi_fraud/settings.py
# Ported from fraud_detection/config/settings.py unchanged — same three
# required env vars, same "no fallback, crash on startup if missing" for
# MODEL_NAME/RISK_THRESHOLD (that's intentional upstream behaviour, not
# something Prism should silently paper over).

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
RISK_THRESHOLD = int(os.getenv("RISK_THRESHOLD"))

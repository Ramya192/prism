# domains/payroll_hr/hr/settings.py
# Minimal settings for hr's job-posting fraud detector -- only what
# HrDetectorAgent needs. Same minimal-first pattern as
# domains/insurance/settings.py.

from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

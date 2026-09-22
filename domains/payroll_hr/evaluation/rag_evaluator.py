"""
domains/payroll_hr/evaluation/rag_evaluator.py
Thin wrapper over core/rag/base_rag_evaluator.py -- see that module's
docstring for why the evaluation logic is shared rather than copy-pasted
per domain. This file's only job is supplying Payroll & HR's own Settings
and ground_truth.json (built from the real EMP001/EMP002/EMP003 demo
payslips -- see domains/payroll_hr/payroll/tools/generate_payroll_register.py).
Settings is imported from the payroll sub-package (payroll_hr's document
side reuses Payroll's existing RAG agents/settings, not a new top-level
payroll_hr/settings.py -- see domains/payroll_hr/pipeline.py).
"""

from pathlib import Path

from core.rag.base_rag_evaluator import EvaluationResult, get_evaluator as _get_evaluator
from domains.payroll_hr.payroll.settings import Settings

_GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

__all__ = ["EvaluationResult", "get_evaluator"]


def get_evaluator():
    """
    Returns evaluator based on Settings.LLM_PROVIDER.
    - ollama  -> CustomRAGEvaluator (local)
    - openai  -> RAGASEvaluator (production)
    """
    return _get_evaluator(Settings, _GROUND_TRUTH_PATH)

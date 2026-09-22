"""
domains/insurance/evaluation/rag_evaluator.py
Thin wrapper over core/rag/base_rag_evaluator.py -- see that module's
docstring for why the evaluation logic is shared rather than copy-pasted
per domain. This file's only job is supplying Insurance's own Settings
and ground_truth.json (built from the real CLM001/CLM002/CLM003 demo
claims -- see domains/insurance/tools/generate_claim_document.py).
"""

from pathlib import Path

from core.rag.base_rag_evaluator import EvaluationResult, get_evaluator as _get_evaluator
from domains.insurance.settings import Settings

_GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

__all__ = ["EvaluationResult", "get_evaluator"]


def get_evaluator():
    """
    Returns evaluator based on Settings.LLM_PROVIDER.
    - ollama  -> CustomRAGEvaluator (local)
    - openai  -> RAGASEvaluator (production)
    """
    return _get_evaluator(Settings, _GROUND_TRUTH_PATH)

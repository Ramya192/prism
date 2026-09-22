"""
domains/banking/documents/evaluation/rag_evaluator.py
Thin wrapper over core/rag/base_rag_evaluator.py -- this file used to hold
all the evaluation logic directly (ported from
document_intelligence_system/evaluation/rag_evaluator.py); that logic is
now shared (Insurance, Payroll & HR, and Financial Services all needed the
same capability -- see the base module's docstring), and this file's only
remaining job is supplying Banking's own Settings and ground_truth.json,
unchanged from before. IMPORTANT: this refactor changes zero logic and zero
PROMPT TEXT for a system that's live in production and already RAGAS-
evaluated -- verified via an exact diff against the pre-refactor version,
including a byte-for-byte comparison of the faithfulness prompt string.
"""

from pathlib import Path

from core.rag.base_rag_evaluator import EvaluationResult, get_evaluator as _get_evaluator
from domains.banking.documents.settings import Settings

# Resolved relative to this file, not the process CWD -- the original used
# Path("evaluation/ground_truth.json"), which only worked when the CWD
# happened to be the repo root. Nesting one level deeper under domains/
# made that assumption unsafe, so this now works regardless of CWD.
_GROUND_TRUTH_PATH = Path(__file__).parent / "ground_truth.json"

__all__ = ["EvaluationResult", "get_evaluator"]


def get_evaluator():
    """
    Returns evaluator based on Settings.LLM_PROVIDER.
    - ollama  -> CustomRAGEvaluator (local)
    - openai  -> RAGASEvaluator (production)
    """
    return _get_evaluator(Settings, _GROUND_TRUTH_PATH)

"""
core/rag/base_rag_evaluator.py
Generalized from domains/banking/documents/evaluation/rag_evaluator.py --
that file was Banking-specific (a fixed `from domains.banking.documents.settings
import Settings` and a hardcoded ground_truth.json path) even though every
line of actual evaluation logic is domain-agnostic: cosine similarity,
LLM-as-judge faithfulness, and RAGAS metrics don't know or care which
domain's chunks they're scoring. With Insurance, Payroll & HR, and
Financial Services all needing the same capability, copy-pasting this
~350-line file three more times would be the exact kind of undeniable
duplication core/rag/base_retriever_agent.py's own docstring describes
extracting a shared base for -- same reasoning applied here.

IMPORTANT: this is Banking's file with zero logic or prompt-text changes --
only `Settings`/`_GROUND_TRUTH_PATH` promoted from module-level imports/
constants to constructor parameters, since Banking's evaluator is already
live in production and RAGAS-evaluated. Verify this if you ever touch the
templates below: a prompt wording change here changes a deployed, measured
system, not a refactor.

Metrics implemented:
  - Context Precision   : cosine(query_embedding, chunk_embeddings) mean
  - Faithfulness        : LLM-as-judge -- is every claim grounded in context?
  - Answer Relevancy    : cosine(query_embedding, answer_embedding)
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _shim_langchain_community_vertexai() -> None:
    """
    ragas 0.4.3 unconditionally imports `ChatVertexAI` from
    `langchain_community.chat_models.vertexai` at module load time, just to
    build an isinstance() check tuple for provider-specific behaviour. That
    submodule was removed in langchain-community 0.4.x (Vertex AI chat
    support moved to the separate `langchain-google-vertexai` package),
    so importing `ragas` at all raises ModuleNotFoundError on our pinned
    langchain-community==0.4.2 -- even though we only ever use the OpenAI
    path. Register a harmless stand-in module before ragas imports it; this
    is a no-op for anyone who doesn't use Vertex AI.
    """
    import sys
    import types

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # placeholder -- never instantiated, only used in isinstance() checks
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


# ── Shared output model ───────────────────────────────────────────────────────
class EvaluationResult:
    def __init__(
        self,
        context_precision: Optional[float],
        faithfulness: float,
        answer_relevancy: float,
        evaluator: str,
        latency_ms: int,
    ):
        # context_precision can be None: it's the "with reference" RAGAS
        # metric, only meaningful when a matching ground-truth answer was
        # found (see RAGASEvaluator.evaluate()). None means "not measurable
        # for this query", not "scored zero" -- don't conflate the two.
        self.context_precision = (
            round(context_precision, 3) if context_precision is not None else None
        )
        self.faithfulness = round(faithfulness, 3)
        self.answer_relevancy = round(answer_relevancy, 3)
        self.evaluator = evaluator
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        return {
            "context_precision": self.context_precision,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "evaluator": self.evaluator,
            "latency_ms": self.latency_ms,
        }


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM EVALUATOR  (Ollama -- local dev)
# ══════════════════════════════════════════════════════════════════════════════
class CustomRAGEvaluator:
    """
    Three metrics using sentence-transformers + Ollama LLM-as-judge.

    Context Precision  : mean cosine similarity between query and each chunk
    Faithfulness       : LLM prompt -- does answer stay within the context?
    Answer Relevancy   : cosine similarity between query and answer
    """

    def __init__(self, settings):
        from sentence_transformers import SentenceTransformer
        from langchain_ollama import ChatOllama

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.llm = ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    # ── cosine similarity (manual, no scipy needed) ───────────────────────────
    @staticmethod
    def _cosine(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── 1. Context Relevancy ─────────────────────────────────────────────────
    def _context_precision(self, query: str, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        texts = [c["text"] for c in chunks if c.get("text")]
        if not texts:
            return 0.0
        query_emb = self.encoder.encode(query).tolist()
        chunk_embs = [self.encoder.encode(t).tolist() for t in texts]
        scores = [self._cosine(query_emb, ce) for ce in chunk_embs]
        return sum(scores) / len(scores)

    # ── 2. Faithfulness (LLM-as-judge) ──────────────────────────────────────
    def _faithfulness(self, answer: str, chunks: list[dict]) -> float:
        if not chunks or not answer:
            return 0.0
        context = "\n\n".join([c.get("text", "") for c in chunks])
        prompt = f"""You are an impartial evaluator assessing faithfulness of an answer.

Context (retrieved documents):
{context}

Answer to evaluate:
{answer}

Task: Check if the factual claims in the answer are broadly supported by the context.
Be lenient with minor wording differences — if the meaning is the same, it is faithful.
- Score 1.0 if all claims are grounded in the context (minor wording differences are acceptable).
- Score 0.5 if some claims are grounded but others are clearly not in the context.
- Score 0.0 only if the answer contains facts that directly contradict or are completely absent from the context.

Respond ONLY with a JSON object, no other text:
{{"score": 0.0, "reason": "brief explanation"}}"""

        try:
            response = self.llm.invoke(prompt)
            raw = response.content.strip()

            # Extract JSON
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw[start:end])
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Faithfulness eval failed: {e}")
            # Also print raw response for debugging
        return 0.5  # neutral fallback

    # ── 3. Answer Relevancy ──────────────────────────────────────────────────
    def _answer_relevancy(self, query: str, answer: str) -> float:
        if not query or not answer:
            return 0.0
        query_emb = self.encoder.encode(query).tolist()
        answer_emb = self.encoder.encode(answer).tolist()
        return self._cosine(query_emb, answer_emb)

    # ── Public evaluate() ────────────────────────────────────────────────────
    def evaluate(
        self,
        query: str,
        chunks: list[dict],
        answer: str,
        source_document: str = None,
    ) -> EvaluationResult:
        start = time.time()

        ctx_rel = self._context_precision(query, chunks)
        faith = self._faithfulness(answer, chunks)
        ans_rel = self._answer_relevancy(query, answer)

        latency_ms = int((time.time() - start) * 1000)

        return EvaluationResult(
            context_precision=ctx_rel,
            faithfulness=faith,
            answer_relevancy=ans_rel,
            evaluator="custom-ollama",
            latency_ms=latency_ms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATOR  (OpenAI -- production)
# ══════════════════════════════════════════════════════════════════════════════
class RAGASEvaluator:
    """
    Uses RAGAS library with OpenAI backend.
    Metrics: context_precision, faithfulness, answer_relevancy
    """

    def __init__(self, settings, ground_truth_path: Path):
        _shim_langchain_community_vertexai()

        import warnings

        with warnings.catch_warnings():
            # ragas 0.4.3 still ships the classic evaluate()/metrics API we use
            # here, but emits a DeprecationWarning nudging toward
            # ragas.metrics.collections (an async, instructor-based rewrite).
            # Not migrating to that yet: it changes the evaluation model
            # (single-example async scoring vs. our batched `datasets.Dataset`
            # + evaluate() call) and isn't required until ragas 1.0 actually
            # removes this path.
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                faithfulness,
                answer_relevancy,
            )
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        self._evaluate = evaluate
        self._metrics = [context_precision, faithfulness, answer_relevancy]
        # RAGAS uses langchain LLM and embeddings under the hood
        self._llm = ChatOpenAI(model=settings.OPENAI_MODEL)
        self._embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

        # Load ground truth
        self.ground_truth = (
            json.load(open(ground_truth_path)) if ground_truth_path.exists() else {}
        )

    def evaluate(
        self,
        query: str,
        chunks: list[dict],
        answer: str,
        source_document: str = None,
    ) -> EvaluationResult:
        from datasets import Dataset

        start = time.time()
        contexts = [c.get("text", "") for c in chunks]

        # Stopwords excluded from the overlap check below -- without this,
        # "what"/"is"/"the" alone were enough to match query "what are large
        # withdrawals?" against ground-truth question "What is the closing
        # balance?" (both contain "what"), silently feeding RAGAS the WRONG
        # reference answer for context_precision/faithfulness. Comparing an
        # answer about withdrawals against a reference about closing balance
        # produces near-zero scores that look like a broken RAG pipeline,
        # when it's actually a broken ground-truth lookup.
        _STOPWORDS = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "there",
            "this", "to", "was", "were", "what", "when", "where", "which",
            "who", "why", "will", "with",
        }

        def _significant_words(text: str) -> set[str]:
            return {w.strip("?.,!") for w in text.lower().split()} - _STOPWORDS

        ground_truth_answer = ""
        if source_document and source_document in self.ground_truth:
            query_words = _significant_words(query)
            best_overlap = 0
            for pair in self.ground_truth[source_document]["qa_pairs"]:
                overlap = len(query_words & _significant_words(pair["question"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    ground_truth_answer = pair["ground_truth"]
            if best_overlap == 0:
                # No genuine content-word overlap with any known eval
                # question -- don't guess. An empty ground truth here is
                # honest: it means "no reference available for this query",
                # not "this answer scored zero".
                ground_truth_answer = ""

        has_ground_truth = bool(ground_truth_answer)
        dataset = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth_answer],
            }
        )

        try:
            # context_precision (the "with reference" variant) needs a real
            # ground truth to mean anything -- computing it against an empty
            # string isn't "0% precision", it's "not measurable". Only
            # request it when we actually matched a known eval question.
            metrics = self._metrics if has_ground_truth else self._metrics[1:]
            result = self._evaluate(
                dataset,
                metrics=metrics,
                llm=self._llm,
                embeddings=self._embeddings,
            )
            df = result.to_pandas()
            ctx_rel = float(df["context_precision"].iloc[0]) if has_ground_truth else None
            faith = float(df["faithfulness"].iloc[0])
            ans_rel = float(df["answer_relevancy"].iloc[0])
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            ctx_rel = faith = ans_rel = 0.0

        latency_ms = int((time.time() - start) * 1000)

        return EvaluationResult(
            context_precision=ctx_rel,
            faithfulness=faith,
            answer_relevancy=ans_rel,
            evaluator="ragas-openai",
            latency_ms=latency_ms,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY -- auto-switches based on settings.LLM_PROVIDER
# ══════════════════════════════════════════════════════════════════════════════
def get_evaluator(settings, ground_truth_path: Path):
    """
    Returns evaluator based on settings.LLM_PROVIDER.
    - ollama  -> CustomRAGEvaluator (local)
    - openai  -> RAGASEvaluator (production)
    """
    if settings.LLM_PROVIDER == "ollama":
        return CustomRAGEvaluator(settings)
    return RAGASEvaluator(settings, ground_truth_path)

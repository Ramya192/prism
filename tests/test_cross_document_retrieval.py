"""
tests/test_cross_document_retrieval.py
Real end-to-end test of cross-document retrieval -- item 3 of the
reference-corpora/HR-handbook/multi-document work, generalizing the same
mechanism core/rag/base_retriever_agent.py's `reference_source` param
already established (a document's own chunks and the domain's static
reference corpus compete in ONE fused HyDE/variant/RRF/rerank ranking,
not two retrieval passes merged after the fact) to more than one real
uploaded document. `source_document` now also accepts a list of
filenames -- see retrieve()'s and UnifiedDomainPipeline.run()'s updated
docstrings.

Uses Insurance's two existing, already-committed demo claim PDFs
(CLM001 -- clean, CLM003 -- approved > billed rule violation) since they
require no new fixtures and their real dollar figures are already known
(see domains/insurance/tools/generate_claim_document.py): CLM001 is
billed $480.00/approved $400.00; CLM003 is billed $500.00/approved
$9,999.00. A single query asking about BOTH claims at once can only be
answered correctly if retrieval actually pulled chunks from both
documents into the same context -- not just whichever one the retriever
happened to rank first.

Requires OPENAI_API_KEY (embeddings) and Ollama running locally
(LLM_PROVIDER=ollama, the default) or LLM_PROVIDER=openai set in .env.
Slow (real LLM calls) -- run standalone, not part of the fast suite.
"""

from pathlib import Path

import pytest

from core.orchestrator import AgentOrchestrator

CLM001 = Path("domains/insurance/data/statements/CLM001_claim.pdf")
CLM003 = Path("domains/insurance/data/statements/CLM003_claim.pdf")


@pytest.fixture(scope="session")
def pipeline():
    orchestrator = AgentOrchestrator()
    return orchestrator.get_pipeline("insurance")


@pytest.mark.skipif(
    not (CLM001.exists() and CLM003.exists()), reason="run generate_claim_document.py first",
)
class TestCrossDocumentRetrieval:
    @pytest.fixture(scope="class", autouse=True)
    def ingested(self, pipeline):
        pipeline.ingest(str(CLM001), filename="CLM001_claim")
        pipeline.ingest(str(CLM003), filename="CLM003_claim")

    def test_single_document_query_unaffected(self, pipeline):
        """A plain string source_document (the pre-existing, single-
        document code path) must still work exactly as before -- this is
        the regression check that the list-handling added to retrieve()
        and every domain's reasoning_agent.py didn't change single-source
        behavior at all."""
        result = pipeline.run("What was the billed claim amount?", source_document="CLM001_claim")
        assert result["status"] == "valid"
        assert "480" in result["data"]["answer"]

    def test_cross_document_query_draws_on_both_documents(self, pipeline):
        """The real test: one query, two source documents. Only
        answerable correctly if chunks from BOTH CLM001 and CLM003 made
        it into the same fused-ranked context -- a single-document
        retrieval (even the "best" one) structurally cannot state both
        figures correctly, since each amount only exists in one claim."""
        result = pipeline.run(
            "What is the billed claim amount on CLM001, and what is the "
            "approved amount on CLM003?",
            source_document=["CLM001_claim", "CLM003_claim"],
        )
        assert result["status"] == "valid"
        answer = result["data"]["answer"]
        assert "480" in answer
        assert "9,999" in answer or "9999" in answer

    def test_chunks_returned_come_from_more_than_one_source(self, pipeline):
        """Direct check on the retrieval layer itself (not just the final
        answer text) -- retrieve() should surface chunks tagged with both
        source filenames when both are requested together."""
        chunks = pipeline.retriever.retrieve(
            "claim amount", source_document=["CLM001_claim", "CLM003_claim"],
        )
        sources = {c["source"] for c in chunks}
        assert "CLM001_claim" in sources
        assert "CLM003_claim" in sources

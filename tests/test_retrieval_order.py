# retrieve() must hand back chunks in fused (RRF) order even when the vector
# store returns collection.get(ids=...) rows in some other order. Offline: the
# collection, embedder and LLM calls are all stubbed.

from types import SimpleNamespace

from core.rag.base_retriever_agent import BaseRetrieverAgent


class _Collection:
    """query() ranks a, b, c, d best-first; get() returns them scrambled."""
    def query(self, **kwargs):
        return {"ids": [["a", "b", "c", "d"]]}

    def get(self, ids, include, where=None):
        order = ["c", "a", "d", "b"]
        return {
            "ids": order,
            "documents": [f"text-{i}" for i in order],
            "metadatas": [{"source": "doc.csv", "chunk_index": 0} for _ in order],
        }


def test_retrieve_keeps_rrf_order_when_no_reranker_is_configured():
    agent = BaseRetrieverAgent.__new__(BaseRetrieverAgent)
    agent.settings = SimpleNamespace(TOP_K=2, LLM_PROVIDER="openai", COHERE_API_KEY=None)
    agent.collection = _Collection()
    agent.embedder = SimpleNamespace(embed_query=lambda q: [0.0])
    agent._hyde = lambda q: [0.0]
    agent._generate_variants = lambda q: []

    chunks = agent.retrieve("anything")

    assert [c["text"] for c in chunks] == ["text-a", "text-b"]   # the two best-ranked, best first

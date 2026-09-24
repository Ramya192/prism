# core/rag/base_retriever_agent.py
# Shared HyDE + multi-query-expansion + RRF fusion + rerank mechanics for
# every document-shaped domain's retriever agent. Extracted once a second,
# near-identical copy (payroll's) made the duplication undeniable --
# payroll's own file docstring already said "identical HyDE + multi-
# query-expansion + RRF fusion + rerank mechanics (none of that is
# banking-specific); only the HyDE/variant prompt strings change persona."
#
# Domain-specific behavior is three class attributes a subclass sets
# (PERSONA, HYDE_DOC_PHRASE, HYDE_INSTRUCTION, VARIANT_DOC_PHRASE), not a
# config flag or copy-pasted method.
#
# IMPORTANT: banking documents is live in production and already RAGAS-
# evaluated. This extraction changes zero logic and, critically, zero
# PROMPT TEXT for it -- its subclass's class attributes are set to
# reproduce its exact original _hyde()/_generate_variants() strings,
# character for character. Verify this if you ever touch the templates
# below: a prompt wording change here would be a real behavior change to
# a deployed, measured system, not a refactor.

import logging

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class BaseRetrieverAgent:
    # Override in subclasses -- see each domain's retriever_agent.py for
    # the exact strings that reproduce pre-refactor behavior.
    PERSONA = "a document analyst"
    HYDE_DOC_PHRASE = "document"
    HYDE_INSTRUCTION = "Be specific with relevant details."
    VARIANT_DOC_PHRASE = "a document"

    def __init__(self, settings):
        self.settings = settings
        self.embedder = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self._init_vector_db()

        if settings.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama
            from sentence_transformers import CrossEncoder

            self.llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=settings.OPENAI_MODEL)
            self.reranker = None

    def _init_vector_db(self):
        import chromadb

        self.chroma_client = chromadb.PersistentClient(path=self.settings.CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(name=self.settings.COLLECTION_NAME)

    def _hyde(self, query: str) -> list[float]:
        prompt = f"""You are {self.PERSONA}.
A user asked: "{query}"
Write a hypothetical {self.HYDE_DOC_PHRASE} excerpt that would answer this.
{self.HYDE_INSTRUCTION} Under 100 words."""
        response = self.llm.invoke(prompt)
        return self.embedder.embed_query(response.content)

    def _generate_variants(self, query: str) -> list[str]:
        prompt = f"""Generate 3 different search query variants for:
"{query}"
Context: searching {self.VARIANT_DOC_PHRASE}.
Return only 3 queries, one per line, no numbering, no bullets."""
        response = self.llm.invoke(prompt)
        variants = [p.strip() for p in response.content.split("\n") if p.strip()]
        return variants[:3]

    def _rrf_merge(self, ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        scores = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += 1 / (k + rank)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if self.settings.LLM_PROVIDER != "ollama":
            if not self.settings.COHERE_API_KEY:
                # No Cohere key configured -- degrade gracefully instead of
                # hard-failing every /analyze call. Chunks are already RRF-
                # merged across the HyDE + query-variant searches, so this
                # is a reasonable order, just without the cross-encoder
                # quality boost Cohere's rerank would add.
                logger.warning("COHERE_API_KEY not set -- skipping rerank, using RRF-merged order")
                for c in chunks:
                    c["rerank_score"] = None
                return chunks

            import cohere
            from cohere.core import ApiError as CohereApiError

            try:
                co = cohere.Client(self.settings.COHERE_API_KEY)
                results = co.rerank(
                    query=query,
                    documents=[c["text"] for c in chunks],
                    model="rerank-english-v3.0",
                )
            except CohereApiError:
                # A configured-but-invalid/expired key (401), rate limit
                # (429), or a Cohere-side outage should degrade the same
                # way an unset key already does above -- rerank is an
                # optional quality boost, not something that should take
                # down every /analyze or chat call when it fails.
                logger.warning("Cohere rerank call failed -- skipping rerank, using RRF-merged order", exc_info=True)
                for c in chunks:
                    c["rerank_score"] = None
                return chunks
            for r in results.results:
                chunks[r.index]["rerank_score"] = r.relevance_score
            return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.reranker.predict(pairs)
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def retrieve(
        self, query: str, source_document: str | list[str] = None, k: int = None,
        reference_source: str = None,
    ) -> list[dict]:
        """`reference_source` (a fixed collection-scoped id per domain,
        e.g. "__reference_corpus__") makes this document's own chunks and
        the domain's static reference corpus compete in the SAME
        HyDE/variant/RRF/rerank pipeline, rather than two separate
        retrieval passes merged after the fact -- see
        docs/UNIFIED_INGESTION_VISION.md's "chat should draw on more than
        just the uploaded document" section. None (the default) preserves
        exact pre-existing single-source behavior for callers that don't
        pass it.

        `source_document` also accepts a list of filenames -- cross-
        document retrieval, the same "let everything compete in one fused
        ranking" mechanism reference_source already established, just
        with more than one real uploaded document instead of one document
        plus the domain's static corpus. A single string keeps its exact
        pre-existing meaning and code path; only a list changes anything,
        so no existing single-document caller is affected."""
        if k is None:
            k = self.settings.TOP_K
        source_list = source_document if isinstance(source_document, list) else [source_document]
        sources = [s for s in (*source_list, reference_source) if s]
        hyde_vector = self._hyde(query)
        variants = self._generate_variants(query)

        all_ranked_lists = []

        if not sources:
            where_filter = None
        elif len(sources) == 1:
            where_filter = {"source": sources[0]}
        else:
            where_filter = {"source": {"$in": sources}}

        hyde_results = self.collection.query(
            query_embeddings=[hyde_vector], n_results=20, where=where_filter,
        )
        all_ranked_lists.append(hyde_results["ids"][0])

        for variant in variants:
            variant_vector = self.embedder.embed_query(variant)
            results = self.collection.query(
                query_embeddings=[variant_vector], n_results=20, where=where_filter,
            )
            all_ranked_lists.append(results["ids"][0])

        merged_ids = self._rrf_merge(all_ranked_lists)
        top20 = self.collection.get(
            ids=merged_ids[:20], include=["documents", "metadatas"], where=where_filter,
        )
        chunks = [
            {
                "text": top20["documents"][i],
                "source": top20["metadatas"][i]["source"],
                "chunk_index": top20["metadatas"][i]["chunk_index"],
                "score": 0,
            }
            for i in range(len(top20["documents"]))
        ]

        reranked = self._rerank(query, chunks)
        return reranked[:k]

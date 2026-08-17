# domains/bfsi_documents/agents/retriever_agent.py
# Ported from document_intelligence_system/agents/retriever_agent.py —
# logic unchanged, import paths moved under domains.bfsi_documents.

import logging

from langchain_openai import OpenAIEmbeddings
from domains.bfsi_documents.settings import Settings

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self):
        self.embedder = OpenAIEmbeddings(model=Settings.EMBEDDING_MODEL)
        self._init_vector_db()

        if Settings.LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama
            from sentence_transformers import CrossEncoder

            self.llm = ChatOllama(
                model=Settings.OLLAMA_MODEL, base_url=Settings.OLLAMA_BASE_URL
            )
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        else:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=Settings.OPENAI_MODEL)
            self.reranker = None

    def _init_vector_db(self):
        if Settings.VECTOR_DB == "qdrant":
            from qdrant_client import QdrantClient

            self.qdrant = QdrantClient(
                url=Settings.QDRANT_URL,
                api_key=Settings.QDRANT_API_KEY,
            )
            self.collection = None  # not used for qdrant
        else:
            import chromadb

            self.chroma_client = chromadb.PersistentClient(path=Settings.CHROMA_PATH)
            self.collection = self.chroma_client.get_or_create_collection(
                name=Settings.COLLECTION_NAME
            )
            self.qdrant = None

    def _hyde(self, query: str) -> list[float]:
        prompt = f"""You are a bank statement analyst.
A user asked: "{query}"
Write a hypothetical bank statement excerpt that would answer this.
Be specific with amounts, dates, transaction types. Under 100 words."""
        response = self.llm.invoke(prompt)
        return self.embedder.embed_query(response.content)

    def _generate_variants(self, query: str) -> list[str]:
        prompt = f"""Generate 3 different search query variants for:
"{query}"
Context: searching a bank statement PDF.
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

    def _search_qdrant(
        self, vector: list[float], n: int = 20, source: str = None
    ) -> list[str]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if source:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            )

        results = self.qdrant.query_points(
            collection_name=Settings.COLLECTION_NAME,
            query=vector,
            limit=n,
            with_payload=True,
            query_filter=query_filter,
        ).points
        return [str(r.id) for r in results]

    def _fetch_qdrant(self, ids: list[str]) -> list[dict]:
        from qdrant_client.models import Filter, HasIdCondition

        results = self.qdrant.scroll(
            collection_name=Settings.COLLECTION_NAME,
            scroll_filter=Filter(must=[HasIdCondition(has_id=[int(i) for i in ids])]),
            limit=len(ids),
            with_payload=True,
        )[0]
        return [
            {
                "text": r.payload.get("text", ""),
                "source": r.payload.get("source", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "score": 0,
            }
            for r in results
        ]

    def _rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        if Settings.LLM_PROVIDER != "ollama":
            if not Settings.COHERE_API_KEY:
                # No Cohere key configured -- degrade gracefully instead of
                # hard-failing every /analyze call. Chunks are already RRF-
                # merged across the HyDE + query-variant searches, so this
                # is a reasonable order, just without the cross-encoder
                # quality boost Cohere's rerank would add.
                logger.warning(
                    "COHERE_API_KEY not set -- skipping rerank, using RRF-merged order"
                )
                for c in chunks:
                    c["rerank_score"] = None
                return chunks

            import cohere

            co = cohere.Client(Settings.COHERE_API_KEY)
            results = co.rerank(
                query=query,
                documents=[c["text"] for c in chunks],
                model="rerank-english-v3.0",
            )
            for r in results.results:
                chunks[r.index]["rerank_score"] = r.relevance_score
            return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.reranker.predict(pairs)
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    def retrieve(
        self, query: str, source_document: str = None, k: int = Settings.TOP_K
    ) -> list[dict]:
        hyde_vector = self._hyde(query)
        variants = self._generate_variants(query)

        all_ranked_lists = []

        if Settings.VECTOR_DB == "qdrant":
            # HyDE search
            all_ranked_lists.append(
                self._search_qdrant(hyde_vector, source=source_document)
            )
            # Variant searches
            for variant in variants:
                variant_vector = self.embedder.embed_query(variant)
                all_ranked_lists.append(
                    self._search_qdrant(variant_vector, source=source_document)
                )

            # RRF merge
            merged_ids = self._rrf_merge(all_ranked_lists)

            # Fetch top-20 chunks
            chunks = self._fetch_qdrant(merged_ids[:20])

        else:
            # HyDE search
            where_filter = {"source": source_document} if source_document else None

            hyde_results = self.collection.query(
                query_embeddings=[hyde_vector],
                n_results=20,
                where=where_filter,
            )

            all_ranked_lists.append(hyde_results["ids"][0])

            # Variant searches
            for variant in variants:
                variant_vector = self.embedder.embed_query(variant)
                results = self.collection.query(
                    query_embeddings=[variant_vector],
                    n_results=20,
                    where=where_filter,
                )
                all_ranked_lists.append(results["ids"][0])

            # RRF merge
            merged_ids = self._rrf_merge(all_ranked_lists)

            # Fetch top-20 chunks
            top20 = self.collection.get(
                ids=merged_ids[:20],
                include=["documents", "metadatas"],
                where=where_filter,
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

        # Rerank and return top-k
        reranked = self._rerank(query, chunks)
        return reranked[:k]

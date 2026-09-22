# domains/insurance/settings.py
# Extended from a fraud-only OPENAI_MODEL-only settings file to the full
# RAG env-var pattern every document-capable domain uses (see
# domains/payroll_hr/payroll/settings.py) once Insurance gained a
# document/RAG capability -- own COLLECTION_NAME so ingested chunks never
# collide with another domain's vector store.

from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ChromaDB local persistent store
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_store")

    ENV = os.getenv("ENV", "local")

    COLLECTION_NAME = os.getenv("INSURANCE_COLLECTION_NAME", "insurance_claims")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama" if ENV == "local" else "openai")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    TOP_K = int(os.getenv("TOP_K", 5))

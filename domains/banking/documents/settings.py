# domains/banking/documents/settings.py
# Ported from document_intelligence_system/settings.py unchanged — same
# env vars, same local-vs-prod defaulting logic (ENV drives the
# LLM_PROVIDER default, same reasoning as the original comment below).

from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # ChromaDB local persistent store
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_store")

    ENV = os.getenv("ENV", "local")

    COLLECTION_NAME = os.getenv("BANKING_COLLECTION_NAME", "bank_statements")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # Unless overridden, only local dev uses
    # Ollama. Without this, LLM_PROVIDER defaulted to "ollama" everywhere,
    # including prod -- which meant sentence-transformers/CrossEncoder (and
    # the torch install it drags in) loaded on every deploy that forgot to
    # set this explicitly, regardless of host. That's almost certainly what
    # blew past Render's free-tier memory limit.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama" if ENV == "local" else "openai")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    TOP_K = int(os.getenv("TOP_K", 5))

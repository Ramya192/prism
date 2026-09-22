# domains/payroll_hr/payroll/settings.py
# Own copy of bfsi_documents/settings.py's env-var pattern — same local-vs-
# prod defaulting logic, own collection name so the two domains' ingested
# chunks never collide in the same vector store. See core/config_loader.py's
# capabilities design note: payroll deliberately does not share a runtime
# Settings instance with bfsi_documents, so a config change to one domain
# can never silently affect the other.

from dotenv import load_dotenv
import os

load_dotenv()


class Settings:
    # ChromaDB local persistent store
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_store")

    ENV = os.getenv("ENV", "local")

    COLLECTION_NAME = os.getenv("PAYROLL_COLLECTION_NAME", "payroll_registers")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama" if ENV == "local" else "openai")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    TOP_K = int(os.getenv("TOP_K", 5))

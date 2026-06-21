"""
Centralized, env-driven configuration.
Nothing in this app should hardcode an API key, model name, or path —
everything flows through here so local dev and the deployed Render
instance behave identically except for what's in the environment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings:
    # --- LLM / Embeddings ---
    # Local-only via Ollama — no API key, no cost, runs entirely on your machine.
    # See README "Local (Ollama) setup" for install + model pull steps.
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "llama3.2")
    JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "llama3.2")  # groundedness checker

    # --- Retrieval ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_RETRIEVE: int = int(os.getenv("TOP_K_RETRIEVE", "10"))   # broad candidate set
    TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "4"))          # after rerank
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.5"))  # dense vs sparse weight in RRF

    # --- Generation ---
    GENERATION_TEMPERATURE: float = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))

    # --- Guardrails ---
    ENABLE_INPUT_GUARDRAILS: bool = os.getenv("ENABLE_INPUT_GUARDRAILS", "true").lower() == "true"
    ENABLE_OUTPUT_GUARDRAILS: bool = os.getenv("ENABLE_OUTPUT_GUARDRAILS", "true").lower() == "true"

    # --- Storage ---
    DATA_DIR: Path = BASE_DIR / "data" / "sample_docs"
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
    COLLECTION_NAME: str = "collections_policy"

    # --- App ---
    APP_ENV: str = os.getenv("APP_ENV", "development")
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()

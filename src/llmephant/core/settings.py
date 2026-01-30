from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    API_HOST: str = "0.0.0.0"
    API_PORT: str = "8080"
    API_HOT_RELOAD: str = "false"
    UPSTREAM_OPENAI_BASE: str = "http://localhost:11434"
    UPSTREAM_OPENAI_API_KEY: str = "local"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "llmephant"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    MEMORY_MODEL_NAME: str = "qwen2.5:14b-instruct"

    # Memory extraction tuning knobs (optional; safe defaults)
    MEMORY_EXTRACT_MAX_TOKENS: Optional[int] = None
    MEMORY_DISTILL_MAX_TOKENS: Optional[int] = None
    MEMORY_VERIFY_MAX_TOKENS: int = 300

    # If supported by the upstream backend, this can reduce latency for models that emit reasoning.
    # Examples: "none", "low", "medium", "high". Leave unset to avoid backend-specific coupling.
    MEMORY_REASONING_EFFORT: Optional[str] = None

    MEMORY_MIN_CONFIDENCE: float = 0.9
    MEMORY_SIMILARITY_THRESHOLD: float = 0.55
    # Near-duplicate threshold for semantic dedupe (stricter than retrieval).
    MEMORY_DEDUPE_THRESHOLD: float = 0.95
    MEMORY_TTL_DAYS: int = 365
    ENABLE_MEMORY_EXTRACTION: str = "true"
    DEFAULT_USER_ID: str = "local-user"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

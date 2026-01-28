from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    API_HOST: str
    API_PORT: str
    API_HOT_RELOAD: str
    TOOLING_CONFIG_FILE: str
    UPSTREAM_OPENAI_BASE: str
    UPSTREAM_OPENAI_API_KEY: str
    QDRANT_URL: str
    QDRANT_COLLECTION: str
    EMBEDDING_MODEL_NAME: str
    MEMORY_MODEL_NAME: str

    # Memory extraction tuning knobs (optional; safe defaults)
    MEMORY_EXTRACT_MAX_TOKENS: Optional[int] = None
    MEMORY_DISTILL_MAX_TOKENS: Optional[int] = None
    MEMORY_VERIFY_MAX_TOKENS: int = 300

    # If supported by the upstream backend, this can reduce latency for models that emit reasoning.
    # Examples: "none", "low", "medium", "high". Leave unset to avoid backend-specific coupling.
    MEMORY_REASONING_EFFORT: Optional[str] = None

    MEMORY_MIN_CONFIDENCE: float
    MEMORY_SIMILARITY_THRESHOLD: float
    # Near-duplicate threshold for semantic dedupe (stricter than retrieval).
    MEMORY_DEDUPE_THRESHOLD: float = 0.95
    MEMORY_TTL_DAYS: int
    ENABLE_MEMORY_EXTRACTION: str
    DEFAULT_USER_ID: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()

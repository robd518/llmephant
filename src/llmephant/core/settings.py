from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_HOST: str
    API_PORT: str
    API_HOT_RELOAD: str
    UPSTREAM_OPENAI_BASE: str
    UPSTREAM_OPENAI_API_KEY: str
    QDRANT_URL: str
    QDRANT_COLLECTION: str
    EMBEDDING_MODEL_NAME: str
    MEMORY_MODEL_NAME: str
    MEMORY_MIN_CONFIDENCE: float
    MEMORY_SIMILARITY_THRESHOLD: float
    MEMORY_TTL_DAYS: int
    ENABLE_MEMORY_EXTRACTION: str
    DEFAULT_USER_ID: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

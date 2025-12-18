class LLMephantError(Exception):
    """Base exception for the service."""


class UpstreamLLMError(LLMephantError):
    """Raised when the upstream LLM returns an error."""


class MemoryExtractionError(LLMephantError):
    """Raised when memory extraction JSON cannot be parsed."""


class VectorStoreError(LLMephantError):
    """Raised for Qdrant/vector DB issues."""
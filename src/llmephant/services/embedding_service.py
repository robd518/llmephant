import logging
import threading
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from llmephant.core.settings import settings

logger = logging.getLogger(__name__)

embedder: Optional[SentenceTransformer] = None
vector_dim: Optional[int] = None
_init_lock = threading.Lock()


def init_embedder(*, force: bool = False) -> None:
    """Initialize the global sentence-transformers embedder.

    This is safe to call multiple times; it will no-op unless `force=True`.
    """
    global embedder, vector_dim

    if embedder is not None and not force:
        return

    with _init_lock:
        if embedder is not None and not force:
            return

        model_name = settings.EMBEDDING_MODEL_NAME
        logger.info("Initializing embedder", extra={"embedding_model": model_name})
        emb = SentenceTransformer(model_name)
        dim = emb.get_sentence_embedding_dimension()

        embedder = emb
        vector_dim = dim
        logger.info("Embedder initialized", extra={"embedding_model": model_name, "vector_dim": dim})


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts.

    Lazily initializes the embedder if needed. Raises RuntimeError with a clear
    message if embeddings cannot be produced.
    """
    if not texts:
        return []

    global embedder
    if embedder is None:
        try:
            init_embedder()
        except Exception as e:
            logger.exception("Failed to initialize embedder")
            raise RuntimeError(
                "Embedding model is not initialized (failed to init). "
                "Set EMBEDDING_MODEL_NAME and ensure the model can be loaded."
            ) from e

    # At this point embedder should be available.
    assert embedder is not None

    try:
        return embedder.encode(texts, convert_to_numpy=True).tolist()
    except Exception as e:
        logger.exception("Embedding encode() failed")
        raise RuntimeError("Embedding encode() failed") from e

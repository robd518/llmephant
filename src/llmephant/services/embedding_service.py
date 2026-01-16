from sentence_transformers import SentenceTransformer
from llmephant.core.settings import settings

embedder = None
vector_dim = None


def init_embedder():
    global embedder, vector_dim
    embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    vector_dim = embedder.get_sentence_embedding_dimension()


def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True).tolist()

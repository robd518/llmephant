from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from llmephant.core.settings import settings
import datetime
import uuid

client = None


def init_qdrant():
    global client
    client = QdrantClient(url=settings.QDRANT_URL, prefer_grpc=False)

    existing = client.get_collections()
    if settings.QDRANT_COLLECTION not in {c.name for c in existing.collections}:
        from services.embedding_service import vector_dim

        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )


def qdrant_search(user_id, query_vec, top_k, *, category: str | None = None):
    must = [{"key": "user_id", "match": {"value": user_id}}]
    if category:
        must.append({"key": "category", "match": {"value": category}})
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter={"must": must},
    )
    return [
        {
            "text": p.payload.get("text", ""),
            "score": p.score,
            "created_at": p.payload.get("created_at"),
            "category": p.payload.get("category"),
        }
        for p in results.points
    ]


def qdrant_upsert(user_id, texts, vectors, *, categories=None):
    now = datetime.datetime.utcnow().isoformat() + "Z"
    expiry = (
        datetime.datetime.utcnow() + datetime.timedelta(days=settings.MEMORY_TTL_DAYS)
    ).isoformat() + "Z"

    if categories is None:
        categories = [None] * len(texts)
    if len(categories) != len(texts):
        raise ValueError("categories length must match texts length")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "user_id": user_id,
                "text": text,
                "created_at": now,
                "expires_at": expiry,
                **({"category": category} if category else {}),
            },
        )
        for text, vec, category in zip(texts, vectors, categories)
    ]

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points, wait=True)


def delete_expired():
    now = datetime.datetime.utcnow().isoformat() + "Z"
    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        filter={"must": [{"key": "expires_at", "range": {"lt": now}}]},
    )

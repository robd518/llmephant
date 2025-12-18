import json
from typing import List, Optional, TypedDict

from llmephant.core.settings import settings
from llmephant.core.logger import setup_logger
from llmephant.models.chat_model import ChatRequest, ChatMessage
from llmephant.repositories.qdrant_repository import qdrant_search, qdrant_upsert
from llmephant.services.embedding_service import embed_texts
from llmephant.services.normalization import normalize_fact
from llmephant.services.upstream_llm import chat_upstream

logger = setup_logger(__name__)


class MemoryHit(TypedDict):
    text: str
    score: float
    created_at: Optional[str]


def handle_explicit_remember_request(user_id: str, last_msg: str) -> None:
    if not last_msg:
        return
    if last_msg.lower().startswith("remember that"):
        fact = last_msg.replace("remember that", "", 1).strip()
        store_facts(user_id, [fact])


def search_relevant_memories(user_id: str, query: str) -> List[MemoryHit]:
    if not query or not query.strip():
        return []
    query_vec = embed_texts([query])[0]
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD
    raw = qdrant_search(user_id, query_vec, top_k=8)
    memories: List[MemoryHit] = []
    for m in raw:
        score = float(m.get("score", 0.0) or 0.0)
        if score < threshold:
            continue
        memories.append(
            {
                "text": m.get("text", ""),
                "score": score,
                "created_at": m.get("created_at"),
            }
        )
    return memories


def augment_messages_with_memory(
    messages: List[ChatMessage],
    memories: List[MemoryHit],
) -> List[ChatMessage]:
    if not memories:
        return messages

    verified = "\n".join(f"- {m['text']}" for m in memories)
    prefix = ChatMessage(
        role="system",
        content=(
            "The following are VERIFIED FACTS about the USER.\n"
            "Treat these as true ground truth and use naturally.\n\n"
            f"{verified}"
        ),
    )
    return [prefix, *messages]


def _strip_code_fences(text: str) -> str:
    """
    Remove surrounding ``` or ```json fences if present.
    """
    t = (text or "").strip()
    if not t.startswith("```"):
        return t

    # drop the first fence line (``` or ```json)
    first_nl = t.find("\n")
    if first_nl != -1:
        t = t[first_nl + 1 :]
    # drop trailing fence
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Best-effort extraction of a JSON object from `text`.
    Handles:
      - raw JSON object
      - JSON wrapped in code fences
      - extra prose where the first {...} is the JSON object
    """
    if not text:
        return None

    candidate = _strip_code_fences(text)

    # Fast path: whole-string JSON
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Heuristic: take the first {...} span
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(candidate[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


async def extract_and_store_memory(
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
) -> bool:
    """
    Returns True only when memories were extracted AND stored.
    Returns False when skipped, parsing fails, or confidence is too low.
    """
    if not settings.ENABLE_MEMORY_EXTRACTION:
        return False

    # Only use USER utterances to avoid extracting assistant hallucinations as "user facts".
    transcript = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in messages
        if m.role == "user" and m.content
    ).strip()

    if not transcript:
        return False

    chosen_model = settings.MEMORY_MODEL_NAME or model_name
    if not settings.MEMORY_MODEL_NAME:
        logger.warning(
            "MEMORY_MODEL_NAME is not set; falling back to the chat model for memory extraction. "
            "If your chat model does not reliably output JSON in message.content, set MEMORY_MODEL_NAME."
        )

    convo = [
        ChatMessage(
            role="system",
            content=(
                "You are a memory extraction assistant.\n"
                "Extract long-term, stable facts about the USER.\n\n"
                "Rules:\n"
                "- Only include durable facts (preferences, identity, long-term habits).\n"
                "- Write facts in third person.\n"
                "- Do NOT include assistant opinions or responses.\n"
                "- Respond ONLY as valid JSON (no markdown, no prose).\n\n"
                "JSON format:\n"
                "{\n"
                '  \"confidence\": float between 0 and 1,\n'
                '  \"facts\": [string, ...]\n'
                "}"
            ),
        ),
        ChatMessage(role="user", content=transcript),
    ]

    resp = await chat_upstream(
        ChatRequest(
            model=chosen_model,
            messages=convo,
            temperature=0.0,
        )
    )

    if not isinstance(resp, dict):
        logger.error(f"Memory extractor returned non-dict response: {resp}")
        return False
    if "error" in resp:
        logger.error(f"Memory extractor upstream error: {resp}")
        return False
    if "choices" not in resp:
        logger.error(f"Memory extractor missing 'choices': {resp}")
        return False

    try:
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or "").strip()

        # Some backends populate reasoning_content but leave content empty.
        # This is not ideal, but we can attempt parsing as a fallback.
        if not content:
            rc = (msg.get("reasoning_content") or "").strip()
            if rc:
                logger.warning(
                    "Memory extractor returned empty message.content; attempting JSON parse from reasoning_content fallback."
                )
                content = rc

        obj = _extract_json_object(content)
        if obj is None:
            logger.error(
                "Failed to parse memory extractor response as JSON object. "
                f"Raw startswith={content[:160]!r}"
            )
            return False

        conf = float(obj.get("confidence", 0.0) or 0.0)
        facts = obj.get("facts", [])

        if not isinstance(facts, list):
            logger.error(f"Memory extractor 'facts' is not a list: {facts!r}")
            return False

        facts = [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception as e:
        logger.error(
            f"Failed to parse memory extractor response. Raw choices: {resp.get('choices')}. Error: {str(e)}"
        )
        return False

    if conf < settings.MEMORY_MIN_CONFIDENCE:
        return False

    store_facts(user_id, facts)
    return True


def store_facts(user_id: str, facts: List[str]) -> None:
    cleaned = [normalize_fact(f) for f in facts if isinstance(f, str) and f.strip()]
    if not cleaned:
        return

    vecs = embed_texts(cleaned)
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD

    # Minimal dedupe: avoid inserting if the top hit is already very similar.
    existing = qdrant_search(user_id, vecs[0], top_k=1)
    if existing and existing[0].get("score", 0.0) >= threshold:
        return

    qdrant_upsert(user_id, cleaned, vecs)

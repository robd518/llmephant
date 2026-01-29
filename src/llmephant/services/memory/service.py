import json
from typing import Any, Dict, List, Optional, TypedDict

from llmephant.core.logger import setup_logger
from llmephant.core.settings import settings
from llmephant.models.chat_model import ChatMessage, ChatRequest
from llmephant.repositories.qdrant_repository import qdrant_search, qdrant_upsert
from llmephant.services.embedding_service import embed_texts
from llmephant.services.normalization import normalize_fact
from llmephant.services.upstream_llm import chat_upstream

from .json_utils import extract_json_object, get_finish_reason, strip_code_fences
from .prompts import EXTRACT_PROMPT

logger = setup_logger(__name__)

# Hard cutover: ALL memories live in ONE bucket under base `user_id`.
# No workspace/profile/analysis namespaces. No lane-specific logic. No verify pass.


# Keep stored notes compact.
MEMORY_MAX_NOTE_CHARS = 6000
MEMORY_SUMMARY_CAP = 2500
MEMORY_MAX_OBSERVABLES = 30
MEMORY_MAX_TAGS = 15
MEMORY_MAX_NOTES = 5


class MemoryHit(TypedDict):
    text: str
    score: float
    created_at: Optional[str]
    category: Optional[str]


class MemoryItem(TypedDict):
    text: str
    category: str


def _log_memory(event: str, **fields: Any) -> None:
    """Structured-ish memory logging without leaking message content."""
    safe: Dict[str, Any] = {k: v for k, v in fields.items() if v is not None}
    parts = " ".join(f"{k}={safe[k]!r}" for k in sorted(safe.keys()))
    logger.info(f"memory.{event}" + (f" {parts}" if parts else ""))


def _clip_text(s: str, *, head: int = 240, tail: int = 240) -> str:
    """Return a compact head/tail preview for diagnostics."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    if len(s) <= head + tail + 16:
        return s
    return f"{s[:head]}…<snip {len(s) - head - tail} chars>…{s[-tail:]}"


def _sanitize_json_control_chars(s: str) -> str:
    """Best-effort removal of control characters that can break JSON parsing."""
    if not isinstance(s, str) or not s:
        return "" if s is None else str(s)

    out: List[str] = []
    for ch in s:
        if ch == "\t":
            out.append(" ")
            continue
        o = ord(ch)
        if o < 32 and ch not in ("\n", "\r"):
            continue
        out.append(ch)
    return "".join(out)


def _filter_chatrequest_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to keys supported by ChatRequest (Pydantic v1/v2 safe)."""
    fields = getattr(ChatRequest, "model_fields", None) or getattr(ChatRequest, "__fields__", None)
    allowed = set(fields.keys()) if isinstance(fields, dict) else None

    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if allowed is not None and k not in allowed:
            continue
        out[k] = v
    return out


def _build_user_transcript(messages: List[ChatMessage]) -> str:
    """Join user-only messages into a single transcript for evidence checks."""
    parts: List[str] = []
    for m in messages:
        if getattr(m, "role", None) != "user":
            continue
        content = getattr(m, "content", None)
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts).strip()


def _normalize_for_match(text: str) -> str:
    """Deterministic normalization for evidence matching (collapse whitespace)."""
    if not isinstance(text, str) or not text:
        return ""
    return " ".join(text.split()).casefold()


def _evidence_in_source(evidence: str, source: str) -> bool:
    """True if evidence appears verbatim in the source (after whitespace normalization)."""
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    if not isinstance(source, str) or not source.strip():
        return False
    return _normalize_for_match(evidence) in _normalize_for_match(source)


def _memory_request_knobs() -> Dict[str, Any]:
    """Backend-agnostic knobs for the extraction pass."""
    max_tokens = getattr(settings, "MEMORY_DISTILL_MAX_TOKENS", None)

    effort = getattr(settings, "MEMORY_REASONING_EFFORT", None)
    reasoning_obj = {"effort": effort} if isinstance(effort, str) and effort.strip() else None

    return {
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
        "reasoning": reasoning_obj,
    }


def handle_explicit_remember_request(user_id: str, last_msg: str) -> None:
    if not last_msg:
        return
    lowered = last_msg.lower()
    prefix = "remember that"
    if lowered.startswith(prefix):
        fact = last_msg[len(prefix):].strip()
        if fact:
            store_facts(user_id, [fact])


def _try_embed_query(user_id: str, query: str, *, purpose: str) -> Optional[List[float]]:
    """Best-effort embed of a single query. Returns None if embeddings are unavailable."""
    try:
        return embed_texts([query])[0]
    except Exception as e:
        _log_memory("embed.unavailable", user_id=user_id, purpose=purpose, err=type(e).__name__)
        logger.warning(
            f"Embeddings unavailable for purpose={purpose!r}; skipping memory op. "
            f"err={type(e).__name__}: {e}"
        )
        return None


def search_relevant_memories(user_id: str, query: str) -> List[MemoryHit]:
    """Single-bucket retrieval."""
    if not query or not query.strip():
        return []

    query_vec = _try_embed_query(user_id, query, purpose="recall")
    if query_vec is None:
        return []

    threshold = settings.MEMORY_SIMILARITY_THRESHOLD
    raw = qdrant_search(user_id, query_vec, top_k=8)

    _log_memory(
        "search.debug.raw",
        user_id=user_id,
        query_preview=query[:120],
        threshold=settings.MEMORY_SIMILARITY_THRESHOLD,
        top_scores=[float(r.get("score", 0.0) or 0.0) for r in raw[:5]],
        top_texts=[(r.get("text","")[:80]) for r in raw[:3]],
    )

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
                "category": m.get("category"),
            }
        )
    return memories


def augment_messages_with_memory(messages: List[ChatMessage], memories: List[MemoryHit]) -> List[ChatMessage]:
    if not memories:
        return messages

    def _fmt(m: MemoryHit) -> str:
        cat = m.get("category")
        label = f"[{cat}] " if cat else ""
        return f"- {label}{m['text']}"

    ctx = "\n".join(_fmt(m) for m in memories)
    prefix = ChatMessage(
        role="system",
        content=(
            "The following are RELEVANT MEMORIES for the USER.\n"
            "Use these as helpful context and recall.\n\n"
            f"{ctx}"
        ),
    )
    return [prefix, *messages]


# --- Compatibility wrappers (still ONE bucket) ---
def search_relevant_workspace_memories(user_id: str, query: str) -> List[MemoryHit]:
    return search_relevant_memories(user_id, query)


def augment_messages_with_workspace_memories(messages: List[ChatMessage], memories: List[MemoryHit]) -> List[ChatMessage]:
    return augment_messages_with_memory(messages, memories)


def search_relevant_analysis_memories(user_id: str, query: str) -> List[MemoryHit]:
    return search_relevant_memories(user_id, query)


def augment_messages_with_analysis_memories(messages: List[ChatMessage], memories: List[MemoryHit]) -> List[ChatMessage]:
    return augment_messages_with_memory(messages, memories)


def _collect_tool_names(messages: List[ChatMessage]) -> List[str]:
    """Best-effort tool name extraction without depending on a provider schema."""
    names: List[str] = []
    for m in messages:
        if getattr(m, "role", None) == "tool":
            n = getattr(m, "name", None) or getattr(m, "tool_name", None)
            if isinstance(n, str) and n.strip():
                names.append(n.strip())

        tool_calls = getattr(m, "tool_calls", None)
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    n = fn.get("name") or tc.get("name")
                else:
                    fn = getattr(tc, "function", None)
                    if isinstance(fn, dict):
                        n = fn.get("name")
                    else:
                        n = getattr(fn, "name", None)
                    n = n or getattr(tc, "name", None)

                if isinstance(n, str) and n.strip():
                    names.append(n.strip())

    seen = set()
    out: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _build_tool_transcript(
    messages: List[ChatMessage],
    *,
    max_total_chars: int = 4000,
    max_per_tool_chars: int = 1200,
) -> str:
    """Compact transcript of tool outputs for distillation input."""
    parts: List[str] = []
    total = 0

    for m in messages:
        if getattr(m, "role", None) != "tool":
            continue
        name = getattr(m, "name", None) or getattr(m, "tool_name", None) or "tool"
        content = getattr(m, "content", None)
        if not isinstance(content, str) or not content.strip():
            continue

        cleaned = content.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
        if len(cleaned) > max_per_tool_chars:
            cleaned = cleaned[: max_per_tool_chars - 1].rstrip() + "…"

        chunk = f"[{name}] {cleaned}"
        if total + len(chunk) + 1 > max_total_chars:
            remaining = max_total_chars - total
            if remaining <= 16:
                break
            chunk = chunk[: remaining - 1].rstrip() + "…"
            parts.append(chunk)
            break

        parts.append(chunk)
        total += len(chunk) + 1

    return "\n".join(parts).strip()


def _filter_verbatim_observables(observables: List[str], source_text: str) -> List[str]:
    """Guardrail: keep only observables that appear verbatim in the provided text."""
    hay = (source_text or "").lower()
    out: List[str] = []
    for o in observables or []:
        if not isinstance(o, str):
            continue
        s = o.strip()
        if not s:
            continue
        if s.lower() in hay:
            out.append(s)

    seen = set()
    final: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        final.append(s)
    return final


def _dedupe_for_upsert(
    user_id: str, texts: List[str], categories: Optional[List[Optional[str]]] = None
) -> tuple[List[str], List[List[float]], List[Optional[str]], int]:
    """Return (texts_to_insert, vecs_to_insert, categories_to_insert, skipped) after similarity-based dedupe."""
    try:
        vecs = embed_texts(texts)
    except Exception as e:
        raise RuntimeError("Embeddings unavailable") from e

    threshold = getattr(settings, "MEMORY_DEDUPE_THRESHOLD", settings.MEMORY_SIMILARITY_THRESHOLD)
    if categories is None:
        categories = [None] * len(texts)

    to_insert_texts: List[str] = []
    to_insert_vecs: List[List[float]] = []
    to_insert_categories: List[Optional[str]] = []
    skipped = 0

    for text, vec, category in zip(texts, vecs, categories):
        existing = qdrant_search(user_id, vec, top_k=1, category=category)
        if existing and float(existing[0].get("score", 0.0) or 0.0) >= threshold:
            skipped += 1
            continue
        to_insert_texts.append(text)
        to_insert_vecs.append(vec)
        to_insert_categories.append(category)

    return to_insert_texts, to_insert_vecs, to_insert_categories, skipped


def store_facts(user_id: str, facts: List[str], *, category: Optional[str] = None) -> int:
    cleaned = [normalize_fact(f) for f in facts if isinstance(f, str) and f.strip()]
    if not cleaned:
        return 0

    try:
        categories = [category] * len(cleaned) if category else None
        to_insert_texts, to_insert_vecs, to_insert_categories, skipped = _dedupe_for_upsert(
            user_id, cleaned, categories
        )
    except RuntimeError as e:
        _log_memory("store.skip.embed_unavailable", user_id=user_id)
        logger.warning(f"Skipping memory store: {e}")
        return 0

    if not to_insert_texts:
        _log_memory("store.none_deduped", user_id=user_id, skipped=skipped)
        return 0

    qdrant_upsert(user_id, to_insert_texts, to_insert_vecs, categories=to_insert_categories)
    _log_memory("store.upsert", user_id=user_id, inserted=len(to_insert_texts), skipped=skipped)
    return len(to_insert_texts)


def store_workspace_facts(user_id: str, facts: List[str]) -> int:
    return store_facts(user_id, facts, category="workspace")


def store_memory_items(user_id: str, items: List[MemoryItem]) -> int:
    cleaned_texts: List[str] = []
    categories: List[Optional[str]] = []

    for item in items:
        text = item.get("text")
        category = item.get("category")
        if not isinstance(text, str) or not text.strip():
            continue
        if category not in {"profile", "workspace"}:
            continue
        cleaned_texts.append(normalize_fact(text))
        categories.append(category)

    if not cleaned_texts:
        return 0

    try:
        to_insert_texts, to_insert_vecs, to_insert_categories, skipped = _dedupe_for_upsert(
            user_id, cleaned_texts, categories
        )
    except RuntimeError as e:
        _log_memory("store.skip.embed_unavailable", user_id=user_id)
        logger.warning(f"Skipping memory store: {e}")
        return 0

    if not to_insert_texts:
        _log_memory("store.none_deduped", user_id=user_id, skipped=skipped)
        return 0

    qdrant_upsert(user_id, to_insert_texts, to_insert_vecs, categories=to_insert_categories)
    _log_memory("store.upsert", user_id=user_id, inserted=len(to_insert_texts), skipped=skipped)
    return len(to_insert_texts)


async def _distill_memories(
    *,
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
    req_id: Optional[str] = None,
) -> List[MemoryItem]:
    """Extract recall-friendly notes from user-only transcript with evidence gating."""
    if not assistant_reply or not assistant_reply.strip():
        return []

    user_transcript = _build_user_transcript(messages)
    if not user_transcript:
        _log_memory("extract.skip.no_user_transcript", user_id=user_id, req_id=req_id)
        return []

    chosen_model = settings.MEMORY_MODEL_NAME or model_name
    knobs = _memory_request_knobs()

    convo = [
        ChatMessage(role="system", content=EXTRACT_PROMPT),
        ChatMessage(
            role="user",
            content=(
                user_transcript
                + "\n\nOUTPUT REQUIREMENTS:\n"
                + "- Return ONLY a single JSON object matching the extractor schema.\n"
                + "- Do NOT include markdown or prose.\n"
                + "- Evidence MUST be an exact substring from the user transcript.\n"
                + "- Avoid literal tab characters; use spaces.\n"
            ),
        ),
    ]

    req_kwargs = _filter_chatrequest_kwargs(
        {
            "model": chosen_model,
            "messages": convo,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            **knobs,
        }
    )

    resp = await chat_upstream(ChatRequest(**req_kwargs), req_id=req_id)
    finish_reason = get_finish_reason(resp)

    if not isinstance(resp, dict) or "choices" not in resp or "error" in resp:
        _log_memory("extract.skip.upstream_error", user_id=user_id, model=chosen_model, finish_reason=finish_reason, req_id=req_id)
        logger.error(f"Memory extractor upstream error/shape: {resp}")
        return []

    msg = resp["choices"][0]["message"]
    content = _sanitize_json_control_chars((msg.get("content") or "").strip())

    if not content:
        rc = (msg.get("reasoning_content") or "").strip()
        if rc:
            logger.warning("Memory extractor returned empty message.content; attempting JSON parse from reasoning_content fallback.")
            content = _sanitize_json_control_chars(rc)

    _log_memory("extract.raw.preview", user_id=user_id, model=chosen_model, finish_reason=finish_reason, content_len=len(content), preview=_clip_text(content), req_id=req_id)

    obj = extract_json_object(content)
    if obj is None:
        parse_err: Optional[str] = None
        try:
            json.loads(strip_code_fences(content))
        except Exception as e:
            parse_err = repr(e)

        _log_memory("extract.skip.parse_failed", user_id=user_id, model=chosen_model, finish_reason=finish_reason, content_len=len(content), req_id=req_id)
        logger.error(
            "Failed to parse memory extractor response as JSON object. "
            f"finish_reason={finish_reason!r} "
            + (f"err={parse_err} " if parse_err else "")
            + f"head={content[:160]!r} tail={content[-240:]!r}"
        )
        return []

    items = obj.get("items", [])
    if not isinstance(items, list) or not items:
        _log_memory("extract.skip.no_items", user_id=user_id, model=chosen_model, finish_reason=finish_reason, req_id=req_id)
        return []

    out_items: List[MemoryItem] = []
    dropped = 0
    for m in items[:MEMORY_MAX_NOTES]:
        if not isinstance(m, dict):
            dropped += 1
            continue

        text = m.get("text")
        category = m.get("category")
        evidence = m.get("evidence")

        if not isinstance(text, str) or not text.strip():
            dropped += 1
            continue
        if category not in {"profile", "workspace"}:
            dropped += 1
            continue
        if not _evidence_in_source(evidence, user_transcript):
            dropped += 1
            continue

        s = text.strip()
        if len(s) > MEMORY_SUMMARY_CAP:
            s = s[: MEMORY_SUMMARY_CAP - 1].rstrip() + "…"

        if len(s) > MEMORY_MAX_NOTE_CHARS:
            s = s[: MEMORY_MAX_NOTE_CHARS - 1].rstrip() + "…"

        out_items.append({"text": s, "category": category})

    _log_memory(
        "extract.items.filtered",
        user_id=user_id,
        kept=len(out_items),
        dropped=dropped,
        req_id=req_id,
    )

    return out_items


async def extract_and_store_memory(
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
    req_id: Optional[str] = None,
) -> bool:
    """Hard cutover: one pass (extract) + one bucket (store_facts under base user_id)."""
    if not settings.ENABLE_MEMORY_EXTRACTION:
        _log_memory("skip.disabled", user_id=user_id, req_id=req_id)
        return False

    chosen_model = settings.MEMORY_MODEL_NAME or model_name
    if not settings.MEMORY_MODEL_NAME:
        logger.warning(
            "MEMORY_MODEL_NAME is not set; falling back to the chat model for memory extraction. "
            "If your chat model does not reliably output JSON in message.content, set MEMORY_MODEL_NAME."
        )

    knobs = _memory_request_knobs()
    _log_memory(
        "models.selected",
        user_id=user_id,
        req_id=req_id,
        extract_model=chosen_model,
        extract_max_tokens=knobs.get("max_tokens"),
        reasoning_effort=knobs.get("reasoning_effort"),
        reasoning=knobs.get("reasoning"),
    )

    try:
        items = await _distill_memories(
            user_id=user_id,
            messages=messages,
            assistant_reply=assistant_reply,
            model_name=model_name,
            req_id=req_id,
        )
    except Exception as e:
        logger.error(f"Failed to extract memories: {e}")
        return False

    if not items:
        _log_memory("store.skip.no_notes", user_id=user_id, req_id=req_id)
        return False

    inserted = store_memory_items(user_id, items)
    if inserted > 0:
        _log_memory("store.ok", user_id=user_id, inserted=inserted, req_id=req_id)
        return True

    _log_memory("store.deduped", user_id=user_id, skipped=len(items), req_id=req_id)
    return False

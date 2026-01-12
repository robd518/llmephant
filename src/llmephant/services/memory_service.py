import json
from typing import Any, Dict, List, Optional, TypedDict

from llmephant.core.settings import settings
from llmephant.core.logger import setup_logger
from llmephant.models.chat_model import ChatRequest, ChatMessage
from llmephant.repositories.qdrant_repository import qdrant_search, qdrant_upsert
from llmephant.services.embedding_service import embed_texts
from llmephant.services.normalization import normalize_fact
from llmephant.services.upstream_llm import chat_upstream

logger = setup_logger(__name__)


def _log_memory(event: str, **fields: Any) -> None:
    """Structured-ish memory logging without leaking message content."""
    safe: Dict[str, Any] = {k: v for k, v in fields.items() if v is not None}
    # Keep logs compact and stable.
    parts = " ".join(f"{k}={safe[k]!r}" for k in sorted(safe.keys()))
    logger.info(f"memory.{event}" + (f" {parts}" if parts else ""))



class MemoryHit(TypedDict):
    text: str
    score: float
    created_at: Optional[str]


class AnalysisMemoryCandidate(TypedDict, total=False):
    text: str
    observables: List[str]
    tags: List[str]


class AnalysisDistillResult(TypedDict, total=False):
    store: bool
    memories: List[AnalysisMemoryCandidate]


def handle_explicit_remember_request(user_id: str, last_msg: str) -> None:
    if not last_msg:
        return

    lowered = last_msg.lower()
    prefix = "remember that"
    if lowered.startswith(prefix):
        fact = last_msg[len(prefix):].strip()
        if fact:
            store_facts(user_id, [fact])
def _workspace_namespace_user_id(user_id: str) -> str:
    """Namespace current projects / work context separately from durable user profile facts."""
    return f"{user_id}::workspace"


def search_relevant_workspace_memories(user_id: str, query: str) -> List[MemoryHit]:
    """Search current work context (workspace namespace) relevant to the current query."""
    if not query or not query.strip():
        return []

    query_vec = embed_texts([query])[0]
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD
    raw = qdrant_search(_workspace_namespace_user_id(user_id), query_vec, top_k=8)

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
def augment_messages_with_workspace_memories(
    messages: List[ChatMessage],
    memories: List[MemoryHit],
) -> List[ChatMessage]:
    """Inject current work context as an advisory system message."""
    if not memories:
        return messages

    ctx = "\n".join(f"- {m['text']}" for m in memories)
    prefix = ChatMessage(
        role="system",
        content=(
            "The following are CURRENT WORK CONTEXT items for the USER.\n"
            "These are likely true recently, but may change over time; treat as helpful context.\n\n"
            f"{ctx}"
        ),
    )
    return [prefix, *messages]
class UserMemoryItem(TypedDict, total=False):
    text: str
    category: str  # "profile" | "workspace"
    evidence: str  # verbatim substring from USER transcript


class UserMemoryExtractResult(TypedDict, total=False):
    confidence: float
    items: List[UserMemoryItem]
def _evidence_is_verbatim(evidence: str, transcript: str) -> bool:
    if not evidence or not transcript:
        return False
    # Case-sensitive match is intentional for "verbatim".
    return evidence in transcript


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


def _analysis_namespace_user_id(user_id: str) -> str:
    """Namespace analysis/investigation notes separately from user facts."""
    return f"{user_id}::analysis"


def search_relevant_analysis_memories(user_id: str, query: str) -> List[MemoryHit]:
    """Search investigation notes (analysis namespace) relevant to the current query."""
    if not query or not query.strip():
        return []

    query_vec = embed_texts([query])[0]
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD
    raw = qdrant_search(_analysis_namespace_user_id(user_id), query_vec, top_k=8)

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


def augment_messages_with_analysis_memories(
    messages: List[ChatMessage],
    memories: List[MemoryHit],
) -> List[ChatMessage]:
    """Inject prior investigation notes as an advisory system message."""
    if not memories:
        return messages

    notes = "\n".join(f"- {m['text']}" for m in memories)
    prefix = ChatMessage(
        role="system",
        content=(
            "The following are PRIOR INVESTIGATION NOTES for the USER.\n"
            "Use these as helpful context, but do not treat them as immutable ground truth.\n\n"
            f"{notes}"
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


def _collect_tool_names(messages: List[ChatMessage]) -> List[str]:
    """Best-effort tool name extraction without depending on a specific provider schema."""
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
                    n = None
                    if isinstance(fn, dict):
                        n = fn.get("name")
                    else:
                        n = getattr(fn, "name", None)
                    n = n or getattr(tc, "name", None)

                if isinstance(n, str) and n.strip():
                    names.append(n.strip())

    # Preserve order, remove duplicates
    seen = set()
    out: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _filter_verbatim_observables(observables: List[str], source_text: str) -> List[str]:
    """Guardrail: keep only observables that appear verbatim in the provided text.

    We keep implementation simple and avoid regex/rulesets.
    """
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

    # De-dupe preserve order
    seen = set()
    final: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        final.append(s)
    return final


def store_investigation_memories(user_id: str, notes: List[str]) -> int:
    """Store investigation notes in a separate namespace."""
    cleaned = [normalize_fact(n) for n in notes if isinstance(n, str) and n.strip()]
    if not cleaned:
        return 0

    ns_user_id = _analysis_namespace_user_id(user_id)
    vecs = embed_texts(cleaned)
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD

    to_insert_texts: List[str] = []
    to_insert_vecs: List[List[float]] = []
    skipped = 0

    for text, vec in zip(cleaned, vecs):
        existing = qdrant_search(ns_user_id, vec, top_k=1)
        if existing and float(existing[0].get("score", 0.0) or 0.0) >= threshold:
            skipped += 1
            continue
        to_insert_texts.append(text)
        to_insert_vecs.append(vec)

    if not to_insert_texts:
        _log_memory("analysis.store.none_deduped", user_id=user_id, skipped=skipped)
        return 0

    qdrant_upsert(ns_user_id, to_insert_texts, to_insert_vecs)
    _log_memory("analysis.store.upsert", user_id=user_id, inserted=len(to_insert_texts), skipped=skipped)
    return len(to_insert_texts)


async def _distill_investigation_memories(
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
    tool_names: List[str],
) -> List[str]:
    """Distill recall-friendly investigation notes from the user question + assistant answer.

    - No manual regex rules.
    - No confidence ratings.
    - Guardrail: extracted observables must appear verbatim in provided text.

    Returns a list of note texts to store in the analysis namespace.
    """
    if not assistant_reply or not assistant_reply.strip():
        return []

    # Use the last user message as the "question".
    last_user = ""
    for m in reversed(messages):
        if getattr(m, "role", None) == "user" and getattr(m, "content", None):
            last_user = (m.content or "").strip()
            break

    source_text = f"USER: {last_user}\nASSISTANT: {assistant_reply}".strip()

    chosen_model = settings.MEMORY_MODEL_NAME or model_name
    tools_str = ", ".join(tool_names) if tool_names else "none"

    # Hard caps to keep stored analysis notes compact.
    MAX_NOTE_CHARS = 1200
    MAX_OBSERVABLES = 12
    MAX_TAGS = 8

    convo = [
        ChatMessage(
            role="system",
            content=(
                "You are a memory distiller for cybersecurity investigations.\n"
                "Given the USER question, the ASSISTANT final answer, and the tool names used, decide whether to store durable investigation notes for future recall.\n\n"
                "Constraints:\n"
                "- Do NOT store personal profile facts about the user (preferences, identity, habits).\n"
                "- Prefer durable investigation takeaways: key findings, hypotheses, procedures, IOCs/observables mentioned.\n"
                "- Keep memory texts concise (<= 300 characters each).\n"
                "- Include at most 12 observables and 8 tags per memory.\n"
                "- If nothing is worth remembering, set store=false and return an empty list.\n"
                "- Observables MUST be copied exactly from the provided text (verbatim substrings).\n"
                "- Return ONLY valid JSON (no markdown, no prose).\n\n"
                "JSON format:\n"
                "{\n"
                '  \"store\": true|false,\n'
                '  \"memories\": [\n'
                "    {\n"
                '      \"text\": string,\n'
                '      \"observables\": [string, ...],\n'
                '      \"tags\": [string, ...]\n'
                "    }, ...\n"
                "  ]\n"
                "}\n"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"TOOLS USED: {tools_str}\n\n"
                f"{source_text}"
            ),
        ),
    ]

    resp = await chat_upstream(
        ChatRequest(
            model=chosen_model,
            messages=convo,
            temperature=0.0,
        )
    )

    if not isinstance(resp, dict) or "choices" not in resp or "error" in resp:
        _log_memory("analysis.skip.upstream_error", user_id=user_id, model=chosen_model)
        logger.error(f"Investigation memory distiller upstream error/shape: {resp}")
        return []

    msg = resp["choices"][0]["message"]
    content = (msg.get("content") or "").strip()

    if not content:
        rc = (msg.get("reasoning_content") or "").strip()
        if rc:
            logger.warning(
                "Investigation distiller returned empty message.content; attempting JSON parse from reasoning_content fallback."
            )
            content = rc

    obj = _extract_json_object(content)
    if obj is None:
        _log_memory("analysis.skip.parse_failed", user_id=user_id, model=chosen_model, content_len=len(content))
        logger.error(
            "Failed to parse investigation distiller response as JSON object. "
            f"Raw startswith={content[:160]!r}"
        )
        return []

    store = bool(obj.get("store", False))
    if not store:
        _log_memory("analysis.skip.store_false", user_id=user_id, model=chosen_model)
        return []

    memories = obj.get("memories", [])
    if not isinstance(memories, list):
        _log_memory("analysis.skip.bad_shape", user_id=user_id, model=chosen_model)
        return []

    out_texts: List[str] = []
    for m in memories[:3]:
        if not isinstance(m, dict):
            continue
        text = m.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        observables = m.get("observables", [])
        if not isinstance(observables, list):
            observables = []
        observables = _filter_verbatim_observables(observables, source_text)

        tags = m.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()][:MAX_TAGS]

        parts = [f"Tools: {tools_str}", text.strip()]
        if observables:
            obs_joined = ", ".join(observables[:MAX_OBSERVABLES])
            parts.append(f"Observables: {obs_joined}")
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        note = "\n".join(parts)
        if len(note) > MAX_NOTE_CHARS:
            note = note[: MAX_NOTE_CHARS - 1].rstrip() + "…"

        out_texts.append(note)

    _log_memory("analysis.distilled", user_id=user_id, model=chosen_model, n_notes=len(out_texts), n_tools=len(tool_names))
    return out_texts


async def extract_and_store_memory(
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
) -> bool:
    """
    Returns True only when any memories were extracted AND stored.

    Notes:
      - USER-facts extraction is confidence-gated.
      - Investigation-note distillation is separate and may still store notes even when USER-facts are skipped.
    """
    if not settings.ENABLE_MEMORY_EXTRACTION:
        _log_memory("skip.disabled", user_id=user_id)
        return False

    stored_any = False

    # Investigation-note distillation is independent of USER-facts extraction.
    try:
        tool_names = _collect_tool_names(messages)
        notes = await _distill_investigation_memories(
            user_id=user_id,
            messages=messages,
            assistant_reply=assistant_reply,
            model_name=model_name,
            tool_names=tool_names,
        )
        if notes:
            inserted_notes = store_investigation_memories(user_id, notes)
            if inserted_notes > 0:
                stored_any = True
                _log_memory("analysis.store.ok", user_id=user_id, inserted=inserted_notes)
            else:
                _log_memory("analysis.store.deduped", user_id=user_id)
        else:
            _log_memory("analysis.skip.no_notes", user_id=user_id, n_tools=len(tool_names))
    except Exception as e:
        logger.error(f"Failed to distill/store investigation memories: {e}")

    # Only use USER utterances to avoid extracting assistant hallucinations as "user facts".
    transcript = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in messages
        if m.role == "user" and m.content
    ).strip()

    n_user_msgs = sum(1 for m in messages if m.role == "user" and m.content)
    transcript_len = len(transcript)
    _log_memory("extract.start", user_id=user_id, n_user_msgs=n_user_msgs, transcript_len=transcript_len)

    if not transcript:
        _log_memory("skip.empty_user_transcript", user_id=user_id, n_user_msgs=n_user_msgs)
        return stored_any

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
                "Extract stable, useful facts stated by the USER (from USER messages only).\n\n"
                "You must provide verbatim evidence for each item: the evidence must be an exact substring from the USER transcript.\n\n"
                "Categories:\n"
                "- profile: durable personal preferences/identity/long-term habits\n"
                "- workspace: current projects/tools/work context (useful, but time-varying)\n\n"
                "Rules:\n"
                "- Only include facts that are explicitly supported by the USER transcript (no inference).\n"
                "- Write items in third person.\n"
                "- Do NOT include assistant opinions or responses.\n"
                "- Keep items short (<= 120 chars each).\n"
                "- Respond ONLY as valid JSON (no markdown, no prose).\n\n"
                "JSON format:\n"
                "{\n"
                '  \"confidence\": float between 0 and 1,\n'
                '  \"items\": [\n'
                "    {\n"
                '      \"text\": string,\n'
                '      \"category\": \"profile\"|\"workspace\",\n'
                '      \"evidence\": string\n'
                "    }, ...\n"
                "  ]\n"
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
        return stored_any
    if "error" in resp:
        logger.error(f"Memory extractor upstream error: {resp}")
        return stored_any
    if "choices" not in resp:
        logger.error(f"Memory extractor missing 'choices': {resp}")
        return stored_any

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
            _log_memory(
                "skip.parse_failed",
                user_id=user_id,
                model=chosen_model,
                content_len=len(content),
            )
            logger.error(
                "Failed to parse memory extractor response as JSON object. "
                f"Raw startswith={content[:160]!r}"
            )
            return stored_any

        conf = float(obj.get("confidence", 0.0) or 0.0)

        items = obj.get("items")
        # Back-compat: tolerate older schema "facts": [string, ...]
        if items is None:
            facts = obj.get("facts", [])
            if not isinstance(facts, list):
                logger.error(f"Memory extractor 'facts' is not a list: {facts!r}")
                return stored_any
            items = [{"text": f, "category": "profile", "evidence": f} for f in facts]

        if not isinstance(items, list):
            logger.error(f"Memory extractor 'items' is not a list: {items!r}")
            return stored_any

        profile_texts: List[str] = []
        workspace_texts: List[str] = []

        kept = 0
        dropped_no_evidence = 0
        dropped_bad_shape = 0

        for it in items[:12]:
            if not isinstance(it, dict):
                dropped_bad_shape += 1
                continue
            text = it.get("text")
            cat = (it.get("category") or "profile")
            evidence = it.get("evidence")

            if not isinstance(text, str) or not text.strip():
                dropped_bad_shape += 1
                continue
            if not isinstance(evidence, str) or not evidence:
                dropped_no_evidence += 1
                continue
            if not _evidence_is_verbatim(evidence, transcript):
                dropped_no_evidence += 1
                continue

            kept += 1
            cat_l = str(cat).strip().lower()
            if cat_l == "workspace":
                workspace_texts.append(text.strip())
            else:
                profile_texts.append(text.strip())

        _log_memory(
            "extract.parsed",
            user_id=user_id,
            model=chosen_model,
            confidence=conf,
            n_items=len(items),
            kept=kept,
            n_profile=len(profile_texts),
            n_workspace=len(workspace_texts),
            dropped_no_evidence=dropped_no_evidence,
            dropped_bad_shape=dropped_bad_shape,
        )
    except Exception as e:
        logger.error(
            f"Failed to parse memory extractor response. Raw choices: {resp.get('choices')}. Error: {str(e)}"
        )
        return stored_any

    if conf < settings.MEMORY_MIN_CONFIDENCE:
        _log_memory(
            "skip.low_confidence",
            user_id=user_id,
            model=chosen_model,
            confidence=conf,
            min_confidence=settings.MEMORY_MIN_CONFIDENCE,
        )
        return stored_any

    if not profile_texts and not workspace_texts:
        _log_memory(
            "skip.no_facts",
            user_id=user_id,
            model=chosen_model,
            confidence=conf,
        )
        return stored_any

    inserted_profile = store_facts(user_id, profile_texts)
    inserted_workspace = store_workspace_facts(user_id, workspace_texts)

    if inserted_profile > 0:
        _log_memory("store.ok", user_id=user_id, inserted=inserted_profile)
        stored_any = True
    elif profile_texts:
        _log_memory("skip.deduped", user_id=user_id, inserted=inserted_profile)

    if inserted_workspace > 0:
        _log_memory("workspace.store.ok", user_id=user_id, inserted=inserted_workspace)
        stored_any = True
    elif workspace_texts:
        _log_memory("workspace.skip.deduped", user_id=user_id, inserted=inserted_workspace)

    return stored_any


def store_facts(user_id: str, facts: List[str]) -> int:
    cleaned = [normalize_fact(f) for f in facts if isinstance(f, str) and f.strip()]
    if not cleaned:
        return 0

    vecs = embed_texts(cleaned)
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD

    to_insert_texts: List[str] = []
    to_insert_vecs: List[List[float]] = []
    skipped = 0

    for text, vec in zip(cleaned, vecs):
        existing = qdrant_search(user_id, vec, top_k=1)
        if existing and float(existing[0].get("score", 0.0) or 0.0) >= threshold:
            skipped += 1
            continue
        to_insert_texts.append(text)
        to_insert_vecs.append(vec)

    if not to_insert_texts:
        _log_memory("store.none_deduped", user_id=user_id, skipped=skipped)
        return 0

    qdrant_upsert(user_id, to_insert_texts, to_insert_vecs)
    _log_memory("store.upsert", user_id=user_id, inserted=len(to_insert_texts), skipped=skipped)
    return len(to_insert_texts)

def store_workspace_facts(user_id: str, facts: List[str]) -> int:
    cleaned = [normalize_fact(f) for f in facts if isinstance(f, str) and f.strip()]
    if not cleaned:
        return 0

    ns_user_id = _workspace_namespace_user_id(user_id)
    vecs = embed_texts(cleaned)
    threshold = settings.MEMORY_SIMILARITY_THRESHOLD

    to_insert_texts: List[str] = []
    to_insert_vecs: List[List[float]] = []
    skipped = 0

    for text, vec in zip(cleaned, vecs):
        existing = qdrant_search(ns_user_id, vec, top_k=1)
        if existing and float(existing[0].get("score", 0.0) or 0.0) >= threshold:
            skipped += 1
            continue
        to_insert_texts.append(text)
        to_insert_vecs.append(vec)

    if not to_insert_texts:
        _log_memory("workspace.store.none_deduped", user_id=user_id, skipped=skipped)
        return 0

    qdrant_upsert(ns_user_id, to_insert_texts, to_insert_vecs)
    _log_memory("workspace.store.upsert", user_id=user_id, inserted=len(to_insert_texts), skipped=skipped)
    return len(to_insert_texts)
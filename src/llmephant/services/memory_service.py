import json
import re
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


# --- Added helpers for memory request knob plumbing ---
def _filter_chatrequest_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to keys supported by ChatRequest (Pydantic v1/v2 safe)."""
    fields = getattr(ChatRequest, "model_fields", None) or getattr(
        ChatRequest, "__fields__", None
    )
    allowed = set(fields.keys()) if isinstance(fields, dict) else None
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if allowed is not None and k not in allowed:
            continue
        out[k] = v
    return out


def _memory_request_knobs(pass_name: str) -> Dict[str, Any]:
    """Return backend-agnostic knobs for a given memory pass.

    We keep Python "dumb": only numeric/config knobs live here. Any semantic filtering lives in prompts.

    pass_name: one of {"extract", "distill", "verify"}.
    """

    max_tokens: Optional[int]
    if pass_name == "extract":
        max_tokens = getattr(settings, "MEMORY_EXTRACT_MAX_TOKENS", None)
    elif pass_name == "distill":
        max_tokens = getattr(settings, "MEMORY_DISTILL_MAX_TOKENS", None)
    elif pass_name == "verify":
        # verifier is intentionally cheap; settings provides a default
        max_tokens = getattr(settings, "MEMORY_VERIFY_MAX_TOKENS", 300)
    else:
        max_tokens = None

    effort = getattr(settings, "MEMORY_REASONING_EFFORT", None)
    # Different backends expose this differently; include both and let ChatRequest filtering decide.
    reasoning_obj = (
        {"effort": effort} if isinstance(effort, str) and effort.strip() else None
    )

    return {
        "max_tokens": max_tokens,
        "reasoning_effort": effort,
        "reasoning": reasoning_obj,
    }


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
        fact = last_msg[len(prefix) :].strip()
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


class UserMemoryVerifyResult(TypedDict, total=False):
    items: List[UserMemoryItem]


async def _verify_user_memory_items(
    *,
    user_id: str,
    transcript: str,
    candidates: List[UserMemoryItem],
    model_name: str,
    req_id: Optional[str] = None,
) -> List[UserMemoryItem]:
    """Second-pass verifier to filter low-quality/incorrect USER facts without encoding English rules in code.

    The verifier decides which items should be stored. Python enforces only structural checks (shape + verbatim evidence).
    """

    if not transcript or not transcript.strip() or not candidates:
        return []

    # Keep the verifier cheap.
    MAX_VERIFY_ITEMS = 12

    cand = candidates[:MAX_VERIFY_ITEMS]

    _log_memory(
        "verify.start",
        user_id=user_id,
        model=model_name,
        n_candidates=len(cand),
        req_id=req_id,
    )

    convo = [
        ChatMessage(
            role="system",
            content=(
                "You are a memory verification assistant.\n"
                "You will be given a USER transcript (USER messages only) and a list of candidate memory items.\n"
                "Your job is to return ONLY the items that should be STORED as durable memories.\n\n"
                "Rules:\n"
                "- Only keep items that are explicitly supported by the transcript evidence.\n"
                "- Drop instructions/requests/tasks/to-dos, even if rewritten as a statement (e.g., 'User requires a narrative summary...').\n"
                "- Drop ephemeral or one-off statements unless they are useful as short-lived workspace context.\n"
                "- profile: only long-term durable personal facts/preferences explicitly stated by the user.\n"
                "- workspace: current project/tools/work context that will likely remain useful for at least days/weeks.\n"
                "- Do NOT store 'how to respond' preferences unless the user explicitly states it as a durable preference.\n"
                "- Evidence MUST be a verbatim substring from the transcript.\n"
                "- Return ONLY valid JSON (no markdown, no prose).\n\n"
                "JSON format:\n"
                "{\n"
                '  "items": [\n'
                "    {\n"
                '      "text": string,\n'
                '      "category": "profile"|"workspace",\n'
                '      "evidence": string\n'
                "    }, ...\n"
                "  ]\n"
                "}"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                "TRANSCRIPT:\n"
                + transcript
                + "\n\n"
                + "CANDIDATES (JSON):\n"
                + json.dumps(cand, ensure_ascii=False)
            ),
        ),
    ]

    knobs = _memory_request_knobs("verify")

    req_kwargs = _filter_chatrequest_kwargs(
        {
            "model": model_name,
            "messages": convo,
            "temperature": 0.0,
            **knobs,
        }
    )

    resp = await chat_upstream(ChatRequest(**req_kwargs), req_id=req_id)

    finish_reason = _get_finish_reason(resp)

    if not isinstance(resp, dict) or "choices" not in resp or "error" in resp:
        _log_memory(
            "verify.skip.upstream_error",
            user_id=user_id,
            model=model_name,
            finish_reason=finish_reason,
            req_id=req_id,
        )
        logger.error(f"User memory verifier upstream error/shape: {resp}")
        return []

    msg = resp["choices"][0]["message"]
    content = (msg.get("content") or "").strip()

    if not content:
        rc = (msg.get("reasoning_content") or "").strip()
        if rc:
            logger.warning(
                "User memory verifier returned empty message.content; attempting JSON parse from reasoning_content fallback."
            )
            content = rc

    if finish_reason == "length":
        _log_memory(
            "verify.warn.truncated",
            user_id=user_id,
            model=model_name,
            content_len=len(content),
            req_id=req_id,
        )

    obj = _extract_json_object(content)
    if obj is None:
        _log_memory(
            "verify.skip.parse_failed",
            finish_reason=finish_reason,
            user_id=user_id,
            model=model_name,
            content_len=len(content),
            req_id=req_id,
        )
        parse_err: Optional[str] = None
        try:
            json.loads(_strip_code_fences(content))
        except Exception as e:
            parse_err = repr(e)

        log_msg = (
            "Failed to parse user memory verifier response as JSON object. "
            + f"finish_reason={finish_reason!r} "
            + (f"err={parse_err} " if parse_err else "")
            + f"head={content[:160]!r} "
            + f"tail={content[-240:]!r}"
        )
        logger.error(log_msg)
        return []

    items = obj.get("items", [])
    if not isinstance(items, list):
        _log_memory(
            "verify.skip.bad_shape", user_id=user_id, model=model_name, req_id=req_id
        )
        return []

    kept: List[UserMemoryItem] = []
    dropped_bad_shape = 0
    dropped_no_evidence = 0

    for it in items[:MAX_VERIFY_ITEMS]:
        if not isinstance(it, dict):
            dropped_bad_shape += 1
            continue
        text = it.get("text")
        cat = it.get("category")
        evidence = it.get("evidence")

        if not isinstance(text, str) or not text.strip():
            dropped_bad_shape += 1
            continue
        if not isinstance(cat, str):
            dropped_bad_shape += 1
            continue
        cat_l = cat.strip().lower()
        if cat_l not in ("profile", "workspace"):
            dropped_bad_shape += 1
            continue

        if not isinstance(evidence, str) or not evidence:
            dropped_no_evidence += 1
            continue
        if not _evidence_is_verbatim(evidence, transcript):
            dropped_no_evidence += 1
            continue

        kept.append({"text": text.strip(), "category": cat_l, "evidence": evidence})

    _log_memory(
        "verify.parsed",
        finish_reason=finish_reason,
        user_id=user_id,
        model=model_name,
        n_in=len(cand),
        n_out=len(kept),
        dropped_bad_shape=dropped_bad_shape,
        dropped_no_evidence=dropped_no_evidence,
        req_id=req_id,
    )

    return kept


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


def _get_finish_reason(resp: Any) -> Optional[str]:
    """Best-effort extraction of choice.finish_reason from an upstream response."""
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    if isinstance(c0, dict):
        fr = c0.get("finish_reason")
        return fr if isinstance(fr, str) else None
    return None


def _repair_json_common(s: str) -> str:
    """Best-effort repair for common LLM JSON mistakes.

    Repairs (string/escape-aware):
      - Trailing commas before `}` or `]`.
      - Normalizes unicode line separators that can break JSON parsing.
      - Strips UTF-8 BOM if present.

    This is intentionally conservative: if no obvious repair is needed, returns input unchanged.
    """
    if not s:
        return s

    # Strip BOM
    s = s.lstrip("\ufeff")

    # Normalize line separators
    s = s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")

    # Mask out characters inside JSON strings so we don't remove commas inside strings.
    in_string = False
    escape = False
    mask_chars: List[str] = []

    for ch in s:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            mask_chars.append("X")
            continue

        if ch == '"':
            in_string = True
            mask_chars.append("X")
            continue

        mask_chars.append(ch)

    masked = "".join(mask_chars)

    # Remove commas followed by only whitespace and then a closing brace/bracket.
    comma_positions = [m.start() for m in re.finditer(r",(?=\s*[}\]])", masked)]
    if not comma_positions:
        return s

    chars = list(s)
    for idx in reversed(comma_positions):
        chars[idx] = ""
    return "".join(chars)


def _extract_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON object from `text`.

    Best-effort extraction that is robust to:
      - raw JSON object
      - JSON wrapped in code fences
      - extra prose before/after

    Uses a balanced-brace scanner that is string/escape aware (so braces inside strings
    do not affect nesting depth).
    """
    if not text:
        return None

    candidate = _strip_code_fences(text)

    # Fast path: whole-string JSON
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        repaired = _repair_json_common(candidate)
        if repaired != candidate:
            try:
                obj = json.loads(repaired)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

    s = candidate

    start: Optional[int] = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        # Not currently in a string
        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
            continue

        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                chunk = s[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    repaired = _repair_json_common(chunk)
                    if repaired != chunk:
                        try:
                            obj = json.loads(repaired)
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            pass
                    # Keep scanning: there may be another object later.
                    start = None
                    continue

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
    _log_memory(
        "analysis.store.upsert",
        user_id=user_id,
        inserted=len(to_insert_texts),
        skipped=skipped,
    )
    return len(to_insert_texts)


async def _distill_investigation_memories(
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
    tool_names: List[str],
    req_id: Optional[str] = None,
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
                '- If you cannot fit valid JSON, return {"store": false, "memories": []}.\n'
                "- Return ONLY valid JSON (no markdown, no prose).\n\n"
                "JSON format:\n"
                "{\n"
                '  "store": true|false,\n'
                '  "memories": [\n'
                "    {\n"
                '      "text": string,\n'
                '      "observables": [string, ...],\n'
                '      "tags": [string, ...]\n'
                "    }, ...\n"
                "  ]\n"
                "}\n"
            ),
        ),
        ChatMessage(
            role="user",
            content=(f"TOOLS USED: {tools_str}\n\n{source_text}"),
        ),
    ]

    knobs = _memory_request_knobs("distill")
    req_kwargs = _filter_chatrequest_kwargs(
        {
            "model": chosen_model,
            "messages": convo,
            "temperature": 0.0,
            **knobs,
        }
    )

    resp = await chat_upstream(ChatRequest(**req_kwargs), req_id=req_id)
    finish_reason = _get_finish_reason(resp)

    if not isinstance(resp, dict) or "choices" not in resp or "error" in resp:
        _log_memory(
            "analysis.skip.upstream_error",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            req_id=req_id,
        )
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

    if finish_reason == "length":
        _log_memory(
            "analysis.warn.truncated",
            user_id=user_id,
            model=chosen_model,
            content_len=len(content),
            req_id=req_id,
        )

    obj = _extract_json_object(content)
    if obj is None:
        _log_memory(
            "analysis.skip.parse_failed",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            content_len=len(content),
            req_id=req_id,
        )
        parse_err: Optional[str] = None
        try:
            json.loads(_strip_code_fences(content))
        except Exception as e:
            parse_err = repr(e)

        msg_prefix = (
            "Failed to parse investigation distiller response as JSON object due to truncation. "
            if finish_reason == "length"
            else "Failed to parse investigation distiller response as JSON object. "
        )

        log_msg = (
            msg_prefix
            + f"finish_reason={finish_reason!r} "
            + (f"err={parse_err} " if parse_err else "")
            + f"head={content[:160]!r} "
            + f"tail={content[-240:]!r}"
        )

        if finish_reason == "length":
            logger.warning(log_msg)
        else:
            logger.error(log_msg)
        return []

    store = bool(obj.get("store", False))
    if not store:
        _log_memory(
            "analysis.skip.store_false",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            req_id=req_id,
        )
        return []

    memories = obj.get("memories", [])
    if not isinstance(memories, list):
        _log_memory(
            "analysis.skip.bad_shape",
            user_id=user_id,
            model=chosen_model,
            req_id=req_id,
        )
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

    _log_memory(
        "analysis.distilled",
        finish_reason=finish_reason,
        user_id=user_id,
        model=chosen_model,
        n_notes=len(out_texts),
        n_tools=len(tool_names),
        req_id=req_id,
    )
    return out_texts


async def extract_and_store_memory(
    user_id: str,
    messages: List[ChatMessage],
    assistant_reply: str,
    model_name: str,
    req_id: Optional[str] = None,
) -> bool:
    """
    Returns True only when any memories were extracted AND stored.

    Notes:
      - USER-facts extraction is confidence-gated.
      - Investigation-note distillation is separate and may still store notes even when USER-facts are skipped.
    """
    if not settings.ENABLE_MEMORY_EXTRACTION:
        _log_memory("skip.disabled", user_id=user_id, req_id=req_id)
        return False

    stored_any = False

    # Source-of-truth: log models and knobs for memory distillation/extraction.
    chosen_model = settings.MEMORY_MODEL_NAME or model_name
    if not settings.MEMORY_MODEL_NAME:
        logger.warning(
            "MEMORY_MODEL_NAME is not set; falling back to the chat model for memory extraction. "
            "If your chat model does not reliably output JSON in message.content, set MEMORY_MODEL_NAME."
        )

    extract_knobs = _memory_request_knobs("extract")
    distill_knobs = _memory_request_knobs("distill")
    verify_knobs = _memory_request_knobs("verify")

    _log_memory(
        "models.selected",
        req_id=req_id,
        user_id=user_id,
        distill_model=chosen_model,
        extract_model=chosen_model,
        verify_model=chosen_model,
        temperature=0.0,
        extract_max_tokens=extract_knobs.get("max_tokens"),
        distill_max_tokens=distill_knobs.get("max_tokens"),
        verify_max_tokens=verify_knobs.get("max_tokens"),
        reasoning_effort=extract_knobs.get("reasoning_effort"),
        reasoning=extract_knobs.get("reasoning"),
    )

    knobs = extract_knobs

    # Investigation-note distillation is independent of USER-facts extraction.
    try:
        tool_names = _collect_tool_names(messages)
        notes = await _distill_investigation_memories(
            user_id=user_id,
            messages=messages,
            assistant_reply=assistant_reply,
            model_name=model_name,
            tool_names=tool_names,
            req_id=req_id,
        )
        if notes:
            inserted_notes = store_investigation_memories(user_id, notes)
            if inserted_notes > 0:
                stored_any = True
                _log_memory(
                    "analysis.store.ok",
                    user_id=user_id,
                    inserted=inserted_notes,
                    req_id=req_id,
                )
            else:
                _log_memory("analysis.store.deduped", user_id=user_id, req_id=req_id)
        else:
            _log_memory(
                "analysis.skip.no_notes",
                user_id=user_id,
                n_tools=len(tool_names),
                req_id=req_id,
            )
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
    _log_memory(
        "extract.start",
        user_id=user_id,
        n_user_msgs=n_user_msgs,
        transcript_len=transcript_len,
        req_id=req_id,
    )

    if not transcript:
        _log_memory(
            "skip.empty_user_transcript",
            user_id=user_id,
            n_user_msgs=n_user_msgs,
            req_id=req_id,
        )
        return stored_any

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
                "- Only include facts explicitly supported by the USER transcript (no inference).\n"
                "- NEVER store instructions, requests, tasks, or to-do items.\n"
                "  - If the user is telling the assistant to DO something, that is not a memory.\n"
                "- NEVER store 'how to respond' preferences unless the user states it as a durable preference (e.g., 'I prefer X in general').\n"
                "- Profile is extremely strict: only durable facts likely true in 30+ days (identity/role/preferences explicitly stated).\n"
                "- If a fact is situational (testing, this run, right now), classify as workspace or drop it.\n"
                "- Write items in third person, as declarative statements (not commands).\n"
                "- Keep items short (<= 140 chars each).\n"
                "- Evidence MUST be a verbatim substring from USER transcript.\n"
                "- Respond ONLY as valid JSON (no markdown, no prose).\n"
                "\n"
                "Examples (DO NOT STORE):\n"
                '- "Re-run the last tool" (task)\n'
                '- "Generate a narrative summary" (task)\n'
                '- "Redo that last analysis but..." (task)\n'
                "\n"
                "Examples (OK to store as workspace):\n"
                '- "User is testing a memory service locally in an isolated environment."\n'
                "JSON format:\n"
                "{\n"
                '  "confidence": float between 0 and 1,\n'
                '  "items": [\n'
                "    {\n"
                '      "text": string,\n'
                '      "category": "profile"|"workspace",\n'
                '      "evidence": string\n'
                "    }, ...\n"
                "  ]\n"
                "}"
            ),
        ),
        ChatMessage(role="user", content=transcript),
    ]

    req_kwargs = _filter_chatrequest_kwargs(
        {
            "model": chosen_model,
            "messages": convo,
            "temperature": 0.0,
            **knobs,
        }
    )

    resp = await chat_upstream(ChatRequest(**req_kwargs), req_id=req_id)
    finish_reason = _get_finish_reason(resp)

    if not isinstance(resp, dict):
        _log_memory(
            "extract.skip.upstream_error",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            reason="non_dict_response",
            req_id=req_id,
        )
        logger.error(f"Memory extractor returned non-dict response: {resp}")
        return stored_any
    if "error" in resp:
        _log_memory(
            "extract.skip.upstream_error",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            reason="upstream_error",
            req_id=req_id,
        )
        logger.error(f"Memory extractor upstream error: {resp}")
        return stored_any
    if "choices" not in resp:
        _log_memory(
            "extract.skip.upstream_error",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            reason="missing_choices",
            req_id=req_id,
        )
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

        if finish_reason == "length":
            _log_memory(
                "extract.warn.truncated",
                user_id=user_id,
                model=chosen_model,
                content_len=len(content),
                req_id=req_id,
            )

        obj = _extract_json_object(content)
        if obj is None:
            _log_memory(
                "extract.skip.parse_failed",
                finish_reason=finish_reason,
                user_id=user_id,
                model=chosen_model,
                content_len=len(content),
                req_id=req_id,
            )
            parse_err: Optional[str] = None
            try:
                json.loads(_strip_code_fences(content))
            except Exception as e:
                parse_err = repr(e)

            msg_prefix = (
                "Failed to parse memory extractor response as JSON object due to truncation. "
                if finish_reason == "length"
                else "Failed to parse memory extractor response as JSON object. "
            )

            log_msg = (
                msg_prefix
                + f"finish_reason={finish_reason!r} "
                + (f"err={parse_err} " if parse_err else "")
                + f"head={content[:160]!r} "
                + f"tail={content[-240:]!r}"
            )

            if finish_reason == "length":
                logger.warning(log_msg)
            else:
                logger.error(log_msg)
            return stored_any

        conf = float(obj.get("confidence", 0.0) or 0.0)

        items = obj.get("items")
        # Back-compat: tolerate older schema "facts": [string, ...]
        if items is None:
            facts = obj.get("facts", [])
            if not isinstance(facts, list):
                logger.error(f"Memory extractor 'facts' is not a list: {facts!r}")
                return stored_any
            items = [
                {
                    "text": f,
                    "category": "profile",
                    "evidence": f,
                }
                for f in facts
            ]

        if not isinstance(items, list):
            logger.error(f"Memory extractor 'items' is not a list: {items!r}")
            return stored_any

        candidate_items: List[UserMemoryItem] = []

        kept = 0
        dropped_no_evidence = 0
        dropped_bad_shape = 0

        for it in items[:12]:
            if not isinstance(it, dict):
                dropped_bad_shape += 1
                continue
            text = it.get("text")
            cat = it.get("category")
            evidence = it.get("evidence")

            if not isinstance(text, str) or not text.strip():
                dropped_bad_shape += 1
                continue

            if not isinstance(cat, str):
                dropped_bad_shape += 1
                continue
            cat_l = cat.strip().lower()
            if cat_l not in ("profile", "workspace"):
                dropped_bad_shape += 1
                continue

            if not isinstance(evidence, str) or not evidence:
                dropped_no_evidence += 1
                continue
            if not _evidence_is_verbatim(evidence, transcript):
                dropped_no_evidence += 1
                continue

            kept += 1
            candidate_items.append(
                {"text": text.strip(), "category": cat_l, "evidence": evidence}
            )

        _log_memory(
            "extract.parsed",
            finish_reason=finish_reason,
            user_id=user_id,
            model=chosen_model,
            confidence=conf,
            n_items=len(items),
            n_candidates=len(candidate_items),
            kept=kept,
            dropped_no_evidence=dropped_no_evidence,
            dropped_bad_shape=dropped_bad_shape,
            req_id=req_id,
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
            req_id=req_id,
        )
        return stored_any

    if not candidate_items:
        _log_memory(
            "skip.no_facts",
            user_id=user_id,
            model=chosen_model,
            confidence=conf,
            req_id=req_id,
        )
        return stored_any

    verified_items = await _verify_user_memory_items(
        user_id=user_id,
        transcript=transcript,
        candidates=candidate_items,
        model_name=chosen_model,
        req_id=req_id,
    )

    if not verified_items:
        _log_memory(
            "skip.no_verified_facts",
            user_id=user_id,
            model=chosen_model,
            confidence=conf,
            req_id=req_id,
        )
        return stored_any

    profile_texts = [
        i["text"] for i in verified_items if i.get("category") == "profile"
    ]
    workspace_texts = [
        i["text"] for i in verified_items if i.get("category") == "workspace"
    ]

    inserted_profile = store_facts(user_id, profile_texts)
    inserted_workspace = store_workspace_facts(user_id, workspace_texts)

    if inserted_profile > 0:
        _log_memory(
            "store.ok", user_id=user_id, inserted=inserted_profile, req_id=req_id
        )
        stored_any = True
    elif profile_texts:
        _log_memory(
            "skip.deduped", user_id=user_id, inserted=inserted_profile, req_id=req_id
        )

    if inserted_workspace > 0:
        _log_memory(
            "workspace.store.ok",
            user_id=user_id,
            inserted=inserted_workspace,
            req_id=req_id,
        )
        stored_any = True
    elif workspace_texts:
        _log_memory(
            "workspace.skip.deduped",
            user_id=user_id,
            inserted=inserted_workspace,
            req_id=req_id,
        )

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
    _log_memory(
        "store.upsert", user_id=user_id, inserted=len(to_insert_texts), skipped=skipped
    )
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
    _log_memory(
        "workspace.store.upsert",
        user_id=user_id,
        inserted=len(to_insert_texts),
        skipped=skipped,
    )
    return len(to_insert_texts)

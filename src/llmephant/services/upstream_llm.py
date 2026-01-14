import httpx
import json
import time
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from llmephant.core.settings import settings
from llmephant.models.chat_model import ChatRequest
from llmephant.core.logger import setup_logger

logger = setup_logger(__name__)

UPSTREAM_OPENAI_BASE = settings.UPSTREAM_OPENAI_BASE.rstrip("/")


def _safe_payload_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact, non-content preview of an OpenAI-style chat payload.

    Intentionally avoids logging full messages/prompts.
    """
    messages = payload.get("messages") or []
    n_messages = len(messages) if isinstance(messages, list) else 0

    # Count roles without recording content.
    role_counts: Dict[str, int] = {}
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict):
                role = str(m.get("role") or "unknown")
            else:
                role = str(getattr(m, "role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1

    tools = payload.get("tools")
    n_tools = len(tools) if isinstance(tools, list) else 0

    def g(key: str) -> Optional[Any]:
        return payload.get(key)

    preview: Dict[str, Any] = {
        "model": g("model"),
        "stream": bool(g("stream")) if g("stream") is not None else None,
        "temperature": g("temperature"),
        "max_tokens": g("max_tokens"),
        "max_completion_tokens": g("max_completion_tokens"),
        "top_p": g("top_p"),
        "presence_penalty": g("presence_penalty"),
        "frequency_penalty": g("frequency_penalty"),
        "stop": g("stop"),
        "response_format": g("response_format"),
        "reasoning": g("reasoning"),
        "reasoning_effort": g("reasoning_effort"),
        "stream_options": (g("stream_options") or None),
        "tool_choice": g("tool_choice"),
        "n_tools": n_tools,
        "n_messages": n_messages,
        "role_counts": role_counts or None,
    }

    # Sanitize stream_options: only keep include_usage if present.
    so = preview.get("stream_options")
    if isinstance(so, dict):
        preview["stream_options"] = {"include_usage": so.get("include_usage")} if "include_usage" in so else None

    # Drop Nones to keep logs compact.
    return {k: v for k, v in preview.items() if v is not None}


def _safe_usage_preview(usage: Any) -> Optional[Dict[str, Any]]:
    """Return a compact preview of usage stats if present."""
    if not isinstance(usage, dict):
        return None
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
    )
    out = {k: usage.get(k) for k in keys if usage.get(k) is not None}
    return out or None


def _log_upstream_payload(prefix: str, payload: Dict[str, Any], req_id: Optional[str] = None) -> None:
    """Log a consistent, safe payload preview."""
    try:
        preview = _safe_payload_preview(payload)
        if req_id:
            preview = {"req_id": req_id, **preview}
        logger.info("%s %s", prefix, json.dumps(preview, ensure_ascii=False, sort_keys=True))
    except Exception as e:
        logger.warning("%s <failed to build payload preview: %s>", prefix, e)


async def chat_upstream(req: ChatRequest, *, req_id: Optional[str] = None):
    """Non-streaming upstream call.

    Accept a ChatRequest Pydantic model and safely convert it into a JSON-serializable
    payload for the upstream LLM.
    """
    payload = req.to_upstream_payload()
    _log_upstream_payload("Upstream payload prepared", payload, req_id=req_id)

    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=60.0) as client:
        try:
            logger.info("POST %s/chat/completions", UPSTREAM_OPENAI_BASE)
            t0 = time.perf_counter()
            resp = await client.post("/chat/completions", json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            data = resp.json()
            finish_reason = None
            try:
                choices = data.get("choices") or []
                if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
            except Exception:
                finish_reason = None

            usage_preview = _safe_usage_preview(data.get("usage"))
            logger.info(
                "Upstream LLM responded successfully req_id=%s status=%s elapsed_ms=%.1f finish_reason=%s usage=%s",
                req_id,
                resp.status_code,
                elapsed_ms,
                finish_reason,
                usage_preview,
            )
            return data
        except httpx.HTTPStatusError as e:
            try:
                snippet = (e.response.text or "")[:800]
            except Exception:
                snippet = "<unavailable>"
            logger.error("Upstream returned HTTP error status=%s body_snippet=%r", e.response.status_code, snippet)
            raise
        except Exception as e:
            logger.error("Unexpected upstream error type=%s err=%s", type(e).__name__, e)
            raise


async def chat_upstream_stream(req: ChatRequest, *, req_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
    """Stream an OpenAI-style SSE response and yield parsed JSON frames.

    This is intentionally lightweight (close to the original behavior), but includes
    diagnostic counters so we can debug "silent" completions.
    """
    payload = req.to_upstream_payload()
    _log_upstream_payload("Streaming upstream payload prepared", payload, req_id=req_id)

    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=60.0) as client:
        lines_seen = 0
        data_lines_seen = 0
        parsed_frames = 0
        last_finish_reason: Optional[str] = None
        last_usage: Optional[Dict[str, Any]] = None

        try:
            t0 = time.perf_counter()
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                logger.info(
                    "Upstream stream opened req_id=%s status=%s url=%s",
                    req_id,
                    resp.status_code,
                    resp.request.url,
                )

                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    # Best-effort: read a small snippet for logs.
                    try:
                        body = await resp.aread()
                        snippet = body[:500].decode("utf-8", errors="replace")
                    except Exception:
                        snippet = "<unavailable>"
                    logger.error(
                        f"Upstream returned HTTP error {e.response.status_code}: {snippet}"
                    )
                    raise

                async for line in resp.aiter_lines():
                    lines_seen += 1
                    if not line:
                        continue

                    stripped = line.strip()

                    # Original behavior: treat OpenAI-style sentinel as end-of-stream.
                    if stripped in ("data: [DONE]", "[DONE]"):
                        break

                    # Original behavior: only parse `data:` lines.
                    if not line.startswith("data:"):
                        continue

                    data_lines_seen += 1
                    data_str = line[len("data:") :].strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        parsed = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to decode upstream SSE JSON line: {data_str[:300]}"
                        )
                        continue

                    parsed_frames += 1
                    try:
                        choices = parsed.get("choices") or []
                        if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                            fr = choices[0].get("finish_reason")
                            if fr:
                                last_finish_reason = fr
                    except Exception:
                        pass

                    up_usage = _safe_usage_preview(parsed.get("usage"))
                    if up_usage:
                        last_usage = up_usage

                    yield parsed

        except Exception as e:
            logger.error(
                "Upstream streaming request failed req_id=%s type=%s err=%s lines_seen=%s data_lines=%s parsed_frames=%s",
                req_id,
                type(e).__name__,
                e,
                lines_seen,
                data_lines_seen,
                parsed_frames,
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 if 't0' in locals() else None
            if elapsed_ms is not None:
                logger.info(
                    "Upstream stream finished req_id=%s lines_seen=%s data_lines=%s parsed_frames=%s elapsed_ms=%.1f finish_reason=%s usage=%s",
                    req_id,
                    lines_seen,
                    data_lines_seen,
                    parsed_frames,
                    elapsed_ms,
                    last_finish_reason,
                    last_usage,
                )
            else:
                logger.info(
                    "Upstream stream finished req_id=%s lines_seen=%s data_lines=%s parsed_frames=%s finish_reason=%s usage=%s",
                    req_id,
                    lines_seen,
                    data_lines_seen,
                    parsed_frames,
                    last_finish_reason,
                    last_usage,
                )
            if parsed_frames == 0:
                logger.warning(
                    "Upstream stream ended without any parsed JSON frames. This usually means SSE framing differs from expectations,"
                    " the upstream returned an empty stream, or the client disconnected early."
                )


async def list_models(*, req_id: Optional[str] = None) -> Dict:
    logger.info("Requesting model list from upstream. req_id=%s", req_id)
    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=20.0) as client:
        resp = await client.get("/models")
        resp.raise_for_status()
        logger.info("Model list retrieved successfully. req_id=%s", req_id)
        return resp.json()


async def check_upstream_health(client=None, *, req_id: Optional[str] = None) -> bool:
    logger.info("Checking upstream LLM health. req_id=%s", req_id)
    async with (client or httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=3)) as c:
        resp = await c.get("/models")
        resp.raise_for_status()
        return resp.status_code == 200

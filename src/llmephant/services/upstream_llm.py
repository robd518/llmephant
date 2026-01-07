import httpx
import json
from typing import Any, AsyncIterator, Dict
from llmephant.core.settings import settings
from llmephant.models.chat_model import ChatRequest
from llmephant.core.logger import setup_logger

logger = setup_logger(__name__)

UPSTREAM_OPENAI_BASE = settings.UPSTREAM_OPENAI_BASE.rstrip("/")


async def chat_upstream(req: ChatRequest):
    """Non-streaming upstream call.

    Accept a ChatRequest Pydantic model and safely convert it into a JSON-serializable
    payload for the upstream LLM.
    """
    payload = req.to_upstream_payload()
    logger.info(f"Upstream payload prepared for model '{req.model}'")

    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=60.0) as client:
        try:
            logger.info(f"POST {UPSTREAM_OPENAI_BASE}/chat/completions")
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            logger.info("Upstream LLM responded successfully.")
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Upstream returned HTTP error {e.response.status_code}: {e.response.text}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected upstream error: {str(e)}")
            raise


async def chat_upstream_stream(req: ChatRequest) -> AsyncIterator[Dict[str, Any]]:
    """Stream an OpenAI-style SSE response and yield parsed JSON frames.

    This is intentionally lightweight (close to the original behavior), but includes
    diagnostic counters so we can debug "silent" completions.
    """
    payload = req.to_upstream_payload()
    logger.info(f"Streaming upstream payload for model '{req.model}'")

    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=60.0) as client:
        lines_seen = 0
        data_lines_seen = 0
        parsed_frames = 0

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                logger.info(
                    f"Upstream stream opened status={resp.status_code} url={resp.request.url}"
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
                    yield parsed

        except Exception:
            logger.exception(
                f"Upstream streaming request failed (lines_seen={lines_seen} data_lines={data_lines_seen} parsed_frames={parsed_frames})"
            )
            raise
        finally:
            logger.info(
                f"Upstream stream finished (lines_seen={lines_seen} data_lines={data_lines_seen} parsed_frames={parsed_frames})"
            )
            if parsed_frames == 0:
                logger.warning(
                    "Upstream stream ended without any parsed JSON frames. This usually means SSE framing differs from expectations,"
                    " the upstream returned an empty stream, or the client disconnected early."
                )


async def list_models() -> Dict:
    logger.info("Requesting model list from upstream.")
    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=20.0) as client:
        resp = await client.get("/models")
        resp.raise_for_status()
        logger.info("Model list retrieved successfully.")
        return resp.json()


async def check_upstream_health(client=None) -> bool:
    logger.info("Checking upstream LLM health.")
    async with (client or httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=3)) as c:
        resp = await c.get("/models")
        resp.raise_for_status()
        return resp.status_code == 200

import httpx
import json
from llmephant.core.settings import settings
from llmephant.models.chat_model import ChatMessage, ChatRequest
from llmephant.core.logger import setup_logger
from typing import Dict

logger = setup_logger(__name__)

UPSTREAM_OPENAI_BASE = settings.UPSTREAM_OPENAI_BASE.rstrip("/")

async def chat_upstream(req: ChatRequest):
    """
    Accept a ChatRequest Pydantic model and safely convert
    it into a JSON-serializable payload for the upstream LLM.
    Supports only non-streaming requests.
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
            logger.error(f"Upstream returned HTTP error {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected upstream error: {str(e)}")
            raise

async def chat_upstream_stream(req: ChatRequest):
    payload = req.to_upstream_payload()
    logger.info(f"Streaming upstream payload for model '{req.model}'")

    async with httpx.AsyncClient(base_url=UPSTREAM_OPENAI_BASE, timeout=60.0) as client:
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue

                # Upstream uses OpenAI-style SSE framing.
                if line.strip() == "data: [DONE]":
                    break

                if line.startswith("data:"):
                    data_str = line[len("data:"):].strip()
                    try:
                        parsed = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode upstream SSE JSON line: {data_str}")
                        continue
                    yield parsed

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
        if resp.status_code == 200:
            return True
        else:
            return False

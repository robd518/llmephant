from __future__ import annotations
from typing import Any, Dict, Optional
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def health() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {"status": "ok", "service": "llmephant-memory-api"}


@router.get("/tools")
async def health_tools(raw_req: Request) -> Dict[str, Any]:
    """Tooling health/diagnostics.

    This endpoint is intentionally lightweight and reads from app.state, so it works
    even in degraded mode.
    """
    state = raw_req.app.state

    tools_enabled = bool(getattr(state, "tools_enabled", False))
    tooling_init_error: Optional[str] = getattr(state, "tooling_init_error", None)

    registry = getattr(state, "registry", None)
    executor = getattr(state, "executor", None)

    # Best-effort counts (do not raise if registry is missing or misconfigured).
    tool_count = 0
    try:
        if registry is not None and hasattr(registry, "openai_tools"):
            tool_count = len(registry.openai_tools())
    except Exception:
        # If registry.openai_tools() explodes, that's useful signal but shouldn't 500 health.
        tool_count = 0
        if tooling_init_error is None:
            tooling_init_error = "registry.openai_tools() raised while computing tool_count"

    payload: Dict[str, Any] = {
        "tools_enabled": tools_enabled,
        "tool_count": tool_count,
        "registry_present": registry is not None,
        "executor_present": executor is not None,
    }

    # Only include init error if present (keeps the happy-path response clean).
    if tooling_init_error:
        payload["tooling_init_error"] = tooling_init_error

    return payload
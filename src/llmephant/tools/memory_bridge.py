from __future__ import annotations
import hashlib
import json
from typing import Any, Dict, List, Optional
from llmephant.core.logger import setup_logger
from llmephant.models.memory_model import MemoryItem
from llmephant.models.tool_model import ToolCall, ToolDefinition, ToolResult

logger = setup_logger(__name__)


def _safe_json_dumps(obj: Any) -> Optional[str]:
    """Best-effort stable JSON serialization for hashing/metadata.

    Returns None if the object cannot be serialized.
    """
    try:
        return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        return None


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _get_call_args(call: ToolCall) -> Dict[str, Any]:
    """Extract tool call arguments in a defensive, provider-agnostic way."""
    for attr in ("arguments", "args", "tool_arguments", "kwargs"):
        v = getattr(call, attr, None)
        if isinstance(v, dict):
            return v

    v = getattr(call, "arguments_json", None)
    if isinstance(v, str) and v:
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return {}


def _summarize_value(value: Any) -> Dict[str, Any]:
    """Return a compact, data-agnostic shape summary.

    NOTE: intentionally does not include raw content.

    We also avoid full JSON serialization of large payloads (e.g., DOMs, log blobs) for hashing.
    The hash is based on a bounded representation.
    """
    summary: Dict[str, Any] = {"kind": type(value).__name__}

    # Bound how much content we ever hash.
    MAX_HASH_CHARS = 4096
    MAX_KEY_PREVIEW = 25

    if isinstance(value, dict):
        keys = list(value.keys())
        top_keys = [str(k) for k in keys[:MAX_KEY_PREVIEW]]
        summary.update({
            "n_keys": len(keys),
            "top_keys": top_keys,
        })
        basis = _safe_json_dumps({"kind": "dict", "n_keys": len(keys), "top_keys": top_keys})
        if basis is None:
            basis = f"dict:{len(keys)}:{','.join(top_keys)}"

    elif isinstance(value, (list, tuple, set)):
        n_items = len(value)
        summary["n_items"] = n_items
        basis = f"{type(value).__name__}:{n_items}"

    elif isinstance(value, (bytes, bytearray)):
        n_bytes = len(value)
        summary["n_bytes"] = n_bytes
        prefix = bytes(value[:MAX_HASH_CHARS])
        basis = f"bytes:{n_bytes}:" + hashlib.sha256(prefix).hexdigest()

    elif isinstance(value, str):
        n_chars = len(value)
        summary["n_chars"] = n_chars
        prefix = value[:MAX_HASH_CHARS]
        basis = f"str:{n_chars}:" + _sha256_hex(prefix)

    else:
        # Last-resort bounded repr.
        r = repr(value)
        if len(r) > MAX_HASH_CHARS:
            r = r[:MAX_HASH_CHARS]
        basis = f"{type(value).__name__}:{r}"

    summary["sha256"] = _sha256_hex(basis)
    return summary


def tool_result_to_memory_items(
    *,
    tool: ToolDefinition,
    call: ToolCall,
    result: ToolResult,
    user_id: str,
) -> List[MemoryItem]:
    """
    Convert a tool result into memory items according to the tool's MemoryPolicy.

    This function is intentionally pure (no DB writes). The caller decides where to store.
    """
    policy = tool.memory_policy
    if not policy or not policy.store:
        return []

    # Minimal, tool-agnostic default behavior:
    # - store a derived "index memory" (summary + provenance)
    args = _get_call_args(call)

    # Summarize either the normal result or the error payload, without storing raw content.
    payload = getattr(result, "error", None) if result.is_error else result.result
    result_summary = _summarize_value(payload)

    status = "error" if result.is_error else "ok"
    text = f"Tool '{tool.name}' {status}. Result summary: {result_summary}."

    metadata: Dict[str, Any] = {
        "tool": tool.name,
        "provider": tool.provider,
        "call_id": call.call_id,
        "memory_type": policy.memory_type,
        "extractor": policy.extractor,
        "is_error": result.is_error,
        "arguments_keys": sorted([str(k) for k in args.keys()])[:50],
        "arguments_sha256": _sha256_hex(_safe_json_dumps(args) or repr(args)),
        "result_summary": result_summary,
    }

    return [
        MemoryItem(
            user_id=user_id,
            text=text,
            scope=policy.scope,
            metadata=metadata,
            source=f"tool:{tool.name}",
        )
    ]
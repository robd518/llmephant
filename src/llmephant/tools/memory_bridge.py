

from __future__ import annotations

from typing import Any, List, Optional

from llmephant.core.logger import setup_logger
from llmephant.models.memory_model import MemoryItem
from llmephant.models.tool_model import ToolCall, ToolDefinition, ToolResult

logger = setup_logger(__name__)


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

    # Minimal default behavior:
    # - store the raw result as a "fact" memory
    # - include provenance for later debugging/traceability
    text = f"Tool '{tool.name}' result: {result.result!r}"
    return [
        MemoryItem(
            user_id=user_id,
            text=text,
            scope=policy.scope,
            metadata={
                "tool": tool.name,
                "provider": tool.provider,
                "call_id": call.call_id,
                "memory_type": policy.memory_type,
                "extractor": policy.extractor,
                "is_error": result.is_error,
            },
            source=f"tool:{tool.name}",
        )
    ]
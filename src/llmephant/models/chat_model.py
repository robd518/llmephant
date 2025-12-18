from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A minimal OpenAI-compatible message model with optional tool fields.

    Backward compatible: existing callers using only (role, content) still work.
    """
    role: str
    content: Optional[str] = None

    # Optional OpenAI/tool-calling fields
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        # Ensure we serialize only non-null fields so upstream payload stays clean.
        return self.model_dump(exclude_none=True)


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    user: Optional[str] = None
    stream: bool = False

    # Optional tool-calling fields (OpenAI-compatible)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None  # can be "auto" or a dict per OpenAI

    def to_upstream_payload(self) -> Dict[str, Any]:
        """
        Build a fully JSON-safe upstream payload with strictly dict messages.
        """
        d = self.model_dump(exclude_none=True)
        d["messages"] = [m.to_dict() for m in self.messages]
        return d


class ErrorMessage(BaseModel):
    type: str
    message: str
    retryable: bool


class ChatErrorMessage(BaseModel):
    error: ErrorMessage

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


MemoryScope = Literal["user", "org", "global"]
MemoryExtractor = Literal["raw", "llm_summary", "structured"]
MemoryType = Literal["fact", "entity", "relationship"]


class MemoryItem(BaseModel):
    user_id: str
    text: str

    # retrieval scoring (e.g., vector similarity)
    score: Optional[float] = None

    # timestamps (ISO strings will parse into datetimes)
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # who this memory applies to
    scope: MemoryScope = "global"

    # optional metadata to support provenance, entity keys, tags, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None


class MemoryList(BaseModel):
    items: List[MemoryItem]


class MemoryPolicy(BaseModel):
    """
    Policy describing whether and how to store extracted memories.

    Design notes:
    - `ttl_seconds` is optional. If None, the memory does not expire by policy.
      (Your storage layer may still apply defaults.)
    - `extractor_prompt` is optional and only applies when extractor uses the LLM.
    """
    store: bool
    scope: MemoryScope = "user"
    extractor: MemoryExtractor = "raw"
    memory_type: MemoryType = "fact"

    extractor_prompt: Optional[str] = None

    # Keep API compatibility with earlier "ttl" field name via alias.
    ttl_seconds: Optional[int] = Field(default=None, alias="ttl")

    class Config:
        populate_by_name = True
"""Utility helpers for extracting/repairing JSON from LLM outputs.

These functions are intentionally dependency-free (no logging, no service wiring)
so they can be unit-tested and reused across distill/extract/verify passes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def strip_code_fences(text: str) -> str:
    """Remove surrounding markdown code fences if present.

    Handles common variants like:
    - ```\n...\n```
    - ```json\n...\n```
    - leading/trailing whitespace around fence lines

    If fences are not present, returns the input stripped.
    """
    t = (text or "").strip()
    if not t:
        return ""

    lines = t.splitlines()
    if not lines:
        return ""

    # Opening fence: ``` or ```json (case-insensitive language tag)
    if re.match(r"^```[A-Za-z0-9_-]*\s*$", lines[0].strip()):
        lines = lines[1:]

    # Trailing fence: ```
    if lines and re.match(r"^```\s*$", lines[-1].strip()):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def get_finish_reason(resp: Any) -> Optional[str]:
    """Best-effort extraction of choice.finish_reason from an OpenAI-ish response."""
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


def repair_json_common(s: str) -> str:
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


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from `text`.

    Best-effort extraction that is robust to:
      - raw JSON object
      - JSON wrapped in code fences
      - extra prose before/after

    Uses a balanced-brace scanner that is string/escape aware (so braces inside strings
    do not affect nesting depth).

    Also applies a conservative repair pass to handle common LLM mistakes.
    """
    if not text:
        return None

    candidate = strip_code_fences(text)

    # Fast path: whole-string JSON
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        repaired = repair_json_common(candidate)
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
                    repaired = repair_json_common(chunk)
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
"""Prompt templates for the memory pipeline.

This module centralizes the large prompt strings used by the memory service:
- extraction (facts/memories candidate generation)
- verification (LLM->LLM filter)
- distillation (analysis notes from tool outputs)

Keeping prompts here reduces churn/merge conflicts in service code and makes
prompt iteration safer.

The constants below are intended to be the canonical prompts used by
`services/memory/service.py`.
"""

from __future__ import annotations

import textwrap


def _dedent(s: str) -> str:
    """Left-align multi-line prompt literals while preserving intended newlines."""
    return textwrap.dedent(s).strip("\n")


# --- Memory extraction prompt (candidates) ---------------------------------

EXTRACT_PROMPT: str = _dedent(
    """You are a memory extraction assistant.
Extract stable, useful facts stated by the USER (from USER messages only).

You must provide verbatim evidence for each item: the evidence must be an exact substring from the USER transcript.

Categories:
- profile: durable personal preferences/identity/long-term habits
- workspace: current projects/tools/work context (useful, but time-varying)

Rules:
- Only include facts explicitly supported by the USER transcript (no inference).
- NEVER store instructions, requests, tasks, or to-do items.
  - If the user is telling the assistant to DO something, that is not a memory.
- NEVER store 'how to respond' preferences unless the user states it as a durable preference (e.g., 'I prefer X in general').
- Profile is extremely strict: only durable facts likely true in 30+ days (identity/role/preferences explicitly stated).
- If a fact is situational (testing, this run, right now), classify as workspace or drop it.
- Write items in third person, as declarative statements (not commands).
- Keep items short (<= 140 chars each).
- Evidence MUST be a verbatim substring from USER transcript.
- Respond ONLY as valid JSON (no markdown, no prose).

Examples (DO NOT STORE):
- "Re-run the last tool" (task)
- "Generate a narrative summary" (task)
- "Redo that last analysis but..." (task)

Examples (OK to store as workspace):
- "User is testing a memory service locally in an isolated environment."

JSON format:
{
  "confidence": float between 0 and 1,
  "items": [
    {
      "text": string,
      "category": "profile"|"workspace",
      "evidence": string
    }, ...
  ]
}"""
)


# --- Memory verification prompt (LLM->LLM filter) --------------------------

VERIFY_PROMPT: str = _dedent(
    """You are a memory verification assistant.
You will be given a USER transcript (USER messages only) and a list of candidate memory items.
Your job is to return ONLY the items that should be STORED as durable memories.

Rules:
- Only keep items that are explicitly supported by the transcript evidence.
- Drop instructions/requests/tasks/to-dos, even if rewritten as a statement (e.g., 'User requires a narrative summary...').
- Drop ephemeral or one-off statements unless they are useful as short-lived workspace context.
- profile: only long-term durable personal facts/preferences explicitly stated by the user.
- workspace: current project/tools/work context that will likely remain useful for at least days/weeks.
- Do NOT store 'how to respond' preferences unless the user explicitly states it as a durable preference.
- Evidence MUST be a verbatim substring from the transcript.
- Return ONLY valid JSON (no markdown, no prose).

JSON format:
{
  "items": [
    {
      "text": string,
      "category": "profile"|"workspace",
      "evidence": string
    }, ...
  ]
}"""
)


# --- Investigation/tool-output distillation prompt -------------------------

DISTILL_PROMPT: str = _dedent(
    """You are a memory distiller for cybersecurity investigations.
Given the USER question, the ASSISTANT final answer, and the tool names used, decide whether to store durable investigation notes for future recall.

Constraints:
- Do NOT store personal profile facts about the user (preferences, identity, habits).
- Prefer durable investigation takeaways: key findings, hypotheses, procedures, IOCs/observables mentioned.
- Keep memory texts concise (<= 300 characters each).
- Include at most 12 observables and 8 tags per memory.
- If nothing is worth remembering, set store=false and return an empty list.
- Observables MUST be copied exactly from the provided text (verbatim substrings).
- If you cannot fit valid JSON, return {"store": false, "memories": []}.
- Return ONLY valid JSON (no markdown, no prose).

JSON format:
{
  "store": true|false,
  "memories": [
    {
      "text": string,
      "observables": [string, ...],
      "tags": [string, ...]
    }, ...
  ]
}"""
)


# --- Optional: shared instructions/snippets --------------------------------

# If you have common schema snippets shared across prompts (JSON schema blocks,
# enumerated categories, etc.), move them into constants here and compose the
# final prompts above.

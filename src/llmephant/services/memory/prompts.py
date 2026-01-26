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

CRITICAL OUTPUT RULES:
- You MUST return a strict SUBSET of the provided candidate items.
- Do NOT create new items.
- Do NOT rewrite or normalize fields.
  - For any kept item, copy `text`, `category`, and `evidence` EXACTLY as provided in the candidate list.
- Preserve order: output items must appear in the same order as the input candidates.

Rules:
- Only keep items that are explicitly supported by the transcript evidence.
- Drop instructions/requests/tasks/to-dos, even if rewritten as a statement (e.g., 'User requires a narrative summary...').
- Drop ephemeral or one-off statements unless they are useful as short-lived workspace context.
- profile: only long-term durable personal facts/preferences explicitly stated by the user.
- workspace: current project/tools/work context that will likely remain useful for at least days/weeks.
- Do NOT store 'how to respond' preferences unless the user explicitly states it as a durable preference.
- Evidence MUST be a verbatim substring from the transcript.
- Return ONLY valid JSON (no markdown, no prose).

If nothing should be stored, return exactly: {"items": []}

JSON format:
{
  "items": [
    {
      "text": string,
      "category": "profile"|"workspace",
      "evidence": string
    }, ...
  ]
}
"""
)


# --- Investigation/tool-output distillation prompt -------------------------

DISTILL_PROMPT: str = _dedent(
    """You are a Memory Compiler for investigation notes.

You will be given the USER question, the ASSISTANT final answer, and tool activity (tool names and any tool outputs).
Your job is to produce compact, durable investigation memories suitable for future recall.

OUTPUT (STRICT):
- Return ONLY a single valid JSON object. No markdown, no code fences, no prose.
- Output MUST match this JSON shape exactly:

{
  "memories": [
    {
      "lane": "all",
      "scope": "user" | "workspace" | "notes" | "global" | "unknown",
      "kind": "fact" | "preference" | "plan" | "decision" | "definition" | "event" | "note" | "other",
      "summary": "string",
      "evidence": "string" | null,
      "tags": ["string", "..."],
      "details": {"any": "json"},
      "confidence": 0.0,
      "salience": 0.0
    }
  ]
}

RULES:
- This compiler is for INVESTIGATION NOTES ONLY (analysis/tool distillation).
- Do NOT store personal profile facts about the user (name, identity, bio, preferences).
- Do NOT store tasks/to-dos/instructions.
- "lane" MUST always be "all" for investigation memories.
- For investigation memories, set:
  - scope = "notes"
  - kind = "note" (or "event" if it is a concrete observed event)
- "summary" must be concise and retrieval-friendly:
  - 800 characters max.
  - Single-line only: do NOT include newline/tab characters and do NOT include markdown tables.
  - Use short sentences separated by semicolons if needed.

OBSERVABLES (IMPORTANT):
- If any observables appear verbatim in the provided input (USER question, ASSISTANT answer, tool names/outputs), collect them.
- Observables include: domains, URLs, IPs, file names, process/service names, registry keys, email addresses, hashes (MD5/SHA1/SHA256), and user/host identifiers.
- Do NOT invent observables. Only include strings that appear verbatim in the provided input.
- Put observables in details.observables as a JSON array of strings.
- Limit details.observables to at most 20 items.
- If details.observables is present and non-empty, the summary MUST mention at least one of the highest-salience observables (e.g., the domain or hash).
- If no observables are present in the input, omit details.observables.

- "tags" is optional but recommended (0-8 tags). Keep tags short.
- "details" is optional but recommended when helpful. Use it for structured extras such as:
  - {"tools_used": ["tool_a", "tool_b"], "observables": ["..."], "hosts": ["..."], "files": ["..."], "campaign": "..."}
  - Do NOT invent tool calls, observables, or other entities.
  - Only include tool names/observables that appear in the provided input.
  - HARD RULE: If the provided input includes a non-empty list of tool names, you MUST set details.tools_used to exactly that list (same tool names, same order). If details is omitted, create it.
  - HARD RULE: If the provided input indicates zero tool names, do NOT include tools_used in details.
- "evidence": always null for investigation memories.
- "confidence" and "salience" are numbers 0..1.
- Create at most 6 memories. Prefer fewer, higher-signal items.
- If nothing is worth remembering, return exactly: {"memories": []}

Examples (valid minimal):
- {"memories": []}
- {"memories": [{"lane":"all","scope":"notes","kind":"note","summary":"Observed domain example.org associated with phishing; recommend blocking; monitor similar SMTP anomalies.","evidence":null,"tags":["phishing"],"details":{"observables":["example.org"],"tools_used":["ti_lookup"]},"confidence":0.7,"salience":0.6}]}
"""
)


# --- Optional: shared instructions/snippets --------------------------------

# If you have common schema snippets shared across prompts (JSON schema blocks,
# enumerated categories, etc.), move them into constants here and compose the
# final prompts above.

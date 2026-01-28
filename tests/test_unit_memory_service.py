
import json
import inspect
from types import SimpleNamespace

import pytest


def _msg(role: str, content: str | None = None, *, name: str | None = None, tool_name: str | None = None):
    """Create a minimal message object compatible with the memory service helpers.

    The memory service uses `getattr` for fields like role/name/content, so a SimpleNamespace is sufficient.
    """

    ns = SimpleNamespace(role=role)
    if content is not None:
        ns.content = content
    if name is not None:
        ns.name = name
    if tool_name is not None:
        ns.tool_name = tool_name
    return ns


@pytest.fixture()
def memory_service():
    # Import lazily so test collection doesn't fail if the package layout changes.
    from llmephant.services.memory import service as svc

    return svc


def test_tool_transcript_includes_only_tool_messages(memory_service):
    build = getattr(memory_service, "_build_tool_transcript")

    msgs = [
        _msg("user", "hi"),
        _msg("assistant", "hello"),
        _msg("tool", "result one\nline2", name="tool_a"),
        _msg("tool", "result two", tool_name="tool_b"),
        _msg("assistant", "done"),
    ]

    out = build(msgs, max_total_chars=4000, max_per_tool_chars=1200)

    # Includes tool name and cleaned content (no newlines)
    assert "[tool_a]" in out
    assert "result one line2" in out

    assert "[tool_b]" in out
    assert "result two" in out

    # Does not include non-tool message content
    assert "hello" not in out
    assert "done" not in out


def test_tool_transcript_truncates_per_tool(memory_service):
    build = getattr(memory_service, "_build_tool_transcript")

    long = "x" * 500
    msgs = [_msg("tool", long, name="tool_a")]

    out = build(msgs, max_total_chars=4000, max_per_tool_chars=50)

    # Should be truncated and end with ellipsis.
    assert out.startswith("[tool_a] ")
    assert out.endswith("…")
    assert len(out) <= len("[tool_a] ") + 50


def test_tool_transcript_truncates_total(memory_service):
    build = getattr(memory_service, "_build_tool_transcript")

    msgs = [
        _msg("tool", "a" * 200, name="tool_a"),
        _msg("tool", "b" * 200, name="tool_b"),
        _msg("tool", "c" * 200, name="tool_c"),
    ]

    out = build(msgs, max_total_chars=140, max_per_tool_chars=1200)

    assert len(out) <= 140
    # At least one tool included.
    assert "[tool_a]" in out or "[tool_b]" in out or "[tool_c]" in out


def test_filter_verbatim_observables_keeps_only_present(memory_service):
    filt = getattr(memory_service, "_filter_verbatim_observables")

    source = "Observed domain example.org and IP 1.2.3.4; file shipdocs.pdf"
    observables = ["example.org", "missing.example", "1.2.3.4", "", None, "shipdocs.pdf"]

    out = filt(observables, source)

    assert "example.org" in out
    assert "1.2.3.4" in out
    assert "shipdocs.pdf" in out
    assert "missing.example" not in out


def test_build_user_transcript_only_user_messages(memory_service):
    build = getattr(memory_service, "_build_user_transcript")

    msgs = [
        _msg("user", "First"),
        _msg("assistant", "Ignore this"),
        _msg("user", "Second"),
    ]

    assert build(msgs) == "First\nSecond"


def test_extractor_is_hard_cutover_no_facts_backcompat(memory_service):
    """Regression guard: extraction must not accept legacy `{facts:[...]}` schema."""

    fn = getattr(memory_service, "extract_and_store_memory")
    src = inspect.getsource(fn)

    assert 'obj.get("facts"' not in src
    assert "Back-compat" not in src


def test_memory_tuning_constants_are_module_level(memory_service):
    """Ensure the main tuning knobs remain easy to find/edit."""

    for name in (
        "MEMORY_MAX_NOTE_CHARS",
        "MEMORY_SUMMARY_CAP",
        "MEMORY_MAX_NOTES",
        "MEMORY_MAX_TAGS",
        "MEMORY_MAX_OBSERVABLES",
    ):
        assert hasattr(memory_service, name), f"Missing tuning constant: {name}"


@pytest.mark.asyncio
async def test_extract_parser_filters_by_evidence(monkeypatch, memory_service):
    """Smoke test of extraction + evidence gating."""

    fn = getattr(memory_service, "_distill_memories")

    extract_obj = {
        "items": [
            {
                "text": "User prefers Python for side projects.",
                "category": "profile",
                "evidence": "I prefer Python",
            },
            {
                "text": "User prefers Rust.",
                "category": "profile",
                "evidence": "I prefer Rust",
            },
        ]
    }

    async def _fake_chat_upstream(*args, **kwargs):
        # get_finish_reason(...) typically reads the choice-level finish_reason.
        return {
            "choices": [
                {
                    "message": {"content": json.dumps(extract_obj)},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(memory_service, "chat_upstream", _fake_chat_upstream)

    msgs = [
        _msg("user", "I prefer Python for side projects."),
        _msg("assistant", "Got it."),
    ]

    out = await fn(
        user_id="u::analysis",
        messages=msgs,
        assistant_reply="assistant final",
        model_name="dummy",
        req_id="req",
    )

    assert isinstance(out, list)
    assert out == [
        {"text": "User prefers Python for side projects.", "category": "profile"}
    ]


def test_evidence_in_source_normalizes_case_and_whitespace(memory_service):
    fn = getattr(memory_service, "_evidence_in_source")

    assert fn("I prefer Python", "I prefer   python for side projects.") is True
    assert fn("I prefer Rust", "I prefer Python for side projects.") is False

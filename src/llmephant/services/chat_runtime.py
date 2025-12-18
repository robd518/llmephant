from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

import json
from fastapi import Request
from llmephant.core.logger import setup_logger
from llmephant.models.chat_model import ChatRequest, ErrorMessage, ChatErrorMessage
from llmephant.models.chat_model import ChatMessage
from llmephant.models.tool_model import ToolCall
from llmephant.services.memory_service import (
    augment_messages_with_memory,
    extract_and_store_memory,
    handle_explicit_remember_request,
    search_relevant_memories,
)
from llmephant.services.upstream_llm import chat_upstream, chat_upstream_stream
from llmephant.utils.text import get_last_user_message

logger = setup_logger(__name__)


# Runtime emits events that the transport layer can adapt to HTTP/SSE.
# event["type"] in {"chunk", "token", "final", "error", "done"}
# - chunk: a raw upstream chunk (streaming mode; safe-to-forward output chunks only)
# - token: optional raw token text (used by some upstreams / fallbacks)
# - final: a complete upstream response (non-streaming mode)
# - error: an error payload compatible with ChatErrorMessage.model_dump()
# - done: indicates completion (always emitted once at the end)


def _is_openwebui_sidecar_prompt(text: str) -> bool:
    """
    OpenWebUI can send extra non-user-facing tasks (follow-ups/tags/title) to the same chat endpoint.
    These prompts often start with '### Task:' and include '<chat_history>'.

    For these prompts, we should NOT:
      - inject user memories
      - advertise tools / run tool loop
      - run post-completion memory extraction
    """
    if not text:
        return False
    t = text.strip()
    if t.startswith("### Task:") and "<chat_history>" in t:
        return True
    if "Suggest 3-5 relevant follow-up questions" in t and "<chat_history>" in t:
        return True
    if "\"follow_ups\"" in t and "<chat_history>" in t:
        return True
    if "\"tags\"" in t and "<chat_history>" in t:
        return True
    if "\"title\"" in t and "<chat_history>" in t:
        return True
    return False


def _accumulate_tool_call_deltas(
    acc: Dict[int, Dict[str, Any]],
    deltas: List[Dict[str, Any]],
) -> None:
    """Accumulate OpenAI-style streamed tool call deltas into complete tool_calls.

    OpenAI streaming emits tool calls in `choices[0].delta.tool_calls` with partial
    `function.name` and `function.arguments` fragments. This function merges those
    fragments into a stable list of tool_calls we can execute.
    """
    for tc in deltas or []:
        idx = int(tc.get("index", 0))
        cur = acc.setdefault(
            idx,
            {
                "id": tc.get("id"),
                "type": tc.get("type") or "function",
                "function": {"name": "", "arguments": ""},
            },
        )

        if tc.get("id"):
            cur["id"] = tc.get("id")
        if tc.get("type"):
            cur["type"] = tc.get("type")

        fn_delta = tc.get("function") or {}
        if fn_delta.get("name"):
            # Name typically arrives once; prefer setting rather than concatenating.
            cur["function"]["name"] = fn_delta.get("name")
        if fn_delta.get("arguments"):
            cur["function"]["arguments"] += fn_delta.get("arguments")


async def _finalize_completion(
    *,
    user_id: str,
    messages: List[Any],
    assistant_reply: str,
    model_name: str,
) -> None:
    """
    Called exactly once when an assistant response is fully complete.
    Responsible for memory extraction + storage.
    """
    if not assistant_reply or not str(assistant_reply).strip():
        logger.info("Skipping memory extraction: empty assistant reply.")
        return

    try:
        ok = await extract_and_store_memory(
            user_id=user_id,
            messages=messages[-6:],  # sliding window
            assistant_reply=assistant_reply,
            model_name=model_name,
        )
        if ok:
            logger.info("Memory extraction completed and facts were accepted.")
        else:
            logger.info("Memory extraction ran but produced no storable facts (or was skipped).")
    except Exception:
        logger.exception("Memory extraction failed (non-fatal).")


async def run_chat_runtime_stream(
    req: ChatRequest,
    *,
    user_id: str,
    request: Request,
) -> AsyncIterator[Dict[str, Any]]:
    """Streaming runtime with tool support.

    Implements a multi-phase loop:
      stream tokens -> detect tool calls -> execute tools -> restart streaming

    It yields runtime events; the transport layer is responsible for SSE framing.
    """
    logger.info(f"Runtime(stream) start for user_id={user_id}")

    last_msg = get_last_user_message(req.messages)

    is_sidecar = _is_openwebui_sidecar_prompt(last_msg)
    if is_sidecar:
        logger.info("Detected OpenWebUI sidecar task prompt; disabling tools and memory side-effects.")
        req.tools = None
        req.tool_choice = None
    else:
        handle_explicit_remember_request(user_id, last_msg)
        memories = search_relevant_memories(user_id, last_msg)
        req.messages = augment_messages_with_memory(req.messages, memories)

    # Sidecar streaming: passthrough streaming, no tools, no memory extraction.
    if is_sidecar:
        try:
            async for chunk in chat_upstream_stream(
                ChatRequest(
                    model=req.model,
                    messages=req.messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    user=user_id,
                    stream=True,
                )
            ):
                yield {"type": "chunk", "chunk": chunk}
        except Exception as e:
            logger.exception("Streaming upstream failed (sidecar)")
            err = ChatErrorMessage(
                error=ErrorMessage(
                    type="upstream_stream_error",
                    message=str(e),
                    retryable=True,
                )
            )
            yield {"type": "error", "payload": err.model_dump()}
            yield {"type": "done"}
            return

        yield {"type": "done"}
        return

    # Tooling is created once at app startup (FastAPI lifespan) and stored on the FastAPI app state.
    state = request.app.state
    registry = getattr(state, "registry", None)
    executor = getattr(state, "executor", None)

    if registry is None:
        raise RuntimeError(
            "ToolRegistry not initialized. Ensure FastAPI lifespan sets request.app.state.registry before handling requests."
        )
    if executor is None:
        raise RuntimeError(
            "ToolExecutor not initialized. Ensure FastAPI lifespan sets request.app.state.executor before handling requests."
        )

    # Advertise tools to the upstream LLM (OpenAI tool schema)
    req.tools = registry.openai_tools()
    req.tool_choice = "auto"

    # We allow a small number of tool iterations to avoid infinite loops.
    max_tool_iterations = 5
    iteration = 0

    # Buffer assistant text across phases for post-completion memory extraction.
    full_reply_parts: List[str] = []

    while True:
        iteration += 1
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        saw_tool_calls = False

        try:
            async for chunk in chat_upstream_stream(
                ChatRequest(
                    model=req.model,
                    messages=req.messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    user=user_id,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    stream=True,
                )
            ):
                choice0 = (chunk.get("choices") or [{}])[0] or {}
                delta = choice0.get("delta") or {}

                # Content tokens: safe to forward.
                content = delta.get("content")
                if content:
                    full_reply_parts.append(content)
                    yield {"type": "chunk", "chunk": chunk}

                # Tool call deltas: DO NOT forward to client.
                tc_deltas = delta.get("tool_calls") or []
                if tc_deltas:
                    if not saw_tool_calls:
                        indices = [int(t.get("index", 0)) for t in (tc_deltas or [])]
                        logger.info(
                            f"[tool-stream] detected tool_call deltas (iteration={iteration}) indices={indices} count={len(tc_deltas)}"
                        )
                    saw_tool_calls = True
                    _accumulate_tool_call_deltas(tool_calls_acc, tc_deltas)

        except Exception as e:
            logger.exception("Streaming upstream failed (runtime)")
            err = ChatErrorMessage(
                error=ErrorMessage(
                    type="upstream_stream_error",
                    message=str(e),
                    retryable=True,
                )
            )
            yield {"type": "error", "payload": err.model_dump()}
            yield {"type": "done"}
            return

        # If tool calls were requested, execute them and restart streaming.
        if saw_tool_calls and tool_calls_acc:
            if iteration > max_tool_iterations:
                err = ChatErrorMessage(
                    error=ErrorMessage(
                        type="tool_loop_exceeded",
                        message=f"Exceeded max tool iterations ({max_tool_iterations}).",
                        retryable=False,
                    )
                )
                yield {"type": "error", "payload": err.model_dump()}
                yield {"type": "done"}
                return

            tool_calls_raw = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]

            assembled_summary = []
            for tc in tool_calls_raw:
                fn = tc.get("function") or {}

            args = fn.get("arguments") or ""
            preview = args[:60] if isinstance(args, str) else str(args)[:60]
            if isinstance(preview, str):
                preview = preview.replace("\n", "\\n").replace("\r", "\\r")

            assembled_summary.append(
                {
                    "name": fn.get("name"),
                    "args_len": len(args) if isinstance(args, str) else len(str(args)),
                    "has_id": bool(tc.get("id")),
                    "args_preview": repr(preview),
                }
            )
            logger.info(f"[tool-stream] assembled tool_calls (iteration={iteration}): {assembled_summary}")

            # Ensure each tool call has an id for tool responses.
            for i, tc in enumerate(tool_calls_raw):
                if not tc.get("id"):
                    tc["id"] = f"call_{iteration}_{i}"

            # Append assistant message that requested the tool calls.
            req.messages.append(
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=tool_calls_raw,
                )
            )

            for tc in tool_calls_raw:
                call_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name")
                args_str = fn.get("arguments") or "{}"

                logger.info(
                    f"[tool-stream] executing tool (iteration={iteration}) name={name} call_id={call_id} args_len={len(args_str) if isinstance(args_str, str) else 'n/a'}"
                )

                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception as e:
                    logger.warning(
                        f"[tool-stream] failed to parse tool args as JSON (iteration={iteration}) name={name} call_id={call_id}: {e}"
                    )
                    args = {}

                call = ToolCall(name=name, arguments=args, call_id=call_id, raw=tc)

                # Backward-compatible execution: support either executor.execute(call)
                # or executor.execute(name, args) depending on implementation.
                try:
                    tool_result = await executor.execute(call)
                except TypeError:
                    tool_result = await executor.execute(call.name, call.arguments)

                is_error = bool(tool_result.get("is_error"))
                error_msg = tool_result.get("error") or "Tool execution failed"
                result_val = tool_result.get("result")

                logger.info(
                    f"[tool-stream] tool complete (iteration={iteration}) name={name} call_id={call_id} is_error={is_error} result_type={type(result_val).__name__}"
                )

                if is_error:
                    content = json.dumps({"error": error_msg}, ensure_ascii=False)
                else:
                    content = result_val if isinstance(result_val, str) else json.dumps(result_val, ensure_ascii=False)

                req.messages.append(
                    ChatMessage(
                        role="tool",
                        content=content,
                        tool_call_id=call_id,
                        name=name,
                    )
                )

            # Continue the loop now that tool outputs are in messages.
            continue

        # No tool calls: assistant reply is complete.
        assistant_reply = "".join(full_reply_parts).strip()
        if assistant_reply:
            await _finalize_completion(
                user_id=user_id,
                messages=req.messages,
                assistant_reply=assistant_reply,
                model_name=req.model,
            )

        yield {"type": "done"}
        return


async def run_chat_runtime(
    req: ChatRequest,
    *,
    user_id: str,
    request: Request,
) -> AsyncIterator[Dict[str, Any]]:
    """Non-streaming runtime (tool loop + final payload).

    Streaming requests should use `run_chat_runtime_stream` (or we delegate to it for
    backward compatibility).
    """
    logger.info(f"Runtime start for user_id={user_id}")

    last_msg = get_last_user_message(req.messages)

    # Backward compatibility: if a caller routes streaming requests here,
    # delegate to the streaming runtime.
    if getattr(req, "stream", False):
        async for ev in run_chat_runtime_stream(req, user_id=user_id, request=request):
            yield ev
        return

    is_sidecar = _is_openwebui_sidecar_prompt(last_msg)
    if is_sidecar:
        logger.info("Detected OpenWebUI sidecar task prompt; disabling tools and memory side-effects.")
        req.tools = None
        req.tool_choice = None
    else:
        handle_explicit_remember_request(user_id, last_msg)

        memories = search_relevant_memories(user_id, last_msg)
        req.messages = augment_messages_with_memory(req.messages, memories)

    # ---- Non-streaming mode: call upstream and emit final payload ----
    if is_sidecar:
        # Plain one-shot completion: no tool loop, no memory extraction.
        try:
            upstream_resp = await chat_upstream(
                ChatRequest(
                    model=req.model,
                    messages=req.messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    user=user_id,
                    stream=False,
                )
            )
        except Exception as e:
            logger.exception("Sidecar upstream failed (runtime)")
            err = ChatErrorMessage(
                error=ErrorMessage(
                    type="upstream_error",
                    message=str(e),
                    retryable=True,
                )
            )
            yield {"type": "error", "payload": err.model_dump()}
            yield {"type": "done"}
            return

        yield {"type": "final", "payload": upstream_resp}
        yield {"type": "done"}
        return

    # Tooling is created once at app startup (FastAPI lifespan) and stored on the FastAPI app state.
    # `request` is required to avoid circular imports and to ensure app-scoped state is the single source of truth.
    state = request.app.state

    registry = getattr(state, "registry", None)
    executor = getattr(state, "executor", None)

    if registry is None:
        raise RuntimeError(
            "ToolRegistry not initialized. Ensure FastAPI lifespan sets request.app.state.registry before handling requests."
        )

    if executor is None:
        raise RuntimeError(
            "ToolExecutor not initialized. Ensure FastAPI lifespan sets request.app.state.executor before handling requests."
        )

    # Advertise tools to the upstream LLM (OpenAI tool schema)
    req.tools = registry.openai_tools()
    req.tool_choice = "auto"

    # We allow a small number of tool iterations to avoid infinite loops.
    max_tool_iterations = 5
    iteration = 0

    while True:
        iteration += 1
        try:
            upstream_resp = await chat_upstream(
                ChatRequest(
                    model=req.model,
                    messages=req.messages,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    max_tokens=req.max_tokens,
                    user=user_id,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    stream=False,
                )
            )
        except Exception as e:
            logger.exception("Non-streaming upstream failed (runtime)")
            err = ChatErrorMessage(
                error=ErrorMessage(
                    type="upstream_error",
                    message=str(e),
                    retryable=True,
                )
            )
            yield {"type": "error", "payload": err.model_dump()}
            yield {"type": "done"}
            return

        if "error" in upstream_resp:
            # Preserve previous behavior: return upstream error payload as-is
            yield {"type": "final", "payload": upstream_resp}
            yield {"type": "done"}
            return

        if "choices" not in upstream_resp:
            err = ChatErrorMessage(
                error=ErrorMessage(
                    type="unexpected_response",
                    message="Upstream LLM returned a malformed response (missing 'choices').",
                    retryable=False,
                )
            )
            yield {"type": "error", "payload": err.model_dump()}
            yield {"type": "done"}
            return

        message = upstream_resp["choices"][0]["message"]
        tool_calls_raw = message.get("tool_calls") or []

        # If the model requested tools, execute them and continue the loop.
        if tool_calls_raw:
            logger.info(f"Tool calls requested: {tool_calls_raw}")
            if iteration > max_tool_iterations:
                err = ChatErrorMessage(
                    error=ErrorMessage(
                        type="tool_loop_exceeded",
                        message=f"Exceeded max tool iterations ({max_tool_iterations}).",
                        retryable=False,
                    )
                )
                yield {"type": "error", "payload": err.model_dump()}
                yield {"type": "done"}
                return

            # Append the assistant message that contained the tool_calls
            req.messages.append(
                ChatMessage(
                    role="assistant",
                    content=message.get("content"),
                    tool_calls=tool_calls_raw,
                )
            )

            for tc in tool_calls_raw:
                # OpenAI format: {"id": "...", "type":"function", "function":{"name":"add","arguments":"{...}"}}
                call_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name")
                args_str = fn.get("arguments") or "{}"

                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except Exception:
                    args = {}

                call = ToolCall(name=name, arguments=args, call_id=call_id, raw=tc)
                # Support either executor.execute(call) or executor.execute(name, args) depending on implementation.
                try:
                    tool_result = await executor.execute(call)
                except TypeError:
                    tool_result = await executor.execute(call.name, call.arguments)

                # Append tool message for the model to consume
                is_error = bool(tool_result.get("is_error"))
                error_msg = tool_result.get("error") or "Tool execution failed"
                result_val = tool_result.get("result")

                if is_error:
                    content = json.dumps({"error": error_msg}, ensure_ascii=False)
                else:
                    # Avoid double-encoding strings (json.dumps("hi") -> '"hi"')
                    content = result_val if isinstance(result_val, str) else json.dumps(result_val, ensure_ascii=False)

                req.messages.append(
                    ChatMessage(
                        role="tool",
                        content=content,
                        tool_call_id=call_id,
                        name=name,
                    )
                )

            # Continue to ask the model, now that tool outputs are in messages
            continue

        # Otherwise we have a normal assistant reply; finalize and return.
        assistant_reply = message.get("content") or ""
        await _finalize_completion(
            user_id=user_id,
            messages=req.messages,
            assistant_reply=assistant_reply,
            model_name=req.model,
        )

        yield {"type": "final", "payload": upstream_resp}
        yield {"type": "done"}
        return

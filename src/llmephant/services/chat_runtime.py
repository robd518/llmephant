from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

import json

from llmephant.core.logger import setup_logger
from llmephant.models.chat_model import ChatRequest, ErrorMessage, ChatErrorMessage
from llmephant.models.chat_model import ChatMessage
from llmephant.models.tool_model import ToolCall
from llmephant.tools.executor import ToolExecutor
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
# event["type"] in {"chunk", "final", "error", "done"}
# - chunk: a raw upstream chunk (streaming mode)
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


async def run_chat_runtime(
    req: ChatRequest,
    *,
    user_id: str,
    request: Any | None = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Unified execution loop for BOTH streaming and non-streaming chat requests.

    Parity goals:
    - Same memory injection logic as existing non-streaming flow
    - Same SSE chunk payloads as existing streaming passthrough flow
    - Same post-completion memory extraction as existing finalize_completion
    """
    logger.info(f"Runtime start for user_id={user_id}")

    last_msg = get_last_user_message(req.messages)
    logger.info(f"Last user message: {last_msg}")

    is_sidecar = _is_openwebui_sidecar_prompt(last_msg)
    if is_sidecar:
        logger.info("Detected OpenWebUI sidecar task prompt; disabling tools and memory side-effects.")
        req.tools = None
        req.tool_choice = None
    else:
        handle_explicit_remember_request(user_id, last_msg)

        memories = search_relevant_memories(user_id, last_msg)
        req.messages = augment_messages_with_memory(req.messages, memories)

    # ---- Streaming mode: emit raw upstream chunks (SSE passthrough parity) ----
    if getattr(req, "stream", False):
        # Preserve parity: do not advertise tools during streaming yet.
        # (Once streaming tool-interrupt support is implemented, we can enable this.)
        req.tools = None
        req.tool_choice = None
        full_reply_parts: List[str] = []

        try:
            async for chunk in chat_upstream_stream(req):
                # Buffer assistant text for post-stream memory extraction (parity)
                try:
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        full_reply_parts.append(content)
                except Exception:
                    # Never let buffering issues break streaming
                    pass

                # Emit the raw chunk exactly as upstream provided it
                yield {"type": "chunk", "chunk": chunk}

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

        # Post-stream memory extraction (only if we have a complete reply)
        assistant_reply = "".join(full_reply_parts).strip()
        if assistant_reply and not is_sidecar:
            await _finalize_completion(
                user_id=user_id,
                messages=req.messages,
                assistant_reply=assistant_reply,
                model_name=req.model,
            )

        yield {"type": "done"}
        return

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
    # Prefer request.app.state to avoid circular imports; fallback to importing llmephant.app if request isn't provided.
    state = None
    if request is not None:
        try:
            state = request.app.state
        except Exception:
            state = None

    if state is None:
        try:
            # Fallback (works if the app module is already loaded)
            from llmephant.app import app as fastapi_app
            state = getattr(fastapi_app, "state", None)
        except Exception:
            state = None

    registry = getattr(state, "registry", None) if state is not None else None
    executor = getattr(state, "executor", None) if state is not None else None

    if registry is None:
        raise RuntimeError(
            "ToolRegistry not initialized. Ensure FastAPI lifespan sets app.state.registry before handling requests."
        )

    if executor is None:
        executor = ToolExecutor(registry)
        if state is not None:
            state.executor = executor

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

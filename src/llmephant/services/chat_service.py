from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from llmephant.models.chat_model import ChatRequest, ErrorMessage, ChatErrorMessage
from llmephant.services.chat_runtime import run_chat_runtime, run_chat_runtime_stream
from llmephant.core.logger import setup_logger
import json

logger = setup_logger(__name__)


async def dispatch_chat_request(req: ChatRequest, raw_req: Request):
    """
    Entry point for chat requests.
    Delegates execution to the unified chat runtime (streaming + non-streaming).
    """

    # Immediately fail if app.state.executor or app.state.registry is not set up.
    missing = [k for k in ("registry", "executor") if getattr(raw_req.app.state, k, None) is None]
    if missing:
        err = ChatErrorMessage(
            error=ErrorMessage(
                type="server_error",
                message=f"Service unavailable: tooling not initialized (missing: {', '.join(missing)}).",
                retryable=True,
            )
        )
        logger.error("Tooling not initialized on app.state. Missing=%s state_keys=%s", missing, list(vars(raw_req.app.state).keys()))
        return JSONResponse(content=err.model_dump(), status_code=503)

    user_id = raw_req.headers.get("x-user-id") or req.user or "user"
    logger.info(f"Dispatching chat request for user_id={user_id}")

    if getattr(req, "stream", False):
        logger.info("Dispatching streaming chat request (runtime).")

        async def sse_generator():
            async for event in run_chat_runtime_stream(req, user_id=user_id, request=raw_req):
                etype = event.get("type")

                # Forward only streamed output to the client.
                if etype in ("chunk", "token"):
                    if "chunk" in event:
                        yield f"data: {json.dumps(event['chunk'])}\n\n"
                    elif "text" in event:
                        # Fallback: wrap raw token text in a minimal OpenAI-style chunk.
                        yield f"data: {json.dumps({'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': event['text']}, 'finish_reason': None}]})}\n\n"
                elif etype == "error":
                    yield f"data: {json.dumps(event['payload'])}\n\n"
                elif etype == "done":
                    yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    logger.info("Dispatching non-streaming chat request (runtime).")

    final_payload = None
    async for event in run_chat_runtime(req, user_id=user_id, request=raw_req):
        etype = event.get("type")
        if etype == "final":
            final_payload = event.get("payload")
        elif etype == "error":
            final_payload = event.get("payload")

    if final_payload is None:
        err = ChatErrorMessage(
            error=ErrorMessage(
                type="runtime_error",
                message="Chat runtime returned no payload.",
                retryable=True,
            )
        )
        final_payload = err.model_dump()

    return JSONResponse(content=final_payload)
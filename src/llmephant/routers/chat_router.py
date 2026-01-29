from fastapi import APIRouter, Request
from llmephant.models.chat_model import ChatRequest
from llmephant.services.chat_service import dispatch_chat_request
from llmephant.utils.get_user import get_user_from_request

router = APIRouter()


@router.post("/completions")
async def chat_completions(raw_req: Request):
    body = await raw_req.json()

    user_from_header = get_user_from_request(raw_req)
    if user_from_header is not None:
        body["user"] = user_from_header  # override only when it exists

    req = ChatRequest(**body)
    return await dispatch_chat_request(req, raw_req)

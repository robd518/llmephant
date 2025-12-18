from fastapi import APIRouter
from llmephant.services.upstream_llm import list_models

router = APIRouter()

@router.get("/models")
async def get_models():
    return await list_models()
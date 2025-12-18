from fastapi import APIRouter
from llmephant.repositories.qdrant_repository import delete_expired

router = APIRouter()

@router.delete("/expired")
def purge_expired():
    delete_expired()
    return {"status": "purged"}
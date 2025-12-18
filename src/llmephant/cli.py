import uvicorn
from llmephant.core.settings import settings

def main():
    uvicorn.run(
        "llmephant.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_HOT_RELOAD
    )
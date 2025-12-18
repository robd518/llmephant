from contextlib import asynccontextmanager
from fastapi import FastAPI
from llmephant.core.logger import setup_logger
from llmephant.routers import chat_router, memory_router, models_router
from llmephant.services.embedding_service import init_embedder
from llmephant.repositories.qdrant_repository import init_qdrant
from llmephant.tools.registry import ToolRegistry, import_mcp_tools
from llmephant.tools.executor import ToolExecutor
from llmephant.tools.providers.mcp_provider import MCPToolProvider

logger = setup_logger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        mcp = MCPToolProvider(
            name="Test MCP",
            url="http://macbook-pro.cinnebar-mamba.ts.net:8000/mcp",
            tool_name_prefix="mcp__"
        )
        logger.info("🐘🚀 Starting up llmephant!")
        init_embedder()
        init_qdrant()
        app.state.registry = ToolRegistry()
        await import_mcp_tools(app.state.registry, mcp)
        app.state.executor = ToolExecutor(app.state.registry)
        yield
        logger.info("🐘🛑 Shutting down.")

    except RuntimeError as e:
        logger.critical(f"FastAPI failed to start: {str(e)}")
        raise RuntimeError(f"FastAPI failed to start: {str(e)}")

    finally:
        logger.info("Application stopped.")


app = FastAPI(title="LLMephant Memory API", lifespan=lifespan)

app.include_router(chat_router.router, prefix="/v1")
app.include_router(memory_router.router, prefix="/v1/memory")
app.include_router(models_router.router, prefix="/v1")

@app.get("/")
def root():
    return {"status": "ok", "service": "llmephant-memory-api"}
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
        logger.info("🐘🚀 Starting up llmephant!")
        mcp = MCPToolProvider(
            name="Test MCP",
            url="http://macbook-pro.cinnebar-mamba.ts.net:8000/mcp",
            tool_name_prefix="test_mcp__"
        )
        init_embedder()
        init_qdrant()
        app.state.registry = ToolRegistry()
        await import_mcp_tools(app.state.registry, mcp)
        app.state.executor = ToolExecutor(app.state.registry)
        yield
        logger.info("🐘🛑 Shutting down.")

        # Clean up provider resources if the provider exposes a close hook.
        if hasattr(mcp, "aclose"):
            await mcp.aclose()
        elif hasattr(mcp, "close"):
            mcp.close()

    except Exception as e:
        logger.exception("FastAPI failed to start")
        raise

    finally:
        logger.info("Application stopped.")


app = FastAPI(title="LLMephant Memory API", lifespan=lifespan)

app.include_router(chat_router.router, prefix="/v1")
app.include_router(memory_router.router, prefix="/v1/memory")
app.include_router(models_router.router, prefix="/v1")

@app.get("/")
def root():
    return {"status": "ok", "service": "llmephant-memory-api"}
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from llmephant.core.logger import setup_logger
from llmephant.routers import chat_router, memory_router, models_router, health_router
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

        # Tooling is optional per-turn: always create the registry/executor,
        # but treat MCP import failures as a degraded mode (no tools).
        app.state.registry = ToolRegistry()
        app.state.executor = ToolExecutor(app.state.registry)
        app.state.tools_enabled = False
        app.state.tooling_init_error = None

        try:
            await import_mcp_tools(app.state.registry, mcp)
            app.state.tools_enabled = True
            logger.info("✅ Tooling initialized (%d tools)", len(app.state.registry.openai_tools()))
        except Exception as e:
            app.state.tooling_init_error = str(e)
            logger.exception("⚠️ Tooling initialization failed; continuing without tools")

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
router = APIRouter(prefix="/api/v1")

router.include_router(chat_router.router, prefix="/chat")
router.include_router(memory_router.router, prefix="/memory")
router.include_router(models_router.router, prefix="/models")
router.include_router(health_router.router, prefix="/health")

app.include_router(router)
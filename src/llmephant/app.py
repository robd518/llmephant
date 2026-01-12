from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
import httpx
from llmephant.core.logger import setup_logger
from llmephant.core.settings import settings
from llmephant.core.tooling_config import ToolingConfigError, load_tooling_config, tooling_snapshot
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
        init_embedder()
        init_qdrant()

        # Tooling is optional per-turn: always create the registry/executor,
        # but treat MCP import failures as a degraded mode (no tools).
        app.state.registry = ToolRegistry()
        app.state.executor = ToolExecutor(app.state.registry)
        app.state.tools_enabled = False
        app.state.tooling_init_error = None

        # Track providers so we can close them on shutdown.
        app.state.tool_providers = []

        tooling_errors: dict[str, str] = {}
        tools_imported = 0

        try:
            cfg = load_tooling_config(settings.TOOLING_CONFIG_FILE)
            logger.info("Tooling config loaded: %s", tooling_snapshot(cfg))

            if not cfg.enabled:
                logger.info("Tooling disabled by config (enabled=false); continuing without tools")
            else:
                for server in cfg.mcp_servers:
                    if not server.enabled:
                        continue

                    provider = MCPToolProvider(
                        name=server.name,
                        url=server.url,
                        tool_name_prefix=server.tool_name_prefix,
                        headers=server.headers,
                        timeout_s=server.timeout_s,
                        allow_tools=set(server.allow_tools) if server.allow_tools else None,
                        deny_tools=set(server.deny_tools) if server.deny_tools else None,
                    )
                    app.state.tool_providers.append(provider)

                    try:
                        await import_mcp_tools(app.state.registry, provider)
                        tools_imported += 1
                        logger.info(
                            "✅ MCP server '%s' registered (%d tools total)",
                            server.name,
                            len(app.state.registry.openai_tools()),
                        )

                    except httpx.ConnectError as e:
                        logger.warning(f"Tool server unreachable; skipping provider='{provider.name}' err='{e}'")

                    except Exception as e:
                        tooling_errors[server.name] = str(e)
                        logger.exception(
                            "⚠️ MCP server '%s' failed to initialize; continuing without this server",
                            server.name,
                        )

            app.state.tools_enabled = tools_imported > 0

            # Only set tooling_init_error when something actually went wrong.
            app.state.tooling_init_error = tooling_errors or None

            if app.state.tools_enabled:
                logger.info(
                    "✅ Tooling initialized (%d tools)",
                    len(app.state.registry.openai_tools()),
                )
            else:
                logger.warning("⚠️ Tooling not enabled (no servers loaded successfully)")

        except ToolingConfigError as e:
            app.state.tooling_init_error = {"config": str(e)}
            logger.exception("⚠️ Tooling config invalid/unavailable; continuing without tools")

        except Exception as e:
            app.state.tooling_init_error = {"startup": str(e)}
            logger.exception("⚠️ Tooling initialization failed; continuing without tools")

        yield
        logger.info("🐘🛑 Shutting down.")

        # Clean up provider resources if providers expose close hooks.
        for provider in getattr(app.state, "tool_providers", []) or []:
            try:
                if hasattr(provider, "aclose"):
                    await provider.aclose()
                elif hasattr(provider, "close"):
                    provider.close()
            except Exception:
                logger.exception("Failed to close tool provider cleanly")

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
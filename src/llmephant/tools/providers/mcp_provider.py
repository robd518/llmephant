from __future__ import annotations
import asyncio
import json
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx


@dataclass(frozen=True)
class ProviderTool:
    # tool name as exposed by the provider (MCP server)
    name: str
    title: Optional[str]
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ToolCallResult:
    # what we feed back as the tool "content" to the LLM
    text: str
    # raw MCP payload for debugging / memory extraction / auditing
    raw: Dict[str, Any]
    is_error: bool = False


class MCPToolProvider:
    """
    Minimal MCP client for tools/list + tools/call over JSON-RPC 2.0.

    Assumes an HTTP endpoint that accepts JSON-RPC POSTs.
    """
    def __init__(
        self,
        *,
        name: str = "mcp",
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 30.0,
        tool_name_prefix: Optional[str] = None,
        allow_tools: Optional[set[str]] = None,
        deny_tools: Optional[set[str]] = None,
    ) -> None:
        self.name = name
        self.url = url.rstrip("/")
        base_headers: Dict[str, str] = dict(headers or {})
        # Some MCP servers require explicit content negotiation.
        base_headers.setdefault("Accept", "application/json, text/event-stream")
        base_headers.setdefault("Content-Type", "application/json")
        self._headers = base_headers
        # Some MCP servers require a session header. They may return `mcp-session-id`
        # on the first response and expect clients to send it on subsequent requests.
        self._session_id: Optional[str] = None
        # MCP servers commonly require an initialize handshake per session before tool operations.
        self._initialized: bool = False
        self._initialized_session_id: Optional[str] = None
        self._timeout_s = timeout_s
        self._ids = itertools.count(1)
        # Optional persistent client (not currently used); kept for future reuse and clean shutdown.
        self._client: Optional[httpx.AsyncClient] = None
        self.tool_name_prefix = tool_name_prefix or f"{self.name}__"
        self.allow_tools = allow_tools
        self.deny_tools = deny_tools

    def _full_name(self, provider_tool_name: str) -> str:
        return f"{self.tool_name_prefix}{provider_tool_name}"

    async def _rpc(self, client: httpx.AsyncClient, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        req_id = next(self._ids)
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params:
            payload["params"] = params

        async def do_post() -> httpx.Response:
            # Merge base headers + session header (if known)
            headers = dict(self._headers)
            if self._session_id:
                headers["mcp-session-id"] = self._session_id
            return await client.post(self.url, json=payload, headers=headers)

        r = await do_post()

        # Handle servers that require a session id and provide it on first failure.
        if r.status_code == 400:
            sid = r.headers.get("mcp-session-id")
            try:
                ct_400 = (r.headers.get("content-type") or "").lower()
                if "text/event-stream" in ct_400:
                    data_lines_400: List[str] = []
                    for line in r.text.splitlines():
                        if line.startswith("data:"):
                            data_lines_400.append(line[len("data:"):].strip())
                    data_400 = json.loads(data_lines_400[-1]) if data_lines_400 else None
                else:
                    data_400 = r.json()
            except Exception:
                data_400 = None

            missing_session = False
            if isinstance(data_400, dict) and "error" in data_400:
                msg = (data_400.get("error") or {}).get("message") or ""
                if "Missing session ID" in msg:
                    missing_session = True

            if sid and missing_session:
                # Persist session id and retry once.
                self._session_id = sid
                self._headers["mcp-session-id"] = sid
                r = await do_post()

        r.raise_for_status()

        # Some MCP servers respond with SSE (text/event-stream) even for single responses.
        content_type = (r.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            data_lines: List[str] = []
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
            if not data_lines:
                raise RuntimeError("MCP SSE response contained no data lines")
            data = json.loads(data_lines[-1])
        else:
            data = r.json()

        # Capture session id on any successful response as well.
        sid = r.headers.get("mcp-session-id")
        if sid and sid != self._session_id:
            self._session_id = sid
            self._headers["mcp-session-id"] = sid
            # New session => must re-run initialization handshake.
            self._initialized = False
            self._initialized_session_id = None

        if "error" in data:
            # JSON-RPC protocol error
            code = data["error"].get("code")
            msg = data["error"].get("message")
            raise RuntimeError(f"MCP JSON-RPC error {code}: {msg}")

        if data.get("id") != req_id:
            # not fatal, but suspicious
            raise RuntimeError(f"MCP JSON-RPC id mismatch (sent {req_id}, got {data.get('id')})")

        return data["result"]

    async def _notify(self, client: httpx.AsyncClient, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no `id`). Some servers may still respond with an SSE/JSON envelope."""
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params

        async def do_post() -> httpx.Response:
            headers = dict(self._headers)
            if self._session_id:
                headers["mcp-session-id"] = self._session_id
            return await client.post(self.url, json=payload, headers=headers)

        r = await do_post()

        # Handle servers that require a session id and provide it on first failure.
        if r.status_code == 400:
            sid = r.headers.get("mcp-session-id")
            try:
                ct_400 = (r.headers.get("content-type") or "").lower()
                if "text/event-stream" in ct_400:
                    data_lines_400: List[str] = []
                    for line in r.text.splitlines():
                        if line.startswith("data:"):
                            data_lines_400.append(line[len("data:"):].strip())
                    data_400 = json.loads(data_lines_400[-1]) if data_lines_400 else None
                else:
                    data_400 = r.json()
            except Exception:
                data_400 = None

            missing_session = False
            if isinstance(data_400, dict) and "error" in data_400:
                msg = (data_400.get("error") or {}).get("message") or ""
                if "Missing session ID" in msg:
                    missing_session = True

            if sid and missing_session:
                self._session_id = sid
                self._headers["mcp-session-id"] = sid
                # New session => must re-run initialization handshake.
                self._initialized = False
                self._initialized_session_id = None
                r = await do_post()

        r.raise_for_status()

        # If the server sends a JSON-RPC envelope back, surface JSON-RPC errors.
        content_type = (r.headers.get("content-type") or "").lower()
        if "text/event-stream" in content_type:
            data_lines: List[str] = []
            for line in r.text.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[len("data:"):].strip())
            if data_lines:
                data = json.loads(data_lines[-1])
                if isinstance(data, dict) and "error" in data:
                    code = (data["error"] or {}).get("code")
                    msg = (data["error"] or {}).get("message")
                    raise RuntimeError(f"MCP JSON-RPC error {code}: {msg}")
        else:
            # Many servers return an empty body for notifications; if not empty, it may be JSON.
            if r.content:
                try:
                    data = r.json()
                    if isinstance(data, dict) and "error" in data:
                        code = (data["error"] or {}).get("code")
                        msg = (data["error"] or {}).get("message")
                        raise RuntimeError(f"MCP JSON-RPC error {code}: {msg}")
                except Exception:
                    pass

        # Capture session id on any response.
        sid = r.headers.get("mcp-session-id")
        if sid and sid != self._session_id:
            self._session_id = sid
            self._headers["mcp-session-id"] = sid
            self._initialized = False
            self._initialized_session_id = None

    async def _ensure_initialized(self, client: httpx.AsyncClient) -> None:
        """Run MCP initialize handshake once per session."""
        if self._initialized and self._initialized_session_id == self._session_id:
            return

        # MCP initialize request
        await self._rpc(
            client,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "llmephant", "version": "0.0.0"},
            },
        )

        # MCP initialized notification
        await self._notify(client, "notifications/initialized")

        self._initialized = True
        self._initialized_session_id = self._session_id

    async def list_tools(self) -> List[ProviderTool]:
        tools: List[ProviderTool] = []

        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout_s) as client:
            await self._ensure_initialized(client)
            cursor: Optional[str] = None
            while True:
                params: Dict[str, Any] = {}
                if cursor:
                    params["cursor"] = cursor

                result = await self._rpc(client, "tools/list", params)
                for t in result.get("tools", []):
                    name = t["name"]
                    if self.allow_tools is not None and name not in self.allow_tools:
                        continue
                    if self.deny_tools is not None and name in self.deny_tools:
                        continue

                    tools.append(
                        ProviderTool(
                            name=name,
                            title=t.get("title"),
                            description=t.get("description", ""),
                            input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
                            output_schema=t.get("outputSchema"),
                            annotations=t.get("annotations"),
                        )
                    )

                cursor = result.get("nextCursor")
                if not cursor:
                    break
        return tools

    async def call_tool(self, provider_tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout_s) as client:
            await self._ensure_initialized(client)
            result = await self._rpc(
                client,
                "tools/call",
                {"name": provider_tool_name, "arguments": arguments or {}},
            )

        # Normalize MCP content -> a single text blob for your existing tool loop.
        content_items = result.get("content", []) or []
        structured = result.get("structuredContent", None)

        parts: List[str] = []
        for item in content_items:
            t = item.get("type")
            if t == "text":
                parts.append(item.get("text", ""))
            else:
                # Don’t lose non-text results; preserve them as JSON.
                parts.append(json.dumps(item, ensure_ascii=False))

        if structured is not None:
            # Helpful for memory extraction / downstream parsing
            parts.append(json.dumps({"structuredContent": structured}, ensure_ascii=False))

        text = "\n".join(p for p in parts if p is not None and p != "")

        return ToolCallResult(
            text=text if text else json.dumps(result, ensure_ascii=False),
            raw=result,
            is_error=bool(result.get("isError", False)),
        )


    async def aclose(self) -> None:
        """Release any resources held by this provider.

        Note: The current implementation creates an AsyncClient per call (via `async with`),
        so there is typically nothing to close. This hook exists so the FastAPI lifespan
        can always safely call `await provider.aclose()`.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def close(self) -> None:
        """Best-effort synchronous close.

        If called while an event loop is running, schedules `aclose()`.
        """
        if self._client is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.aclose())
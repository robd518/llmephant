from __future__ import annotations
import asyncio
import json
import itertools
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional
import httpx
from llmephant.core.logger import setup_logger

logger = setup_logger(__name__)


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

    def _req_auth_diag(self, r: httpx.Response) -> tuple[list[str], bool, Optional[str]]:
        """Return (header_keys, has_auth, auth_scheme) without leaking secret values."""
        req = r.request
        req_hdrs = dict(req.headers) if req is not None else {}
        header_keys = sorted(list(req_hdrs.keys()))
        auth_val = req_hdrs.get("authorization") or req_hdrs.get("Authorization")
        has_auth = auth_val is not None and str(auth_val).strip() != ""
        auth_scheme: Optional[str] = None
        if has_auth:
            try:
                auth_scheme = str(auth_val).split()[0]
            except Exception:
                auth_scheme = "<unparseable>"
        return header_keys, has_auth, auth_scheme

    def _extract_error_message(self, r: httpx.Response) -> str:
        """Best-effort extraction of an error message from JSON/SSE/plain responses."""
        msg = ""
        try:
            ct = (r.headers.get("content-type") or "").lower()
            if "text/event-stream" in ct:
                data_lines: List[str] = []
                for line in (r.text or "").splitlines():
                    if line.startswith("data:"):
                        data_lines.append(line[len("data:"):].strip())
                data = json.loads(data_lines[-1]) if data_lines else None
            else:
                data = r.json()
            if isinstance(data, dict) and "error" in data:
                msg = (data.get("error") or {}).get("message") or ""
        except Exception:
            pass

        if not msg:
            try:
                msg = r.text or ""
            except Exception:
                msg = ""
        return msg

    def _is_session_problem(self, msg: str) -> bool:
        m = (msg or "").lower()
        return (
            "missing session id" in m
            or "no valid session id" in m
            or "invalid session" in m
        )

    def _capture_session_from_response(self, r: httpx.Response) -> None:
        """Capture/refresh the MCP session id from any response.

        IMPORTANT: We do NOT persist the session id into base headers (`self._headers`).
        We attach it per request from `self._session_id` to avoid sticky stale sessions.
        """
        sid = r.headers.get("mcp-session-id")
        if sid and sid != self._session_id:
            self._session_id = sid
            # New/changed session => must re-run initialization handshake.
            self._initialized = False
            self._initialized_session_id = None

    def _log_request_diag(self, r: httpx.Response, *, kind: str, method: str, payload_id: Optional[int]) -> None:
        """Lightweight request diagnostics (debug only; never logs secrets)."""
        try:
            req = r.request
            header_keys, has_auth, auth_scheme = self._req_auth_diag(r)
            logger.debug(
                "MCP %s sent method=%s url=%s jsonrpc_id=%s status=%s has_session=%s initialized=%s header_keys=%s has_auth=%s auth_scheme=%s",
                kind,
                method,
                str(req.url) if req is not None else "<no-request>",
                payload_id,
                r.status_code,
                bool(self._session_id),
                bool(self._initialized),
                header_keys,
                has_auth,
                auth_scheme,
            )
        except Exception:
            pass

    def _log_http_error_diag(self, r: httpx.Response, *, kind: str, method: str, payload_id: Optional[int]) -> None:
        """HTTP error diagnostics (error level; never logs secrets)."""
        try:
            req = r.request
            header_keys, has_auth, auth_scheme = self._req_auth_diag(r)
            ct = (r.headers.get("content-type") or "").lower()
            try:
                body_preview = (r.text or "")[:1200]
            except Exception:
                body_preview = "<unavailable>"

            logger.error(
                "MCP %s HTTP error status=%s url=%s rpc_method=%s jsonrpc_id=%s content_type=%s header_keys=%s has_auth=%s auth_scheme=%s has_session=%s initialized=%s body_preview=%r",
                kind,
                r.status_code,
                str(req.url) if req is not None else "<no-request>",
                method,
                payload_id,
                ct,
                header_keys,
                has_auth,
                auth_scheme,
                bool(self._session_id),
                bool(self._initialized),
                body_preview,
            )
        except Exception:
            logger.exception("Failed to log MCP %s HTTP error diagnostics", kind)

    async def _maybe_recover_session(
        self,
        *,
        client: httpx.AsyncClient,
        kind: str,
        method: str,
        payload_id: Optional[int],
        do_post: Callable[[], Awaitable[httpx.Response]],
        r: httpx.Response,
    ) -> httpx.Response:
        """Recover from missing/invalid session errors by resetting state and retrying once."""
        if r.status_code != 400:
            return r

        sid = r.headers.get("mcp-session-id")
        msg = self._extract_error_message(r)

        if not self._is_session_problem(msg):
            return r

        logger.info("MCP session problem (%s): %r; resetting session and re-initializing", kind, msg[:200])

        # Use new session id if server provided it; otherwise clear it.
        self._session_id = sid
        self._initialized = False
        self._initialized_session_id = None

        # For non-handshake methods, redo handshake then retry once.
        if method not in ("initialize", "notifications/initialized"):
            await self._ensure_initialized(client)

        return await do_post()

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

        self._log_request_diag(r, kind="RPC", method=method, payload_id=req_id)

        # Recover if server reports missing/invalid session.
        r = await self._maybe_recover_session(
            client=client,
            kind="RPC",
            method=method,
            payload_id=req_id,
            do_post=do_post,
            r=r,
        )

        if r.status_code >= 400:
            self._log_http_error_diag(r, kind="RPC", method=method, payload_id=req_id)

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
        self._capture_session_from_response(r)

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
        self._capture_session_from_response(r)

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

        self._log_request_diag(r, kind="notify", method=method, payload_id=None)

        # Recover if server reports missing/invalid session.
        r = await self._maybe_recover_session(
            client=client,
            kind="notify",
            method=method,
            payload_id=None,
            do_post=do_post,
            r=r,
        )

        if r.status_code >= 400:
            self._log_http_error_diag(r, kind="notify", method=method, payload_id=None)

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
        self._capture_session_from_response(r)


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
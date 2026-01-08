from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ToolingConfigError(RuntimeError):
    """Raised when the tooling YAML is missing or invalid."""


def resolve_tooling_config_path(path: str) -> Path:
    """Resolve a tooling config path deterministically.

    Accepts absolute paths, or relative paths resolved against:
      1) the current working directory
      2) the project root (derived from this file location)

    This makes `TOOLING_CONFIG_FILE=./tooling_config.yml` work reliably in Docker/uvicorn,
    even when the process working directory differs.
    """
    if not path or not str(path).strip():
        raise ToolingConfigError("Tooling config path is empty")

    # Expand env vars / ~ in the PATH itself.
    raw = os.path.expandvars(str(path))
    raw = os.path.expanduser(raw)

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate

    tried: list[Path] = []

    # 1) Relative to CWD
    cwd_candidate = (Path.cwd() / candidate).resolve()
    tried.append(cwd_candidate)
    if cwd_candidate.exists():
        return cwd_candidate

    # 2) Relative to project root (repo root): .../<repo>/src/llmephant/core/tooling_config.py
    # parents: core -> llmephant -> src -> <repo>
    repo_root = Path(__file__).resolve().parents[3]
    repo_candidate = (repo_root / candidate).resolve()
    tried.append(repo_candidate)
    if repo_candidate.exists():
        return repo_candidate

    # Provide a helpful error with attempted locations.
    tried_str = ", ".join(str(p) for p in tried)
    raise ToolingConfigError(
        f"Tooling config file not found. Path='{path}'. Tried: {tried_str}"
    )


def _has_reserved_chars(value: str) -> bool:
    # Tool names are typically composed later as "provider:tool".
    # If provider names/prefixes contain ":", collisions and parsing bugs are likely.
    return ":" in value


class MCPServerConfig(BaseModel):
    """Declarative config for a single MCP server."""

    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    enabled: bool = True

    # Prefix used when registering tools from this server.
    # Example: "weather__" -> "weather__get_forecast"
    tool_name_prefix: Optional[str] = None

    # Optional HTTP headers for auth (e.g., Authorization: Bearer ...).
    # Values may be populated via environment expansion in the YAML loader.
    headers: Optional[Dict[str, str]] = None

    # Request timeout (seconds) for calls to this MCP server.
    timeout_s: float = 30.0

    # Optional tool allow/deny lists (matched against the *raw* MCP tool name before prefixing).
    allow_tools: Optional[List[str]] = None
    deny_tools: Optional[List[str]] = None

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "MCPServerConfig":
        name = self.name.strip()
        if not name:
            raise ValueError("MCP server name cannot be empty")
        if _has_reserved_chars(name):
            raise ValueError("MCP server name may not contain ':'")

        prefix = (self.tool_name_prefix or f"{name}__").strip()
        if not prefix:
            raise ValueError("tool_name_prefix cannot be empty")
        if _has_reserved_chars(prefix):
            raise ValueError("tool_name_prefix may not contain ':'")

        # Validate headers (keys must be non-empty strings; values must be strings).
        hdrs = self.headers
        if hdrs is not None:
            cleaned: Dict[str, str] = {}
            for k, v in hdrs.items():
                k2 = str(k).strip()
                if not k2:
                    raise ValueError("headers contains an empty key")
                # Preserve header key casing, but strip whitespace.
                v2 = str(v)
                cleaned[k2] = v2
            object.__setattr__(self, "headers", cleaned)

        # Validate timeout.
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        # Validate allow/deny lists.
        allow = [t.strip() for t in (self.allow_tools or []) if t and t.strip()]
        deny = [t.strip() for t in (self.deny_tools or []) if t and t.strip()]

        if allow and deny:
            overlap = sorted(set(allow).intersection(set(deny)))
            if overlap:
                raise ValueError(
                    f"allow_tools and deny_tools overlap for server '{name}': {overlap}"
                )

        object.__setattr__(self, "allow_tools", allow or None)
        object.__setattr__(self, "deny_tools", deny or None)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tool_name_prefix", prefix)
        return self


class ToolingConfig(BaseModel):
    """Top-level tooling config loaded from YAML."""

    enabled: bool = True
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> "ToolingConfig":
        # Enforce uniqueness across enabled servers.
        names: set[str] = set()
        prefixes: set[str] = set()

        for s in self.mcp_servers:
            if not s.enabled:
                continue

            if s.name in names:
                raise ValueError(f"Duplicate MCP server name among enabled servers: {s.name}")
            names.add(s.name)

            if s.tool_name_prefix in prefixes:
                raise ValueError(
                    f"Duplicate tool_name_prefix among enabled servers: {s.tool_name_prefix}"
                )
            prefixes.add(s.tool_name_prefix)

        return self


def load_tooling_config(path: str, *, env_expand: bool = True) -> ToolingConfig:
    """Load tooling configuration from a YAML file.

    The YAML may be either:
      - top-level keys {enabled, mcp_servers}
      - or nested under a top-level `tooling:` key

    Environment variables in the file (e.g. ${MCP_URL}) may be expanded.
    """
    p = resolve_tooling_config_path(path)

    raw = p.read_text(encoding="utf-8")
    if env_expand:
        raw = os.path.expandvars(raw)

    try:
        data = yaml.safe_load(raw) or {}
    except Exception as e:
        raise ToolingConfigError(f"Failed to parse tooling YAML: {e}") from e

    if not isinstance(data, dict):
        raise ToolingConfigError("Tooling YAML root must be a mapping/object")

    # Allow nesting under `tooling:`.
    if "tooling" in data and isinstance(data["tooling"], dict):
        data = data["tooling"]

    # Normalize key name.
    if "mcp" in data and "mcp_servers" not in data and isinstance(data["mcp"], dict):
        # Support an alternative shape:
        # tooling:
        #   enabled: true
        #   mcp:
        #     servers: [...]  (or mcp_servers: [...])
        mcp = data["mcp"]
        if "servers" in mcp and "mcp_servers" not in data:
            data = {**data, "mcp_servers": mcp.get("servers", [])}

    try:
        return ToolingConfig.model_validate(data)
    except Exception as e:
        raise ToolingConfigError(f"Invalid tooling config: {e}") from e


def tooling_snapshot(cfg: ToolingConfig) -> Dict[str, Any]:
    """A small, stable summary for logs/health endpoints."""
    enabled_servers = [s for s in cfg.mcp_servers if s.enabled]
    return {
        "enabled": cfg.enabled,
        "mcp_server_count": len(cfg.mcp_servers),
        "mcp_enabled_server_count": len(enabled_servers),
        "mcp_servers": [
            {
                "name": s.name,
                "enabled": s.enabled,
                "url": s.url,
                "tool_name_prefix": s.tool_name_prefix,
                "timeout_s": s.timeout_s,
                "header_keys": sorted(list((s.headers or {}).keys())),
                "allow_tools": s.allow_tools,
                "deny_tools": s.deny_tools,
            }
            for s in cfg.mcp_servers
        ],
    }

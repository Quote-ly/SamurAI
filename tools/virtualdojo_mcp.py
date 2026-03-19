"""VirtualDojo MCP client — OAuth SSO + dynamic CRM tool calling."""

import hashlib
import base64
import secrets
import os
import time
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Token store — per-user tokens keyed by Teams user ID
# In production, swap this for Redis or a database.
# ---------------------------------------------------------------------------
_token_store: dict[str, dict] = {}
# Pending OAuth flows keyed by state param
_pending_auth: dict[str, dict] = {}
# Registered client credentials (populated on first use)
_client_creds: dict | None = None

MCP_URL = os.environ.get("VIRTUALDOJO_MCP_URL", "https://dev.virtualdojo.com/mcp/v1")
BOT_CALLBACK_URL = os.environ.get(
    "BOT_CALLBACK_URL",
    "https://samurai-bot-1019610148219.us-central1.run.app/api/oauth/callback",
)


# ---------------------------------------------------------------------------
# OAuth helpers
# ---------------------------------------------------------------------------

def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


async def _ensure_client_registered() -> dict:
    """Register this bot as an OAuth client with the MCP server (once)."""
    global _client_creds
    if _client_creds:
        return _client_creds

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{MCP_URL}/oauth/register",
            json={
                "client_name": "SamurAI Teams Bot",
                "redirect_uris": [BOT_CALLBACK_URL],
            },
        )
        resp.raise_for_status()
        _client_creds = resp.json()
        return _client_creds


def get_login_url(user_id: str) -> str | None:
    """Build the OAuth authorize URL for a user. Returns URL or None if already authed."""
    if user_id in _token_store and _is_token_valid(user_id):
        return None

    # This will be called from an async context, but we need sync for the tool.
    # The actual registration happens in start_oauth_flow (async).
    return f"_pending:{user_id}"


async def start_oauth_flow(user_id: str) -> str:
    """Start the OAuth flow and return the authorize URL."""
    creds = await _ensure_client_registered()
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)

    _pending_auth[state] = {
        "user_id": user_id,
        "code_verifier": verifier,
        "client_id": creds["client_id"],
    }

    params = {
        "client_id": creds["client_id"],
        "redirect_uri": BOT_CALLBACK_URL,
        "response_type": "code",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "scope": "tools",
        "state": state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{MCP_URL}/oauth/authorize?{query}"


async def exchange_code(code: str, state: str) -> dict | None:
    """Exchange an OAuth authorization code for tokens."""
    flow = _pending_auth.pop(state, None)
    if not flow:
        return None

    creds = await _ensure_client_registered()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{MCP_URL}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": BOT_CALLBACK_URL,
                "client_id": creds["client_id"],
                "code_verifier": flow["code_verifier"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        tokens = resp.json()

    _token_store[flow["user_id"]] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens.get("refresh_token"),
        "expires_at": time.time() + tokens.get("expires_in", 1800),
    }
    return tokens


def _is_token_valid(user_id: str) -> bool:
    """Check if a user's token is still valid (with 60s buffer)."""
    t = _token_store.get(user_id)
    if not t:
        return False
    return t["expires_at"] > time.time() + 60


async def _get_access_token(user_id: str) -> str | None:
    """Get a valid access token for the user, refreshing if needed."""
    t = _token_store.get(user_id)
    if not t:
        return None

    if _is_token_valid(user_id):
        return t["access_token"]

    # Try refresh
    if not t.get("refresh_token"):
        del _token_store[user_id]
        return None

    creds = await _ensure_client_registered()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{MCP_URL}/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": t["refresh_token"],
                    "client_id": creds["client_id"],
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            tokens = resp.json()

        _token_store[user_id] = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token", t["refresh_token"]),
            "expires_at": time.time() + tokens.get("expires_in", 1800),
        }
        return tokens["access_token"]
    except Exception:
        del _token_store[user_id]
        return None


def is_user_authenticated(user_id: str) -> bool:
    """Check if a user has stored tokens."""
    return user_id in _token_store


# ---------------------------------------------------------------------------
# MCP tool discovery and execution
# ---------------------------------------------------------------------------

async def list_mcp_tools(user_id: str) -> list[dict]:
    """List all MCP tools available to the authenticated user."""
    token = await _get_access_token(user_id)
    if not token:
        return []

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{MCP_URL}/tools/list",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("tools", [])


async def call_mcp_tool(user_id: str, tool_name: str, arguments: dict) -> dict:
    """Call an MCP tool on behalf of the authenticated user."""
    token = await _get_access_token(user_id)
    if not token:
        return {"error": "Not authenticated. Please sign in to VirtualDojo first."}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{MCP_URL}/tools/call",
            json={"name": tool_name, "arguments": arguments},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# LangGraph tool wrappers
# ---------------------------------------------------------------------------

def _make_crm_tool_description() -> str:
    """Generic description for the CRM query tool."""
    return (
        "Query the VirtualDojo CRM system. Use this to search contacts, accounts, "
        "opportunities, quotes, and other CRM records. The user must be authenticated "
        "to VirtualDojo first. If not authenticated, tell them to sign in."
    )


class VirtualDojoQueryInput(BaseModel):
    tool_name: str = Field(description="The MCP tool name to call (e.g., 'search_records', 'list_objects', 'describe_object', 'create_record')")
    arguments: str = Field(description='JSON string of arguments to pass to the tool (e.g., \'{"object_type": "contacts", "limit": 10}\')')


def create_virtualdojo_tool(user_id: str) -> StructuredTool:
    """Create a LangGraph-compatible tool that calls VirtualDojo MCP tools for a specific user."""

    async def _call_virtualdojo(tool_name: str, arguments: str) -> str:
        import json
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return f"Invalid JSON arguments: {arguments}"

        if not is_user_authenticated(user_id):
            return (
                "You are not signed in to VirtualDojo. "
                "Please click the sign-in link I'll provide to connect your CRM account."
            )

        result = await call_mcp_tool(user_id, tool_name, args)
        if "error" in result:
            return f"Error: {result['error']}"

        # MCP returns content as a list of content blocks
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            return "\n".join(texts) if texts else str(result)
        return str(content)

    return StructuredTool.from_function(
        coroutine=_call_virtualdojo,
        name="virtualdojo_crm",
        description=(
            "Query the VirtualDojo CRM system. Use this for anything related to "
            "CRM data: contacts, accounts, opportunities, quotes, compliance records, etc. "
            "Common tool_name values: 'search_records', 'list_objects', 'describe_object', "
            "'create_record', 'update_record', 'get_record'. "
            "Pass arguments as a JSON string."
        ),
        args_schema=VirtualDojoQueryInput,
    )


class VirtualDojoListToolsInput(BaseModel):
    pass


def create_virtualdojo_list_tools(user_id: str) -> StructuredTool:
    """Create a tool that lists available VirtualDojo CRM tools for the user."""

    async def _list_tools() -> str:
        if not is_user_authenticated(user_id):
            return "You are not signed in to VirtualDojo. Please sign in first."

        tools = await list_mcp_tools(user_id)
        if not tools:
            return "No CRM tools available. You may not have the right permissions."

        lines = []
        for t in tools:
            lines.append(f"- **{t['name']}**: {t.get('description', 'No description')}")
        return f"Available VirtualDojo CRM tools ({len(tools)}):\n" + "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=_list_tools,
        name="virtualdojo_list_tools",
        description="List all available VirtualDojo CRM tools for the authenticated user.",
        args_schema=VirtualDojoListToolsInput,
    )

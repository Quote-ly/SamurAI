"""Tests for app.py — aiohttp server and Bot Framework handlers."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web


@pytest.fixture
def patched_app():
    """Import app.py with Bot Framework adapter and agent mocked."""
    with (
        patch("langchain_google_vertexai.ChatVertexAI", MagicMock()),
        patch("botbuilder.core.BotFrameworkAdapter") as mock_adapter_cls,
    ):
        mock_adapter = MagicMock()
        mock_adapter.process_activity = AsyncMock()
        mock_adapter_cls.return_value = mock_adapter

        import app as app_module
        importlib.reload(app_module)
        app_module.adapter = mock_adapter
        yield app_module


@pytest.fixture
async def client(patched_app, aiohttp_client):
    return await aiohttp_client(patched_app.app)


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status == 200
    text = await resp.text()
    assert text == "ok"


@pytest.mark.asyncio
async def test_messages_returns_415_for_non_json(client):
    resp = await client.post(
        "/api/messages",
        data="not json",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status == 415


@pytest.mark.asyncio
async def test_messages_returns_200_for_valid_json(client, patched_app):
    resp = await client.post(
        "/api/messages",
        json={"type": "message", "text": "hello"},
        headers={"Content-Type": "application/json"},
    )
    assert resp.status == 200
    patched_app.adapter.process_activity.assert_called_once()


# --- Unit tests for handler functions ---


@pytest.mark.asyncio
async def test_on_message_calls_run_agent(patched_app):
    ctx = MagicMock()
    ctx.activity.text = "show logs"
    ctx.send_activity = AsyncMock()

    with patch.object(patched_app, "run_agent", new_callable=AsyncMock, return_value="here are logs"):
        await patched_app.on_message(ctx)

    # Should have sent typing + response
    assert ctx.send_activity.call_count == 2


@pytest.mark.asyncio
async def test_on_message_sends_typing_indicator(patched_app):
    ctx = MagicMock()
    ctx.activity.text = "hi"
    ctx.send_activity = AsyncMock()

    with patch.object(patched_app, "run_agent", new_callable=AsyncMock, return_value="hey"):
        await patched_app.on_message(ctx)

    first_call = ctx.send_activity.call_args_list[0]
    activity = first_call[0][0]
    assert activity.type == "typing"


@pytest.mark.asyncio
async def test_on_message_ignores_empty_text(patched_app):
    ctx = MagicMock()
    ctx.activity.text = None
    ctx.send_activity = AsyncMock()

    with patch.object(patched_app, "run_agent", new_callable=AsyncMock) as mock_agent:
        await patched_app.on_message(ctx)

    mock_agent.assert_not_called()
    ctx.send_activity.assert_not_called()


@pytest.mark.asyncio
async def test_on_error_sends_apology(patched_app):
    ctx = MagicMock()
    ctx.send_activity = AsyncMock()

    await patched_app.on_error(ctx, Exception("boom"))
    ctx.send_activity.assert_called_once()
    msg = ctx.send_activity.call_args[0][0]
    assert "something went wrong" in msg.lower() or "something went wrong" in str(msg).lower()

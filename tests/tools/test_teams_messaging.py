"""Tests for tools.teams_messaging -- sending messages, looking up and listing team members."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

CONV_ID = "test-conv-123"
USER_EMAIL = "devin@virtualdojo.com"

MEMBER_RECORD = {
    "email": "alice@virtualdojo.com",
    "display_name": "Alice Smith",
    "teams_id": "29:alice-teams-id",
    "service_url": "https://smba.trafficmanager.net/amer/",
    "tenant_id": "tenant-001",
}


# ---------------------------------------------------------------------------
# send_teams_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_send_message_success(mock_get_store):
    import tools.teams_messaging as mod

    mock_store = AsyncMock()
    mock_store.get_team_member.return_value = MEMBER_RECORD
    mock_get_store.return_value = mock_store

    # Build a fake adapter whose create_conversation calls the callback
    mock_adapter = AsyncMock()

    async def fake_create_conversation(conv_ref, callback, conv_params):
        # Simulate the SDK calling back with a TurnContext
        fake_turn = AsyncMock()
        await callback(fake_turn)

    mock_adapter.create_conversation.side_effect = fake_create_conversation

    original_adapter = mod._adapter
    original_app_id = mod._app_id
    mod._adapter = mock_adapter
    mod._app_id = "bot-app-id"
    try:
        from tools.teams_messaging import send_teams_message

        result = await send_teams_message.ainvoke(
            {
                "recipient_email": "alice@virtualdojo.com",
                "message": "Hey Alice, please review PR #42.",
                "conversation_id": CONV_ID,
                "user_email": USER_EMAIL,
            }
        )

        assert "Alice Smith" in result
        assert "alice@virtualdojo.com" in result
        mock_adapter.create_conversation.assert_awaited_once()
    finally:
        mod._adapter = original_adapter
        mod._app_id = original_app_id


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_send_message_unknown_member(mock_get_store):
    import tools.teams_messaging as mod

    mock_store = AsyncMock()
    mock_store.get_team_member.return_value = None
    mock_get_store.return_value = mock_store

    original_adapter = mod._adapter
    mod._adapter = MagicMock()  # adapter is set, so we pass the init check
    try:
        from tools.teams_messaging import send_teams_message

        result = await send_teams_message.ainvoke(
            {
                "recipient_email": "nobody@example.com",
                "message": "Hello?",
                "conversation_id": CONV_ID,
                "user_email": USER_EMAIL,
            }
        )

        assert "not found" in result
        assert "nobody@example.com" in result
    finally:
        mod._adapter = original_adapter


@pytest.mark.asyncio
async def test_send_message_adapter_not_initialized():
    import tools.teams_messaging as mod

    original_adapter = mod._adapter
    mod._adapter = None
    try:
        from tools.teams_messaging import send_teams_message

        result = await send_teams_message.ainvoke(
            {
                "recipient_email": "alice@virtualdojo.com",
                "message": "Hello",
                "conversation_id": CONV_ID,
                "user_email": USER_EMAIL,
            }
        )

        assert "not initialized" in result
    finally:
        mod._adapter = original_adapter


# ---------------------------------------------------------------------------
# lookup_team_member
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_lookup_member_found(mock_get_store):
    from tools.teams_messaging import lookup_team_member

    mock_store = AsyncMock()
    mock_store.get_team_member.return_value = MEMBER_RECORD
    mock_get_store.return_value = mock_store

    result = await lookup_team_member.ainvoke(
        {
            "email": "alice@virtualdojo.com",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "Alice Smith" in result
    assert "alice@virtualdojo.com" in result
    assert "29:alice-teams-id" in result


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_lookup_member_not_found(mock_get_store):
    from tools.teams_messaging import lookup_team_member

    mock_store = AsyncMock()
    mock_store.get_team_member.return_value = None
    mock_get_store.return_value = mock_store

    result = await lookup_team_member.ainvoke(
        {
            "email": "nobody@example.com",
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "No team member found" in result
    assert "nobody@example.com" in result


# ---------------------------------------------------------------------------
# list_team_members
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_list_members_empty(mock_get_store):
    from tools.teams_messaging import list_team_members

    mock_store = AsyncMock()
    mock_store.list_team_members.return_value = []
    mock_get_store.return_value = mock_store

    result = await list_team_members.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "No team members" in result


@pytest.mark.asyncio
@patch("task_store.get_task_store", new_callable=AsyncMock)
async def test_list_members_populated(mock_get_store):
    from tools.teams_messaging import list_team_members

    mock_store = AsyncMock()
    mock_store.list_team_members.return_value = [
        {"display_name": "Alice Smith", "email": "alice@virtualdojo.com"},
        {"display_name": "Bob Jones", "email": "bob@virtualdojo.com"},
    ]
    mock_get_store.return_value = mock_store

    result = await list_team_members.ainvoke(
        {
            "conversation_id": CONV_ID,
            "user_email": USER_EMAIL,
        }
    )

    assert "Team Roster" in result
    assert "2 members" in result
    assert "Alice Smith" in result
    assert "Bob Jones" in result
    assert "alice@virtualdojo.com" in result
    assert "bob@virtualdojo.com" in result

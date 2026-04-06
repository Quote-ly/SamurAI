"""Tests for cards.actions — Action.Submit callback handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cards.actions import (
    handle_card_action,
    handle_schedule_date_reply,
    is_awaiting_schedule_date,
    store_card_activity_id,
    _card_activity_ids,
    _awaiting_schedule_date,
)


@pytest.fixture(autouse=True)
def clear_state():
    """Clear stored activity IDs and schedule flags before each test."""
    _card_activity_ids.clear()
    _awaiting_schedule_date.clear()
    yield
    _card_activity_ids.clear()
    _awaiting_schedule_date.clear()


def _make_turn_context():
    ctx = MagicMock()
    ctx.send_activity = AsyncMock(return_value=MagicMock(id="new-activity-id"))
    ctx.update_activity = AsyncMock()
    return ctx


# --- store_card_activity_id ---


def test_store_and_retrieve_activity_id():
    store_card_activity_id("conv-1", "act-1")
    assert _card_activity_ids["conv-1"] == "act-1"


# --- handle_card_action: approve ---


@pytest.mark.asyncio
@patch("tools.social_media.social_publish_post")
async def test_approve_calls_publish(mock_publish):
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Test post",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }
    mock_publish.invoke = MagicMock(return_value="Post published!")

    ctx = _make_turn_context()
    await handle_card_action(
        ctx, {"action": "social_approve", "conversation_id": "conv-1"}
    )

    mock_publish.invoke.assert_called_once_with(
        {
            "conversation_id": "conv-1",
            "user_email": "devin@virtualdojo.com",
        }
    )
    _pending_posts.pop("conv-1", None)


@pytest.mark.asyncio
async def test_approve_no_draft_sends_error():
    from tools.social_media import _pending_posts

    _pending_posts.pop("conv-1", None)

    ctx = _make_turn_context()
    await handle_card_action(
        ctx, {"action": "social_approve", "conversation_id": "conv-1"}
    )

    ctx.send_activity.assert_called()
    call_args = ctx.send_activity.call_args[0][0]
    assert "No pending post" in call_args.text


@pytest.mark.asyncio
@patch("tools.social_media.social_publish_post")
async def test_approve_updates_original_card(mock_publish):
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Test",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }
    mock_publish.invoke = MagicMock(return_value="Published!")
    store_card_activity_id("conv-1", "original-card-id")

    ctx = _make_turn_context()
    await handle_card_action(
        ctx, {"action": "social_approve", "conversation_id": "conv-1"}
    )

    # Should have tried to update the original card
    ctx.update_activity.assert_called_once()
    updated = ctx.update_activity.call_args[0][0]
    assert updated.id == "original-card-id"
    assert len(updated.attachments) == 1

    _pending_posts.pop("conv-1", None)


# --- handle_card_action: schedule ---


@pytest.mark.asyncio
@patch("cards.actions._fetch_scheduled_calendar", new_callable=AsyncMock)
async def test_show_calendar_sets_awaiting_flag(mock_fetch):
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Test",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }
    mock_fetch.return_value = "**Scheduled Posts**\nNo upcoming posts."

    ctx = _make_turn_context()
    await handle_card_action(
        ctx, {"action": "social_show_calendar", "conversation_id": "conv-1"}
    )

    assert is_awaiting_schedule_date("conv-1")
    ctx.send_activity.assert_called()
    call_args = ctx.send_activity.call_args[0][0]
    assert "When would you like to schedule" in call_args.text
    _pending_posts.pop("conv-1", None)


@pytest.mark.asyncio
@patch("tools.social_media.social_schedule_post")
async def test_schedule_date_reply_calls_schedule(mock_schedule):
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Scheduled post",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }
    mock_schedule.invoke = MagicMock(return_value="Scheduled!")
    _awaiting_schedule_date["conv-1"] = True

    ctx = _make_turn_context()
    await handle_schedule_date_reply(ctx, "conv-1", "2026-03-25T09:00:00Z")

    mock_schedule.invoke.assert_called_once_with(
        {
            "scheduled_date": "2026-03-25T09:00:00Z",
            "conversation_id": "conv-1",
            "user_email": "devin@virtualdojo.com",
        }
    )
    assert not is_awaiting_schedule_date("conv-1")
    _pending_posts.pop("conv-1", None)


# --- handle_card_action: reject ---


@pytest.mark.asyncio
async def test_reject_clears_draft():
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Test",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }

    ctx = _make_turn_context()
    await handle_card_action(
        ctx, {"action": "social_reject", "conversation_id": "conv-1"}
    )

    assert "conv-1" not in _pending_posts


@pytest.mark.asyncio
async def test_reject_updates_card_to_cancelled():
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Test",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }
    store_card_activity_id("conv-1", "card-id-1")

    ctx = _make_turn_context()
    await handle_card_action(
        ctx, {"action": "social_reject", "conversation_id": "conv-1"}
    )

    ctx.update_activity.assert_called_once()
    updated = ctx.update_activity.call_args[0][0]
    assert updated.id == "card-id-1"


# --- handle_card_action: unknown ---


@pytest.mark.asyncio
async def test_unknown_action_sends_error():
    ctx = _make_turn_context()
    await handle_card_action(ctx, {"action": "something_weird"})

    ctx.send_activity.assert_called()
    call_args = ctx.send_activity.call_args[0][0]
    assert "Unknown action" in call_args.text


# --- fallback when update_activity fails ---


@pytest.mark.asyncio
@patch("tools.social_media.social_publish_post")
async def test_approve_fallback_when_update_fails(mock_publish):
    from tools.social_media import _pending_posts

    _pending_posts["conv-1"] = {
        "text": "Test",
        "platforms": ["linkedin"],
        "user_email": "devin@virtualdojo.com",
        "image_url": "",
    }
    mock_publish.invoke = MagicMock(return_value="Published!")
    store_card_activity_id("conv-1", "card-id-1")

    ctx = _make_turn_context()
    ctx.update_activity = AsyncMock(side_effect=Exception("Not allowed"))

    await handle_card_action(
        ctx, {"action": "social_approve", "conversation_id": "conv-1"}
    )

    # Should fall back to send_activity
    ctx.send_activity.assert_called()
    _pending_posts.pop("conv-1", None)

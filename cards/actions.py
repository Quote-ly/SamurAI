"""Handle Adaptive Card Action.Submit callbacks from Teams."""

import os
from datetime import datetime

import httpx
from botbuilder.core import CardFactory, TurnContext
from botbuilder.schema import Activity

from cards.social import (
    build_social_published_card,
    build_social_rejected_card,
)

# Stores activity IDs of sent preview cards so we can update them later.
# Key: conversation_id, Value: activity_id
_card_activity_ids: dict[str, str] = {}

# Tracks conversations waiting for a schedule date reply.
# Key: conversation_id, Value: True
_awaiting_schedule_date: dict[str, bool] = {}


def store_card_activity_id(conversation_id: str, activity_id: str):
    """Store the activity ID of a sent card for later updates."""
    _card_activity_ids[conversation_id] = activity_id


def is_awaiting_schedule_date(conversation_id: str) -> bool:
    """Check if the bot is waiting for a schedule date from this conversation."""
    return _awaiting_schedule_date.get(conversation_id, False)


def clear_awaiting_schedule_date(conversation_id: str):
    """Clear the awaiting schedule date flag."""
    _awaiting_schedule_date.pop(conversation_id, None)


async def handle_card_action(turn_context: TurnContext, value: dict):
    """Dispatch an Action.Submit callback from an Adaptive Card.

    Args:
        turn_context: The bot TurnContext.
        value: The action data dict from activity.value.
    """
    action = value.get("action", "")
    conversation_id = value.get("conversation_id", "")

    if action == "social_approve":
        await _handle_approve(turn_context, conversation_id)
    elif action == "social_show_calendar":
        await _handle_show_calendar(turn_context, conversation_id)
    elif action == "social_schedule":
        schedule_date = value.get("schedule_date", "")
        await _handle_schedule(turn_context, conversation_id, schedule_date)
    elif action == "social_reject":
        await _handle_reject(turn_context, conversation_id)
    else:
        await turn_context.send_activity(
            Activity(type="message", text=f"Unknown action: {action}")
        )


async def handle_schedule_date_reply(
    turn_context: TurnContext, conversation_id: str, user_message: str
):
    """Handle a text reply with a schedule date after the calendar was shown."""
    from tools.social_media import social_schedule_post, _pending_posts

    clear_awaiting_schedule_date(conversation_id)

    draft = _pending_posts.get(conversation_id)
    if not draft:
        await turn_context.send_activity(
            Activity(
                type="message",
                text="No pending post found. The draft may have expired.",
            )
        )
        return

    user_email = draft.get("user_email", "")
    result = social_schedule_post.invoke(
        {
            "scheduled_date": user_message.strip(),
            "conversation_id": conversation_id,
            "user_email": user_email,
        }
    )

    updated_card = build_social_published_card(
        text=draft["text"],
        platforms=draft["platforms"],
        post_id=conversation_id,
        image_url=draft.get("image_url", ""),
    )
    await _update_or_send_card(turn_context, conversation_id, updated_card, result)


async def _handle_approve(turn_context: TurnContext, conversation_id: str):
    """Publish the pending draft and update the preview card."""
    from tools.social_media import social_publish_post, _pending_posts

    draft = _pending_posts.get(conversation_id)
    if not draft:
        await turn_context.send_activity(
            Activity(
                type="message",
                text="No pending post found. The draft may have expired.",
            )
        )
        return

    user_email = draft.get("user_email", "")
    result = social_publish_post.invoke(
        {
            "conversation_id": conversation_id,
            "user_email": user_email,
        }
    )

    updated_card = build_social_published_card(
        text=draft["text"],
        platforms=draft["platforms"],
        post_id=conversation_id,
        image_url=draft.get("image_url", ""),
    )
    await _update_or_send_card(turn_context, conversation_id, updated_card, result)


async def _handle_show_calendar(turn_context: TurnContext, conversation_id: str):
    """Fetch scheduled posts from Ayrshare and show them, then ask for a date."""
    from tools.social_media import _pending_posts

    draft = _pending_posts.get(conversation_id)
    if not draft:
        await turn_context.send_activity(
            Activity(
                type="message",
                text="No pending post found. The draft may have expired.",
            )
        )
        return

    # Fetch scheduled posts from Ayrshare
    calendar_text = await _fetch_scheduled_calendar()

    # Mark this conversation as awaiting a schedule date
    _awaiting_schedule_date[conversation_id] = True

    await turn_context.send_activity(
        Activity(
            type="message",
            text=(
                f"{calendar_text}\n\n"
                "**When would you like to schedule your post?**\n"
                "Reply with a date and time "
                "(e.g. `2026-03-25T09:00:00Z` or `Tuesday at 9am ET`)."
            ),
        )
    )


async def _fetch_scheduled_calendar() -> str:
    """Fetch scheduled posts from Ayrshare and format as a calendar view."""
    try:
        api_key = os.environ.get("AYRSHARE_API_KEY", "")
        if not api_key:
            return "**Scheduled Posts**\n(Could not fetch - no API key configured)"

        resp = httpx.get(
            "https://api.ayrshare.com/api/history",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            params={"lastDays": 30},
            timeout=15,
        )
        resp.raise_for_status()
        posts = resp.json()
    except Exception:
        return "**Scheduled Posts**\n(Could not fetch scheduled posts)"

    scheduled = [
        p for p in posts if p.get("status") in ("scheduled", "pending", "awaiting")
    ]

    if not scheduled:
        return (
            "**Scheduled Posts**\nNo upcoming scheduled posts. You can pick any date!"
        )

    scheduled.sort(key=lambda p: p.get("scheduleDate", ""))

    lines = ["**Scheduled Posts**\n"]
    current_date = None
    for post in scheduled:
        sched = post.get("scheduleDate", "")
        try:
            dt = datetime.fromisoformat(sched.replace("Z", "+00:00"))
            date_str = dt.strftime("%a %b %d")
            time_str = dt.strftime("%I:%M %p UTC")
        except (ValueError, AttributeError):
            date_str = "Unknown"
            time_str = ""

        if date_str != current_date:
            current_date = date_str
            lines.append(f"**{date_str}**")

        platforms = ", ".join(post.get("platforms", []))
        text_preview = post.get("post", "")[:50]
        if len(post.get("post", "")) > 50:
            text_preview += "..."
        lines.append(f'  {time_str} - {platforms}: "{text_preview}"')

    return "\n".join(lines)


async def _handle_schedule(
    turn_context: TurnContext,
    conversation_id: str,
    schedule_date: str,
):
    """Schedule the pending draft and update the preview card."""
    from tools.social_media import social_schedule_post, _pending_posts

    if not schedule_date:
        await turn_context.send_activity(
            Activity(type="message", text="Please provide a schedule date.")
        )
        return

    draft = _pending_posts.get(conversation_id)
    if not draft:
        await turn_context.send_activity(
            Activity(
                type="message",
                text="No pending post found. The draft may have expired.",
            )
        )
        return

    user_email = draft.get("user_email", "")
    result = social_schedule_post.invoke(
        {
            "scheduled_date": schedule_date,
            "conversation_id": conversation_id,
            "user_email": user_email,
        }
    )

    updated_card = build_social_published_card(
        text=draft["text"],
        platforms=draft["platforms"],
        post_id=conversation_id,
        image_url=draft.get("image_url", ""),
    )
    await _update_or_send_card(turn_context, conversation_id, updated_card, result)


async def _handle_reject(turn_context: TurnContext, conversation_id: str):
    """Cancel the pending draft and update the preview card."""
    from tools.social_media import _pending_posts

    draft = _pending_posts.pop(conversation_id, None)
    if not draft:
        await turn_context.send_activity(
            Activity(type="message", text="No pending post found.")
        )
        return

    updated_card = build_social_rejected_card(
        text=draft["text"],
        platforms=draft["platforms"],
        image_url=draft.get("image_url", ""),
    )
    await _update_or_send_card(
        turn_context,
        conversation_id,
        updated_card,
        "Post cancelled. Let me know if you'd like to start over or make changes.",
    )


async def _update_or_send_card(
    turn_context: TurnContext,
    conversation_id: str,
    card: dict,
    text_response: str,
):
    """Try to update the original card; fall back to sending a new one."""
    attachment = CardFactory.adaptive_card(card)
    original_id = _card_activity_ids.pop(conversation_id, None)

    if original_id:
        try:
            updated = Activity(
                type="message",
                id=original_id,
                attachments=[attachment],
            )
            await turn_context.update_activity(updated)
            await turn_context.send_activity(
                Activity(type="message", text=text_response)
            )
            return
        except Exception:
            pass  # Fall through to sending new card

    # Fallback: send new card + text
    await turn_context.send_activity(
        Activity(type="message", attachments=[attachment], text=text_response)
    )

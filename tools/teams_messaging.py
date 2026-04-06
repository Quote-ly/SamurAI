"""Agent tools for sending proactive Teams messages to team members."""

import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Set by scheduler.init_scheduler() at startup
_adapter = None
_app_id: str = ""


@tool
async def send_teams_message(
    recipient_email: str,
    message: str,
    conversation_id: str,
    user_email: str,
) -> str:
    """Send a 1:1 Teams message to a team member.

    The recipient must be in the team roster (anyone who has previously
    messaged the bot, or was discovered via team membership).

    Args:
        recipient_email: The email address of the person to message.
        message: The message text to send.
        conversation_id: The current conversation ID (from context brackets).
        user_email: The sender's email (from context brackets).
    """
    from botbuilder.schema import (
        Activity,
        ChannelAccount,
        ConversationAccount,
        ConversationParameters,
        ConversationReference,
    )
    from botbuilder.core import TurnContext

    from task_store import get_task_store

    if not _adapter:
        return "Error: Teams messaging is not initialized. The bot is still starting up."

    store = await get_task_store()
    member = await store.get_team_member(recipient_email)
    if not member:
        return (
            f"Team member '{recipient_email}' not found in the roster. "
            "They need to message the bot first, or the bot needs to be "
            "installed in a shared team channel to discover members."
        )

    try:
        # Build a ConversationReference to create a new 1:1 conversation
        conv_params = ConversationParameters(
            is_group=False,
            bot=ChannelAccount(id=_app_id),
            members=[
                ChannelAccount(
                    id=member["teams_id"],
                    name=member["display_name"],
                )
            ],
            tenant_id=member.get("tenant_id", ""),
        )

        sent_ok = False

        async def _send(turn_context: TurnContext):
            nonlocal sent_ok
            await turn_context.send_activity(
                Activity(type="message", text=message)
            )
            sent_ok = True

        await _adapter.create_conversation(
            ConversationReference(
                channel_id="msteams",
                service_url=member["service_url"],
                bot=ChannelAccount(id=_app_id),
            ),
            _send,
            conv_params,
        )

        if sent_ok:
            logger.info(
                "Sent Teams message to %s from %s", recipient_email, user_email
            )
            return f"Message sent to {member['display_name']} ({recipient_email})."
        return f"Message delivery to {recipient_email} could not be confirmed."

    except Exception as e:
        logger.error("Failed to send Teams message to %s: %s", recipient_email, e)
        return f"Failed to send message to {recipient_email}: {e}"


@tool
async def lookup_team_member(
    email: str, conversation_id: str, user_email: str
) -> str:
    """Look up a team member by email to check if they are in the roster.

    Args:
        email: The email address to look up.
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
    """
    from task_store import get_task_store

    store = await get_task_store()
    member = await store.get_team_member(email)
    if not member:
        return f"No team member found with email '{email}'."
    return (
        f"Found: {member['display_name']} ({member['email']})\n"
        f"Teams ID: {member['teams_id']}"
    )


@tool
async def list_team_members(conversation_id: str, user_email: str) -> str:
    """List all known team members in the roster.

    Args:
        conversation_id: The current conversation ID (from context brackets).
        user_email: The user's email (from context brackets).
    """
    from task_store import get_task_store

    store = await get_task_store()
    members = await store.list_team_members()
    if not members:
        return "No team members in the roster yet. Members are added automatically when they message the bot."

    lines = [f"**Team Roster** ({len(members)} members)\n"]
    for m in members:
        lines.append(f"- {m['display_name']} ({m['email']})")
    return "\n".join(lines)


TEAMS_MESSAGING_TOOLS = [
    send_teams_message,
    lookup_team_member,
    list_team_members,
]

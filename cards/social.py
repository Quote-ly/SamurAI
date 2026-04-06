"""Adaptive Card builders for social media previews, confirmations, and scheduled posts."""

from datetime import datetime


def build_social_preview_card(
    text: str,
    platforms: list[str],
    conversation_id: str,
    image_url: str = "",
    scheduled_date: str = "",
) -> dict:
    """Build an Adaptive Card for a social media post preview with Approve/Reject buttons."""
    body = []

    # Header
    body.append(
        {
            "type": "TextBlock",
            "text": "Social Media Post — Ready for Review",
            "weight": "bolder",
            "size": "medium",
            "wrap": True,
        }
    )

    # Image (if available)
    if image_url:
        body.append(
            {
                "type": "Image",
                "url": image_url,
                "size": "large",
                "altText": "Post preview image",
            }
        )

    # Post content
    body.append(
        {
            "type": "TextBlock",
            "text": text,
            "wrap": True,
        }
    )

    # Separator
    body.append(
        {
            "type": "TextBlock",
            "text": " ",
            "spacing": "small",
            "separator": True,
        }
    )

    # Metadata as FactSet
    facts = [
        {"title": "Platforms", "value": ", ".join(platforms)},
    ]
    if scheduled_date:
        facts.append({"title": "Scheduled", "value": scheduled_date})
    else:
        facts.append({"title": "Timing", "value": "Immediate (on approval)"})
    if image_url:
        facts.append({"title": "Image", "value": "AI-generated image attached"})
    facts.append({"title": "Status", "value": "Pending Approval"})

    body.append({"type": "FactSet", "facts": facts})

    # Action buttons
    actions = [
        {
            "type": "Action.Submit",
            "title": "Approve",
            "data": {
                "action": "social_approve",
                "conversation_id": conversation_id,
            },
        },
        {
            "type": "Action.Submit",
            "title": "Request Changes",
            "data": {
                "action": "social_reject",
                "conversation_id": conversation_id,
            },
        },
    ]

    if not scheduled_date:
        # Schedule button shows calendar of existing posts, then asks for date
        actions.insert(
            1,
            {
                "type": "Action.Submit",
                "title": "Schedule",
                "data": {
                    "action": "social_show_calendar",
                    "conversation_id": conversation_id,
                },
            },
        )

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": body,
        "actions": actions,
    }


def build_social_published_card(
    text: str,
    platforms: list[str],
    post_id: str = "",
    image_url: str = "",
) -> dict:
    """Build a confirmation card after a post is published (no action buttons)."""
    body = [
        {
            "type": "TextBlock",
            "text": "Post Published",
            "weight": "bolder",
            "size": "medium",
            "color": "good",
            "wrap": True,
        },
    ]

    if image_url:
        body.append(
            {
                "type": "Image",
                "url": image_url,
                "size": "large",
                "altText": "Published post image",
            }
        )

    body.append(
        {
            "type": "TextBlock",
            "text": text,
            "wrap": True,
        }
    )

    facts = [
        {"title": "Platforms", "value": ", ".join(platforms)},
        {"title": "Status", "value": "Published"},
    ]
    if post_id:
        facts.append({"title": "Post ID", "value": post_id})

    body.append({"type": "FactSet", "facts": facts})

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": body,
    }


def build_social_rejected_card(
    text: str,
    platforms: list[str],
    image_url: str = "",
) -> dict:
    """Build a card for a rejected/cancelled post (no action buttons)."""
    body = [
        {
            "type": "TextBlock",
            "text": "Post Cancelled",
            "weight": "bolder",
            "size": "medium",
            "color": "attention",
            "wrap": True,
        },
    ]

    if image_url:
        body.append(
            {
                "type": "Image",
                "url": image_url,
                "size": "medium",
                "altText": "Cancelled post image",
            }
        )

    body.append(
        {
            "type": "TextBlock",
            "text": text,
            "wrap": True,
            "isSubtle": True,
        }
    )

    body.append(
        {
            "type": "FactSet",
            "facts": [
                {"title": "Platforms", "value": ", ".join(platforms)},
                {"title": "Status", "value": "Cancelled"},
            ],
        }
    )

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": body,
    }


def build_scheduled_post_card(
    text: str,
    platforms: list[str],
    scheduled_date: str,
    post_id: str,
) -> dict:
    """Build a card for a single scheduled post (used in carousel)."""
    # Format the date nicely
    time_str = scheduled_date
    try:
        dt = datetime.fromisoformat(scheduled_date.replace("Z", "+00:00"))
        time_str = dt.strftime("%a %b %d, %I:%M %p UTC")
    except (ValueError, AttributeError):
        pass

    # Truncate long post text
    display_text = text[:200] + "..." if len(text) > 200 else text

    return {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": [
            {
                "type": "TextBlock",
                "text": time_str,
                "weight": "bolder",
                "size": "medium",
                "wrap": True,
            },
            {
                "type": "TextBlock",
                "text": display_text,
                "wrap": True,
            },
            {
                "type": "FactSet",
                "facts": [
                    {"title": "Platforms", "value": ", ".join(platforms)},
                    {"title": "Post ID", "value": post_id},
                ],
            },
        ],
    }


def build_scheduled_posts_cards(posts: list[dict]) -> list[dict]:
    """Build a list of Adaptive Cards for scheduled posts (one per post, for carousel)."""
    cards = []
    for post in posts[:10]:  # Teams supports max 10 cards in a carousel
        cards.append(
            build_scheduled_post_card(
                text=post.get("post", "(no text)"),
                platforms=post.get("platforms", []),
                scheduled_date=post.get("scheduleDate", "Unknown"),
                post_id=post.get("id", "N/A"),
            )
        )
    return cards

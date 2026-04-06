"""Social media tools — draft, preview, schedule, and publish posts via Ayrshare."""

import os
from datetime import datetime, timezone

import httpx
from langchain_core.tools import tool

AYRSHARE_BASE = "https://api.ayrshare.com/api"

AUTHORIZED_SOCIAL_USERS = {
    "cyrus@virtualdojo.com",
    "devin@virtualdojo.com",
}

# In-memory stores keyed by conversation_id
_pending_posts: dict[str, dict] = {}
_pending_images: dict[str, str] = {}  # base64 image data
_pending_cards: dict[str, dict] = {}  # card render data for app.py to pick up


def _ayrshare_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['AYRSHARE_API_KEY']}",
        "Content-Type": "application/json",
    }


def _check_auth(user_email: str) -> str | None:
    """Return error message if user is not authorized, else None."""
    if user_email.lower() not in AUTHORIZED_SOCIAL_USERS:
        return "You are not authorized to use social media tools."
    return None


async def _generate_image_vertex(prompt: str) -> str:
    """Call Vertex AI Gemini to generate an image, return base64 PNG."""
    from google.auth import default as google_auth_default
    from google.auth.transport.requests import Request as GoogleAuthRequest

    credentials, project = google_auth_default()
    credentials.refresh(GoogleAuthRequest())

    project_id = os.environ.get("GCP_PROJECT_ID", project)
    url = (
        f"https://aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_id}/locations/global/"
        f"publishers/google/models/gemini-3.1-flash-image-preview:generateContent"
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }

    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Extract base64 image from Gemini response
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "inlineData" in part:
                return part["inlineData"]["data"]

    raise ValueError("No image returned from Vertex AI")


@tool
def social_generate_image(prompt: str, conversation_id: str, user_email: str) -> str:
    """Generate an image for a social media post using AI.

    Args:
        prompt: Description of the image to generate. Include brand details for best results.
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Can't use run_until_complete on a running loop — dispatch to it and wait
        future = asyncio.run_coroutine_threadsafe(_generate_image_vertex(prompt), loop)
        image_b64 = future.result(timeout=60)
    else:
        image_b64 = asyncio.run(_generate_image_vertex(prompt))

    _pending_images[conversation_id] = image_b64
    return "Image generated successfully. Call social_preview_post to see the full preview with the image."


@tool
def social_preview_post(
    text: str,
    platforms: str,
    conversation_id: str,
    user_email: str,
    media_urls: str = "",
    scheduled_date: str = "",
) -> str:
    """Draft a social media post and show a preview. Does NOT publish.

    Args:
        text: The post content/copy.
        platforms: Comma-separated platform names (e.g. "linkedin,twitter").
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
        media_urls: Optional comma-separated media URLs to attach.
        scheduled_date: Optional ISO 8601 date to schedule (e.g. "2026-03-21T09:00:00Z").
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    platform_list = [p.strip().lower() for p in platforms.split(",") if p.strip()]
    media_list = (
        [u.strip() for u in media_urls.split(",") if u.strip()] if media_urls else []
    )

    # Attach any pending generated image — upload now so we have a URL for the card
    image_b64 = _pending_images.pop(conversation_id, None)
    image_url = ""
    if image_b64:
        try:
            image_url = _upload_image_to_ayrshare(image_b64)
        except Exception:
            pass  # Image upload failed; continue without image in card

    draft = {
        "text": text,
        "platforms": platform_list,
        "media_urls": media_list,
        "scheduled_date": scheduled_date,
        "image_b64": image_b64,
        "image_url": image_url,
        "user_email": user_email,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _pending_posts[conversation_id] = draft

    # Store card render data for app.py to pick up
    _pending_cards[conversation_id] = {
        "card_type": "social_preview",
        "text": text,
        "platforms": platform_list,
        "scheduled_date": scheduled_date,
        "image_url": image_url,
        "conversation_id": conversation_id,
    }

    # Build plain-text fallback (also returned to the LLM)
    preview_lines = [
        "**Post Preview**",
        "",
        f"**Platforms:** {', '.join(platform_list)}",
        "",
        f"**Content:**\n{text}",
    ]
    if scheduled_date:
        preview_lines.append(f"\n**Scheduled for:** {scheduled_date}")
    else:
        preview_lines.append("\n**Timing:** Immediate (on approval)")
    if media_list:
        preview_lines.append(f"\n**Media:** {len(media_list)} attachment(s)")
    if image_url:
        preview_lines.append("\n**Image:** AI-generated image attached")
    preview_lines.append(
        "\n---\n*Use the Approve / Schedule / Request Changes buttons on the card "
        "to take action, or tell me what you'd like to change.*"
    )
    return "\n".join(preview_lines)


def _upload_image_to_ayrshare(image_b64: str) -> str:
    """Upload a base64 image to Ayrshare and return the hosted URL."""
    resp = httpx.post(
        f"{AYRSHARE_BASE}/media/upload",
        headers=_ayrshare_headers(),
        json={"file": f"data:image/png;base64,{image_b64}"},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    url = result.get("url")
    if not url:
        raise ValueError(f"No URL returned from media upload: {result}")
    return url


def _resolve_media_urls(draft: dict) -> list[str]:
    """Build the mediaUrls list, using already-uploaded URL or uploading base64 image."""
    urls = list(draft.get("media_urls") or [])
    # Prefer the already-uploaded URL from preview time
    if draft.get("image_url"):
        urls.append(draft["image_url"])
    elif draft.get("image_b64"):
        uploaded_url = _upload_image_to_ayrshare(draft["image_b64"])
        urls.append(uploaded_url)
    return urls


@tool
def social_publish_post(conversation_id: str, user_email: str) -> str:
    """Publish the pending draft post immediately. Only call after user explicitly approves.

    Args:
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    draft = _pending_posts.get(conversation_id)
    if not draft:
        return "No pending post found. Use social_preview_post first to create a draft."

    media_urls = _resolve_media_urls(draft)

    payload = {
        "post": draft["text"],
        "platforms": draft["platforms"],
    }
    if media_urls:
        payload["mediaUrls"] = media_urls

    resp = httpx.post(
        f"{AYRSHARE_BASE}/post",
        headers=_ayrshare_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    _pending_posts.pop(conversation_id, None)

    post_ids = result.get("postIds", [])
    status = result.get("status", "unknown")
    return (
        f"Post published successfully!\n"
        f"Status: {status}\n"
        f"Post IDs: {', '.join(str(pid) for pid in post_ids)}\n"
        f"ID: {result.get('id', 'N/A')}"
    )


@tool
def social_schedule_post(
    scheduled_date: str, conversation_id: str, user_email: str
) -> str:
    """Schedule the pending draft post for a future time. Only call after user explicitly approves.

    Args:
        scheduled_date: ISO 8601 datetime to schedule the post (e.g. "2026-03-21T09:00:00Z").
        conversation_id: The current conversation ID (provided automatically).
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    draft = _pending_posts.get(conversation_id)
    if not draft:
        return "No pending post found. Use social_preview_post first to create a draft."

    media_urls = _resolve_media_urls(draft)

    payload = {
        "post": draft["text"],
        "platforms": draft["platforms"],
        "scheduleDate": scheduled_date,
    }
    if media_urls:
        payload["mediaUrls"] = media_urls

    resp = httpx.post(
        f"{AYRSHARE_BASE}/post",
        headers=_ayrshare_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    _pending_posts.pop(conversation_id, None)

    return (
        f"Post scheduled successfully!\n"
        f"Scheduled for: {scheduled_date}\n"
        f"ID: {result.get('id', 'N/A')}\n"
        f"Status: {result.get('status', 'unknown')}"
    )


@tool
def social_list_scheduled(
    user_email: str, conversation_id: str = "", days_ahead: int = 30
) -> str:
    """List all scheduled and pending social media posts.

    Args:
        user_email: The user's email address (provided automatically).
        conversation_id: The current conversation ID (provided automatically).
        days_ahead: Number of days ahead to look (default 30).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    resp = httpx.get(
        f"{AYRSHARE_BASE}/history",
        headers=_ayrshare_headers(),
        params={"lastDays": days_ahead},
        timeout=30,
    )
    resp.raise_for_status()
    posts = resp.json()

    if not posts:
        return "No scheduled posts found."

    # Filter to scheduled/pending posts and group by date
    scheduled = []
    for post in posts:
        status = post.get("status", "")
        if status in ("scheduled", "pending", "awaiting"):
            scheduled.append(post)

    if not scheduled:
        return "No scheduled posts found."

    # Sort by schedule date
    scheduled.sort(key=lambda p: p.get("scheduleDate", ""))

    lines = ["**📅 Scheduled Posts**\n"]
    current_date = None
    for post in scheduled:
        sched = post.get("scheduleDate", "")
        try:
            dt = datetime.fromisoformat(sched.replace("Z", "+00:00"))
            date_str = dt.strftime("%a %b %d")
            time_str = dt.strftime("%I:%M %p")
        except (ValueError, AttributeError):
            date_str = "Unknown date"
            time_str = ""

        if date_str != current_date:
            current_date = date_str
            lines.append(f"**{date_str}**")

        platforms = ", ".join(post.get("platforms", []))
        text_preview = post.get("post", "")[:60]
        if len(post.get("post", "")) > 60:
            text_preview += "..."
        post_id = post.get("id", "N/A")
        lines.append(f'  {time_str} — {platforms}: "{text_preview}" [ID: {post_id}]')

    # Store card data for carousel rendering
    if conversation_id:
        _pending_cards[conversation_id] = {
            "card_type": "scheduled_posts",
            "posts": scheduled,
        }

    return "\n".join(lines)


@tool
def social_get_post(post_id: str, user_email: str) -> str:
    """Get details of a specific social media post by ID.

    Args:
        post_id: The Ayrshare post ID.
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    resp = httpx.get(
        f"{AYRSHARE_BASE}/post/{post_id}",
        headers=_ayrshare_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    post = resp.json()

    platforms = ", ".join(post.get("platforms", []))
    status = post.get("status", "unknown")
    text = post.get("post", "(no text)")
    sched = post.get("scheduleDate", "N/A")
    created = post.get("created", "N/A")

    return (
        f"**Post Details**\n"
        f"ID: {post_id}\n"
        f"Status: {status}\n"
        f"Platforms: {platforms}\n"
        f"Scheduled: {sched}\n"
        f"Created: {created}\n"
        f"Content:\n{text}"
    )


@tool
def social_update_post(
    post_id: str,
    user_email: str,
    text: str = "",
    scheduled_date: str = "",
) -> str:
    """Edit a scheduled post's text or scheduled time.

    Args:
        post_id: The Ayrshare post ID to update.
        user_email: The user's email address (provided automatically).
        text: New post text (optional — omit to keep current text).
        scheduled_date: New ISO 8601 schedule time (optional — omit to keep current time).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    payload: dict = {"id": post_id}
    if text:
        payload["post"] = text
    if scheduled_date:
        payload["scheduleDate"] = scheduled_date

    resp = httpx.put(
        f"{AYRSHARE_BASE}/post",
        headers=_ayrshare_headers(),
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    return (
        f"Post updated successfully!\n"
        f"ID: {post_id}\n"
        f"Status: {result.get('status', 'unknown')}"
    )


@tool
def social_delete_post(post_id: str, user_email: str) -> str:
    """Delete a scheduled or published social media post.

    Args:
        post_id: The Ayrshare post ID to delete.
        user_email: The user's email address (provided automatically).
    """
    auth_err = _check_auth(user_email)
    if auth_err:
        return auth_err

    resp = httpx.request(
        "DELETE",
        f"{AYRSHARE_BASE}/post",
        headers=_ayrshare_headers(),
        json={"id": post_id},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    return (
        f"Post deleted.\n"
        f"ID: {post_id}\n"
        f"Status: {result.get('status', 'unknown')}"
    )


# All social tools for easy import
SOCIAL_TOOLS = [
    social_generate_image,
    social_preview_post,
    social_publish_post,
    social_schedule_post,
    social_list_scheduled,
    social_get_post,
    social_update_post,
    social_delete_post,
]

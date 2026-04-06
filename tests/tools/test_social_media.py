"""Tests for tools.social_media — social media posting, scheduling, and image generation."""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from tools.social_media import (
    _pending_posts,
    _pending_images,
    _pending_cards,
)

AUTHORIZED_EMAIL = "devin@virtualdojo.com"
UNAUTHORIZED_EMAIL = "hacker@example.com"
CONV_ID = "test-conv-123"


@pytest.fixture(autouse=True)
def clear_pending():
    """Clear pending posts/images/cards before each test."""
    _pending_posts.clear()
    _pending_images.clear()
    _pending_cards.clear()
    yield
    _pending_posts.clear()
    _pending_images.clear()
    _pending_cards.clear()


# --- Access control ---


def test_unauthorized_user_blocked():
    from tools.social_media import social_preview_post

    result = social_preview_post.invoke(
        {
            "text": "Hello world",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        }
    )
    assert "not authorized" in result


def test_authorized_user_allowed():
    from tools.social_media import social_preview_post

    result = social_preview_post.invoke(
        {
            "text": "Hello world",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert "Preview" in result


def test_auth_case_insensitive():
    from tools.social_media import social_preview_post

    result = social_preview_post.invoke(
        {
            "text": "Hello world",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": "Devin@VirtualDojo.com",
        }
    )
    assert "Preview" in result


# --- social_preview_post ---


def test_preview_post_stores_draft():
    from tools.social_media import social_preview_post

    social_preview_post.invoke(
        {
            "text": "SEWP quoting just got faster",
            "platforms": "linkedin,twitter",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    assert CONV_ID in _pending_posts
    draft = _pending_posts[CONV_ID]
    assert draft["text"] == "SEWP quoting just got faster"
    assert draft["platforms"] == ["linkedin", "twitter"]


def test_preview_post_shows_platforms():
    from tools.social_media import social_preview_post

    result = social_preview_post.invoke(
        {
            "text": "Test post",
            "platforms": "linkedin,twitter",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert "linkedin" in result
    assert "twitter" in result


def test_preview_post_shows_scheduled_date():
    from tools.social_media import social_preview_post

    result = social_preview_post.invoke(
        {
            "text": "Test post",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
            "scheduled_date": "2026-03-21T09:00:00Z",
        }
    )
    assert "2026-03-21T09:00:00Z" in result


@patch(
    "tools.social_media._upload_image_to_ayrshare",
    return_value="https://ayrshare.com/uploaded/img.png",
)
def test_preview_post_attaches_pending_image(mock_upload):
    from tools.social_media import social_preview_post

    _pending_images[CONV_ID] = "fakebase64data"

    result = social_preview_post.invoke(
        {
            "text": "Test post",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert "Image" in result
    # Image should be moved to the draft
    assert CONV_ID not in _pending_images
    assert _pending_posts[CONV_ID]["image_b64"] == "fakebase64data"
    assert (
        _pending_posts[CONV_ID]["image_url"] == "https://ayrshare.com/uploaded/img.png"
    )
    # Card data should be stored
    assert CONV_ID in _pending_cards
    assert (
        _pending_cards[CONV_ID]["image_url"] == "https://ayrshare.com/uploaded/img.png"
    )


def test_preview_post_stores_card_data():
    from tools.social_media import social_preview_post

    social_preview_post.invoke(
        {
            "text": "Card test",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert CONV_ID in _pending_cards
    card = _pending_cards[CONV_ID]
    assert card["card_type"] == "social_preview"
    assert card["text"] == "Card test"
    assert card["platforms"] == ["linkedin"]
    assert card["conversation_id"] == CONV_ID


def test_preview_post_immediate_timing():
    from tools.social_media import social_preview_post

    result = social_preview_post.invoke(
        {
            "text": "Test post",
            "platforms": "linkedin",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert "Immediate" in result


# --- social_publish_post ---


@patch("tools.social_media.httpx.post")
def test_publish_post_sends_to_ayrshare(mock_post):
    from tools.social_media import social_publish_post

    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value={
                "status": "success",
                "postIds": ["post_123"],
                "id": "abc123",
            }
        ),
    )
    mock_post.return_value.raise_for_status = MagicMock()

    _pending_posts[CONV_ID] = {
        "text": "Test post",
        "platforms": ["linkedin"],
        "media_urls": [],
        "image_b64": None,
    }

    result = social_publish_post.invoke(
        {
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    assert "published" in result.lower()
    assert "post_123" in result
    mock_post.assert_called_once()
    # Draft should be cleared
    assert CONV_ID not in _pending_posts


@patch("tools.social_media.httpx.post")
def test_publish_post_includes_image(mock_post):
    from tools.social_media import social_publish_post

    # First call is media upload, second is the post
    upload_resp = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"url": "https://ayrshare.com/uploaded/img.png"}),
    )
    upload_resp.raise_for_status = MagicMock()
    post_resp = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"status": "success", "postIds": [], "id": "x"}),
    )
    post_resp.raise_for_status = MagicMock()
    mock_post.side_effect = [upload_resp, post_resp]

    _pending_posts[CONV_ID] = {
        "text": "Post with image",
        "platforms": ["linkedin"],
        "media_urls": [],
        "image_b64": "base64imagedata",
    }

    social_publish_post.invoke(
        {
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    # Verify upload was called, then post with the uploaded URL
    assert mock_post.call_count == 2
    post_json = mock_post.call_args_list[1][1]["json"]
    assert "https://ayrshare.com/uploaded/img.png" in post_json.get("mediaUrls", [])


def test_publish_post_no_draft():
    from tools.social_media import social_publish_post

    result = social_publish_post.invoke(
        {
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert "No pending post" in result


# --- social_schedule_post ---


@patch("tools.social_media.httpx.post")
def test_schedule_post_includes_date(mock_post):
    from tools.social_media import social_schedule_post

    mock_post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"status": "scheduled", "id": "sched123"}),
    )
    mock_post.return_value.raise_for_status = MagicMock()

    _pending_posts[CONV_ID] = {
        "text": "Scheduled post",
        "platforms": ["linkedin"],
        "media_urls": [],
        "image_b64": None,
    }

    result = social_schedule_post.invoke(
        {
            "scheduled_date": "2026-03-21T09:00:00Z",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    assert "scheduled" in result.lower()
    call_json = mock_post.call_args[1]["json"]
    assert call_json["scheduleDate"] == "2026-03-21T09:00:00Z"
    assert CONV_ID not in _pending_posts


def test_schedule_post_no_draft():
    from tools.social_media import social_schedule_post

    result = social_schedule_post.invoke(
        {
            "scheduled_date": "2026-03-21T09:00:00Z",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )
    assert "No pending post" in result


# --- social_list_scheduled ---


@patch("tools.social_media.httpx.get")
def test_list_scheduled_formats_output(mock_get):
    from tools.social_media import social_list_scheduled

    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value=[
                {
                    "status": "scheduled",
                    "scheduleDate": "2026-03-23T13:00:00Z",
                    "platforms": ["linkedin"],
                    "post": "Win more SEWP contracts with faster quoting",
                    "id": "abc123",
                },
                {
                    "status": "scheduled",
                    "scheduleDate": "2026-03-24T18:00:00Z",
                    "platforms": ["linkedin", "twitter"],
                    "post": "SEWP quoting just got easier",
                    "id": "def456",
                },
            ]
        ),
    )
    mock_get.return_value.raise_for_status = MagicMock()

    result = social_list_scheduled.invoke({"user_email": AUTHORIZED_EMAIL})

    assert "abc123" in result
    assert "def456" in result
    assert "linkedin" in result


@patch("tools.social_media.httpx.get")
def test_list_scheduled_empty(mock_get):
    from tools.social_media import social_list_scheduled

    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=[]),
    )
    mock_get.return_value.raise_for_status = MagicMock()

    result = social_list_scheduled.invoke({"user_email": AUTHORIZED_EMAIL})
    assert "No scheduled posts" in result


# --- social_get_post ---


@patch("tools.social_media.httpx.get")
def test_get_post_returns_details(mock_get):
    from tools.social_media import social_get_post

    mock_get.return_value = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value={
                "platforms": ["linkedin"],
                "status": "scheduled",
                "post": "SEWP quoting feature launch",
                "scheduleDate": "2026-03-23T13:00:00Z",
                "created": "2026-03-20T10:00:00Z",
            }
        ),
    )
    mock_get.return_value.raise_for_status = MagicMock()

    result = social_get_post.invoke(
        {
            "post_id": "abc123",
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    assert "abc123" in result
    assert "linkedin" in result
    assert "SEWP quoting" in result


# --- social_update_post ---


@patch("tools.social_media.httpx.put")
def test_update_post_sends_new_text(mock_put):
    from tools.social_media import social_update_post

    mock_put.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"status": "updated"}),
    )
    mock_put.return_value.raise_for_status = MagicMock()

    result = social_update_post.invoke(
        {
            "post_id": "abc123",
            "user_email": AUTHORIZED_EMAIL,
            "text": "Updated headline here",
        }
    )

    assert "updated" in result.lower()
    call_json = mock_put.call_args[1]["json"]
    assert call_json["post"] == "Updated headline here"
    assert call_json["id"] == "abc123"


@patch("tools.social_media.httpx.put")
def test_update_post_sends_new_schedule(mock_put):
    from tools.social_media import social_update_post

    mock_put.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"status": "updated"}),
    )
    mock_put.return_value.raise_for_status = MagicMock()

    social_update_post.invoke(
        {
            "post_id": "abc123",
            "user_email": AUTHORIZED_EMAIL,
            "scheduled_date": "2026-03-25T14:00:00Z",
        }
    )

    call_json = mock_put.call_args[1]["json"]
    assert call_json["scheduleDate"] == "2026-03-25T14:00:00Z"


# --- social_delete_post ---


@patch("tools.social_media.httpx.request")
def test_delete_post_calls_ayrshare(mock_request):
    from tools.social_media import social_delete_post

    mock_request.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value={"status": "deleted"}),
    )
    mock_request.return_value.raise_for_status = MagicMock()

    result = social_delete_post.invoke(
        {
            "post_id": "abc123",
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    assert "deleted" in result.lower()
    call_json = mock_request.call_args[1]["json"]
    assert call_json["id"] == "abc123"


def test_delete_post_unauthorized():
    from tools.social_media import social_delete_post

    result = social_delete_post.invoke(
        {
            "post_id": "abc123",
            "user_email": UNAUTHORIZED_EMAIL,
        }
    )
    assert "not authorized" in result


# --- social_generate_image ---


@patch("tools.social_media._generate_image_vertex", new_callable=AsyncMock)
def test_generate_image_stores_result(mock_gen):
    from tools.social_media import social_generate_image

    mock_gen.return_value = "fakebase64imagedata"

    result = social_generate_image.invoke(
        {
            "prompt": "Modern GovCon infographic in terra cotta and black",
            "conversation_id": CONV_ID,
            "user_email": AUTHORIZED_EMAIL,
        }
    )

    assert "generated" in result.lower() or "Image" in result
    assert _pending_images.get(CONV_ID) == "fakebase64imagedata"


@patch("tools.social_media._generate_image_vertex", new_callable=AsyncMock)
def test_generate_image_unauthorized(mock_gen):
    from tools.social_media import social_generate_image

    result = social_generate_image.invoke(
        {
            "prompt": "test",
            "conversation_id": CONV_ID,
            "user_email": UNAUTHORIZED_EMAIL,
        }
    )

    assert "not authorized" in result
    mock_gen.assert_not_called()

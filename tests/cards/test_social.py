"""Tests for cards.social — Adaptive Card builders for social media."""

import json

from cards.social import (
    build_social_preview_card,
    build_social_published_card,
    build_social_rejected_card,
    build_scheduled_posts_cards,
)

# --- Adaptive Card schema validation helpers ---


def _assert_valid_adaptive_card(card: dict):
    """Assert that a card dict is a valid Adaptive Card v1.5 structure."""
    assert card["type"] == "AdaptiveCard"
    assert card["version"] == "1.3"
    assert "$schema" in card
    assert "body" in card
    assert isinstance(card["body"], list)
    assert len(card["body"]) > 0
    # Every body element must have a type
    for elem in card["body"]:
        assert "type" in elem, f"Body element missing 'type': {elem}"


def _find_elements(card: dict, element_type: str) -> list[dict]:
    """Find all elements of a given type in the card body."""
    return [e for e in card["body"] if e.get("type") == element_type]


def _find_actions(card: dict, action_type: str) -> list[dict]:
    """Find all actions of a given type."""
    return [a for a in card.get("actions", []) if a.get("type") == action_type]


# --- build_social_preview_card ---


def test_preview_card_valid_schema():
    card = build_social_preview_card(
        text="Test post",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    _assert_valid_adaptive_card(card)


def test_preview_card_contains_post_text():
    card = build_social_preview_card(
        text="SEWP quoting made easy",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    text_blocks = _find_elements(card, "TextBlock")
    texts = [tb["text"] for tb in text_blocks]
    assert any("SEWP quoting made easy" in t for t in texts)


def test_preview_card_shows_platforms_in_factset():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin", "twitter"],
        conversation_id="conv-123",
    )
    factsets = _find_elements(card, "FactSet")
    assert len(factsets) >= 1
    facts = factsets[0]["facts"]
    platform_fact = next(f for f in facts if f["title"] == "Platforms")
    assert "linkedin" in platform_fact["value"]
    assert "twitter" in platform_fact["value"]


def test_preview_card_has_approve_button():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    submit_actions = _find_actions(card, "Action.Submit")
    approve = [a for a in submit_actions if a["title"] == "Approve"]
    assert len(approve) == 1
    assert approve[0]["data"]["action"] == "social_approve"
    assert approve[0]["data"]["conversation_id"] == "conv-123"


def test_preview_card_has_reject_button():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    submit_actions = _find_actions(card, "Action.Submit")
    reject = [a for a in submit_actions if a["title"] == "Request Changes"]
    assert len(reject) == 1
    assert reject[0]["data"]["action"] == "social_reject"


def test_preview_card_has_schedule_option_when_immediate():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    # Should have a Schedule submit button that triggers the calendar flow
    submit_actions = _find_actions(card, "Action.Submit")
    schedule = [a for a in submit_actions if a["title"] == "Schedule"]
    assert len(schedule) == 1
    assert schedule[0]["data"]["action"] == "social_show_calendar"


def test_preview_card_no_schedule_option_when_already_scheduled():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
        scheduled_date="2026-03-25T09:00:00Z",
    )
    show_cards = [a for a in card.get("actions", []) if a["type"] == "Action.ShowCard"]
    assert len(show_cards) == 0


def test_preview_card_shows_scheduled_date():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
        scheduled_date="2026-03-25T09:00:00Z",
    )
    factsets = _find_elements(card, "FactSet")
    facts = factsets[0]["facts"]
    sched_fact = next(f for f in facts if f["title"] == "Scheduled")
    assert "2026-03-25" in sched_fact["value"]


def test_preview_card_shows_image():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
        image_url="https://example.com/img.png",
    )
    images = _find_elements(card, "Image")
    assert len(images) == 1
    assert images[0]["url"] == "https://example.com/img.png"


def test_preview_card_no_image_when_empty():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    images = _find_elements(card, "Image")
    assert len(images) == 0


def test_preview_card_shows_status_pending():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
    )
    factsets = _find_elements(card, "FactSet")
    facts = factsets[0]["facts"]
    status_fact = next(f for f in facts if f["title"] == "Status")
    assert status_fact["value"] == "Pending Approval"


# --- build_social_published_card ---


def test_published_card_valid_schema():
    card = build_social_published_card(
        text="Published post",
        platforms=["linkedin"],
    )
    _assert_valid_adaptive_card(card)


def test_published_card_no_action_buttons():
    card = build_social_published_card(
        text="Published post",
        platforms=["linkedin"],
    )
    assert "actions" not in card or len(card.get("actions", [])) == 0


def test_published_card_shows_status():
    card = build_social_published_card(
        text="Published post",
        platforms=["linkedin"],
        post_id="abc123",
    )
    factsets = _find_elements(card, "FactSet")
    facts = factsets[0]["facts"]
    status_fact = next(f for f in facts if f["title"] == "Status")
    assert status_fact["value"] == "Published"


def test_published_card_shows_post_id():
    card = build_social_published_card(
        text="Post",
        platforms=["linkedin"],
        post_id="xyz789",
    )
    factsets = _find_elements(card, "FactSet")
    facts = factsets[0]["facts"]
    id_fact = next(f for f in facts if f["title"] == "Post ID")
    assert id_fact["value"] == "xyz789"


def test_published_card_shows_image():
    card = build_social_published_card(
        text="Post",
        platforms=["linkedin"],
        image_url="https://example.com/img.png",
    )
    images = _find_elements(card, "Image")
    assert len(images) == 1


def test_published_card_header_color_good():
    card = build_social_published_card(
        text="Post",
        platforms=["linkedin"],
    )
    header = card["body"][0]
    assert header["color"] == "good"


# --- build_social_rejected_card ---


def test_rejected_card_valid_schema():
    card = build_social_rejected_card(
        text="Rejected post",
        platforms=["linkedin"],
    )
    _assert_valid_adaptive_card(card)


def test_rejected_card_no_action_buttons():
    card = build_social_rejected_card(
        text="Rejected post",
        platforms=["linkedin"],
    )
    assert "actions" not in card or len(card.get("actions", [])) == 0


def test_rejected_card_shows_cancelled_status():
    card = build_social_rejected_card(
        text="Post",
        platforms=["linkedin"],
    )
    factsets = _find_elements(card, "FactSet")
    facts = factsets[0]["facts"]
    status_fact = next(f for f in facts if f["title"] == "Status")
    assert status_fact["value"] == "Cancelled"


def test_rejected_card_header_color_attention():
    card = build_social_rejected_card(
        text="Post",
        platforms=["linkedin"],
    )
    header = card["body"][0]
    assert header["color"] == "attention"


# --- build_scheduled_posts_cards ---


def test_scheduled_cards_returns_list():
    posts = [
        {
            "post": "Post 1",
            "platforms": ["linkedin"],
            "scheduleDate": "2026-03-23T13:00:00Z",
            "id": "a1",
        },
        {
            "post": "Post 2",
            "platforms": ["twitter"],
            "scheduleDate": "2026-03-24T18:00:00Z",
            "id": "b2",
        },
    ]
    cards = build_scheduled_posts_cards(posts)
    assert isinstance(cards, list)
    assert len(cards) == 2


def test_scheduled_cards_each_valid():
    posts = [
        {
            "post": "Post 1",
            "platforms": ["linkedin"],
            "scheduleDate": "2026-03-23T13:00:00Z",
            "id": "a1",
        },
    ]
    cards = build_scheduled_posts_cards(posts)
    for card in cards:
        _assert_valid_adaptive_card(card)


def test_scheduled_cards_max_10():
    posts = [
        {
            "post": f"Post {i}",
            "platforms": ["linkedin"],
            "scheduleDate": f"2026-03-{20+i}T09:00:00Z",
            "id": f"id{i}",
        }
        for i in range(15)
    ]
    cards = build_scheduled_posts_cards(posts)
    assert len(cards) == 10


def test_scheduled_cards_show_post_id():
    posts = [
        {
            "post": "Test",
            "platforms": ["linkedin"],
            "scheduleDate": "2026-03-23T13:00:00Z",
            "id": "abc123",
        },
    ]
    cards = build_scheduled_posts_cards(posts)
    factsets = _find_elements(cards[0], "FactSet")
    facts = factsets[0]["facts"]
    id_fact = next(f for f in facts if f["title"] == "Post ID")
    assert id_fact["value"] == "abc123"


def test_scheduled_cards_empty_input():
    cards = build_scheduled_posts_cards([])
    assert cards == []


def test_scheduled_card_truncates_long_text():
    posts = [
        {
            "post": "A" * 300,
            "platforms": ["linkedin"],
            "scheduleDate": "2026-03-23T13:00:00Z",
            "id": "x",
        },
    ]
    cards = build_scheduled_posts_cards(posts)
    text_blocks = _find_elements(cards[0], "TextBlock")
    post_text = next(tb["text"] for tb in text_blocks if "A" in tb["text"])
    assert len(post_text) <= 210  # 200 chars + "..."
    assert post_text.endswith("...")


# --- JSON serialization sanity check ---


def test_preview_card_json_serializable():
    card = build_social_preview_card(
        text="Test",
        platforms=["linkedin"],
        conversation_id="conv-123",
        image_url="https://example.com/img.png",
        scheduled_date="2026-03-25T09:00:00Z",
    )
    # Must be JSON-serializable (this is what CardFactory.adaptive_card passes to Teams)
    serialized = json.dumps(card)
    assert isinstance(serialized, str)
    roundtrip = json.loads(serialized)
    assert roundtrip == card

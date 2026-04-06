"""Tests for tools/google_search.py — Google web search tool."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_genai():
    """Mock the google.genai client."""
    with patch("tools.google_search.genai") as mock:  # now module-level import
        yield mock


def _make_response(text, sources=None):
    """Build a mock GenerateContentResponse."""
    part = MagicMock()
    part.text = text

    candidate = MagicMock()
    candidate.content.parts = [part]

    # Grounding metadata
    metadata = MagicMock()
    if sources:
        supports = []
        chunks = []
        for i, (title, uri) in enumerate(sources):
            support = MagicMock()
            support.grounding_chunk_indices = [i]
            supports.append(support)
            chunk = MagicMock()
            chunk.web.title = title
            chunk.web.uri = uri
            chunks.append(chunk)
        metadata.grounding_supports = supports
        metadata.grounding_chunks = chunks
    else:
        metadata.grounding_supports = None
        metadata.grounding_chunks = None
    candidate.grounding_metadata = metadata

    response = MagicMock()
    response.candidates = [candidate]
    return response


def test_google_search_returns_text(mock_genai):
    from tools.google_search import google_search

    mock_genai.Client.return_value.models.generate_content.return_value = (
        _make_response("Python 3.12 was released in October 2023.")
    )

    result = google_search.invoke({"query": "python 3.12 release date"})
    assert "Python 3.12" in result


def test_google_search_includes_sources(mock_genai):
    from tools.google_search import google_search

    mock_genai.Client.return_value.models.generate_content.return_value = (
        _make_response(
            "Cloud Run supports volumes.",
            sources=[("Google Cloud Docs", "https://cloud.google.com/run/docs")],
        )
    )

    result = google_search.invoke({"query": "cloud run volumes"})
    assert "Sources:" in result
    assert "Google Cloud Docs" in result


def test_google_search_no_results(mock_genai):
    from tools.google_search import google_search

    part = MagicMock()
    part.text = ""
    candidate = MagicMock()
    candidate.content.parts = [part]
    response = MagicMock()
    response.candidates = [candidate]
    mock_genai.Client.return_value.models.generate_content.return_value = response

    result = google_search.invoke({"query": "asdfghjkl nonsense"})
    assert "No search results" in result


def test_google_search_passes_query(mock_genai):
    from tools.google_search import google_search

    mock_genai.Client.return_value.models.generate_content.return_value = (
        _make_response("result")
    )

    google_search.invoke({"query": "test query"})

    call_args = mock_genai.Client.return_value.models.generate_content.call_args
    assert call_args[1]["contents"] == "test query"

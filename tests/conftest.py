"""Shared fixtures — sets safe env vars so no test hits a real service."""

import os
import tempfile

import pytest

# Set test data directory at module level (before any test imports read it)
os.environ.setdefault("SAMURAI_DATA_DIR", tempfile.mkdtemp(prefix="samurai_test_"))


@pytest.fixture(autouse=True, scope="session")
def env_vars():
    os.environ.setdefault("GCP_PROJECT_ID", "test-project")
    os.environ.setdefault("MICROSOFT_APP_ID", "test-app-id")
    os.environ.setdefault("MICROSOFT_APP_PASSWORD", "test-app-password")
    os.environ.setdefault("GITHUB_APP_ID", "12345")
    os.environ.setdefault("GITHUB_APP_PRIVATE_KEY", "fake-private-key")
    os.environ.setdefault("AYRSHARE_API_KEY", "test-ayrshare-key")

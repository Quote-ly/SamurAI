"""Shared fixtures — sets safe env vars so no test hits a real service."""

import os
import pytest


@pytest.fixture(autouse=True, scope="session")
def env_vars():
    os.environ.setdefault("GCP_PROJECT_ID", "test-project")
    os.environ.setdefault("MICROSOFT_APP_ID", "test-app-id")
    os.environ.setdefault("MICROSOFT_APP_PASSWORD", "test-app-password")
    os.environ.setdefault("GITHUB_APP_ID", "12345")
    os.environ.setdefault("GITHUB_APP_PRIVATE_KEY", "fake-private-key")

import os
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _set_env(monkeypatch):
    """Set required env vars for tests and clear settings cache."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")

    from app.config import get_settings
    get_settings.cache_clear()


@pytest.fixture
def mock_pipeline_repo():
    return AsyncMock()


@pytest.fixture
def mock_eval_repo():
    return AsyncMock()


@pytest.fixture
def mock_invocation_repo():
    return AsyncMock()

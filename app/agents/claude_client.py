from __future__ import annotations

import anthropic

from app.config import get_settings

_async_client: anthropic.AsyncAnthropic | None = None


def get_async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)
    return _async_client

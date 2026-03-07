from __future__ import annotations

import asyncio
import json

import httpx

from app.config import get_settings
from app.core.exceptions import ASRProviderError
from app.providers.asr.base import ASRProvider, TranscriptionResult


class DeepgramProvider(ASRProvider):
    API_URL = "https://api.deepgram.com/v1/listen"

    async def transcribe(self, audio_bytes: bytes, hints: list[str] | None = None) -> TranscriptionResult:
        params: dict[str, str] = {
            "model": "nova-2",
            "smart_format": "true",
        }
        if hints:
            params["keywords"] = ",".join(hints)

        headers = {
            "Authorization": f"Token {get_settings().deepgram_api_key}",
            "Content-Type": "audio/wav",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.API_URL,
                    params=params,
                    headers=headers,
                    content=audio_bytes,
                )
                response.raise_for_status()
                data = response.json()

            alt = data["results"]["channels"][0]["alternatives"][0]
            return TranscriptionResult(
                text=alt["transcript"],
                confidence=alt["confidence"],
            )
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise ASRProviderError(f"Deepgram transcription failed: {e}") from e

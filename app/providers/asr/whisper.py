from __future__ import annotations

import httpx

from app.config import get_settings
from app.core.exceptions import ASRProviderError
from app.providers.asr.base import ASRProvider, TranscriptionResult


class WhisperProvider(ASRProvider):
    API_URL = "https://api.openai.com/v1/audio/transcriptions"

    async def transcribe(self, audio_bytes: bytes, hints: list[str] | None = None) -> TranscriptionResult:
        headers = {"Authorization": f"Bearer {get_settings().openai_api_key}"}

        prompt = ", ".join(hints) if hints else None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                data: dict[str, str] = {"model": "whisper-1", "response_format": "verbose_json"}
                if prompt:
                    data["prompt"] = prompt

                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()

            return TranscriptionResult(
                text=result["text"],
                confidence=1.0,
                language=result.get("language", "en"),
            )
        except (httpx.HTTPError, KeyError) as e:
            raise ASRProviderError(f"Whisper transcription failed: {e}") from e

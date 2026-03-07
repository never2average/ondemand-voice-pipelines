from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str = "en"
    metadata: dict | None = None


class ASRProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, hints: list[str] | None = None) -> TranscriptionResult:
        ...

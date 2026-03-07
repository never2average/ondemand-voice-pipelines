from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from app.core.exceptions import ASRProviderError
from app.providers.asr.base import ASRProvider, TranscriptionResult


class SampleProvider(ASRProvider):
    """Deterministic local ASR for committed demo voice fixtures."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        hints: list[str] | None = None,
    ) -> TranscriptionResult:
        if not audio_bytes:
            raise ASRProviderError("Sample transcription failed: audio payload was empty.")

        manifest = _load_sample_manifest()
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        sample = manifest.get(audio_hash)
        if sample is None:
            known_samples = ", ".join(
                sorted(entry["sample_name"] for entry in manifest.values())
            )
            raise ASRProviderError(
                "Sample transcription failed: audio sample is not registered. "
                f"Known samples: {known_samples or 'none'}."
            )

        return TranscriptionResult(
            text=sample["transcript"],
            confidence=float(sample.get("confidence", 0.99)),
            language=str(sample.get("language", "en")),
            metadata={
                "sample_name": sample["sample_name"],
                "sha256": audio_hash,
                "keyword_hints": hints or [],
            },
        )


@lru_cache(maxsize=1)
def _load_sample_manifest() -> dict[str, dict[str, object]]:
    manifest_path = (
        Path(__file__).resolve().parents[3] / "examples" / "voice_samples" / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text())
    return {
        str(sample["sha256"]): sample
        for sample in payload.get("samples", [])
    }

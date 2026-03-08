from app.providers.asr.base import ASRProvider
from app.providers.asr.deepgram import DeepgramProvider
from app.providers.asr.whisper import WhisperProvider

_PROVIDERS: dict[str, type[ASRProvider]] = {
    "deepgram": DeepgramProvider,
    "whisper": WhisperProvider,
}


def get_asr_provider(name: str) -> ASRProvider:
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown ASR provider: {name}. Available: {list(_PROVIDERS.keys())}")
    return cls()

import pytest

from app.providers.asr.deepgram import DeepgramProvider
from app.providers.asr.factory import get_asr_provider
from app.providers.asr.sample import SampleProvider
from app.providers.asr.whisper import WhisperProvider


def test_get_whisper_provider():
    provider = get_asr_provider("whisper")
    assert isinstance(provider, WhisperProvider)


def test_get_deepgram_provider():
    provider = get_asr_provider("deepgram")
    assert isinstance(provider, DeepgramProvider)


def test_get_sample_provider():
    provider = get_asr_provider("sample")
    assert isinstance(provider, SampleProvider)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown ASR provider"):
        get_asr_provider("does-not-exist")

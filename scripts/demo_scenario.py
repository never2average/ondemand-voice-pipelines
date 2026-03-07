from __future__ import annotations

from pathlib import Path

DEFAULT_PIPELINE_NAME = "retail-banking-voice-router"
DEFAULT_PIPELINE_DESCRIPTION = (
    "Route inbound retail banking voice calls by intent, optimizing for intent "
    "error rate instead of transcript fidelity."
)
DEFAULT_INTENT_PROMPT = "\n".join(
    [
        "I'm setting up a phone support line for a retail bank.",
        "Customers usually say things like:",
        '- "I want to check my balance"',
        '- "I need to transfer money between accounts"',
        '- "I need to dispute a charge on my card"',
        '- "I need a replacement card because mine is lost"',
        "If it's something else, send it to unknown.",
    ]
)
DEFAULT_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "voice_samples"
    / "check-my-balance.wav"
)

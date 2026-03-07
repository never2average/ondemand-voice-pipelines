from unittest.mock import patch

import pytest

from app.providers.asr.base import TranscriptionResult
from app.providers.intent.base import IntentExtractor, IntentResult
from app.schemas.artifacts import (
    ArtifactRef,
    AsrComponentSpec,
    DecisionPolicyComponentSpec,
    IntentClassifierComponentSpec,
    IntentDefinition,
    IntentSchemaArtifact,
    PipelineGraphArtifact,
    TranscriptNormalizerComponentSpec,
)
from app.schemas.invocation import InvokeRequest
from app.services.graph_runner import PipelineGraphRunner


class FakeIntentExtractor(IntentExtractor):
    def __init__(self, detected_intent: str, confidence: float):
        self._detected_intent = detected_intent
        self._confidence = confidence

    async def extract(self, text, intent_schema_artifact, classifier_component):
        return IntentResult(
            detected_intent=self._detected_intent,
            confidence=self._confidence,
            candidates=[
                {"intent_name": self._detected_intent, "confidence": self._confidence},
                {"intent_name": "unknown", "confidence": 0.1},
            ],
        )


def _build_intent_schema() -> IntentSchemaArtifact:
    return IntentSchemaArtifact(
        artifact_id="intent-schema-1",
        pipeline_id="pipe-1",
        source_prompt="check balance, transfer funds",
        fallback_intent="unknown",
        intents=[
            IntentDefinition(intent_name="check_balance", description="Check account balance"),
            IntentDefinition(intent_name="transfer_funds", description="Move money between accounts"),
        ],
    )


def _build_graph() -> PipelineGraphArtifact:
    intent_schema_ref = ArtifactRef(
        artifact_id="intent-schema-1",
        artifact_type="intent_schema",
        pipeline_id="pipe-1",
        version=1,
    )
    return PipelineGraphArtifact(
        artifact_id="graph-1",
        pipeline_id="pipe-1",
        entry_mode="audio",
        asr_component=AsrComponentSpec(
            component_id="asr",
            component_name="ASR",
            provider="whisper",
            model="whisper-1",
        ),
        normalizer_component=TranscriptNormalizerComponentSpec(
            component_id="normalizer",
            component_name="Normalizer",
            lowercase=True,
            strip_fillers=True,
            punctuation_policy="preserve",
            canonical_replacements={"wanna": "want to"},
            custom_rules=[],
        ),
        intent_classifier_component=IntentClassifierComponentSpec(
            component_id="classifier",
            component_name="Classifier",
            provider="claude",
            model="claude-sonnet-4-20250514",
            system_prompt="prompt",
            candidate_count=2,
            intent_schema=intent_schema_ref,
            few_shot_examples=[],
        ),
        decision_policy_component=DecisionPolicyComponentSpec(
            component_id="policy",
            component_name="Policy",
            confidence_threshold=0.6,
            margin_threshold=0.05,
            fallback_intent="unknown",
            allow_abstain=True,
            out_of_domain_strategy="fallback_intent",
        ),
    )


@pytest.mark.asyncio
async def test_graph_runner_normalizes_text_and_returns_traces():
    runner = PipelineGraphRunner(intent_extractor=FakeIntentExtractor("check_balance", 0.92))
    result = await runner.run_invoke_request(
        pipeline_id="pipe-1",
        invoke_request=InvokeRequest(input_type="text", input_text="Um CHECK balance"),
        pipeline_graph_artifact=_build_graph(),
        intent_schema_artifact=_build_intent_schema(),
    )
    assert result.normalized_text == "check balance"
    assert result.detected_intent == "check_balance"
    assert len(result.component_traces) == 3


@pytest.mark.asyncio
async def test_graph_runner_applies_fallback_decision_policy():
    runner = PipelineGraphRunner(intent_extractor=FakeIntentExtractor("check_balance", 0.3))
    result = await runner.run_invoke_request(
        pipeline_id="pipe-1",
        invoke_request=InvokeRequest(input_type="text", input_text="check balance"),
        pipeline_graph_artifact=_build_graph(),
        intent_schema_artifact=_build_intent_schema(),
    )
    assert result.detected_intent == "unknown"


@pytest.mark.asyncio
async def test_graph_runner_handles_audio_inputs():
    class FakeAsrProvider:
        async def transcribe(self, audio_bytes, hints=None):
            return TranscriptionResult(text="check balance", confidence=0.88)

    runner = PipelineGraphRunner(intent_extractor=FakeIntentExtractor("check_balance", 0.92))
    with patch("app.services.graph_runner.get_asr_provider", return_value=FakeAsrProvider()):
        result = await runner.run_invoke_request(
            pipeline_id="pipe-1",
            invoke_request=InvokeRequest(input_type="audio", input_audio_base64="aGVsbG8="),
            pipeline_graph_artifact=_build_graph(),
            intent_schema_artifact=_build_intent_schema(),
        )
    assert result.transcript_text == "check balance"
    assert result.component_traces[0].component_kind == "asr"

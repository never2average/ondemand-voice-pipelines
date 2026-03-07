from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field

from app.core.exceptions import PipelineNotReadyError
from app.providers.asr.factory import get_asr_provider
from app.providers.intent.base import IntentExtractor
from app.providers.intent.claude_intent import ClaudeIntentExtractor
from app.schemas.artifacts import ComponentTrace, IntentSchemaArtifact, PipelineGraphArtifact
from app.schemas.invocation import InvokeRequest


@dataclass
class GraphInvocationResult:
    transcript_text: str
    normalized_text: str
    detected_intent: str
    confidence: float
    intent_candidates: list[dict[str, float | str]] = field(default_factory=list)
    component_traces: list[ComponentTrace] = field(default_factory=list)


class PipelineGraphRunner:
    def __init__(self, intent_extractor: IntentExtractor | None = None):
        self._intent_extractor = intent_extractor or ClaudeIntentExtractor()

    async def run_invoke_request(
        self,
        pipeline_id: str,
        invoke_request: InvokeRequest,
        pipeline_graph_artifact: PipelineGraphArtifact,
        intent_schema_artifact: IntentSchemaArtifact,
    ) -> GraphInvocationResult:
        audio_bytes = (
            base64.b64decode(invoke_request.input_audio_base64)
            if invoke_request.input_audio_base64
            else None
        )
        return await self.run(
            pipeline_id=pipeline_id,
            input_type=invoke_request.input_type,
            input_text=invoke_request.input_text,
            input_audio_bytes=audio_bytes,
            pipeline_graph_artifact=pipeline_graph_artifact,
            intent_schema_artifact=intent_schema_artifact,
        )

    async def run(
        self,
        pipeline_id: str,
        input_type: str,
        input_text: str | None,
        input_audio_bytes: bytes | None,
        pipeline_graph_artifact: PipelineGraphArtifact,
        intent_schema_artifact: IntentSchemaArtifact,
    ) -> GraphInvocationResult:
        component_traces: list[ComponentTrace] = []
        transcript_text = input_text or ""

        if input_type == "audio":
            if pipeline_graph_artifact.asr_component is None:
                raise PipelineNotReadyError(pipeline_id, "ready (but missing ASR component)")
            start_time = time.monotonic()
            transcription = await get_asr_provider(
                pipeline_graph_artifact.asr_component.provider
            ).transcribe(
                input_audio_bytes or b"",
                hints=pipeline_graph_artifact.asr_component.keyword_hints,
            )
            transcript_text = transcription.text
            component_traces.append(
                ComponentTrace(
                    component_id=pipeline_graph_artifact.asr_component.component_id,
                    component_kind="asr",
                    input_snapshot={"input_type": "audio"},
                    output_snapshot={
                        "transcript_text": transcript_text,
                        "confidence": transcription.confidence,
                    },
                    latency_ms=int((time.monotonic() - start_time) * 1000),
                )
            )
        elif not transcript_text:
            raise PipelineNotReadyError(pipeline_id, "ready (but missing text input)")

        normalized_text = transcript_text
        if pipeline_graph_artifact.normalizer_component:
            start_time = time.monotonic()
            normalized_text = self._normalize_text(
                transcript_text,
                pipeline_graph_artifact.normalizer_component,
            )
            component_traces.append(
                ComponentTrace(
                    component_id=pipeline_graph_artifact.normalizer_component.component_id,
                    component_kind="normalizer",
                    input_snapshot={"transcript_text": transcript_text},
                    output_snapshot={"normalized_text": normalized_text},
                    latency_ms=int((time.monotonic() - start_time) * 1000),
                )
            )

        start_time = time.monotonic()
        intent_result = await self._intent_extractor.extract(
            text=normalized_text,
            intent_schema_artifact=intent_schema_artifact,
            classifier_component=pipeline_graph_artifact.intent_classifier_component,
        )
        component_traces.append(
            ComponentTrace(
                component_id=pipeline_graph_artifact.intent_classifier_component.component_id,
                component_kind="intent_classifier",
                input_snapshot={"normalized_text": normalized_text},
                output_snapshot={
                    "detected_intent": intent_result.detected_intent,
                    "confidence": intent_result.confidence,
                    "intent_candidates": intent_result.candidates,
                },
                latency_ms=int((time.monotonic() - start_time) * 1000),
            )
        )

        decision_start_time = time.monotonic()
        detected_intent, confidence = self._apply_decision_policy(
            pipeline_graph_artifact,
            intent_schema_artifact,
            intent_result.candidates,
        )
        component_traces.append(
            ComponentTrace(
                component_id=pipeline_graph_artifact.decision_policy_component.component_id,
                component_kind="decision_policy",
                input_snapshot={"intent_candidates": intent_result.candidates},
                output_snapshot={
                    "detected_intent": detected_intent,
                    "confidence": confidence,
                },
                latency_ms=int((time.monotonic() - decision_start_time) * 1000),
            )
        )

        return GraphInvocationResult(
            transcript_text=transcript_text,
            normalized_text=normalized_text,
            detected_intent=detected_intent,
            confidence=confidence,
            intent_candidates=intent_result.candidates,
            component_traces=component_traces,
        )

    def _normalize_text(self, transcript_text: str, normalizer_component) -> str:
        normalized_text = transcript_text
        if normalizer_component.lowercase:
            normalized_text = normalized_text.lower()
        if normalizer_component.strip_fillers:
            normalized_text = re.sub(r"\b(um|uh|like|you know)\b", " ", normalized_text)
        for source_value, target_value in normalizer_component.canonical_replacements.items():
            normalized_text = re.sub(
                rf"\b{re.escape(source_value)}\b",
                target_value,
                normalized_text,
            )
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        return normalized_text

    def _apply_decision_policy(
        self,
        pipeline_graph_artifact: PipelineGraphArtifact,
        intent_schema_artifact: IntentSchemaArtifact,
        candidates: list[dict[str, float | str]],
    ) -> tuple[str, float]:
        if not candidates:
            return (
                intent_schema_artifact.fallback_intent,
                0.0,
            )

        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: float(candidate.get("confidence", 0.0)),
            reverse=True,
        )
        top_candidate = ordered_candidates[0]
        top_confidence = float(top_candidate.get("confidence", 0.0))
        second_confidence = (
            float(ordered_candidates[1].get("confidence", 0.0))
            if len(ordered_candidates) > 1
            else 0.0
        )
        margin = top_confidence - second_confidence
        decision_policy = pipeline_graph_artifact.decision_policy_component

        if (
            top_confidence < decision_policy.confidence_threshold
            or margin < decision_policy.margin_threshold
        ) and decision_policy.allow_abstain:
            return decision_policy.fallback_intent, top_confidence

        return str(top_candidate.get("intent_name", intent_schema_artifact.fallback_intent)), top_confidence

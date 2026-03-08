from __future__ import annotations

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext, new_artifact_id
from app.schemas.artifacts import (
    ArtifactRef,
    AsrComponentSpec,
    DecisionPolicyComponentSpec,
    IntentClassifierComponentSpec,
    IntentSchemaArtifact,
    PipelineEdge,
    PipelineGraphArtifact,
    TranscriptNormalizerComponentSpec,
)


class BaselineGraphPlannerAgent(BaseArtifactAgent[PipelineGraphArtifact]):
    agent_name = "baseline_graph_planner_agent"

    async def run(
        self,
        build_context: PipelineBuildContext,
        intent_schema_artifact: IntentSchemaArtifact,
        intent_schema_ref: ArtifactRef,
    ) -> ArtifactGenerationResult[PipelineGraphArtifact]:
        classifier_prompt = self._build_classifier_prompt(intent_schema_artifact)
        asr_component = AsrComponentSpec(
            component_id="asr_primary",
            component_name="Primary ASR",
            provider=build_context.asr_provider,
            model=self._resolve_asr_model(build_context.asr_provider),
            language="en",
            keyword_hints=[intent.intent_name.replace("_", " ") for intent in intent_schema_artifact.intents],
            n_best=1,
            endpointing_config={"mode": "default"},
        )
        normalizer_component = TranscriptNormalizerComponentSpec(
            component_id="transcript_normalizer",
            component_name="Transcript Normalizer",
            lowercase=True,
            strip_fillers=True,
            punctuation_policy="preserve",
            canonical_replacements={
                "wanna": "want to",
                "gonna": "going to",
                "lemme": "let me",
                "uh": "",
                "um": "",
            },
            custom_rules=[
                "Collapse repeated whitespace.",
                "Trim leading and trailing filler phrases.",
            ],
        )
        classifier_component = IntentClassifierComponentSpec(
            component_id="intent_classifier",
            component_name="Intent Classifier",
            provider="claude",
            model="claude-sonnet-4-20250514",
            system_prompt=classifier_prompt,
            candidate_count=3,
            intent_schema=intent_schema_ref,
            few_shot_examples=[
                f"{intent.positive_examples[0]} -> {intent.intent_name}"
                for intent in intent_schema_artifact.intents
                if intent.positive_examples
            ],
        )
        decision_policy_component = DecisionPolicyComponentSpec(
            component_id="decision_policy",
            component_name="Intent Decision Policy",
            confidence_threshold=0.7,
            margin_threshold=0.08,
            fallback_intent=intent_schema_artifact.fallback_intent,
            allow_abstain=True,
            out_of_domain_strategy="fallback_intent",
        )
        artifact = PipelineGraphArtifact(
            artifact_id=new_artifact_id(),
            pipeline_id=build_context.pipeline_id,
            entry_mode="audio",
            asr_component=asr_component,
            normalizer_component=normalizer_component,
            intent_classifier_component=classifier_component,
            decision_policy_component=decision_policy_component,
            candidate_reranker_component=None,
            edges=[
                PipelineEdge(
                    from_component_id=asr_component.component_id,
                    from_output_key="transcript_text",
                    to_component_id=normalizer_component.component_id,
                    to_input_key="transcript_text",
                ),
                PipelineEdge(
                    from_component_id=normalizer_component.component_id,
                    from_output_key="normalized_text",
                    to_component_id=classifier_component.component_id,
                    to_input_key="utterance_text",
                ),
                PipelineEdge(
                    from_component_id=classifier_component.component_id,
                    from_output_key="intent_candidates",
                    to_component_id=decision_policy_component.component_id,
                    to_input_key="intent_candidates",
                ),
            ],
        )
        summary = "Constructed the baseline pipeline graph with ASR, normalization, classification, and decision policy."
        return ArtifactGenerationResult(artifact=artifact, summary=summary)

    def _build_classifier_prompt(self, intent_schema_artifact: IntentSchemaArtifact) -> str:
        sections: list[str] = [
            "You are an intent classification engine.",
            "Classify the user utterance into one of the allowed intents below.",
            f"If no intent clearly matches, return {intent_schema_artifact.fallback_intent}.",
            "",
            "Allowed intents:",
        ]
        for intent_definition in intent_schema_artifact.intents:
            sections.append(f"- {intent_definition.intent_name}: {intent_definition.description}")
            if intent_definition.disambiguation_rules:
                sections.append("  Disambiguation rules:")
                sections.extend([f"  - {rule}" for rule in intent_definition.disambiguation_rules])
        sections.extend(
            [
                "",
                "Return JSON with fields intent, confidence, and candidates.",
            ]
        )
        return "\n".join(sections)

    def _resolve_asr_model(self, asr_provider: str) -> str:
        if asr_provider == "whisper":
            return "whisper-1"
        return "nova-2"

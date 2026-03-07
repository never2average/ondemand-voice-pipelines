from __future__ import annotations

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext, new_artifact_id
from app.schemas.artifacts import ArtifactRef, EvalDatasetArtifact, EvalExample, IntentSchemaArtifact


class EvalDatasetCuratorAgent(BaseArtifactAgent[EvalDatasetArtifact]):
    agent_name = "eval_dataset_curator_agent"

    async def run(
        self,
        build_context: PipelineBuildContext,
        intent_schema_artifact: IntentSchemaArtifact,
        intent_schema_ref: ArtifactRef,
    ) -> ArtifactGenerationResult[EvalDatasetArtifact]:
        examples: list[EvalExample] = []
        for intent_definition in intent_schema_artifact.intents:
            examples.extend(self._examples_for_intent(intent_definition.intent_name, intent_definition.description))

        examples.extend(self._fallback_examples(intent_schema_artifact.fallback_intent))
        coverage_summary = {
            split_name: sum(1 for example in examples if example.split == split_name)
            for split_name in ("train", "dev", "holdout")
        }
        artifact = EvalDatasetArtifact(
            artifact_id=new_artifact_id(),
            pipeline_id=build_context.pipeline_id,
            intent_schema=intent_schema_ref,
            examples=examples,
            coverage_summary=coverage_summary,
        )
        summary = (
            f"Curated {len(examples)} evaluation examples with immutable holdout coverage "
            f"for {len(intent_schema_artifact.intents)} intents."
        )
        return ArtifactGenerationResult(artifact=artifact, summary=summary)

    def _examples_for_intent(self, intent_name: str, description: str) -> list[EvalExample]:
        display_phrase = intent_name.replace("_", " ")
        return [
            EvalExample(
                example_id=new_artifact_id(),
                split="train",
                source="curated",
                modality="text",
                utterance_text=display_phrase,
                expected_intent=intent_name,
                phenomenon_tags=["direct"],
            ),
            EvalExample(
                example_id=new_artifact_id(),
                split="train",
                source="curated",
                modality="text",
                utterance_text=f"I need help with {display_phrase}",
                expected_intent=intent_name,
                phenomenon_tags=["natural_language"],
            ),
            EvalExample(
                example_id=new_artifact_id(),
                split="train",
                source="curated",
                modality="text",
                utterance_text=f"Can you handle {display_phrase} for me?",
                expected_intent=intent_name,
                phenomenon_tags=["question"],
            ),
            EvalExample(
                example_id=new_artifact_id(),
                split="dev",
                source="curated",
                modality="text",
                utterance_text=f"please {display_phrase}",
                expected_intent=intent_name,
                phenomenon_tags=["polite"],
            ),
            EvalExample(
                example_id=new_artifact_id(),
                split="dev",
                source="curated",
                modality="text",
                utterance_text=f"uh can you {display_phrase}",
                expected_intent=intent_name,
                phenomenon_tags=["filler_words"],
            ),
            EvalExample(
                example_id=new_artifact_id(),
                split="holdout",
                source="curated",
                modality="text",
                utterance_text=f"could you help me with {display_phrase}",
                expected_intent=intent_name,
                phenomenon_tags=["holdout", "polite"],
            ),
            EvalExample(
                example_id=new_artifact_id(),
                split="holdout",
                source="curated",
                modality="text",
                utterance_text=f"I'm calling about {description.lower()}",
                expected_intent=intent_name,
                phenomenon_tags=["holdout", "description_based"],
            ),
        ]

    def _fallback_examples(self, fallback_intent: str) -> list[EvalExample]:
        fallback_texts = [
            ("train", "Tell me a joke"),
            ("dev", "What's the weather like?"),
            ("holdout", "I just wanted to chat about something random"),
        ]
        return [
            EvalExample(
                example_id=new_artifact_id(),
                split=split_name,
                source="curated",
                modality="text",
                utterance_text=utterance_text,
                expected_intent=fallback_intent,
                phenomenon_tags=["out_of_domain"],
            )
            for split_name, utterance_text in fallback_texts
        ]

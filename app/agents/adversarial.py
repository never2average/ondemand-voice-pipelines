from __future__ import annotations

from collections import defaultdict

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext, new_artifact_id
from app.schemas.artifacts import (
    AdversarialFindingsArtifact,
    ArtifactRef,
    EvalDatasetArtifact,
    EvalExample,
    EvaluationReportArtifact,
    FailureCluster,
    IntentSchemaArtifact,
)


class AdversarialDatasetAgent(BaseArtifactAgent[AdversarialFindingsArtifact]):
    agent_name = "adversarial_dataset_agent"

    async def run(
        self,
        build_context: PipelineBuildContext,
        intent_schema_artifact: IntentSchemaArtifact,
        eval_dataset_artifact: EvalDatasetArtifact,
        evaluation_report_artifact: EvaluationReportArtifact,
        evaluation_report_ref: ArtifactRef,
    ) -> ArtifactGenerationResult[AdversarialFindingsArtifact]:
        failure_clusters = self._cluster_failures(evaluation_report_artifact)
        proposed_examples = self._propose_examples(evaluation_report_artifact)
        recommended_component_changes = [
            "Add failed-example keywords to ASR hints.",
            "Expand classifier disambiguation rules for confused intent pairs.",
            "Lower the abstain threshold slightly for ambiguous but in-domain utterances.",
        ]
        artifact = AdversarialFindingsArtifact(
            artifact_id=new_artifact_id(),
            pipeline_id=build_context.pipeline_id,
            evaluation_report=evaluation_report_ref,
            failure_clusters=failure_clusters,
            proposed_examples=proposed_examples,
            recommended_component_changes=recommended_component_changes,
        )
        summary = (
            f"Clustered {len(evaluation_report_artifact.hard_cases)} hard cases into "
            f"{len(failure_clusters)} failure groups and proposed {len(proposed_examples)} adversarial examples."
        )
        return ArtifactGenerationResult(artifact=artifact, summary=summary)

    def _cluster_failures(self, evaluation_report_artifact: EvaluationReportArtifact) -> list[FailureCluster]:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for case_result in evaluation_report_artifact.hard_cases:
            if case_result.correct:
                continue
            grouped[(case_result.expected_intent, case_result.predicted_intent)].append(case_result.example_id)

        clusters: list[FailureCluster] = []
        for (expected_intent, predicted_intent), example_ids in grouped.items():
            clusters.append(
                FailureCluster(
                    cluster_id=new_artifact_id(),
                    title=f"{expected_intent} confused with {predicted_intent}",
                    description=(
                        f"Examples for {expected_intent} are being routed to {predicted_intent}; "
                        "tighten intent definitions and decision policy."
                    ),
                    affected_intents=[expected_intent, predicted_intent],
                    suspected_component_ids=["intent_classifier", "decision_policy"],
                    example_ids=example_ids,
                )
            )
        return clusters

    def _propose_examples(self, evaluation_report_artifact: EvaluationReportArtifact) -> list[EvalExample]:
        proposed_examples: list[EvalExample] = []
        for index, case_result in enumerate(evaluation_report_artifact.hard_cases):
            if case_result.correct:
                continue
            source_text = case_result.normalized_text or case_result.transcript_text or case_result.expected_intent
            revised_split = "train" if index % 2 == 0 else "dev"
            variants = [
                f"um {source_text}",
                f"can you please {source_text}",
                f"{source_text} right now",
            ]
            for variant_text in variants:
                proposed_examples.append(
                    EvalExample(
                        example_id=new_artifact_id(),
                        split=revised_split,
                        source="adversarial",
                        modality="text",
                        utterance_text=variant_text,
                        expected_intent=case_result.expected_intent,
                        phenomenon_tags=["adversarial", "filler_words", "failure_driven"],
                        confusable_with=[case_result.predicted_intent],
                    )
                )
        return proposed_examples

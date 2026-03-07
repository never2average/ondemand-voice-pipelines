from __future__ import annotations

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext, new_artifact_id
from app.schemas.artifacts import (
    AdversarialFindingsArtifact,
    EvaluationReportArtifact,
    IntentSchemaArtifact,
    PipelineGraphArtifact,
)


class GraphRevisionAgent(BaseArtifactAgent[PipelineGraphArtifact]):
    agent_name = "graph_revision_agent"

    async def run(
        self,
        build_context: PipelineBuildContext,
        intent_schema_artifact: IntentSchemaArtifact,
        pipeline_graph_artifact: PipelineGraphArtifact,
        evaluation_report_artifact: EvaluationReportArtifact,
        adversarial_findings_artifact: AdversarialFindingsArtifact,
    ) -> ArtifactGenerationResult[PipelineGraphArtifact]:
        revised_graph = pipeline_graph_artifact.model_copy(deep=True)
        revised_graph.artifact_id = new_artifact_id()

        failed_tokens = self._extract_failure_tokens(evaluation_report_artifact)
        if revised_graph.asr_component:
            merged_hints = list(dict.fromkeys(revised_graph.asr_component.keyword_hints + failed_tokens))
            revised_graph.asr_component.keyword_hints = merged_hints[:25]

        if revised_graph.normalizer_component:
            revised_graph.normalizer_component.custom_rules.extend(
                [
                    "Normalize courtesy prefixes like please and kindly.",
                    "Collapse repeated filler words before intent classification.",
                ]
            )

        revision_notes = [
            f"- {cluster.title}: {cluster.description}"
            for cluster in adversarial_findings_artifact.failure_clusters
        ]
        if revision_notes:
            revised_graph.intent_classifier_component.system_prompt = (
                revised_graph.intent_classifier_component.system_prompt
                + "\n\nAdditional disambiguation guidance:\n"
                + "\n".join(revision_notes)
            )

        revised_graph.decision_policy_component.confidence_threshold = max(
            0.5,
            revised_graph.decision_policy_component.confidence_threshold - 0.05,
        )
        revised_graph.decision_policy_component.margin_threshold = max(
            0.03,
            revised_graph.decision_policy_component.margin_threshold - 0.01,
        )

        summary = (
            "Revised the pipeline graph by expanding ASR hints, normalization rules, "
            "classifier disambiguation guidance, and decision thresholds."
        )
        return ArtifactGenerationResult(artifact=revised_graph, summary=summary)

    def _extract_failure_tokens(self, evaluation_report_artifact: EvaluationReportArtifact) -> list[str]:
        unique_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for case_result in evaluation_report_artifact.hard_cases:
            text = (case_result.normalized_text or case_result.transcript_text or "").lower()
            for token in text.split():
                cleaned_token = token.strip(".,!?")
                if len(cleaned_token) < 4:
                    continue
                if cleaned_token in seen_tokens:
                    continue
                seen_tokens.add(cleaned_token)
                unique_tokens.append(cleaned_token)
        return unique_tokens

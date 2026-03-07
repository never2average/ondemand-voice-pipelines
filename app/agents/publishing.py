from __future__ import annotations

from dataclasses import dataclass

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext
from app.core.exceptions import AgentError
from app.schemas.artifacts import ArtifactRef
from app.schemas.pipeline import PipelineSpec, PipelineStatus


@dataclass
class PublishCandidate:
    intent_schema_ref: ArtifactRef
    eval_dataset_ref: ArtifactRef
    published_graph_ref: ArtifactRef
    latest_evaluation_report_ref: ArtifactRef
    latest_adversarial_findings_ref: ArtifactRef | None
    holdout_intent_error_rate: float


class PublishingAgent(BaseArtifactAgent[PipelineSpec]):
    agent_name = "publishing_agent"

    async def run(
        self,
        build_context: PipelineBuildContext,
        publish_candidate: PublishCandidate,
        created_at,
        updated_at,
    ) -> ArtifactGenerationResult[PipelineSpec]:
        if publish_candidate.holdout_intent_error_rate > build_context.optimization_objective.target_intent_error_rate:
            raise AgentError(
                "Holdout intent error rate did not meet the publish threshold."
            )

        pipeline_spec = PipelineSpec(
            pipeline_id=build_context.pipeline_id,
            name=build_context.pipeline_name,
            description=build_context.pipeline_description,
            status=PipelineStatus.ready,
            asr_provider=build_context.asr_provider,
            optimization_objective=build_context.optimization_objective,
            intent_schema=publish_candidate.intent_schema_ref,
            eval_dataset=publish_candidate.eval_dataset_ref,
            published_graph=publish_candidate.published_graph_ref,
            latest_evaluation_report=publish_candidate.latest_evaluation_report_ref,
            latest_adversarial_findings=publish_candidate.latest_adversarial_findings_ref,
            created_at=created_at,
            updated_at=updated_at,
        )
        return ArtifactGenerationResult(
            artifact=pipeline_spec,
            summary=(
                f"Published graph version {publish_candidate.published_graph_ref.version} "
                f"with holdout IER {publish_candidate.holdout_intent_error_rate:.4f}."
            ),
        )

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.agents.adversarial import AdversarialDatasetAgent
from app.agents.base import PipelineBuildContext, new_artifact_id
from app.agents.baseline_graph_planner import BaselineGraphPlannerAgent
from app.agents.eval_curator import EvalDatasetCuratorAgent
from app.agents.graph_revision import GraphRevisionAgent
from app.agents.intent_schema import IntentSchemaAgent
from app.agents.pipeline_evaluator import PipelineEvaluatorAgent
from app.agents.publishing import PublishCandidate, PublishingAgent
from app.core.exceptions import AgentError
from app.db.repositories.eval_dataset_repo import EvalDatasetRepository
from app.db.repositories.eval_repo import EvalExampleRepository
from app.db.repositories.pipeline_artifact_repo import PipelineArtifactRepository
from app.db.repositories.pipeline_build_step_repo import PipelineBuildStepRepository
from app.db.repositories.pipeline_repo import PipelineRepository
from app.schemas.artifacts import (
    AdversarialFindingsArtifact,
    ArtifactRef,
    ArtifactType,
    EvalDatasetArtifact,
    EvaluationReportArtifact,
    IntentSchemaArtifact,
    PipelineGraphArtifact,
)


@dataclass
class CandidateVersion:
    graph_artifact: PipelineGraphArtifact
    graph_ref: ArtifactRef
    eval_dataset_artifact: EvalDatasetArtifact
    eval_dataset_ref: ArtifactRef
    optimization_report_artifact: EvaluationReportArtifact
    optimization_report_ref: ArtifactRef
    latest_adversarial_findings_ref: ArtifactRef | None

    @property
    def dev_intent_error_rate(self) -> float:
        return self.optimization_report_artifact.split_intent_error_rate.get(
            "dev",
            self.optimization_report_artifact.overall_intent_error_rate,
        )


class PipelineOrchestrator:
    def __init__(
        self,
        pipeline_repo: PipelineRepository,
        pipeline_artifact_repo: PipelineArtifactRepository,
        pipeline_build_step_repo: PipelineBuildStepRepository,
        eval_dataset_repo: EvalDatasetRepository,
        eval_example_repo: EvalExampleRepository,
        intent_schema_agent: IntentSchemaAgent | None = None,
        eval_dataset_curator_agent: EvalDatasetCuratorAgent | None = None,
        baseline_graph_planner_agent: BaselineGraphPlannerAgent | None = None,
        pipeline_evaluator_agent: PipelineEvaluatorAgent | None = None,
        adversarial_dataset_agent: AdversarialDatasetAgent | None = None,
        graph_revision_agent: GraphRevisionAgent | None = None,
        publishing_agent: PublishingAgent | None = None,
    ):
        self._pipeline_repo = pipeline_repo
        self._pipeline_artifact_repo = pipeline_artifact_repo
        self._pipeline_build_step_repo = pipeline_build_step_repo
        self._eval_dataset_repo = eval_dataset_repo
        self._eval_example_repo = eval_example_repo
        self._intent_schema_agent = intent_schema_agent or IntentSchemaAgent()
        self._eval_dataset_curator_agent = eval_dataset_curator_agent or EvalDatasetCuratorAgent()
        self._baseline_graph_planner_agent = baseline_graph_planner_agent or BaselineGraphPlannerAgent()
        self._pipeline_evaluator_agent = pipeline_evaluator_agent or PipelineEvaluatorAgent()
        self._adversarial_dataset_agent = adversarial_dataset_agent or AdversarialDatasetAgent()
        self._graph_revision_agent = graph_revision_agent or GraphRevisionAgent()
        self._publishing_agent = publishing_agent or PublishingAgent()

    async def build_pipeline(self, build_context: PipelineBuildContext) -> None:
        current_step_name = "intent_schema_generation"
        best_candidate: CandidateVersion | None = None
        latest_adversarial_findings_ref: ArtifactRef | None = None

        pipeline_row = await self._pipeline_repo.get_by_id(build_context.pipeline_id)
        created_at = pipeline_row["created_at"] if pipeline_row else self._now_iso()

        try:
            await self._pipeline_repo.update(
                build_context.pipeline_id,
                {
                    "status": "building",
                    "current_build_step": current_step_name,
                    "updated_at": self._now_iso(),
                },
            )

            intent_schema_artifact, intent_schema_ref = await self._run_artifact_step(
                build_context=build_context,
                step_name="intent_schema_generation",
                input_artifact_refs=[],
                artifact_type=ArtifactType.intent_schema,
                agent_name=self._intent_schema_agent.agent_name,
                artifact_result=await self._intent_schema_agent.run(build_context),
            )
            await self._pipeline_repo.update(
                build_context.pipeline_id,
                {
                    "intent_schema_artifact_id": intent_schema_ref.artifact_id,
                    "intent_schema_artifact_version": intent_schema_ref.version,
                    "updated_at": self._now_iso(),
                },
            )

            eval_dataset_artifact, eval_dataset_ref = await self._run_artifact_step(
                build_context=build_context,
                step_name="eval_dataset_curation",
                input_artifact_refs=[intent_schema_ref],
                artifact_type=ArtifactType.eval_dataset,
                agent_name=self._eval_dataset_curator_agent.agent_name,
                artifact_result=await self._eval_dataset_curator_agent.run(
                    build_context,
                    intent_schema_artifact,
                    intent_schema_ref,
                ),
            )
            await self._pipeline_repo.update(
                build_context.pipeline_id,
                {
                    "eval_dataset_artifact_id": eval_dataset_ref.artifact_id,
                    "eval_dataset_artifact_version": eval_dataset_ref.version,
                    "updated_at": self._now_iso(),
                },
            )

            current_graph_artifact, current_graph_ref = await self._run_artifact_step(
                build_context=build_context,
                step_name="baseline_graph_planning",
                input_artifact_refs=[intent_schema_ref],
                artifact_type=ArtifactType.pipeline_graph,
                agent_name=self._baseline_graph_planner_agent.agent_name,
                artifact_result=await self._baseline_graph_planner_agent.run(
                    build_context,
                    intent_schema_artifact,
                    intent_schema_ref,
                ),
            )
            current_eval_dataset_artifact = eval_dataset_artifact
            current_eval_dataset_ref = eval_dataset_ref
            await self._pipeline_repo.update(
                build_context.pipeline_id,
                {
                    "config": self._build_compatibility_config(current_graph_artifact),
                    "updated_at": self._now_iso(),
                },
            )

            for round_index in range(build_context.optimization_objective.max_optimization_rounds):
                evaluation_step_name = f"optimization_evaluation_round_{round_index + 1}"
                optimization_report_artifact, optimization_report_ref = await self._run_artifact_step(
                    build_context=build_context,
                    step_name=evaluation_step_name,
                    input_artifact_refs=[current_graph_ref, current_eval_dataset_ref],
                    artifact_type=ArtifactType.evaluation_report,
                    agent_name=self._pipeline_evaluator_agent.agent_name,
                    artifact_result=await self._pipeline_evaluator_agent.run(
                        build_context,
                        intent_schema_artifact,
                        current_eval_dataset_artifact,
                        current_graph_artifact,
                        current_graph_ref,
                        current_eval_dataset_ref,
                        include_splits=["train", "dev"],
                    ),
                )
                await self._pipeline_repo.update(
                    build_context.pipeline_id,
                    {
                        "current_intent_error_rate": optimization_report_artifact.split_intent_error_rate.get(
                            "dev",
                            optimization_report_artifact.overall_intent_error_rate,
                        ),
                        "latest_evaluation_report_artifact_id": optimization_report_ref.artifact_id,
                        "latest_evaluation_report_version": optimization_report_ref.version,
                        "metrics": self._build_metrics_snapshot(
                            optimization_report_artifact=optimization_report_artifact,
                            published_graph_ref=current_graph_ref,
                            holdout_intent_error_rate=None,
                        ),
                        "updated_at": self._now_iso(),
                    },
                )

                candidate_version = CandidateVersion(
                    graph_artifact=current_graph_artifact,
                    graph_ref=current_graph_ref,
                    eval_dataset_artifact=current_eval_dataset_artifact,
                    eval_dataset_ref=current_eval_dataset_ref,
                    optimization_report_artifact=optimization_report_artifact,
                    optimization_report_ref=optimization_report_ref,
                    latest_adversarial_findings_ref=latest_adversarial_findings_ref,
                )
                if best_candidate is None or candidate_version.dev_intent_error_rate < best_candidate.dev_intent_error_rate:
                    best_candidate = candidate_version

                if candidate_version.dev_intent_error_rate <= build_context.optimization_objective.target_intent_error_rate:
                    break

                current_step_name = f"adversarial_analysis_round_{round_index + 1}"
                adversarial_findings_artifact, latest_adversarial_findings_ref = await self._run_artifact_step(
                    build_context=build_context,
                    step_name=current_step_name,
                    input_artifact_refs=[optimization_report_ref, current_eval_dataset_ref],
                    artifact_type=ArtifactType.adversarial_findings,
                    agent_name=self._adversarial_dataset_agent.agent_name,
                    artifact_result=await self._adversarial_dataset_agent.run(
                        build_context,
                        intent_schema_artifact,
                        current_eval_dataset_artifact,
                        optimization_report_artifact,
                        optimization_report_ref,
                    ),
                )
                await self._pipeline_repo.update(
                    build_context.pipeline_id,
                    {
                        "latest_adversarial_findings_artifact_id": latest_adversarial_findings_ref.artifact_id,
                        "latest_adversarial_findings_version": latest_adversarial_findings_ref.version,
                        "updated_at": self._now_iso(),
                    },
                )

                current_step_name = f"dataset_revision_round_{round_index + 1}"
                current_eval_dataset_artifact, current_eval_dataset_ref = await self._run_dataset_revision_step(
                    build_context=build_context,
                    step_name=current_step_name,
                    current_eval_dataset_artifact=current_eval_dataset_artifact,
                    current_eval_dataset_ref=current_eval_dataset_ref,
                    adversarial_findings_artifact=adversarial_findings_artifact,
                    adversarial_findings_ref=latest_adversarial_findings_ref,
                )
                await self._pipeline_repo.update(
                    build_context.pipeline_id,
                    {
                        "eval_dataset_artifact_id": current_eval_dataset_ref.artifact_id,
                        "eval_dataset_artifact_version": current_eval_dataset_ref.version,
                        "updated_at": self._now_iso(),
                    },
                )

                current_step_name = f"graph_revision_round_{round_index + 1}"
                current_graph_artifact, current_graph_ref = await self._run_artifact_step(
                    build_context=build_context,
                    step_name=current_step_name,
                    input_artifact_refs=[
                        current_graph_ref,
                        optimization_report_ref,
                        latest_adversarial_findings_ref,
                    ],
                    artifact_type=ArtifactType.pipeline_graph,
                    agent_name=self._graph_revision_agent.agent_name,
                    artifact_result=await self._graph_revision_agent.run(
                        build_context,
                        intent_schema_artifact,
                        current_graph_artifact,
                        optimization_report_artifact,
                        adversarial_findings_artifact,
                    ),
                )
                await self._pipeline_repo.update(
                    build_context.pipeline_id,
                    {
                        "config": self._build_compatibility_config(current_graph_artifact),
                        "updated_at": self._now_iso(),
                    },
                )

            if best_candidate is None:
                raise AgentError("No evaluation artifacts were produced during the pipeline build.")

            current_step_name = "holdout_evaluation"
            holdout_report_artifact, holdout_report_ref = await self._run_artifact_step(
                build_context=build_context,
                step_name=current_step_name,
                input_artifact_refs=[best_candidate.graph_ref, best_candidate.eval_dataset_ref],
                artifact_type=ArtifactType.evaluation_report,
                agent_name=self._pipeline_evaluator_agent.agent_name,
                artifact_result=await self._pipeline_evaluator_agent.run(
                    build_context,
                    intent_schema_artifact,
                    best_candidate.eval_dataset_artifact,
                    best_candidate.graph_artifact,
                    best_candidate.graph_ref,
                    best_candidate.eval_dataset_ref,
                    include_splits=["holdout"],
                ),
            )

            current_step_name = "publishing"
            publish_result = await self._publishing_agent.run(
                build_context=build_context,
                publish_candidate=PublishCandidate(
                    intent_schema_ref=intent_schema_ref,
                    eval_dataset_ref=best_candidate.eval_dataset_ref,
                    published_graph_ref=best_candidate.graph_ref,
                    latest_evaluation_report_ref=holdout_report_ref,
                    latest_adversarial_findings_ref=best_candidate.latest_adversarial_findings_ref,
                    holdout_intent_error_rate=holdout_report_artifact.overall_intent_error_rate,
                ),
                created_at=created_at,
                updated_at=self._now_iso(),
            )
            await self._run_completion_step(
                build_context=build_context,
                step_name=current_step_name,
                input_artifact_refs=[best_candidate.graph_ref, holdout_report_ref],
                summary=publish_result.summary,
            )

            await self._pipeline_repo.update(
                build_context.pipeline_id,
                {
                    "status": "ready",
                    "current_build_step": "published",
                    "intent_schema_artifact_id": intent_schema_ref.artifact_id,
                    "intent_schema_artifact_version": intent_schema_ref.version,
                    "eval_dataset_artifact_id": best_candidate.eval_dataset_ref.artifact_id,
                    "eval_dataset_artifact_version": best_candidate.eval_dataset_ref.version,
                    "published_graph_artifact_id": best_candidate.graph_ref.artifact_id,
                    "published_graph_version": best_candidate.graph_ref.version,
                    "latest_evaluation_report_artifact_id": holdout_report_ref.artifact_id,
                    "latest_evaluation_report_version": holdout_report_ref.version,
                    "latest_adversarial_findings_artifact_id": (
                        best_candidate.latest_adversarial_findings_ref.artifact_id
                        if best_candidate.latest_adversarial_findings_ref
                        else None
                    ),
                    "latest_adversarial_findings_version": (
                        best_candidate.latest_adversarial_findings_ref.version
                        if best_candidate.latest_adversarial_findings_ref
                        else None
                    ),
                    "current_intent_error_rate": best_candidate.dev_intent_error_rate,
                    "holdout_intent_error_rate": holdout_report_artifact.overall_intent_error_rate,
                    "config": self._build_compatibility_config(best_candidate.graph_artifact),
                    "metrics": self._build_metrics_snapshot(
                        optimization_report_artifact=best_candidate.optimization_report_artifact,
                        published_graph_ref=best_candidate.graph_ref,
                        holdout_intent_error_rate=holdout_report_artifact.overall_intent_error_rate,
                    ),
                    "updated_at": self._now_iso(),
                },
            )

        except Exception as exc:
            await self._fail_step(
                build_context=build_context,
                step_name=current_step_name,
                error_message=str(exc),
            )
            await self._pipeline_repo.update(
                build_context.pipeline_id,
                {
                    "status": "failed",
                    "current_build_step": current_step_name,
                    "metrics": {
                        "error": str(exc),
                        "current_intent_error_rate": best_candidate.dev_intent_error_rate
                        if best_candidate
                        else None,
                    },
                    "updated_at": self._now_iso(),
                },
            )

    async def _run_artifact_step(
        self,
        build_context: PipelineBuildContext,
        step_name: str,
        input_artifact_refs: list[ArtifactRef],
        artifact_type: ArtifactType,
        agent_name: str,
        artifact_result,
    ) -> tuple[Any, ArtifactRef]:
        started_at = self._now_iso()
        await self._start_step(build_context, step_name, input_artifact_refs, started_at)
        artifact_ref = await self._persist_artifact(
            build_context=build_context,
            artifact_type=artifact_type,
            artifact_payload=artifact_result.artifact,
            producer_agent=agent_name,
            summary=artifact_result.summary,
        )
        await self._complete_step(
            build_context=build_context,
            step_name=step_name,
            started_at=started_at,
            input_artifact_refs=input_artifact_refs,
            output_artifact_refs=[artifact_ref],
            summary=artifact_result.summary,
        )
        return artifact_result.artifact, artifact_ref

    async def _run_dataset_revision_step(
        self,
        build_context: PipelineBuildContext,
        step_name: str,
        current_eval_dataset_artifact: EvalDatasetArtifact,
        current_eval_dataset_ref: ArtifactRef,
        adversarial_findings_artifact: AdversarialFindingsArtifact,
        adversarial_findings_ref: ArtifactRef,
    ) -> tuple[EvalDatasetArtifact, ArtifactRef]:
        started_at = self._now_iso()
        input_refs = [current_eval_dataset_ref, adversarial_findings_ref]
        await self._start_step(build_context, step_name, input_refs, started_at)

        revised_eval_dataset_artifact = current_eval_dataset_artifact.model_copy(deep=True)
        revised_eval_dataset_artifact.artifact_id = new_artifact_id()
        revised_eval_dataset_artifact.examples.extend(
            [example.model_copy(deep=True) for example in adversarial_findings_artifact.proposed_examples]
        )
        revised_eval_dataset_artifact.coverage_summary = {
            split_name: sum(
                1
                for example in revised_eval_dataset_artifact.examples
                if example.split == split_name
            )
            for split_name in ("train", "dev", "holdout")
        }

        artifact_ref = await self._persist_artifact(
            build_context=build_context,
            artifact_type=ArtifactType.eval_dataset,
            artifact_payload=revised_eval_dataset_artifact,
            producer_agent="dataset_revision_step",
            summary="Extended the train/dev dataset with adversarial examples while keeping holdout immutable.",
        )
        await self._complete_step(
            build_context=build_context,
            step_name=step_name,
            started_at=started_at,
            input_artifact_refs=input_refs,
            output_artifact_refs=[artifact_ref],
            summary="Extended the train/dev dataset with adversarial examples while keeping holdout immutable.",
        )
        return revised_eval_dataset_artifact, artifact_ref

    async def _run_completion_step(
        self,
        build_context: PipelineBuildContext,
        step_name: str,
        input_artifact_refs: list[ArtifactRef],
        summary: str,
    ) -> None:
        started_at = self._now_iso()
        await self._start_step(build_context, step_name, input_artifact_refs, started_at)
        await self._complete_step(
            build_context=build_context,
            step_name=step_name,
            started_at=started_at,
            input_artifact_refs=input_artifact_refs,
            output_artifact_refs=[],
            summary=summary,
        )

    async def _persist_artifact(
        self,
        build_context: PipelineBuildContext,
        artifact_type: ArtifactType,
        artifact_payload,
        producer_agent: str,
        summary: str,
    ) -> ArtifactRef:
        latest_artifact_row = await self._pipeline_artifact_repo.get_latest_by_type(
            build_context.pipeline_id,
            artifact_type.value,
        )
        next_version = (int(latest_artifact_row["version"]) + 1) if latest_artifact_row else 1
        await self._pipeline_artifact_repo.create(
            {
                "id": artifact_payload.artifact_id,
                "pipeline_id": build_context.pipeline_id,
                "artifact_type": artifact_type.value,
                "version": next_version,
                "payload": artifact_payload.model_dump(mode="json"),
                "producer_agent": producer_agent,
                "summary": summary,
                "created_at": self._now_iso(),
            }
        )
        if artifact_type == ArtifactType.eval_dataset:
            await self._persist_eval_dataset(
                artifact_payload=artifact_payload,
                artifact_version=next_version,
            )
        return ArtifactRef(
            artifact_id=artifact_payload.artifact_id,
            artifact_type=artifact_type,
            pipeline_id=build_context.pipeline_id,
            version=next_version,
        )

    async def _persist_eval_dataset(
        self,
        artifact_payload: EvalDatasetArtifact,
        artifact_version: int,
    ) -> None:
        await self._eval_dataset_repo.create(
            {
                "id": artifact_payload.artifact_id,
                "pipeline_id": artifact_payload.pipeline_id,
                "intent_schema_artifact_id": artifact_payload.intent_schema.artifact_id,
                "artifact_version": artifact_version,
                "coverage_summary": artifact_payload.coverage_summary,
                "created_at": self._now_iso(),
            }
        )
        await self._eval_example_repo.create_many(
            [
                {
                    "id": example.example_id,
                    "dataset_id": artifact_payload.artifact_id,
                    "pipeline_id": artifact_payload.pipeline_id,
                    "split": example.split,
                    "source": example.source,
                    "modality": example.modality,
                    "utterance_text": example.utterance_text,
                    "audio_uri": example.audio_uri,
                    "expected_intent": example.expected_intent,
                    "phenomenon_tags": example.phenomenon_tags,
                    "confusable_with": example.confusable_with,
                    "metadata": example.metadata,
                    "created_at": self._now_iso(),
                }
                for example in artifact_payload.examples
            ]
        )

    async def _start_step(
        self,
        build_context: PipelineBuildContext,
        step_name: str,
        input_artifact_refs: list[ArtifactRef],
        started_at: str,
    ) -> None:
        await self._pipeline_build_step_repo.save(
            {
                "pipeline_id": build_context.pipeline_id,
                "step_name": step_name,
                "status": "running",
                "started_at": started_at,
                "completed_at": None,
                "input_artifacts": [artifact_ref.model_dump(mode="json") for artifact_ref in input_artifact_refs],
                "output_artifacts": [],
                "summary": "",
                "error": None,
                "updated_at": self._now_iso(),
            }
        )
        await self._pipeline_repo.update(
            build_context.pipeline_id,
            {
                "current_build_step": step_name,
                "updated_at": self._now_iso(),
            },
        )

    async def _complete_step(
        self,
        build_context: PipelineBuildContext,
        step_name: str,
        started_at: str,
        input_artifact_refs: list[ArtifactRef],
        output_artifact_refs: list[ArtifactRef],
        summary: str,
    ) -> None:
        await self._pipeline_build_step_repo.save(
            {
                "pipeline_id": build_context.pipeline_id,
                "step_name": step_name,
                "status": "completed",
                "started_at": started_at,
                "completed_at": self._now_iso(),
                "input_artifacts": [
                    artifact_ref.model_dump(mode="json") for artifact_ref in input_artifact_refs
                ],
                "output_artifacts": [
                    artifact_ref.model_dump(mode="json") for artifact_ref in output_artifact_refs
                ],
                "summary": summary,
                "error": None,
                "updated_at": self._now_iso(),
            }
        )

    async def _fail_step(
        self,
        build_context: PipelineBuildContext,
        step_name: str,
        error_message: str,
    ) -> None:
        await self._pipeline_build_step_repo.save(
            {
                "pipeline_id": build_context.pipeline_id,
                "step_name": step_name,
                "status": "failed",
                "started_at": self._now_iso(),
                "completed_at": self._now_iso(),
                "input_artifacts": [],
                "output_artifacts": [],
                "summary": "",
                "error": error_message,
                "updated_at": self._now_iso(),
            }
        )

    def _build_compatibility_config(self, pipeline_graph_artifact: PipelineGraphArtifact) -> dict[str, Any]:
        return {
            "asr_component": (
                pipeline_graph_artifact.asr_component.model_dump(mode="json")
                if pipeline_graph_artifact.asr_component
                else None
            ),
            "normalizer_component": (
                pipeline_graph_artifact.normalizer_component.model_dump(mode="json")
                if pipeline_graph_artifact.normalizer_component
                else None
            ),
            "intent_classifier_component": pipeline_graph_artifact.intent_classifier_component.model_dump(
                mode="json"
            ),
            "decision_policy_component": pipeline_graph_artifact.decision_policy_component.model_dump(
                mode="json"
            ),
        }

    def _build_metrics_snapshot(
        self,
        optimization_report_artifact: EvaluationReportArtifact,
        published_graph_ref: ArtifactRef,
        holdout_intent_error_rate: float | None,
    ) -> dict[str, Any]:
        return {
            "overall_intent_error_rate": optimization_report_artifact.overall_intent_error_rate,
            "split_intent_error_rate": optimization_report_artifact.split_intent_error_rate,
            "per_intent_accuracy": optimization_report_artifact.per_intent_accuracy,
            "published_graph_version": published_graph_ref.version,
            "holdout_intent_error_rate": holdout_intent_error_rate,
        }

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

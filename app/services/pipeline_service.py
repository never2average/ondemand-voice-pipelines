from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from fastapi import BackgroundTasks

from app.agents.base import PipelineBuildContext
from app.config import get_settings
from app.core.exceptions import PipelineNotFoundError, PipelineNotReadyError
from app.db.repositories.eval_dataset_repo import EvalDatasetRepository
from app.db.repositories.eval_repo import EvalExampleRepository
from app.db.repositories.invocation_repo import InvocationRepository
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
    PipelineBuildStep,
    PipelineGraphArtifact,
)
from app.schemas.invocation import IntentCandidate, InvokeRequest, InvokeResponse
from app.schemas.pipeline import (
    OptimizationObjective,
    PipelineArtifactRecord,
    PipelineCompatibilitySnapshot,
    PipelineCreateRequest,
    PipelineDetailResponse,
    PipelineListResponse,
    PipelineSpec,
    PipelineStatus,
    PipelineSummaryResponse,
)
from app.services.graph_runner import PipelineGraphRunner
from app.services.orchestrator import PipelineOrchestrator


class PipelineService:
    def __init__(
        self,
        pipeline_repo: PipelineRepository,
        eval_dataset_repo: EvalDatasetRepository,
        eval_example_repo: EvalExampleRepository,
        pipeline_artifact_repo: PipelineArtifactRepository,
        pipeline_build_step_repo: PipelineBuildStepRepository,
        invocation_repo: InvocationRepository,
        graph_runner: PipelineGraphRunner | None = None,
        orchestrator_factory: Callable[[], PipelineOrchestrator] | None = None,
    ):
        self._pipeline_repo = pipeline_repo
        self._eval_dataset_repo = eval_dataset_repo
        self._eval_example_repo = eval_example_repo
        self._pipeline_artifact_repo = pipeline_artifact_repo
        self._pipeline_build_step_repo = pipeline_build_step_repo
        self._invocation_repo = invocation_repo
        self._graph_runner = graph_runner or PipelineGraphRunner()
        self._orchestrator_factory = orchestrator_factory or (
            lambda: PipelineOrchestrator(
                pipeline_repo=self._pipeline_repo,
                pipeline_artifact_repo=self._pipeline_artifact_repo,
                pipeline_build_step_repo=self._pipeline_build_step_repo,
                eval_dataset_repo=self._eval_dataset_repo,
                eval_example_repo=self._eval_example_repo,
            )
        )

    async def create_pipeline(
        self,
        payload: PipelineCreateRequest,
        background_tasks: BackgroundTasks,
    ) -> PipelineDetailResponse:
        optimization_objective = payload.optimization_objective or self._default_optimization_objective()
        pipeline_id = str(uuid.uuid4())
        now = self._now_iso()
        row = await self._pipeline_repo.create(
            {
                "id": pipeline_id,
                "name": payload.name,
                "description": payload.description,
                "intent_prompt": payload.intent_prompt,
                "status": "pending",
                "config": {},
                "metrics": {},
                "asr_provider": payload.asr_provider,
                "optimization_objective": optimization_objective.model_dump(mode="json"),
                "intent_schema_artifact_id": None,
                "intent_schema_artifact_version": None,
                "eval_dataset_artifact_id": None,
                "eval_dataset_artifact_version": None,
                "published_graph_artifact_id": None,
                "published_graph_version": None,
                "latest_evaluation_report_artifact_id": None,
                "latest_evaluation_report_version": None,
                "latest_adversarial_findings_artifact_id": None,
                "latest_adversarial_findings_version": None,
                "current_intent_error_rate": None,
                "holdout_intent_error_rate": None,
                "current_build_step": "queued",
                "created_at": now,
                "updated_at": now,
            }
        )
        background_tasks.add_task(
            self._orchestrator_factory().build_pipeline,
            PipelineBuildContext(
                pipeline_id=pipeline_id,
                pipeline_name=payload.name,
                pipeline_description=payload.description,
                intent_prompt=payload.intent_prompt,
                asr_provider=payload.asr_provider,
                optimization_objective=optimization_objective,
            ),
        )
        return await self._build_pipeline_detail(row=row, build_steps=[])

    async def get_pipeline(self, pipeline_id: str) -> PipelineDetailResponse:
        row = await self._pipeline_repo.get_by_id(pipeline_id)
        if row is None:
            raise PipelineNotFoundError(pipeline_id)
        build_step_rows = await self._pipeline_build_step_repo.list_by_pipeline(pipeline_id)
        return await self._build_pipeline_detail(row=row, build_steps=build_step_rows)

    async def list_pipelines(self) -> PipelineListResponse:
        rows = await self._pipeline_repo.list_all()
        return PipelineListResponse(
            pipelines=[self._build_pipeline_summary(row) for row in rows]
        )

    async def invoke_pipeline(
        self,
        pipeline_id: str,
        request: InvokeRequest,
    ) -> InvokeResponse:
        pipeline_row = await self._pipeline_repo.get_by_id(pipeline_id)
        if pipeline_row is None:
            raise PipelineNotFoundError(pipeline_id)
        if pipeline_row["status"] != PipelineStatus.ready.value:
            raise PipelineNotReadyError(pipeline_id, pipeline_row["status"])

        graph_row = await self._require_artifact_row(
            pipeline_id,
            pipeline_row.get("published_graph_artifact_id"),
            "ready (but missing published graph)",
        )
        intent_schema_row = await self._require_artifact_row(
            pipeline_id,
            pipeline_row.get("intent_schema_artifact_id"),
            "ready (but missing intent schema)",
        )
        pipeline_graph_artifact = PipelineGraphArtifact.model_validate(graph_row["payload"])
        intent_schema_artifact = IntentSchemaArtifact.model_validate(intent_schema_row["payload"])

        start_time = datetime.now(timezone.utc)
        invocation_result = await self._graph_runner.run_invoke_request(
            pipeline_id=pipeline_id,
            invoke_request=request,
            pipeline_graph_artifact=pipeline_graph_artifact,
            intent_schema_artifact=intent_schema_artifact,
        )
        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        invocation_id = str(uuid.uuid4())
        created_at = self._now_iso()

        intent_candidates = [
            IntentCandidate(
                intent_name=str(candidate.get("intent_name", pipeline_graph_artifact.decision_policy_component.fallback_intent)),
                confidence=float(candidate.get("confidence", 0.0)),
            )
            for candidate in invocation_result.intent_candidates
        ]
        await self._invocation_repo.create(
            {
                "id": invocation_id,
                "pipeline_id": pipeline_id,
                "input_type": request.input_type,
                "input_text": invocation_result.transcript_text,
                "normalized_text": invocation_result.normalized_text,
                "detected_intent": invocation_result.detected_intent,
                "confidence": invocation_result.confidence,
                "intent_candidates": [candidate.model_dump(mode="json") for candidate in intent_candidates],
                "latency_ms": latency_ms,
                "component_traces": [
                    component_trace.model_dump(mode="json")
                    for component_trace in invocation_result.component_traces
                ],
                "pipeline_graph_artifact_id": graph_row["id"],
                "pipeline_graph_version": graph_row["version"],
                "metadata": {},
                "created_at": created_at,
            }
        )

        return InvokeResponse(
            id=invocation_id,
            pipeline_id=pipeline_id,
            input_type=request.input_type,
            input_text=invocation_result.transcript_text,
            normalized_text=invocation_result.normalized_text,
            detected_intent=invocation_result.detected_intent,
            confidence=invocation_result.confidence,
            intent_candidates=intent_candidates,
            latency_ms=latency_ms,
            component_traces=invocation_result.component_traces,
            pipeline_graph_artifact=ArtifactRef(
                artifact_id=graph_row["id"],
                artifact_type=ArtifactType.pipeline_graph,
                pipeline_id=pipeline_id,
                version=graph_row["version"],
            ),
            pipeline_graph_version=graph_row["version"],
            metadata={"invoked_at": created_at},
            created_at=created_at,
        )

    async def _build_pipeline_detail(
        self,
        row: dict,
        build_steps: list[dict],
    ) -> PipelineDetailResponse:
        artifact_rows = await self._pipeline_artifact_repo.list_by_pipeline(row["id"])
        intent_schema_artifact = await self._load_artifact_model(
            row.get("intent_schema_artifact_id"),
            IntentSchemaArtifact,
        )
        eval_dataset_artifact = await self._load_artifact_model(
            row.get("eval_dataset_artifact_id"),
            EvalDatasetArtifact,
        )
        published_graph_artifact = await self._load_artifact_model(
            row.get("published_graph_artifact_id"),
            PipelineGraphArtifact,
        )
        latest_evaluation_report_artifact = await self._load_artifact_model(
            row.get("latest_evaluation_report_artifact_id"),
            EvaluationReportArtifact,
        )
        latest_adversarial_findings_artifact = await self._load_artifact_model(
            row.get("latest_adversarial_findings_artifact_id"),
            AdversarialFindingsArtifact,
        )

        return PipelineDetailResponse(
            pipeline=self._build_pipeline_spec(row),
            intent_schema_artifact=intent_schema_artifact,
            eval_dataset_artifact=eval_dataset_artifact,
            published_graph_artifact=published_graph_artifact,
            latest_evaluation_report_artifact=latest_evaluation_report_artifact,
            latest_adversarial_findings_artifact=latest_adversarial_findings_artifact,
            artifact_history=[
                self._build_pipeline_artifact_record(artifact_row)
                for artifact_row in artifact_rows
            ],
            build_steps=[
                PipelineBuildStep.model_validate(step_row)
                for step_row in build_steps
            ],
            compatibility_snapshot=PipelineCompatibilitySnapshot(
                config=row.get("config") or {},
                metrics=row.get("metrics") or {},
            ),
        )

    def _build_pipeline_summary(self, row: dict) -> PipelineSummaryResponse:
        return PipelineSummaryResponse(
            pipeline_id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            status=PipelineStatus(row["status"]),
            asr_provider=row["asr_provider"],
            current_intent_error_rate=row.get("current_intent_error_rate"),
            holdout_intent_error_rate=row.get("holdout_intent_error_rate"),
            published_graph_version=row.get("published_graph_version"),
            active_build_step=row.get("current_build_step"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _build_pipeline_spec(self, row: dict) -> PipelineSpec:
        return PipelineSpec(
            pipeline_id=row["id"],
            name=row["name"],
            description=row.get("description", ""),
            status=PipelineStatus(row["status"]),
            asr_provider=row["asr_provider"],
            optimization_objective=OptimizationObjective.model_validate(
                row.get("optimization_objective") or self._default_optimization_objective().model_dump(mode="json")
            ),
            intent_schema=self._build_artifact_ref_from_row(
                row,
                "intent_schema_artifact_id",
                "intent_schema_artifact_version",
                ArtifactType.intent_schema,
            ),
            eval_dataset=self._build_artifact_ref_from_row(
                row,
                "eval_dataset_artifact_id",
                "eval_dataset_artifact_version",
                ArtifactType.eval_dataset,
            ),
            published_graph=self._build_artifact_ref_from_row(
                row,
                "published_graph_artifact_id",
                "published_graph_version",
                ArtifactType.pipeline_graph,
            ),
            latest_evaluation_report=self._build_artifact_ref_from_row(
                row,
                "latest_evaluation_report_artifact_id",
                "latest_evaluation_report_version",
                ArtifactType.evaluation_report,
            ),
            latest_adversarial_findings=self._build_artifact_ref_from_row(
                row,
                "latest_adversarial_findings_artifact_id",
                "latest_adversarial_findings_version",
                ArtifactType.adversarial_findings,
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _build_artifact_ref_from_row(
        self,
        row: dict,
        artifact_id_key: str,
        artifact_version_key: str,
        artifact_type: ArtifactType,
    ) -> ArtifactRef | None:
        artifact_id = row.get(artifact_id_key)
        artifact_version = row.get(artifact_version_key)
        if not artifact_id or artifact_version is None:
            return None
        return ArtifactRef(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            pipeline_id=row["id"],
            version=artifact_version,
        )

    async def _load_artifact_model(self, artifact_id: str | None, model_type):
        if not artifact_id:
            return None
        artifact_row = await self._pipeline_artifact_repo.get_by_id(artifact_id)
        if artifact_row is None:
            return None
        return model_type.model_validate(artifact_row["payload"])

    def _build_pipeline_artifact_record(self, artifact_row: dict) -> PipelineArtifactRecord:
        artifact_type = ArtifactType(artifact_row["artifact_type"])
        return PipelineArtifactRecord(
            artifact_id=artifact_row["id"],
            artifact_type=artifact_type,
            version=int(artifact_row["version"]),
            producer_agent=artifact_row.get("producer_agent", ""),
            summary=artifact_row.get("summary", ""),
            created_at=artifact_row["created_at"],
            payload=self._parse_artifact_payload(artifact_type, artifact_row["payload"]),
        )

    def _parse_artifact_payload(self, artifact_type: ArtifactType, payload: dict):
        model_by_type = {
            ArtifactType.intent_schema: IntentSchemaArtifact,
            ArtifactType.eval_dataset: EvalDatasetArtifact,
            ArtifactType.pipeline_graph: PipelineGraphArtifact,
            ArtifactType.evaluation_report: EvaluationReportArtifact,
            ArtifactType.adversarial_findings: AdversarialFindingsArtifact,
        }
        return model_by_type[artifact_type].model_validate(payload)

    async def _require_artifact_row(
        self,
        pipeline_id: str,
        artifact_id: str | None,
        status_message: str,
    ) -> dict:
        if artifact_id is None:
            raise PipelineNotReadyError(pipeline_id, status_message)
        artifact_row = await self._pipeline_artifact_repo.get_by_id(artifact_id)
        if artifact_row is None:
            raise PipelineNotReadyError(pipeline_id, status_message)
        return artifact_row

    def _default_optimization_objective(self) -> OptimizationObjective:
        settings = get_settings()
        return OptimizationObjective(
            target_intent_error_rate=settings.ier_target,
            max_optimization_rounds=settings.max_improvement_iterations,
        )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

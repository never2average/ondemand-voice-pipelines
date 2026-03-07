from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

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


class PipelineStatus(str, Enum):
    pending = "pending"
    building = "building"
    ready = "ready"
    failed = "failed"


class PipelineBuildPhase(str, Enum):
    intent_schema_design = "intent_schema_design"
    evaluation_dataset = "evaluation_dataset"
    pipeline_graph = "pipeline_graph"
    evaluation = "evaluation"
    adversarial_analysis = "adversarial_analysis"
    build_output = "build_output"


class OptimizationObjective(BaseModel):
    primary_metric: Literal["intent_error_rate"] = "intent_error_rate"
    target_intent_error_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    max_latency_ms: int | None = Field(default=None, ge=1)
    max_cost_per_invocation_usd: float | None = Field(default=None, ge=0.0)
    max_optimization_rounds: int = Field(default=3, ge=1, le=10)


class PipelineCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str = ""
    intent_prompt: str = Field(..., min_length=1)
    asr_provider: str = Field(
        default="whisper",
        description="ASR provider used for audio invocation.",
    )
    optimization_objective: OptimizationObjective | None = None


class PipelineSpec(BaseModel):
    pipeline_id: str
    name: str
    description: str
    status: PipelineStatus
    asr_provider: str
    optimization_objective: OptimizationObjective
    intent_schema: ArtifactRef | None = None
    eval_dataset: ArtifactRef | None = None
    published_graph: ArtifactRef | None = None
    latest_evaluation_report: ArtifactRef | None = None
    latest_adversarial_findings: ArtifactRef | None = None
    created_at: datetime
    updated_at: datetime


class PipelineCompatibilitySnapshot(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class PipelineSummaryResponse(BaseModel):
    pipeline_id: str
    name: str
    description: str
    status: PipelineStatus
    asr_provider: str
    current_intent_error_rate: float | None = None
    holdout_intent_error_rate: float | None = None
    published_graph_version: int | None = None
    active_build_step: str | None = None
    created_at: datetime
    updated_at: datetime


class PipelineListResponse(BaseModel):
    pipelines: list[PipelineSummaryResponse]


class PipelineArtifactRecord(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    version: int
    build_phase: PipelineBuildPhase
    summary: str
    created_at: datetime
    payload: (
        IntentSchemaArtifact
        | EvalDatasetArtifact
        | PipelineGraphArtifact
        | EvaluationReportArtifact
        | AdversarialFindingsArtifact
    )


class PipelineDetailResponse(BaseModel):
    pipeline: PipelineSpec
    intent_schema_artifact: IntentSchemaArtifact | None = None
    eval_dataset_artifact: EvalDatasetArtifact | None = None
    published_graph_artifact: PipelineGraphArtifact | None = None
    latest_evaluation_report_artifact: EvaluationReportArtifact | None = None
    latest_adversarial_findings_artifact: AdversarialFindingsArtifact | None = None
    artifact_history: list[PipelineArtifactRecord] = Field(default_factory=list)
    build_steps: list[PipelineBuildStep] = Field(default_factory=list)
    compatibility_snapshot: PipelineCompatibilitySnapshot = Field(
        default_factory=PipelineCompatibilitySnapshot
    )

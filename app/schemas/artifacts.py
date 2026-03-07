from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    intent_schema = "intent_schema"
    eval_dataset = "eval_dataset"
    pipeline_graph = "pipeline_graph"
    evaluation_report = "evaluation_report"
    adversarial_findings = "adversarial_findings"


class ArtifactRef(BaseModel):
    artifact_id: str
    artifact_type: ArtifactType
    pipeline_id: str
    version: int


class IntentDefinition(BaseModel):
    intent_name: str
    description: str
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    disambiguation_rules: list[str] = Field(default_factory=list)
    out_of_scope_examples: list[str] = Field(default_factory=list)


class IntentSchemaArtifact(BaseModel):
    artifact_id: str
    pipeline_id: str
    source_prompt: str
    fallback_intent: str = "unknown"
    intents: list[IntentDefinition] = Field(default_factory=list)


class EvalExample(BaseModel):
    example_id: str
    split: Literal["train", "dev", "holdout"]
    source: Literal["seed", "curated", "adversarial", "production"]
    modality: Literal["text", "audio"]
    utterance_text: str | None = None
    audio_uri: str | None = None
    expected_intent: str
    phenomenon_tags: list[str] = Field(default_factory=list)
    confusable_with: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDatasetArtifact(BaseModel):
    artifact_id: str
    pipeline_id: str
    intent_schema: ArtifactRef
    examples: list[EvalExample] = Field(default_factory=list)
    coverage_summary: dict[str, int] = Field(default_factory=dict)


class AsrComponentSpec(BaseModel):
    component_id: str
    component_name: str
    provider: str
    model: str
    language: str = "en"
    keyword_hints: list[str] = Field(default_factory=list)
    n_best: int = 1
    endpointing_config: dict[str, Any] = Field(default_factory=dict)


class TranscriptNormalizerComponentSpec(BaseModel):
    component_id: str
    component_name: str
    lowercase: bool = True
    strip_fillers: bool = True
    punctuation_policy: str = "preserve"
    canonical_replacements: dict[str, str] = Field(default_factory=dict)
    custom_rules: list[str] = Field(default_factory=list)


class IntentClassifierComponentSpec(BaseModel):
    component_id: str
    component_name: str
    provider: str
    model: str
    system_prompt: str
    candidate_count: int = 3
    intent_schema: ArtifactRef
    few_shot_examples: list[str] = Field(default_factory=list)


class CandidateRerankerComponentSpec(BaseModel):
    component_id: str
    component_name: str
    provider: str
    model: str
    rerank_prompt: str
    top_k: int = 3


class DecisionPolicyComponentSpec(BaseModel):
    component_id: str
    component_name: str
    confidence_threshold: float = 0.7
    margin_threshold: float = 0.05
    fallback_intent: str = "unknown"
    allow_abstain: bool = True
    out_of_domain_strategy: str = "fallback_intent"


class PipelineEdge(BaseModel):
    from_component_id: str
    from_output_key: str
    to_component_id: str
    to_input_key: str


class PipelineGraphArtifact(BaseModel):
    artifact_id: str
    pipeline_id: str
    entry_mode: Literal["text", "audio"] = "audio"
    asr_component: AsrComponentSpec | None = None
    normalizer_component: TranscriptNormalizerComponentSpec | None = None
    intent_classifier_component: IntentClassifierComponentSpec
    decision_policy_component: DecisionPolicyComponentSpec
    candidate_reranker_component: CandidateRerankerComponentSpec | None = None
    edges: list[PipelineEdge] = Field(default_factory=list)


class ComponentTrace(BaseModel):
    component_id: str
    component_kind: str
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int
    warnings: list[str] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    example_id: str
    split: Literal["train", "dev", "holdout"]
    expected_intent: str
    predicted_intent: str
    confidence: float
    correct: bool
    transcript_text: str | None = None
    normalized_text: str | None = None
    component_traces: list[ComponentTrace] = Field(default_factory=list)


class EvaluationReportArtifact(BaseModel):
    artifact_id: str
    pipeline_id: str
    pipeline_graph: ArtifactRef
    eval_dataset: ArtifactRef
    overall_intent_error_rate: float
    split_intent_error_rate: dict[str, float] = Field(default_factory=dict)
    per_intent_accuracy: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    hard_cases: list[EvalCaseResult] = Field(default_factory=list)


class FailureCluster(BaseModel):
    cluster_id: str
    title: str
    description: str
    affected_intents: list[str] = Field(default_factory=list)
    suspected_component_ids: list[str] = Field(default_factory=list)
    example_ids: list[str] = Field(default_factory=list)


class AdversarialFindingsArtifact(BaseModel):
    artifact_id: str
    pipeline_id: str
    evaluation_report: ArtifactRef
    failure_clusters: list[FailureCluster] = Field(default_factory=list)
    proposed_examples: list[EvalExample] = Field(default_factory=list)
    recommended_component_changes: list[str] = Field(default_factory=list)


class PipelineBuildStep(BaseModel):
    step_name: str
    status: Literal["pending", "running", "completed", "failed"]
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None

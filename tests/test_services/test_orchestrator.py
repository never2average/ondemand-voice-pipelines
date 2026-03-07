from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from app.agents.base import ArtifactGenerationResult, PipelineBuildContext
from app.agents.publishing import PublishingAgent
from app.schemas.artifacts import (
    AdversarialFindingsArtifact,
    ArtifactRef,
    ArtifactType,
    AsrComponentSpec,
    DecisionPolicyComponentSpec,
    EvalCaseResult,
    EvalDatasetArtifact,
    EvalExample,
    EvaluationReportArtifact,
    FailureCluster,
    IntentClassifierComponentSpec,
    IntentDefinition,
    IntentSchemaArtifact,
    PipelineGraphArtifact,
    TranscriptNormalizerComponentSpec,
)
from app.schemas.pipeline import OptimizationObjective
from app.services.orchestrator import PipelineOrchestrator


class InMemoryPipelineRepository:
    def __init__(self, row):
        self.row = row

    async def get_by_id(self, pipeline_id):
        return self.row if self.row["id"] == pipeline_id else None

    async def update(self, pipeline_id, data):
        assert self.row["id"] == pipeline_id
        self.row.update(data)
        return self.row


class InMemoryArtifactRepository:
    def __init__(self):
        self.rows = []

    async def create(self, data):
        self.rows.append(data)
        return data

    async def get_latest_by_type(self, pipeline_id, artifact_type):
        matching = [
            row for row in self.rows
            if row["pipeline_id"] == pipeline_id and row["artifact_type"] == artifact_type
        ]
        if not matching:
            return None
        return sorted(matching, key=lambda row: row["version"])[-1]

    async def get_by_id(self, artifact_id):
        for row in self.rows:
            if row["id"] == artifact_id:
                return row
        return None


class InMemoryBuildStepRepository:
    def __init__(self):
        self.rows = {}

    async def save(self, data):
        self.rows[(data["pipeline_id"], data["step_name"])] = data
        return data

    async def list_by_pipeline(self, pipeline_id):
        return [row for (stored_pipeline_id, _), row in self.rows.items() if stored_pipeline_id == pipeline_id]


class InMemoryEvalDatasetRepository:
    def __init__(self):
        self.rows = []

    async def create(self, data):
        self.rows.append(data)
        return data


class InMemoryEvalExampleRepository:
    def __init__(self):
        self.rows = []

    async def create_many(self, records):
        self.rows.extend(records)
        return records


class StubIntentSchemaAgent:
    agent_name = "intent_schema_agent"

    async def run(self, build_context):
        artifact = IntentSchemaArtifact(
            artifact_id="intent-schema-1",
            pipeline_id=build_context.pipeline_id,
            source_prompt=build_context.intent_prompt,
            fallback_intent="unknown",
            intents=[
                IntentDefinition(intent_name="check_balance", description="Check account balance"),
                IntentDefinition(intent_name="transfer_funds", description="Transfer funds"),
            ],
        )
        return ArtifactGenerationResult(artifact=artifact, summary="schema created")


class StubEvalDatasetAgent:
    agent_name = "eval_dataset_curator_agent"

    async def run(self, build_context, intent_schema_artifact, intent_schema_ref):
        artifact = EvalDatasetArtifact(
            artifact_id="eval-dataset-1",
            pipeline_id=build_context.pipeline_id,
            intent_schema=intent_schema_ref,
            examples=[
                EvalExample(
                    example_id="ex-train-1",
                    split="train",
                    source="curated",
                    modality="text",
                    utterance_text="check balance",
                    expected_intent="check_balance",
                ),
                EvalExample(
                    example_id="ex-dev-1",
                    split="dev",
                    source="curated",
                    modality="text",
                    utterance_text="transfer funds",
                    expected_intent="transfer_funds",
                ),
                EvalExample(
                    example_id="ex-holdout-1",
                    split="holdout",
                    source="curated",
                    modality="text",
                    utterance_text="check my balance",
                    expected_intent="check_balance",
                ),
            ],
            coverage_summary={"train": 1, "dev": 1, "holdout": 1},
        )
        return ArtifactGenerationResult(artifact=artifact, summary="dataset created")


class StubBaselineGraphPlannerAgent:
    agent_name = "baseline_graph_planner_agent"

    async def run(self, build_context, intent_schema_artifact, intent_schema_ref):
        artifact = PipelineGraphArtifact(
            artifact_id="graph-1",
            pipeline_id=build_context.pipeline_id,
            entry_mode="audio",
            asr_component=AsrComponentSpec(
                component_id="asr",
                component_name="ASR",
                provider="whisper",
                model="whisper-1",
            ),
            normalizer_component=TranscriptNormalizerComponentSpec(
                component_id="normalizer",
                component_name="Normalizer",
                lowercase=True,
                strip_fillers=True,
                punctuation_policy="preserve",
                canonical_replacements={},
                custom_rules=[],
            ),
            intent_classifier_component=IntentClassifierComponentSpec(
                component_id="classifier",
                component_name="Classifier",
                provider="claude",
                model="claude-sonnet-4-20250514",
                system_prompt="prompt",
                candidate_count=2,
                intent_schema=intent_schema_ref,
                few_shot_examples=[],
            ),
            decision_policy_component=DecisionPolicyComponentSpec(
                component_id="policy",
                component_name="Policy",
                confidence_threshold=0.6,
                margin_threshold=0.05,
                fallback_intent="unknown",
                allow_abstain=True,
                out_of_domain_strategy="fallback_intent",
            ),
        )
        return ArtifactGenerationResult(artifact=artifact, summary="graph created")


class StubPipelineEvaluatorAgent:
    agent_name = "pipeline_evaluator_agent"

    def __init__(self):
        self.optimization_call_count = 0

    async def run(self, build_context, intent_schema_artifact, eval_dataset_artifact, pipeline_graph_artifact, pipeline_graph_ref, eval_dataset_ref, include_splits):
        if include_splits == ["holdout"]:
            split_ier = {"holdout": 0.0}
            overall_ier = 0.0
            hard_cases = []
            artifact_id = "holdout-report"
        else:
            self.optimization_call_count += 1
            overall_ier = 0.4 if self.optimization_call_count == 1 else 0.0
            split_ier = {"train": overall_ier, "dev": overall_ier}
            hard_cases = [] if overall_ier == 0.0 else [
                EvalCaseResult(
                    example_id="ex-dev-1",
                    split="dev",
                    expected_intent="transfer_funds",
                    predicted_intent="check_balance",
                    confidence=0.4,
                    correct=False,
                    transcript_text="transfer funds",
                    normalized_text="transfer funds",
                    component_traces=[],
                )
            ]
            artifact_id = f"optimization-report-{self.optimization_call_count}"

        artifact = EvaluationReportArtifact(
            artifact_id=artifact_id,
            pipeline_id=build_context.pipeline_id,
            pipeline_graph=pipeline_graph_ref,
            eval_dataset=eval_dataset_ref,
            overall_intent_error_rate=overall_ier,
            split_intent_error_rate=split_ier,
            per_intent_accuracy={"check_balance": 1.0, "transfer_funds": 1.0 - overall_ier},
            confusion_matrix={"transfer_funds": {"check_balance": 1}} if hard_cases else {},
            hard_cases=hard_cases,
        )
        return ArtifactGenerationResult(artifact=artifact, summary="evaluation complete")


class StubAdversarialDatasetAgent:
    agent_name = "adversarial_dataset_agent"

    async def run(self, build_context, intent_schema_artifact, eval_dataset_artifact, evaluation_report_artifact, evaluation_report_ref):
        artifact = AdversarialFindingsArtifact(
            artifact_id="adversarial-findings-1",
            pipeline_id=build_context.pipeline_id,
            evaluation_report=evaluation_report_ref,
            failure_clusters=[
                FailureCluster(
                    cluster_id="cluster-1",
                    title="transfer confused with balance",
                    description="Need better separation",
                    affected_intents=["transfer_funds", "check_balance"],
                    suspected_component_ids=["classifier"],
                    example_ids=["ex-dev-1"],
                )
            ],
            proposed_examples=[
                EvalExample(
                    example_id="adv-example-1",
                    split="train",
                    source="adversarial",
                    modality="text",
                    utterance_text="please transfer funds right now",
                    expected_intent="transfer_funds",
                )
            ],
            recommended_component_changes=["Expand intent disambiguation"],
        )
        return ArtifactGenerationResult(artifact=artifact, summary="findings created")


class StubGraphRevisionAgent:
    agent_name = "graph_revision_agent"

    async def run(self, build_context, intent_schema_artifact, pipeline_graph_artifact, evaluation_report_artifact, adversarial_findings_artifact):
        revised_graph = pipeline_graph_artifact.model_copy(deep=True)
        revised_graph.artifact_id = "graph-2"
        revised_graph.decision_policy_component.confidence_threshold = 0.5
        return ArtifactGenerationResult(artifact=revised_graph, summary="graph revised")


@pytest.mark.asyncio
async def test_pipeline_orchestrator_persists_artifacts_and_publishes_pipeline():
    created_at = datetime.now(timezone.utc).isoformat()
    pipeline_repo = InMemoryPipelineRepository(
        {
            "id": "pipe-1",
            "name": "Banking Pipeline",
            "description": "Intent pipeline",
            "intent_prompt": "check balance, transfer funds",
            "status": "pending",
            "config": {},
            "metrics": {},
            "asr_provider": "whisper",
            "optimization_objective": OptimizationObjective(target_intent_error_rate=0.1).model_dump(mode="json"),
            "created_at": created_at,
            "updated_at": created_at,
        }
    )
    artifact_repo = InMemoryArtifactRepository()
    build_step_repo = InMemoryBuildStepRepository()
    eval_dataset_repo = InMemoryEvalDatasetRepository()
    eval_example_repo = InMemoryEvalExampleRepository()

    orchestrator = PipelineOrchestrator(
        pipeline_repo=pipeline_repo,
        pipeline_artifact_repo=artifact_repo,
        pipeline_build_step_repo=build_step_repo,
        eval_dataset_repo=eval_dataset_repo,
        eval_example_repo=eval_example_repo,
        intent_schema_agent=StubIntentSchemaAgent(),
        eval_dataset_curator_agent=StubEvalDatasetAgent(),
        baseline_graph_planner_agent=StubBaselineGraphPlannerAgent(),
        pipeline_evaluator_agent=StubPipelineEvaluatorAgent(),
        adversarial_dataset_agent=StubAdversarialDatasetAgent(),
        graph_revision_agent=StubGraphRevisionAgent(),
        publishing_agent=PublishingAgent(),
    )

    await orchestrator.build_pipeline(
        PipelineBuildContext(
            pipeline_id="pipe-1",
            pipeline_name="Banking Pipeline",
            pipeline_description="Intent pipeline",
            intent_prompt="check balance, transfer funds",
            asr_provider="whisper",
            optimization_objective=OptimizationObjective(target_intent_error_rate=0.1, max_optimization_rounds=2),
        )
    )

    assert pipeline_repo.row["status"] == "ready"
    assert pipeline_repo.row["published_graph_version"] == 2
    assert pipeline_repo.row["current_intent_error_rate"] == 0.0
    assert pipeline_repo.row["holdout_intent_error_rate"] == 0.0
    assert len(artifact_repo.rows) >= 7
    assert any(row["artifact_type"] == ArtifactType.intent_schema.value for row in artifact_repo.rows)
    assert any(row["artifact_type"] == ArtifactType.pipeline_graph.value for row in artifact_repo.rows)
    assert ("pipe-1", "publishing") in build_step_repo.rows
    assert len(eval_dataset_repo.rows) == 2
    assert len(eval_example_repo.rows) >= 4

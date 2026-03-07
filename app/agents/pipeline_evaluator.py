from __future__ import annotations

from collections import defaultdict

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext, new_artifact_id
from app.schemas.artifacts import (
    ArtifactRef,
    EvalCaseResult,
    EvalDatasetArtifact,
    EvaluationReportArtifact,
    IntentSchemaArtifact,
    PipelineGraphArtifact,
)
from app.services.graph_runner import PipelineGraphRunner


class PipelineEvaluatorAgent(BaseArtifactAgent[EvaluationReportArtifact]):
    agent_name = "pipeline_evaluator_agent"

    def __init__(self, graph_runner: PipelineGraphRunner | None = None):
        self._graph_runner = graph_runner or PipelineGraphRunner()

    async def run(
        self,
        build_context: PipelineBuildContext,
        intent_schema_artifact: IntentSchemaArtifact,
        eval_dataset_artifact: EvalDatasetArtifact,
        pipeline_graph_artifact: PipelineGraphArtifact,
        pipeline_graph_ref: ArtifactRef,
        eval_dataset_ref: ArtifactRef,
        include_splits: list[str],
    ) -> ArtifactGenerationResult[EvaluationReportArtifact]:
        filtered_examples = [
            example
            for example in eval_dataset_artifact.examples
            if example.split in include_splits
        ]
        case_results: list[EvalCaseResult] = []
        split_totals: dict[str, int] = defaultdict(int)
        split_errors: dict[str, int] = defaultdict(int)
        per_intent_totals: dict[str, int] = defaultdict(int)
        per_intent_correct: dict[str, int] = defaultdict(int)
        confusion_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for example in filtered_examples:
            invocation_result = await self._graph_runner.run(
                pipeline_id=build_context.pipeline_id,
                input_type="text",
                input_text=example.utterance_text,
                input_audio_bytes=None,
                pipeline_graph_artifact=pipeline_graph_artifact,
                intent_schema_artifact=intent_schema_artifact,
            )
            correct = invocation_result.detected_intent == example.expected_intent
            split_totals[example.split] += 1
            split_errors[example.split] += 0 if correct else 1
            per_intent_totals[example.expected_intent] += 1
            per_intent_correct[example.expected_intent] += 1 if correct else 0
            confusion_matrix[example.expected_intent][invocation_result.detected_intent] += 1
            case_results.append(
                EvalCaseResult(
                    example_id=example.example_id,
                    split=example.split,
                    expected_intent=example.expected_intent,
                    predicted_intent=invocation_result.detected_intent,
                    confidence=invocation_result.confidence,
                    correct=correct,
                    transcript_text=invocation_result.transcript_text,
                    normalized_text=invocation_result.normalized_text,
                    component_traces=invocation_result.component_traces,
                )
            )

        total_examples = len(case_results)
        total_errors = sum(1 for case_result in case_results if not case_result.correct)
        overall_intent_error_rate = total_errors / total_examples if total_examples else 1.0
        split_intent_error_rate = {
            split_name: (split_errors[split_name] / split_totals[split_name] if split_totals[split_name] else 1.0)
            for split_name in include_splits
        }
        per_intent_accuracy = {
            intent_name: (
                per_intent_correct[intent_name] / per_intent_totals[intent_name]
                if per_intent_totals[intent_name]
                else 0.0
            )
            for intent_name in per_intent_totals
        }
        hard_cases = sorted(
            case_results,
            key=lambda case_result: (case_result.correct, case_result.confidence),
        )[:20]

        artifact = EvaluationReportArtifact(
            artifact_id=new_artifact_id(),
            pipeline_id=build_context.pipeline_id,
            pipeline_graph=pipeline_graph_ref,
            eval_dataset=eval_dataset_ref,
            overall_intent_error_rate=overall_intent_error_rate,
            split_intent_error_rate=split_intent_error_rate,
            per_intent_accuracy=per_intent_accuracy,
            confusion_matrix={
                expected_intent: dict(predicted_counts)
                for expected_intent, predicted_counts in confusion_matrix.items()
            },
            hard_cases=hard_cases,
        )
        summary = (
            f"Evaluated {total_examples} examples across splits {', '.join(include_splits)} "
            f"with overall IER {overall_intent_error_rate:.4f}."
        )
        return ArtifactGenerationResult(artifact=artifact, summary=summary)

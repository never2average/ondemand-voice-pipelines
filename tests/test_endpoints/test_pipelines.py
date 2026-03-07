from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_pipeline_service
from app.main import app
from app.schemas.artifacts import ArtifactRef, ArtifactType
from app.schemas.artifacts import PipelineBuildStep
from app.schemas.invocation import IntentCandidate, InvokeResponse
from app.schemas.pipeline import (
    OptimizationObjective,
    PipelineCompatibilitySnapshot,
    PipelineDetailResponse,
    PipelineListResponse,
    PipelineSpec,
    PipelineStatus,
    PipelineSummaryResponse,
)


@pytest.fixture
def mock_service():
    return AsyncMock()


@pytest.fixture
def client(mock_service):
    app.dependency_overrides[get_pipeline_service] = lambda: mock_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _make_pipeline_detail_response(status: PipelineStatus = PipelineStatus.pending) -> PipelineDetailResponse:
    now = datetime.now(timezone.utc)
    return PipelineDetailResponse(
        pipeline=PipelineSpec(
            pipeline_id="pipe-1",
            name="Banking Pipeline",
            description="Intent pipeline",
            status=status,
            asr_provider="whisper",
            optimization_objective=OptimizationObjective(),
            intent_schema=None,
            eval_dataset=None,
            published_graph=None,
            latest_evaluation_report=None,
            latest_adversarial_findings=None,
            created_at=now,
            updated_at=now,
        ),
        build_steps=[
            PipelineBuildStep(
                step_name="intent_schema_generation",
                status="completed",
                started_at=now,
                completed_at=now,
                summary="done",
            )
        ],
        compatibility_snapshot=PipelineCompatibilitySnapshot(),
    )


def _make_pipeline_summary(status: PipelineStatus = PipelineStatus.pending) -> PipelineSummaryResponse:
    now = datetime.now(timezone.utc)
    return PipelineSummaryResponse(
        pipeline_id="pipe-1",
        name="Banking Pipeline",
        description="Intent pipeline",
        status=status,
        asr_provider="whisper",
        current_intent_error_rate=0.1,
        holdout_intent_error_rate=0.05,
        published_graph_version=2,
        active_build_step="publishing",
        created_at=now,
        updated_at=now,
    )


class TestListPipelines:
    def test_returns_empty_list(self, client, mock_service):
        mock_service.list_pipelines.return_value = PipelineListResponse(pipelines=[])
        response = client.get("/api/v1/pipelines")
        assert response.status_code == 200
        assert response.json()["pipelines"] == []

    def test_returns_pipeline_summaries(self, client, mock_service):
        mock_service.list_pipelines.return_value = PipelineListResponse(
            pipelines=[_make_pipeline_summary(status=PipelineStatus.ready)]
        )
        response = client.get("/api/v1/pipelines")
        assert response.status_code == 200
        assert response.json()["pipelines"][0]["published_graph_version"] == 2
        assert response.json()["pipelines"][0]["status"] == "ready"


class TestCreatePipeline:
    def test_returns_accepted_detail_response(self, client, mock_service):
        mock_service.create_pipeline.return_value = _make_pipeline_detail_response()
        response = client.post(
            "/api/v1/pipelines",
            json={
                "name": "Banking Pipeline",
                "intent_prompt": "check balance, transfer funds",
            },
        )
        assert response.status_code == 202
        assert response.json()["pipeline"]["pipeline_id"] == "pipe-1"

    def test_rejects_missing_prompt(self, client, mock_service):
        response = client.post(
            "/api/v1/pipelines",
            json={"name": "Banking Pipeline"},
        )
        assert response.status_code == 422


class TestGetPipeline:
    def test_returns_pipeline_detail(self, client, mock_service):
        mock_service.get_pipeline.return_value = _make_pipeline_detail_response(
            status=PipelineStatus.ready
        )
        response = client.get("/api/v1/pipelines/pipe-1")
        assert response.status_code == 200
        assert response.json()["pipeline"]["status"] == "ready"
        assert response.json()["artifact_history"] == []
        assert len(response.json()["build_steps"]) == 1

    def test_returns_404_for_missing_pipeline(self, client, mock_service):
        from app.core.exceptions import PipelineNotFoundError

        mock_service.get_pipeline.side_effect = PipelineNotFoundError("missing")
        response = client.get("/api/v1/pipelines/missing")
        assert response.status_code == 404


class TestInvokePipeline:
    def test_returns_graph_aware_invoke_response(self, client, mock_service):
        now = datetime.now(timezone.utc)
        mock_service.invoke_pipeline.return_value = InvokeResponse(
            id="inv-1",
            pipeline_id="pipe-1",
            input_type="text",
            input_text="check balance",
            normalized_text="check balance",
            detected_intent="check_balance",
            confidence=0.93,
            intent_candidates=[IntentCandidate(intent_name="check_balance", confidence=0.93)],
            latency_ms=45,
            component_traces=[],
            pipeline_graph_artifact=ArtifactRef(
                artifact_id="graph-1",
                artifact_type=ArtifactType.pipeline_graph,
                pipeline_id="pipe-1",
                version=2,
            ),
            pipeline_graph_version=2,
            metadata={},
            created_at=now,
        )
        response = client.post(
            "/api/v1/pipelines/pipe-1/invoke",
            json={"input_text": "check balance", "input_type": "text"},
        )
        assert response.status_code == 200
        assert response.json()["pipeline_graph_version"] == 2
        assert response.json()["detected_intent"] == "check_balance"

    def test_returns_409_when_pipeline_not_ready(self, client, mock_service):
        from app.core.exceptions import PipelineNotReadyError

        mock_service.invoke_pipeline.side_effect = PipelineNotReadyError("pipe-1", "building")
        response = client.post(
            "/api/v1/pipelines/pipe-1/invoke",
            json={"input_text": "check balance", "input_type": "text"},
        )
        assert response.status_code == 409

    def test_rejects_missing_text_for_text_invocation(self, client, mock_service):
        response = client.post(
            "/api/v1/pipelines/pipe-1/invoke",
            json={"input_type": "text"},
        )
        assert response.status_code == 422

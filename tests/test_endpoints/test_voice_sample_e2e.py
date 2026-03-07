from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db.repositories.in_memory import (
    InMemoryArtifactRepository,
    InMemoryBuildStepRepository,
    InMemoryEvalDatasetRepository,
    InMemoryEvalExampleRepository,
    InMemoryInvocationRepository,
    InMemoryPipelineRepository,
)
from app.dependencies import get_pipeline_service
from app.main import app
from app.services.pipeline_service import PipelineService


@pytest.fixture
def voice_sample_client():
    service = PipelineService(
        pipeline_repo=InMemoryPipelineRepository(),
        eval_dataset_repo=InMemoryEvalDatasetRepository(),
        eval_example_repo=InMemoryEvalExampleRepository(),
        pipeline_artifact_repo=InMemoryArtifactRepository(),
        pipeline_build_step_repo=InMemoryBuildStepRepository(),
        invocation_repo=InMemoryInvocationRepository(),
    )
    app.dependency_overrides[get_pipeline_service] = lambda: service
    with TestClient(app) as client:
        yield client, service
    app.dependency_overrides.clear()


def test_voice_sample_flows_across_create_list_detail_and_invoke(voice_sample_client):
    client, service = voice_sample_client
    sample_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "voice_samples"
        / "check-my-balance.wav"
    )
    audio_payload = base64.b64encode(sample_path.read_bytes()).decode("utf-8")

    create_response = client.post(
        "/api/v1/pipelines",
        json={
            "name": "banking-voice-sample",
            "description": "Exercise the local sample voice workflow.",
            "intent_prompt": "\n".join(
                [
                    "check balance: Handle requests about checking an account balance.",
                    "transfer funds: Handle requests about moving money between accounts.",
                ]
            ),
            "asr_provider": "sample",
        },
    )
    assert create_response.status_code == 202
    pipeline_id = create_response.json()["pipeline"]["pipeline_id"]

    list_response = client.get("/api/v1/pipelines")
    assert list_response.status_code == 200
    assert list_response.json()["pipelines"][0]["pipeline_id"] == pipeline_id
    assert list_response.json()["pipelines"][0]["asr_provider"] == "sample"

    detail_payload = None
    for _ in range(20):
        detail_response = client.get(f"/api/v1/pipelines/{pipeline_id}")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        if detail_payload["pipeline"]["status"] == "ready":
            break
        time.sleep(0.01)

    assert detail_payload is not None
    assert detail_payload["pipeline"]["status"] == "ready"
    assert detail_payload["published_graph_artifact"]["asr_component"]["provider"] == "sample"
    assert detail_payload["published_graph_artifact"]["asr_component"]["model"] == "fixture-transcript-v1"
    assert [artifact["producer_agent"] for artifact in detail_payload["artifact_history"]] == [
        "intent_schema_agent",
        "eval_dataset_curator_agent",
        "baseline_graph_planner_agent",
        "pipeline_evaluator_agent",
        "pipeline_evaluator_agent",
    ]
    assert [artifact["artifact_type"] for artifact in detail_payload["artifact_history"]] == [
        "intent_schema",
        "eval_dataset",
        "pipeline_graph",
        "evaluation_report",
        "evaluation_report",
    ]
    assert detail_payload["artifact_history"][0]["payload"]["intents"][0]["intent_name"] == "check_balance"

    invoke_response = client.post(
        f"/api/v1/pipelines/{pipeline_id}/invoke",
        json={
            "input_type": "audio",
            "input_audio_base64": audio_payload,
        },
    )
    assert invoke_response.status_code == 200
    invoke_payload = invoke_response.json()
    assert invoke_payload["input_text"] == "check my balance"
    assert invoke_payload["normalized_text"] == "check my balance"
    assert invoke_payload["detected_intent"] == "check_balance"
    assert invoke_payload["pipeline_graph_version"] == 1
    assert [trace["component_kind"] for trace in invoke_payload["component_traces"]] == [
        "asr",
        "normalizer",
        "intent_classifier",
        "decision_policy",
    ]

    persisted_invocation = next(iter(service._invocation_repo.rows.values()))
    assert persisted_invocation["pipeline_id"] == pipeline_id
    assert persisted_invocation["input_type"] == "audio"
    assert persisted_invocation["input_text"] == "check my balance"

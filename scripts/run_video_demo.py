from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx


DEFAULT_INTENT_PROMPT = "\n".join(
    [
        "I'm setting up a phone support line for a retail bank.",
        "Customers usually say things like:",
        '- "I want to check my balance"',
        '- "I need to transfer money between accounts"',
        '- "I need to dispute a charge on my card"',
        '- "I need a replacement card because mine is lost"',
        "If it's something else, send it to unknown.",
    ]
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create, monitor, list, and invoke a realistic banking voice pipeline "
            "through the public API."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument(
        "--sample-path",
        default=str(
            Path(__file__).resolve().parents[1]
            / "examples"
            / "voice_samples"
            / "check-my-balance.wav"
        ),
        help="Path to the WAV file used for the invoke step",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=15.0,
        help="How long to poll for pipeline readiness",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.5,
        help="Polling cadence for build monitoring",
    )
    parser.add_argument(
        "--pipeline-name",
        default="retail-banking-voice-router",
        help="Pipeline name used in the create request",
    )
    parser.add_argument(
        "--pipeline-description",
        default=(
            "Route inbound retail banking voice calls by intent, optimizing for intent "
            "error rate instead of transcript fidelity."
        ),
        help="Pipeline description used in the create request",
    )
    parser.add_argument(
        "--intent-prompt",
        default=DEFAULT_INTENT_PROMPT,
        help="Intent prompt used to generate the pipeline",
    )
    parser.add_argument(
        "--asr-provider",
        default="sample",
        help="ASR provider for the generated pipeline",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample_path)
    if not sample_path.exists():
        raise FileNotFoundError(f"Audio sample not found: {sample_path}")

    audio_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")
    create_payload = {
        "name": args.pipeline_name,
        "description": args.pipeline_description,
        "intent_prompt": args.intent_prompt,
        "asr_provider": args.asr_provider,
        "optimization_objective": {
            "target_intent_error_rate": 0.05,
            "max_optimization_rounds": 3,
        },
    }

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        print("== Create Pipeline ==")
        print(json.dumps(create_payload, indent=2))
        create_response = client.post("/api/v1/pipelines", json=create_payload)
        create_response.raise_for_status()
        create_body = create_response.json()
        pipeline_id = create_body["pipeline"]["pipeline_id"]
        print(f"pipeline_id={pipeline_id}")
        print(f"initial_status={create_body['pipeline']['status']}")
        print()

        print("== Monitor Build ==")
        detail = create_body
        _monitor_pipeline_until_ready(
            client=client,
            pipeline_id=pipeline_id,
            initial_detail=detail,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        detail_response = client.get(f"/api/v1/pipelines/{pipeline_id}")
        detail_response.raise_for_status()
        detail = detail_response.json()
        print()

        print("== List Active Pipelines ==")
        list_response = client.get("/api/v1/pipelines")
        list_response.raise_for_status()
        pipelines = list_response.json()["pipelines"]
        matching_pipeline = next(
            pipeline
            for pipeline in pipelines
            if pipeline["pipeline_id"] == pipeline_id
        )
        print(json.dumps(matching_pipeline, indent=2))
        print()

        print("== Invoke With Audio ==")
        invoke_payload = {
            "input_type": "audio",
            "input_audio_base64": audio_b64,
        }
        invoke_response = client.post(
            f"/api/v1/pipelines/{pipeline_id}/invoke",
            json=invoke_payload,
        )
        invoke_response.raise_for_status()
        invoke_body = invoke_response.json()
        print(f"transcript={invoke_body['input_text']}")
        print(f"normalized_text={invoke_body['normalized_text']}")
        print(f"detected_intent={invoke_body['detected_intent']}")
        print(f"confidence={invoke_body['confidence']}")
        print("intent_candidates=" + json.dumps(invoke_body["intent_candidates"], indent=2))
        print("component_traces=" + json.dumps(invoke_body["component_traces"], indent=2))
        print()

    print("== Final Summary ==")
    summary = {
        "pipeline_id": pipeline_id,
        "status": detail["pipeline"]["status"],
        "published_graph_version": detail["pipeline"]["published_graph"]["version"],
        "current_intent_error_rate": matching_pipeline.get("current_intent_error_rate"),
        "holdout_intent_error_rate": matching_pipeline.get("holdout_intent_error_rate"),
        "invoked_intent": invoke_body["detected_intent"],
        "sample_path": str(sample_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _monitor_pipeline_until_ready(
    client: httpx.Client,
    pipeline_id: str,
    initial_detail: dict,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    seen_step_states: set[tuple[str, str]] = set()
    seen_artifacts: set[tuple[str, int]] = set()
    last_status: str | None = None
    last_active_step: str | None = None
    deadline = time.time() + timeout_seconds
    detail = initial_detail

    while True:
        pipeline = detail["pipeline"]
        status = pipeline["status"]
        active_step = _derive_active_step(detail.get("build_steps", []))

        if status != last_status or active_step != last_active_step:
            print(
                f"status={status} active_build_step={active_step}"
            )
            last_status = status
            last_active_step = active_step

        for step in detail.get("build_steps", []):
            key = (step["step_name"], step["status"])
            if key in seen_step_states:
                continue
            seen_step_states.add(key)
            print(
                f"step={step['step_name']} status={step['status']} "
                f"summary={step.get('summary', '') or '-'}"
            )
            if step.get("error"):
                print(f"step_error={step['error']}")

        for artifact in detail.get("artifact_history", []):
            key = (artifact["artifact_id"], int(artifact["version"]))
            if key in seen_artifacts:
                continue
            seen_artifacts.add(key)
            print(
                f"artifact={artifact['artifact_type']} version={artifact['version']} "
                f"build_phase={artifact['build_phase']} summary={artifact['summary']}"
            )

        if status == "ready":
            return
        if status == "failed":
            raise RuntimeError(f"Pipeline {pipeline_id} failed during build.")
        if time.time() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for pipeline {pipeline_id} to become ready."
            )

        time.sleep(poll_interval_seconds)
        detail_response = client.get(f"/api/v1/pipelines/{pipeline_id}")
        detail_response.raise_for_status()
        detail = detail_response.json()


def _derive_active_step(build_steps: list[dict]) -> str | None:
    running_steps = [step for step in build_steps if step.get("status") == "running"]
    if running_steps:
        return str(running_steps[-1]["step_name"])
    if build_steps:
        return str(build_steps[-1]["step_name"])
    return None


if __name__ == "__main__":
    raise SystemExit(main())

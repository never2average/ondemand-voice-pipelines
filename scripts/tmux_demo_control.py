from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import httpx

from demo_scenario import (
    DEFAULT_INTENT_PROMPT,
    DEFAULT_PIPELINE_DESCRIPTION,
    DEFAULT_PIPELINE_NAME,
    DEFAULT_SAMPLE_PATH,
)
from demo_state import ensure_state_dir, write_done, write_pipeline_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive create and invoke actions for the four-pane tmux demo."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--state-dir", default="artifacts/tmux-demo")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.4)
    parser.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
    parser.add_argument("--asr-provider", default="sample")
    args = parser.parse_args()

    state_dir = ensure_state_dir(args.state_dir)
    sample_path = Path(args.sample_path)
    audio_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")

    create_payload = {
        "name": DEFAULT_PIPELINE_NAME,
        "description": DEFAULT_PIPELINE_DESCRIPTION,
        "intent_prompt": DEFAULT_INTENT_PROMPT,
        "asr_provider": args.asr_provider,
        "optimization_objective": {
            "target_intent_error_rate": 0.05,
            "max_optimization_rounds": 3,
        },
    }

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        _wait_for_health(client, args.timeout_seconds)

        print("[create] POST /api/v1/pipelines")
        print(json.dumps(create_payload, indent=2))
        create_response = client.post("/api/v1/pipelines", json=create_payload)
        create_response.raise_for_status()
        create_body = create_response.json()
        pipeline_id = create_body["pipeline"]["pipeline_id"]
        write_pipeline_id(state_dir, pipeline_id)
        print(f"[create] pipeline_id={pipeline_id}")
        print(f"[create] status={create_body['pipeline']['status']}")
        print()

        ready_detail = _wait_for_ready(
            client=client,
            pipeline_id=pipeline_id,
            timeout_seconds=args.timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        print(f"[ready] status={ready_detail['pipeline']['status']}")
        time.sleep(1.0)

        print("[invoke] POST /api/v1/pipelines/{pipeline_id}/invoke")
        invoke_response = client.post(
            f"/api/v1/pipelines/{pipeline_id}/invoke",
            json={
                "input_type": "audio",
                "input_audio_base64": audio_b64,
            },
        )
        invoke_response.raise_for_status()
        invoke_body = invoke_response.json()
        print(f"[invoke] transcript={invoke_body['input_text']}")
        print(f"[invoke] normalized_text={invoke_body['normalized_text']}")
        print(f"[invoke] detected_intent={invoke_body['detected_intent']}")
        print(f"[invoke] confidence={invoke_body['confidence']}")
        print("[invoke] intent_candidates=" + json.dumps(invoke_body["intent_candidates"], indent=2))
        print("[invoke] component_traces=" + json.dumps(invoke_body["component_traces"], indent=2))

        write_done(
            state_dir,
            {
                "pipeline_id": pipeline_id,
                "status": ready_detail["pipeline"]["status"],
                "invoked_intent": invoke_body["detected_intent"],
                "sample_path": str(sample_path),
            },
        )
    return 0


def _wait_for_health(client: httpx.Client, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                print("[health] API server is ready")
                print()
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError("Timed out waiting for the local demo server to become healthy.")


def _wait_for_ready(
    client: httpx.Client,
    pipeline_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict:
    deadline = time.time() + timeout_seconds
    last_status: str | None = None

    while time.time() < deadline:
        response = client.get(f"/api/v1/pipelines/{pipeline_id}")
        response.raise_for_status()
        detail = response.json()
        status = detail["pipeline"]["status"]
        if status != last_status:
            print(f"[build] status={status}")
            last_status = status
        if status == "ready":
            return detail
        if status == "failed":
            raise RuntimeError(f"Pipeline {pipeline_id} failed during build.")
        time.sleep(poll_interval_seconds)

    raise TimeoutError(f"Timed out waiting for pipeline {pipeline_id} to become ready.")


if __name__ == "__main__":
    raise SystemExit(main())

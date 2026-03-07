from __future__ import annotations

import argparse
import time

import httpx

from demo_state import done_path, wait_for_pipeline_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch the pipeline detail endpoint during the tmux demo."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--state-dir", default="artifacts/tmux-demo")
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.4)
    args = parser.parse_args()

    pipeline_id = wait_for_pipeline_id(args.state_dir, args.timeout_seconds)
    print(f"[detail] monitoring pipeline_id={pipeline_id}")

    seen_step_states: set[tuple[str, str]] = set()
    seen_artifacts: set[tuple[str, int]] = set()
    last_status: str | None = None
    ready_seen = False

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        deadline = time.time() + args.timeout_seconds
        while time.time() < deadline:
            response = client.get(f"/api/v1/pipelines/{pipeline_id}")
            response.raise_for_status()
            detail = response.json()
            status = detail["pipeline"]["status"]

            if status != last_status:
                print(f"[detail] status={status}")
                last_status = status

            for step in detail.get("build_steps", []):
                step_key = (step["step_name"], step["status"])
                if step_key in seen_step_states:
                    continue
                seen_step_states.add(step_key)
                print(
                    f"[detail] step={step['step_name']} status={step['status']} "
                    f"summary={step.get('summary', '') or '-'}"
                )

            for artifact in detail.get("artifact_history", []):
                artifact_key = (artifact["artifact_id"], int(artifact["version"]))
                if artifact_key in seen_artifacts:
                    continue
                seen_artifacts.add(artifact_key)
                print(
                    f"[detail] artifact={artifact['artifact_type']} "
                    f"version={artifact['version']} build_phase={artifact['build_phase']}"
                )

            if status == "ready" and not ready_seen:
                ready_seen = True
                print(
                    "[detail] published_graph_version="
                    f"{detail['pipeline']['published_graph']['version']}"
                )

            if ready_seen and done_path(args.state_dir).exists():
                return 0

            time.sleep(args.poll_interval_seconds)

    raise TimeoutError(f"Timed out monitoring pipeline detail for {pipeline_id}.")


if __name__ == "__main__":
    raise SystemExit(main())

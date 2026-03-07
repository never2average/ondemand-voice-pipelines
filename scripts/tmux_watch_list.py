from __future__ import annotations

import argparse
import json
import time

import httpx

from demo_state import done_path, read_pipeline_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch the pipeline list endpoint during the tmux demo."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--state-dir", default="artifacts/tmux-demo")
    parser.add_argument("--timeout-seconds", type=float, default=25.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.7)
    args = parser.parse_args()

    last_snapshot: str | None = None
    target_ready_seen = False

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        deadline = time.time() + args.timeout_seconds
        while time.time() < deadline:
            response = client.get("/api/v1/pipelines")
            response.raise_for_status()
            pipelines = response.json()["pipelines"]
            target_pipeline_id = read_pipeline_id(args.state_dir)

            if target_pipeline_id:
                filtered = [
                    pipeline
                    for pipeline in pipelines
                    if pipeline["pipeline_id"] == target_pipeline_id
                ]
            else:
                filtered = pipelines[:3]

            snapshot = json.dumps(filtered, indent=2, sort_keys=True)
            if snapshot != last_snapshot:
                print("[list] GET /api/v1/pipelines")
                print(snapshot)
                print()
                last_snapshot = snapshot

            if filtered and filtered[0]["status"] == "ready":
                target_ready_seen = True
            if target_ready_seen and done_path(args.state_dir).exists():
                return 0

            time.sleep(args.poll_interval_seconds)

    raise TimeoutError("Timed out monitoring the pipeline list.")


if __name__ == "__main__":
    raise SystemExit(main())

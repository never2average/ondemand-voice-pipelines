from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the committed voice sample through create/list/detail/invoke APIs."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument(
        "--sample-path",
        default=str(Path(__file__).resolve().parents[1] / "examples" / "voice_samples" / "check-my-balance.wav"),
        help="Path to the demo WAV file",
    )
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Polling timeout for pipeline readiness")
    args = parser.parse_args()

    sample_path = Path(args.sample_path)
    audio_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")
    create_payload = {
        "name": "banking-voice-sample-demo",
        "description": "Local demo pipeline exercised with the committed voice sample.",
        "intent_prompt": "\n".join(
            [
                "check balance: Handle requests about checking an account balance.",
                "transfer funds: Handle requests about moving money between accounts.",
            ]
        ),
        "asr_provider": "sample",
    }

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        created = client.post("/api/v1/pipelines", json=create_payload)
        created.raise_for_status()
        pipeline_id = created.json()["pipeline"]["pipeline_id"]

        deadline = time.time() + args.timeout_seconds
        detail = created.json()
        while detail["pipeline"]["status"] in {"pending", "building"} and time.time() < deadline:
            time.sleep(0.25)
            detail_response = client.get(f"/api/v1/pipelines/{pipeline_id}")
            detail_response.raise_for_status()
            detail = detail_response.json()

        if detail["pipeline"]["status"] != "ready":
            raise RuntimeError(
                f"Pipeline {pipeline_id} did not become ready. Final status: {detail['pipeline']['status']}"
            )

        listed = client.get("/api/v1/pipelines")
        listed.raise_for_status()

        invoked = client.post(
            f"/api/v1/pipelines/{pipeline_id}/invoke",
            json={"input_type": "audio", "input_audio_base64": audio_b64},
        )
        invoked.raise_for_status()

    output = {
        "created_pipeline_id": pipeline_id,
        "list_count": len(listed.json()["pipelines"]),
        "detail_status": detail["pipeline"]["status"],
        "invoke_detected_intent": invoked.json()["detected_intent"],
        "invoke_transcript": invoked.json()["input_text"],
        "pipeline_graph_version": invoked.json()["pipeline_graph_version"],
        "sample_path": str(sample_path),
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

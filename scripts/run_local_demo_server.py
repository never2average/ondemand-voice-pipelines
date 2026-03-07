from __future__ import annotations

import argparse
import os

import uvicorn

os.environ.setdefault("SUPABASE_URL", "https://local.demo.supabase.invalid")
os.environ.setdefault("SUPABASE_KEY", "local-demo-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram")
os.environ.setdefault("OPENAI_API_KEY", "test-openai")

from app.config import get_settings
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


def build_demo_service() -> PipelineService:
    return PipelineService(
        pipeline_repo=InMemoryPipelineRepository(),
        eval_dataset_repo=InMemoryEvalDatasetRepository(),
        eval_example_repo=InMemoryEvalExampleRepository(),
        pipeline_artifact_repo=InMemoryArtifactRepository(),
        pipeline_build_step_repo=InMemoryBuildStepRepository(),
        invocation_repo=InMemoryInvocationRepository(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the API with in-memory persistence for tmux/video recording."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    get_settings.cache_clear()
    service = build_demo_service()
    app.dependency_overrides[get_pipeline_service] = lambda: service
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

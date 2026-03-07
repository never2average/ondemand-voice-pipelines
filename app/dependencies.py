from functools import lru_cache

from app.config import get_settings
from app.db.client import get_supabase_client
from app.db.repositories.eval_dataset_repo import EvalDatasetRepository
from app.db.repositories.eval_repo import EvalExampleRepository
from app.db.repositories.in_memory import (
    InMemoryArtifactRepository,
    InMemoryBuildStepRepository,
    InMemoryEvalDatasetRepository,
    InMemoryEvalExampleRepository,
    InMemoryInvocationRepository,
    InMemoryPipelineRepository,
)
from app.db.repositories.invocation_repo import InvocationRepository
from app.db.repositories.pipeline_artifact_repo import PipelineArtifactRepository
from app.db.repositories.pipeline_build_step_repo import PipelineBuildStepRepository
from app.db.repositories.pipeline_repo import PipelineRepository
from app.services.pipeline_service import PipelineService


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    settings = get_settings()
    if settings.demo_mode:
        return PipelineService(
            pipeline_repo=InMemoryPipelineRepository(),
            eval_dataset_repo=InMemoryEvalDatasetRepository(),
            eval_example_repo=InMemoryEvalExampleRepository(),
            pipeline_artifact_repo=InMemoryArtifactRepository(),
            pipeline_build_step_repo=InMemoryBuildStepRepository(),
            invocation_repo=InMemoryInvocationRepository(),
        )

    client = get_supabase_client()
    return PipelineService(
        pipeline_repo=PipelineRepository(client),
        eval_dataset_repo=EvalDatasetRepository(client),
        eval_example_repo=EvalExampleRepository(client),
        pipeline_artifact_repo=PipelineArtifactRepository(client),
        pipeline_build_step_repo=PipelineBuildStepRepository(client),
        invocation_repo=InvocationRepository(client),
    )

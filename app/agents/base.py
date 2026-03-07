from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.schemas.pipeline import OptimizationObjective

ArtifactModel = TypeVar("ArtifactModel", bound=BaseModel)


def new_artifact_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class PipelineBuildContext:
    pipeline_id: str
    pipeline_name: str
    pipeline_description: str
    intent_prompt: str
    asr_provider: str
    optimization_objective: OptimizationObjective


@dataclass
class ArtifactGenerationResult(Generic[ArtifactModel]):
    artifact: ArtifactModel
    summary: str
    warnings: list[str] = field(default_factory=list)


class BaseArtifactAgent(ABC, Generic[ArtifactModel]):
    agent_name: str

    @abstractmethod
    async def run(self, *args, **kwargs) -> ArtifactGenerationResult[ArtifactModel]:
        ...

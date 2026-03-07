from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.schemas.artifacts import IntentClassifierComponentSpec, IntentSchemaArtifact

@dataclass
class IntentResult:
    detected_intent: str
    confidence: float
    candidates: list[dict[str, Any]] = field(default_factory=list)


class IntentExtractor(ABC):
    @abstractmethod
    async def extract(
        self,
        text: str,
        intent_schema_artifact: IntentSchemaArtifact,
        classifier_component: IntentClassifierComponentSpec,
    ) -> IntentResult:
        ...

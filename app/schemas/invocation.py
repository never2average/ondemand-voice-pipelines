from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.schemas.artifacts import ArtifactRef, ComponentTrace


class IntentCandidate(BaseModel):
    intent_name: str
    confidence: float


class InvokeRequest(BaseModel):
    input_text: str | None = None
    input_audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio bytes for audio invocation requests.",
    )
    input_type: str = Field(default="text", pattern="^(text|audio)$")

    @model_validator(mode="after")
    def validate_input(self) -> "InvokeRequest":
        if self.input_type == "text" and not self.input_text:
            raise ValueError("input_text is required when input_type is text")
        if self.input_type == "audio" and not self.input_audio_base64:
            raise ValueError("input_audio_base64 is required when input_type is audio")
        return self


class InvokeResponse(BaseModel):
    id: str
    pipeline_id: str
    input_type: str
    input_text: str
    normalized_text: str | None = None
    detected_intent: str
    confidence: float
    intent_candidates: list[IntentCandidate]
    latency_ms: int
    component_traces: list[ComponentTrace] = Field(default_factory=list)
    pipeline_graph_artifact: ArtifactRef
    pipeline_graph_version: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

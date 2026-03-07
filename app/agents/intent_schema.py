from __future__ import annotations

import re

from app.agents.base import ArtifactGenerationResult, BaseArtifactAgent, PipelineBuildContext, new_artifact_id
from app.schemas.artifacts import IntentDefinition, IntentSchemaArtifact


class IntentSchemaAgent(BaseArtifactAgent[IntentSchemaArtifact]):
    agent_name = "intent_schema_agent"

    async def run(self, build_context: PipelineBuildContext) -> ArtifactGenerationResult[IntentSchemaArtifact]:
        intent_lines = self._extract_intent_lines(build_context.intent_prompt)
        intent_definitions = [self._build_intent_definition(line) for line in intent_lines]

        if not intent_definitions:
            intent_definitions = [
                IntentDefinition(
                    intent_name="general_request",
                    description=build_context.intent_prompt.strip(),
                    positive_examples=[build_context.intent_prompt.strip()],
                    negative_examples=[],
                    disambiguation_rules=["Use this intent when no narrower intent matches."],
                    out_of_scope_examples=[],
                )
            ]

        artifact = IntentSchemaArtifact(
            artifact_id=new_artifact_id(),
            pipeline_id=build_context.pipeline_id,
            source_prompt=build_context.intent_prompt,
            fallback_intent="unknown",
            intents=intent_definitions,
        )
        summary = f"Derived {len(intent_definitions)} intents from the source prompt."
        return ArtifactGenerationResult(artifact=artifact, summary=summary)

    def _extract_intent_lines(self, intent_prompt: str) -> list[str]:
        raw_lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in intent_prompt.splitlines()
            if line.strip()
        ]
        if len(raw_lines) == 1:
            comma_segments = [segment.strip() for segment in re.split(r"[;,]", raw_lines[0]) if segment.strip()]
            if len(comma_segments) > 1:
                return comma_segments
        return raw_lines

    def _build_intent_definition(self, line: str) -> IntentDefinition:
        if ":" in line:
            candidate_name, description = [part.strip() for part in line.split(":", 1)]
        else:
            candidate_name, description = line.strip(), f"Handle requests about {line.strip()}."

        intent_name = self._normalize_name(candidate_name)
        display_phrase = candidate_name.replace("_", " ").strip()

        return IntentDefinition(
            intent_name=intent_name,
            description=description,
            positive_examples=[
                display_phrase,
                f"I need help with {display_phrase}",
                f"Can you handle {display_phrase}?",
            ],
            negative_examples=[],
            disambiguation_rules=[
                f"Choose {intent_name} only when the request is primarily about {display_phrase}.",
                f"If the request is unrelated to {display_phrase}, prefer another intent or unknown.",
            ],
            out_of_scope_examples=[
                f"This is not about {display_phrase}.",
            ],
        )

    def _normalize_name(self, value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
        return normalized or "general_request"

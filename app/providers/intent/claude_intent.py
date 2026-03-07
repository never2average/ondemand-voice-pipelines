from __future__ import annotations

import json
import re

import anthropic

from app.config import get_settings
from app.core.exceptions import IntentExtractionError
from app.providers.intent.base import IntentExtractor, IntentResult
from app.schemas.artifacts import IntentClassifierComponentSpec, IntentSchemaArtifact


class ClaudeIntentExtractor(IntentExtractor):
    def __init__(self):
        api_key = get_settings().anthropic_api_key
        self._client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else None
        self._api_key = api_key

    async def extract(
        self,
        text: str,
        intent_schema_artifact: IntentSchemaArtifact,
        classifier_component: IntentClassifierComponentSpec,
    ) -> IntentResult:
        if not self._should_use_remote_model():
            return self._fallback_extract(text, intent_schema_artifact, classifier_component)

        try:
            response = await self._client.messages.create(
                model=classifier_component.model,
                max_tokens=1024,
                system=classifier_component.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Classify this utterance using the provided intent schema.\n\n"
                            f"Utterance: {text}\n\n"
                            f"Intent schema: {intent_schema_artifact.model_dump_json()}"
                        ),
                    }
                ],
            )

            raw = response.content[0].text
            parsed = json.loads(raw)

            candidates = [
                {
                    "intent_name": candidate.get("intent_name", candidate.get("intent", "unknown")),
                    "confidence": float(candidate.get("confidence", 0.0)),
                }
                for candidate in parsed.get("candidates", [])
            ]
            detected = parsed.get("intent", "unknown")
            confidence = parsed.get("confidence", 0.0)
            if not candidates:
                candidates = [{"intent_name": detected, "confidence": confidence}]

            return IntentResult(
                detected_intent=detected,
                confidence=confidence,
                candidates=candidates,
            )
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise IntentExtractionError(f"Failed to parse intent response: {e}") from e
        except anthropic.APIError as e:
            if self._client is None:
                raise IntentExtractionError(f"Claude API error: {e}") from e
            return self._fallback_extract(text, intent_schema_artifact, classifier_component)

    def _should_use_remote_model(self) -> bool:
        if self._client is None:
            return False
        lowered_key = self._api_key.lower()
        return "test" not in lowered_key and "dummy" not in lowered_key

    def _fallback_extract(
        self,
        text: str,
        intent_schema_artifact: IntentSchemaArtifact,
        classifier_component: IntentClassifierComponentSpec,
    ) -> IntentResult:
        text_tokens = self._tokenize(text)
        scored_candidates: list[dict[str, float | str]] = []

        for intent_definition in intent_schema_artifact.intents:
            intent_tokens = self._tokenize(
                " ".join(
                    [
                        intent_definition.intent_name.replace("_", " "),
                        intent_definition.description,
                        " ".join(intent_definition.positive_examples),
                    ]
                )
            )
            if not intent_tokens:
                continue
            overlap = len(text_tokens & intent_tokens)
            score = overlap / max(len(text_tokens), 1)
            if overlap:
                score += min(0.25, overlap / max(len(intent_tokens), 1))
            scored_candidates.append(
                {
                    "intent_name": intent_definition.intent_name,
                    "confidence": min(0.99, round(score, 4)),
                }
            )

        scored_candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
        top_candidates = scored_candidates[: classifier_component.candidate_count]

        if not top_candidates or top_candidates[0]["confidence"] < 0.2:
            return IntentResult(
                detected_intent=intent_schema_artifact.fallback_intent,
                confidence=0.35,
                candidates=[
                    {
                        "intent_name": intent_schema_artifact.fallback_intent,
                        "confidence": 0.35,
                    }
                ],
            )

        top_candidate = top_candidates[0]
        normalized_candidates = [
            {
                "intent_name": str(candidate["intent_name"]),
                "confidence": float(candidate["confidence"]),
            }
            for candidate in top_candidates
        ]
        return IntentResult(
            detected_intent=str(top_candidate["intent_name"]),
            confidence=float(top_candidate["confidence"]),
            candidates=normalized_candidates,
        )

    def _tokenize(self, value: str) -> set[str]:
        tokens = {token for token in re.split(r"[^a-zA-Z0-9]+", value.lower()) if token}
        return {token for token in tokens if token not in {"the", "a", "an", "to", "for", "with", "help"}}

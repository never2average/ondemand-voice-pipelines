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
        quoted_candidates = self._extract_quoted_candidates(intent_prompt)
        if quoted_candidates:
            return quoted_candidates

        raw_lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in intent_prompt.splitlines()
            if line.strip()
        ]

        extracted_candidates: list[str] = []
        collecting_list_items = False

        for line in raw_lines:
            if self._is_fallback_instruction(line):
                collecting_list_items = False
                continue

            if self._is_section_header(line):
                collecting_list_items = True
                _, _, remainder = line.partition(":")
                extracted_candidates.extend(self._split_candidate_segments(remainder))
                continue

            if collecting_list_items and self._looks_like_list_item(line):
                extracted_candidates.extend(self._split_candidate_segments(line))
                continue

            collecting_list_items = False

            if self._looks_like_explicit_intent_definition(line):
                extracted_candidates.append(line)
                continue

            if self._looks_like_short_candidate(line):
                extracted_candidates.extend(self._split_candidate_segments(line))

        if extracted_candidates:
            return self._dedupe_preserve_order(extracted_candidates)

        inline_candidates = self._extract_inline_candidates(intent_prompt)
        if inline_candidates:
            return inline_candidates

        if len(raw_lines) == 1:
            comma_segments = self._split_candidate_segments(raw_lines[0])
            if len(comma_segments) > 1:
                return comma_segments

        return raw_lines

    def _extract_quoted_candidates(self, intent_prompt: str) -> list[str]:
        quoted_segments = re.findall(r'["“](.+?)["”]', intent_prompt)
        cleaned_segments = [
            self._clean_candidate(segment)
            for segment in quoted_segments
        ]
        return self._dedupe_preserve_order(
            [segment for segment in cleaned_segments if segment]
        )

    def _extract_inline_candidates(self, intent_prompt: str) -> list[str]:
        lowered_prompt = re.sub(r"\s+", " ", intent_prompt.strip())
        markers = [
            "customers usually say things like:",
            "people usually call about:",
            "the requests i need routed are:",
            "common requests are:",
            "common calls are:",
            "common intents are:",
        ]
        for marker in markers:
            marker_index = lowered_prompt.lower().find(marker)
            if marker_index == -1:
                continue
            tail = lowered_prompt[marker_index + len(marker):]
            tail = re.split(r"\.\s+|!\s+|\?\s+", tail, maxsplit=1)[0]
            candidates = self._split_candidate_segments(tail)
            if candidates:
                return candidates
        return []

    def _split_candidate_segments(self, value: str) -> list[str]:
        if not value.strip():
            return []
        segments = [
            self._clean_candidate(segment)
            for segment in re.split(r"[;,]", value)
        ]
        return self._dedupe_preserve_order([segment for segment in segments if segment])

    def _clean_candidate(self, value: str) -> str:
        cleaned = value.strip().strip('"').strip("'").strip()
        cleaned = re.sub(r"^(?:and|or)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:customers|callers|people)\s+(?:who\s+)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^(?:i|we)\s+(?:want|need|would like)(?:\s+to)?\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:i'm|i am)\s+calling\s+(?:about|to)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:please\s+)?(?:help\s+me\s+)?(?:with\s+)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,:;!-")
        return cleaned

    def _is_section_header(self, value: str) -> bool:
        if ":" not in value:
            return False
        header, _, _ = value.partition(":")
        lowered_header = header.strip().lower()
        return any(
            phrase in lowered_header
            for phrase in (
                "requests",
                "common calls",
                "common requests",
                "common intents",
                "customers usually say",
                "people usually call about",
                "need routed",
                "route",
            )
        )

    def _looks_like_explicit_intent_definition(self, value: str) -> bool:
        if ":" not in value or self._is_section_header(value):
            return False
        header, _, _ = value.partition(":")
        return len(header.split()) <= 6 and "." not in header

    def _looks_like_list_item(self, value: str) -> bool:
        cleaned = self._clean_candidate(value)
        return bool(cleaned) and len(cleaned.split()) <= 10

    def _looks_like_short_candidate(self, value: str) -> bool:
        cleaned = self._clean_candidate(value)
        if not cleaned:
            return False
        if self._is_context_line(cleaned) or self._is_fallback_instruction(cleaned):
            return False
        return len(cleaned.split()) <= 6

    def _is_context_line(self, value: str) -> bool:
        lowered_value = value.lower()
        return any(
            phrase in lowered_value
            for phrase in (
                "build a voice router",
                "building a voice router",
                "phone support line",
                "customer support",
                "retail bank",
                "support line",
            )
        )

    def _is_fallback_instruction(self, value: str) -> bool:
        lowered_value = value.lower()
        return "unknown" in lowered_value and any(
            phrase in lowered_value
            for phrase in (
                "anything else",
                "something else",
                "doesn't fit",
                "does not fit",
                "route it to",
                "send it to",
                "fallback",
            )
        )

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        deduped_values: list[str] = []
        seen_values: set[str] = set()
        for value in values:
            normalized_value = value.lower()
            if normalized_value in seen_values:
                continue
            seen_values.add(normalized_value)
            deduped_values.append(value)
        return deduped_values

    def _build_intent_definition(self, line: str) -> IntentDefinition:
        if ":" in line:
            candidate_name, description = [part.strip() for part in line.split(":", 1)]
        else:
            candidate_name = self._clean_candidate(line)
            description = f"Handle requests about {candidate_name}."

        intent_name = self._normalize_name(candidate_name)
        display_phrase = self._clean_candidate(candidate_name).replace("_", " ").strip()

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
        normalized_value = self._clean_candidate(value).lower()
        normalized_value = re.sub(
            r"\b(?:i|we|my|mine|their|the|a|an|please|help|me|with|about|for|to|is|are|on|because)\b",
            " ",
            normalized_value,
        )
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized_value).strip("_")
        return normalized or "general_request"

from __future__ import annotations

import asyncio
from typing import Any

from supabase import Client


class PipelineArtifactRepository:
    def __init__(self, client: Client):
        self._client = client
        self._table = "pipeline_artifacts"

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table).insert(data).execute()
        )
        return result.data[0]

    async def get_by_id(self, artifact_id: str) -> dict[str, Any] | None:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("id", artifact_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def list_by_pipeline(self, pipeline_id: str) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("pipeline_id", pipeline_id)
            .order("created_at")
            .execute()
        )
        return result.data

    async def get_latest_by_type(self, pipeline_id: str, artifact_type: str) -> dict[str, Any] | None:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("pipeline_id", pipeline_id)
            .eq("artifact_type", artifact_type)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

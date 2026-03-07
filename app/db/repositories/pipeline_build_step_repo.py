from __future__ import annotations

import asyncio
from typing import Any

from supabase import Client


class PipelineBuildStepRepository:
    def __init__(self, client: Client):
        self._client = client
        self._table = "pipeline_build_steps"

    async def save(self, data: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .upsert(data, on_conflict="pipeline_id,step_name")
            .execute()
        )
        return result.data[0]

    async def list_by_pipeline(self, pipeline_id: str) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("pipeline_id", pipeline_id)
            .order("started_at")
            .execute()
        )
        return result.data

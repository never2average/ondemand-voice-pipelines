from __future__ import annotations

import asyncio
from typing import Any

from supabase import Client


class PipelineRepository:
    def __init__(self, client: Client):
        self._client = client
        self._table = "pipelines"

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table).insert(data).execute()
        )
        return result.data[0]

    async def get_by_id(self, pipeline_id: str) -> dict[str, Any] | None:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("id", pipeline_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def list_all(self) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    async def update(self, pipeline_id: str, data: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .update(data)
            .eq("id", pipeline_id)
            .execute()
        )
        return result.data[0]

from __future__ import annotations

import asyncio
from typing import Any

from supabase import Client


class InvocationRepository:
    def __init__(self, client: Client):
        self._client = client
        self._table = "invocations"

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table).insert(data).execute()
        )
        return result.data[0]

    async def list_by_pipeline(self, pipeline_id: str) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("pipeline_id", pipeline_id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

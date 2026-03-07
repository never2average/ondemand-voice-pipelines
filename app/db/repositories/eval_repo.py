from __future__ import annotations

import asyncio
from typing import Any

from supabase import Client


class EvalExampleRepository:
    def __init__(self, client: Client):
        self._client = client
        self._table = "eval_examples"

    async def create_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table).insert(records).execute()
        )
        return result.data

    async def list_by_dataset_id(self, dataset_id: str) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("dataset_id", dataset_id)
            .order("created_at")
            .execute()
        )
        return result.data

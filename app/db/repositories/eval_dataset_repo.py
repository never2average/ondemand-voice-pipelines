from __future__ import annotations

import asyncio
from typing import Any

from supabase import Client


class EvalDatasetRepository:
    def __init__(self, client: Client):
        self._client = client
        self._table = "eval_datasets"

    async def create(self, data: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table).insert(data).execute()
        )
        return result.data[0]

    async def get_by_id(self, dataset_id: str) -> dict[str, Any] | None:
        result = await asyncio.to_thread(
            lambda: self._client.table(self._table)
            .select("*")
            .eq("id", dataset_id)
            .execute()
        )
        return result.data[0] if result.data else None

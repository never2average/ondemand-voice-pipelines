from __future__ import annotations

from collections.abc import Iterable


class InMemoryPipelineRepository:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def create(self, data: dict) -> dict:
        row = dict(data)
        self.rows[row["id"]] = row
        return dict(row)

    async def get_by_id(self, pipeline_id: str) -> dict | None:
        row = self.rows.get(pipeline_id)
        return dict(row) if row else None

    async def list_all(self) -> list[dict]:
        return sorted(
            (dict(row) for row in self.rows.values()),
            key=lambda row: row["created_at"],
            reverse=True,
        )

    async def update(self, pipeline_id: str, data: dict) -> dict:
        if pipeline_id not in self.rows:
            raise KeyError(pipeline_id)
        self.rows[pipeline_id].update(data)
        return dict(self.rows[pipeline_id])


class InMemoryArtifactRepository:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def create(self, data: dict) -> dict:
        row = dict(data)
        self.rows[row["id"]] = row
        return dict(row)

    async def get_latest_by_type(self, pipeline_id: str, artifact_type: str) -> dict | None:
        matching = [
            row
            for row in self.rows.values()
            if row["pipeline_id"] == pipeline_id and row["artifact_type"] == artifact_type
        ]
        if not matching:
            return None
        latest = max(matching, key=lambda row: int(row["version"]))
        return dict(latest)

    async def get_by_id(self, artifact_id: str) -> dict | None:
        row = self.rows.get(artifact_id)
        return dict(row) if row else None

    async def list_by_pipeline(self, pipeline_id: str) -> list[dict]:
        matching = [
            dict(row)
            for row in self.rows.values()
            if row["pipeline_id"] == pipeline_id
        ]
        return sorted(matching, key=lambda row: row["created_at"])


class InMemoryBuildStepRepository:
    def __init__(self):
        self.rows: dict[tuple[str, str], dict] = {}
        self._order: list[tuple[str, str]] = []

    async def save(self, data: dict) -> dict:
        key = (data["pipeline_id"], data["step_name"])
        if key not in self.rows:
            self._order.append(key)
        self.rows[key] = dict(data)
        return dict(self.rows[key])

    async def list_by_pipeline(self, pipeline_id: str) -> list[dict]:
        return [dict(self.rows[key]) for key in self._order if key[0] == pipeline_id]


class InMemoryEvalDatasetRepository:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def create(self, data: dict) -> dict:
        row = dict(data)
        self.rows[row["id"]] = row
        return dict(row)


class InMemoryEvalExampleRepository:
    def __init__(self):
        self.rows: list[dict] = []

    async def create_many(self, records: Iterable[dict]) -> list[dict]:
        created = [dict(record) for record in records]
        self.rows.extend(created)
        return created


class InMemoryInvocationRepository:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    async def create(self, data: dict) -> dict:
        row = dict(data)
        self.rows[row["id"]] = row
        return dict(row)

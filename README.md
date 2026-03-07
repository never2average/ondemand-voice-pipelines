# On-Demand Voice Pipelines

FastAPI service for building and invoking voice intent pipelines that optimize intent error rate instead of word error rate.

## What it does

- Creates pipelines from an intent prompt.
- Builds pipelines asynchronously through typed subagents.
- Persists build artifacts, build steps, evaluation reports, and invocation traces.
- Invokes the published pipeline graph over text or base64-encoded audio.

## Architecture

- `IntentSchemaAgent`: turns the prompt into a typed intent schema.
- `EvalDatasetCuratorAgent`: creates train, dev, and immutable holdout eval sets.
- `BaselineGraphPlannerAgent`: produces the first typed pipeline graph.
- `PipelineEvaluatorAgent`: runs the graph and computes intent-error metrics.
- `AdversarialDatasetAgent`: clusters failures and proposes adversarial examples.
- `GraphRevisionAgent`: revises graph components and thresholds.
- `PublishingAgent`: publishes the best graph only if holdout intent error rate meets the target.

## Core types

- `IntentSchemaArtifact`
- `EvalDatasetArtifact`
- `PipelineGraphArtifact`
- `EvaluationReportArtifact`
- `AdversarialFindingsArtifact`
- `PipelineSpec`
- `PipelineBuildStep`

## API

- `GET /health`
- `GET /api/v1/pipelines`
- `POST /api/v1/pipelines`
- `GET /api/v1/pipelines/{pipeline_id}`
- `POST /api/v1/pipelines/{pipeline_id}/invoke`

The generated OpenAPI document is stored in `openapi.json`.

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

## Tests

```bash
.venv/bin/pytest
```

## Supabase

Apply `supabase_migration.sql` in the Supabase SQL editor before running against a real database.

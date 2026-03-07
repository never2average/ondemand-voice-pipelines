# On-Demand Voice Pipelines

Voice intent pipelines that optimize for intent error rate (IER) instead of word error rate (WER). You give the API a natural-language description of your use case, the system builds a pipeline around that intent space, evaluates it, and publishes a graph you can invoke with text or audio.

![Four-pane terminal demo](docs/demo/four-pane-demo.gif)

## What the demo shows

The demo is built around a realistic retail-banking support scenario:

- create a new pipeline from a normal human prompt
- monitor the pipeline build through the existing detail API
- watch the pipeline appear as `ready` in the list API
- send an audio sample to the published pipeline and inspect the output trace

The prompt used in the demo is intentionally non-technical:

```text
I'm setting up a phone support line for a retail bank.
Customers usually say things like:
- "I want to check my balance"
- "I need to transfer money between accounts"
- "I need to dispute a charge on my card"
- "I need a replacement card because mine is lost"
If it's something else, send it to unknown.
```

## Demo paths

### 1. Four-pane tmux recording

Use this when you want the same flow shown in the GIF or want to regenerate the asset.

```bash
python3 scripts/run_tmux_demo.py
```

What it does:

- starts a local API server through [run_local_demo_server.py](/Users/priyeshsrivastava/ondemand-voice-pipelines/scripts/run_local_demo_server.py)
- keeps the public HTTP contract the same
- uses in-memory repositories only for recording repeatability
- opens four tmux panes: server logs, create/invoke control, detail watcher, list watcher
- drives the create -> monitor -> list -> invoke flow
- renders the pane snapshots into [four-pane-demo.gif](/Users/priyeshsrivastava/ondemand-voice-pipelines/docs/demo/four-pane-demo.gif)

The local recording harness is separate from the product runtime path. It does not reintroduce a runtime demo mode.

### 2. Single-pane live walkthrough against the API

Use this when you already have the API running and want one straightforward end-to-end terminal run.

```bash
.venv/bin/python scripts/run_video_demo.py
```

This prints:

- the `POST /api/v1/pipelines` request
- build progress from `GET /api/v1/pipelines/{pipeline_id}`
- the matching row from `GET /api/v1/pipelines`
- the final `POST /api/v1/pipelines/{pipeline_id}/invoke` response

## Public API

- `GET /health`
- `GET /api/v1/pipelines`
- `POST /api/v1/pipelines`
- `GET /api/v1/pipelines/{pipeline_id}`
- `POST /api/v1/pipelines/{pipeline_id}/invoke`

The generated OpenAPI document is stored in [openapi.json](/Users/priyeshsrivastava/ondemand-voice-pipelines/openapi.json).

## What to inspect during generation

Use `GET /api/v1/pipelines/{pipeline_id}` while the build is running.

The most useful fields for narration are:

- `build_steps[]`: step-by-step progress through generation and publishing
- `artifact_history[]`: every persisted artifact with `build_phase`, `artifact_type`, `version`, and summary
- `intent_schema_artifact`: the intent contract derived from the user prompt
- `eval_dataset_artifact`: the grounded train/dev/holdout dataset
- `published_graph_artifact`: the assembled ASR, normalization, classification, and decision-policy components
- `latest_evaluation_report_artifact`: the measured IER and confusion data

Use `GET /api/v1/pipelines` after publication to show the pipeline in the active list with summary status and graph version.

Use `POST /api/v1/pipelines/{pipeline_id}/invoke` to show:

- transcript text
- normalized text
- detected intent
- confidence and intent candidates
- per-component traces

## Architecture

- `IntentSchemaAgent`: turns the prompt into a typed `IntentSchemaArtifact`
- `EvalDatasetCuratorAgent`: generates train/dev/holdout examples with phenomenon tags
- `BaselineGraphPlannerAgent`: creates the first `PipelineGraphArtifact`
- `PipelineEvaluatorAgent`: measures IER through the executable graph runner
- `AdversarialDatasetAgent`: groups failures and proposes new hard examples
- `GraphRevisionAgent`: revises ASR hints, normalization rules, classifier guidance, and thresholds
- `PublishingAgent`: publishes the best graph only if holdout IER meets the threshold

Typed artifacts are defined in [artifacts.py](/Users/priyeshsrivastava/ondemand-voice-pipelines/app/schemas/artifacts.py):

- `IntentSchemaArtifact`
- `EvalDatasetArtifact`
- `PipelineGraphArtifact`
- `EvaluationReportArtifact`
- `AdversarialFindingsArtifact`
- `PipelineSpec`
- `PipelineBuildStep`

## Local voice sample

The committed audio fixture is [check-my-balance.wav](/Users/priyeshsrivastava/ondemand-voice-pipelines/examples/voice_samples/check-my-balance.wav). The repo also includes a local `sample` ASR provider for offline audio invocation in tests and recording workflows.

The end-to-end sample path is covered by [test_voice_sample_e2e.py](/Users/priyeshsrivastava/ondemand-voice-pipelines/tests/test_endpoints/test_voice_sample_e2e.py).

## Local setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

For the real runtime path, apply [supabase_migration.sql](/Users/priyeshsrivastava/ondemand-voice-pipelines/supabase_migration.sql) in Supabase and configure `.env`.

## Tests

```bash
.venv/bin/pytest
```

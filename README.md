# On-Demand Voice Pipelines

Self-improving voice intent pipelines that optimize for intent error rate (IER) instead of word error rate (WER). Create a pipeline from an intent prompt — the system builds, evaluates, and hardens it automatically.

## What it does

- Creates voice intent pipelines from a natural language intent prompt.
- Builds pipelines asynchronously through a multi-agent improvement loop (Eval Curator, ETL Pipeline, Adversarial).
- Produces typed, versioned artifacts at each build step — intent schemas, eval datasets, pipeline graphs, evaluation reports, adversarial findings.
- Iterates until intent error rate meets the target or the iteration budget is exhausted.
- Invokes the published pipeline graph over text or base64-encoded audio, returning intent, confidence, candidates, and full component traces.
- Includes a committed sample WAV plus a local `sample` ASR provider so the entire API flow can run offline.

## Architecture

- `IntentSchemaAgent`: turns the intent prompt into a typed `IntentSchemaArtifact` with definitions, disambiguation rules, and examples.
- `EvalDatasetCuratorAgent`: generates diverse test utterances with phenomenon tags (filler words, polite forms, out-of-domain, etc.) split into train/dev/immutable holdout.
- `BaselineGraphPlannerAgent`: produces the first `PipelineGraphArtifact` — ASR, Normalizer, Classifier, Reranker, Decision Policy.
- `PipelineEvaluatorAgent`: runs the eval suite against the graph, produces `EvaluationReportArtifact` with IER, confusion matrix, and component traces.
- `AdversarialDatasetAgent`: clusters failures by confused intent pairs, generates targeted edge cases, proposes component-level fixes.
- `GraphRevisionAgent`: rewrites classifier prompts, thresholds, and ASR hints based on failure data.
- `PublishingAgent`: publishes the best graph only if holdout IER meets the target.

## Core types

Defined in `app/schemas/artifacts.py`:

- `IntentSchemaArtifact` — parsed intents with positive/negative examples and disambiguation rules
- `EvalDatasetArtifact` — test utterances with splits, phenomenon tags, confusable-with annotations
- `PipelineGraphArtifact` — component specs (ASR, Normalizer, Classifier, Reranker, Decision Policy) and edges
- `EvaluationReportArtifact` — IER, confusion matrix, per-intent accuracy, hard cases with component traces
- `AdversarialFindingsArtifact` — failure clusters, proposed adversarial examples, recommended component changes
- `PipelineSpec` / `PipelineBuildStep` — pipeline metadata and build progress

## Developer workflow

### 1. Create a pipeline

```bash
curl -X POST /api/v1/pipelines \
  -H "Content-Type: application/json" \
  -d '{
    "name": "support-router",
    "intent_prompt": "billing inquiry, cancellation, account status, technical support",
    "asr_provider": "whisper",
    "optimization_objective": {
      "target_intent_error_rate": 0.08,
      "max_optimization_rounds": 3
    }
  }'
# Returns 202 Accepted with pipeline_id, status: "building"
```

### 2. Poll for build progress

```bash
curl /api/v1/pipelines/{pipeline_id}
# Response includes:
#   status: "building" | "ready" | "failed"
#   build_steps[]: per-step status (eval_curation, etl_optimization, adversarial, ...)
#   active_build_step: current step name
#   current_intent_error_rate: IER at latest eval
```

### 3. Inspect the built pipeline (when ready)

```bash
curl /api/v1/pipelines/{pipeline_id}
# Full detail: intent_schema_artifact, published_graph_artifact,
# latest_evaluation_report (IER, confusion matrix, per-intent accuracy),
# latest_adversarial_findings (failure clusters, component fix recommendations),
# artifact_history[] (every persisted build artifact with build_phase, version, summary, and typed payload)
```

### 4. Invoke

```bash
curl -X POST /api/v1/pipelines/{pipeline_id}/invoke \
  -H "Content-Type: application/json" \
  -d '{"input_text": "I want to cancel my subscription", "input_type": "text"}'
# Returns:
#   detected_intent: "cancellation"
#   confidence: 0.94
#   intent_candidates: [{"intent": "cancellation", "confidence": 0.94}, ...]
#   component_traces: [{component_id, input_snapshot, output_snapshot, latency_ms}, ...]
#   latency_ms: 320
```

## API

- `GET /health`
- `GET /api/v1/pipelines` — list all pipelines with summary status
- `POST /api/v1/pipelines` — create pipeline, starts async build (202)
- `GET /api/v1/pipelines/{pipeline_id}` — full detail with artifacts and build steps
- `POST /api/v1/pipelines/{pipeline_id}/invoke` — classify an utterance using the optimized pipeline

The generated OpenAPI document is stored in `openapi.json`.

## Local voice sample

The repository includes [examples/voice_samples/check-my-balance.wav](/Users/priyeshsrivastava/ondemand-voice-pipelines/examples/voice_samples/check-my-balance.wav) and a `sample` ASR provider used by tests and local development.

After starting a normally configured API instance, you can execute:

```bash
.venv/bin/python scripts/run_voice_sample_demo.py
```

The script exercises all four pipeline APIs in order:

1. `POST /api/v1/pipelines` with `asr_provider="sample"`
2. `GET /api/v1/pipelines`
3. `GET /api/v1/pipelines/{pipeline_id}`
4. `POST /api/v1/pipelines/{pipeline_id}/invoke` with the base64-encoded WAV sample

This path is covered by `tests/test_endpoints/test_voice_sample_e2e.py`.

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

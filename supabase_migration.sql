-- Run this in the Supabase SQL Editor (https://supabase.com/dashboard -> SQL Editor)

create extension if not exists "uuid-ossp";

create table pipelines (
    id uuid primary key default uuid_generate_v4(),
    name text not null,
    description text not null default '',
    intent_prompt text not null,
    status text not null default 'pending' check (status in ('pending', 'building', 'ready', 'failed')),
    config jsonb not null default '{}',
    metrics jsonb not null default '{}',
    asr_provider text not null default 'whisper',
    optimization_objective jsonb not null default '{}',
    intent_schema_artifact_id uuid,
    intent_schema_artifact_version integer,
    eval_dataset_artifact_id uuid,
    eval_dataset_artifact_version integer,
    published_graph_artifact_id uuid,
    published_graph_version integer,
    latest_evaluation_report_artifact_id uuid,
    latest_evaluation_report_version integer,
    latest_adversarial_findings_artifact_id uuid,
    latest_adversarial_findings_version integer,
    current_intent_error_rate double precision,
    holdout_intent_error_rate double precision,
    current_build_step text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table pipeline_artifacts (
    id uuid primary key,
    pipeline_id uuid not null references pipelines(id) on delete cascade,
    artifact_type text not null check (
        artifact_type in (
            'intent_schema',
            'eval_dataset',
            'pipeline_graph',
            'evaluation_report',
            'adversarial_findings'
        )
    ),
    version integer not null,
    payload jsonb not null,
    producer_agent text not null,
    summary text not null default '',
    created_at timestamptz not null default now()
);

create unique index idx_pipeline_artifacts_pipeline_type_version
    on pipeline_artifacts(pipeline_id, artifact_type, version);

create table pipeline_build_steps (
    id uuid primary key default uuid_generate_v4(),
    pipeline_id uuid not null references pipelines(id) on delete cascade,
    step_name text not null,
    status text not null check (status in ('pending', 'running', 'completed', 'failed')),
    started_at timestamptz,
    completed_at timestamptz,
    input_artifacts jsonb not null default '[]',
    output_artifacts jsonb not null default '[]',
    summary text not null default '',
    error text,
    updated_at timestamptz not null default now()
);

create unique index idx_pipeline_build_steps_pipeline_step
    on pipeline_build_steps(pipeline_id, step_name);

create table eval_datasets (
    id uuid primary key,
    pipeline_id uuid not null references pipelines(id) on delete cascade,
    intent_schema_artifact_id uuid not null,
    artifact_version integer not null,
    coverage_summary jsonb not null default '{}',
    created_at timestamptz not null default now()
);

create table eval_examples (
    id uuid primary key,
    dataset_id uuid not null references eval_datasets(id) on delete cascade,
    pipeline_id uuid not null references pipelines(id) on delete cascade,
    split text not null check (split in ('train', 'dev', 'holdout')),
    source text not null check (source in ('seed', 'curated', 'adversarial', 'production')),
    modality text not null check (modality in ('text', 'audio')),
    utterance_text text,
    audio_uri text,
    expected_intent text not null,
    phenomenon_tags jsonb not null default '[]',
    confusable_with jsonb not null default '[]',
    metadata jsonb not null default '{}',
    created_at timestamptz not null default now()
);

create index idx_eval_examples_dataset_id on eval_examples(dataset_id);
create index idx_eval_examples_pipeline_id on eval_examples(pipeline_id);

create table invocations (
    id uuid primary key default uuid_generate_v4(),
    pipeline_id uuid not null references pipelines(id) on delete cascade,
    input_type text not null default 'text',
    input_text text not null,
    normalized_text text,
    detected_intent text not null,
    confidence double precision not null,
    intent_candidates jsonb not null default '[]',
    latency_ms integer not null,
    component_traces jsonb not null default '[]',
    pipeline_graph_artifact_id uuid,
    pipeline_graph_version integer,
    metadata jsonb not null default '{}',
    created_at timestamptz not null default now()
);

create index idx_invocations_pipeline_id on invocations(pipeline_id);

# Three Agents, One Improvement Loop

The improvement loop is the pipeline. Everything else is configuration.

That statement sounds reductive until you see what "configuration" means in practice. In a traditional voice pipeline, configuration means a YAML file someone wrote six months ago. In this system, configuration means the output of three specialized agents — each with tool access, each producing typed artifacts, each operating on the failures of the previous iteration.

The agents are not autonomous in the science-fiction sense. They are autonomous in the engineering sense: given a well-defined objective and a set of tools, they execute a multi-step workflow without human intervention. The human defines the intents. The agents build, test, and harden the pipeline.

## Agent 1: The Eval Curator

**Role:** Generate a diverse, structured evaluation dataset from an intent schema.

The Eval Curator takes the parsed `IntentSchemaArtifact` — a list of intents with definitions, examples, and disambiguation rules — and produces an `EvalDatasetArtifact` containing test utterances across three splits.

**Why three splits matter:**

| Split | Purpose | Mutation Policy |
|-------|---------|-----------------|
| `train` | Visible to optimization loop. The ETL agent sees these results. | Can be expanded with adversarial examples |
| `dev` | Validation set. Used to detect overfitting to train. | Can be expanded, never moved to train |
| `holdout` | Immutable. Never seen during optimization. Final integrity check. | Never modified, never expanded |

The holdout split is sacred. If the system achieves 5% IER on train and dev but 20% IER on holdout, the optimization loop overfit. The holdout catches this.

**What the curator generates:**

For each intent, the curator produces utterances with explicit phenomenon tags:

```python
EvalExample(
    example_id=new_artifact_id(),
    split="dev",
    source="curated",
    modality="text",
    utterance_text="uh can you check my balance",
    expected_intent="check_balance",
    phenomenon_tags=["filler_words"],
)
```

Phenomenon tags mark *what makes this example hard*:

- `direct` — Bare intent name as utterance ("check balance")
- `natural_language` — Conversational framing ("I need help with checking my balance")
- `question` — Question form ("Can you handle check balance for me?")
- `polite` — Polite prefix ("please check my balance")
- `filler_words` — Disfluency markers ("uh can you check my balance")
- `out_of_domain` — Utterances that should map to fallback ("Tell me a joke")
- `description_based` — Uses the intent description, not the name ("I'm calling about viewing my current account balance")

The tags serve two purposes. First, they enable *phenomenon-level analysis* — if 80% of filler-word examples fail, the normalizer's filler-stripping rules need work. Second, they guide the adversarial agent's example generation — it generates more examples in the phenomenon categories that show the highest failure rates.

**Coverage tracking:**

```python
coverage_summary = {
    split_name: sum(1 for example in examples if example.split == split_name)
    for split_name in ("train", "dev", "holdout")
}
```

The coverage summary is not a vanity metric. It's a constraint checker. If holdout has zero examples for an intent, the system cannot validate that intent's accuracy on unseen data. The summary surfaces gaps before the eval loop runs.

## Agent 2: The ETL Pipeline Agent

**Role:** Run evaluations, analyze failures, rewrite pipeline configuration, and re-evaluate — in a tool-use loop.

The ETL Pipeline Agent is the workhorse. It has access to five tools, each defined as a JSON schema and executed against shared agent context:

```
store_evals       → Persist test cases to the evaluation set
run_eval_suite    → Execute all evals against current pipeline config, return IER
analyze_failures  → Return failed cases with expected vs. actual intent
update_prompt_config → Rewrite the classifier system prompt, threshold, and ASR hints
classify_utterance   → Spot-check a single utterance against current config
```

The agent operates in a loop. Here's the actual flow:

1. **Store evals** — The agent receives the curated eval set and stores it in context
2. **Run eval suite** — Every example is classified against the current pipeline config. The tool returns IER, correct/total counts, and up to 10 failure details
3. **Analyze failures** — The agent inspects which utterances were misclassified and why
4. **Update prompt config** — Based on failure analysis, the agent rewrites the system prompt. It may add disambiguation instructions, adjust the confidence threshold, or update ASR keyword hints
5. **Run eval suite again** — Measure the impact. Did IER improve?
6. **Repeat** until satisfied or iteration budget is exhausted

This is not a single LLM call. It's a multi-turn tool-use loop where the agent decides which tool to call next based on the results of the previous call. The agent is Claude, running via the Anthropic API with `tools=` parameter, making autonomous decisions about how to optimize the pipeline.

**What the eval suite actually measures:**

```python
async def _exec_run_eval_suite(context):
    for ev in context.evals:
        intent_result = await extractor.extract(
            text=ev["utterance"],
            intent_prompt=context.intent_prompt,
            config=context.config,
        )
        results.append({
            "utterance": utterance,
            "expected": expected,
            "actual": intent_result.detected_intent,
            "confidence": intent_result.confidence,
            "correct": intent_result.detected_intent == expected,
        })

    ier = 1.0 - (correct / total)
```

Every utterance goes through the full pipeline — intent prompt + current config → Claude API → parsed result → match against expected intent. IER is the percentage that got it wrong. This is the number the system is trying to minimize.

**What "update prompt config" means concretely:**

The agent rewrites three things:
- `system_prompt` — The full system prompt sent to Claude for intent classification
- `confidence_threshold` — The minimum confidence below which the system abstains
- `asr_hints` — Keywords hinted to the ASR provider for better transcription

The system prompt is the biggest lever. When the agent sees that "cancel subscription" and "billing inquiry" are confused, it adds explicit disambiguation instructions to the prompt. When it sees that filler-word utterances are failing, it adds instructions to ignore disfluency markers. The prompt evolves through the loop based on empirical failure data, not human intuition.

## Agent 3: The Adversarial Agent

**Role:** Cluster failures by confused intent pairs, generate targeted edge cases, and propose component-level fixes.

The Adversarial Agent is the red team. It takes the `EvaluationReportArtifact` from the most recent eval run and produces an `AdversarialFindingsArtifact` containing three things: failure clusters, proposed adversarial examples, and recommended component changes.

**Failure clustering:**

```python
def _cluster_failures(self, report):
    grouped = defaultdict(list)
    for case in report.hard_cases:
        if not case.correct:
            grouped[(case.expected_intent, case.predicted_intent)].append(case.example_id)

    return [
        FailureCluster(
            title=f"{expected} confused with {predicted}",
            affected_intents=[expected, predicted],
            suspected_component_ids=["intent_classifier", "decision_policy"],
            example_ids=example_ids,
        )
        for (expected, predicted), example_ids in grouped.items()
    ]
```

The clustering is not random. It groups failures by *confused intent pairs* — "cancel_subscription confused with billing_inquiry" is a cluster. "check_balance confused with account_status" is a different cluster. Each cluster identifies the affected intents and the suspected component.

**Adversarial example generation:**

For each failed case, the agent generates variants designed to stress-test the same failure mode:

```python
variants = [
    f"um {source_text}",           # filler word prefix
    f"can you please {source_text}", # polite framing
    f"{source_text} right now",      # urgency suffix
]
```

Each variant is tagged with `phenomenon_tags=["adversarial", "filler_words", "failure_driven"]` and annotated with `confusable_with=[predicted_intent]`. These examples are added to the eval set and the ETL agent re-optimizes — now with harder examples that target the known failure modes.

**Component-level recommendations:**

The adversarial agent doesn't just generate examples. It proposes specific fixes:

- *"Add failed-example keywords to ASR hints"* — When misclassification traces to transcription errors
- *"Expand classifier disambiguation rules for confused intent pairs"* — When the classifier prompt doesn't distinguish similar intents
- *"Lower the abstain threshold slightly for ambiguous but in-domain utterances"* — When the decision policy is too aggressive about falling back

These recommendations are stored in `recommended_component_changes` and inform the next ETL optimization cycle.

## The Loop in Motion

Here's what a complete build looks like:

```
Iteration 0:
  Eval Curator → 49 examples (7 per intent × 6 intents + 7 fallback)
  ETL Agent → runs eval, IER = 0.22, rewrites prompt
  ETL Agent → re-runs eval, IER = 0.14

Iteration 1:
  IER 0.14 > target 0.08
  Adversarial Agent → clusters 7 failures into 3 groups
  Adversarial Agent → generates 21 adversarial examples
  ETL Agent → re-optimizes with 70 examples, IER = 0.10

Iteration 2:
  IER 0.10 > target 0.08
  Adversarial Agent → clusters 7 failures into 2 groups
  Adversarial Agent → generates 21 more adversarial examples
  ETL Agent → re-optimizes with 91 examples, IER = 0.07

  IER 0.07 ≤ target 0.08 → Pipeline ready
```

Three iterations. Zero human intervention. The pipeline went from 22% intent error rate to 7% by generating its own test cases, finding its own failures, and fixing its own configuration.

## What Changes Operationally

**Before this system:**
1. Product manager writes intent descriptions
2. Engineer writes a classifier prompt
3. Engineer tests with 5-10 examples
4. Deploy and hope
5. Customer complaints surface failures weeks later
6. Engineer manually adjusts prompt, repeat from step 3

**After this system:**
1. Product manager writes intent descriptions via a prompt
2. System builds, tests, and hardens the pipeline automatically
3. Status polling shows build progress — eval curator complete, ETL optimizing, IER at 0.12...
4. Pipeline reaches target IER and becomes ready to invoke
5. Invoke uses the optimized config — the prompt, thresholds, and hints that survived the improvement loop

The human contribution is the *what* — which intents matter. The system handles the *how* — which prompt works, which threshold is right, which examples break it.

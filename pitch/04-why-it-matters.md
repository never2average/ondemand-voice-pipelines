# From Prompt-and-Pray to Continuous Improvement

Every voice pipeline in production today is exactly as good as the day it was deployed. This one gets better.

That's not a tagline. It's an architectural property. The system doesn't just classify intents — it generates evaluations, measures its own performance, finds its own failures, and iterates its own configuration. The pipeline *is* the improvement loop. Deployment is not the end of the process. It's the beginning.

## What "Prompt-and-Pray" Actually Costs

The current state of voice pipeline engineering is roughly where web development was before automated testing. People build things, poke at them manually, convince themselves they work, and ship.

The process looks like this:

1. Write an intent classification prompt
2. Test it on the five utterances you can think of
3. They all work — ship it
4. A month later, customer complaints reveal that "reschedule my appointment" routes to "cancel appointment" 40% of the time
5. Fix the prompt for that case. Break two other cases in the process
6. Repeat until the next reorg

This is not engineering. This is folk medicine. And it persists not because engineers are careless, but because the infrastructure for doing it right didn't exist. There was no way to automatically generate comprehensive eval suites. There was no way to run them against the pipeline systematically. There was no way to analyze failures, propose fixes, and re-evaluate in a loop.

Now there is.

## What Changes Operationally

Three shifts that matter:

**No manual prompt tuning.** The ETL Pipeline Agent rewrites the classifier system prompt based on empirical failure data. It sees which utterances are misclassified, which intents are confused, and what patterns the failures share. It rewrites the prompt to address those specific failure modes. Then it re-evaluates to verify the fix worked. This is not "try a different prompt and see" — it's measure-diagnose-fix-verify, automated.

**No hand-curated test sets.** The Eval Curator Agent generates diverse test utterances with phenomenon coverage — filler words, polite forms, question framing, out-of-domain, description-based. It splits them into train, dev, and immutable holdout. The adversarial agent extends the set with targeted edge cases that stress-test known failure modes. The eval set is comprehensive *because it was generated to be*, not because someone remembered to add an example for "uh can you check my balance."

**No "it works on my examples."** The holdout split exists specifically to catch overfitting. The system can optimize IER on train and dev all day — if holdout IER doesn't improve, the optimization is memorizing patterns instead of learning them. The three-way split is the difference between "it works" and "it generalizes."

## The Tradeoff You're Actually Navigating

Every voice pipeline makes implicit choices along three axes:

```
          Accuracy
            /\
           /  \
          /    \
         / your \
        / pipeline\
       /    here   \
      /--------------\
   Cost            Adaptability
```

**Accuracy:** How often does the pipeline get the intent right? Measured by IER on holdout.

**Cost:** How many API calls, how much compute, how much latency per classification? Every improvement iteration costs Claude API calls. Every reranker step adds latency. Every additional eval example extends build time.

**Adaptability:** How quickly can the pipeline adjust to new intents, new utterance patterns, new failure modes? A hand-tuned prompt is cheap and accurate but brittle — it adapts slowly. A fully automated loop is expensive to run but adapts in minutes.

Most teams are pinned to the **Cost–Accuracy edge**: they hand-tune a prompt until it's accurate enough on the examples they know about, and they keep costs low by not running evaluations. They sacrifice adaptability entirely.

This system lets you move along the surface deliberately:

- **High accuracy, high cost, high adaptability:** Run the full build loop with multiple adversarial iterations. IER drops to target. Build takes minutes and consumes API calls. New intents can be added and the loop re-runs.
- **Moderate accuracy, low cost, moderate adaptability:** Run a single eval-and-optimize pass. Skip adversarial iterations. Good enough for low-stakes applications.
- **High accuracy, moderate cost, moderate adaptability:** Set a tight IER target but cap iteration count. The system optimizes as far as it can within budget.

The `OptimizationObjective` in the build context controls where on this surface the system operates. The artifact architecture makes the tradeoff explicit — you can see exactly how many iterations it took, how many API calls were consumed, and what IER was achieved at each step.

## What the Artifact Architecture Enables

The typed artifact chain is not just good engineering hygiene. It enables three capabilities that unstructured systems can't provide:

**Reproducibility.** Every build produces a chain of artifacts — `IntentSchemaArtifact` → `EvalDatasetArtifact` → `PipelineGraphArtifact` → `EvaluationReportArtifact` → `AdversarialFindingsArtifact`. Each artifact references its inputs via `ArtifactRef`. You can reconstruct any build, re-run any evaluation, and diff any two pipeline configurations.

**Debuggability.** When IER regresses, the `EvaluationReportArtifact` contains per-intent accuracy, a full confusion matrix, and component-level traces for hard cases:

```python
class EvalCaseResult(BaseModel):
    example_id: str
    expected_intent: str
    predicted_intent: str
    confidence: float
    correct: bool
    component_traces: list[ComponentTrace]

class ComponentTrace(BaseModel):
    component_id: str
    component_kind: str
    input_snapshot: dict[str, Any]
    output_snapshot: dict[str, Any]
    latency_ms: int
```

Each hard case carries a full trace — what went into each component, what came out, how long it took. This is attribution, not guessing.

**Composability.** The `PipelineGraphArtifact` separates component specs from the graph topology. You can swap the ASR provider without touching the classifier. You can add a reranker without changing the decision policy. You can version each component independently. The graph structure makes the pipeline modular in practice, not just in theory.

## Who This Is For

This system is for teams that:

- Run voice pipelines where intent accuracy directly affects customer experience or revenue
- Have enough volume that a 5% IER improvement is worth automating
- Are tired of manually tuning classifier prompts and never knowing if they made things better or worse
- Want to add new intents without re-testing the entire pipeline by hand
- Need auditability — the ability to explain *why* the pipeline classifies a given utterance the way it does

This system is *not* for teams that:

- Have three intents and can test them manually in five minutes
- Don't have tolerance for build-time API costs (the improvement loop makes Claude calls)
- Need sub-100ms classification latency (the system uses LLM-based classification, not a lightweight model)

## The Closing Insight

There is a category error embedded in most voice pipeline work. Teams treat the pipeline as a *static artifact* — something you build, configure, deploy, and maintain. The ASR is the pipeline. The classifier prompt is the pipeline. The confidence threshold is the pipeline.

But the ASR model will encounter utterance patterns it hasn't seen. The classifier prompt will face intent combinations it doesn't handle. The confidence threshold will be wrong for edge cases that haven't surfaced yet.

The pipeline isn't the ASR model or the classifier. The pipeline is the loop that improves them.

A static artifact degrades from the moment it's deployed, because the world changes and the artifact doesn't. A loop that evaluates, identifies failures, generates adversarial tests, and re-optimizes configuration — that's a system that *responds* to the world instead of assuming it stays fixed.

The question is not "how good is your voice pipeline today." The question is "how does your voice pipeline get better tomorrow."

This system is the answer: not a better model, but a better loop. Not a smarter prompt, but a system that writes, tests, and rewrites its own prompts until the failures stop. Not configuration — continuous improvement, encoded as architecture.

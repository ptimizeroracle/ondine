# Troubleshooting

Common issues and how to fix them. If your problem isn't here, please
[open an issue](https://github.com/ptimizeroracle/ondine/issues).

## "Authentication failed" / missing API key

**Symptom:** the run fails with an authentication or invalid-API-key error as
soon as it makes its first call.

**Cause:** the provider's API key isn't set. Ondine reads keys from environment
variables by convention — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`,
and so on.

**Fix:** set the key for your provider, then re-run:

```bash
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

Run `ondine list-providers` to see the providers Ondine knows about. You can
also pass the key explicitly instead of via the environment:

```python
.with_llm(provider="openai", model="gpt-4o-mini", api_key="sk-...")  # pragma: allowlist secret
```

To find out what a run *would* cost before spending anything — and without a key —
estimate first:

```bash
ondine process --config config.yaml --dry-run
```

## "Rate limit exceeded" (HTTP 429)

**Symptom:** intermittent `429` errors, especially on a free tier or a large run.

**Cause:** you're sending requests faster than the provider allows.

**Fix:** cap the request rate. Ondine paces calls to stay under the limit:

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="groq", model="llama-3.3-70b-versatile")
    .with_rate_limit(30)   # requests per minute — match your provider's limit
    .build()
)
```

Transient `429`s are also retried automatically; see
[Error Handling & Retries](error-handling.md).

## "Context window exceeded" / prompt too long

**Symptom:** an error about the context length or maximum tokens being exceeded.

**Cause:** with batching, one call packs many rows into a single prompt. Too many
rows (or very long rows) overflow the model's context window.

**Fix:** lower the batch size so each call carries fewer rows:

```python
.with_batch_size(10)   # fewer rows per call = shorter prompts
```

If individual rows are long, reduce it further (even `1`). Smaller batches make
more calls but each one fits.

## "Budget exceeded" — the run stopped early

**Symptom:** the run halts partway with a budget-exceeded error.

**Cause:** you set a spending cap with `.with_max_budget()` and the run reached it.
This is working as intended — the cap is a safety net.

**Fix:** estimate the true cost first, then set the budget above it (or remove it):

```python
pipeline = (
    PipelineBuilder.create()
    .from_csv("data.csv", input_columns=["text"], output_columns=["result"])
    .with_prompt("Process: {text}")
    .with_llm(provider="openai", model="gpt-4o-mini")
    .with_max_budget(50.0)   # raise to fit the estimate, or drop this line
    .build()
)

print(pipeline.estimate_cost().total_cost)   # check before running
```

See [Cost Estimation & Budgets](cost-control.md).

## Resuming after a crash

**Symptom:** a long run died (network blip, Ctrl+C, machine slept) and you don't
want to pay to redo the rows that already finished.

**Fix:** if checkpointing was on, resume from the session that failed. Every LLM
response already paid for is reused, not re-requested:

```python
try:
    result = pipeline.execute()
except Exception:
    # session_id is set as soon as the run starts, so it's here even after a crash
    result = pipeline.execute(resume_from=pipeline.session_id)
```

Full details, including streaming caveats, in
[Checkpointing & Recovery](checkpointing.md).

## A hook or CI step fails for reasons unrelated to my change

If you're working in a git worktree and a pre-commit hook fails inexplicably,
check which worktree you're committing from — hooks are shared and resolve the
environment from the current directory. See the note in
[Contributing](../contributing.md).

# ondine.plan() -- Intent Layer

`ondine.plan()` turns a plain-language goal and a sample of your data into a drafted pipeline configuration -- **without running anything**. You inspect the draft, check the projected cost, and only then build a real `Pipeline`.

```python
from decimal import Decimal
from ondine import plan

drafted = plan(
    data="reviews.csv",
    goal="Extract sentiment and a one-line summary per review",
    budget=Decimal("5.0"),
)

print(drafted.preview_yaml())      # inspect / edit before committing
print(drafted.estimated_cost)      # projected spend, no LLM call needed

pipeline = drafted.build()         # materialize a normal Pipeline
result = pipeline.execute()        # deterministic, budget-capped, as always
```

## This Is Not an Agent Loop

`plan()` makes **exactly one** structured LLM call to draft a `PipelineSpecifications` from your goal + a data sample. There is no re-planning, no tool use, no multi-step reasoning loop, and no execution inside `plan()` itself.

**The safety model:** nondeterminism is confined to the single drafting call. The plan it produces is inspectable and approvable *before* anything executes. Once you call `.build()`, execution runs through the exact same deterministic, budget-capped `Pipeline` as every other Ondine entry point -- no LLM decides what happens next mid-run.

## The `Plan` Object

`plan()` returns a `Plan` -- an immutable handoff, not a running job:

| Attribute / Method | What it gives you |
|---|---|
| `.specifications` | The drafted `PipelineSpecifications` (input/output columns, prompt template, response format). |
| `.goal` | The goal string you passed in, unmodified. |
| `.rationale` | One-sentence explanation the LLM gave for its choices. |
| `.estimated_cost` | A lazily-computed `CostEstimate` -- reuses `Pipeline.estimate_cost()`'s sample-based token/pricing estimator, so checking it makes **no extra LLM call**. |
| `.preview_yaml()` | The spec as YAML, round-trippable through `ConfigLoader.from_yaml` -- edit on disk, reload, or just read it. |
| `.build()` | Routes through the standard `PipelineBuilder` to produce a real `Pipeline`. No separate execution path. |

## Approve Before You Spend

Check the projected cost before committing to `.build()`:

```python
drafted = plan(data="reviews.csv", goal="Extract sentiment", budget=Decimal("5.0"))

if drafted.estimated_cost.total_cost > 2.0:
    print("Too expensive, refine the goal or lower budget")
else:
    result = drafted.build().execute()
```

`budget` is mandatory on `plan()` -- there's no sensible default for "how much should this cost," and it flows straight onto the drafted spec's `processing.max_budget`, enforced the same way as any hand-built pipeline.

## What the LLM Decides vs. What You Decide

The drafting call only fills in the knobs a non-expert can't be expected to know how to set: which existing columns feed the prompt, which new columns to produce, the prompt template itself, and the response format (raw text vs. JSON for multi-column output).

Anything you *can* reasonably set yourself -- budget, model, rate limits, concurrency -- is a caller-supplied constraint, never guessed by the LLM.

```python
drafted = plan(
    data="products.csv",
    goal="Categorize products and score confidence",
    budget=Decimal("2.5"),
    model="anthropic/claude-3-5-sonnet",   # used both to draft AND to run
)
```

## Defensive Validation

The draft is checked against your real data before a `Plan` is returned:

- Drafted `input_columns` that don't exist in your data -> `ValueError`, not a hallucinated column name shipped to execution.
- Overlapping `input_columns` / `output_columns` -> `ValueError`.
- An empty `goal` or non-positive `budget` -> `ValueError`, before a single token is spent.

## Related

- [enrich()](enrich.md) -- the one-call front door when you already know exactly what you want
- [Structured Output (Pydantic)](structured-output.md) -- what `response_format="json"` maps to under the hood
- [Cost Estimation & Budgets](cost-control.md) -- the estimator `.estimated_cost` reuses

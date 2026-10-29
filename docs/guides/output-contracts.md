# Output Contracts

An output contract says what a deliverable value looks like. Ondine checks every model answer against it before it delivers the value. A value that breaks the contract is not delivered silently: the row is marked, explained and listed in a quarantine table.

Contracts are opt-in. A pipeline with no contract behaves exactly as before.

## Quick Reference

| Builder method | Declares | Example |
|---|---|---|
| `with_output_field()` | Rules for one output column | Allowed labels, a number range, a pattern |
| `with_output_relation()` | A rule between two columns | `end >= start` |
| `with_structured_output(..., as_contract=True)` | The Pydantic schema's rules | `Literal` values, bounds, validators |
| `with_contract_policy()` | What to do with failing rows | Reject or deliver flagged |

## What a contract checks

You declare rules per output column. Ondine reads each answer into the declared type, applies the cleanups you allowed, then runs the rules. It runs a field-to-field rule only when both fields passed their own rules. One bad value then gives one reason, not two.

A contract is checked when you call `build()`. Every problem is listed at once, for example a minimum above its maximum or a rule that names an unknown column. Nothing is spent on a broken contract.

Columns you do not declare are not checked.

## The six statuses

Every input row ends the run with exactly one status. The status decides what happens to the row's value.

| Status | Meaning | Value delivered? |
|---|---|---|
| `accepted` | The answer passed every check without a fix. | Yes |
| `repaired` | A check failed first, then the answer passed after a format fix. | Yes |
| `rejected` | An answer came back but failed a check. | No, blanked |
| `unverified` | The answer failed a check, but you chose `deliver_flagged`. | Yes, flagged |
| `failed` | No usable answer: the provider or a stage failed, or the run stopped first. | No, blanked |
| `unchecked` | No contract was declared. | Yes, as before |

A format fix happens only when exactly one allowed value appears in a wordy answer. For example, "I think it is refund." becomes `refund`. An answer that names two allowed values is rejected.

`repaired` means a check failed before the fix. A row that auto-retry recovers after a provider failure is `accepted`: its answer failed no check.

## Declaring a contract

### Allowed values for one column

```python
builder.with_output_field(
    "intent",
    allowed=["refund", "billing", "shipping"],
    aliases={"money back": "refund"},
    normalise=["strip", "casefold"],
)
```

An alias is delivered as the allowed value it names. The cleanups run before checking. Choose them from `strip`, `collapse_whitespace`, `strip_punctuation` and `casefold`.

### A numeric range

```python
builder.with_output_field("score", type="integer", minimum=1, maximum=5)
```

The types are `text` (the default), `integer`, `number`, `boolean` and `any`. Bounds are inclusive.

Other keywords of `with_output_field()`: `pattern`, `min_length`, `max_length`, `required` and `absent`. An optional column (`required=False`) may be absent. Use `absent="n/a"` to name the answer that means "no value". It is delivered empty.

### A field-to-field rule

```python
builder.with_output_relation("end_date", ">=", "start_date")

# Compare an output with an input column of the same row
builder.with_output_relation("refund", "<=", input_column="order_total")
```

The comparisons are `<`, `<=`, `==`, `!=`, `>=` and `>`.

### Enforcing a structured-output schema

```python
from typing import Literal

from pydantic import BaseModel, Field


class Ticket(BaseModel):
    intent: Literal["refund", "billing", "shipping"]
    urgency: int = Field(ge=1, le=5)


builder.with_structured_output(Ticket, as_contract=True)
```

Without `as_contract=True`, the schema shapes the answer but rejects nothing. With it, the schema's types, `Literal` and `Enum` values, bounds, patterns, lengths and validators become checks. A validator's own error message becomes the reason a row fails.

For a batch schema with one `items: list[Row]` field, the contract is read from `Row`.

## Run policy

Set the policy with `with_contract_policy()`. Each call sets the whole policy: an option left out takes its default.

| Option | Values | Default | Effect |
|---|---|---|---|
| `on_failure` | `"reject"`, `"deliver_flagged"` | `"reject"` | `reject` blanks a failing row. `deliver_flagged` keeps its value with the status `unverified`. |
| `partial_rows` | `"whole_row"`, `"keep_valid_fields"` | `"whole_row"` | `whole_row`: one failed required field rejects the whole row. `keep_valid_fields`: the row is still rejected, but the fields that passed keep their values. |
| `early_stop_fraction` | a share, or `None` | `0.5` | Declared, not enforced yet (see Current limits). |
| `early_stop_window` | a row count | `50` | Declared, not enforced yet. |

```python
builder.with_contract_policy(on_failure="deliver_flagged")
```

A policy with nothing declared to check fails `build()`.

## What you get back

### Five extra columns

With a contract, the result's dataframe has five more columns after your output columns:

| Column | Content |
|---|---|
| `status` | One of the six statuses. |
| `reason` | The check that blocked delivery last, then the path taken, for example `[path: attempt 1 failed check]`. |
| `checks` | The result per field, for example `intent: accepted; score: rejected`. |
| `repair_method` | `none` or `format`. `re-ask` is reserved for when re-ask exists (see Current limits). |
| `delivered_by` | The model that produced the delivered value. |

If an input or output column already uses one of these names, `build()` refuses the pipeline and lists every clash. A column in your data that the prompt never reads, such as a ticket's own `status`, is refused when the data loads, before the first model call. Either way nothing is spent. Rename the column. Without a contract, these names are free: no column is added.

### The quarantine table

`result.quarantine` lists every row whose status is `rejected`, `unverified` or `failed`. Each entry holds:

- the row's position (`row`) and its input values (`inputs`);
- the `status`, the `reason` and any `sub_reason`;
- the raw answer of each attempt that failed a check or could not be read (`raw_answers`), for the first 1,000 quarantined rows. A rejected, unverified or unparseable row has them. An attempt whose provider call failed has no answer, so it has none;
- every failed check, with its field, rule and reason (`failed_checks`);
- the models used (`models`) and the cost of the row (`cost`).

Counts stay exact past 1,000 rows. Only the raw answers are capped.

### The run report

`result.report` summarises the run:

- the rows per status, which always sum to the input row count;
- the coverage: accepted plus repaired, over all rows;
- the most frequent rejection reasons;
- the pass rate of each rule;
- the cost: total, per delivered row, and spent on rejected rows;
- warnings, for example a rule that fails on most rows.

Print it with `print(result.report.render())`.

### Rejected rows are not lost rows

| Attribute | With a contract |
|---|---|
| `result.rejected_row_indices` | Rows with the status `rejected`. |
| `result.lost_row_indices` | Rows with the status `failed`. Rejected rows are not in it. |
| `result.errors` | Read from the row statuses, so it agrees with the `status` column. |
| `result.is_complete` | True only when every row is `accepted` or `repaired`. |

```python
result = pipeline.execute()
df = result.to_pandas()

good = df[df["status"].isin(["accepted", "repaired"])]
print(result.report.render())
print(result.quarantine)
```

### Retries and resume

Auto-retry re-runs only `failed` rows, never `rejected` ones.

A checkpoint records the contract it was written under. Resuming with a different contract is refused before any model call. The message names the part that changed, for example `the output contract changed since it started (run policy changed)` or `(field 'intent' changed)`. Resume with the original contract, or start a new run.

A checkpoint written before contracts existed cannot prove which rules it used. Ondine refuses to resume it under a contract. It resumes as before without one.

### Streaming

With `execute_stream()`, each chunk's result holds that chunk's delivered dataframe with its statuses, and `chunk.report` covers that chunk only. `pipeline.run_report` covers the whole run, every chunk merged in. Read it when the stream ends.

## With no contract

Nothing changes:

- the dataframe has the same values and no extra columns;
- `result.errors`, `result.lost_row_indices`, `result.is_complete` and the metrics are the same;
- resume works as before, for old and new checkpoints.

Two outputs are new and sit next to the result: `result.quarantine` lists the failed rows, and `result.report` counts every row as `unchecked` or `failed`.

## Current limits

| Limit | What happens today |
|---|---|
| No early stop yet | `early_stop_fraction` and `early_stop_window` are validated, but a run does not stop early. |
| No provider-batch mode | `submit()` with a contract raises an error. See [Provider Batch API Mode](provider-batch.md). |
| No re-ask or escalation yet | A rejected row is not sent back to the model, and no stronger model is tried. |

## Related

- [Structured Output (Pydantic)](structured-output.md)
- [Error Handling & Retries](error-handling.md)
- [Checkpointing & Recovery](checkpointing.md)

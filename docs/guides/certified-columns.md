# Certified Columns

A contract blocks values that are malformed. It cannot see a value that is well formed but wrong, such as the label `card_delivery_estimate` delivered for a question about `card_arrival`. On our test sets, 6–11% of delivered values were wrong in that way.

Certification measures it. You give labels you trust, and Ondine says how many delivered values are right, with an exact 95% range, the most frequent confusions, and the wrong rows.

Certification is opt-in. It never changes the delivered frame or the run report.

## Quick Reference

| Call | Use it when | You get |
|---|---|---|
| `pipeline.certify(result, gold="true_label")` | You have a label for every delivered row (a labelled dev set) | The exact accuracy |
| `pipeline.draw_sample(result)`, then `pipeline.certify(result, gold=labels, sample=sample)` | You label only the rows Ondine draws | The accuracy with an exact 95% range |
| `pipeline.certify(result, gold={row: label, ...})` | You labelled rows you picked yourself | A **check**: agreement on those rows only |
| `pipeline.export_review(result, "review/")`, then `pipeline.read_review("review/")` and `pipeline.certify(labels.run, gold=labels)` | You have no labels, and people who can label a sample | The accuracy against **blind human labels**, with an exact 95% range |
| `pipeline.audit(result, model="openai/gpt-6-luna", max_cost=0.50)` | You have no labels, and a stronger model to compare with | An **audit**: agreement with that model, with an exact 95% range. Never accuracy |

## Labels for every row

```python
result = pipeline.execute()
cert = pipeline.certify(result, gold="true_intent")
print(cert.render())
```

`gold` names a column of your data. It can also be a list aligned with the input rows, or a mapping from row position to label. With several output fields, pass a mapping from field to labels.

## Labels for a sample Ondine draws

```python
sample = pipeline.draw_sample(result, size=200)   # or precision=0.03
labels = my_labelling(sample.inputs)             # {row: label}, any way you like
cert = pipeline.certify(result, gold=labels, sample=sample)
```

- `sample.inputs` holds the input columns the prompt reads, for the drawn rows only. It never holds the model's answer, so labellers are not led by it.
- Ondine chooses the seed from the run itself. Drawing again on the same result returns the same rows. You cannot redraw until the number looks good.
- A sample belongs to one result. Certifying another result with it is refused.
- `precision=0.03` sets the size before drawing, so that the range is at most ±3 points at worst.
- If the run delivered no more rows than the size, every row is used. That is a census, with no range.

## What the certificate shows

```
Certificate: accuracy against your labels, on a sample Ondine drew
  Basis:  sample of 100 of 231 delivered rows (100 rows)

  answer (labels)
    Accuracy: 80.0% (95% range 73.2%–85.3%)
    Compared: 100 of 231 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    By status: accepted 80.0% [73.2%–85.3%] (n=100)
    By model: openai/gpt-6-luna 80.0% [73.2%–85.3%] (n=100)
    ...
```

This is a real run: GPT-6 Luna on 231 bank-support messages with 77 intents. Its accuracy against the labels of every row was 81.4%, inside the sample's range.

- **Accuracy** is the share of compared rows whose delivered value equals the label. The range contains the column's true accuracy in at least 95% of repeated certifications.
- **By status, model and value**: each line has its own exact range. A line marked "few rows" has fewer than 20 compared rows; read it with care. Reading many lines at once is not covered jointly.
- **Confusions** are the most frequent delivered → label pairs. **Wrong rows** lists the compared rows the labels contradict.
- `cert.to_dict()` gives the same content as data.

## How values are compared

Each label is read by the field's contract rules, exactly as the model's answer was read: cleanups, aliases and allowed values. A label that breaks the rules is an error. Every such label is listed in one error, and nothing is certified until you fix them. The rules are never loosened for labels: without `casefold`, `Refund` is not `refund`.

- A cell may hold a list of acceptable answers. The delivered value is right if it equals any of them.
- An empty cell on a compared row makes the row **unresolved**. For the range, the low end counts it wrong and the high end counts it right, so the range stays valid whatever the row holds.
- Numbers compare exactly unless you pass `tolerance={"field": 0.5}`.

## Which rows are certified

Each field is certified on its own rows: the rows that delivered a value for it. Rejected and failed rows are delivered empty, so they are counted apart and never certified. Under `partial_rows="keep_valid_fields"`, a rejected row still delivers the fields that passed, and those are certified. With several fields, the certificate also gives the share of rows with every field right.

v1 certifies fields with allowed values, booleans, integers and numbers. Free-text fields are listed as not certified, with the reason.

## Labels from people

When you have no labels, ask people to label a sample. They work blind, in a spreadsheet. The claim is "accuracy against blind human labels".

```python
result = pipeline.execute()
review = pipeline.export_review(result, "review/")   # 200 rows, 2 labellers, xlsx
print(review.render())                               # who gets which file

# Days later, in a new process built from the same pipeline code:
labels = pipeline.read_review("review/")             # every problem listed at once
print(labels.render())                               # progress per sheet
cert = pipeline.certify(labels.run, gold=labels)
print(cert.render())
```

`export_review` takes `sample=` (default: the rows `draw_sample` gives, the same rows an audit uses), `labellers=` (1 or 2), `double=` (at least 0.2), `format=` (`"xlsx"` or `"csv"`), `task=` and `fields=`. Give `task=` your own labelling guide. Otherwise the sheet shows the prompt and system prompt as written.

### What goes where

```
review/for_labellers/   sheet-A.xlsx, sheet-B.xlsx      (csv: sheet-A.csv, sheet-B.csv, task.txt)
review/keep_private/    manifest.json, run.json
```

- **Send each person their own file, and nothing else.** Never send `keep_private/`. `run.json` holds the run's answers. `manifest.json` maps each item to its row.
- Each sheet has an `item` column, one `input: ...` column per input column, one `label: ...` column per certified field, and `notes`.
- A labeller never sees the delivered value, the harness columns, other source columns, few-shot examples or the rendered prompt. The Task text shows the system prompt as written. If it holds examples or anything labellers must not see, pass `task=` instead.
- Rows are shuffled in each sheet, from a seed Ondine derives. Item ids are opaque. Exporting again gives the same sheets, so nobody can reshuffle until agreement looks good.
- XLSX sheets also hold a `Task` worksheet, and a dropdown of the allowed values. The dropdown only warns, so `a | b` can still be typed.
- XLSX needs the `excel` extra: `pip install 'ondine[excel]'`. Without it, use `format="csv"`.
- Spreadsheet apps can change CSV inputs, such as leading zeros and long numbers, and Ondine then refuses the edited input. Prefer XLSX. In a CSV, a cell that starts with `=`, `+`, `-` or `@` gets a leading `'`, so a customer's text is never run as a formula. Ondine removes it on import.
- Ondine refuses a folder that is not empty. Sheets that came back are never overwritten.

### What labellers type

| Cell | Means |
|---|---|
| `card_arrival` | One answer. It is read by the field's contract rules, as gold would be. |
| `card_arrival \| card_delivery_estimate` | Either one is right. |
| `?` alone | Cannot tell. The row is unresolved. |
| empty | Not labelled. The row is unresolved. |
| `-` | No value. Optional fields only. It is an error on a required field. |
| `true`, `false` | A boolean, in any case. Excel's TRUE and FALSE work. |
| `12`, `3.5`, `1e3` | A number, with a dot. `3,5` is an error: "use a dot". |

Ondine reads the whole cell first, so an allowed value that contains `|` still works. The rules are never looser than for gold. Without `casefold` in the contract, `Refund` is not `refund`. A typo is an error, never an unresolved row.

### Reading the sheets back

`read_review` checks everything, then raises one `ReviewSheetError` that lists every problem. Each problem names the file, the sheet row, the item and the column. A label that nearly matches an allowed value gets a suggestion. `error.problems` holds the same as data. The first 50 are printed.

Ondine accepts re-sorted rows, extra columns and the `notes` column. It lists:

- a missing or unknown `label:` or `input:` column;
- a missing, unknown or duplicate item, or an item on the other labeller's file;
- an input cell that differs from the exported one, which also catches an input or item column sorted alone. A label column sorted alone is not detected;
- each label that breaks the cell syntax or the field's rules.

It refuses outright, in one line:

- sheets exported from another run, other delivered values or another output contract;
- a `run.json` that is not the run the manifest names;
- a sample that Ondine cannot redraw, such as after a numpy change. The message names both numpy versions;
- a sheet whose items belong to another sheet id.

A run with no output contract needs the same `fields=` at `read_review` and `certify` as at `export_review`. `read_review(folder, result=result)` also checks that `result` is the exported run. `read_review(folder, sheets={"A": "mail/ada.xlsx"})` reads a returned file kept elsewhere.

### Two labellers

With `labellers=2`, each person gets their own file. Each person sees only their own labels. A random share of the rows (`double`, 20% by default) goes to both. Doubled rows are not marked.

Ondine judges each person's answers against the delivered value:

- Both right: right.
- Both wrong: wrong.
- One right and one wrong: unresolved, "labellers disagree".
- One answered and the other gave `?` or nothing: the answer counts.
- Neither answered: unresolved, "cannot tell" if either typed `?`, else "not labelled".

Unresolved rows widen the range both ways, so the range stays valid whatever they hold. That includes disagreement that clusters on wrong rows.

The certificate prints how far the two agreed, per field:

```
    Labellers: 2 (one sheet each); 40 rows labelled twice: agree on 36 (90.0%), kappa 0.89 on 37 single-answer rows; 3 split on the delivered value (unresolved)
```

Kappa is Cohen's kappa, for labels and booleans, on the rows where each person gave one answer. If agreement is under 80%, or kappa is under 0.60, the certificate warns that the labels are noisy and accuracy against them is loose. With fewer than 20 rows answered by both, the line says "few rows". With one labeller, it says agreement is not measured.

### What is checked, and what is not

- **Checked:** the range's coverage (exact; see `benchmarks/ruler/`). The sheet's rules are tested end to end: blindness, order, import problems, the two-labeller rule and both file formats.
- **Not yet checked:** how well people agree on a real task, how long labelling takes, and whether our adjudicated gold matches people. The human pilot (`benchmarks/ruler/PILOT.md`) measures these. It has not been run. Until `benchmarks/ruler/results/pilot.md` exists, treat agreement and labelling time as unknown.
- **Never covered:** what the labels themselves get wrong. A labeller who is wrong makes the accuracy wrong too. The agreement line shows how noisy the labels are. It cannot show a mistake both people share.

## A run with no contract

Say how the outputs are read:

```python
from ondine.harness.contract import OutputField

cert = pipeline.certify(
    result, gold="true_intent", fields=[OutputField("intent", allowed=LABELS)]
)
```

## Audit against a model

With no labels, a stronger model can stand in for them. Ondine asks it the sampled rows again, through the run's own prompt, and compares its answers with the delivered values.

```python
estimate = pipeline.estimate_audit(result, model="openai/gpt-6-luna")
print(estimate.render())   # what leaves, to whom, and the most it costs; spends nothing

audit = pipeline.audit(result, model="openai/gpt-6-luna", max_cost=0.50)
print(audit.render())
```

**An audit is not accuracy.** Models share mistakes, across vendors too. When a reference agrees with a wrong answer, the audit counts it as right. On our golden sets, agreement with a stronger model overstated accuracy by 4 to 12.5 points. In a live run, gpt-oss-20b classified 231 bank-support messages into 77 intents. On 200 sampled rows, it agreed with GPT-6 Luna on 90.5% (range 88.7%–91.7%), but matched the labels on only 79.5%. Luna flagged 19 rows, and 13 of them were really wrong. It agreed with the other 28 wrong answers, so the audit missed them. Use an audit to compare runs and to find wrong rows to review. For a number you can claim, certify against labels.

- **What the reference sees:** the prompt the run's model saw for each sampled row, with the same system prompt, examples, preprocessing and retrieved passages. It never sees the run's answer. `estimate.render()` names everything that leaves.
- **What it shares with the run:** nothing. No context store, no response cache, no checkpoints, no output file. Its answers never reach the run's ledger. The result and its run report are unchanged.
- **How its answers are read:** by the run's contract. An answer that breaks it is asked again once, with the reason. A row still without a valid answer is **unresolved** ("broke the contract"), and so is a provider failure ("provider"). The certificate prints the causes next to the count.
- **Cost:** `max_cost` is required. It is a hard cap for the audit, separate from the run's budget. Ondine asks the rows in chunks of 50. It asks only the rows whose estimate fits what is left of the cap. The first chunk is priced at the high estimate. Later chunks are priced at what the reference has actually charged so far, times a safety factor of 2, never above the high estimate. The cap holds unless later rows cost more than twice as much, relative to their size, as the rows already asked. Rows left over are unresolved ("over the cost cap").
- **Asking the rest later:** pass the first audit back. Its answers are reused, and only its unresolved rows are asked:

  ```python
  audit = pipeline.audit(result, model="openai/gpt-6-luna", max_cost=1.0, previous=audit)
  ```

  An audit of another run or another delivered frame, by another model or with another prompt, is refused.
- **Refused:** the run's own model as the reference ("a model cannot audit itself"), a run with no contract, and `max_cost` of 0 or less.
- **The estimate errs high,** sometimes far above the real cost: it prices every token of the run's answers (reasoning included) at both the reference's rates, for two calls. In the live run above it said at most $0.23; the audit spent $0.016.
- `model=` takes the same options as `with_llm()`. `client=` takes a client instead. Ondine never picks the reference for you.

## Current limits

| Limit | What happens today |
|---|---|
| No stream certification | Certify the result of `execute()`. A streaming chunk is refused. |
| Two labellers at most | There is no adjudication round. A row the two split on is unresolved. |
| No free-text labels | A sheet holds the certifiable fields only. |
| No free text | Free-text fields are listed as not certified. |

## Related

- [Output Contracts](output-contracts.md)

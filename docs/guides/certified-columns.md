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

## A run with no contract

Say how the outputs are read:

```python
from ondine.harness.contract import OutputField

cert = pipeline.certify(
    result, gold="true_intent", fields=[OutputField("intent", allowed=LABELS)]
)
```

## Current limits

| Limit | What happens today |
|---|---|
| No stream certification | Certify the result of `execute()`. A streaming chunk is refused. |
| Labels only | Audits against a stronger model, and review sheets for people, come later. |
| No free text | Free-text fields are listed as not certified. |

## Related

- [Output Contracts](output-contracts.md)

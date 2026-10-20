# Golden sets

The rows the harness benchmark is measured on. Three tasks, one per family
the harness must lift. No rows are committed; `build.py` downloads the
sources, checks them against pinned hashes, and carves the splits with a
fixed seed. `manifest.json` records the sources, the seed, and the hash of
every split file, so a rebuild can be checked byte for byte.

```bash
uv run python benchmarks/golden/build.py
```

## Tasks

| Task | Source | Row shape | Gold | Dev | Holdout |
|---|---|---|---|---|---|
| `intent` | [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) test split | one customer-service query | one of 77 intents | 231 (3 per intent) | 308 (4 per intent) |
| `attributes` | [WDC-PAVE](https://webdatacommons.org/structureddata/wdc-pave/), extraction variant | product title + description, 5 categories | 8–20 attributes per category, each stated in the offer text; `n/a` when absent | 199 | 300 |
| `faithfulness` | [SummEdits](https://github.com/salesforce/factualNLG), 8 business domains | a document + a candidate summary | `consistent` / `inconsistent` | 200 (25 per domain, labels balanced) | 304 (38 per domain) |

Why these three:

- **intent** is a plain classification with enough classes (77) that cheap
  models visibly confuse near neighbours (`card_arrival` vs
  `card_delivery_estimate`). 2023 figures: GPT-3.5 75% vs GPT-4 80% micro-F1.
- **attributes** is structured extraction where the easy fields come free
  and the hard ones (dimensions, product type, pack quantity) need checking.
  The benchmark paper reports GPT-3.5 F1 88.0 vs GPT-4 90.5 on the
  normalized variant. We use the *extraction* variant: 99.8% of its gold
  values appear in the offer text, against 41% for the normalized variant,
  whose gold relies on unit conversions and a fixed taxonomy the prompt
  would have to spell out. Normalization is left for v2.
- **faithfulness** is a judgment where the cheap-vs-big gap is large and
  documented (balanced accuracy: 7B open models ≈ 50–58, GPT-3.5 71, GPT-4
  82, humans 91) and the yes/no answer suits agreement-based checks.

## Label quality, honestly

None of the three publishes inter-annotator agreement in a form we could
read, except SummEdits (0.92 reported). What we know:

- **BANKING77**: labelling process undocumented. One 2022 paper estimated
  about 14% of *training* rows as possibly mislabeled by an automated method;
  the test split, which we use, was not audited.
- **WDC-PAVE**: annotated with a custom tool and a second review round;
  no agreement figure; no noise audit.
- **SummEdits**: edits made by GPT-3.5, then judged by two professional
  annotators plus the authors; borderline rows dropped.

So the "200 hand-checked rows" per task are produced like this, not
assumed: label the dev split with two strong models from different
families, hand-adjudicate every row where both disagree with the gold, and
record the verdicts in `adjudications.jsonl` (row id, original label,
corrected label, reason). Report results on the original gold and on the
adjudicated gold. The holdout gets the same treatment only at release
time, by someone who has not seen it.

## Adjudication, first pass (2026-10-20)

Dev split labelled by Sol (`openai/gpt-6.1-sol`) and Sonnet 5.5
(`anthropic/claude-sonnet-5-5`), prompts in `label.py`, candidates found by
`adjudicate.py`. Total spend about $3.4, tracked by Ondine.

| Task | Sol wrong | Sonnet wrong | Both wrong (candidates) |
|---|---|---|---|
| `intent` (231 rows) | 16.9% | 18.2% | 35 rows (15.2%) |
| `faithfulness` (200 rows) | 14.0% | 16.0% | 25 rows (12.5%) |
| `attributes` (3,074 attribute values, lenient match) | 9.3% | 7.8% | 163 values (5.3%) |

What the candidates show, so far read by a model, not yet by a human:

- `intent`: real gold noise ("I need my PIN" is labelled `get_physical_card`;
  "friends would like to top-up my account" is labelled `topping_up_by_card`)
  next to genuine near-neighbour confusions (`compromised_card` vs
  `card_payment_not_recognised`).
- `attributes`: gold `n/a` where both models found a stated value (an
  under-annotated gold), and vocabulary differences (`Ladies` vs `Women`).
- Sonnet ignored "reply with only the label" on 9% of `intent` rows and 19%
  of `faithfulness` rows (reasoning first, label last); Sol never did. The
  scorer takes the last line. A harness that parses strictly would lose
  those rows.

Tiebreak (same day): Opus 5.5 relabelled the candidates (`label.py
--candidates`, $1.24; the attributes run hit its $0.80 cap and was
dropped). Opus sided with the gold on 13 of 35 intent rows and 2 of 25
faithfulness rows, leaving 22 + 23 rows where all three models disagree
with the gold. Those go to a human.

Verdicts go in `adjudications.jsonl`. Until a human has ruled on a
candidate, report both the original and the adjudicated gold.

## Splits and contamination

- `dev` is for building and tuning. Every dev row carries a `canary`
  field; if that string appears in a model's output or a public corpus,
  the dev set leaked.
- `holdout` is for release gates only. It is not committed and carries no
  canary (a canary in the holdout would itself mark the row as a test
  item). SummEdits rows are split by *document*, so no document appears in
  both splits.
- All three sources have been public since 2020–2024, so frontier models
  may have seen them. Gaps measured on the holdout are the ones to trust;
  a cheap-vs-big gap much smaller than the published one is a
  contamination signal, not good news.

## Licences

| Source | Licence | Redistribution of rows |
|---|---|---|
| BANKING77 | CC-BY-4.0 (PolyAI card; the mteb mirror says MIT) | allowed with attribution |
| SummEdits | Apache-2.0 (repository); podcast domain dropped because its transcripts may carry their own terms | allowed |
| WDC-PAVE | not stated; the site says "publicly available for download" | **not until confirmed** with the Mannheim group |

One rule for all three keeps it simple: rows stay out of git. Only the
build script, the manifest, and our own adjudications are committed.

## Scoring

- `intent`: exact match on the label.
- `attributes`: per attribute, a prediction is correct if it equals any
  accepted value (the source stores several accepted spellings) or `n/a`
  when the attribute is absent; report attribute-level F1 and row-level
  all-correct rate.
- `faithfulness`: balanced accuracy, because 37% of rows are consistent.

Cost per *successful* row is the headline cost metric: every call made for
a row, including retries, samples, and escalations, counts whether or not
the row ended up right.

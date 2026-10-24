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

## Adjudication, verdicts (2026-10-24)

The 45 rows were ruled on by Claude (Fable 5.1) in the maintainer's
session, reading each input, the dataset's label and the three model
answers, and for SummEdits the exact edit the dataset made to the seed
summary. **These are model verdicts, not human ones**, and the maintainer
delegated the call. They are in `adjudications.jsonl` with a reason and a
confidence per row.

| Task | Dataset right | Models right | Dropped (ambiguous) | Adjudicated dev size |
|---|---|---|---|---|
| intent | 0 | 9 | 13 | 218 of 231 |
| faithfulness | 1 | 20 (+1 non-candidate, below) | 2 | 198 of 200 |

What the rulings found:

- **intent**: the dataset labels purchases declined on a card as
  `declined_transfer`, a card fee as `transfer_fee_charged`, a missing
  card as `activate_my_card`. Thirteen messages have no single defensible
  intent ("Is a SWIFT transfer okay?", "What is the transfer policy?")
  and were dropped rather than guessed.
- **faithfulness**: 20 of 23 candidates had a wrong gold. Six "consistent"
  rows hide a real change (a 20% discount became 15%; "Binance" became
  "Binance's customers"; "by region" became "by country"). Fourteen
  "inconsistent" rows are synonym swaps only. Two seed summaries are
  themselves unfaithful: AutoNation's "over 100 stores by the end of 2025"
  (the transcript says over 50 by then) and Dolby's "Q3 revenue lower than
  Q2" (no Q2 revenue in the transcript; the Q3 EPS part is supported, so
  this verdict rests on "unsupported counts as inconsistent"). Every dev
  row carrying those claims is scored `inconsistent`, including the Dolby
  seed row `DLB_q2_2021_og`, which was not a candidate.
- **attributes**: not adjudicated. The disputes are mostly the gold saying
  `n/a` where the offer states a value, which needs a rule, not 163 rulings.

**Circularity, stated plainly.** Candidates were chosen because Sol, Sonnet
and Opus disagreed with the gold, and most verdicts went the models' way.
Scoring on the adjudicated gold therefore favours those three models, and
any model that errs like them, more than it favours the dataset. Report
both golds. The holdout is untouched by any of this; it is the only split
that can settle a claim.

Until a human has ruled on a candidate, report both the original and the
adjudicated gold.

## Baseline, dev split (2026-10-24)

Five models over the dev splits through `label.py`, scored by `score.py`.
Figures below were re-checked after fixing two verdicts (see git log).
Accuracy on the dataset's original gold and on the adjudicated gold; cost
is Ondine's own tracking for the run, and "per 1k" is cost per successful
row on the adjudicated gold, times 1,000. Opus intent was run twice
(0.892, then 0.883 on the original gold, both at temperature 0): run-to-run
noise is about a point even before sampling noise.

| Task, metric | Haiku 5.5 | GPT-6 Luna | Sonnet 5.5 | GPT-6.1 Sol | Opus 5.5 |
|---|---|---|---|---|---|
| intent, accuracy (original → adjudicated) | 0.810 → 0.894 | 0.788 → 0.876 | 0.818 → 0.908 | 0.831 → 0.917 | 0.883 → 0.972 |
| intent, $ per 1k successful rows | 0.15 | 0.09 | 2.33 | 1.31 | 5.42 |
| faithfulness, balanced acc. (original → adjudicated) | 0.820 → 0.934 | 0.781 → 0.869 | 0.840 → 0.944 | 0.860 → 0.975 | 0.824 → 0.939 |
| faithfulness, $ per 1k successful rows | 0.19 | 0.17 | 3.12 | 1.85 | 6.83 |
| attributes, lenient attribute acc. | 0.916 | 0.906 | 0.922 | 0.907 | 0.917 |
| attributes, strict attribute acc. | 0.824 | 0.804 | 0.828 | 0.799 | 0.810 |
| attributes, rows with every attribute right (of 199) | 61 | 57 | 69 | 58 | 63 |
| attributes, $ per 1k successful rows (all attributes right) | 1.19 | 0.72 | 7.86 | 9.06 | 36.34 |

What it says so far, on about 200 rows per task. A 95% interval on a
single rate is about ±4 points at 90% accuracy and ±5 to 6 points at 80%,
and balanced accuracy on about 100 rows per class is wider still, so
differences under 5 points are ties:

- **intent**: the gap the harness has to close is real, and it holds on
  both golds: Haiku 0.810 / Luna 0.788 against Opus 0.883 on the original
  gold, 0.894 / 0.876 against 0.972 on the adjudicated one, at 36x to 60x
  less cost per successful row. Opus helped pick the candidates, so its
  adjudicated score is the most inflated; the original-gold gap is the
  safer number. Sonnet and Sol sit between, closer to the cheap models.
- **faithfulness**: no gap to close. Haiku (0.934), Sonnet (0.944) and
  Opus (0.939) tie; Sol leads (0.975), so Opus is not the ceiling here.
  Luna lags (0.869). Haiku gets Opus-level results at 36x less cost per
  successful row with no harness at all.
- **attributes**: a tie across all five, Opus included, lenient (0.906 to 0.922) and
  strict (0.799 to 0.828). Only about a third of rows get every attribute
  right for any model. The task as scored does not separate cheap from
  big models; the gold's missing values (`n/a` where the offer states a
  value) may be capping everyone.
- Costs are from one run each at temperature 0 (Haiku at 1, as required).
  Haiku used 1.5x to 1.8x Luna's tokens for the same rows.

**What this means for the harness plan.** Only `intent` has a cheap-vs-big
gap worth closing. On `faithfulness` and `attributes` the cheap models
already match the most expensive one, so the honest result there is "use
the cheap model", and a harness can only add cost. The next experiment
runs on `intent`; the other two need a harder slice before they can show
any lift.

Thirteen intent rows and two faithfulness rows are dropped from the
adjudicated gold as ambiguous; the original-gold columns use every row.

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

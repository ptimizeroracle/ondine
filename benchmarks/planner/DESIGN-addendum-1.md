Status: pre-registered 2026-11-17, before any extra sample was drawn. Parent: `DESIGN.md` (sha256 `3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42`), PREREG `01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a`, verdict KILL. The parent stays locked.

# Addendum 1: does Haiku self-consistency find the rows the cheap pair gets wrong together?

## 0. Disclosure (existing labels only; no sample drawn)
Intent, 231 rows: Haiku wrong on 44; Haiku and Luna agree and are wrong on 33 (the blind spot); Opus right on 24 of the 44 and 17 of the 33, wrong on 7 of Haiku's 187 right rows. Adjudicated gold: 26 of the 33 remain, 17 still wrong; Haiku wrong on 23 of 218. Faithfulness: Sol right on 2 of the 22 blind-spot rows; 2 of the 22 stay wrong on adjudicated gold. Arithmetic, not a result: a perfect detector gives C4 ΔA ≈ +7.8; recall 0.5 of Haiku's errors at 10% false flags gives ΔA ≈ +2.4, D2 ≈ +0.5.

## 1. Data
- **k = 3** new Haiku samples per row, intent (231) and faithfulness (200). Expected $0.19; hard cap $0.30 over all `haiku-s*` cost files, $0.05 per run. Rejected: k = 2 (weaker, saves $0.06); k = 5 (over budget).
- Same prompt as `label.py` `prepare`, same `MODEL_PARAMS["haiku"]` (temperature 1.0, reasoning_effort "low"), concurrency 6, error policy skip, no response cache. Route: `openrouter/anthropic/claude-haiku-5-5`, explicit prices $0.10/M input, $0.50/M output (LiteLLM's `claude-haiku-5-5` entry, as in the original run).
- Files: `benchmarks/golden/data/labels/<task>/haiku-s<i>.jsonl` and `.cost.json` (route, model id, input and output tokens). `haiku.jsonl` is never written. Data is gitignored: results embed each file's sha256.
- Answer 0 is `haiku.jsonl`; answers 1–3 are s1–s3; compared as `norm(final_answer(·))`.
- Attributes skipped: no cheap-vs-big gap (Haiku 0.916, best 0.922), and JSON agreement would need new per-field rules.

## 2. Threat: the route change
Same weights expected, but OpenRouter may map "low" to another thinking budget, drop it, or use another host. Checks per sample file:
- **V1:** intent accuracy within 81.0 ± 5.0 points; faithfulness balanced accuracy within 0.820 ± 0.06 (≈ 2 SE).
- **V2:** output tokens per row within [0.5×, 2×] of the original's implied value, (cost − 1e-7·T)/4e-7/rows: ≈ 95 intent, ≈ 110 faithfulness.
- **V3:** mean agreement among new samples minus their mean agreement with the original, within ±5 points (exchangeability).
- **V4:** every dev id; ≤ 3% null or invalid; no two samples identical on every row.
A file failing only V4 may be replaced once by `haiku-s4` within the cap. If V1–V3 fail on intent, the verdict is INCONCLUSIVE; new data needs Addendum 2.

## 3. Policies (2 confirmatory)
- **C4 self-consistency:** ask Haiku 4 times. If all 4 answers are valid and equal, deliver the original; else ask s and deliver its answer.
- **C5:** route when C4 routes OR Haiku and Luna disagree.

s: intent opus, faithfulness sol. Unanimity, not majority: on unanimous rows the original equals the majority, so no voting gain leaks into ΔA, and every split row is routed and paid for. Rejected: "deliver the majority, route only without one": a pass could be the vote, not the trigger.

Cost: the parent's flat and length-weighted models. C4 pays 4·u_haiku + routed·u_s; C5 adds u_luna. Calls are priced at the original Haiku unit cost; measured costs are reported.

Exploratory (≤ 8 per task, never decisive): C4 on j = 1, 2 samples; C4 and C5 with the other strong models; "route if the modal answer has < 3 of 4 votes, else deliver it"; Haiku 4-vote as a point at 4·u_haiku.

## 4. Comparator, statistics, verdict
Parent unchanged: the 5-model hull from the original runs, ΔA and D2, S = 1000 splits at n_L = 100, seed 20261117 (same test rows), cluster bootstrap B = 1000, LIFT and CUT under both cost models. Bonferroni over 2: LB = 0.025 quantile. Integrity: the run reproduces the parent's intent C1 accuracy and ΔA to 1e-9.

Intent decides, on original gold:
1. **KILL STANDS** if neither C4 nor C5 passes.
2. **BUILD_SC_TRIGGER** if C4 passes, unless C5 passes and beats C4 by ΔA ≥ 2 points with paired LB > 0.
3. **BUILD_SC_PLUS_PAIR_TRIGGER** if only C5 passes, or C5 clears that margin.
4. Scope as parent rule 4.

A pass yields an escalation trigger for Ondine's existing escalation step, not a Planner: neither policy chooses among models. The Planner KILL stands in every outcome. Adjudicated gold: reported; a sign flip is flagged, not a kill.

## 5. Mechanism diagnostic (all intent rows, every outcome)
Flag = C4's trigger. R33 = flagged share of the 33 blind-spot rows; R44 = of Haiku's 44 wrong rows; FF = of its 187 right rows; Wilson 95% intervals.
- **Useful:** R33 ≥ 0.40 (≥ 14) AND FF ≤ 0.10 (≤ 18). Basis: with C1's 11 caught rows, recall ≈ 0.55, near §0's break-even.
- **Not useful:** R33 ≤ 0.20 (≤ 6): the shared mistakes are stable.
- Otherwise **weak**. A pass with "not useful" is flagged. Faithfulness: descriptive.

## 6. Label-noise control
On adjudicated gold: R and FF over Haiku's 23 wrong and 195 right rows, and flag rates of three groups of the 33: A = 17 still wrong, B = 9 now right, C = 7 without a defensible label. flag(A) ≥ 0.40: it finds real errors. flag(B ∪ C) − flag(A) ≥ 20 points: it tracks ambiguity, so escalation is the wrong use. Groups this small describe; they decide nothing.

## 7. Risks
| Risk | Likelihood | Cheapest check |
|---|---|---|
| Thinking budget changed or dropped | medium | 3-row smoke: V2 |
| Wrong slug or host | low–medium | smoke response `model`; V1, V3 |
| Cache fakes unanimity | low | `--sample` refuses a cache; V4 |
| Samples near-unanimous | unknown: the question | `--check` reports flag rates |
| Low power (131 test rows) | high | accepted; CUT path |
| Gold noise | high | §6 |

## 8. Not doing
No attributes, logprobs, verifier model, new strong-model calls, k change, parent edit, held-out rows, or product code before the verdict.

Order: commit this file, `addendum1.py`, `label.py` options, tests → smoke → collect → `addendum1.py --check` → one full run.

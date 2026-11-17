Status: pre-registered 2026-11-17, before any policy number was computed.
Author: Opus planner (high effort); accepted as written by the lead.

# Planner replay: experiment design and pre-registered rules

## 0. Grounding (verified by the planner)
- All 5 models answer every dev row: 231 intent, 200 faithfulness, 199 attributes, no null answers. The adjudicated gold has 218 intent and 198 faithfulness rows. Attributes has no adjudicated file; `score.py:105` falls back to the original gold.
- Reuse `score.score` (`benchmarks/golden/score.py:63`), `final_answer`/`norm`/`attribute_errors`/`parse_json_object` (`adjudicate.py`) and `strict_attribute_errors` (`score.py:50`), imported with the `sys.path` pattern of `benchmarks/ruler/simulate.py`. The script lives in `benchmarks/`, adds no product code.
- `data/intent/labels.json` holds the 77 intent names, used for the "invalid value" signal.
- Not verified: the Opus `cost.json` files have no `"source"` key (likely computed by `label.py:152`); Opus intent was run twice and it is unclear which run its cost file records.
- Disclosure: one result was seen before this pre-registration: "Haiku and Luna agree but are wrong" on 17 of 202 intent rows (certified-columns design §1.2).

## 1. Comparator
**The single-model hull:** the upper concave envelope of the 5 (cost per row, metric) points, on the same evaluation rows. A random mix of two models reaches any point on it, so it is the cheapest strategy that needs no Planner.

Rejected: "match the best single model at ≤50% of its cost". A single model already passes it (attributes: Haiku 0.916 at $0.07 vs Sonnet 0.922 at $0.54; faithfulness: Haiku 0.820 within noise of Sol 0.860), so it cannot tell routing from "use the cheap model".

Two paired statistics on the same rows:
- **ΔA = a_p − F(c_p)**: accuracy lift over the hull at equal cost.
- **D2 = a_p − F(2·c_p)**: whether the policy at cost c matches what the hull buys with twice the money (a cost cut ≥ 50%).

## 2. Policies
Signals: other models' answers, a value's validity (a contract check), and labels on the training subset only. Every policy routes a row with an invalid value to the strong model (intent: final line not one of the 77 names; faithfulness: not consistent/inconsistent; attributes: JSON does not parse).

`s` = best single model on original gold in `baseline-report.json`: intent opus, faithfulness sol, attributes sonnet.

**Confirmatory (only these decide the verdict; 3 per task):**
- **C1 cascade:** ask Haiku and Luna; if `norm(final_answer)` agrees, deliver Haiku's answer, else ask `s` and deliver its answer. Learns nothing; evaluated on the same test rows as C2/C3.
- **C2 value router (Ruler-fed):** ask Haiku on all rows. On n_L = 100 labelled rows, shrunk precision of each delivered value v: `p̂(v) = (k_v + m·p̄) / (n_v + m)`, k_v right, n_v seen, p̄ Haiku's overall precision on the labelled rows, m = 4. Unseen value: p̂ = p̄. Route to `s` when p̂(v) < τ = 0.8.
- **C3 combined:** route when Haiku and Luna disagree OR p̂(Haiku's value) < 0.8.

At p̄ ≈ 0.81 on intent, one right observation gives 0.85 (stays), one wrong 0.65 (routed), one right + one wrong 0.71 (routed): in practice C2 routes "values seen wrong at least once". Shrinkage replaces minimum counts; no hand-made pooling of intents.

**Exploratory grid** (frontier plot only, cannot change the verdict, ≤ 75 configs per task): C1 with s ∈ {sonnet, sol, opus}; C2 with c ∈ {haiku, luna} × s ∈ {sonnet, sol, opus} × n_L ∈ {50, 100} × τ ∈ {0.6, 0.7, 0.8, 0.9}; C3 with s × n_L × τ; m = 4 everywhere.

**Unreachable references (labelled as such):** O1 per row the cheapest right model (pays it only); O2 any-model-right ceiling; O3 C2 with precision fitted on all rows including test rows (O3 − C2 measures learning optimism).

**Cost:** flat per-row u_m = cost_m / rows_m; a policy pays u_m for every model it asks on the row. Report $ per 1k rows, % of s, $ per successful row. Sensitivity: length-weighted u_m,i = cost_m · len_i / Σlen (len = input characters). Flat averaging understates routed rows' cost (likely longer; favours the policy) and ignores prompt caching on a second call (disfavours it). A task passes only if it passes under both cost models (point estimates for the length-weighted one). Human labels are not priced in $; n_L is reported as their price.

## 3. Out-of-sample procedure
- **Splits:** S = 1000 seeded splits (seed 20261117). Draw n_L rows uniformly without replacement (the Ruler's plain sample), fit on them, evaluate on the rest (131 test rows for intent at n_L = 100). Hull F on the same test rows. Report mean and 5–95% range over splits.
- **Uncertainty:** cluster bootstrap over rows, B = 1000: multinomial weights over unique ids; in each draw 20 splits by unique id (copies of a row never on both sides); weighted metrics; mean ΔA and mean D2. Lower bound = the α = 0.05/3 quantile (Bonferroni over 3 confirmatory policies per task, one-sided).

## 4. Metrics (consistent with score.py)
- intent: accuracy, `norm(final_answer) == norm(label)`.
- faithfulness: balanced accuracy (mean recall over gold classes on the test rows); the hull mix is linear in expectation, so F stays valid.
- attributes: primary lenient attribute-level accuracy, micro over fields (`attribute_errors`); secondary strict, and rows with every attribute right.
- Verdict on original gold. Adjudicated gold reported; it favours sol/sonnet/opus, so is optimistic for routing to them. If ΔA changes sign on adjudicated gold, flag it; not a kill.

## 5. Attributes
Primary per-field routing. Field cell = (category, attribute, delivered value is "n/a"). C1 compares the two cheap models field by field with `norm` equality; C2/C3 route fields whose cell has p̂ < 0.8; a routed field takes s's value. Cost conservative: a row pays the full u_s if any field is routed. Whole-row routing is exploratory only (with 8–20 fields it collapses to "use s").

## 6. Pre-registered pass and kill rules
A confirmatory policy passes on a task when either holds:
- **LIFT:** mean ΔA ≥ +3.0 points and its lower bound > 0.
- **CUT:** mean D2 ≥ 0, lower bound of D2 ≥ −2.0 points, and mean ΔA ≥ +1.0.

Thresholds: a paired difference with 10% discordant rows at n = 231 has SE ≈ 2.1 points, so LIFT detects ≈ 4 points or more (accepted: a smaller gain could not be shown on user data either, the Ruler being ±4 at 200 labels). The 2× cost margin covers the 20–30% error of cost averaging.

**Verdict:**
1. **KILL the Planner** if no confirmatory policy passes on intent (the only task with a ≥ 5-point gap).
2. **Build the Ruler-fed Planner** if C2 or C3 passes on intent, AND either C1 fails, OR ΔA(C2 or C3) − ΔA(C1) ≥ 2 points with a paired lower bound > 0 at ≤ C1's cost.
3. **Build only a disagreement cascade** if only C1 passes (an escalation trigger, no labels).
4. **General vs narrow:** intent plus faithfulness or attributes → a general floor; intent alone → classification-only Planner, flagged single-task, re-tested on holdout at release.

"2 of 3 tasks" was rejected: README shows no cheap-vs-big gap on 2 tasks, so it would pre-register a kill by construction.

## 7. Script and outputs
`benchmarks/planner/replay.py`, a `PREREG` dict with every constant above (its hash goes into the results), numpy only.

Functions: `load_task(task, gold) -> Task` (ids, gold, final values per model, `correct[m]` rows or rows × fields, `valid[m]`, `u_flat`, `u_len`); `check_integrity()` (id coverage; `score.score` reproduces `baseline-report.json` to 1e-9 for 15 model × task pairs and both golds); `shrunk_precision(codes_L, right_L, m)`; policies `(task, train_idx, test_idx, params) -> (right_test, cost_test, routed_test)`: `single`, `cascade_agree`, `value_router`, `agree_or_value`, `oracle_cheapest_right`, `oracle_in_sample`; `metric(task, right, gold, weights)`; `hull(points) -> F`; `run_splits`, `cluster_bootstrap`, `verdict(results, PREREG)`; `write_markdown`, `write_svg` (hand-written SVG).

Holdout guard: the loader accepts only `dev.jsonl`, `dev-adjudicated.jsonl`, `labels/*`; a test asserts "holdout" appears only in the guard.

Outputs in `benchmarks/planner/results/`: `replay.md` (integrity; per policy and task: cost $/1k, % of s, accuracy with split range, ΔA and D2 with lower bounds, routed share, adjudicated accuracy, length-weighted ΔA/D2; verdict; diagnostics: shared-mistake rate, share of routed rows where Haiku was wrong, oracle ceilings, O3 − C2), `replay.json`, `frontier-<task>.svg`, `benchmarks/planner/README.md`.

Runtime < 60 s full, < 10 s with `--quick` (B = 100). Exit 2 on integrity failure; else exit 0 and print the verdict.

Order: the script with `PREREG` is reviewed before the full run; results cite the PREREG hash.

## 8. Risks
| Risk | Likelihood | Cheapest early check |
|---|---|---|
| Shared cheap mistakes cap C1 at ≈ 92.6% on intent | certain, size known | diagnostics table |
| Thin per-value data: C2 ≈ "route values seen wrong once", noisy at n_L = 50 | high | O3 − C2; n_L 50 vs 100 |
| Low power: a real 3-point gain called a kill | medium | SE per config; accepted |
| Cost averaging bias | medium | must pass under length weighting |
| Gold noise, adjudication circularity | high | original gold decides; both reported |
| Forking paths | controlled | 9 fixed configs, Bonferroni, capped grid, review before run |
| Dev reuse (s picked on dev), Haiku at temperature 1, Opus ±1 point between runs | medium | holdout replay at release only |

The hull is computed knowing the test rows, so the comparator is stronger than reality: conservative.

## 9. Not doing
No learned text classifier; no per-row token reconstruction; no holdout reading, no API calls; no τ tuning on test rows; no vote-of-three; no product code until the verdict.

Status: pre-registered 2026-11-17, before any ranker number was computed. Parents, both locked: `benchmarks/planner/DESIGN.md` (sha256 `3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42`, PREREG `01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a`, KILL) and `DESIGN-addendum-1.md` (sha256 `420032cc81b4d9af9ef17d4f77a820b3c78af0aa446053fbcea853b51abf4f33`, PREREG `28d2f6a99ae5e8959a7ea60c7e01fc59317727e204d02e84316bce8d44e73cef`, KILL STANDS). This study is not blind: §0 lists what was seen.

# Review-next: which unsampled rows are wrong?

The Ruler labels n_L = 100 sampled rows and reports an error rate. Review-next ranks the other delivered rows, so a person who reviews the top k finds more errors per row than random review. Question: does a ranking signal beat the signal the Ruler gives free (per-value error rates) by enough to ship?

## 0. Disclosure (existing labels; seen before this design)
- Base error, original gold: Haiku wrong on 44/231 intent rows (19.0%), 36/200 faithfulness rows (18.0%). Adjudicated: 23/218 (10.6%), 13/198 (6.6%). Test rows per split (n − 100): 131 and 100; adjudicated 118 and 98. No invalid delivered value.
- Dissent D = how many of the 3 extra Haiku samples differ from the delivered answer. All rows, not crossed with correctness: intent D = 0/1/2/3 on 211/11/5/4 rows; faithfulness 181/5/7/7; adjudicated rows 201/10/4/3 and 179/5/7/7.
- Pair disagreement P (Haiku ≠ Luna): 22 intent rows, 35 faithfulness rows. D ≥ 1 and P together: 11 and 12. Haiku delivers 73 distinct intents; on faithfulness it delivers 2 values (96/104).
- Seen against correctness (Addendum 1 and replay.md):
  - D ≥ 1 flags 11 of 44 wrong and 9 of 187 right intent rows (precision 55%), and 11 of 36 and 8 of 164 faithfulness rows (58%). Adjudicated: 9/23 and 8/195; 10/13 and 9/185.
  - P: 11 of 22 rows wrong on intent (50%), 14 of 35 on faithfulness (40%).
  - Derived: rows with both D ≥ 1 and P: 7 of 11 wrong (intent), 9 of 12 (faithfulness). Union: 15 of 31 and 16 of 42.
  - The per-value rule (C2, τ 0.8, out of sample) flagged 31.7% of intent test rows at 30.5% wrong, and 31.4% of faithfulness at 14.0% wrong. O3 − C2 = 5.7 points on intent: per-value rates transfer poorly.
- Arithmetic, not a result. At k = 10%, intent: 13 of 131 test rows hold ≈ 11.3 rows with D ≥ 1. R2's precision ≈ (0.55·11.3 + 0.2·1.7)/13 ≈ 0.50, lift ≈ 2.6.
- Faithfulness: 10 of 100 test rows hold ≈ 9.5 rows with D ≥ 1. Precision ≈ 0.56, lift ≈ 3.1. So the lift floor of §5 is implied, not discovered.
- R1 orders only 2 faithfulness values, so its precision ≈ 0.14–0.22, and the mean gain there is near-implied too.
- Open quantities:
  - R1's top-10% precision on intent: unknown. C2 suggests 0.30–0.45.
  - The bounds. Bootstrap SE of R2's precision ≈ √(0.55·0.45/20) ≈ 11 points. So intent clears the gain bound only if R1@10% ≲ 0.25.

## 1. Data and units
- Delivered model: Haiku (`labels/<task>/haiku.jsonl`). D comes from `haiku-s1..s3` (Addendum 1 files; sha256 in PREREG). P comes from `luna.jsonl`. Tasks: intent and faithfulness.
- Attributes are skipped: no extra samples exist, and per-field dissent needs new comparison rules.
- Splits: the parent's, row for row: `replay._SplitCache(task, replay.PREREG).get(100)` (seed 20261117, purpose 0), S = 1000. A ranker reads labels of the 100 labelled rows only and ranks the test rows.
- Gold: the original gold decides. The adjudicated gold is the label-noise control (the same recipe on its own rows; gold index 1).

## 2. Rankers (keys compared lexicographically and exactly; a higher key is reviewed first)
- R0 random: one constant key. This is the null; its precision equals the base rate.
- R1 per-value, free (the comparator): key (1 − p̂(v)). p̂ comes from `replay._fit_precision`, m = 4, on the labelled rows. An unseen value gets p̄.
- R2 self-consistency: key (D, 1 − p̂). Cost: 3 Haiku calls per unsampled row.
- R4 dissent plus pair: key (D + P, 1 − p̂). Cost: 3 Haiku calls + 1 Luna call.
- Confirmatory: R2 and R4. R3 (P alone) is exploratory: it is the parent's C1 signal, already disclosed above.
- R4 is additive and has no parameter: Luna's disagreement counts as one more dissenting answer.
  - Rejected: a logistic fit on 100 labels. About 19 errors and 9 flagged rows would fit noise and need a recipe of their own.
  - Lexicographic (D, P) is exploratory (E3).
- Unit costs, measured ($ per 1k rows):
  - Haiku: 0.129 (intent), 0.174 (faithfulness).
  - Luna: 0.075 and 0.144.
  - R2: 0.39 and 0.52. R4: 0.46 and 0.67.
- Break-even:
  - Setup: h = review cost per row, review budget k = 10% of rows. A ranker that costs c per row on every row beats R1 when 10c < h·(P_R/P_R1 − 1).
  - At h = $0.25 (30 s at $30/h, assumed), R2 needs P_R/P_R1 > 1.02. At h = $0.01 it needs > 1.52.
  - Model cost does not bind. The gain does.
- Exploratory (8 per task, never decisive):
  - E1: R3 = (P, 1 − p̂).
  - E2: R2 without the tie-break, key (D).
  - E3: R4-lex = (D, P, 1 − p̂).
  - E4–E6: R2 on samples {s1,s2}, {s1,s3}, {s2,s3}.
  - E7: R2 on {s1}.
  - E8: R1 fitted on all rows (unreachable).

## 3. Metrics
- k = ⌊f·W + ½⌋, computed in exact fractions. W = the test weight of the split; f ∈ {0.05, 0.10, 0.20}. Plain splits: intent k = 7/13/26, faithfulness 5/10/20.
- Primary f = 0.10, because of reviewer effort: in a 1,000-row run that is about 90 rows, one more session the size of the Ruler's own sample.
  - f = 0.05 only orders rows inside the flagged set; f = 0.20 is mostly R1's fill.
  - Disclosed: 0.10 sits near the 9–10% flag rate.
- Ties: expected hits under uniformly random tie-breaking, computed exactly.
  - Sort the rows into groups of equal key. Each group fully above the cut counts all its errors.
  - The boundary group (weight w_g, errors e_g) adds (k − W_above)·e_g/w_g.
  - A bootstrap weight w counts as w copies of the row.
- Per split: P = hits/k (errors per 100 reviewed = 100·P), b = errors/W, recall = hits/errors.
- Over the S splits, or over the 20 splits of a bootstrap draw:
  - Mean precision P̄ and mean base rate b̄.
  - Lift L = P̄/b̄ (ratio of means).
  - Gain G = 100·(P̄_R − P̄_R1), in points.
  - Mean recall, and rows reviewed per error = 1/P̄.

## 4. Statistics
- Cluster bootstrap as in the parent: `replay.bootstrap_batches`, B = 1000, 20 splits per draw, on the parent's bootstrap stream (`split_rng(…, "original", 1, 100)`).
- Every ranker sees the same weights, so G is paired.
- Family: 2 confirmatory rankers × 2 tasks = 4 one-sided tests. LB = the 0.0125 quantile of the draws (`np.quantile`, linear).

## 5. Verdict (original gold, f = 0.10)
For each ranker R and task t:
- **PASS:** mean L ≥ 2.0, LB(L) > 1.0, mean G ≥ 10 points and LB(G) > 0.
- **KILL:** mean G < 5 points, or mean L < 1.5.
- **INCONCLUSIVE:** otherwise, or V1–V3 failed on t (§7).

Why these thresholds:
- L ≥ 2: the ranked list at least doubles what random review finds.
- G ≥ 10: ten more errors per 100 rows reviewed than R1. This is the smallest gain a reviewer notices in one session; at R1 ≈ 0.33 it cuts rows per error from 3 to 2.

Can the thresholds fail?
- L: not on §0's numbers, since R2's test precision would have to fall below 0.38. It is a floor, not evidence.
- Intent, mean G: fails if R1@10% ≥ 0.40. This is plausible if Haiku's errors cluster on a few intents.
- Intent, LB(G): fails unless R1@10% ≲ 0.25.
- Faithfulness: only LB(G) can fail, and only if its SE is above about 17 points.

Overall verdict:
1. **BUILD_GENERAL(R)** if R passes on both tasks.
2. **BUILD_SCOPED(t, R)** if R passes on one task only. The report prints the other task's status: many-class columns versus binary judgment columns.
3. **KILL** if all four (R, t) pairs are KILL.
4. **INCONCLUSIVE** otherwise.

- If R2 and R4 both pass on a task, choose R2 unless R4 − R2 ≥ 5 points with a paired LB > 0. R2 needs one model only.
- "Intent decides" is rejected. That precedent rested on intent being the only task with a cheap-versus-strong gap, which routing needs. Ranking needs only errors to find, and both tasks have 18–19%.
- Flags on a passing (R, t). They are reported and do not kill:
  - SIGN_FLIP: adjudicated mean G ≤ 0.
  - K_FRAGILE: mean G ≤ 0 at f = 0.05 or f = 0.20.
  - AMBIGUITY: see §6.
  - SUBSET: one of E4–E6 has mean G < 10.

## 6. Mechanism and label-noise control (all rows; flag = D ≥ 1)
- Split Haiku's original-gold wrong rows by what adjudication did: A = still wrong, F = now right, X = dropped. Report flag rates with Wilson 95% intervals.
- Also report the flag rate on N = rows right on the original gold but wrong on the adjudicated gold.
- AMBIGUITY if flag(F ∪ X) − flag(A) ≥ 0.20: the flag finds disputed labels, not errors.
- Adjudicated L and G are reported for every ranker.
- The original gold is conservative for faithfulness: at least 23 of its 36 wrong rows are not still-wrong after adjudication.

## 7. Validity of the extra samples
- The samples came by another route (OpenRouter), so re-run `addendum1.sample_checks`: V1–V4, the parent's C1 reproduction, and Addendum 1's disclosure.
- Every data file's sha256 must equal the PREREG value.
- A hash mismatch, a V4 failure or a hard failure: exit 2, and no study runs.
- A V1–V3 failure on a task: R2 and R4 are INCONCLUSIVE on that task. New data needs a new design.

## 8. Risks
| Risk | Likelihood | Cheapest check |
|---|---|---|
| Faithfulness pass is near-implied: R1 cannot rank 2 values | high | R1's L reported; the holdout gate needs adjudicated L ≥ 2 |
| Intent underpowered (20 flagged rows): a real gain reads INCONCLUSIVE | high | report SE(G) to size a future intent sample |
| Tie handling or row order inflates precision | low | unit tests; R0's L = 1 exactly in `--check` |
| One sample file carries the result | low | SUBSET flag; V3 |
| Gold noise | high | §6; adjudicated reported |
| Dev reuse: D ≥ 1 was found on these rows | certain | the holdout confirms; nothing ships on dev alone |
| Bootstrap tail at q = 0.0125 (≈ 12 draws) | medium | accepted; report B with the bound |

## 9. Not doing
No attributes, no new samples, no API call, no held-out rows, no fitted combiner, no change of k or of a threshold, no product code. No routing claim: the Planner KILL stands.

Order: commit this file, `review_next.py` and its tests → review → `--check` → one full run from the clean commit → README result.

Status: pre-registered 2026-11-17, before any confirm row was built or any model was called on one. Parent: `benchmarks/review_next/DESIGN.md` (sha256 `57e8962a0173c860dc464e2fb4cc2a456c3ece2f711c3209412f17265cfb7bff`), PREREG_REVIEW `85737be9a20a34cc253f665ab37fa5bd10dae2ce674d3b6e382875d9426afecc`, script `94836cab636f6d40a3880ec90c127d502da4f6ec7b55cf3841fbed0879ad7e35`, verdict BUILD_SCOPED(faithfulness, R2). The parent stays locked. The release split stays unread.

# Review-next confirmation: does R2 hold on fresh rows?

## 0. Disclosure (seen before this design)
- Parent, k = 10%, original gold:
  - Faithfulness: R2 54.9 errors per 100 reviewed, L 3.06 (LB 1.72), G +38.8 (LB +11.5), SE(G) 11.7; R1 L 0.90.
  - Intent: R2 52.2, L 2.73 (LB 1.50), G +17.5 (LB −7.4), SE(G) 10.9; R1 L 1.82.
  - R4 − R2: +1.6 / +1.7 points, paired LB < 0.
  - Adjudicated faithfulness: R2 L 7.6, recall 0.78.
  - Flags: intent SUBSET; faithfulness none.
- Parent mechanism: on faithfulness, 10 of the 13 still-wrong rows are flagged, against 1 of the 23 rows that adjudication changed or dropped. On intent: 9/23 against 2/21. The flag tracks real errors, so original-gold noise lowers measured precision.
- Fresh pools, counts and gold labels only. Faithfulness: 66 dev-pool documents, 1,756 rows unused by dev (579 consistent, 1,177 inconsistent); per-domain counts are in the PREREG. Intent: 32–33 unused rows per class.
- No model has seen a confirm row.
- Selection effect: faithfulness was chosen because it passed. Expect regression toward a smaller gain.

## 1. Data
- **Faithfulness confirm, 320 rows.** For each of the 8 domains: 20 consistent and 20 inconsistent rows, from dev-pool documents, never a dev row.
  - Draw order: documents shuffled; for each label, round-robin over the documents.
  - Cap: 5 rows per label per document, so at most 10 rows per document.
  - Random generator: `Random("20261020:faithfulness:confirm")`.
  - scitldr has exactly 20 consistent rows under the cap, so 20/20 per domain is the most this recipe allows.
  - Balance matches dev (12/13 per domain). Rejected: more rows with unequal label shares, because the base rate would no longer be comparable.
- **Intent confirm, 770 rows.** 10 per class from rows unused by both earlier splits, with `Random("20261020:intent:confirm")`. The rows are independent queries.
- Both draws use their own random generators, after the existing draws. `dev.jsonl` and the second split stay byte-identical (§8 of the builder spec). Rows carry the dev canary.
- Gold comes from the sources. No adjudication: the cheapest automatic rule (two strong models agree against the gold) needs Sol plus Opus on every row, about $2.6 for faithfulness, which is over the cap. Per §0, original gold is conservative for R2 on both tasks.
- Partial independence on faithfulness: rows are new, documents are not. Nothing is fitted: R2 has no parameters, and k and the thresholds are fixed. So the claim is "new summaries of documents seen in dev". New documents are the release gate's job.

## 2. Answers
- Haiku ×4 per row, all on one route: `openrouter/anthropic/claude-haiku-5-5`, temperature 1.0, reasoning "low", the prompt of `label.py` `prepare`, no cache.
- Files: `labels/<task>/confirm/haiku-s1..s4`. Column j is file s(j+1).
- The delivered answer is s1, re-drawn on the same route. No direct-route answer exists.

## 3. Arms
- **Confirmatory, one per task: R2**, key (D, 1 − p̂). D counts how many of s2–s4 differ from s1. p̂ is R1's shrunk per-value precision (m = 4) on the n_L = 100 labelled rows.
- Comparators: R0 (random) and R1 (free).
- R4 is dropped: it did not beat R2, and it needs a Luna run.
- Exploratory (6, never decisive):
  - E2: D alone.
  - E4–E6: D on two of the three samples.
  - E7: D on one sample.
  - E8: R1 fitted on all rows.
- **Escalation arm (D ≥ 1 → Sol): not run.**
  - Its equal-cost line needs Sol on every row: 320 × $1.79/1k = $0.57.
  - On dev, the Haiku–Sol chord equals the five-model hull between Haiku's cost and Sol's: Haiku is a hull vertex; Luna is cheaper; Sonnet and Opus are dearer and worse. On new rows that is unverified.
  - Power at 220 test rows: SE(ΔA) ≈ 2–2.5 points against an expected ΔA of +1.5 to +2.7. P(LIFT) ≈ 10–20%. P(CUT) ≈ 40%.
  - Even a pass is a routing claim in a family that is killed twice, and it would cost the intent arm.

## 4. Metric and statistics (the parent's, by import)
- Splits: S = 1000, n_L = 100, seed 20261118 (purpose 0 for splits, 1 for the bootstrap).
- Test rows: 670 intent, 220 faithfulness. k at 5/10/20%: 34/67/134 and 11/22/44.
- Primary k = 10%. P, L = P̄/b̄, G = 100·(P̄_R2 − P̄_R1). Ties are scored exactly, as in the parent.
- Bootstrap, B = 1000, 20 splits per draw:
  - Intent: rows are clusters (`replay.bootstrap_batches`).
  - Faithfulness: documents are clusters. Weights are multinomial over the documents used, and a row takes its document's weight. The 100 labelled rows are drawn over rows, as the Ruler draws.
- Family: 2 tasks × 1 ranker. LB = the 0.025 quantile.

## 5. Verdict
Per task, the parent's rules unchanged:
- **PASS:** mean L ≥ 2.0, LB(L) > 1.0, mean G ≥ 10, LB(G) > 0.
- **KILL:** mean G < 5, or mean L < 1.5.
- **INCONCLUSIVE:** otherwise, or V1–V3 failed on the task.

Can each condition fail?
- Faithfulness: R1 ≈ random on balanced labels, so G ≈ 100·(P − b). With b ≈ 0.18, L ≥ 2 needs P ≥ 0.36. Dev's flag precision is 11/19, Wilson [0.36, 0.77]. At its lower end, L ≈ 1.95 and the test fails. LB(G) > 0 needs mean G ≥ 1.96 × SE ≈ 18 points.
- Intent: dev G +17.5 ± 10.9. Expected SE now 5–6, so the test passes only if the true G ≳ 11. At a true G of 10 it is a coin flip.

Overall verdict:
1. **CONFIRMED_GENERAL**: PASS on both tasks.
2. **CONFIRMED(faithfulness)**: faithfulness PASS, intent not PASS.
3. **NOT_REPLICATED**: faithfulness KILL. The dev BUILD_SCOPED is withdrawn. An intent PASS is then reported as NEW(intent).
4. **INCONCLUSIVE**: otherwise.

Flags (reported, not decisive): K_FRAGILE (mean G ≤ 0 at 5% or 20%); SUBSET (any of E4–E6 below 10); DOMAIN (faithfulness G ≤ 0 in 3 or more of the 8 domains, descriptive).

What each outcome allows:
- A PASS allows a Ruler product note (review list v2: R1 always on, R2 an opt-in at about $0.5 per 1k rows) and a pre-registered release-gate check on the release split at release time. No product code before that check.
- NOT_REPLICATED: binary columns get no ranking; R1 alone stays a later floor.
- INCONCLUSIVE: no new spend this cycle.
- No outcome supports a routing claim.

## 6. Validity of the samples (`addendum1` checks, re-pointed)
- V1: each file's accuracy within 0.810 ± 0.06 (intent) or balanced accuracy within 0.820 ± 0.07 (faithfulness): about 2 SE of the dev-versus-confirm difference.
- V2: output tokens per row within 0.5–2× of 95.0 (intent) or 109.6 (faithfulness).
- V3: agreement of s1 with the others minus agreement among s2–s4, within ±5 points.
- V4: exactly the confirm ids; ≤ 3% null or invalid; route, model and rows as fixed; no two files identical on every row.
- One file failing only V4 may be replaced once by s5, within the cap.
- If V1–V3 fail on a task, that task is INCONCLUSIVE.

## 7. Risks
| Risk | Likelihood | Cheapest check |
|---|---|---|
| Faithfulness shares documents with dev; replication overstates generality | certain, size unknown | doc-cluster SE; DOMAIN flag; the release gate |
| Regression: dev's 58% flag precision was luck | medium | it is the test; Wilson lower end fails it |
| Intent true G about 10: underpowered again | medium | SE(G) reported to size any next step |
| Route or token drift breaks the budget | low | 3-row smoke; V2; per-run caps |
| A build change alters dev or the second split | low | byte-identity test against the manifest |
| Gold noise | high, conservative | §0 mechanism |

## 8. Not doing
No release-split read, no escalation arm, no Luna, Sol or Opus calls, no adjudication, no R4, no threshold change, no product code.

Order:
1. Commit this design, the build and label changes, `confirm.py` and its tests → review.
2. `build.py --confirm` (deterministic; the reviewer re-runs it) → commit the confirm.jsonl hashes.
3. 3-row smoke → 8 runs → commit the data lock → `confirm.py --check` → one full run from the clean commit.

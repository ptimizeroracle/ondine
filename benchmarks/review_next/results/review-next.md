# Review-next: results

Design: `benchmarks/review_next/DESIGN.md` (sha256 `57e8962a0173c860dc464e2fb4cc2a456c3ece2f711c3209412f17265cfb7bff`). PREREG_REVIEW sha256 `85737be9a20a34cc253f665ab37fa5bd10dae2ce674d3b6e382875d9426afecc`. Parents: replay DESIGN `3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42`, PREREG `01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a`; addendum 1 DESIGN `420032cc81b4d9af9ef17d4f77a820b3c78af0aa446053fbcea853b51abf4f33`, PREREG `28d2f6a99ae5e8959a7ea60c7e01fc59317727e204d02e84316bce8d44e73cef`.  
Script sha256 `94836cab636f6d40a3880ec90c127d502da4f6ec7b55cf3841fbed0879ad7e35`; git `a1b4367c1906171c4f617ca51c86f0efd696e2d7` (clean).  
Mode: full; S = 1000 splits, B = 1000 bootstrap draws, seed 20261117; run time 4.8s.

## Verdict

**BUILD_SCOPED(faithfulness, R2)**

Selection rule: R2 unless R4 beats R2 by >= 5 points with paired LB > 0 on every task in scope.

- FLAG R2/faithfulness: none

| Ranker | intent | faithfulness |
|---|---|---|
| R2 | INCONCLUSIVE | PASS |
| R4 | INCONCLUSIVE | PASS |

A pass is a ranking signal for the review step, not a router. The Planner KILL stands in every outcome.

## Checks

```
PREREG_REVIEW sha256 85737be9a20a34cc253f665ab37fa5bd10dae2ce674d3b6e382875d9426afecc
review-next design sha256 57e8962a0173c860dc464e2fb4cc2a456c3ece2f711c3209412f17265cfb7bff
parent designs and PREREG hashes (replay, addendum 1): OK
parent integrity (replay.check_integrity): OK
PREREG_ADDENDUM sha256 28d2f6a99ae5e8959a7ea60c7e01fc59317727e204d02e84316bce8d44e73cef
addendum design sha256 420032cc81b4d9af9ef17d4f77a820b3c78af0aa446053fbcea853b51abf4f33
parent DESIGN.md and PREREG hashes: OK
[intent] the design's disclosed counts reproduce: {'rows': 231, 'haiku_wrong': 44, 'blind': 33, 'strong_right_on_wrong': 24, 'strong_right_on_blind': 17, 'strong_wrong_on_right': 7, 'adjudicated_rows': 218, 'adjudicated_haiku_wrong': 23, 'groups': {'A': 17, 'B': 9, 'C': 7}}
[faithfulness] the design's disclosed counts reproduce: {'rows': 200, 'blind': 22, 'strong_right_on_blind': 2}
parent's intent C1 reproduced: accuracy 0.830709924, Delta A 0.768238106 (max |diff| 0.0e+00)
[intent] original Haiku: 95.0 output tokens per row implied
  haiku-s1: V1 0.805 in [0.760, 0.860] ok; V2 95.7 tokens/row, x1.01 of the original ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 cdc0314b67a0, measured cost $0.0298
  haiku-s2: V1 0.797 in [0.760, 0.860] ok; V2 93.4 tokens/row, x0.98 of the original ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 8e81ebefcaac, measured cost $0.0295
  haiku-s3: V1 0.810 in [0.760, 0.860] ok; V2 97.2 tokens/row, x1.02 of the original ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 dcc67b66ec00, measured cost $0.0300
  V3: agreement among new 0.954, with the original 0.952, difference +0.1 points ok
  flag rates on all 231 rows: C4 0.087 (20 rows), C5 0.134 (31 rows)
[faithfulness] original Haiku: 109.6 output tokens per row implied
  haiku-s1: V1 0.856 in [0.760, 0.880] ok; V2 127.2 tokens/row, x1.16 of the original ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 06b772cc801e, measured cost $0.0365
  haiku-s2: V1 0.856 in [0.760, 0.880] ok; V2 125.4 tokens/row, x1.14 of the original ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 7f0ce6a98642, measured cost $0.0364
  haiku-s3: V1 0.830 in [0.760, 0.880] ok; V2 117.6 tokens/row, x1.07 of the original ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 c76d0b86cf7d, measured cost $0.0356
  V3: agreement among new 0.960, with the original 0.933, difference +2.7 points ok
  flag rates on all 200 rows: C4 0.095 (19 rows), C5 0.210 (42 rows)
[intent] V1-V3: ok
[faithfulness] V1-V3: ok
data files: 19 sha256 values match
[intent] the design's disclosed counts reproduce
[faithfulness] the design's disclosed counts reproduce
[intent/original] test rows per split [131.0], k at 5%/10%/20% = 7/13/26; R0 lift 1 to within 2.2e-16 on 1000 splits
[intent/adjudicated] test rows per split [118.0], k at 5%/10%/20% = 6/12/24; R0 lift 1 to within 1.1e-16 on 1000 splits
[faithfulness/original] test rows per split [100.0], k at 5%/10%/20% = 5/10/20; R0 lift 1 to within 2.2e-16 on 1000 splits
[faithfulness/adjudicated] test rows per split [98.0], k at 5%/10%/20% = 5/10/20; R0 lift 1 to within 2.2e-16 on 1000 splits
loader guard: held-out and unlisted files refused: OK
```

| Data file | sha256 |
|---|---|
| intent/dev.jsonl | `f8d3b90ef99131fd438caf91876766fcd66c1a48b80f7f9f32abc8c0523e9fa0` |
| intent/dev-adjudicated.jsonl | `90ca4889002f5b2813bfe51036a512357f9f11071b0b4e1408f5dd167f033d1c` |
| labels/intent/haiku.jsonl | `160651b02b020a41d1c39a63aafeba50e2c45c72978c4affdbab4aa7fe42949b` |
| labels/intent/luna.jsonl | `80c373eeedd8315190643a2e3398652ff805f4e5b1f9eb50b3c9f5387521715c` |
| labels/intent/haiku-s1.jsonl | `cdc0314b67a036b74708b28ae7c7d5e0b6cd382361656897ff6de535eb476f75` |
| labels/intent/haiku-s2.jsonl | `8e81ebefcaac2afebcf3cd3fde09b2767f869fd2367de6a77c2ee31d46bff263` |
| labels/intent/haiku-s3.jsonl | `dcc67b66ec0029bf6c9961569b4ca88a286871b35cf433fd82eb1116bea26722` |
| intent/labels.json | `e64b29ebfc59f9bdbf4d4756fadd29669ab8fe623203b66790cce6d1e1b8f5f8` |
| labels/intent/haiku.cost.json | `9ae9937c37673d3e6747fd09fbf9df1f1c5f7dcf3fec69e47dff2c1735862b29` |
| labels/intent/luna.cost.json | `2afc06ac46c31a7d2046e3f5f35c8db131343a518bf5fd9e00ffa4dd7e7c92e1` |
| faithfulness/dev.jsonl | `d0fb8fd50518e1d3fb0a6319738611998d5ce3646e22c90aada8220bf4a8f821` |
| faithfulness/dev-adjudicated.jsonl | `0328b6c4fa5a9f35c59cf0f5aef3d0d34185c9e8a218ad7b3176b6f00d1d89e3` |
| labels/faithfulness/haiku.jsonl | `5c86b324538cecbe714213f03928594c635771aac9fafc83f985a97481b7ce40` |
| labels/faithfulness/luna.jsonl | `19719aa36f806d2dcce8a13172bb663f4394e8b32b47ccb60381cdb5bc2ad93f` |
| labels/faithfulness/haiku-s1.jsonl | `06b772cc801e274f0e66d3dc8359ed0e5531c293753ecb971c0373cad57d713c` |
| labels/faithfulness/haiku-s2.jsonl | `7f0ce6a98642f61230fbe0815833aa2ae1074ec11ef090c83088529d1420383f` |
| labels/faithfulness/haiku-s3.jsonl | `c76d0b86cf7db2abc6753e17479c6e2ada693b7f03d0826d148b8eaa7c778949` |
| labels/faithfulness/haiku.cost.json | `c271f1390fb86b19edcd12476ab4b71698604a3317f9385026ffa53fa5feebc5` |
| labels/faithfulness/luna.cost.json | `1027b7863bca45b5dea5141cec39329443fca700bb963b3fedfc6b2c10c656cf` |

## intent

231 rows; 131 test rows per split; k = 7/13/26 at 0.05/0.10/0.20. Primary k fraction 0.10. Bootstrap B = 1000, LB at q = 0.0125.

### Confirmatory, original gold, k fraction 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| R0 | 19.1 | 1.00 |  | -15.6 |  | 0.10 | 5.23 |
| R1 | 34.7 | 1.82 |  | +0.0 |  | 0.18 | 2.88 |
| R2 | 52.2 | 2.73 | 1.50 | +17.5 | -7.4 | 0.27 | 1.91 |
| R4 | 53.9 | 2.82 | 1.63 | +19.2 | -8.0 | 0.28 | 1.86 |

Status: R2 **INCONCLUSIVE**; R4 **INCONCLUSIVE**. Flags: R2 ['SUBSET']; R4 ['SUBSET'].

R4 minus R2 at 0.10: +1.7 points, paired LB -8.9; R4 beats R2 by the rule: False. SE of G (bootstrap): R2 10.9; R4 11.9.

### k sensitivity (mean G, points; original gold)

| Ranker | 0.05 | 0.10 | 0.20 |
|---|---|---|---|
| R2 | +12.2 | +17.5 | +8.1 |
| R4 | +18.7 | +19.2 | +9.6 |

### Label-noise control: adjudicated gold, k fraction 0.10

218 rows; 118 test rows per split.

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| R0 | 10.6 | 1.00 |  | -3.3 |  | 0.10 | 9.47 |
| R1 | 13.9 | 1.32 |  | +0.0 |  | 0.13 | 7.19 |
| R2 | 41.9 | 3.97 |  | +28.0 |  | 0.40 | 2.39 |
| R4 | 41.0 | 3.88 |  | +27.1 |  | 0.40 | 2.44 |
| E1 | 28.8 | 2.72 |  | +14.9 |  | 0.28 | 3.48 |
| E2 | 42.2 | 4.00 |  | +28.3 |  | 0.41 | 2.37 |
| E3 | 42.8 | 4.06 |  | +28.9 |  | 0.41 | 2.34 |
| E4 | 30.1 | 2.85 |  | +16.2 |  | 0.29 | 3.32 |
| E5 | 31.5 | 2.99 |  | +17.6 |  | 0.30 | 3.17 |
| E6 | 42.4 | 4.02 |  | +28.5 |  | 0.41 | 2.36 |
| E7 | 22.6 | 2.14 |  | +8.7 |  | 0.22 | 4.43 |
| E8 | 46.1 | 4.37 |  | +32.2 |  | 0.45 | 2.17 |

### Exploratory (never decisive), original gold, 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| E1 | 47.8 | 2.50 |  | +13.1 |  | 0.25 | 2.09 |
| E2 | 49.3 | 2.58 |  | +14.5 |  | 0.26 | 2.03 |
| E3 | 54.2 | 2.83 |  | +19.4 |  | 0.28 | 1.85 |
| E4 | 43.7 | 2.29 |  | +9.0 |  | 0.23 | 2.29 |
| E5 | 47.4 | 2.48 |  | +12.7 |  | 0.25 | 2.11 |
| E6 | 53.8 | 2.82 |  | +19.1 |  | 0.28 | 1.86 |
| E7 | 42.6 | 2.23 |  | +7.9 |  | 0.22 | 2.35 |
| E8 | 65.9 | 3.45 |  | +31.2 |  | 0.34 | 1.52 |

### Mechanism (flag = D >= 1, all rows)

- Haiku wrong rows flagged: 11/44 = 0.250 [0.146, 0.394]; right rows: 9/187 = 0.048 [0.026, 0.089]
- A (still wrong after adjudication): 9/23 = 0.391 [0.222, 0.592]
- F (now right): 0/9 = 0.000 [-0.000, 0.299]
- X (dropped): 2/12 = 0.167 [0.047, 0.448]
- F and X together: 2/21 = 0.095 [0.027, 0.289]
- N (right on the original gold, wrong on the adjudicated): n/a
- flag(F u X) - flag(A) = -0.296; AMBIGUITY (>= 0.20): False

### Break-even (display only)

| Ranker | $ per 1k rows | observed P_R / P_R1 | needed at h = $0.25 | needed at h = $0.01 |
|---|---|---|---|---|
| R2 | 0.386 | 1.50 | 1.02 | 1.39 |
| R4 | 0.461 | 1.55 | 1.02 | 1.46 |

## faithfulness

200 rows; 100 test rows per split; k = 5/10/20 at 0.05/0.10/0.20. Primary k fraction 0.10. Bootstrap B = 1000, LB at q = 0.0125.

### Confirmatory, original gold, k fraction 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| R0 | 17.9 | 1.00 |  | +1.8 |  | 0.10 | 5.59 |
| R1 | 16.1 | 0.90 |  | +0.0 |  | 0.09 | 6.23 |
| R2 | 54.9 | 3.06 | 1.72 | +38.8 | +11.5 | 0.31 | 1.82 |
| R4 | 56.5 | 3.16 | 1.81 | +40.4 | +12.1 | 0.32 | 1.77 |

Status: R2 **PASS**; R4 **PASS**. Flags: R2 none; R4 none.

R4 minus R2 at 0.10: +1.6 points, paired LB -2.8; R4 beats R2 by the rule: False. SE of G (bootstrap): R2 11.7; R4 11.8.

### k sensitivity (mean G, points; original gold)

| Ranker | 0.05 | 0.10 | 0.20 |
|---|---|---|---|
| R2 | +64.9 | +38.8 | +18.0 |
| R4 | +70.4 | +40.4 | +22.0 |

### Label-noise control: adjudicated gold, k fraction 0.10

198 rows; 98 test rows per split.

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| R0 | 6.6 | 1.00 |  | -1.0 |  | 0.10 | 15.23 |
| R1 | 7.6 | 1.16 |  | +0.0 |  | 0.12 | 13.16 |
| R2 | 49.8 | 7.59 |  | +42.2 |  | 0.78 | 2.01 |
| R4 | 52.6 | 8.02 |  | +45.0 |  | 0.83 | 1.90 |
| E1 | 38.5 | 5.86 |  | +30.9 |  | 0.61 | 2.60 |
| E2 | 49.9 | 7.60 |  | +42.3 |  | 0.78 | 2.00 |
| E3 | 52.2 | 7.96 |  | +44.6 |  | 0.82 | 1.91 |
| E4 | 50.1 | 7.62 |  | +42.5 |  | 0.78 | 2.00 |
| E5 | 49.1 | 7.48 |  | +41.5 |  | 0.77 | 2.04 |
| E6 | 49.0 | 7.47 |  | +41.4 |  | 0.77 | 2.04 |
| E7 | 50.1 | 7.63 |  | +42.5 |  | 0.78 | 2.00 |
| E8 | 8.8 | 1.33 |  | +1.2 |  | 0.14 | 11.42 |

### Exploratory (never decisive), original gold, 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| E1 | 34.0 | 1.90 |  | +17.9 |  | 0.19 | 2.94 |
| E2 | 55.1 | 3.08 |  | +39.0 |  | 0.31 | 1.82 |
| E3 | 56.1 | 3.14 |  | +40.1 |  | 0.31 | 1.78 |
| E4 | 57.1 | 3.19 |  | +41.0 |  | 0.32 | 1.75 |
| E5 | 49.9 | 2.79 |  | +33.8 |  | 0.28 | 2.00 |
| E6 | 54.3 | 3.04 |  | +38.3 |  | 0.31 | 1.84 |
| E7 | 54.1 | 3.02 |  | +38.0 |  | 0.30 | 1.85 |
| E8 | 18.6 | 1.04 |  | +2.5 |  | 0.10 | 5.38 |

### Mechanism (flag = D >= 1, all rows)

- Haiku wrong rows flagged: 11/36 = 0.306 [0.180, 0.469]; right rows: 8/164 = 0.049 [0.025, 0.093]
- A (still wrong after adjudication): 10/13 = 0.769 [0.497, 0.918]
- F (now right): 1/21 = 0.048 [0.008, 0.227]
- X (dropped): 0/2 = 0.000 [0.000, 0.658]
- F and X together: 1/23 = 0.043 [0.008, 0.210]
- N (right on the original gold, wrong on the adjudicated): n/a
- flag(F u X) - flag(A) = -0.726; AMBIGUITY (>= 0.20): False

### Break-even (display only)

| Ranker | $ per 1k rows | observed P_R / P_R1 | needed at h = $0.25 | needed at h = $0.01 |
|---|---|---|---|---|
| R2 | 0.522 | 3.42 | 1.02 | 1.52 |
| R4 | 0.666 | 3.52 | 1.03 | 1.67 |


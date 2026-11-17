# Planner Addendum 1: results

Design: `benchmarks/planner/DESIGN-addendum-1.md` (sha256 `420032cc81b4d9af9ef17d4f77a820b3c78af0aa446053fbcea853b51abf4f33`). PREREG_ADDENDUM sha256 `28d2f6a99ae5e8959a7ea60c7e01fc59317727e204d02e84316bce8d44e73cef`. Parent DESIGN `3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42`, PREREG `01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a`.  
Script sha256 `eee5a5b64d3e407dbcc92c4ca1e2fdd5b1f65abcd230fe21689a49ed8a7c74b2`; git `a9dd6db95476c40e60b4c834cb1ad9621dc13bdd` (clean).  
Mode: full; S = 1000 splits, B = 1000 bootstrap draws, seed 20261117; run time 0.8s.

## Verdict

**KILL_STANDS**

- neither C4 nor C5 passes on intent

| Task | C4 | C5 |
|---|---|---|
| intent | fail | fail |
| faithfulness | pass | fail |

A pass yields an escalation trigger for Ondine's existing escalation step, not a Planner. The Planner KILL stands in every outcome.

## Sample checks

```
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
```

| Task | file | used | sha256 | measured cost |
|---|---|---|---|---|
| intent | haiku-s1 | True | `cdc0314b67a036b74708b28ae7c7d5e0b6cd382361656897ff6de535eb476f75` | $0.0298 |
| intent | haiku-s2 | True | `8e81ebefcaac2afebcf3cd3fde09b2767f869fd2367de6a77c2ee31d46bff263` | $0.0295 |
| intent | haiku-s3 | True | `dcc67b66ec0029bf6c9961569b4ca88a286871b35cf433fd82eb1116bea26722` | $0.0300 |
| faithfulness | haiku-s1 | True | `06b772cc801e274f0e66d3dc8359ed0e5531c293753ecb971c0373cad57d713c` | $0.0365 |
| faithfulness | haiku-s2 | True | `7f0ce6a98642f61230fbe0815833aa2ae1074ec11ef090c83088529d1420383f` | $0.0364 |
| faithfulness | haiku-s3 | True | `c76d0b86cf7db2abc6753e17479c6e2ada693b7f03d0826d148b8eaa7c778949` | $0.0356 |

Measured sample cost, files used: $0.1978.

## intent

231 rows. Metric: accuracy. s = opus. Delta A and D2 in points; accuracy is the mean over splits with the 5-95% range. Flag rates over all rows: C4 0.087 (20 rows), C5 0.134 (31 rows).

### Confirmatory policies (original gold)

| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | adjudicated acc | Delta A, length | D2, length |
|---|---|---|---|---|---|---|---|---|---|
| C4 | 0.940 | 19% | 0.831 [0.802, 0.870] | +0.2 (-2.7) | -1.3 (-4.3) | 0.086 | 0.922 | +0.2 | -1.2 |
| C5 | 1.249 | 25% | 0.840 [0.809, 0.870] | +0.4 (-2.6) | -1.2 (-4.1) | 0.133 | 0.927 | +0.5 | -1.0 |

C5 against C4 (Delta A of C5 minus C4): flat +0.3, length +0.3, lower bound (flat) -0.7.

### Mechanism diagnostic (flag = C4's trigger, all rows)

- R33, flagged share of the blind-spot rows: 4/33 = 0.121 [0.048, 0.273]
- R44, of Haiku's wrong rows: 11/44 = 0.250 [0.146, 0.394]
- FF, of Haiku's right rows: 9/187 = 0.048 [0.026, 0.089]
- Class: **not_useful**

### Label-noise control (adjudicated gold)

- Haiku wrong rows flagged: 9/23 = 0.391 [0.222, 0.592]; right rows: 8/195 = 0.041 [0.021, 0.079]
- A (still wrong): 4/17 = 0.235 [0.096, 0.473]
- B (now right): 0/9 = 0.000 [-0.000, 0.299]
- C (no defensible label, dropped): 0/7 = 0.000 [-0.000, 0.354]
- B and C together: 0/16 = 0.000 [0.000, 0.194]
- flag(A) >= 0.40, finds real errors: False; flag(B u C) - flag(A) = -0.235, tracks ambiguity (>= 0.20): False

### Exploratory (not used for the verdict)

| Config | $/1k | accuracy | Delta A | D2 | routed |
|---|---|---|---|---|---|
| C4:opus:ask2 | 0.405 | 0.813 | -0.3 | -1.3 | 0.030 |
| C4:opus:ask3 | 0.704 | 0.818 | -0.6 | -2.0 | 0.064 |
| C4:sonnet | 0.685 | 0.827 | +0.3 | -1.1 | 0.086 |
| C5:sonnet | 0.854 | 0.822 | -0.5 | -1.9 | 0.133 |
| C4:sol | 0.611 | 0.827 | +0.5 | -0.9 | 0.086 |
| C5:sol | 0.740 | 0.827 | +0.2 | -1.2 | 0.133 |
| modal3of4:opus | 0.619 | 0.809 | -1.2 | -2.6 | 0.021 |
| vote4 | 0.514 | 0.813 | -0.6 | -1.8 | 0.000 |

## faithfulness

200 rows. Metric: balanced. s = sol. Delta A and D2 in points; accuracy is the mean over splits with the 5-95% range. Flag rates over all rows: C4 0.095 (19 rows), C5 0.210 (42 rows).

### Confirmatory policies (original gold)

| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | adjudicated acc | Delta A, length | D2, length |
|---|---|---|---|---|---|---|---|---|---|
| C4 | 0.865 | 48% | 0.866 [0.823, 0.908] | +2.7 (+0.2) | +0.7 (-1.9) | 0.095 | 0.980 | +2.3 | +0.5 |
| C5 | 1.214 | 68% | 0.860 [0.820, 0.900] | +1.3 (-1.2) | -0.1 (-2.0) | 0.209 | 0.975 | +1.0 | -0.1 |

C5 against C4 (Delta A of C5 minus C4): flat -1.4, length -1.3, lower bound (flat) -3.8.

### Mechanism diagnostic (flag = C4's trigger, all rows)

- R_blind, flagged share of the blind-spot rows: 2/22 = 0.091 [0.025, 0.278]
- R_wrong, of Haiku's wrong rows: 11/36 = 0.306 [0.180, 0.469]
- FF, of Haiku's right rows: 8/164 = 0.049 [0.025, 0.093]
- Class: **not_useful** (descriptive only)

### Label-noise control (adjudicated gold)

- Haiku wrong rows flagged: 10/13 = 0.769 [0.497, 0.918]; right rows: 9/185 = 0.049 [0.026, 0.090]
- A (still wrong): 2/2 = 1.000 [0.342, 1.000]
- B (now right): 0/18 = 0.000 [-0.000, 0.176]
- C (no defensible label, dropped): 0/2 = 0.000 [0.000, 0.658]
- B and C together: 0/20 = 0.000 [0.000, 0.161]
- flag(A) >= 0.40, finds real errors: True; flag(B u C) - flag(A) = -1.000, tracks ambiguity (>= 0.20): False

### Exploratory (not used for the verdict)

| Config | $/1k | accuracy | Delta A | D2 | routed |
|---|---|---|---|---|---|
| C4:sol:ask2 | 0.463 | 0.872 | +4.3 | +3.1 | 0.064 |
| C4:sol:ask3 | 0.663 | 0.872 | +3.8 | +2.2 | 0.079 |
| C4:sonnet | 0.972 | 0.851 | +0.9 | -1.0 | 0.095 |
| C5:sonnet | 1.451 | 0.840 | -1.3 | -2.2 | 0.209 |
| C4:opus | 1.297 | 0.845 | -0.4 | -1.6 | 0.095 |
| C5:opus | 2.170 | 0.829 | -3.2 | -3.2 | 0.209 |
| modal3of4:sol | 0.758 | 0.872 | +3.6 | +1.7 | 0.034 |
| vote4 | 0.696 | 0.857 | +2.2 | +0.5 | 0.000 |


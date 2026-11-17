# Planner replay: results

Design: `benchmarks/planner/DESIGN.md` (sha256 `3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42`). PREREG sha256 `01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a`.  
Mode: full; S = 1000 splits, B = 1000 bootstrap draws, seed 20261117; run time 7.2s.

## Verdict

**KILL**

- no confirmatory policy passes on intent

| Task | C1 | C2 | C3 |
|---|---|---|---|
| intent | fail | fail | fail |
| faithfulness | fail | fail | fail |
| attributes | fail | fail | fail |

## Integrity

```
PREREG sha256 01872888f24b359b5bb24ad167aa50e04d7ee41816718d7d93685e9e41bb0e3a
design sha256 3d667e65acd04d682855bf71d868e93b57e7aea53f2b782d4c68daaf051e4a42
loader guard: held-out and unlisted files refused: OK
[intent] dev 231 rows, adjudicated 218
  haiku   231 rows, 0 null answers, 0 invalid values, run cost $0.0297
  luna    231 rows, 0 null answers, 0 invalid values, run cost $0.0174
  sonnet  231 rows, 0 null answers, 0 invalid values, run cost $0.4614
  sol     231 rows, 0 null answers, 0 invalid values, run cost $0.2612
  opus    231 rows, 0 null answers, 0 invalid values, run cost $1.1489  (cost file has no 'source' key)
  best single model on original gold: opus (PREREG s: OK)
[faithfulness] dev 200 rows, adjudicated 198
  haiku   200 rows, 0 null answers, 0 invalid values, run cost $0.0348
  luna    200 rows, 0 null answers, 0 invalid values, run cost $0.0288
  sonnet  200 rows, 0 null answers, 0 invalid values, run cost $0.5836
  sol     200 rows, 0 null answers, 0 invalid values, run cost $0.3577
  opus    200 rows, 0 null answers, 0 invalid values, run cost $1.2708  (cost file has no 'source' key)
  best single model on original gold: sol (PREREG s: OK)
[attributes] dev 199 rows, adjudicated none (falls back to original)
  haiku   199 rows, 0 null answers, 0 invalid values, run cost $0.0725
  luna    199 rows, 0 null answers, 0 invalid values, run cost $0.0412
  sonnet  199 rows, 0 null answers, 0 invalid values, run cost $0.5423
  sol     199 rows, 0 null answers, 0 invalid values, run cost $0.5256
  opus    199 rows, 0 null answers, 0 invalid values, run cost $2.2892  (cost file has no 'source' key)
  best single model on original gold: sonnet (PREREG s: OK)
score.score vs baseline-report.json: 30 model x task x gold pairs, max |diff| 0.0e+00
task arrays vs score.score: max |diff| 0.0e+00
```

Not verified by the design: the Opus cost files carry no `source` key, and Opus intent was run twice, so which run its cost records is unknown.

## intent

231 rows. Metric: accuracy. s = opus. Delta A and D2 in points; a policy's accuracy is the mean over splits with the 5-95% range.

### Single models and the hull

| Model | $ per 1k rows | metric (all rows) |
|---|---|---|
| haiku | 0.129 | 0.810 |
| luna | 0.075 | 0.788 |
| sonnet | 1.997 | 0.818 |
| sol | 1.131 | 0.831 |
| opus | 4.973 | 0.883 |

### Confirmatory policies (original gold)

| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | routed, cheap wrong | $ per 1k successful rows | adjudicated acc | Delta A, length | D2, length |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.672 | 14% | 0.831 [0.794, 0.863] | +0.8 (-1.7) | -0.6 (-3.7) | 0.094 | 0.503 | 0.81 | 0.917 | +0.9 | -0.5 |
| C2 | 1.704 | 34% | 0.840 [0.794, 0.878] | -0.0 (-3.0) | -1.9 (-4.8) | 0.317 | 0.305 | 2.01 | 0.907 | -0.1 | -2.0 |
| C3 | 2.053 | 41% | 0.853 [0.817, 0.893] | +0.9 (-2.0) | -1.3 (-3.7) | 0.372 | 0.324 | 2.39 | 0.925 | +0.8 | -1.3 |

Paired against C1 (Delta A of the policy minus Delta A of C1):

| Policy | flat | length | LB (flat) |
|---|---|---|---|
| C2 | -0.8 | -0.9 | -3.9 |
| C3 | +0.1 | -0.0 | -2.1 |

### Diagnostics

- Cheap pair agrees and is wrong on 33 of 231 units (0.143), 0.750 of the cheap model's 44 wrong units. C1 accuracy ceiling if s were perfect: 0.857.
- O1 (cheapest right model per unit, unreachable): accuracy 0.913, $/1k 0.332. O2 (any model right): 0.913.
- O3 (C2 fitted on all rows, unreachable): accuracy 0.896; O3 minus C2 = 5.66 points (learning optimism).

### Exploratory grid (not used for the verdict)

| Config | $/1k | accuracy | Delta A | D2 | routed |
|---|---|---|---|---|---|
| C1:sonnet | 0.392 | 0.809 | -0.7 | -1.7 | 0.094 |
| C1:sol | 0.310 | 0.818 | +0.4 | -0.4 | 0.094 |
| C1:opus | 0.672 | 0.831 | +0.8 | -0.6 | 0.094 |
| C2:haiku:sonnet:n50:t0.6 | 0.184 | 0.812 | +0.1 | -0.3 | 0.028 |
| C2:haiku:sonnet:n50:t0.7 | 0.346 | 0.816 | +0.1 | -0.6 | 0.109 |
| C2:haiku:sonnet:n50:t0.8 | 0.844 | 0.816 | -0.7 | -2.0 | 0.358 |
| C2:haiku:sonnet:n50:t0.9 | 1.996 | 0.818 | -2.6 | -5.2 | 0.935 |
| C2:haiku:sonnet:n100:t0.6 | 0.204 | 0.812 | +0.1 | -0.4 | 0.038 |
| C2:haiku:sonnet:n100:t0.7 | 0.418 | 0.817 | +0.1 | -0.8 | 0.145 |
| C2:haiku:sonnet:n100:t0.8 | 0.761 | 0.820 | -0.4 | -1.8 | 0.317 |
| C2:haiku:sonnet:n100:t0.9 | 2.079 | 0.817 | -2.9 | -5.5 | 0.976 |
| C2:haiku:sol:n50:t0.6 | 0.160 | 0.812 | +0.1 | -0.2 | 0.028 |
| C2:haiku:sol:n50:t0.7 | 0.251 | 0.816 | +0.3 | -0.2 | 0.109 |
| C2:haiku:sol:n50:t0.8 | 0.533 | 0.819 | +0.1 | -0.8 | 0.358 |
| C2:haiku:sol:n50:t0.9 | 1.186 | 0.829 | -0.3 | -1.9 | 0.935 |
| C2:haiku:sol:n100:t0.6 | 0.171 | 0.812 | +0.1 | -0.3 | 0.038 |
| C2:haiku:sol:n100:t0.7 | 0.292 | 0.817 | +0.4 | -0.3 | 0.145 |
| C2:haiku:sol:n100:t0.8 | 0.487 | 0.822 | +0.5 | -0.6 | 0.317 |
| C2:haiku:sol:n100:t0.9 | 1.233 | 0.830 | -0.6 | -2.2 | 0.976 |
| C2:haiku:opus:n50:t0.6 | 0.266 | 0.813 | +0.1 | -0.4 | 0.028 |
| C2:haiku:opus:n50:t0.7 | 0.669 | 0.822 | +0.1 | -1.0 | 0.109 |
| C2:haiku:opus:n50:t0.8 | 1.909 | 0.837 | -0.1 | -1.5 | 0.358 |
| C2:haiku:opus:n50:t0.9 | 4.778 | 0.877 | -0.2 | -0.4 | 0.935 |
| C2:haiku:opus:n100:t0.6 | 0.316 | 0.813 | -0.1 | -0.8 | 0.038 |
| C2:haiku:opus:n100:t0.7 | 0.849 | 0.824 | -0.2 | -1.6 | 0.145 |
| C2:haiku:opus:n100:t0.8 | 1.704 | 0.840 | -0.0 | -1.9 | 0.317 |
| C2:haiku:opus:n100:t0.9 | 4.985 | 0.879 | -0.3 | -0.4 | 0.976 |
| C2:luna:sonnet:n50:t0.6 | 0.169 | 0.794 | -0.8 | -2.0 | 0.047 |
| C2:luna:sonnet:n50:t0.7 | 0.356 | 0.801 | -1.3 | -2.1 | 0.140 |
| C2:luna:sonnet:n50:t0.8 | 1.058 | 0.808 | -1.9 | -3.4 | 0.492 |
| C2:luna:sonnet:n50:t0.9 | 2.015 | 0.818 | -2.6 | -5.3 | 0.971 |
| C2:luna:sonnet:n100:t0.6 | 0.196 | 0.798 | -1.1 | -1.8 | 0.061 |
| C2:luna:sonnet:n100:t0.7 | 0.449 | 0.805 | -1.2 | -2.3 | 0.187 |
| C2:luna:sonnet:n100:t0.8 | 0.942 | 0.811 | -1.7 | -3.1 | 0.434 |
| C2:luna:sonnet:n100:t0.9 | 2.063 | 0.818 | -2.8 | -5.4 | 0.995 |
| C2:luna:sol:n50:t0.6 | 0.129 | 0.796 | -0.4 | -1.7 | 0.047 |
| C2:luna:sol:n50:t0.7 | 0.234 | 0.803 | -0.7 | -1.4 | 0.140 |
| C2:luna:sol:n50:t0.8 | 0.632 | 0.814 | -0.7 | -1.6 | 0.492 |
| C2:luna:sol:n50:t0.9 | 1.173 | 0.830 | -0.3 | -1.9 | 0.971 |
| C2:luna:sol:n100:t0.6 | 0.144 | 0.800 | -0.6 | -1.3 | 0.061 |
| C2:luna:sol:n100:t0.7 | 0.287 | 0.808 | -0.5 | -1.2 | 0.187 |
| C2:luna:sol:n100:t0.8 | 0.566 | 0.816 | -0.4 | -1.5 | 0.434 |
| C2:luna:sol:n100:t0.9 | 1.200 | 0.831 | -0.4 | -2.0 | 0.995 |
| C2:luna:opus:n50:t0.6 | 0.310 | 0.796 | -0.9 | -2.3 | 0.047 |
| C2:luna:opus:n50:t0.7 | 0.773 | 0.808 | -1.4 | -2.7 | 0.140 |
| C2:luna:opus:n50:t0.8 | 2.522 | 0.835 | -1.2 | -2.4 | 0.492 |
| C2:luna:opus:n50:t0.9 | 4.904 | 0.880 | -0.1 | -0.2 | 0.971 |
| C2:luna:opus:n100:t0.6 | 0.376 | 0.800 | -1.3 | -2.4 | 0.061 |
| C2:luna:opus:n100:t0.7 | 1.007 | 0.813 | -1.7 | -3.2 | 0.187 |
| C2:luna:opus:n100:t0.8 | 2.232 | 0.832 | -1.5 | -3.6 | 0.434 |
| C2:luna:opus:n100:t0.9 | 5.024 | 0.882 | -0.1 | -0.1 | 0.995 |
| C3:haiku:sonnet:n50:t0.6 | 0.442 | 0.812 | -0.5 | -1.4 | 0.119 |
| C3:haiku:sonnet:n50:t0.7 | 0.582 | 0.816 | -0.4 | -1.6 | 0.189 |
| C3:haiku:sonnet:n50:t0.8 | 1.035 | 0.817 | -1.0 | -2.6 | 0.416 |
| C3:haiku:sonnet:n50:t0.9 | 2.084 | 0.818 | -2.6 | -5.4 | 0.941 |
| C3:haiku:sonnet:n100:t0.6 | 0.459 | 0.812 | -0.5 | -1.6 | 0.128 |
| C3:haiku:sonnet:n100:t0.7 | 0.642 | 0.818 | -0.4 | -1.8 | 0.219 |
| C3:haiku:sonnet:n100:t0.8 | 0.947 | 0.820 | -0.8 | -2.3 | 0.372 |
| C3:haiku:sonnet:n100:t0.9 | 2.159 | 0.818 | -2.9 | -5.6 | 0.979 |
| C3:haiku:sol:n50:t0.6 | 0.338 | 0.820 | +0.6 | -0.2 | 0.119 |
| C3:haiku:sol:n50:t0.7 | 0.418 | 0.824 | +0.7 | -0.2 | 0.189 |
| C3:haiku:sol:n50:t0.8 | 0.674 | 0.826 | +0.5 | -0.7 | 0.416 |
| C3:haiku:sol:n50:t0.9 | 1.268 | 0.830 | -0.3 | -2.0 | 0.941 |
| C3:haiku:sol:n100:t0.6 | 0.349 | 0.820 | +0.5 | -0.3 | 0.128 |
| C3:haiku:sol:n100:t0.7 | 0.452 | 0.825 | +0.8 | -0.3 | 0.219 |
| C3:haiku:sol:n100:t0.8 | 0.624 | 0.829 | +0.8 | -0.5 | 0.372 |
| C3:haiku:sol:n100:t0.9 | 1.311 | 0.831 | -0.6 | -2.3 | 0.979 |
| C3:haiku:opus:n50:t0.6 | 0.796 | 0.834 | +0.9 | -0.4 | 0.119 |
| C3:haiku:opus:n50:t0.7 | 1.145 | 0.840 | +0.9 | -0.6 | 0.189 |
| C3:haiku:opus:n50:t0.8 | 2.272 | 0.851 | +0.6 | -0.9 | 0.416 |
| C3:haiku:opus:n50:t0.9 | 4.886 | 0.879 | -0.1 | -0.2 | 0.941 |
| C3:haiku:opus:n100:t0.6 | 0.840 | 0.834 | +0.7 | -0.7 | 0.128 |
| C3:haiku:opus:n100:t0.7 | 1.295 | 0.842 | +0.7 | -1.0 | 0.219 |
| C3:haiku:opus:n100:t0.8 | 2.053 | 0.853 | +0.9 | -1.3 | 0.372 |
| C3:haiku:opus:n100:t0.9 | 5.073 | 0.880 | -0.2 | -0.3 | 0.979 |

## faithfulness

200 rows. Metric: balanced. s = sol. Delta A and D2 in points; a policy's accuracy is the mean over splits with the 5-95% range.

### Single models and the hull

| Model | $ per 1k rows | metric (all rows) |
|---|---|---|
| haiku | 0.174 | 0.820 |
| luna | 0.144 | 0.781 |
| sonnet | 2.918 | 0.840 |
| sol | 1.789 | 0.860 |
| opus | 6.354 | 0.824 |

### Confirmatory policies (original gold)

| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | routed, cheap wrong | $ per 1k successful rows | adjudicated acc | Delta A, length | D2, length |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 0.631 | 35% | 0.850 [0.808, 0.890] | +1.7 (-1.8) | +0.2 (-2.4) | 0.175 | 0.397 | 0.74 | 0.965 | +1.6 | -0.1 |
| C2 | 0.736 | 41% | 0.826 [0.776, 0.892] | -0.6 (-3.5) | -1.6 (-5.2) | 0.314 | 0.140 | 0.87 | 0.934 | -0.6 | -1.6 |
| C3 | 1.090 | 61% | 0.852 [0.808, 0.896] | +1.1 (-2.0) | -0.2 (-2.7) | 0.432 | 0.295 | 1.27 | 0.965 | +1.0 | -0.4 |

Paired against C1 (Delta A of the policy minus Delta A of C1):

| Policy | flat | length | LB (flat) |
|---|---|---|---|
| C2 | -2.3 | -2.2 | -6.0 |
| C3 | -0.6 | -0.6 | -2.4 |

### Diagnostics

- Cheap pair agrees and is wrong on 22 of 200 units (0.110), 0.611 of the cheap model's 36 wrong units. C1 accuracy ceiling if s were perfect: 0.890.
- O1 (cheapest right model per unit, unreachable): accuracy 0.906, $/1k 0.177. O2 (any model right): 0.906.
- O3 (C2 fitted on all rows, unreachable): accuracy 0.821; O3 minus C2 = -0.57 points (learning optimism).

### Exploratory grid (not used for the verdict)

| Config | $/1k | accuracy | Delta A | D2 | routed |
|---|---|---|---|---|---|
| C1:sonnet | 0.829 | 0.834 | -0.3 | -2.3 | 0.175 |
| C1:sol | 0.631 | 0.850 | +1.7 | +0.2 | 0.175 |
| C1:opus | 1.430 | 0.824 | -2.9 | -3.7 | 0.175 |
| C2:haiku:sonnet:n50:t0.6 | 0.176 | 0.820 | -0.0 | -0.4 | 0.001 |
| C2:haiku:sonnet:n50:t0.7 | 0.271 | 0.819 | -0.3 | -0.7 | 0.033 |
| C2:haiku:sonnet:n50:t0.8 | 1.237 | 0.824 | -1.6 | -2.0 | 0.364 |
| C2:haiku:sonnet:n50:t0.9 | 2.822 | 0.837 | -2.2 | -2.3 | 0.908 |
| C2:haiku:sonnet:n100:t0.6 | 0.174 | 0.821 | -0.1 | -0.5 | 0.000 |
| C2:haiku:sonnet:n100:t0.7 | 0.177 | 0.821 | -0.1 | -0.5 | 0.001 |
| C2:haiku:sonnet:n100:t0.8 | 1.091 | 0.819 | -1.9 | -2.3 | 0.314 |
| C2:haiku:sonnet:n100:t0.9 | 3.048 | 0.838 | -2.3 | -2.3 | 0.985 |
| C2:haiku:sol:n50:t0.6 | 0.175 | 0.820 | -0.0 | -0.4 | 0.001 |
| C2:haiku:sol:n50:t0.7 | 0.234 | 0.820 | -0.1 | -0.6 | 0.033 |
| C2:haiku:sol:n50:t0.8 | 0.825 | 0.831 | -0.3 | -1.2 | 0.364 |
| C2:haiku:sol:n50:t0.9 | 1.797 | 0.855 | -0.1 | -0.5 | 0.908 |
| C2:haiku:sol:n100:t0.6 | 0.174 | 0.821 | -0.1 | -0.5 | 0.000 |
| C2:haiku:sol:n100:t0.7 | 0.176 | 0.821 | -0.1 | -0.5 | 0.001 |
| C2:haiku:sol:n100:t0.8 | 0.736 | 0.826 | -0.6 | -1.6 | 0.314 |
| C2:haiku:sol:n100:t0.9 | 1.936 | 0.859 | -0.1 | -0.2 | 0.985 |
| C2:haiku:opus:n50:t0.6 | 0.178 | 0.820 | -0.0 | -0.4 | 0.001 |
| C2:haiku:opus:n50:t0.7 | 0.386 | 0.818 | -0.4 | -0.8 | 0.033 |
| C2:haiku:opus:n50:t0.8 | 2.489 | 0.818 | -2.4 | -2.6 | 0.364 |
| C2:haiku:opus:n50:t0.9 | 5.941 | 0.822 | -3.7 | -3.7 | 0.908 |
| C2:haiku:opus:n100:t0.6 | 0.174 | 0.821 | -0.1 | -0.5 | 0.000 |
| C2:haiku:opus:n100:t0.7 | 0.181 | 0.821 | -0.1 | -0.5 | 0.001 |
| C2:haiku:opus:n100:t0.8 | 2.172 | 0.814 | -2.6 | -2.8 | 0.314 |
| C2:haiku:opus:n100:t0.9 | 6.433 | 0.823 | -3.8 | -3.8 | 0.985 |
| C2:luna:sonnet:n50:t0.6 | 0.172 | 0.782 | -0.0 | -4.1 | 0.010 |
| C2:luna:sonnet:n50:t0.7 | 0.568 | 0.793 | -0.5 | -3.9 | 0.145 |
| C2:luna:sonnet:n50:t0.8 | 1.849 | 0.825 | -1.8 | -2.8 | 0.584 |
| C2:luna:sonnet:n50:t0.9 | 2.896 | 0.840 | -2.0 | -2.1 | 0.943 |
| C2:luna:sonnet:n100:t0.6 | 0.144 | 0.781 | +0.0 | -4.3 | 0.000 |
| C2:luna:sonnet:n100:t0.7 | 0.281 | 0.785 | -0.1 | -4.3 | 0.047 |
| C2:luna:sonnet:n100:t0.8 | 1.954 | 0.833 | -1.7 | -2.4 | 0.620 |
| C2:luna:sonnet:n100:t0.9 | 3.007 | 0.839 | -2.2 | -2.2 | 0.981 |
| C2:luna:sol:n50:t0.6 | 0.161 | 0.782 | +0.0 | -4.1 | 0.010 |
| C2:luna:sol:n50:t0.7 | 0.404 | 0.796 | +0.1 | -3.6 | 0.145 |
| C2:luna:sol:n50:t0.8 | 1.189 | 0.836 | +0.1 | -1.7 | 0.584 |
| C2:luna:sol:n50:t0.9 | 1.831 | 0.859 | +0.1 | -0.1 | 0.943 |
| C2:luna:sol:n100:t0.6 | 0.144 | 0.781 | +0.0 | -4.3 | 0.000 |
| C2:luna:sol:n100:t0.7 | 0.228 | 0.786 | +0.1 | -4.2 | 0.047 |
| C2:luna:sol:n100:t0.8 | 1.253 | 0.845 | +0.4 | -1.2 | 0.620 |
| C2:luna:sol:n100:t0.9 | 1.899 | 0.860 | -0.0 | -0.1 | 0.981 |
| C2:luna:opus:n50:t0.6 | 0.205 | 0.782 | -0.0 | -4.1 | 0.010 |
| C2:luna:opus:n50:t0.7 | 1.067 | 0.790 | -0.8 | -4.2 | 0.145 |
| C2:luna:opus:n50:t0.8 | 3.856 | 0.814 | -3.0 | -4.0 | 0.584 |
| C2:luna:opus:n50:t0.9 | 6.136 | 0.824 | -3.6 | -3.6 | 0.943 |
| C2:luna:opus:n100:t0.6 | 0.144 | 0.781 | +0.0 | -4.3 | 0.000 |
| C2:luna:opus:n100:t0.7 | 0.441 | 0.784 | -0.2 | -4.3 | 0.047 |
| C2:luna:opus:n100:t0.8 | 4.086 | 0.820 | -3.1 | -3.8 | 0.620 |
| C2:luna:opus:n100:t0.9 | 6.378 | 0.824 | -3.7 | -3.7 | 0.981 |
| C3:haiku:sonnet:n50:t0.6 | 0.828 | 0.835 | -0.2 | -2.2 | 0.175 |
| C3:haiku:sonnet:n50:t0.7 | 0.903 | 0.834 | -0.3 | -2.3 | 0.200 |
| C3:haiku:sonnet:n50:t0.8 | 1.703 | 0.836 | -1.3 | -2.3 | 0.475 |
| C3:haiku:sonnet:n50:t0.9 | 3.014 | 0.839 | -2.1 | -2.1 | 0.924 |
| C3:haiku:sonnet:n100:t0.6 | 0.829 | 0.834 | -0.3 | -2.3 | 0.175 |
| C3:haiku:sonnet:n100:t0.7 | 0.832 | 0.834 | -0.3 | -2.3 | 0.176 |
| C3:haiku:sonnet:n100:t0.8 | 1.577 | 0.834 | -1.4 | -2.5 | 0.432 |
| C3:haiku:sonnet:n100:t0.9 | 3.200 | 0.839 | -2.2 | -2.2 | 0.988 |
| C3:haiku:sol:n50:t0.6 | 0.631 | 0.850 | +1.9 | +0.3 | 0.175 |
| C3:haiku:sol:n50:t0.7 | 0.677 | 0.850 | +1.8 | +0.2 | 0.200 |
| C3:haiku:sol:n50:t0.8 | 1.167 | 0.853 | +1.1 | -0.1 | 0.475 |
| C3:haiku:sol:n50:t0.9 | 1.970 | 0.859 | +0.1 | -0.1 | 0.924 |
| C3:haiku:sol:n100:t0.6 | 0.631 | 0.850 | +1.7 | +0.2 | 0.175 |
| C3:haiku:sol:n100:t0.7 | 0.633 | 0.850 | +1.7 | +0.2 | 0.176 |
| C3:haiku:sol:n100:t0.8 | 1.090 | 0.852 | +1.1 | -0.2 | 0.432 |
| C3:haiku:sol:n100:t0.9 | 2.084 | 0.860 | -0.0 | -0.1 | 0.988 |
| C3:haiku:opus:n50:t0.6 | 1.429 | 0.824 | -2.7 | -3.6 | 0.175 |
| C3:haiku:opus:n50:t0.7 | 1.592 | 0.824 | -2.8 | -3.6 | 0.200 |
| C3:haiku:opus:n50:t0.8 | 3.334 | 0.824 | -3.3 | -3.7 | 0.475 |
| C3:haiku:opus:n50:t0.9 | 6.189 | 0.824 | -3.6 | -3.6 | 0.924 |
| C3:haiku:opus:n100:t0.6 | 1.430 | 0.824 | -2.9 | -3.7 | 0.175 |
| C3:haiku:opus:n100:t0.7 | 1.436 | 0.824 | -2.9 | -3.7 | 0.176 |
| C3:haiku:opus:n100:t0.8 | 3.060 | 0.823 | -3.4 | -3.8 | 0.432 |
| C3:haiku:opus:n100:t0.9 | 6.593 | 0.824 | -3.8 | -3.8 | 0.988 |

## attributes

199 rows. Metric: micro. s = sonnet. Delta A and D2 in points; a policy's accuracy is the mean over splits with the 5-95% range.

### Single models and the hull

| Model | $ per 1k rows | metric (all rows) |
|---|---|---|
| haiku | 0.364 | 0.916 |
| luna | 0.207 | 0.906 |
| sonnet | 2.725 | 0.922 |
| sol | 2.641 | 0.907 |
| opus | 11.504 | 0.917 |

### Confirmatory policies (original gold)

| Policy | $/1k | % of s | accuracy | Delta A (LB) | D2 (LB) | routed | routed, cheap wrong | $ per 1k successful rows | adjudicated acc | Delta A, length | D2, length |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 2.765 | 101% | 0.923 [0.914, 0.931] | +0.1 (-0.4) | +0.1 (-0.5) | 0.157 | 0.327 | 8.05 | 0.923 | +0.1 | +0.1 |
| C2 | 2.733 | 100% | 0.928 [0.919, 0.937] | +0.6 (-0.2) | +0.6 (-0.2) | 0.102 | 0.427 | 7.88 | 0.928 | +0.6 | +0.6 |
| C3 | 3.199 | 117% | 0.927 [0.919, 0.935] | +0.5 (-0.0) | +0.5 (-0.0) | 0.221 | 0.307 | 8.77 | 0.927 | +0.5 | +0.5 |

Paired against C1 (Delta A of the policy minus Delta A of C1):

| Policy | flat | length | LB (flat) |
|---|---|---|---|
| C2 | +0.5 | +0.5 | -0.3 |
| C3 | +0.4 | +0.4 | +0.1 |

Attributes, secondary metrics (strict attribute accuracy; rows with every attribute right):

- C1: strict 0.834 [0.821, 0.847], all-right rows 0.347 [0.293, 0.404]
- C2: strict 0.834 [0.820, 0.848], all-right rows 0.350 [0.293, 0.404]
- C3: strict 0.837 [0.825, 0.850], all-right rows 0.368 [0.313, 0.424]

### Diagnostics

- Cheap pair agrees and is wrong on 101 of 3074 units (0.033), 0.391 of the cheap model's 258 wrong units. C1 accuracy ceiling if s were perfect: 0.967.
- O1 (cheapest right model per unit, unreachable): accuracy 0.967, $/1k 1.292. O2 (any model right): 0.967.
- O3 (C2 fitted on all rows, unreachable): accuracy 0.932; O3 minus C2 = 0.35 points (learning optimism).

### Exploratory grid (not used for the verdict)

| Config | $/1k | accuracy | Delta A | D2 | routed |
|---|---|---|---|---|---|
| C1:sonnet | 2.765 | 0.923 | +0.1 | +0.1 | 0.157 |
| C1:sol | 2.698 | 0.917 | -0.5 | -0.5 | 0.157 |
| C1:opus | 9.832 | 0.918 | -0.4 | -0.4 | 0.157 |
| C2:haiku:sonnet:n50:t0.6 | 1.134 | 0.919 | +0.2 | -0.1 | 0.021 |
| C2:haiku:sonnet:n50:t0.7 | 1.801 | 0.923 | +0.3 | +0.1 | 0.045 |
| C2:haiku:sonnet:n50:t0.8 | 2.617 | 0.927 | +0.5 | +0.5 | 0.096 |
| C2:haiku:sonnet:n50:t0.9 | 2.987 | 0.928 | +0.6 | +0.6 | 0.205 |
| C2:haiku:sonnet:n100:t0.6 | 1.521 | 0.922 | +0.3 | +0.0 | 0.034 |
| C2:haiku:sonnet:n100:t0.7 | 2.121 | 0.925 | +0.5 | +0.3 | 0.058 |
| C2:haiku:sonnet:n100:t0.8 | 2.733 | 0.928 | +0.6 | +0.6 | 0.102 |
| C2:haiku:sonnet:n100:t0.9 | 3.047 | 0.929 | +0.7 | +0.7 | 0.214 |
| C2:haiku:sol:n50:t0.6 | 1.110 | 0.919 | +0.2 | -0.1 | 0.021 |
| C2:haiku:sol:n50:t0.7 | 1.757 | 0.923 | +0.4 | +0.1 | 0.045 |
| C2:haiku:sol:n50:t0.8 | 2.548 | 0.926 | +0.5 | +0.4 | 0.096 |
| C2:haiku:sol:n50:t0.9 | 2.906 | 0.925 | +0.3 | +0.3 | 0.205 |
| C2:haiku:sol:n100:t0.6 | 1.485 | 0.923 | +0.4 | +0.1 | 0.034 |
| C2:haiku:sol:n100:t0.7 | 2.067 | 0.926 | +0.6 | +0.4 | 0.058 |
| C2:haiku:sol:n100:t0.8 | 2.660 | 0.928 | +0.6 | +0.6 | 0.102 |
| C2:haiku:sol:n100:t0.9 | 2.964 | 0.925 | +0.3 | +0.3 | 0.214 |
| C2:haiku:opus:n50:t0.6 | 3.612 | 0.919 | -0.2 | -0.3 | 0.021 |
| C2:haiku:opus:n50:t0.7 | 6.431 | 0.922 | +0.0 | +0.0 | 0.045 |
| C2:haiku:opus:n50:t0.8 | 9.873 | 0.926 | +0.4 | +0.4 | 0.096 |
| C2:haiku:opus:n50:t0.9 | 11.435 | 0.927 | +0.5 | +0.5 | 0.205 |
| C2:haiku:opus:n100:t0.6 | 5.247 | 0.922 | +0.0 | -0.0 | 0.034 |
| C2:haiku:opus:n100:t0.7 | 7.780 | 0.925 | +0.2 | +0.2 | 0.058 |
| C2:haiku:opus:n100:t0.8 | 10.362 | 0.927 | +0.5 | +0.5 | 0.102 |
| C2:haiku:opus:n100:t0.9 | 11.687 | 0.928 | +0.5 | +0.5 | 0.214 |
| C2:luna:sonnet:n50:t0.6 | 1.372 | 0.918 | -0.0 | -0.3 | 0.038 |
| C2:luna:sonnet:n50:t0.7 | 1.776 | 0.921 | +0.2 | -0.1 | 0.057 |
| C2:luna:sonnet:n50:t0.8 | 2.331 | 0.925 | +0.4 | +0.3 | 0.100 |
| C2:luna:sonnet:n50:t0.9 | 2.791 | 0.926 | +0.4 | +0.4 | 0.212 |
| C2:luna:sonnet:n100:t0.6 | 1.684 | 0.921 | +0.2 | -0.1 | 0.050 |
| C2:luna:sonnet:n100:t0.7 | 1.976 | 0.925 | +0.4 | +0.2 | 0.070 |
| C2:luna:sonnet:n100:t0.8 | 2.367 | 0.927 | +0.6 | +0.5 | 0.108 |
| C2:luna:sonnet:n100:t0.9 | 2.824 | 0.928 | +0.6 | +0.6 | 0.210 |
| C2:luna:sol:n50:t0.6 | 1.337 | 0.916 | -0.3 | -0.5 | 0.038 |
| C2:luna:sol:n50:t0.7 | 1.728 | 0.919 | -0.1 | -0.3 | 0.057 |
| C2:luna:sol:n50:t0.8 | 2.265 | 0.921 | +0.1 | -0.1 | 0.100 |
| C2:luna:sol:n50:t0.9 | 2.712 | 0.916 | -0.5 | -0.6 | 0.212 |
| C2:luna:sol:n100:t0.6 | 1.639 | 0.919 | -0.0 | -0.3 | 0.050 |
| C2:luna:sol:n100:t0.7 | 1.922 | 0.922 | +0.2 | -0.0 | 0.070 |
| C2:luna:sol:n100:t0.8 | 2.301 | 0.924 | +0.3 | +0.2 | 0.108 |
| C2:luna:sol:n100:t0.9 | 2.743 | 0.918 | -0.4 | -0.5 | 0.210 |
| C2:luna:opus:n50:t0.6 | 5.127 | 0.916 | -0.6 | -0.6 | 0.038 |
| C2:luna:opus:n50:t0.7 | 6.831 | 0.919 | -0.3 | -0.3 | 0.057 |
| C2:luna:opus:n50:t0.8 | 9.172 | 0.922 | -0.0 | -0.0 | 0.100 |
| C2:luna:opus:n50:t0.9 | 11.116 | 0.921 | -0.1 | -0.1 | 0.212 |
| C2:luna:opus:n100:t0.6 | 6.442 | 0.919 | -0.3 | -0.3 | 0.050 |
| C2:luna:opus:n100:t0.7 | 7.675 | 0.922 | -0.0 | -0.0 | 0.070 |
| C2:luna:opus:n100:t0.8 | 9.326 | 0.924 | +0.2 | +0.1 | 0.108 |
| C2:luna:opus:n100:t0.9 | 11.253 | 0.922 | +0.0 | +0.0 | 0.210 |
| C3:haiku:sonnet:n50:t0.6 | 2.893 | 0.925 | +0.3 | +0.3 | 0.168 |
| C3:haiku:sonnet:n50:t0.7 | 3.019 | 0.926 | +0.4 | +0.4 | 0.184 |
| C3:haiku:sonnet:n50:t0.8 | 3.169 | 0.927 | +0.5 | +0.5 | 0.217 |
| C3:haiku:sonnet:n50:t0.9 | 3.247 | 0.926 | +0.4 | +0.4 | 0.298 |
| C3:haiku:sonnet:n100:t0.6 | 2.962 | 0.926 | +0.4 | +0.4 | 0.175 |
| C3:haiku:sonnet:n100:t0.7 | 3.094 | 0.927 | +0.5 | +0.5 | 0.191 |
| C3:haiku:sonnet:n100:t0.8 | 3.199 | 0.927 | +0.5 | +0.5 | 0.221 |
| C3:haiku:sonnet:n100:t0.9 | 3.267 | 0.927 | +0.5 | +0.5 | 0.299 |
| C3:haiku:sol:n50:t0.6 | 2.822 | 0.918 | -0.4 | -0.4 | 0.168 |
| C3:haiku:sol:n50:t0.7 | 2.944 | 0.919 | -0.2 | -0.2 | 0.184 |
| C3:haiku:sol:n50:t0.8 | 3.089 | 0.920 | -0.2 | -0.2 | 0.217 |
| C3:haiku:sol:n50:t0.9 | 3.165 | 0.917 | -0.5 | -0.5 | 0.298 |
| C3:haiku:sol:n100:t0.6 | 2.888 | 0.919 | -0.3 | -0.3 | 0.175 |
| C3:haiku:sol:n100:t0.7 | 3.016 | 0.921 | -0.2 | -0.2 | 0.191 |
| C3:haiku:sol:n100:t0.8 | 3.118 | 0.920 | -0.2 | -0.2 | 0.221 |
| C3:haiku:sol:n100:t0.9 | 3.184 | 0.916 | -0.6 | -0.6 | 0.299 |
| C3:haiku:opus:n50:t0.6 | 10.372 | 0.920 | -0.2 | -0.2 | 0.168 |
| C3:haiku:opus:n50:t0.7 | 10.903 | 0.921 | -0.1 | -0.1 | 0.184 |
| C3:haiku:opus:n50:t0.8 | 11.537 | 0.922 | -0.0 | -0.0 | 0.217 |
| C3:haiku:opus:n50:t0.9 | 11.868 | 0.921 | -0.1 | -0.1 | 0.298 |
| C3:haiku:opus:n100:t0.6 | 10.663 | 0.921 | -0.1 | -0.1 | 0.175 |
| C3:haiku:opus:n100:t0.7 | 11.221 | 0.922 | -0.0 | -0.0 | 0.191 |
| C3:haiku:opus:n100:t0.8 | 11.662 | 0.922 | -0.0 | -0.0 | 0.221 |
| C3:haiku:opus:n100:t0.9 | 11.951 | 0.922 | -0.1 | -0.1 | 0.299 |
| C1:sonnet:row | 2.765 | 0.922 | -0.0 | -0.1 | 0.825 |
| C2:haiku:sonnet:n100:t0.8:row | 2.733 | 0.922 | -0.0 | -0.0 | 0.876 |
| C3:haiku:sonnet:n100:t0.8:row | 3.199 | 0.922 | -0.0 | -0.0 | 0.972 |


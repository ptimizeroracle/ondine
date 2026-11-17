# Review-next confirmation: results

Design: `benchmarks/review_next/DESIGN-confirm.md` (sha256 `b8de9ba652b1bddfde4e586693ef90945cdd9ce2b5de662b3f41a95bd96c539a`). PREREG_CONFIRM sha256 `fb7d2c3ff926bebff322e57aa96cb2c95446c86a2a5e8545bcd3b34b532e9ced`. Parent: DESIGN `57e8962a0173c860dc464e2fb4cc2a456c3ece2f711c3209412f17265cfb7bff`, PREREG `85737be9a20a34cc253f665ab37fa5bd10dae2ce674d3b6e382875d9426afecc`, script `94836cab636f6d40a3880ec90c127d502da4f6ec7b55cf3841fbed0879ad7e35`.  
Script sha256 `2c3670bca858574a0801ea15e7718eb38c905eb64f9335f2c4c78036e30c91d4`; git `e87e8fc8d8d005d23bb7abd2ff7dac35a12ba6db` (clean).  
Mode: full; S = 1000 splits, B = 1000 bootstrap draws, seed 20261118; run time 7.4s.

## Verdict

**INCONCLUSIVE**

- FLAGS intent: none
- FLAGS faithfulness: DOMAIN

| Task | R2 status |
|---|---|
| intent | PASS |
| faithfulness | INCONCLUSIVE |

A pass is a ranking signal for the review step, not a router. The Planner KILL stands in every outcome. Extra samples cost per 1k rows: intent $0.383; faithfulness $0.531.

## Checks

```
PREREG_CONFIRM sha256 fb7d2c3ff926bebff322e57aa96cb2c95446c86a2a5e8545bcd3b34b532e9ced
design, parent design, parent PREREG and parent script hashes: OK
[intent] confirm rows: 770 (77 classes x 10); disjoint from dev; sha256 6ccee5cc9550152c matches the PREREG
[faithfulness] confirm rows: 320 (8 domains x 20 per label; max 5 rows per document and label); disjoint from dev; sha256 6c8e98e0163ee7bc matches the PREREG
  haiku-s1: V1 0.819 in [0.750, 0.870] ok; V2 90.2 tokens/row, x0.95 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 f92585a6237b, cost $0.0971
  haiku-s2: V1 0.817 in [0.750, 0.870] ok; V2 94.9 tokens/row, x1.00 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 4f594f329586, cost $0.0989
  haiku-s3: V1 0.821 in [0.750, 0.870] ok; V2 91.4 tokens/row, x0.96 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 49bc8ce15fe6, cost $0.0975
  haiku-s4: V1 0.821 in [0.750, 0.870] ok; V2 94.4 tokens/row, x0.99 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 ae98bd73b046, cost $0.0987
  V3: agreement of s1 with the others minus among the rest: -0.5 points ok
[intent] V1-V3: ok
  haiku-s1: V1 0.819 in [0.750, 0.890] ok; V2 124.7 tokens/row, x1.14 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 0d8900ec86f2, cost $0.0569
  haiku-s2: V1 0.806 in [0.750, 0.890] ok; V2 126.4 tokens/row, x1.15 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 746ab82df7e7, cost $0.0571
  haiku-s3: V1 0.809 in [0.750, 0.890] ok; V2 120.1 tokens/row, x1.10 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 dcbe8df0364f, cost $0.0561
  haiku-s4: V1 0.816 in [0.750, 0.890] ok; V2 124.2 tokens/row, x1.13 of the design's ok; V4 ids ok, bad 0 (0.0%) ok, provenance ok; sha256 dd2075f1dccb, cost $0.0568
  V3: agreement of s1 with the others minus among the rest: +0.2 points ok
[faithfulness] V1-V3: ok
confirm spend recorded: $0.6206 of the $1.00 cap
lock file: 21 data files match their sha256
[intent] test rows per split [670.0], k at 5%/10%/20% = 34/67/134; R0 lift 1 to within 2.2e-16 on 1000 splits
[faithfulness] test rows per split [220.0], k at 5%/10%/20% = 11/22/44; R0 lift 1 to within 2.2e-16 on 1000 splits
guard: the release split and unlisted files are refused: OK
[intent] confirm_task on the dev Haiku file equals replay.load_task: OK
[faithfulness] confirm_task on the dev Haiku file equals replay.load_task: OK
```

| Data file | sha256 |
|---|---|
| intent/labels.json | `e64b29ebfc59f9bdbf4d4756fadd29669ab8fe623203b66790cce6d1e1b8f5f8` |
| intent/confirm.jsonl | `6ccee5cc9550152cd0c8ab298cc7f4ebb81a09f2a34e49f54d9fa972858899f5` |
| intent/dev.jsonl | `f8d3b90ef99131fd438caf91876766fcd66c1a48b80f7f9f32abc8c0523e9fa0` |
| labels/intent/confirm/haiku-s1.jsonl | `f92585a6237b1a86ea105aa2b57c9fd2f7367f4c7198b90ecccd9e9e8cc78cb6` |
| labels/intent/confirm/haiku-s1.cost.json | `3a4cc7fb627de3c11fba0d68242ffbc7865f35893a9385e9c6f626850ed1e9c2` |
| labels/intent/confirm/haiku-s2.jsonl | `4f594f329586b04464f8dfd6af188cf472768f4eea025b3d0a6014ffc315e267` |
| labels/intent/confirm/haiku-s2.cost.json | `0cae51b6af6207ccc971d7e64cdf01421b503748b3354be531d9b1323239259c` |
| labels/intent/confirm/haiku-s3.jsonl | `49bc8ce15fe6b4bb202c0833dd68f6ef04d24fc4c2fb8ffdb8a796323b34e29d` |
| labels/intent/confirm/haiku-s3.cost.json | `95eb129a20ef5c4996427bf8fd2e021de10f696741e21afc1e47c0e84585dde7` |
| labels/intent/confirm/haiku-s4.jsonl | `ae98bd73b0467621e53ce9be8dfb4043bec10508f90e4abd0862a0a9a62b6370` |
| labels/intent/confirm/haiku-s4.cost.json | `ee30d5b1f02424a25fb52fbd3b47be778b670cd1668dcdd9262b16cd746c0ac6` |
| faithfulness/confirm.jsonl | `6c8e98e0163ee7bceb1d9f7a2efca64f2aa1622edcc7f17a1a59b475cf3fcc9d` |
| faithfulness/dev.jsonl | `d0fb8fd50518e1d3fb0a6319738611998d5ce3646e22c90aada8220bf4a8f821` |
| labels/faithfulness/confirm/haiku-s1.jsonl | `0d8900ec86f243aec68df037e093c062342dd81008c050a386e5dbf38badea02` |
| labels/faithfulness/confirm/haiku-s1.cost.json | `3f500bcb300ab1fadae4304a23ecd7de481bb97f5fd6810ceaa66eac6920c92e` |
| labels/faithfulness/confirm/haiku-s2.jsonl | `746ab82df7e735a51d8d121cfae72adb9f45bc6cd9087463bf38097db3a309e0` |
| labels/faithfulness/confirm/haiku-s2.cost.json | `499805286e8203c2757ea7da5ce3de0b6903f878421ba81de2179c848f5cfdb6` |
| labels/faithfulness/confirm/haiku-s3.jsonl | `dcbe8df0364f35d17bdecbe96c5ecfb6eb2680235b57b90e191fa8d5bbed9f35` |
| labels/faithfulness/confirm/haiku-s3.cost.json | `47573e7f811a0650434da307389ae53115bc5dcf9b90cc8ac41d7dd5a044588d` |
| labels/faithfulness/confirm/haiku-s4.jsonl | `dd2075f1dccb47e2fdb79edd7766229716498099129fc484518a749f92267ead` |
| labels/faithfulness/confirm/haiku-s4.cost.json | `094530707e5179e08f6d8414b3e87f184787fdbd76f08cfb0527f9aae737f6f7` |

## intent

770 rows; 670 test rows per split; k = 34/67/134 at 0.05/0.10/0.20. Primary k fraction 0.10. Bootstrap B = 1000 over 770 row clusters, LB at q = 0.025.

### Confirmatory, original gold, k fraction 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| R0 | 18.0 | 1.00 |  | -14.0 |  | 0.10 | 5.54 |
| R1 | 32.1 | 1.78 |  | +0.0 |  | 0.18 | 3.12 |
| R2 | 53.9 | 2.98 | 2.41 | +21.8 | +10.6 | 0.30 | 1.86 |

Status: R2 **PASS**. Flags: none. SE of G (bootstrap): 6.0.

### k sensitivity (mean G, points)

| Ranker | 0.05 | 0.10 | 0.20 |
|---|---|---|---|
| R2 | +31.1 | +21.8 | +9.7 |

### Exploratory (never decisive), 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| E2 | 51.5 | 2.86 |  | +19.4 |  | 0.29 | 1.94 |
| E4 | 52.4 | 2.90 |  | +20.3 |  | 0.29 | 1.91 |
| E5 | 50.1 | 2.78 |  | +18.0 |  | 0.28 | 2.00 |
| E6 | 51.3 | 2.84 |  | +19.2 |  | 0.28 | 1.95 |
| E7 | 47.3 | 2.62 |  | +15.2 |  | 0.26 | 2.11 |
| E8 | 54.4 | 3.02 |  | +22.3 |  | 0.30 | 1.84 |

## faithfulness

320 rows; 220 test rows per split; k = 11/22/44 at 0.05/0.10/0.20. Primary k fraction 0.10. Bootstrap B = 1000 over 66 doc clusters, LB at q = 0.025.

### Confirmatory, original gold, k fraction 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| R0 | 18.2 | 1.00 |  | -0.8 |  | 0.10 | 5.49 |
| R1 | 19.0 | 1.04 |  | +0.0 |  | 0.10 | 5.26 |
| R2 | 35.0 | 1.92 | 1.16 | +16.0 | +1.8 | 0.19 | 2.86 |

Status: R2 **INCONCLUSIVE**. Flags: ['DOMAIN']. SE of G (bootstrap): 7.0.

### k sensitivity (mean G, points)

| Ranker | 0.05 | 0.10 | 0.20 |
|---|---|---|---|
| R2 | +18.5 | +16.0 | +8.0 |

### Exploratory (never decisive), 0.10

| Ranker | errors per 100 reviewed | L | LB(L) | G (pts) | LB(G) | recall | rows per error |
|---|---|---|---|---|---|---|---|
| E2 | 34.6 | 1.90 |  | +15.6 |  | 0.19 | 2.89 |
| E4 | 29.8 | 1.63 |  | +10.8 |  | 0.16 | 3.36 |
| E5 | 34.3 | 1.88 |  | +15.3 |  | 0.19 | 2.91 |
| E6 | 33.6 | 1.84 |  | +14.6 |  | 0.18 | 2.98 |
| E7 | 24.7 | 1.36 |  | +5.7 |  | 0.14 | 4.05 |
| E8 | 20.7 | 1.14 |  | +1.7 |  | 0.11 | 4.83 |

### G of R2 per domain at 0.10 (descriptive)

| Domain | G (pts) |
|---|---|
| billsum | +32.3 |
| ectsum | +10.5 |
| news | -3.7 |
| qmsumm | -9.6 |
| sales_call | -3.3 |
| sales_email | +8.8 |
| samsum | +19.0 |
| scitldr | +16.9 |


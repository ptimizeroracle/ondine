# Coverage on synthetic populations

135 cells × 2,000 draws, sample 200, seed 20261104, 1 s. Pass rule: coverage ≥ 94% in every cell (Monte Carlo band about ±1 point).

**PASS**: lowest 94.40% — faithfulness/haiku, original gold, N=1,000.

Shared-mistake rate, P(Opus agrees with the run | run wrong): intent/haiku, original 36%, intent/haiku, adjudicated 0%, intent/luna, original 33%, intent/luna, adjudicated 0%, faithfulness/haiku, original 69%, faithfulness/haiku, adjudicated 15%, faithfulness/luna, original 70%, faithfulness/luna, adjudicated 38%. Two-tier cells use the original-gold rate for haiku; the adjudicated gold was settled partly by model votes, Opus's among them.


## Real rows: plain sample and unresolved rows

| cell | accuracy | coverage | mean width |
|---|---:|---:|---:|
| intent/haiku, original gold, N=1,000 | 79.4% | 96.30% | 0.1035 |
| intent/haiku, original gold, N=1,000, 5% unresolved at random | 79.4% | 99.05% | 0.1545 |
| intent/haiku, original gold, N=1,000, 5% unresolved 5x among wrong | 79.4% | 99.80% | 0.1517 |
| intent/haiku, original gold, N=10,000 | 80.7% | 96.05% | 0.1120 |
| intent/haiku, original gold, N=10,000, 5% unresolved at random | 80.7% | 99.45% | 0.1645 |
| intent/haiku, original gold, N=10,000, 5% unresolved 5x among wrong | 80.7% | 99.50% | 0.1604 |
| intent/haiku, original gold, N=100,000 | 80.7% | 96.25% | 0.1131 |
| intent/haiku, original gold, N=100,000, 5% unresolved at random | 80.7% | 99.35% | 0.1654 |
| intent/haiku, original gold, N=100,000, 5% unresolved 5x among wrong | 80.7% | 99.70% | 0.1614 |
| intent/haiku, adjudicated gold, N=1,000 | 88.9% | 95.60% | 0.0815 |
| intent/haiku, adjudicated gold, N=1,000, 5% unresolved at random | 88.9% | 98.45% | 0.1363 |
| intent/haiku, adjudicated gold, N=1,000, 5% unresolved 5x among wrong | 88.9% | 99.95% | 0.1316 |
| intent/haiku, adjudicated gold, N=10,000 | 89.5% | 96.80% | 0.0883 |
| intent/haiku, adjudicated gold, N=10,000, 5% unresolved at random | 89.5% | 99.25% | 0.1436 |
| intent/haiku, adjudicated gold, N=10,000, 5% unresolved 5x among wrong | 89.5% | 99.75% | 0.1392 |
| intent/haiku, adjudicated gold, N=100,000 | 89.5% | 95.90% | 0.0894 |
| intent/haiku, adjudicated gold, N=100,000, 5% unresolved at random | 89.5% | 98.80% | 0.1446 |
| intent/haiku, adjudicated gold, N=100,000, 5% unresolved 5x among wrong | 89.5% | 99.90% | 0.1405 |
| intent/luna, original gold, N=1,000 | 79.7% | 96.10% | 0.1029 |
| intent/luna, original gold, N=1,000, 5% unresolved at random | 79.7% | 99.25% | 0.1545 |
| intent/luna, original gold, N=1,000, 5% unresolved 5x among wrong | 79.7% | 99.80% | 0.1507 |
| intent/luna, original gold, N=10,000 | 79.2% | 96.75% | 0.1152 |
| intent/luna, original gold, N=10,000, 5% unresolved at random | 79.2% | 99.00% | 0.1679 |
| intent/luna, original gold, N=10,000, 5% unresolved 5x among wrong | 79.2% | 99.70% | 0.1632 |
| intent/luna, original gold, N=100,000 | 79.0% | 95.95% | 0.1168 |
| intent/luna, original gold, N=100,000, 5% unresolved at random | 79.0% | 99.45% | 0.1685 |
| intent/luna, original gold, N=100,000, 5% unresolved 5x among wrong | 79.0% | 99.90% | 0.1651 |
| intent/luna, adjudicated gold, N=1,000 | 87.8% | 96.05% | 0.0849 |
| intent/luna, adjudicated gold, N=1,000, 5% unresolved at random | 87.8% | 99.70% | 0.1382 |
| intent/luna, adjudicated gold, N=1,000, 5% unresolved 5x among wrong | 87.8% | 99.65% | 0.1378 |
| intent/luna, adjudicated gold, N=10,000 | 87.7% | 95.90% | 0.0942 |
| intent/luna, adjudicated gold, N=10,000, 5% unresolved at random | 87.7% | 99.20% | 0.1484 |
| intent/luna, adjudicated gold, N=10,000, 5% unresolved 5x among wrong | 87.7% | 99.80% | 0.1442 |
| intent/luna, adjudicated gold, N=100,000 | 87.5% | 95.40% | 0.0956 |
| intent/luna, adjudicated gold, N=100,000, 5% unresolved at random | 87.5% | 98.15% | 0.1500 |
| intent/luna, adjudicated gold, N=100,000, 5% unresolved 5x among wrong | 87.5% | 99.75% | 0.1468 |
| faithfulness/haiku, original gold, N=1,000 | 82.6% | 94.40% | 0.0972 |
| faithfulness/haiku, original gold, N=1,000, 5% unresolved at random | 82.6% | 98.35% | 0.1503 |
| faithfulness/haiku, original gold, N=1,000, 5% unresolved 5x among wrong | 82.6% | 99.85% | 0.1463 |
| faithfulness/haiku, original gold, N=10,000 | 81.7% | 95.70% | 0.1099 |
| faithfulness/haiku, original gold, N=10,000, 5% unresolved at random | 81.7% | 99.10% | 0.1622 |
| faithfulness/haiku, original gold, N=10,000, 5% unresolved 5x among wrong | 81.7% | 99.80% | 0.1589 |
| faithfulness/haiku, original gold, N=100,000 | 81.8% | 96.15% | 0.1109 |
| faithfulness/haiku, original gold, N=100,000, 5% unresolved at random | 81.8% | 99.10% | 0.1633 |
| faithfulness/haiku, original gold, N=100,000, 5% unresolved 5x among wrong | 81.8% | 99.60% | 0.1592 |
| faithfulness/haiku, adjudicated gold, N=1,000 | 93.4% | 97.45% | 0.0656 |
| faithfulness/haiku, adjudicated gold, N=1,000, 5% unresolved at random | 93.4% | 99.60% | 0.1221 |
| faithfulness/haiku, adjudicated gold, N=1,000, 5% unresolved 5x among wrong | 93.4% | 99.85% | 0.1199 |
| faithfulness/haiku, adjudicated gold, N=10,000 | 93.4% | 96.50% | 0.0729 |
| faithfulness/haiku, adjudicated gold, N=10,000, 5% unresolved at random | 93.4% | 98.85% | 0.1291 |
| faithfulness/haiku, adjudicated gold, N=10,000, 5% unresolved 5x among wrong | 93.4% | 99.75% | 0.1263 |
| faithfulness/haiku, adjudicated gold, N=100,000 | 93.1% | 96.65% | 0.0745 |
| faithfulness/haiku, adjudicated gold, N=100,000, 5% unresolved at random | 93.1% | 98.65% | 0.1320 |
| faithfulness/haiku, adjudicated gold, N=100,000, 5% unresolved 5x among wrong | 93.1% | 99.85% | 0.1279 |
| faithfulness/luna, original gold, N=1,000 | 80.1% | 96.20% | 0.1020 |
| faithfulness/luna, original gold, N=1,000, 5% unresolved at random | 80.1% | 99.65% | 0.1535 |
| faithfulness/luna, original gold, N=1,000, 5% unresolved 5x among wrong | 80.1% | 99.85% | 0.1512 |
| faithfulness/luna, original gold, N=10,000 | 79.0% | 95.60% | 0.1156 |
| faithfulness/luna, original gold, N=10,000, 5% unresolved at random | 79.0% | 99.45% | 0.1674 |
| faithfulness/luna, original gold, N=10,000, 5% unresolved 5x among wrong | 79.0% | 99.55% | 0.1641 |
| faithfulness/luna, original gold, N=100,000 | 78.4% | 96.65% | 0.1178 |
| faithfulness/luna, original gold, N=100,000, 5% unresolved at random | 78.4% | 99.20% | 0.1694 |
| faithfulness/luna, original gold, N=100,000, 5% unresolved 5x among wrong | 78.4% | 99.70% | 0.1654 |
| faithfulness/luna, adjudicated gold, N=1,000 | 86.9% | 97.30% | 0.0871 |
| faithfulness/luna, adjudicated gold, N=1,000, 5% unresolved at random | 86.9% | 99.25% | 0.1409 |
| faithfulness/luna, adjudicated gold, N=1,000, 5% unresolved 5x among wrong | 86.9% | 99.85% | 0.1376 |
| faithfulness/luna, adjudicated gold, N=10,000 | 86.9% | 96.10% | 0.0967 |
| faithfulness/luna, adjudicated gold, N=10,000, 5% unresolved at random | 86.9% | 99.10% | 0.1510 |
| faithfulness/luna, adjudicated gold, N=10,000, 5% unresolved 5x among wrong | 86.9% | 99.95% | 0.1465 |
| faithfulness/luna, adjudicated gold, N=100,000 | 86.8% | 96.80% | 0.0981 |
| faithfulness/luna, adjudicated gold, N=100,000, 5% unresolved at random | 86.8% | 99.10% | 0.1519 |
| faithfulness/luna, adjudicated gold, N=100,000, 5% unresolved 5x among wrong | 86.8% | 99.75% | 0.1475 |

## Rows with no defensible label, as unresolved

| cell | accuracy | coverage | mean width |
|---|---:|---:|---:|
| intent/haiku, 49 dropped rows unresolved, N=1,000 | 85.6% | 96.55% | 0.1322 |
| intent/haiku, 527 dropped rows unresolved, N=10,000 | 84.8% | 96.15% | 0.1464 |
| intent/haiku, 5681 dropped rows unresolved, N=100,000 | 84.6% | 96.75% | 0.1514 |
| intent/luna, 51 dropped rows unresolved, N=1,000 | 82.0% | 96.10% | 0.1431 |
| intent/luna, 518 dropped rows unresolved, N=10,000 | 83.6% | 94.95% | 0.1491 |
| intent/luna, 5588 dropped rows unresolved, N=100,000 | 82.8% | 95.05% | 0.1560 |

## Strata (stress population)

| cell | accuracy | coverage | mean width | unstratified width |
|---|---:|---:|---:|---:|
| accepted 90.0% + repaired 5% at 70% + unverified 1% at 50%, N=1,000, equal split | 86.3% | 99.40% | 0.1180 | 0.0887 |
| accepted 90.0% + repaired 5% at 70% + unverified 1% at 50%, N=1,000, weight split | 86.3% | 99.55% | 0.1115 | 0.0887 |
| accepted 90.0% + repaired 5% at 70% + unverified 1% at 50%, N=10,000, equal split | 88.6% | 99.85% | 0.1390 | 0.0912 |
| accepted 90.0% + repaired 5% at 70% + unverified 1% at 50%, N=10,000, weight split | 88.6% | 99.55% | 0.1273 | 0.0915 |
| accepted 90.0% + repaired 5% at 70% + unverified 1% at 50%, N=100,000, equal split | 88.7% | 99.95% | 0.1397 | 0.0915 |
| accepted 90.0% + repaired 5% at 70% + unverified 1% at 50%, N=100,000, weight split | 88.7% | 99.75% | 0.1281 | 0.0919 |
| accepted 97.0% + repaired 5% at 70% + unverified 1% at 50%, N=1,000, equal split | 95.0% | 99.70% | 0.0728 | 0.0581 |
| accepted 97.0% + repaired 5% at 70% + unverified 1% at 50%, N=1,000, weight split | 95.0% | 99.75% | 0.0712 | 0.0578 |
| accepted 97.0% + repaired 5% at 70% + unverified 1% at 50%, N=10,000, equal split | 95.1% | 99.95% | 0.0953 | 0.0640 |
| accepted 97.0% + repaired 5% at 70% + unverified 1% at 50%, N=10,000, weight split | 95.1% | 99.85% | 0.0906 | 0.0639 |
| accepted 97.0% + repaired 5% at 70% + unverified 1% at 50%, N=100,000, equal split | 95.2% | 99.95% | 0.0960 | 0.0635 |
| accepted 97.0% + repaired 5% at 70% + unverified 1% at 50%, N=100,000, weight split | 95.2% | 100.00% | 0.0916 | 0.0638 |
| accepted 99.5% + repaired 5% at 70% + unverified 1% at 50%, N=1,000, equal split | 97.5% | 100.00% | 0.0495 | 0.0431 |
| accepted 99.5% + repaired 5% at 70% + unverified 1% at 50%, N=1,000, weight split | 97.5% | 100.00% | 0.0501 | 0.0433 |
| accepted 99.5% + repaired 5% at 70% + unverified 1% at 50%, N=10,000, equal split | 97.6% | 99.90% | 0.0637 | 0.0467 |
| accepted 99.5% + repaired 5% at 70% + unverified 1% at 50%, N=10,000, weight split | 97.6% | 100.00% | 0.0629 | 0.0469 |
| accepted 99.5% + repaired 5% at 70% + unverified 1% at 50%, N=100,000, equal split | 97.5% | 100.00% | 0.0657 | 0.0481 |
| accepted 99.5% + repaired 5% at 70% + unverified 1% at 50%, N=100,000, weight split | 97.5% | 100.00% | 0.0654 | 0.0478 |

## Two-tier: model on the sample, people on part of it

| cell | accuracy | coverage | mean width | people-only width |
|---|---:|---:|---:|---:|
| intent shared mistakes 36%, accuracy 90.0%, disagreement 10%, 25 people labels, N=10,000 | 90.1% | 100.00% | 0.4480 | 0.2578 |
| intent shared mistakes 36%, accuracy 90.0%, disagreement 10%, 50 people labels, N=10,000 | 90.1% | 100.00% | 0.2940 | 0.1789 |
| intent shared mistakes 36%, accuracy 90.0%, disagreement 10%, 100 people labels, N=10,000 | 90.1% | 100.00% | 0.2157 | 0.1248 |
| intent shared mistakes 36%, accuracy 97.0%, disagreement 3%, 25 people labels, N=10,000 | 97.1% | 100.00% | 0.2840 | 0.1782 |
| intent shared mistakes 36%, accuracy 97.0%, disagreement 3%, 50 people labels, N=10,000 | 97.1% | 100.00% | 0.1760 | 0.1143 |
| intent shared mistakes 36%, accuracy 97.0%, disagreement 3%, 100 people labels, N=10,000 | 97.1% | 100.00% | 0.1277 | 0.0746 |
| intent shared mistakes 36%, accuracy 97.0%, disagreement 10%, 25 people labels, N=10,000 | 96.8% | 99.95% | 0.3786 | 0.1839 |
| intent shared mistakes 36%, accuracy 97.0%, disagreement 10%, 50 people labels, N=10,000 | 96.8% | 100.00% | 0.2166 | 0.1167 |
| intent shared mistakes 36%, accuracy 97.0%, disagreement 10%, 100 people labels, N=10,000 | 96.8% | 100.00% | 0.1416 | 0.0777 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 1%, 25 people labels, N=10,000 | 99.4% | 99.95% | 0.2263 | 0.1465 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 1%, 50 people labels, N=10,000 | 99.4% | 100.00% | 0.1356 | 0.0796 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 1%, 100 people labels, N=10,000 | 99.4% | 100.00% | 0.0908 | 0.0455 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 3%, 25 people labels, N=10,000 | 99.5% | 99.95% | 0.2583 | 0.1444 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 3%, 50 people labels, N=10,000 | 99.5% | 100.00% | 0.1450 | 0.0786 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 3%, 100 people labels, N=10,000 | 99.5% | 100.00% | 0.0949 | 0.0438 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 10%, 25 people labels, N=10,000 | 99.5% | 99.95% | 0.3357 | 0.1446 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 10%, 50 people labels, N=10,000 | 99.5% | 99.95% | 0.1687 | 0.0794 |
| intent shared mistakes 36%, accuracy 99.5%, disagreement 10%, 100 people labels, N=10,000 | 99.5% | 99.90% | 0.0942 | 0.0446 |
| faithfulness shared mistakes 69%, accuracy 90.0%, disagreement 10%, 25 people labels, N=10,000 | 89.9% | 99.90% | 0.4613 | 0.2616 |
| faithfulness shared mistakes 69%, accuracy 90.0%, disagreement 10%, 50 people labels, N=10,000 | 89.9% | 99.90% | 0.3043 | 0.1820 |
| faithfulness shared mistakes 69%, accuracy 90.0%, disagreement 10%, 100 people labels, N=10,000 | 89.9% | 100.00% | 0.2152 | 0.1251 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 1%, 25 people labels, N=10,000 | 97.3% | 99.85% | 0.2492 | 0.1763 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 1%, 50 people labels, N=10,000 | 97.3% | 100.00% | 0.1610 | 0.1124 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 1%, 100 people labels, N=10,000 | 97.3% | 100.00% | 0.1148 | 0.0725 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 3%, 25 people labels, N=10,000 | 96.8% | 99.90% | 0.3021 | 0.1813 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 3%, 50 people labels, N=10,000 | 96.8% | 100.00% | 0.1900 | 0.1166 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 3%, 100 people labels, N=10,000 | 96.8% | 100.00% | 0.1379 | 0.0782 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 10%, 25 people labels, N=10,000 | 97.0% | 99.85% | 0.3724 | 0.1821 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 10%, 50 people labels, N=10,000 | 97.0% | 99.90% | 0.2119 | 0.1149 |
| faithfulness shared mistakes 69%, accuracy 97.0%, disagreement 10%, 100 people labels, N=10,000 | 97.0% | 100.00% | 0.1361 | 0.0765 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 1%, 25 people labels, N=10,000 | 99.5% | 99.50% | 0.2262 | 0.1457 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 1%, 50 people labels, N=10,000 | 99.5% | 100.00% | 0.1354 | 0.0798 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 1%, 100 people labels, N=10,000 | 99.5% | 99.85% | 0.0897 | 0.0449 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 3%, 25 people labels, N=10,000 | 99.5% | 99.75% | 0.2567 | 0.1450 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 3%, 50 people labels, N=10,000 | 99.5% | 99.95% | 0.1452 | 0.0790 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 3%, 100 people labels, N=10,000 | 99.5% | 100.00% | 0.0945 | 0.0441 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 10%, 25 people labels, N=10,000 | 99.4% | 99.95% | 0.3355 | 0.1469 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 10%, 50 people labels, N=10,000 | 99.4% | 99.95% | 0.1699 | 0.0815 |
| faithfulness shared mistakes 69%, accuracy 99.5%, disagreement 10%, 100 people labels, N=10,000 | 99.4% | 99.95% | 0.0965 | 0.0465 |

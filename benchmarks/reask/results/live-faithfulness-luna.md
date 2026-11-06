# Re-ask, live: faithfulness, openai/gpt-6-luna, loosely worded prompt

200 rows in 151 s; statuses {'repaired': 200}.

| | rows |
|---|---:|
| first answer failed a check | 175 |
| asked again | 175 |
| fixed by the re-ask | 175 |
| fixed and right against the labels | 142 |
| still failing after the re-ask | 0 |

Cost: first answers $0.0260, re-asks $0.0199.

```
Certificate: accuracy against your labels, on every delivered row
  Basis:  every delivered row has a label

  answer (labels)
    Accuracy: 78.5% (exact: every delivered row compared)
    Compared: 200 of 200 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    By status: repaired 78.5% (157/200)
    By model: openai/gpt-6-luna 78.5% (157/200)
    By value:
      consistent 75.7% (78/103)
      inconsistent 81.4% (79/97)
    Confusions: consistent → inconsistent (25); inconsistent → consistent (18)
    Wrong rows (43): 2, 3, 13, 20, 22, 23, 24, 25, 26, 28 … 33 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run 8254cbb4…, delivered values f212a787…, contract ea8898da…
```

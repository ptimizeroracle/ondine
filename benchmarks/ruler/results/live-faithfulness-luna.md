# Live certificate: faithfulness, openai/gpt-6-luna

200 rows in 70 s, cost $0.0282; statuses {'accepted': 200}.

## Census (labels for every delivered row)

```
Certificate: accuracy against your labels, on every delivered row
  Basis:  every delivered row has a label

  answer (labels)
    Accuracy: 78.5% (exact: every delivered row compared)
    Compared: 200 of 200 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    By status: accepted 78.5% (157/200)
    By model: openai/gpt-6-luna 78.5% (157/200)
    By value:
      inconsistent 76.1% (89/117)
      consistent 81.9% (68/83)
    Confusions: inconsistent → consistent (28); consistent → inconsistent (15)
    Wrong rows (43): 2, 3, 5, 6, 13, 18, 20, 24, 25, 26 … 33 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run 087e0a87…, delivered values b0632539…, contract 8773a142…
```

## Sample of 100 drawn by Ondine

```
Certificate: accuracy against your labels, on a sample Ondine drew
  Basis:  sample of 100 of 200 delivered rows (100 rows)

  answer (labels)
    Accuracy: 76.0% (95% range 69.5%–81.5%)
    Compared: 100 of 200 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    By status: accepted 76.0% [69.5%–81.5%] (n=100)
    By model: openai/gpt-6-luna 76.0% [69.5%–81.5%] (n=100)
    By value:
      inconsistent 74.1% [65.0%–81.2%] (n=58)
      consistent 78.6% [67.5%–86.7%] (n=42)
    Confusions: inconsistent → consistent (15); consistent → inconsistent (9)
    Wrong rows (24): 2, 3, 5, 20, 24, 25, 26, 30, 31, 32 … 14 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run 087e0a87…, delivered values b0632539…, contract 8773a142…, seed 15457382742178146380
```

# Live audit: intent, openrouter/openai/gpt-oss-20b audited by openai/gpt-6-luna

Run: 231 rows in 649 s, $0.0101. Audit: 54 s, $0.0155 (estimated at most $0.2345).

| measure | value |
|---|---:|
| accuracy on every row (census against the labels) | 80.4% |
| accuracy on the sample (against the labels) | 79.5% (77.4%–81.3%) |
| agreement with openai/gpt-6-luna on the same sample (audit) | 90.5% (88.7%–91.7%) |
| gap on the sample (agreement − accuracy) | +11.0 points |
| sampled rows the audit flags | 19 |
| of which really wrong (labels) | 13 |
| really wrong sampled rows it missed | 28 of 41 |
| unresolved in the audit | 0 |

## Estimate

```
Audit estimate: openai/gpt-6-luna (litellm)
  Sends: 200 rows; columns text
  Price: at most $0.2345 (estimate errs high; the audit stops at its max_cost)
```

## Audit

```
Audit: agreement with openai/gpt-6-luna, on a sample Ondine drew — an audit, not accuracy
  Basis:  sample of 200 of 230 delivered rows (200 rows)
  Warning: an audit, not accuracy: models share mistakes, across vendors too, so agreement can overstate accuracy (by 4 to 12.5 points on our test sets)
  Reference: openai/gpt-6-luna, cost $0.0155

  answer (labels)
    Agreement: 90.5% (95% range 88.7%–91.7%)
    Compared: 200 of 230 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    Not delivered, not certified: 1 failed
    By status: accepted 90.5% [88.7%–91.7%] (n=200)
    By model: openrouter/openai/gpt-oss-20b 90.5% [88.7%–91.7%] (n=200)
    By value:
      transfer_timing 50.0% (3/6, few rows)
      declined_card_payment 83.3% (5/6, few rows)
      lost_or_stolen_card 75.0% [60.0%–80.0%] (n=4, few rows)
      failed_transfer 100.0% [80.0%–100.0%] (n=4, few rows)
      card_delivery_estimate 80.0% (4/5, few rows)
      … 69 more values
    Confusions: reverted_card_payment? → declined_card_payment (2); transfer_timing → automatic_top_up (1); transfer_not_received_by_recipient → receiving_money (1); lost_or_stolen_card → activate_my_card (1); compromised_card → card_payment_not_recognised (1)
    Wrong rows (19): 16, 18, 43, 52, 59, 73, 76, 80, 118, 133 … 9 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run 08d1c726…, delivered values b3aeda50…, contract 02a87c9a…, seed 15009900015146600195
```

## Certificate

```
Certificate: accuracy against your labels, on every delivered row
  Basis:  every delivered row has a label

  answer (labels)
    Accuracy: 80.4% (exact: every delivered row compared)
    Compared: 230 of 230 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    Not delivered, not certified: 1 failed
    By status: accepted 80.4% (185/230)
    By model: openrouter/openai/gpt-oss-20b 80.4% (185/230)
    By value:
      transfer_timing 33.3% (2/6)
      declined_card_payment 50.0% (3/6)
      lost_or_stolen_card 60.0% (3/5)
      failed_transfer 60.0% (3/5)
      card_delivery_estimate 60.0% (3/5)
      … 69 more values
    Confusions: compromised_card → card_payment_not_recognised (3); declined_card_payment → declined_transfer (3); failed_transfer → beneficiary_not_allowed (2); change_pin → get_physical_card (2); transfer_into_account → top_up_by_bank_transfer_charge (2)
    Wrong rows (45): 4, 16, 18, 25, 26, 34, 37, 43, 49, 51 … 35 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run 08d1c726…, delivered values b3aeda50…, contract 02a87c9a…
```

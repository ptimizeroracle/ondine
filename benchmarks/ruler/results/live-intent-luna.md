# Live certificate: intent, openai/gpt-6-luna

231 rows in 65 s, cost $0.0177; statuses {'accepted': 231}.

## Census (labels for every delivered row)

```
Certificate: accuracy against your labels, on every delivered row
  Basis:  every delivered row has a label

  answer (labels)
    Accuracy: 81.4% (exact: every delivered row compared)
    Compared: 231 of 231 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    By status: accepted 81.4% (188/231)
    By model: openai/gpt-6-luna 81.4% (188/231)
    By value:
      receiving_money 37.5% (3/8)
      declined_card_payment 28.6% (2/7)
      failed_transfer 60.0% (3/5)
      country_support 60.0% (3/5)
      change_pin 60.0% (3/5)
      … 68 more values
    Confusions: declined_card_payment → declined_transfer (3); failed_transfer → beneficiary_not_allowed (2); compromised_card → card_payment_not_recognised (2); change_pin → get_physical_card (2); get_physical_card → order_physical_card (2)
    Wrong rows (43): 4, 18, 25, 26, 34, 37, 43, 49, 51, 52 … 33 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run df6d2f0d…, delivered values 1512d6d5…, contract 67b678fe…
```

## Sample of 100 drawn by Ondine

```
Certificate: accuracy against your labels, on a sample Ondine drew
  Basis:  sample of 100 of 231 delivered rows (100 rows)

  answer (labels)
    Accuracy: 80.0% (95% range 73.2%–85.3%)
    Compared: 100 of 231 delivered rows; 0 unresolved; compared as allowed values, after the contract's normalisation and aliases
    By status: accepted 80.0% [73.2%–85.3%] (n=100)
    By model: openai/gpt-6-luna 80.0% [73.2%–85.3%] (n=100)
    By value:
      receiving_money 20.0% [12.5%–50.0%] (n=5, few rows)
      declined_card_payment 0.0% [0.0%–71.4%] (n=2, few rows)
      failed_transfer 75.0% [60.0%–80.0%] (n=4, few rows)
      country_support — (no resolved row of 5)
      change_pin 100.0% [20.0%–100.0%] (n=1, few rows)
      … 68 more values
    Confusions: get_physical_card → order_physical_card (2); receiving_money → verify_source_of_funds (2); receiving_money → balance_not_updated_after_bank_transfer (1); failed_transfer → beneficiary_not_allowed (1); card_delivery_estimate → card_arrival (1)
    Wrong rows (20): 18, 25, 37, 49, 52, 70, 80, 84, 98, 133 … 10 more
  Method: exact hypergeometric 95% range; unresolved rows counted wrong for the low end, right for the high end
  Bound to: run df6d2f0d…, delivered values 1512d6d5…, contract 67b678fe…, seed 15360316605594468081
```

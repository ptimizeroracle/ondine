# Early stop, live: faithfulness, openai/gpt-6-luna

Default policy (stop above 50% of the first 50 rows), `re_ask=1`.

| prompt | stopped | rows asked | calls | cost | time | statuses |
|---|---|---:|---:|---:|---:|---|
| loose | yes | 50 of 200 | 50 | $0.0128 | 40 s | {'repaired': 3, 'rejected': 47, 'failed': 150} |
| strict | no | 200 of 200 | 200 | $0.0281 | 86 s | {'accepted': 200} |

Error raised by the loose run (150 rows never asked, stop reason: early stop):

```
Early stop: 47 of the first 50 rows were rejected on their first answer (94%), above 50%. The prompt or the contract is likely broken.
Most common: 47 x answer.allowed_values (e.g. answer: 'The substantive summary is accurate: the…' is not one of the 2 allowed values).
Fix the prompt or the contract and run again.
To finish this run anyway: pipeline.execute(resume_from=UUID('a4442278-09b1-482f-be9c-f26955eac983')).
To turn early stop off: with_contract_policy(early_stop_fraction=None).
```

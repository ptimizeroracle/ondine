# Escalation, live: faithfulness (loose prompt), openrouter/openai/gpt-oss-20b -> openai/gpt-6-luna

200 rows; `re_ask=0` in all three runs; early stop off.
Right: delivered value equals the dataset's label (rejected rows count
as wrong).

| run | delivered | right | accuracy | escalated (right) | cost | time | statuses | repair methods |
|---|---:|---:|---:|---:|---:|---:|---|---|
| weak | 75 | 59 | 29.5% | 0 (0) | $0.0204 | 341 s | {'rejected': 125, 'repaired': 75} | {'none': 125, 'format': 75} |
| weak + escalation | 200 | 169 | 84.5% | 135 (119) | $0.0486 | 1134 s | {'repaired': 200} | {'format': 65, 'escalate': 135} |
| strong | 12 | 7 | 3.5% | 0 (0) | $0.0260 | 173 s | {'rejected': 188, 'repaired': 12} | {'none': 188, 'format': 12} |

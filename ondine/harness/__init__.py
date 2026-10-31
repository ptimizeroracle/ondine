"""The data-quality harness: what Ondine guarantees about every delivered value.

Three parts, each owning one kind of knowledge:

* ``contract``: what the client declared a deliverable value to be (types,
  rules, run policy), validated before anything is spent.
* ``judge``: turns one raw model answer into a verdict against the contract.
  Deterministic, no model calls, never raises on bad model output.
* ``ledger``: the single record of what happened to every input row, and the
  one place a row's status is decided.

With no contract declared, the harness is "unchecked" and the pipeline's
output is identical to Ondine without it.
"""

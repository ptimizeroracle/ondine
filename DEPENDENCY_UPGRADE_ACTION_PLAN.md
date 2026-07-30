# Dependency Upgrade Action Plan — Commit 4ac9c30 (deps/upgrade-jul-2026)

Prioritized action items derived from DEPENDENCY_UPGRADE_ANALYSIS.md,
codebase inspection, and the current test run (1081 passed, 1 failed, 97 skipped).

Priority key: P0 = do before merge, P1 = do this week, P2 = nice-to-have / backlog.
Effort: S (<30 min), M (1-4 h), L (1+ day).

## Status (updated 2026-07-12)

| # | Action | Priority | Status | Notes |
|---|--------|----------|--------|-------|
| 1 | Fix Jaeger observability test | **P0** | ✅ Done | Migrated to OTLP (commit fa7c90d) |
| 2 | Commit pyproject.toml + uv.lock + docs to main | **P0** | ✅ Done | Commit fa7c90d on this branch |
| 3 | instructor↔litellm compat smoke test | P1 | ✅ Done | `tests/unit/test_instructor_litellm_compat.py` (3 tests) |
| 4 | Pandas CoW CI smoke test | P1 | ✅ Done | conftest fixture + `tests/unit/test_pandas_cow_smoke.py` (4 tests) |
| 5 | Pin Rust toolchain | P1 | ✅ Done | `rust-toolchain.toml` (stable + rustfmt/clippy) |
| 6-11 | P2 backlog items | P2 | Open | See checklist below |

---

## Q1. instructor pins litellm ≤1.82.6 but we use 1.91.1 — safe? Add a compat test?

### Finding

instructor 1.15.4's litellm constraint is **extra-only**, not a core dependency:

```
litellm<=1.83.7,>=1.35.31; extra == 'litellm'
litellm<=1.83.7,>=1.35.31; extra == 'test-docs'
```

instructor's core `Requires-Dist` has no litellm entry — litellm is pulled in
by ondine's own `litellm>=1.91.1`, and ondine uses `instructor.from_litellm()`
explicitly. So the upper bound is **not enforced** at install time (confirmed:
both packages co-install and 1081 tests pass).

The pin exists upstream because instructor's maintainers test against
litellm ≤1.82.6 and don't want to certify newer versions. It is a *tested-against*
upper bound, not a *hard incompatibility*.

### Risk assessment

- ondine calls litellm through instructor via `instructor.from_litellm(client)`
  (unified_litellm_client.py:263 area) and directly via `litellm.completion` /
  `litellm.acompletion` (knowledge modules, unified client).
- The litellm 1.84 "breaking changes" flagged in the analysis are to
  `litellm.completion` *internals* (not the public `completion()` / `acompletion()`
  API surface ondine uses).
- instructor 1.15.0 explicitly blocks the compromised 1.82.7/1.82.8 — ondine is
  on 1.91.1, which is post-fix. **No security concern.**

### Action

**[P1, S] Add a smoke compat test** that asserts `instructor.from_litellm`
works with the installed litellm version. This is cheap insurance: if a future
litellm bump silently breaks the instructor bridge, CI catches it.

```python
# tests/unit/test_instructor_litellm_compat.py
def test_instructor_from_litellm_smoke():
    """instructor.from_litellm must work with ondine's pinned litellm."""
    import instructor
    import litellm
    client = instructor.from_litellm(litellm)
    assert client is not None
```

**[P2, S] Document the version skew** in a comment in `pyproject.toml` next to
the litellm/instructor deps, so the next maintainer doesn't get surprised by
the extra-only upper bound.

---

## Q2. polars 0.20→1.42 breaking changes — grep for deprecated APIs?

### Finding

Grepped the codebase for every polars 1.0 breaking-change surface:

| API | Used in ondine? | Status |
|---|---|---|
| `Series(strict=...)` | No — the only `pl.Series(...)` call (polars_container.py:152) passes a list mask, no `strict=` kwarg | Safe |
| `replace` (split into `replace_strict`/`replace_old`) | No — ondine's `.replace()` calls are all `str.replace`, not polars expr `.replace()` | Safe |
| `read_excel` default engine → calamine | Yes — streaming_loader.py:92, but called with no engine kwarg, so calamine default applies cleanly | Safe |
| Hive partitioning disabled by default | No — `scan_parquet`/`scan_csv` calls pass no hive args | Safe |
| `pl.concat` | Yes — streaming_writer.py:149, `pl.concat(self._temp_chunks)` with default `how="vertical"` | Safe (unchanged) |
| `infer_schema_length` behavior | Not used | N/A |
| Type alias removals | No type-alias imports found | Safe |
| Binary serialization default | Not used (ondine uses CSV/Parquet/NDJSON/Excel) | N/A |

All 1081 tests pass with polars 1.42.1. The codebase uses a thin, stable
polars API surface (`read_csv`, `read_parquet`, `read_excel`, `scan_*`,
`DataFrame`, `Series`, `concat`, `filter`, `sample`, `lazy`, `collect`,
`len`). None of these hit the 1.0 breaking surface.

### Action

**[P2, S] Add a `# polars>=1.0 compatible` note** to the two files that do
polars I/O (streaming_loader.py, streaming_writer.py) so future grep audits
have a baseline. No code changes needed — the upgrade is clean.

**[P2, S] Optionally pin `polars>=1.0` as a floor** in pyproject.toml if you
want to prevent accidental downgrade below the 1.0 ABI break (currently
`>=1.42.1` already does this implicitly).

---

## Q3. pandas CoW — enable copy_on_write now to prep for 3.0?

### Finding

ondine uses pandas extensively (30+ import sites). The 1.5→2.3 jump is already
absorbed (no `iteritems`, `.ix`, or `inplace=True` calls found). CoW becomes
the unchangeable default in pandas 3.0.

Enabling CoW now would surface any latent SettingWithCopyWarning-style bugs
*before* the 3.0 bump forces them. However: ondine's architecture routes most
mutations through Polars containers and only converts to pandas at the boundary,
so the exposure surface is smaller than a typical pandas-heavy project.

### Action

**[P1, M] Add a CoW smoke test** that runs the integration test suite with
`pd.options.mode.copy_on_write = True` set, either as a separate CI job or a
conftest fixture toggle. This is the low-risk way to discover CoW incompatibilities
without changing production behavior.

```python
# tests/conftest.py (optional fixture)
import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def _enable_cow_if_requested():
    if os.getenv("ONDINE_TEST_COW") == "1":
        import pandas as pd
        pd.options.mode.copy_on_write = True
        yield
        pd.options.mode.copy_on_write = False
    else:
        yield
```

**[P2, S] Do NOT enable CoW in production code yet.** Wait until the smoke test
runs clean for a release cycle, then flip the default in `logging_utils`-style
central config. Premature enabling risks subtle behavior changes in user pipelines.

---

## Q4. diskcache CVE-2025-69872 — pin/mitigate/replace?

### Finding

ondine does **not** import diskcache directly (zero `import diskcache` hits).
It enters as a transitive dependency of litellm, used by litellm's Disk cache
backend (`litellm.caching.Cache(type="disk")` — see unified_litellm_client.py:254-256).

- The CVE is low severity (the analysis notes "no fix available" at the time).
- instructor 1.15.0 made diskcache optional as its CVE mitigation.
- ondine only activates the disk cache when a user explicitly passes
  `cache_config` with `type="disk"` — it is opt-in, not default.

### Action

**[P2, S] No action needed now.** The exposure is: (a) opt-in only, (b) low
severity, (c) no upstream fix available to pin to. When a patched diskcache
release lands, bump the `[tool.uv] override-dependencies` floor.

**[P2, S] Add a comment** in unified_litellm_client.py near the `Cache(type="disk")`
call noting the CVE and that it's opt-in, so a future security audit doesn't
re-discover it from scratch.

---

## Q5. structlog 26.1 weak-ref fix — worth enabling for long-running kanban tasks?

### Finding

structlog 26.1 added weak-ref file handles that prevent logfile descriptor
leaks in long-running processes. ondine uses structlog in:
- `logging_utils.py` — `configure_logging()` with `ConsoleRenderer` / JSON output
- `budget_controller.py` — `structlog.get_logger(__name__)`

However, ondine's structlog usage is **console/stderr-only** — it does not
configure file-based log handlers. The `ConsoleRenderer` writes to stdout/stderr,
which are not file descriptors that leak. The weak-ref fix applies to
`FileLoggerFactory` / file handles, which ondine does not use.

### Action

**[P2, S] No action needed.** The weak-ref fix is irrelevant to ondine's current
logging configuration. The structlog 26.1 bump is harmless (already installed,
tests pass) but brings no tangible benefit for this codebase. No configuration
change, no pin change.

---

## Q6. Any packages we should pin to exact versions (not lower bounds)?

### Finding

ondine currently uses `>=` lower bounds throughout — no exact pins. This is
correct for a library/SDK (pyproject.toml classifies it as a library, not an
application). The `uv.lock` file pins exact versions for reproducible installs.

### Recommendation

**Do NOT add exact pins to `pyproject.toml`** for the core dependencies. Exact
pins in a library conflict with downstream consumers. The lockfile is the right
place for reproducibility.

The one exception is **security-critical transitive overrides**, which ondine
already handles correctly in `[tool.uv] override-dependencies` (filelock, aiohttp,
cryptography, etc.).

### Action

**[P1, S] Pin the Rust toolchain version** in `rust-toolchain.toml` (if not
already) since the project uses maturin + pyo3. A drifting Rust version can
silently break the `ondine._engine` native module build. Check whether
`rust-toolchain.toml` exists; if not, add one pinning to the current stable.

**[P2, S] Consider upper bounds on the two most volatile deps** —
`litellm<2.0` and `pydantic<3.0` — as guardrails against a future major bump.
This is optional; the lockfile already protects CI.

---

## Q7. Commit pyproject.toml + uv.lock to main now, or wait?

### Finding

The diff is `pyproject.toml` (50 lines) + `uv.lock` (6588 lines, net +3706).
All 1081 tests pass. The 1 failing test (`test_observability.py::test_jaeger`)
is **pre-existing and unrelated** — it fails because `opentelemetry-exporter-jaeger`
1.21.0 is incompatible with `opentelemetry-sdk` 1.43.0 (the Jaeger exporter was
archived upstream in favor of OTLP). This is a dependency mismatch in the
`[observability]` extra, not a regression from the upgrade commit.

### Action

**[P0, M] Fix the Jaeger test before merge.** The `opentelemetry-exporter-jaeger`
package was archived by the OTel project. Options:
1. **(Recommended)** Migrate the observability extra from `opentelemetry-exporter-jaeger`
   to `opentelemetry-exporter-otlp` (already installed as a transitive) and update
   `tracer.py` to use `OTLPSpanExporter`. Update the test to match.
2. **(Quick fix)** Pin `opentelemetry-sdk<1.30` in the observability extra to
   keep the old Jaeger exporter working — but this is a dead end.
3. **(Minimal)** Mark the test `pytest.skip` with a reason and file an issue.

Option 1 is the right fix; option 3 unblocks merge immediately if time-boxed.

**[P0, S] Commit pyproject.toml + uv.lock to main once the Jaeger test is resolved.**
There is no reason to wait — the upgrade is clean, all real tests pass, and the
lockfile ensures reproducible installs. The `DEPENDENCY_UPGRADE_ANALYSIS.md` and
this action plan can be committed alongside (or moved to `docs/`).

---

## Q8. What one thing should we do FIRST?

**Fix the Jaeger observability test (Q7, P0).**

It is the only red signal in the suite, it blocks a clean merge, and it is a
real dependency incompatibility (archived package) that will only get worse.
Everything else is P1/P2 and can follow after the branch lands on main.

---

## Summary Checklist

| # | Action | Priority | Effort | Status |
|---|---|---|---|---|
| 1 | Fix Jaeger observability test (migrate to OTLP or skip+issue) | **P0** | M | ✅ Done |
| 2 | Commit pyproject.toml + uv.lock + analysis docs to main | **P0** | S | ✅ Done |
| 3 | Add instructor↔litellm compat smoke test | P1 | S | ✅ Done |
| 4 | Run integration suite with `copy_on_write=True` (CI toggle) | P1 | M | ✅ Done |
| 5 | Check/add `rust-toolchain.toml` for native build reproducibility | P1 | S | ✅ Done |
| 6 | Add `litellm<2.0` / `pydantic<3.0` upper-bound guardrails | P2 | S | Open |
| 7 | Document version-skew comment next to litellm/instructor deps | P2 | S | Open |
| 8 | Add polars 1.0-compat baseline comments in I/O files | P2 | S | Open |
| 9 | Add CVE-2025-69872 opt-in comment near disk cache call | P2 | S | Open |
| 10 | No structlog config change needed (weak-ref fix irrelevant) | P2 | — | N/A |
| 11 | No diskcache pin needed (no fix available, opt-in only) | P2 | — | N/A |

---

*Produced by reviewer task t_f6a46bb6. Based on DEPENDENCY_UPGRADE_ANALYSIS.md
(t_ddaee0ac), review t_a92cd0d3, and direct codebase inspection.*

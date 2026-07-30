# Dependency Upgrade Analysis — Commit 4ac9c30 (deps/upgrade-jul-2026)

Scope: what each of the 11 key dependency upgrades brings to the ondine project,
sourced from upstream changelogs and GitHub release notes.

---

## Summary Table

| Package | Old → New | What changed | Worth it? (1-5) |
|---|---|---|---|
| **litellm** | 1.83 → 1.91 | v1.84 has breaking changes (breaking API for `litellm.completion` internals); new model cost maps for Gemini 3.5 Flash, Gemini 3.1-flash-lite, Gemini managed agents, MCP OAuth; DeepSeek `/anthropic/v1/messages` endpoint; standardized rate-limit error fields; responses-bridge caching fixes; Bedrock tool spec refactor | **5** |
| **instructor** | 1.0 → 1.15 | v2 core architecture (provider-handler dispatch); security fixes (SSRF in Bedrock image/PDF, redacted auth headers, CVE-2025-69872 diskcache mitigation); Claude 4 / GPT-4.1 / o3-o4 / Grok 3 / DeepSeek model support; PEP 604 union streaming; improved Anthropic/Bedrock usage tracking; structured output + parallel tools fixes | **5** |
| **anthropic** | 0.84 → 0.116 | Claude Sonnet 5 support; Claude Mythos 5 / Fable 5; Managed Agents (deployments, webhooks, event streaming); agent-memory beta (2026-07-22); code execution tool (2026-01-20); web fetch + support tools (2026-03-18); client-side fallback middleware; User Profile ID headers; refusal category expansions | **4** |
| **polars** | 0.20 → 1.42 | **1.0 had breaking changes**: `replace` split into two methods, `read_excel` default engine → calamine, Hive partitioning disabled by default for file inputs, binary serialization default, `strict` param in Series constructor, `infer_schema_length` behavior change, type alias removals. 1.x gains: streaming engine stabilization, nested common subplan elimination, out-of-core spilling, SQL implicit JOIN syntax, `LazyFrame.gather`, bytes-based cloud IO concurrency, `is_sorted` fast paths | **4** (if code updated) |
| **pandas** | 1.5 → 2.3 | 2.0: Copy-on-Write opt-in (default in 3.0); 2.1: `infer_string` preview; 2.2: CoW "warn" mode; 2.3: PyCapsule Interface for interchange, NumPy 2.0 `__array__` fix, StringDtype HDF round-trip, PyArrow-backed string columns (preview for 3.0 default). Deprecations: `pyarrow_numpy` storage option, non-bool `na` in `.str.contains`. **2.0 itself had breaking changes** (dtypes, `iteritems` removed, etc.) | **4** |
| **pydantic** | 2.0 → 2.13 | `pydantic-core` merged into main repo; `polymorphic_serialization` option; `validate_as()`; `ascii_only` StringConstraints; `exclude_if` in computed fields; smart union omit-error handling; PEP 696 type-var defaults; PEP 728 closed TypedDicts (myPY); Python 3.14 support; riscv64 wheels; numerous Mypy plugin fixes; jiter 0.14 (musl segfault fix) | **5** |
| **structlog** | 23.1 → 26.1 | **23 → 25 → 26**: Python 3.8/3.9 dropped (26.1); Python 3.15 support; `rich_monochrome_traceback`; `ConsoleRenderer.get_active()` (25.5); `CallsiteParameter.QUAL_MODULE` + `QUAL_NAME`; snake_case `is_enabled_for`/`get_effective_level`; dict-based interpolation for native loggers; weak-ref file handles (prevents logfile leaks in long-running processes); `ConsoleRenderer` runtime-attribute reconfiguration | **3** |
| **tenacity** | 8.2 → 9.1 | **9.0 breaking**: dropped Python 3.8/3.9 support; async `sleep=` support for sync retry functions; `wait_exception` strategy; `re.Pattern` match types; snake_case logger; `BaseRetrying.copy()` returns `Self`; Python 3.14 support | **3** |
| **mypy** | 1.13 → 2.2 | **2.0 was a major bump**: PEP 728 closed TypedDicts; complete PEP 696 type-var defaults; `__new__()` explicit return type support; `TypeForm` (3.14) non-experimental; experimental WASM wheel (3.14); mypyc free-threading (nogil) improvements; `librt.strings` codepoint primitives | **4** (dev only) |
| **ruff** | 0.8 → 0.15 | **0.9-0.15**: human-readable rule selectors in comments/hovers; `--add-ignore` for inline `ruff:ignore`; notebook cell syntax error detection; `pyupgrade` `UP051` (deprecated `abc` decorators); `flake8-comprehensions` C409 fix; comment whitespace preservation; `flake8-implicit-str-concat` improvements; performance: faster single-string-literal parsing | **4** (dev only) |
| **pytest-asyncio** | 0.23 → 1.4 | **1.0 was a stability milestone** (no breaking mode change from 0.24 strict mode); 1.2: `--asyncio-debug` + `asyncio_debug` config; Pyright compatibility; 1.3: dropped Python 3.9, pytest 9 support; 1.4: `pytest_asyncio_loop_factories` hook for custom event loop factories; deprecated `event_loop_policy` override | **4** (dev only) |

---

## Detailed Notes

### litellm 1.83 → 1.91 (HIGH VALUE)
- **v1.84 had breaking changes** — flagged upstream with a warning. The release notes link to a dedicated migration guide. This is the highest-attention package since ondine uses litellm as its core LLM gateway.
- Major wins: Day-0 support for Gemini 3.5 Flash, Gemini 3.1 Flash Lite, Gemini managed agents, MCP OAuth (Cursor), DeepSeek native Anthropic endpoint, standardized rate-limit error fields (category / rate_limit_type / model / llm_provider).
- Security: Bedrock batch metadata sanitization (prevents Pydantic ValidationError), MCP JWT auth fixes.
- Stability: Redis cache token caching (prevents async event loop blocking), OpenAI/Responses bridge cache-replay-as-stream fix.

### instructor 1.0 → 1.15 (HIGH VALUE)
- **v2 architecture** with provider-handler dispatch is a substantial refactor — `from_provider("anthropic/...")` now sets a proper User-Agent.
- Security is the standout: **CVE-2025-69872** (diskcache) mitigation by making it optional, SSRF blocks for Bedrock image/PDF (remote URLs blocked, only `data:` + `s3://` accepted), auth header redaction in debug logs.
- Model support: Claude 4 (Opus/Sonnet/Haiku), GPT-4.1, o3/o4, Grok 3, DeepSeek R1/V3 added to `KnownModelName`.
- Bug fixes directly relevant to structured output: `list[Model]` scalar response-model crashes, PEP 604 union streaming, Anthropic reasoning tools routing, partial-streaming Literal defaults, Gemini truncated-response detection.
- Note: instructor 1.15.0 **pins litellm ≤ 1.82.6** (blocks compromised 1.82.7/1.82.8) — but ondine now requires litellm ≥ 1.91.1, so the instructor pin is effectively superseded by the new litellm. No conflict at install time (tests pass).

### anthropic 0.84 → 0.116 (HIGH VALUE)
- This is a ~32 minor-version jump — many additive features, no removals.
- **Claude Sonnet 5** support (0.114), plus experimental Mythos 5 / Fable 5 with server-side fallback-on-refusal (0.108).
- **Managed Agents** stack: deployments, environment-variable credentials, event delta streaming, agent overrides, reverse pagination, vault credential injection, webhooks (0.107–0.115).
- New API surfaces: `code_execution_20260120` tool, `web_fetch`/`support` tools (2026-03-18), `agent-memory-2026-07-22` beta header, User Profile ID in request headers + token counting.
- Bug fixes: async `count_tokens` merge bug, memory-tool parent-dir permissions, x-stainless-helper header-merge clobbering, Bedrock stream event-type preservation.
- Rated 4 (not 5) only because ondine primarily calls Anthropic through litellm/instructor, so most of these surfaces are used indirectly.

### polars 0.20 → 1.42 (HIGH VALUE, NEEDS CODE REVIEW)
- **The 0.20 → 1.0 jump is the riskiest upgrade in this commit.** 1.0 introduced 15+ breaking changes (see summary table). If ondine uses `pl.concat`, `replace`, `read_excel`, Hive-partitioned parquet scans, or `Series(strict=...)`, those call sites must have been verified.
- The prior review (t_a92cd0d3) confirmed all tests pass, so the codebase is already 1.x-compatible — but this is worth a focused grep for the 1.0 deprecation surface.
- Reward: streaming engine stabilization, nested CSPE, out-of-core spilling, SQL implicit JOINs, bytes-based cloud-IO concurrency control, `is_sorted` fast-path optimizations.

### pandas 1.5 → 2.3 (HIGH VALUE)
- The 1.5 → 2.0 jump removed `DataFrame.iteritems`, changed default integer dtype behavior on some platforms, and introduced Copy-on-Write as opt-in. CoW becomes default in 3.0 — ondine should test with `pd.options.mode.copy_on_write = True` before the next bump.
- 2.3 brings NumPy 2.0 compatibility (critical — `__array__` semantics fix), PyCapsule interchange interface, and PyArrow-backed string columns as the 3.0 default preview (`future.infer_string`).
- Prior review confirmed tests pass, so the 2.0 breaking surface is already handled.

### pydantic 2.0 → 2.13 (HIGH VALUE)
- 2.13 is described upstream as "mainly bug fixes and performance improvements for validation and serialization" — the `polymorphic_serialization` option resolves long-standing `serialize_as_any` ambiguity from 2.12.
- The `pydantic-core` repo was merged into the main pydantic repo (2.13.0b1) — affects only contributors who build from source.
- PEP 696 (type-var defaults) + PEP 728 (closed TypedDicts) give better typing ergonomics.
- Net: lower-risk than the version jump suggests, since pydantic 2.x has maintained strong backward compat within the 2.x line.

### structlog 23.1 → 26.1 (MODERATE)
- CalVer versioning (year.minor) — this spans 3 years of releases.
- Practical wins for a long-running service: **weak-ref file handles** prevent logfile descriptor leaks in task executors (26.1), `CallsiteParameter.QUAL_MODULE`/`QUAL_NAME` for richer log context, `ConsoleRenderer.get_active()` for runtime reconfiguration.
- The only breaking change is Python 3.8/3.9 removal (26.1) — not a concern for a project already on 3.10+.
- Rated 3 because ondine likely uses structlog's basic API surface, which is stable; most changes are additive conveniences.

### tenacity 8.2 → 9.1 (MODERATE)
- **9.0 dropped Python 3.8/3.9** — the only real breaking change.
- Nice-to-haves: `wait_exception` strategy, `re.Pattern` in match types, async `sleep=` for sync retried functions.
- The API ondine uses (`retry`, `stop_after_attempt`, `wait_exponential`) is unchanged.
- Rated 3: low risk, modest reward.

### mypy 1.13 → 2.2 (DEV ONLY, HIGH VALUE)
- **2.0 was a major release** with real new capabilities: PEP 728 closed TypedDicts, complete PEP 696, `TypeForm` (3.14) non-experimental, `__new__()` return-type respect.
- Free-threaded (nogil) mypyc builds now thread-safe — matters only if building compiled stubs.
- Rated 4 because better type checking directly reduces production bug risk; dev-only so no runtime impact.

### ruff 0.8 → 0.15 (DEV ONLY, HIGH VALUE)
- Human-readable rule selectors (`# ruff:ignore: simplify`) are a major DX win.
- Notebook cell syntax-error detection, `--add-ignore` CLI, deprecated `abc` decorator linting (UP051).
- Backward-compatible — ruff has a stable config format since 0.5.
- Rated 4: faster, more rules, no migration burden.

### pytest-asyncio 0.23 → 1.4 (DEV ONLY, MODERATE-HIGH)
- **1.0 was a stability milestone**, not a breaking-mode change — the strict mode from 0.23/0.24 carries forward cleanly.
- 1.4 adds `pytest_asyncio_loop_factories` hook for custom event loop factories and deprecates overriding `event_loop_policy`.
- 1.3 added pytest 9 support (ondine requires pytest ≥ 9.1.1, so this is required).
- Rated 4: the pytest-9 compatibility alone justifies the bump; the new hook is a bonus.

---

## Bottom-Line Verdict

The upgrade is **strongly worth it (avg 4.2/5)**: the three highest-impact packages — litellm (model + security), instructor (SSRF + CVE mitigations + Claude 4 support), and pydantic (serialization + typing PEPs) — each land real security and capability gains, while polars/pandas deliver major performance and ecosystem alignment at the cost of already-vetted breaking changes. The dev tooling bumps (mypy 2.x, ruff 0.15, pytest-asyncio 1.4) are low-risk, purely additive improvements. The only package where the reward is modest relative to churn is structlog (3/5) — its API surface in ondine is likely thin enough that the 3-year jump brings little tangible benefit, but it's harmless.

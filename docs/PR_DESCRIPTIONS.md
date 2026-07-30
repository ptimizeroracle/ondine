# PR Descriptions — ondine (Jul 2026)

Copy-paste-ready descriptions for the 12 ready branches. Ordered by merge
dependency. Each block is self-contained: title, summary, test status, and
merge order note.

---

## wt/t1-enrich

**Title:** feat(api): add enrich() front-door facade

**What:** Adds `ondine.enrich()` — a single-call entry point that accepts a
file path or DataFrame, a prompt template, and optional output columns / schema
/ budget, then builds and runs a QuickPipeline internally. Unknown kwargs are
rejected against an explicit allowlist (temperature, max_tokens, batch_size,
concurrency, provider) so the surface stays auditable.

**Diff:** +327 / −0 (3 files: `ondine/api/enrich.py`, `ondine/__init__.py`,
`tests/unit/test_enrich.py`)

**Tests:** 12/12 pass (`tests/unit/test_enrich.py`)

**Merge order:** 1st — no dependencies. Independent of all other branches.

---

## wt/t2-run-registry

**Title:** feat(orchestration): add RunRegistry persistent job index

**What:** Adds a crash-safe SQLite-backed job registry (`runs.db`, WAL mode)
that tracks the lifecycle of every pipeline run across process boundaries.
Exposes four operations — `create`, `get`, `list`, `transition` — hiding all
SQL and serialization. `RunHandle` is an immutable snapshot read fresh from
disk on every access (no cache — a second process polling must see committed
state). `RegistryObserver` provides the extension point for live progress
feeds, notified inside the same transaction commit. `Pipeline.execute()` gains
an optional `run_id=` parameter; passing `None` (the default) leaves existing
behavior untouched.

**Diff:** +1070 / −1 (4 files: `ondine/orchestration/run_registry.py`,
`ondine/api/pipeline.py`, `ondine/orchestration/__init__.py`,
`tests/unit/test_run_registry.py`)

**Tests:** 17/17 pass (`tests/unit/test_run_registry.py`)

**Merge order:** 1st — no dependencies. Foundation for t4-mcp-server and
t35-unified, but those branches already contain this commit.

---

## wt/t4-mcp-server

**Title:** feat(mcp): add FastMCP 3.x server with 4 ondine tools

**What:** Adds `ondine.mcp` — an L5 front door exposing pipeline operations as
MCP tools. Four tools: `ondine_estimate` (cost preview), `ondine_run` (launch
with mandatory budget cap, returns `run_id` immediately, runs on a daemon
thread), `ondine_status` (live progress via the shared RunRegistry),
`ondine_collect` (retrieve results). The MCP wire layer (FastMCP 3.x) is a
pass-through; all behaviour lives in `MCPService` so it's unit-testable without
an MCP client. FastMCP is imported lazily — users without the `[mcp]` extra pay
no import cost. A budget cap is enforced on every `ondine_run` so the engine's
BudgetController can abort runaway jobs end-to-end.

**Diff:** +2199 / −1 (9 files: `ondine/mcp/server.py`, `ondine/mcp/progress.py`,
`ondine/mcp/__init__.py`, `ondine/orchestration/run_registry.py`,
`ondine/api/pipeline.py`, `ondine/orchestration/__init__.py`, `pyproject.toml`,
`tests/unit/test_mcp_server.py`, `tests/unit/test_run_registry.py`)

**Tests:** 30/30 pass (17 run-registry + 13 mcp-server)

**Merge order:** After t2-run-registry (already contained in branch history).
Fast-forwards onto t2.

---

## wt/t35-unified

**Title:** feat(orchestration): ProviderBatchBackend + LiveBackend on submit/poll/collect protocol

**What:** Unifies §3 (ExecutionBackend) and §5 (ProviderBatchBackend) into a
single branch. Introduces a job-lifecycle `ExecutionBackend` protocol
(submit → poll → collect) that the pipeline's middle layer plugs into.
`ProviderBatchBackend` compiles a JSONL, submits a provider-native batch job
(OpenAI or Anthropic), and collects results later — enabling non-blocking,
cost-cheap bulk processing. `LiveBackend` ports the existing asyncio
engine onto the same protocol (synchronous submit in v1 — documented in
T3_T5_PORT_VALIDATION.md). Adds the CLI batch commands (`submit`, `poll`,
`collect`), the E2E DeepSeek test, and the fastmcp dev-extras fix.

Supersedes the discarded `wt/t3-backend-extract` (invoke-shape protocol) and
subsumes `wt/t5-provider-batch` (t35 contains t5's full commit plus the
LiveBackend port).

**Diff:** +4615 / −4 (19 files including `backends/base.py`,
`backends/live.py`, `backends/provider_batch.py`, `cli/main.py`, E2E test, 5
test files)

**Tests:** t5's 45 + LiveBackend unit tests + E2E (gated on `DEEPSEEK_API_KEY`)

**Merge order:** After t2-run-registry and t4-mcp-server (both contained in
branch history). Fast-forwards. Resolves §3 and §5 together; do NOT merge
t3-backend-extract, t5-provider-batch, or t35-unified-backend.

---

## deps/upgrade-jul-2026

**Title:** deps: upgrade all deps (Jul 2026) + OTLP migration + toolchain pin

**What:** Upgrades 16 direct and dev dependencies to latest compatible
versions (pandas, polars, litellm, instructor, pydantic, openai, anthropic,
pyyaml, diskcache, structlog, typer, rich, pytest stack, opentelemetry, etc.).
Migrates observability tracing from the archived `opentelemetry-exporter-jaeger`
to `opentelemetry-exporter-otlp` (OTLP/HTTP). Adds `rust-toolchain.toml` to pin
the Rust toolchain for `polars` builds. Adds instructor/litellm compatibility
tests, a pandas Copy-on-Write smoke test, and two analysis docs
(`DEPENDENCY_UPGRADE_ANALYSIS.md`, `DEPENDENCY_UPGRADE_ACTION_PLAN.md`).

Independent of all architecture branches — can merge in parallel with Train 1.

**Diff:** +4334 / −3008 (10 files; bulk is `uv.lock` at +6588/−2882)

**Tests:** 1088 pass (up from 1081 — +7 new compat/smoke tests), 97 skip

**Merge order:** Independent. Merge anytime relative to architecture branches.

---

## rp/readme

**Title:** docs(readme): rewrite per repositioning plan

**What:** Rewrites README.md to lead with the pain point (LLM labeling is
expensive and fiddly), positions ondine as an agentic-bridge, and adds
use-case wedges. Includes the `enrich()` one-liner quickstart doc.

**Diff:** +350 / −115 (4 files: `README.md`, `ondine/api/enrich.py`,
`ondine/__init__.py`, `tests/unit/test_enrich.py`)

**Tests:** n/a (docs). Note: this branch carries an earlier enrich() commit
that will conflict with wt/t1-enrich if both merge — resolve by taking
t1-enrich's version of `enrich.py` and the readme from this branch.

**Merge order:** Independent. Merge after t1-enrich if both are taken, to
avoid the enrich.py divergence.

---

## rp/roadmap

**Title:** docs: add ROADMAP.md

**What:** Adds `ROADMAP.md` covering repositioning status, shipped features
(enrich + Colab), and next priorities (MCP server, Batch mode). Linked from
README and docs/SUMMARY.md.

**Diff:** +107 / −0 (3 files: `ROADMAP.md`, `README.md`, `docs/SUMMARY.md`)

**Tests:** n/a (docs)

**Merge order:** Independent. May conflict trivially with rp/docs on
`docs/SUMMARY.md` — resolve by combining both edits.

---

## rp/docs

**Title:** docs: re-tier SUMMARY.md per repositioning plan

**What:** Restructures `docs/SUMMARY.md` into Tier 1 (core use cases) and
Tier 2 (Advanced), reflecting the repositioning decision to foreground common
paths and push edge-case docs deeper.

**Diff:** +17 / −9 (1 file: `docs/SUMMARY.md`)

**Tests:** n/a (docs)

**Merge order:** Independent. May conflict trivially with rp/roadmap on
`docs/SUMMARY.md` — resolve by combining both edits.

---

## rp/colab

**Title:** feat(docs): add Colab quickstart notebook + Open in Colab badge

**What:** Adds `examples/ondine_quickstart.ipynb` — a zero-install Colab
notebook that walks through loading data, calling `enrich()`, and inspecting
results. Adds an "Open in Colab" badge to README.md.

**Diff:** +145 / −0 (2 files: `examples/ondine_quickstart.ipynb`, `README.md`)

**Tests:** n/a (docs/notebook)

**Merge order:** Independent. Note: also touches README.md — merge alongside
rp/readme and rp/roadmap, resolving README conflicts by taking the union of
additions.

---

## rp/metadata

**Title:** chore(metadata): refine project description + keywords

**What:** Refines `pyproject.toml` project description and keywords to match
the repositioning (pain-first framing, discoverable search terms).

**Diff:** +2 / −2 (1 file: `pyproject.toml`)

**Tests:** n/a (config)

**Merge order:** Independent. May conflict with deps/upgrade-jul-2026 on
`pyproject.toml` — resolve by taking deps branch's dependency changes and
this branch's description/keywords line.

---

## rp/benchmark

**Title:** feat(benchmarks): 3-arm repositioning benchmark + RESULTS.md

**What:** Adds a three-arm benchmark (`benchmarks/repositioning.py`) comparing
repositioning approaches, with dataset generation (`generate_dataset.py`), raw
results (`results.json`), and a human-readable `RESULTS.md`. Used to validate
the repositioning claims with measured data.

**Diff:** +1281 / −0 (6 files: `benchmarks/repositioning.py`,
`benchmarks/generate_dataset.py`, `benchmarks/results.json`,
`benchmarks/RESULTS.md`, `README.md`, `.gitignore`)

**Tests:** n/a (benchmark — not a test suite)

**Merge order:** Independent. Touches README.md — coordinate with rp/readme
and rp/colab.

---

## wt/t6-plan

**Title:** feat(intent): add ondine.plan() intent layer

**What:** Adds `ondine.plan()` — a deep module that uses a single structured
LLM call to turn a goal + a sample of tabular data into a fully-formed
`PipelineSpecifications`. Returns an immutable `Plan` handoff the user
inspects (YAML preview) and then builds via `Plan.build()`. The LLM client is
an injectable boundary for testability. No agent loop, no re-planning, no
execution — plan is a one-shot draft.

Marked v2 in the architecture proposal ("after 1–5 ship + market signal").
Built and tested but the product call is to hold or merge at discretion.

**Diff:** +828 / −0 (4 files: `ondine/orchestration/intent/planner.py`,
`ondine/orchestration/intent/__init__.py`, `ondine/__init__.py`,
`tests/unit/test_planner.py`)

**Tests:** 12/12 pass (`tests/unit/test_planner.py`)

**Merge order:** Independent at runtime. Hold for v2 per proposal, or merge
after §1–§5 if shipping now.

---

## Merge order summary

**Train 1 — Architecture (fast-forward chain):**
1. wt/t1-enrich
2. wt/t2-run-registry
3. wt/t4-mcp-server
4. wt/t35-unified
5. wt/t6-plan (optional — v2 candidate)

**Train 2 — Dependencies (parallel):**
- deps/upgrade-jul-2026

**Train 3 — Repositioning (any order, docs-only conflicts):**
- rp/readme, rp/roadmap, rp/docs, rp/colab, rp/metadata, rp/benchmark

**Do NOT merge:** wt/t3-backend-extract (discarded), wt/t5-provider-batch
(subsumed by t35), wt/t35-unified-backend (stale checkpoint).

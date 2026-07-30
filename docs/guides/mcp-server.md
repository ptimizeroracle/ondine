# MCP Server (`ondine-mcp`)

Agents are good at reasoning over one row at a time and bad at bulk column work -- loop-per-row means loop-per-API-call, no batching, no shared budget cap, no checkpoint. The `ondine-mcp` server exposes Ondine's pipeline as four [MCP](https://modelcontextprotocol.io) tools so an agent delegates the whole column-fill job to Ondine instead of looping row-by-row itself.

## Install

```bash
pip install ondine[mcp]
```

This installs the `ondine-mcp` console script, built on [FastMCP](https://github.com/jlowin/fastmcp), serving over stdio.

## Register With an MCP Client

```json
{
  "mcpServers": {
    "ondine": {
      "command": "ondine-mcp"
    }
  }
}
```

Drop that into your client's MCP config (Claude Desktop's `claude_desktop_config.json`, Claude Code's `.mcp.json`, etc.) under `mcpServers`. If `ondine-mcp` isn't on `PATH` in the client's environment, point `command` at the full interpreter path instead, e.g. `"/path/to/venv/bin/ondine-mcp"`.

## The Four Tools

| Tool | Blocks? | Purpose |
|---|---|---|
| `ondine_estimate` | No | Cost/token/row estimate for a config, without running it. |
| `ondine_run` | No | Launch a run. Returns `run_id` immediately. |
| `ondine_status` | No | Poll progress: state, `%`, rows done, cost so far. |
| `ondine_collect` | No | Terminal readout: summary + output path, once finished. |

### `ondine_run` Never Blocks

`ondine_run` hands back a `run_id` immediately -- the pipeline runs on a background thread, and its state (rows processed, cost, final status) is written to the durable `RunRegistry` (SQLite, in the checkpoint dir). The agent (or any other MCP client, even in a different process) polls `ondine_status` and calls `ondine_collect` once the run reaches a terminal state. This is the same non-blocking contract the CLI's `ondine submit` / `ondine status` / `ondine collect` commands use.

### `ondine_collect` Never Streams Row Data

`ondine_collect` returns a summary -- `run_id`, `status`, `rows_done`, `total_rows`, `cost`, `output_path`, `error` -- and the path to the output file. It does not stream row-by-row results back through the MCP channel. If the agent needs the data, it reads the output file itself.

### A Positive Budget Is Mandatory

```python
# ondine_run(config_yaml, input_path, output_path, budget)
```

`budget` is a required, positive argument on `ondine_run`. An agent must not be able to start an uncapped-spend job over MCP -- the tool call itself raises before any LLM call is made if `budget` is missing or `<= 0`:

```
ValueError: A budget cap is mandatory for ondine_run. Pass a positive
budget (USD) to set the maximum spend for this run.
```

The budget is injected into the pipeline's `ProcessingSpec.max_budget` before the run starts, so the same `BudgetController` that enforces every other Ondine run enforces this one.

## Example Flow

1. Agent calls `ondine_estimate(config_yaml=...)` to sanity-check cost before committing.
2. Agent calls `ondine_run(config_yaml=..., input_path="reviews.csv", output_path="out.csv", budget=5.0)` -> gets `{"run_id": "..."}` back instantly.
3. Agent polls `ondine_status(run_id)` periodically for `progress_pct` / `rows_done`.
4. Once `status` is terminal (`succeeded`, `failed`, or `partial`), agent calls `ondine_collect(run_id)` for the final summary and reads `output_path`.

## Related

- [Provider Batch API Mode](provider-batch.md) -- the RunRegistry that backs `ondine_status`/`ondine_collect` also tracks provider batch jobs
- [Cost Estimation & Budgets](cost-control.md) -- how `max_budget` is enforced during execution
- [CLI](cli.md) -- the `ondine submit` / `status` / `collect` commands that share the same RunRegistry

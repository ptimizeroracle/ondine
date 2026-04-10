# CLI Reference

The CLI tool is called `ondine` (`hermes` still works as an alias). It handles dataset processing, cost estimation, config validation, file inspection, checkpoint management, and provider listing. No Python required.

## Installation

```bash
pip install ondine
```

Check it worked:

```bash
ondine --version
```

## Configuration File

Every pipeline command needs a YAML (or JSON) config file via `--config` / `-c`. This is where you define your data source, prompt, LLM, processing behavior, and output.

**Minimal example (`config.yaml`):**

```yaml
data:
  source:
    type: csv
    path: data.csv
  input_columns: [text]
  output_columns: [sentiment]

prompt:
  template: "Classify sentiment: {text}"
  system_message: "You are a sentiment classifier. Reply positive, negative, or neutral."

llm:
  provider: openai
  model: gpt-4o-mini

processing:
  concurrency: 5
  rate_limit: 60        # maps to rate_limit_rpm

output:
  format: csv
  path: results.csv
```

Quick notes on the fields:
- `data.source.type` -- `csv`, `excel`, `parquet`
- `data.source.path` -- file path
- `processing.rate_limit` -- requests per minute
- `output.format` -- `csv`, `excel`, or `parquet`

---

## Commands

### `process`

This is the command you'll run most of the time. It takes your config and runs the actual LLM pipeline against your dataset.

```
ondine process [OPTIONS]
```

**Options:**

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--config PATH` | `-c` | Yes | YAML/JSON configuration file |
| `--input PATH` | `-i` | No | Input data file (overrides config) |
| `--output PATH` | `-o` | No | Output file (overrides config) |
| `--provider CHOICE` | | No | Override LLM provider from config |
| `--model TEXT` | | No | Override model name from config |
| `--max-budget FLOAT` | | No | Override maximum budget in USD |
| `--batch-size INT` | | No | Override rows-per-prompt-batch |
| `--concurrency INT` | | No | Override concurrency level |
| `--checkpoint-dir PATH` | | No | Override checkpoint directory |
| `--dry-run` | | No | Validate and estimate only, skip execution |
| `--verbose` | `-v` | No | Print full traceback on errors |

**Examples:**

```bash
# Basic run using config file
ondine process -c config.yaml

# Override input/output paths
ondine process -c config.yaml -i data.csv -o results.csv

# Switch provider and model at runtime
ondine process -c config.yaml -i data.csv -o results.csv \
    --provider anthropic --model claude-sonnet-4-20250514

# Set a cost ceiling of $5
ondine process -c config.yaml -i data.csv -o results.csv \
    --max-budget 5.0

# Estimate cost without processing anything
ondine process -c config.yaml -i data.csv --dry-run

# Enable multi-row batching (100 rows per API call)
ondine process -c config.yaml -i data.csv -o results.csv \
    --batch-size 100 --concurrency 10

# Store checkpoints in a custom directory
ondine process -c config.yaml -i data.csv \
    --checkpoint-dir /tmp/my-checkpoints
```

When it finishes, you get two tables: an **Execution Results** table (rows processed/failed/skipped, duration, total cost, cost per row) and a **Quality Report** (success rate, null/empty outputs, quality score).

---

### `estimate`

Run this before `process` to see what the bill looks like. No data gets sent anywhere.

```
ondine estimate [OPTIONS]
```

**Options:**

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--config PATH` | `-c` | Yes | YAML/JSON configuration file |
| `--input PATH` | `-i` | Yes | Input data file |
| `--provider CHOICE` | | No | Override LLM provider |
| `--model TEXT` | | No | Override model name |

**Examples:**

```bash
# Estimate cost for a dataset
ondine estimate -c config.yaml -i data.csv

# Compare cost of two models
ondine estimate -c config.yaml -i data.csv --model gpt-4o-mini
ondine estimate -c config.yaml -i data.csv --model gpt-4o
```

Output shows total cost, total/input/output tokens, rows to process, confidence level, and cost per row. You'll get a warning if the estimate goes above $10.

---

### `validate`

Catches config mistakes before you burn API credits. Does not call any LLM.

```
ondine validate [OPTIONS]
```

**Options:**

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--config PATH` | `-c` | Yes | YAML/JSON configuration file |
| `--verbose` | `-v` | No | Print configuration summary table |

**Examples:**

```bash
# Basic validation
ondine validate -c config.yaml

# Verbose -- also shows configuration summary
ondine validate -c config.yaml --verbose
```

Exit 0 means you're good. Exit 1 means something is wrong. Warnings (non-fatal) still print even on a passing run.

---

### `resume`

Pick up where you left off after a crash or Ctrl-C. Needs the session ID that was printed when the original run started.

```
ondine resume [OPTIONS]
```

**Options:**

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--session-id UUID` | `-s` | Yes | Session UUID to resume |
| `--config PATH` | `-c` | No* | Original config file for the run |
| `--checkpoint-dir PATH` | | No | Checkpoint directory (default: `.checkpoints`) |
| `--input PATH` | `-i` | No | Override input data source |
| `--output PATH` | `-o` | No | Override output path |
| `--verbose` | `-v` | No | Show full traceback on errors |

*`--config` is required once the checkpoint is found -- the command will error if you skip it.

**Examples:**

```bash
# Resume using session ID shown when the original run was interrupted
ondine resume -s 3f2a1b4c-1234-5678-abcd-ef0123456789 -c config.yaml

# Resume with a custom checkpoint directory
ondine resume -s 3f2a1b4c-1234-5678-abcd-ef0123456789 \
    -c config.yaml \
    --checkpoint-dir /tmp/my-checkpoints

# Resume and write output to a different file
ondine resume -s 3f2a1b4c-1234-5678-abcd-ef0123456789 \
    -c config.yaml -o results_resumed.csv
```

Before it starts processing again, you'll see a **Checkpoint Information** table with the session ID, rows already done, total rows, last pipeline stage, timestamp, and cost so far.

---

### `list-checkpoints`

See what checkpoints exist. Useful when you don't remember the session ID.

```
ondine list-checkpoints [OPTIONS]
```

**Options:**

| Flag | Required | Description |
|------|----------|-------------|
| `--checkpoint-dir PATH` | No | Directory to scan (default: `.checkpoints`) |

**Examples:**

```bash
# List checkpoints in the default directory
ondine list-checkpoints

# List from a custom directory
ondine list-checkpoints --checkpoint-dir /tmp/my-checkpoints
```

Prints a table: session ID (truncated), rows processed, total rows, last stage, cost so far, timestamp, file path.

---

### `inspect`

Quick look at a data file before you write your config. Shows metadata, columns, and a row preview.

```
ondine inspect [OPTIONS]
```

**Options:**

| Flag | Short | Required | Description |
|------|-------|----------|-------------|
| `--input PATH` | `-i` | Yes | File to inspect (CSV, Excel, Parquet) |
| `--head INT` | | No | Number of rows to preview (default: 5) |

**Examples:**

```bash
# Inspect a CSV
ondine inspect -i data.csv

# Preview first 20 rows of a Parquet file
ondine inspect -i data.parquet --head 20

# Inspect an Excel workbook
ondine inspect -i data.xlsx
```

You get the file path, type, total rows, total columns, memory usage, per-column dtype and null counts, and a text row preview.

Heads up: Excel files need `pip install ondine[excel]` and Parquet files need `pip install ondine[parquet]`.

---

### `list-providers`

Dumps the full provider table. No flags.

```
ondine list-providers
```

**Supported providers:**

| Provider ID | Name | Platform | Cost |
|-------------|------|----------|------|
| `openai` | OpenAI | Cloud | $$ |
| `azure_openai` | Azure OpenAI | Cloud | $$ |
| `anthropic` | Anthropic Claude | Cloud | $$$ |
| `groq` | Groq | Cloud | Free tier |
| `openai_compatible` | OpenAI-Compatible | Custom/Local/Cloud | Varies |
| `mlx` | Apple MLX | macOS M1/M2/M3/M4 | Free |
| `litellm` | LiteLLM (Universal) | Cloud | Varies |

Also prints per-provider requirement notes and usage examples.

---

## Global Flags

```bash
ondine --version   # Print version and exit
ondine --help      # Print top-level help
ondine COMMAND --help  # Print help for a specific command
```

---

## Config File Reference

Every key the YAML config accepts:

```yaml
# Data source
data:
  source:
    type: csv          # csv | excel | parquet
    path: data.csv
  input_columns: [col1, col2]
  output_columns: [result]
  delimiter: ","       # CSV only, default ","
  encoding: utf-8      # default utf-8
  sheet_name: 0        # Excel only, default 0

# Prompt
prompt:
  template: "Analyze: {col1}"
  system_message: "You are an expert."
  batch_size: 1        # Multi-row batching (1 = off)
  batch_strategy: json # json | csv
  response_format: raw # raw | json | regex

# LLM
llm:
  provider: openai     # See list-providers
  model: gpt-4o-mini
  temperature: 0.0
  max_tokens: 1024
  top_p: 1.0
  # Azure-specific
  azure_endpoint: https://...
  azure_deployment: my-deployment
  use_managed_identity: false
  # Custom/compatible APIs
  base_url: http://localhost:11434/v1

# Processing
processing:
  batch_size: 100      # Internal rows-per-batch for I/O (default 100)
  concurrency: 5       # Parallel LLM requests (default 5)
  rate_limit: 60       # Requests per minute
  max_budget: 10.0     # USD ceiling
  max_retries: 3
  retry_delay: 1.0
  checkpoint_interval: 500
  checkpoint_dir: .checkpoints
  error_policy: skip   # skip | retry | fail | use_default

# Output
output:
  format: csv          # csv | excel | parquet
  path: results.csv
```

---

## Typical Workflow

```bash
# 1. Inspect your data
ondine inspect -i data.csv

# 2. Write config.yaml then validate it
ondine validate -c config.yaml --verbose

# 3. Estimate cost before committing
ondine estimate -c config.yaml -i data.csv

# 4. Process with a budget guard
ondine process -c config.yaml -i data.csv -o results.csv --max-budget 5.0

# 5. If interrupted, resume from checkpoint
ondine list-checkpoints
ondine resume -s <session-id> -c config.yaml -o results.csv
```

Always set `--max-budget` on your first run with a new dataset. Better to hit the ceiling early than to find out the hard way.

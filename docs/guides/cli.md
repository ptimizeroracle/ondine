# CLI Reference

The CLI is called `ondine` (`hermes` still works as an alias). It runs pipelines, estimates costs, validates configs, inspects data files, manages checkpoints, and lists providers. No Python scripting required.

## Installation

```bash
pip install ondine
```

Check it worked:

```bash
ondine --version
```

## Configuration File

Every pipeline command reads a YAML (or JSON) config file passed with `--config` / `-c`. The config defines your data source, prompt, LLM, processing behavior, and output destination.

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

The main event. Takes your config and runs the LLM pipeline against the dataset.

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

On completion you get two tables: **Execution Results** (rows processed/failed/skipped, duration, total cost, cost per row) and **Quality Report** (success rate, null/empty outputs, quality score).

---

### `estimate`

Run this before `process` to preview the bill. Nothing leaves your machine.

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

Output includes total cost, token breakdown (input/output), rows to process, confidence level, and cost per row. Estimates above $10 trigger a warning.

---

### `validate`

Catches config mistakes before you burn API credits. No LLM calls, no network access.

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

Exit 0: valid. Exit 1: broken. Non-fatal warnings still print on a passing run.

---

### `resume`

Picks up where a crashed or interrupted run left off. You need the session ID that was printed when the original run started.

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

Before processing resumes, a **Checkpoint Information** table shows the session ID, rows completed, total rows, last pipeline stage, timestamp, and cost so far.

---

### `list-checkpoints`

Lists existing checkpoints. Useful when you have lost track of session IDs.

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

Peek at a data file before writing your config. Shows metadata, columns, and a sample of rows.

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

Output includes file path, type, row/column counts, memory usage, per-column dtypes and null counts, plus a text preview of the first rows.

Excel support requires `pip install ondine[excel]`. Parquet requires `pip install ondine[parquet]`.

---

### `list-providers`

Prints the full provider table. No flags needed.

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

Full list of accepted YAML keys:

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

Set `--max-budget` on your first run with any new dataset. Hitting a $5 ceiling beats discovering a $200 surprise.

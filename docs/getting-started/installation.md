# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

### Using pip

```bash
pip install ondine
```

### Using uv (Recommended)

uv resolves and installs packages significantly faster than pip:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ondine
uv pip install ondine
```

## Optional Dependencies

Install only what you need. Each extra pulls in a focused set of packages.

### MLX (Apple Silicon Local Inference)

Run models locally on Apple Silicon (M1/M2/M3/M4) with MLX:

```bash
pip install ondine[mlx]
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ RAM recommended

**Supported models:**
- Qwen-2.5 series
- Llama models
- Mistral models
- Any MLX-compatible model from Hugging Face

### Observability

OpenTelemetry-based tracing and metrics:

```bash
pip install ondine[observability]
```

Adds distributed tracing (Jaeger), custom metrics export, and performance monitoring.

### Development Tools

For contributing to Ondine:

```bash
pip install ondine[dev]
```

Pulls in pytest, ruff, mypy, pre-commit hooks, and security scanners (bandit, pip-audit).

### Redis (Response Caching)

LiteLLM Redis-backed response caching:

```bash
pip install ondine[redis]
```

**Installs:** `redis>=5.0.0`

### Excel Support

Read and write `.xlsx` / `.xls` files:

```bash
pip install ondine[excel]
```

**Installs:** `openpyxl>=3.1.0`

### Parquet Support

Read and write Parquet files:

```bash
pip install ondine[parquet]
```

**Installs:** `pyarrow>=15.0.0`

### TUI (Terminal User Interface)

Interactive terminal dashboard:

```bash
pip install ondine[tui]
```

**Installs:** `textual>=1.0.0`

### Performance (uvloop)

Faster async I/O on Linux and macOS:

```bash
pip install ondine[performance]
```

**Installs:** `uvloop>=0.19.0` (not installed on Windows)

### Knowledge Base / RAG

PDF ingestion and embedding-based knowledge stores:

```bash
pip install ondine[knowledge]
```

**Installs:** `pymupdf>=1.24`, `sentence-transformers>=3.0`

### Zep (Long-Term Memory)

Zep Cloud-backed memory and conversation history:

```bash
pip install ondine[zep]
```

**Installs:** `zep-cloud>=2.0`

### Azure (Managed Identity)

Required for Azure OpenAI with Managed Identity authentication:

```bash
pip install ondine[azure]
```

**Installs:** `azure-identity>=1.15.0`

See [Azure Managed Identity Guide](../guides/azure-managed-identity.md) for setup details.

### Install All Optional Dependencies

```bash
pip install ondine[all]
```

`[all]` installs every optional extra except `[dev]`, `[observability]`, `[docs]`, and `[mlx]`. Install those separately as needed:

```bash
pip install ondine[all,mlx,observability]
```

## Verify Installation

```python
import ondine

print(ondine.__version__)
print("Ondine installed successfully!")
```

Or use the CLI:

```bash
ondine --version
```

## API Keys Setup

Set provider keys as environment variables:

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

### Azure OpenAI

**Option 1: Managed Identity (Recommended for Production)**

```bash
# Install Azure dependencies
pip install ondine[azure]

# For local development, login with Azure CLI
az login

# No API keys needed! Uses Managed Identity automatically.
```

**Option 2: API Key (Traditional)**

```bash
export AZURE_OPENAI_API_KEY="..."  # pragma: allowlist secret
export AZURE_OPENAI_ENDPOINT="https://..."
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

See [Azure Managed Identity Guide](../guides/azure-managed-identity.md) for detailed setup.

### Anthropic Claude

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # pragma: allowlist secret
```

### Groq

```bash
export GROQ_API_KEY="gsk_..."  # pragma: allowlist secret
```

### Environment File

Or put everything in a `.env` file at your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

Ondine loads `.env` files automatically via python-dotenv.

## Upgrade

```bash
pip install --upgrade ondine
```

## Troubleshooting

### Import Error: No module named 'ondine'

Verify you're in the right Python environment:

```bash
python -c "import sys; print(sys.executable)"
pip list | grep ondine
```

### MLX Installation Issues

MLX only works on Apple Silicon. If imports fail, check three things: `uname -m` should print `arm64`, Xcode Command Line Tools must be installed (`xcode-select --install`), and a clean reinstall often fixes stale state: `pip uninstall mlx mlx-lm && pip install ondine[mlx]`.

### API Key Not Found

Authentication errors usually mean the key isn't in scope. Run `echo $OPENAI_API_KEY` to confirm the variable is set, check that your `.env` file is in the project root, and restart your Python session after any changes.

## Next Steps

- [Quickstart Guide](quickstart.md) - Build your first pipeline
- [Core Concepts](core-concepts.md) - Understand the architecture
- [Provider Configuration](../guides/providers/openai.md) - Configure LLM providers

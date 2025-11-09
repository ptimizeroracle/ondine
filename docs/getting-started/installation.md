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

uv is a fast Python package installer and resolver:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ondine
uv pip install ondine
```

## Optional Dependencies

Ondine provides optional dependency groups for specific features:

### MLX (Apple Silicon Local Inference)

For running models locally on Apple Silicon (M1/M2/M3/M4) with MLX:

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

For OpenTelemetry-based observability and tracing:

```bash
pip install ondine[observability]
```

**Features:**
- Distributed tracing with Jaeger
- Custom metrics export
- Performance monitoring

### Development Tools

For contributing or development:

```bash
pip install ondine[dev]
```

**Includes:**
- pytest and test utilities
- ruff for linting
- mypy for type checking
- pre-commit hooks
- Security scanners (bandit, pip-audit)

### Install All Optional Dependencies

```bash
pip install ondine[mlx,observability,dev]
```

## Verify Installation

Check that Ondine is installed correctly:

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

Ondine requires API keys for LLM providers. Set them as environment variables:

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

For convenience, create a `.env` file in your project root:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
```

Ondine automatically loads `.env` files using python-dotenv.

## Upgrade

To upgrade to the latest version:

```bash
pip install --upgrade ondine
```

## Troubleshooting

### Import Error: No module named 'ondine'

Make sure you're in the correct Python environment:

```bash
python -c "import sys; print(sys.executable)"
pip list | grep ondine
```

### MLX Installation Issues

MLX only works on Apple Silicon. If you get import errors:

1. Verify you're on Apple Silicon: `uname -m` should show `arm64`
2. Install Xcode Command Line Tools: `xcode-select --install`
3. Try reinstalling: `pip uninstall mlx mlx-lm && pip install ondine[mlx]`

### API Key Not Found

If you get authentication errors:

1. Check environment variables: `echo $OPENAI_API_KEY`
2. Verify `.env` file is in the correct directory
3. Restart your Python session after setting environment variables

## Next Steps

- [Quickstart Guide](quickstart.md) - Build your first pipeline
- [Core Concepts](core-concepts.md) - Understand the architecture
- [Provider Configuration](../guides/providers/openai.md) - Configure LLM providers

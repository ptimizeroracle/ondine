# Contributing to Ondine

Thank you for your interest in contributing to Ondine!

## Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Ondine.git
   cd Ondine
   ```

2. **Set Up Development Environment**
   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --extra dev --extra observability

   # Install pre-commit hooks
   uv run pre-commit install
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=ondine --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_pipeline_builder.py

# Run with verbose output
uv run pytest -v
```

### Code Quality

```bash
# Format code
uv run ruff format ondine/ tests/

# Lint code
uv run ruff check ondine/ tests/

# Type check
uv run mypy ondine/

# Run all quality checks
just lint
```

### Using Justfile

We provide a `justfile` for common tasks:

```bash
# Run tests
just test

# Run tests with coverage
just test-coverage

# Format and lint
just format
just lint

# Run all checks
just check

# View all available commands
   just --list
```

## Code Guidelines

### Style

- Follow [PEP 8](https://pep8.org/) and the [Zen of Python](https://www.python.org/dev/peps/pep-0020/)
- Use type hints for all function signatures
- Keep functions small and focused (KISS principle)
- Write self-documenting code with clear variable names

### Testing

- Write tests for all new features (TDD encouraged)
- Maintain or improve test coverage (currently 95%+)
- Include both unit and integration tests where appropriate
- Use descriptive test names: `test_<what>_<when>_<expected>`

### Documentation

- Update README.md if adding user-facing features
- Add docstrings to all public functions and classes
- Include examples for new features in `examples/` directory
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/)

### Commits

- Write clear, descriptive commit messages
- Use conventional commits format:
  - `feat:` - New feature
  - `fix:` - Bug fix
  - `docs:` - Documentation changes
  - `test:` - Test additions or changes
  - `refactor:` - Code refactoring
  - `chore:` - Maintenance tasks

Example:
```
feat: add support for custom retry strategies

- Implement RetryStrategy interface
- Add exponential backoff with jitter
- Update documentation
```

## Architecture

Ondine follows a **5-layer clean architecture**:

1. **Core** - Domain models and business logic
2. **Adapters** - External integrations (LLM clients, I/O)
3. **Stages** - Pipeline processing stages
4. **Orchestration** - Execution strategies and state management
5. **API** - Public interfaces (builders, composers)

### Plugin System

Ondine uses decorators for extensibility:

- `@provider` - Register custom LLM providers
- `@stage` - Register custom pipeline stages

See `examples/15_custom_llm_provider.py` and `examples/16_custom_pipeline_stage.py` for details.

## Adding New Features

### Adding a New LLM Provider

1. Create a new class inheriting from `BaseLLMProvider`
2. Implement `invoke()` and `estimate_tokens()` methods
3. Register with `@provider("your_provider_name")`
4. Add tests in `tests/unit/test_providers.py`
5. Add example in `examples/`
6. Update README.md

### Adding a New Pipeline Stage

1. Create a new class inheriting from `PipelineStage`
2. Implement `execute()` method
3. Register with `@stage("your_stage_name")`
4. Add tests in `tests/unit/test_stages.py`
5. Add example in `examples/`
6. Update documentation

## Reporting Bugs

1. Check if the bug is already reported in [Issues](https://github.com/ptimizeroracle/Ondine/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, Ondine version)
   - Minimal reproducible example

## Suggesting Features

1. Check [existing feature requests](https://github.com/ptimizeroracle/Ondine/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
2. Open a new issue with:
   - Clear use case description
   - Proposed API or interface
   - Example usage
   - Why this would benefit users

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass: `just test`
   - Run code quality checks: `just check`
   - Update documentation if needed
   - Add entry to CHANGELOG.md

2. **Submit PR:**
   - Write a clear title and description
   - Reference related issues (e.g., "Fixes #123")
   - Ensure CI passes (tests, linting, security)
   - Respond to code review feedback

3. **After approval:**
   - Maintainers will merge your PR
   - Your contribution will be included in the next release!

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/ptimizeroracle/Ondine/labels/good%20first%20issue) - these are great for newcomers!

## Getting Help

- **Questions?** Open a [Discussion](https://github.com/ptimizeroracle/Ondine/discussions)
- **Bugs?** Open an [Issue](https://github.com/ptimizeroracle/Ondine/issues)
- **Chat?** Join our community (link coming soon)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## Thank You!

Every contribution, no matter how small, makes Ondine better. We appreciate your time and effort!

---

**Happy coding!**

"""Unit tests for ConfigLoader env-var interpolation (issue #166)."""

import textwrap
from pathlib import Path

import pytest

from ondine.config import ConfigLoader


@pytest.fixture
def minimal_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid YAML config and return its path."""
    content = textwrap.dedent("""\
        dataset:
          source_type: "csv"
          source_path: "/data/in.csv"
          input_columns: ["name"]
          output_columns: ["label"]

        prompt:
          template: "Label this: {{ name }}"

        llm:
          provider: "groq"
          model: "groq/llama-3.1-8b-instant"
          api_key: "${ONDINE_TEST_API_KEY}"

        output:
          destination_type: "csv"
          destination_path: "/data/out.csv"
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def test_env_var_in_api_key_is_expanded(minimal_yaml: Path, monkeypatch):
    monkeypatch.setenv(
        "ONDINE_TEST_API_KEY", "sk-test-abc123"
    )  # pragma: allowlist secret

    specs = ConfigLoader.from_yaml(minimal_yaml)

    assert specs.llm.api_key == "sk-test-abc123"  # pragma: allowlist secret


def test_unset_env_var_raises_value_error(minimal_yaml: Path, monkeypatch):
    monkeypatch.delenv("ONDINE_TEST_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ONDINE_TEST_API_KEY"):
        ConfigLoader.from_yaml(minimal_yaml)


def test_empty_env_var_raises_value_error(minimal_yaml: Path, monkeypatch):
    monkeypatch.setenv("ONDINE_TEST_API_KEY", "")

    with pytest.raises(ValueError, match="ONDINE_TEST_API_KEY"):
        ConfigLoader.from_yaml(minimal_yaml)


def test_bare_dollar_sign_preserved(tmp_path: Path):
    """Bare $VAR (no braces) must not be expanded — literal $ in prompts stays intact."""
    content = textwrap.dedent("""\
        dataset:
          source_type: "csv"
          source_path: "/data/in.csv"
          input_columns: ["name"]
          output_columns: ["label"]

        prompt:
          template: "Price is $10 for $PATH items about {name}"

        llm:
          provider: "groq"
          model: "groq/llama-3.1-8b-instant"
          api_key: "sk-hardcoded"  # pragma: allowlist secret

        output:
          destination_type: "csv"
          destination_path: "/data/out.csv"
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)

    specs = ConfigLoader.from_yaml(p)

    assert specs.prompt.template == "Price is $10 for $PATH items about {name}"


def test_plain_string_not_mangled(tmp_path: Path):
    content = textwrap.dedent("""\
        dataset:
          source_type: "csv"
          source_path: "/data/in.csv"
          input_columns: ["name"]
          output_columns: ["label"]

        prompt:
          template: "Label this: {{ name }}"

        llm:
          provider: "groq"
          model: "groq/llama-3.1-8b-instant"
          api_key: "sk-hardcoded"  # pragma: allowlist secret

        output:
          destination_type: "csv"
          destination_path: "/data/out.csv"
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)

    specs = ConfigLoader.from_yaml(p)

    assert specs.llm.api_key == "sk-hardcoded"  # pragma: allowlist secret


def test_env_var_in_nested_field_expands(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ONDINE_TEST_BASE_URL", "https://custom.api.example.com/v1")
    content = textwrap.dedent("""\
        dataset:
          source_type: "csv"
          source_path: "/data/in.csv"
          input_columns: ["name"]
          output_columns: ["label"]

        prompt:
          template: "Label: {{ name }}"

        llm:
          provider: "openai"
          model: "gpt-4o-mini"
          api_key: "sk-test"  # pragma: allowlist secret
          base_url: "${ONDINE_TEST_BASE_URL}"

        output:
          destination_type: "csv"
          destination_path: "/data/out.csv"
    """)
    p = tmp_path / "config.yaml"
    p.write_text(content)

    specs = ConfigLoader.from_yaml(p)

    assert specs.llm.base_url == "https://custom.api.example.com/v1"

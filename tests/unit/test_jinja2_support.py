"""Tests for Jinja2 template support in Ondine."""

import pandas as pd

from ondine.core.specifications import PromptSpec
from ondine.orchestration.execution_context import ExecutionContext
from ondine.stages.prompt_formatter_stage import PromptFormatterStage


class TestJinja2Support:
    """Test Jinja2 template rendering in PromptFormatterStage."""

    def test_jinja2_conditionals(self):
        """Should render Jinja2 conditionals correctly."""
        df = pd.DataFrame({"product": ["Apple", "Banana"], "qty": [5, 15]})

        template = """{{ product }}
{% if qty > 10 %}
HIGH
{% else %}
LOW
{% endif %}"""

        spec = PromptSpec(template=template)
        formatter = PromptFormatterStage(batch_size=10, use_jinja2=True)
        batches = formatter.process((df, spec), ExecutionContext())

        prompts = batches[0].prompts
        assert "Apple" in prompts[0]
        assert "LOW" in prompts[0]
        assert "Banana" in prompts[1]
        assert "HIGH" in prompts[1]

    def test_jinja2_loops(self):
        """Should render Jinja2 loops correctly."""
        df = pd.DataFrame({"items": [["a", "b", "c"]]})

        template = """Items:
{% for item in items %}
- {{ item }}
{% endfor %}"""

        spec = PromptSpec(template=template)
        formatter = PromptFormatterStage(batch_size=10, use_jinja2=True)
        batches = formatter.process((df, spec), ExecutionContext())

        prompt = batches[0].prompts[0]
        assert "- a" in prompt
        assert "- b" in prompt
        assert "- c" in prompt

    def test_jinja2_filters(self):
        """Should apply Jinja2 filters correctly."""
        df = pd.DataFrame({"text": ["hello world"]})

        template = """{{ text | upper }}"""

        spec = PromptSpec(template=template)
        formatter = PromptFormatterStage(batch_size=10, use_jinja2=True)
        batches = formatter.process((df, spec), ExecutionContext())

        prompt = batches[0].prompts[0]
        assert "HELLO WORLD" in prompt

    def test_format_style_still_works(self):
        """Should fall back to .format() when use_jinja2=False."""
        df = pd.DataFrame({"name": ["Alice"], "age": [30]})

        template = "Name: {name}, Age: {age}"

        spec = PromptSpec(template=template)
        formatter = PromptFormatterStage(batch_size=10, use_jinja2=False)
        batches = formatter.process((df, spec), ExecutionContext())

        prompt = batches[0].prompts[0]
        assert "Name: Alice, Age: 30" in prompt

    def test_simple_variables_work_in_both_modes(self):
        """Simple variable substitution works in both modes (different syntax)."""
        df = pd.DataFrame({"product": ["Apple"]})
        template = "Product: {product}"

        spec = PromptSpec(template=template)

        # Test with .format() mode
        formatter_format = PromptFormatterStage(batch_size=10, use_jinja2=False)
        batches_format = formatter_format.process((df, spec), ExecutionContext())
        assert "Product: Apple" in batches_format[0].prompts[0]

        # Test with Jinja2 mode (should convert {x} to {{ x }})
        # Note: This will fail because Jinja2 uses {{ }} not {}
        # User needs to use {{ product }} for Jinja2
        template_jinja = "Product: {{ product }}"
        spec_jinja = PromptSpec(template=template_jinja)
        formatter_jinja = PromptFormatterStage(batch_size=10, use_jinja2=True)
        batches_jinja = formatter_jinja.process((df, spec_jinja), ExecutionContext())
        assert "Product: Apple" in batches_jinja[0].prompts[0]

    def test_jinja2_with_missing_variable_renders_empty(self):
        """Jinja2 renders missing variables as empty strings (by design)."""
        df = pd.DataFrame({"name": ["Alice"]})
        template = "Hello {{ missing_column }} and {{ name }}"

        spec = PromptSpec(template=template)
        formatter = PromptFormatterStage(batch_size=10, use_jinja2=True)

        # Jinja2 (with autoescape=False) silently renders undefined as empty
        batches = formatter.process((df, spec), ExecutionContext())
        prompt = batches[0].prompts[0]
        # Should have name but missing_column will be empty
        assert "Alice" in prompt
        assert "Hello  and Alice" in prompt  # missing_column renders as empty

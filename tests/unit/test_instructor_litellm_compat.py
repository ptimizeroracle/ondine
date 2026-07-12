"""
Compatibility smoke tests for the instructor <-> litellm bridge.

Background
----------
instructor pins ``litellm<=1.83.7`` in its ``litellm``/``test-docs`` extras, but
that upper bound is *extra-only* — it is NOT enforced against ondine's core
``litellm>=1.91.1`` pin. ondine calls litellm both directly
(``litellm.completion`` / ``litellm.acompletion``) and through the instructor
bridge (``instructor.from_litellm`` in unified_litellm_client.py).

These tests are cheap insurance: if a future litellm bump silently breaks the
instructor bridge, CI catches it here instead of failing at runtime in a
user pipeline.

See DEPENDENCY_UPGRADE_ACTION_PLAN.md (Q1) for the full analysis.
"""

import importlib.metadata as importlib_metadata

import instructor
import litellm
import pytest


def _installed_version(package: str) -> str:
    """Return the installed distribution version for *package*."""
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


class TestInstructorLiteLLMCompat:
    """Assert the instructor/litellm versions ondine ships actually interoperate."""

    @pytest.fixture(autouse=True)
    def _record_versions(self, record_property):
        """Surface the installed versions in the CI/test report for triage."""
        record_property("instructor_version", _installed_version("instructor"))
        record_property("litellm_version", _installed_version("litellm"))

    def test_from_litellm_module_smoke(self):
        """instructor.from_litellm must accept the litellm module.

        This is the exact call shape used in
        ``UnifiedLiteLLMClient.__init__`` when no router is configured.
        """
        # instructor's stubs type the param as Callable, but the litellm module
        # itself exposes __getattr__-proxied callables — this is the documented
        # usage and matches the action plan example.
        client = instructor.from_litellm(litellm)  # type: ignore[arg-type]
        assert client is not None
        assert isinstance(client, instructor.Instructor)

    def test_from_litellm_callable_smoke(self):
        """instructor.from_litellm must accept litellm.acompletion.

        This is the call shape used when a LiteLLM router is present
        (``router.acompletion`` is passed instead of the module).
        """
        client = instructor.from_litellm(litellm.acompletion)
        assert client is not None
        assert isinstance(client, instructor.Instructor)

    def test_default_mode_is_instructor_enum(self):
        """The default mode produced by the bridge must be a valid Instructor Mode.

        Catches silent API drift where from_litellm stops returning a Mode enum
        and starts returning a string (which would break mode-detection logic).
        """
        client = instructor.from_litellm(litellm)  # type: ignore[arg-type]
        assert isinstance(client.mode, instructor.Mode)

"""Tests for cross-row contradiction detection."""

import pytest

from ondine.api.pipeline import _values_contradict
from ondine.context.memory_store import InMemoryContextStore
from ondine.context.protocol import EvidenceRecord
from ondine.context.rust_store import RustContextStore


@pytest.fixture(params=["rust", "memory"])
def store(request):
    if request.param == "rust":
        return RustContextStore(":memory:")
    return InMemoryContextStore()


class TestContradictionDetection:
    def test_add_and_get_contradictions(self, store):
        id_a = store.store(EvidenceRecord(text="Product X is Cereals"))
        id_b = store.store(EvidenceRecord(text="Product X is Snacks"))

        store.add_contradiction(id_a, id_b)

        assert id_b in store.get_contradictions(id_a)
        assert id_a in store.get_contradictions(id_b)

    def test_no_contradictions_by_default(self, store):
        cid = store.store(EvidenceRecord(text="Product Y is Beverages"))
        assert store.get_contradictions(cid) == []

    def test_multiple_contradictions(self, store):
        id_a = store.store(EvidenceRecord(text="X is A"))
        id_b = store.store(EvidenceRecord(text="X is B"))
        id_c = store.store(EvidenceRecord(text="X is C"))

        store.add_contradiction(id_a, id_b)
        store.add_contradiction(id_a, id_c)

        contras = store.get_contradictions(id_a)
        assert id_b in contras
        assert id_c in contras


class TestValuesContradict:
    """Tests for the tolerance-aware contradiction comparison."""

    def test_identical_values_never_contradict(self):
        assert _values_contradict("3.0", "3.0", None) is False
        assert _values_contradict("3.0", "3.0", 1) is False

    def test_no_tolerance_any_difference_is_contradiction(self):
        assert _values_contradict("3.0", "4.0", None) is True

    def test_tolerance_ignores_small_numeric_difference(self):
        assert _values_contradict("3.0", "4.0", 1) is False
        assert _values_contradict("3.0", "3.5", 1) is False

    def test_tolerance_catches_large_numeric_difference(self):
        assert _values_contradict("1.0", "5.0", 1) is True
        assert _values_contradict("0.0", "3.0", 2) is True

    def test_non_numeric_values_fall_back_to_exact_match(self):
        assert _values_contradict("Cereals", "Snacks", 1) is True
        assert _values_contradict("Cereals", "Cereals", 1) is False

    def test_boundary_at_tolerance_not_contradiction(self):
        # Exactly at tolerance boundary: abs(2.0 - 4.0) = 2.0, tolerance=2 → NOT contradiction
        assert _values_contradict("2.0", "4.0", 2) is False

    def test_just_above_tolerance_is_contradiction(self):
        # abs(2.0 - 4.1) = 2.1, tolerance=2 → contradiction
        assert _values_contradict("2.0", "4.1", 2) is True

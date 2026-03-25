"""Tests for cross-row contradiction detection."""

import pytest

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

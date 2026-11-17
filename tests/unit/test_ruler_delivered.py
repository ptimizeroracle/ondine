"""The Ruler's saveable view of a run: plain data out, the same view back.

``DeliveredValues.to_dict`` / ``from_dict`` carry a run across processes, so
values must round-trip exactly, and anything altered must be refused.

Design: ``.claude/designs/certified-columns.md`` section 13.0.1.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime

import pytest

from ondine.harness.status import RowStatus
from ondine.ruler import CertificationError
from ondine.ruler.delivered import DeliveredValues, _fingerprint

#: One of every plain type, so a round trip that turns 1 into 1.0 or True shows.
VALUES = {
    "intent": ("refund", None, "", " x ", "é"),
    "count": (1, 0, None, -7, 10**20),
    "score": (1.0, 0.1 + 0.2, None, -2.5e-9, 3.0),
    "flag": (True, False, None, True, False),
}


def _view(values=None) -> DeliveredValues:
    values = VALUES if values is None else values
    return DeliveredValues(
        run_id="3f1c",
        row_count=5,
        statuses=(RowStatus.ACCEPTED, RowStatus.FAILED) * 2 + (RowStatus.REPAIRED,),
        models=("m1", None, "m2", "m1", None),
        values=values,
        fingerprint=_fingerprint(values),
        checked=True,
        contract_fingerprint="abc123",
    )


def _round_trip(view: DeliveredValues) -> DeliveredValues:
    return DeliveredValues.from_dict(json.loads(json.dumps(view.to_dict())))


def test_every_plain_value_comes_back_equal_and_of_the_same_type():
    view = _view()

    back = _round_trip(view)

    assert back == view
    for name, cells in view.values.items():
        assert [type(v) for v in back.values[name]] == [type(v) for v in cells]


def test_an_unchecked_view_round_trips_with_no_contract():
    view = replace(_view(), checked=False, contract_fingerprint=None)

    assert _round_trip(view) == view


@pytest.mark.parametrize(
    "value",
    [["a", "b"], {"k": 1}, datetime(2026, 1, 1), float("inf"), float("-inf")],
)
def test_a_value_json_cannot_hold_exactly_is_refused_not_coerced(value):
    values = {"intent": ("refund", value)}
    view = replace(_view(values), row_count=2)

    with pytest.raises(CertificationError, match=r"row 1.*survive a save"):
        view.to_dict()


def test_a_changed_value_is_refused_by_the_fingerprint():
    data = json.loads(json.dumps(_view().to_dict()))
    data["values"]["intent"][0] = "billing"

    with pytest.raises(CertificationError, match="fingerprint"):
        DeliveredValues.from_dict(data)


def test_the_fingerprint_is_the_one_read_computes():
    # from_dict must hash as read() does, or a saved view never loads. The
    # view's own fingerprint here comes from the shared hashing function; an
    # independent copy of the formula pins it.
    import hashlib

    expected = hashlib.sha256(
        json.dumps(sorted(VALUES.items()), default=str, sort_keys=True).encode()
    ).hexdigest()

    assert _view().fingerprint == expected


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda d: d.update(version="delivered-v0"), "version"),
        (lambda d: d.pop("run_id"), "malformed"),
        (lambda d: d["statuses"].append("failed"), "malformed"),
        (lambda d: d["statuses"].__setitem__(0, "bogus"), "malformed"),
        (lambda d: d["values"]["count"].pop(), "malformed"),
    ],
)
def test_a_malformed_view_is_refused_with_a_clear_message(damage, message):
    data = copy.deepcopy(_view().to_dict())
    damage(data)

    with pytest.raises(CertificationError, match=message):
        DeliveredValues.from_dict(data)


# ---------------------------------------------------------------------------
# Integrity: the whole saved view, not only its values
# ---------------------------------------------------------------------------


def _data() -> dict:
    return json.loads(json.dumps(_view().to_dict()))


def test_changed_statuses_or_run_id_or_models_are_refused_by_the_digest():
    # None of these touch the values, so the fingerprint cannot see them; they
    # change what a certificate says (not_delivered, by_status, by_model, the
    # run it is bound to).
    for damage in (
        lambda d: d["statuses"].__setitem__(1, "accepted"),
        lambda d: d.update(run_id="other-run"),
        lambda d: d["models"].__setitem__(0, "m9"),
        lambda d: d.update(contract_fingerprint="def456"),
        lambda d: d.update(checked=False),
    ):
        data = _data()
        damage(data)

        with pytest.raises(CertificationError, match="digest"):
            DeliveredValues.from_dict(data)


def test_an_unknown_entry_added_to_the_saved_data_is_refused():
    data = _data()
    data["note"] = "hand edit"

    with pytest.raises(CertificationError, match="digest"):
        DeliveredValues.from_dict(data)


def test_an_untouched_view_passes_the_digest_after_a_json_round_trip():
    assert _round_trip(_view()) == _view()


# ---------------------------------------------------------------------------
# No coercion: an entry of another type is refused, not converted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.update(checked="false"),  # bool("false") is True
        lambda d: d.update(checked=1),
        lambda d: d.update(row_count=5.9),  # int(5.9) is 5
        lambda d: d.update(row_count="5"),
        lambda d: d.update(row_count=True),
        lambda d: d.update(run_id=3),
        lambda d: d.update(fingerprint=None),
        lambda d: d.update(contract_fingerprint=7),
        lambda d: d["models"].__setitem__(0, 5),
        lambda d: d["statuses"].__setitem__(0, 1),
        lambda d: d["values"].__setitem__("count", tuple(d["values"]["count"])),
        lambda d: d["values"].__setitem__("count", "abcde"),
        lambda d: d["values"]["count"].__setitem__(0, [1]),
        lambda d: d.update(digest=None),
    ],
)
def test_an_entry_of_another_type_is_refused_not_coerced(damage):
    data = _data()
    damage(data)

    with pytest.raises(CertificationError, match="malformed"):
        DeliveredValues.from_dict(data)


def test_a_view_with_no_contract_keeps_a_none_contract_fingerprint():
    view = replace(_view(), checked=False, contract_fingerprint=None)

    assert _round_trip(view).contract_fingerprint is None


# ---------------------------------------------------------------------------
# The fields a view must hold
# ---------------------------------------------------------------------------


def test_a_view_must_hold_exactly_the_fields_asked_for():
    view = _view()
    everything = list(VALUES)

    view.require_fields(everything[::-1])  # order is not part of it
    with pytest.raises(CertificationError, match="missing 'extra_field'"):
        view.require_fields([*everything, "extra_field"])
    with pytest.raises(CertificationError, match="extra 'flag'"):
        view.require_fields([name for name in everything if name != "flag"])

"""The judge: turns one raw model answer into a verdict, given the contract.

``judge(contract, answer, inputs)`` is the only entry point. For each output
field the verdict holds a deliverable value, "absent" (a valid answer for an
optional field), or "failed" with reasons; whether a format fix was applied;
and overall, which required fields failed.

Why the judge is built the way it is:

* **A bad answer is a verdict, not an error.** Model output is untrusted and
  often broken: empty, truncated, wrapped in prose, the wrong shape. The judge
  is *total*: it returns a verdict for every input and raises only on its own
  defects. A raise here would be a crashed run over one bad row; a swallowed
  defect would be a silent error, so defects still crash loudly.

* **It is deterministic and has no side effects.** No model call, no network,
  no shared state, no set-ordering or hash dependence. Resume re-judges cached
  answers in a new process and must reach the same verdicts; that is what lets
  resume work with no extra stored state.

* **Time grows with the answer, not beyond.** Parsing is linear and the
  format fix scans the answer once per allowed value. Every pattern was
  proven at build time to take linear time per match attempt (the contract
  refuses exponential and polynomial ones, without a timeout, which would
  make a verdict depend on machine speed). ``search`` retries an unanchored
  pattern from each start, so a value pattern only runs on values up to
  :data:`~ondine.harness.contract.MAX_PATTERN_INPUT` characters. Extraction
  patterns run on the whole answer; under the unchecked contract they are not
  checked at all, a known risk kept for parity (see ``build_contract``).

* **Order is fixed: parse, normalise, single-field rules cheapest first,
  field-to-field rules, then the schema's own validation.** A field stops at
  its first failed rule, and a relation runs only between fields that passed
  their own rules, so one bad field yields one reason instead of a cascade.
  The format fix runs on a failed field before relations, so relations see
  the fixed value; this is the same result as "check, fix, re-check all".

* **Under the unchecked contract it is today's parse step, exactly.** The
  parse stage now asks the judge for every answer, so this is the only copy
  of that step. Values and failure messages equal what ``ResponseParserStage``
  produced before the move (a frozen copy in
  ``tests/unit/test_parser_stage_judge.py`` pins it), including its
  fallbacks. One of them is a known silent-error source: a dict answer
  without the expected key delivers *whichever value comes first*. It is
  kept only under "unchecked", for byte-identical output; any real contract
  replaces it with a reasoned "missing" failure.
"""

from __future__ import annotations

import enum
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from ondine.harness._phrases import TYPE_PHRASES, schema_reason_text
from ondine.harness.contract import (
    MAX_PATTERN_INPUT,
    FieldRules,
    OutputContract,
    normalise,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _Missing:
    """A field the answer did not contain at all (distinct from null)."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING: Any = _Missing()

_ROLE_PREFIXES = ("assistant:", "user:", "system:")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_LONGEST_NUMBER_TEXT = 100
_PREVIEW_CHARS = 40
#: Words that change what a value right after them says: negations and degree
#: words. Doubt ("probably refund") is absent on purpose: it still names refund.
_QUALIFIERS = frozenset(
    {
        "almost",
        "barely",
        "broadly",
        "fairly",
        "generally",
        "hardly",
        "largely",
        "mainly",
        "mostly",
        "nearly",
        "neither",
        "never",
        "no",
        "nor",
        "not",
        "partially",
        "partly",
        "roughly",
        "slightly",
        "somewhat",
        "without",
    }
)
#: How many words before a value can qualify it ("not entirely consistent"),
#: and how many characters to read to find them, so that a check stays constant
#: time however long the answer is.
_QUALIFIER_WORDS = 3
_QUALIFIER_REACH = 80
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")
_COMPARE_WORDS = {
    "<": "less than",
    "<=": "at most",
    "==": "equal to",
    "!=": "different from",
    ">=": "at least",
    ">": "greater than",
}
_MIRROR = {"<": ">", "<=": ">=", "==": "==", "!=": "!=", ">=": "<=", ">": "<"}

# Why a field is missing, by what was wrong with the answer as a whole.
_NO_ANSWER = "no answer was returned"
_NOT_OBJECT = "missing, the answer is not a JSON object"
_NOT_IN_ANSWER = "missing from the answer"
_PARSER_FAILED = "missing, the parser could not read the answer"


@dataclass(frozen=True)
class ParsedAnswer:
    """An answer the client's own parser already read, with the text it read.

    A client may bring its own parser (``with_parser``, or structured output's
    parser). Its output is what the judge checks, but the judge cannot tell a
    parser that returned the string ``"42"`` from a raw answer ``"42"`` it
    must read itself, and a reason should quote what the model wrote, not the
    parser's output. Wrapping the two keeps both apart.

    Attributes:
        value: What the parser returned, of any type. Never read again: a
            ``str`` here is a value, not text to parse.
        text: The raw answer the parser read; failure reasons quote it.
        error: What the parser raised, as text, if it raised. ``value`` is
            then unused. Kept as text, not as the exception, so two equal
            failures give equal verdicts.
    """

    value: Any
    text: str
    error: str | None = None


@dataclass(frozen=True)
class Reason:
    """Why one check failed: the field, the rule, and a short human sentence.

    ``rule`` is one of: required, type, allowed_values, minimum, maximum,
    min_length, max_length, pattern, relation, schema, and, under the
    unchecked contract only, parse.
    """

    field: str
    rule: str
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class FieldVerdict:
    """The judge's result for one field.

    Attributes:
        name: The output field.
        outcome: "value" (deliverable), "absent" (a valid empty answer for an
            optional field) or "failed".
        value: The deliverable value. For a failed field, the best candidate
            the answer held, which the run delivers only under the
            deliver-flagged policy.
        reasons: Every failed check, in the order found. Empty unless failed.
        format_fixed: True when the value was found inside a wordy answer.
        required: Whether a failure of this field fails the row.
        fixed_reasons: For a format-fixed field, why the answer failed before
            the fix. The field passed, so ``reasons`` is empty; these keep the
            path taken, so a "repaired" row can say what was repaired.
    """

    name: str
    outcome: str
    value: Any = None
    reasons: tuple[Reason, ...] = ()
    format_fixed: bool = False
    required: bool = True
    fixed_reasons: tuple[Reason, ...] = ()


@dataclass(frozen=True)
class Verdict:
    """The judge's result for one answer: one :class:`FieldVerdict` per field.

    ``checked`` is False under the unchecked contract, where the values are
    today's parse output and every column counts as required.
    """

    checked: bool
    fields: tuple[FieldVerdict, ...]

    def __getitem__(self, name: str) -> FieldVerdict:
        for item in self.fields:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def values(self) -> dict[str, Any]:
        """What each column delivers: the value, or None if absent or failed."""
        return {
            item.name: item.value if item.outcome == "value" else None
            for item in self.fields
        }

    @property
    def failed_fields(self) -> tuple[str, ...]:
        """The required fields that failed, in column order."""
        return tuple(
            item.name
            for item in self.fields
            if item.outcome == "failed" and item.required
        )

    @property
    def passed(self) -> bool:
        """True when no required field failed."""
        return not self.failed_fields

    @property
    def format_fixed(self) -> bool:
        """True when any field was delivered through the format fix."""
        return any(item.format_fixed for item in self.fields)

    @property
    def reasons(self) -> tuple[Reason, ...]:
        """Every reason, in column order, each distinct sentence once."""
        seen: set[str] = set()
        ordered = []
        for item in self.fields:
            for reason in item.reasons:
                if reason.text not in seen:
                    seen.add(reason.text)
                    ordered.append(reason)
        return tuple(ordered)


def judge(
    contract: OutputContract,
    answer: Any,
    inputs: Mapping[str, Any] | None = None,
) -> Verdict:
    """Judge one raw answer against ``contract``.

    Args:
        contract: From :func:`~ondine.harness.contract.build_contract`.
        answer: The model's answer, one of: text (plain, JSON, or JSON in a
            code fence); an object already parsed (a dict, a structured-output
            Pydantic instance, or a custom parser's output); a
            :class:`ParsedAnswer` (a client parser's output with the text it
            read); or ``None`` when the provider returned no answer. A bare
            ``str`` is always read as raw text to parse.
        inputs: The row's input values, read by relations that name an input
            column.

    Returns:
        The verdict. Never raises because of the answer's content.
    """
    if not contract.checked:
        return _judge_unchecked(contract, answer)
    return _judge_checked(contract, answer, inputs or {})


# ---------------------------------------------------------------------------
# Unchecked: today's parse step, reproduced exactly
# ---------------------------------------------------------------------------


def _judge_unchecked(contract: OutputContract, answer: Any) -> Verdict:
    columns = list(contract.output_columns)
    if isinstance(answer, ParsedAnswer):
        text, failure = answer.text, answer.error
    else:
        text, failure = (answer if isinstance(answer, str) else str(answer)), None
    row: dict[str, Any]
    message: str | None
    try:
        if failure is not None:
            row, message = dict.fromkeys(columns), _unparsed(failure)
        else:
            if isinstance(answer, ParsedAnswer):
                parsed = answer.value
            elif isinstance(answer, str):
                parsed = _parse_like_today(contract, answer)
            else:
                parsed = answer
            row, message = _map_like_today(parsed, columns, text)
    except Exception as error:  # today's stage catches everything here too
        row, message = dict.fromkeys(columns), _unparsed(str(error))
    if message is None:
        fields = tuple(FieldVerdict(name, "value", row[name]) for name in columns)
    else:
        fields = tuple(
            FieldVerdict(name, "failed", row[name], (Reason(name, "parse", message),))
            for name in columns
        )
    return Verdict(checked=False, fields=fields)


def _unparsed(error: str) -> str:
    return f"response could not be parsed: {error}"


def _parse_like_today(contract: OutputContract, text: str) -> Any:
    """What the parser chosen by ``create_response_parser`` returns.

    The parse stage hands the judge the text whenever no client parser is
    configured, so this is where a pipeline's default parsing happens. The
    parser classes produce the same result for a client that uses them
    directly; ``tests/unit/test_parser_stage_judge.py`` pins the two together.
    """
    if contract.parse_mode == "text":
        return {"output": _clean_text(text)}
    if contract.parse_mode == "regex":
        result: dict[str, Any] = {}
        for rules in contract.rules:
            match = rules.extractor.search(text) if rules.extractor else None
            if match:
                result[rules.name] = (
                    match.group(1) if match.groups() else match.group(0)
                )
            else:
                result[rules.name] = None
        return result
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        for fence in ("```json", "```"):
            if fence in text:
                start = text.find(fence) + len(fence)
                end = text.find("```", start)
                return json.loads(text[start:end].strip())
        return {"output": text.strip()}


def _map_like_today(
    parsed: Any, columns: list[str], text: str
) -> tuple[dict[str, Any], str | None]:
    """Today's per-row column mapping, and the failure it records, if any.

    The only copy: ``ResponseParserStage`` used to hold this mapping and now
    asks the judge. The third single-column branch is the known silent-error
    source: a dict without the expected key delivers its first value.
    """
    if parsed is None:
        return dict.fromkeys(columns), None
    if len(columns) == 1:
        column = columns[0]
        if isinstance(parsed, dict) and column in parsed:
            return {column: parsed[column]}, None
        if isinstance(parsed, dict) and "output" in parsed:
            return {column: parsed["output"]}, None
        if isinstance(parsed, dict):
            return {column: next(iter(parsed.values()))}, None
        return {column: parsed}, None

    source: Any = parsed if isinstance(parsed, dict) else {}
    if not any(column in source for column in columns):
        if "items" in source and isinstance(source["items"], list):
            item = source["items"][0] if source["items"] else {}
            if isinstance(item, dict):
                source = item.get("result", item)
        elif "result" in source and isinstance(source["result"], dict):
            source = source["result"]
        if hasattr(source, "model_dump"):
            source = source.model_dump()
        elif not isinstance(source, dict):
            source = {}
    message = None
    if not any(column in source for column in columns):
        message = (
            f"response is not a JSON object with any of the output columns "
            f"{list(columns)}: {text.strip()[:120]!r}"
        )
    return {column: source.get(column) for column in columns}, message


def _clean_text(text: str) -> str:
    cleaned = text.strip()
    for prefix in _ROLE_PREFIXES:
        if cleaned.lower().startswith(prefix):
            return cleaned[len(prefix) :].strip()
    return cleaned


# ---------------------------------------------------------------------------
# Checked: extract, then check each field, then relations, then the schema
# ---------------------------------------------------------------------------


def _judge_checked(
    contract: OutputContract, answer: Any, inputs: Mapping[str, Any]
) -> Verdict:
    raws, missing_reason = _extract(contract, answer)
    verdicts = {
        rules.name: _check_field(rules, raws.get(rules.name, _MISSING), missing_reason)
        for rules in contract.rules
    }
    _check_relations(contract, verdicts, inputs)
    if contract.schema is not None:
        _check_schema(contract.schema, verdicts, raws)
    return Verdict(
        checked=True, fields=tuple(verdicts[name] for name in contract.output_columns)
    )


def _extract(contract: OutputContract, answer: Any) -> tuple[dict[str, Any], str]:
    """The raw value of each field the answer holds, and why others are missing.

    No guessing: a field is present only under its own name. The one
    structural allowance is the batch wrapper (``{"items": [{"result": …}]}``
    or ``{"result": …}``), which carries the same fields one level down.
    """
    columns = list(contract.output_columns)
    already_read = isinstance(answer, ParsedAnswer)
    if already_read:
        if answer.error is not None:
            return {}, f"{_PARSER_FAILED}: {answer.error}"
        answer = answer.value
    if answer is None:
        return {}, _NO_ANSWER
    if hasattr(answer, "model_dump") and not isinstance(answer, type):
        try:
            answer = answer.model_dump()
        except Exception:
            return {}, _NOT_OBJECT
    if isinstance(answer, str) and not already_read:
        if contract.parse_mode == "text":
            return {columns[0]: _clean_text(answer)}, _NOT_IN_ANSWER
        if contract.parse_mode == "regex":
            return _extract_by_pattern(contract, answer), _NOT_IN_ANSWER
        found, parsed = _loads_json(answer)
        if not found:
            if len(columns) == 1:
                return {columns[0]: answer.strip()}, _NOT_IN_ANSWER
            return {}, _NOT_OBJECT
        answer = parsed
    if isinstance(answer, Mapping):
        return _unwrap(dict(answer), columns), _NOT_IN_ANSWER
    if len(columns) == 1:
        return {columns[0]: answer}, _NOT_IN_ANSWER
    return {}, _NOT_OBJECT


def _extract_by_pattern(contract: OutputContract, text: str) -> dict[str, Any]:
    raws = {}
    for rules in contract.rules:
        match = rules.extractor.search(text) if rules.extractor else None
        if match:
            raws[rules.name] = match.group(1) if match.groups() else match.group(0)
    return raws


def _loads_json(text: str) -> tuple[bool, Any]:
    """JSON from the answer, or from its first code fence; never raises."""
    try:
        return True, json.loads(text.strip())
    except (ValueError, RecursionError):
        pass
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.find("```", start)
            try:
                return True, json.loads(text[start:end].strip())
            except (ValueError, RecursionError):
                return False, None
    return False, None


def _unwrap(source: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    if any(column in source for column in columns):
        return source
    inner: Any = source
    items = source.get("items")
    if isinstance(items, list):
        inner = items[0] if items else {}
        if isinstance(inner, dict):
            inner = inner.get("result", inner)
    elif isinstance(source.get("result"), dict):
        inner = source["result"]
    return inner if isinstance(inner, dict) else {}


def _check_field(rules: FieldRules, raw: Any, missing_reason: str) -> FieldVerdict:
    declaration = rules.declaration
    name = declaration.name
    required = declaration.required
    if raw is _MISSING:
        if not required:
            return FieldVerdict(name, "absent", required=False)
        return _failed(rules, None, "required", f"{name}: {missing_reason}")
    if declaration.type == "any":
        return FieldVerdict(name, "value", raw, required=required)
    if isinstance(raw, enum.Enum):
        raw = raw.value
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if not required:
            return FieldVerdict(name, "absent", required=False)
        return _failed(rules, raw, "required", f"{name}: is empty")
    if rules.absent_key is not None and _key(rules, raw) == rules.absent_key:
        return FieldVerdict(name, "absent", required=required)

    checked = _apply_rules(rules, raw)
    if checked.outcome == "failed" and rules.lookup and isinstance(raw, str):
        fixed = _format_fix(rules, raw)
        if fixed is not _MISSING:
            again = _apply_rules(rules, fixed)
            if again.outcome == "value":
                return replace(again, format_fixed=True, fixed_reasons=checked.reasons)
    return checked


def _apply_rules(rules: FieldRules, raw: Any) -> FieldVerdict:
    """Single-field rules, cheapest first, stopping at the first failure."""
    declaration = rules.declaration
    name = declaration.name
    value = _MISSING
    if rules.lookup:
        value = _lookup(rules, raw)
    if value is _MISSING:
        parsed, ok = _parse(declaration.type, raw)
        if not ok:
            phrase = TYPE_PHRASES[declaration.type]
            return _failed(
                rules, raw, "type", f"{name}: {_preview(raw)} is not {phrase}"
            )
        if declaration.type == "text":
            parsed = normalise(parsed, declaration.normalise)
        if rules.lookup:
            value = _lookup(rules, parsed)
            if value is _MISSING:
                return _failed(
                    rules,
                    raw,
                    "allowed_values",
                    f"{name}: {_preview(raw)} is not one of the "
                    f"{len(declaration.allowed or ())} allowed values",
                )
        else:
            value = parsed

    if declaration.minimum is not None and value < declaration.minimum:
        return _failed(
            rules,
            value,
            "minimum",
            f"{name}: {value} is below the minimum {declaration.minimum}",
        )
    if declaration.maximum is not None and value > declaration.maximum:
        return _failed(
            rules,
            value,
            "maximum",
            f"{name}: {value} is above the maximum {declaration.maximum}",
        )
    if declaration.type == "text":
        length = len(value)
        if declaration.min_length is not None and length < declaration.min_length:
            return _failed(
                rules,
                value,
                "min_length",
                f"{name}: is {length} characters long, shorter than the minimum "
                f"{declaration.min_length}",
            )
        if declaration.max_length is not None and length > declaration.max_length:
            return _failed(
                rules,
                value,
                "max_length",
                f"{name}: is {length} characters long, longer than the maximum "
                f"{declaration.max_length}",
            )
        if rules.pattern is not None:
            if length > MAX_PATTERN_INPUT:
                return _failed(
                    rules,
                    value,
                    "pattern",
                    f"{name}: is {length} characters long, too long to check "
                    f"against the pattern (limit {MAX_PATTERN_INPUT})",
                )
            if not rules.pattern.search(value):
                return _failed(
                    rules,
                    value,
                    "pattern",
                    f"{name}: {_preview(value)} does not match the pattern "
                    f"{declaration.pattern}",
                )
    return FieldVerdict(name, "value", value, required=declaration.required)


def _failed(rules: FieldRules, value: Any, rule: str, text: str) -> FieldVerdict:
    name = rules.name
    return FieldVerdict(
        name,
        "failed",
        value,
        (Reason(name, rule, text),),
        required=rules.declaration.required,
    )


def _key(rules: FieldRules, value: Any) -> Any:
    return rules.key(value) if isinstance(value, str) else value


def _lookup(rules: FieldRules, value: Any) -> Any:
    """The allowed value ``value`` names, or ``_MISSING``.

    True == 1 in Python, so a boolean only ever names a boolean.
    """
    try:
        hit = rules.lookup.get(_key(rules, value), _MISSING)
    except TypeError:  # an unhashable answer (a list, a dict) names nothing
        return _MISSING
    if hit is not _MISSING and isinstance(value, bool) != isinstance(hit, bool):
        return _MISSING
    return hit


def _parse(field_type: str, raw: Any) -> tuple[Any, bool]:
    """``raw`` as the declared type, and whether that worked."""
    if field_type == "text":
        if isinstance(raw, str):
            return raw, True
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return str(raw), True
        return None, False
    if field_type == "boolean":
        if isinstance(raw, bool):
            return raw, True
        if isinstance(raw, str) and raw.strip().casefold() in ("true", "false"):
            return raw.strip().casefold() == "true", True
        return None, False
    if field_type in ("integer", "number"):
        return _parse_number(raw, whole=field_type == "integer")
    return raw, True


def _parse_number(raw: Any, *, whole: bool) -> tuple[Any, bool]:
    """``raw`` as an int or float, and whether that worked.

    Text becomes the same Python number ``json.loads`` gives for that text:
    "3" is 3, "3.0" and "1e3" are floats. A number column then holds the same
    value whether the model wrote 3 or "3".
    """
    written_whole = isinstance(raw, str) and not any(c in raw for c in ".eE")
    if isinstance(raw, bool):
        return None, False
    if isinstance(raw, int):
        return raw, True
    if isinstance(raw, float):
        if not math.isfinite(raw):
            return None, False
        if whole:
            return (int(raw), True) if raw.is_integer() else (None, False)
        return raw, True
    if isinstance(raw, str):
        text = raw.strip()
        if len(text) > _LONGEST_NUMBER_TEXT or not _NUMBER.fullmatch(text):
            return None, False
        try:
            number = Decimal(text)
        except InvalidOperation:  # pragma: no cover - the pattern rules it out
            return None, False
        raw = number
    if isinstance(raw, Decimal):
        if not raw.is_finite() or raw.adjusted() > _LONGEST_NUMBER_TEXT:
            return None, False
        if whole or (written_whole and raw == raw.to_integral_value()):
            if raw != raw.to_integral_value():
                return None, False
            return int(raw), True
        as_float = float(raw)
        return (as_float, True) if math.isfinite(as_float) else (None, False)
    return None, False


def _format_fix(rules: FieldRules, raw: str) -> Any:
    """The one allowed value written inside a wordy answer, or ``_MISSING``.

    Every allowed value and alias is searched for as a whole word. A match
    inside a longer match ("card" inside "card lost") does not count. Exactly
    one distinct allowed value must remain; two or more is ambiguous and the
    field stays failed. Numbers must stand alone: "2" is not in "12", "-2" or
    "2.5". A value the answer qualifies stays failed too: "not consistent" and
    "mostly consistent" do not say "consistent", and delivering it would state
    what the model denied or doubted.
    """
    numeric = rules.declaration.type in ("integer", "number")
    haystack = rules.key(raw)
    spans: list[tuple[int, int, Any]] = []
    for key, allowed in rules.lookup.items():
        needle = key if isinstance(key, str) else str(key)
        if not needle:
            continue
        start = haystack.find(needle)
        while start != -1:
            end = start + len(needle)
            if _stands_alone(haystack, start, end, numeric=numeric):
                spans.append((start, -end, allowed))
            start = haystack.find(needle, start + 1)

    found: list[Any] = []
    furthest = -1
    for start, negative_end, allowed in sorted(spans, key=lambda span: span[:2]):
        end = -negative_end
        if end <= furthest:
            continue  # inside a longer match that starts at or before it
        furthest = end
        if _is_qualified(haystack, start):
            return _MISSING
        if allowed not in found:
            found.append(allowed)
    return found[0] if len(found) == 1 else _MISSING


def _is_qualified(text: str, start: int) -> bool:
    """Does a negation or a degree word lead up to the value at ``start``?

    Looks at the few words before the value, within its own sentence or
    clause. English only: in another language the answer is read as if
    unqualified.
    """
    before = text[max(0, start - _QUALIFIER_REACH) : start].casefold()
    clause = re.split(r"[.,;:!?\n]", before)[-1]
    return any(
        word in _QUALIFIERS or word.endswith(("n't", "n’t"))
        for word in _WORD.findall(clause)[-_QUALIFIER_WORDS:]
    )


def _stands_alone(text: str, start: int, end: int, *, numeric: bool) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before and (before.isalnum() or before == "_"):
        return False
    if after and (after.isalnum() or after == "_"):
        return False
    if numeric:
        if before and before in "+-.":
            return False
        following = text[end + 1] if end + 1 < len(text) else ""
        if after in (".", ",") and following.isdigit():
            return False
    return True


def _check_relations(
    contract: OutputContract,
    verdicts: dict[str, FieldVerdict],
    inputs: Mapping[str, Any],
) -> None:
    """Field-to-field rules, only between fields that passed their own rules."""
    passed = {name for name, item in verdicts.items() if item.outcome == "value"}
    failures: list[Reason] = []
    for relation in contract.relations:
        left = relation.left
        if left not in passed:
            continue
        left_value = verdicts[left].value
        words = _COMPARE_WORDS[relation.op]
        if relation.right is not None:
            right = relation.right
            if right not in passed:
                continue
            right_value = verdicts[right].value
            holds = _compare(left_value, relation.op, right_value)
            if holds:
                continue
            if holds is None:
                text = f"{left}: {_show(left_value)} cannot be compared with {right}"
                failures.append(Reason(left, "relation", text))
                continue
            mirror = _COMPARE_WORDS[_MIRROR[relation.op]]
            failures.append(
                Reason(
                    left,
                    "relation",
                    f"{left}: {_show(left_value)} is not {words} {right} "
                    f"({_show(right_value)})",
                )
            )
            failures.append(
                Reason(
                    right,
                    "relation",
                    f"{right}: {_show(right_value)} is not {mirror} {left} "
                    f"({_show(left_value)})",
                )
            )
            continue

        column = relation.input_column
        label = f"input column {column}"
        other = inputs.get(column, None) if column is not None else None
        if other is None or (isinstance(other, float) and math.isnan(other)):
            failures.append(
                Reason(
                    left, "relation", f"{left}: cannot be compared, {label} is empty"
                )
            )
            continue
        left_type = contract.rules_for(left).declaration.type
        compare_type = "number" if left_type in ("integer", "number") else left_type
        parsed, ok = _parse(compare_type, other)
        holds = _compare(left_value, relation.op, parsed) if ok else None
        if holds is None:
            failures.append(
                Reason(
                    left,
                    "relation",
                    f"{left}: cannot be compared, {label} holds {_preview(other)}, "
                    f"which is not {TYPE_PHRASES.get(compare_type, 'comparable')}",
                )
            )
        elif not holds:
            failures.append(
                Reason(
                    left,
                    "relation",
                    f"{left}: {_show(left_value)} is not {words} {label} "
                    f"({_show(parsed)})",
                )
            )
    _attach(verdicts, failures)


def _compare(left: Any, op: str, right: Any) -> bool | None:
    try:
        return bool(
            {
                "<": lambda: left < right,
                "<=": lambda: left <= right,
                "==": lambda: left == right,
                "!=": lambda: left != right,
                ">=": lambda: left >= right,
                ">": lambda: left > right,
            }[op]()
        )
    except TypeError:
        return None


def _check_schema(
    schema: type, verdicts: dict[str, FieldVerdict], raws: Mapping[str, Any]
) -> None:
    """Run the client's own model validation; its messages become reasons.

    Runs last, as the most expensive check. Only fields that passed every
    declarative rule can gain a reason here, so a field never carries both a
    declarative and a schema reason for the same fault. A validator that
    crashes is a reason on every passing field, never a crashed run.
    """
    from pydantic import ValidationError

    model_fields: Mapping[str, Any] = getattr(schema, "model_fields", {})
    candidate: dict[str, Any] = {}
    for name, info in model_fields.items():
        item = verdicts.get(name)
        if item is None:
            continue
        if item.outcome == "value":
            candidate[name] = item.value
        elif item.outcome == "failed" and name in raws:
            candidate[name] = raws[name]
        elif item.outcome == "absent" and info.is_required():
            candidate[name] = None

    passing = [name for name, item in verdicts.items() if item.outcome == "value"]
    failures: list[Reason] = []
    try:
        schema.model_validate(candidate)  # type: ignore[attr-defined]
        return
    except ValidationError as error:
        problems = error.errors()
    except Exception as error:  # a client validator's own bug
        for name in passing:
            failures.append(
                Reason(
                    name,
                    "schema",
                    f"{name}: schema validation raised {type(error).__name__}: {error}",
                )
            )
        _attach(verdicts, failures)
        return

    for problem in problems:
        location = problem.get("loc") or ()
        message = schema_reason_text(str(problem.get("msg", "")))
        target = location[0] if location and location[0] in verdicts else None
        targets = [target] if target is not None else passing
        for name in targets:
            if name in passing:
                failures.append(Reason(name, "schema", f"{name}: {message}"))
    _attach(verdicts, failures)


def _attach(verdicts: dict[str, FieldVerdict], failures: Sequence[Reason]) -> None:
    """Fail each named field, adding its reasons in order, each sentence once."""
    for reason in failures:
        item = verdicts[reason.field]
        if reason in item.reasons:
            continue
        verdicts[reason.field] = replace(
            item, outcome="failed", reasons=(*item.reasons, reason)
        )


def _preview(value: Any) -> str:
    """A short, one-line rendering of a value for a reason sentence."""
    if isinstance(value, str):
        text = " ".join(value[: _PREVIEW_CHARS * 4].split())
        if len(text) > _PREVIEW_CHARS:
            text = text[:_PREVIEW_CHARS].rstrip() + "…"
        return repr(text)
    text = repr(value)
    if len(text) > _PREVIEW_CHARS:
        text = text[:_PREVIEW_CHARS].rstrip() + "…"
    return text


def _show(value: Any) -> str:
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return str(value)
    return _preview(value)

"""The output contract: what the client declared a deliverable value to be.

A contract says, per output field, what a model answer must look like before
Ondine delivers it: its type, whether it is required (or which value means
"absent"), the rules it must satisfy, and the cleanups allowed before checking.
It also holds the run policy: reject or deliver flagged, keep partial rows or
not, and when to stop a run that is rejecting most of its rows.

Three properties make the contract the backbone of the harness:

* **It fails before money is spent.** :func:`build_contract` checks the whole
  declaration and raises :class:`ContractError` listing *every* problem at
  once: a minimum above its maximum, a rule naming an unknown field, two
  allowed values that collide after normalisation, a pattern that could take
  unbounded time. A client fixes a broken contract in one pass, not one paid
  run per typo.

* **It is frozen and fingerprinted.** Resume re-judges cached answers, so it
  must refuse a contract that changed since the first attempt. The fingerprint
  is a SHA-256 over a canonical rendering of every declaration, so it is the
  same in every process (it never uses ``hash()``, which Python randomises per
  process) and moves when any declaration moves.

* **Nothing declared is still a contract.** With no field, relation or schema,
  :func:`build_contract` returns the *unchecked* contract. The judge then
  reproduces today's parsing exactly, so every caller handles one kind of
  object and "no contract" stays byte-identical to the pipeline before the
  harness existed.

What this module hides: how rules are represented (:class:`FieldRules`, the
compiled form the judge reads), how normalisation and aliases collapse into one
lookup, how a pattern is proven safe, and how a Pydantic schema is read.
"""

from __future__ import annotations

import enum
import functools
import hashlib
import json
import re
import string
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Union

from ondine.harness._phrases import type_phrase

try:  # Python 3.11+
    import re._parser as _sre_parse  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Python 3.10
    import sre_parse as _sre_parse  # type: ignore[no-redef]

# Field types a declaration may name. "any" is never declared by a client: it
# is what an undeclared output column, or a schema field Ondine cannot read
# (a nested model, a list), becomes — present or not, it passes.
FIELD_TYPES = ("text", "integer", "number", "boolean", "any")

# Cleanups a client may allow before checking. They always run in this order,
# whatever order they were declared in, so two declarations listing the same
# cleanups mean the same thing and share a fingerprint.
NORMALISATIONS = ("strip", "collapse_whitespace", "strip_punctuation", "casefold")

COMPARISONS = ("<", "<=", "==", "!=", ">=", ">")

# A pattern is only run on values up to this many characters. Build-time
# analysis proves that one match attempt takes time linear in the value (it
# refuses exponential and polynomial patterns). ``search`` retries an
# unanchored pattern from every start, which can square that; this cap bounds
# it. A pattern anchored with ``^`` makes one attempt and stays linear.
MAX_PATTERN_INPUT = 10_000

_PUNCTUATION = string.punctuation + string.whitespace + "“”‘’«»…"


class ContractError(ValueError):
    """A contract that cannot be satisfied or cannot be run safely.

    Raised by :func:`build_contract` with every problem found, never only the
    first, because each fix otherwise costs the client another build.
    """

    def __init__(self, problems: Sequence[str]):
        self.problems: tuple[str, ...] = tuple(problems)
        count = len(self.problems)
        noun = "problem" if count == 1 else "problems"
        listing = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"The output contract has {count} {noun}:\n{listing}")


@dataclass(frozen=True)
class OutputField:
    """The declaration of one output field.

    Attributes:
        name: The output column this declaration governs.
        type: "text" (default), "integer", "number" or "boolean".
        required: A required field must hold a value. An optional one may be
            absent: missing, null, empty, or equal to ``absent``.
        absent: For an optional field, the answer that means "no value"
            (for example "n/a"). It is delivered as empty, never as itself.
        allowed: The only values the field may take. ``None`` means no such
            rule; an empty collection is a contract problem.
        aliases: Other spellings, each mapped to one allowed value. An alias
            is delivered as the allowed value it names.
        minimum, maximum: Inclusive bounds, for integer and number fields.
        min_length, max_length: Bounds on a text value's character count.
        pattern: A regular expression the text value must contain a match
            for (search semantics, as Pydantic uses: anchor with ``^...$``).
        normalise: Cleanups allowed before checking, from
            :data:`NORMALISATIONS`. A text value is delivered cleaned.

    Collections are copied on creation, so a caller mutating its own list
    afterwards cannot change a contract built from this declaration.
    """

    name: str
    type: str = "text"
    required: bool = True
    absent: Any = None
    allowed: Sequence[Any] | None = None
    aliases: Mapping[str, Any] = field(default_factory=dict)
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    normalise: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.allowed is not None:
            object.__setattr__(self, "allowed", tuple(self.allowed))
        object.__setattr__(self, "aliases", types.MappingProxyType(dict(self.aliases)))
        known = [step for step in NORMALISATIONS if step in self.normalise]
        unknown = [step for step in self.normalise if step not in NORMALISATIONS]
        object.__setattr__(self, "normalise", tuple(known + unknown))


@dataclass(frozen=True)
class Relation:
    """A field-to-field rule: ``left <op> right`` or ``left <op> input column``.

    Exactly one of ``right`` (another output field) and ``input_column`` (a
    column of the row's inputs) is given. The judge evaluates a relation only
    when every output field it names passed its own rules, so one bad field
    produces one reason, not a cascade.
    """

    left: str
    op: str
    right: str | None = None
    input_column: str | None = None

    def describe(self) -> str:
        """The relation as a client would write it, for reasons and problems."""
        if self.input_column is not None and self.right is None:
            return f"{self.left} {self.op} input column {self.input_column}"
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True)
class RunPolicy:
    """Run-level choices, with the defaults decided in the harness design.

    Attributes:
        on_failure: "reject" (default) blanks a row that fails its checks;
            "deliver_flagged" delivers it marked unverified.
        partial_rows: "whole_row" (default) rejects a row when any required
            field fails; "keep_valid_fields" delivers the fields that passed.
        early_stop_fraction: Stop the run when more than this share of the
            first ``early_stop_window`` rows is rejected. ``None`` turns early
            stop off.
        early_stop_window: How many rows, by input order, the fraction is
            measured over.
    """

    on_failure: str = "reject"
    partial_rows: str = "whole_row"
    early_stop_fraction: float | None = 0.5
    early_stop_window: int = 50


@dataclass(frozen=True)
class FieldRules:
    """The compiled form of one field's declaration, read by the judge.

    Normalisation, allowed values and aliases collapse into one ``lookup``
    from a normalised key to the allowed value delivered, so checking a label
    is one dictionary lookup whatever the client declared.

    Attributes:
        declaration: The field as declared (or implied, for an undeclared
            output column under a checked contract).
        lookup: Normalised key of every allowed value and alias, mapped to the
            allowed value it delivers. Empty when there is no allowed rule.
        absent_key: Normalised key of the absent marker, or ``None``.
        pattern: The compiled ``declaration.pattern``.
        extractor: The compiled ``response_format='regex'`` pattern that
            pulls this field out of a text answer, or ``None``.
    """

    declaration: OutputField
    lookup: Mapping[Any, Any]
    absent_key: Any
    pattern: re.Pattern[str] | None
    extractor: re.Pattern[str] | None

    @property
    def name(self) -> str:
        return self.declaration.name

    def key(self, text: str) -> str:
        """The form of ``text`` that lookups and comparisons use.

        Text fields apply their declared cleanups; other fields only strip
        surrounding whitespace, since "42 " is plainly the number 42.
        """
        if self.declaration.type == "text":
            return normalise(text, self.declaration.normalise)
        return text.strip()


@dataclass(frozen=True)
class OutputContract:
    """A validated, frozen contract. Build it with :func:`build_contract`.

    Attributes:
        output_columns: Every output column, in order.
        fields: One declaration per output column, in the same order. Under a
            checked contract an undeclared column is an optional "any" field.
        relations: Field-to-field rules.
        policy: The run policy.
        parse_mode: How a text answer is read: "text" (one field, the answer
            is the value), "json" (an object, possibly in a code fence) or
            "regex" (one extraction pattern per field). Resolved from the
            response format exactly as the parser factory does today.
        regex_patterns: The extraction patterns, as (column, pattern) pairs.
        schema: The Pydantic model the contract was derived from, if any. Its
            validators run in the judge, and their messages become reasons.
        checked: False for the unchecked contract (nothing declared).
        fingerprint: Stable across processes; equal iff declarations are.
    """

    output_columns: tuple[str, ...]
    fields: tuple[OutputField, ...]
    relations: tuple[Relation, ...]
    policy: RunPolicy
    parse_mode: str
    regex_patterns: tuple[tuple[str, str], ...]
    schema: type | None
    checked: bool
    fingerprint: str
    rules: tuple[FieldRules, ...] = field(repr=False, compare=False)

    def rules_for(self, name: str) -> FieldRules:
        """The compiled rules of output field ``name``."""
        for rules in self.rules:
            if rules.name == name:
                return rules
        raise KeyError(name)


def normalise(text: str, steps: Sequence[str]) -> str:
    """Apply the declared cleanups to ``text``, always in canonical order."""
    if "strip" in steps:
        text = text.strip()
    if "collapse_whitespace" in steps:
        text = " ".join(text.split())
    if "strip_punctuation" in steps:
        text = text.strip(_PUNCTUATION)
    if "casefold" in steps:
        text = text.casefold()
    return text


def build_contract(
    output_columns: Sequence[str] | None = None,
    *,
    fields: Sequence[OutputField] = (),
    relations: Sequence[Relation] = (),
    policy: RunPolicy | None = None,
    schema: type | None = None,
    input_columns: Sequence[str] | None = None,
    response_format: str = "raw",
    regex_patterns: Mapping[str, str] | None = None,
) -> OutputContract:
    """Validate the declarations and return a frozen contract.

    Args:
        output_columns: The pipeline's output columns. Defaults to the
            declared fields (or the schema's fields), in order.
        fields: Plain declarations, one per output column at most.
        relations: Field-to-field rules.
        policy: Run policy; the decided defaults when omitted.
        schema: Opt-in: a Pydantic model whose types, ``Literal``/``Enum``
            values, bounds, patterns and lengths become rules, and whose
            validators run in the judge.
        input_columns: The row's input columns, which relations may name.
        response_format: "raw", "json" or "regex", as on the prompt spec.
        regex_patterns: Extraction patterns for ``response_format='regex'``.
            They run on the whole answer, so under a checked contract they
            must pass the same safety check as field patterns. Under the
            unchecked contract they are compiled but not checked, exactly as
            today's regex parser runs them: a known, pre-existing risk (a
            pattern like ``(a+)+b`` can hang on a long answer), kept so that
            "no contract" stays byte-identical. Declaring any field closes it.

    Returns:
        The checked contract, or the unchecked one when nothing is declared.

    Raises:
        ContractError: listing every problem found.
    """
    problems: list[str] = []
    declared = list(fields)
    if schema is not None:
        derived, schema_problems = _fields_from_schema(schema)
        problems.extend(schema_problems)
        explicit = {item.name for item in declared}
        for item in derived:
            if item.name in explicit:
                problems.append(
                    f"{item.name}: declared both in the schema and as a field"
                )
        declared = [item for item in derived if item.name not in explicit] + declared

    columns = _output_columns(output_columns, declared, problems)
    checked = bool(declared or relations or schema is not None)
    policy = policy or RunPolicy()
    parse_mode = _parse_mode(response_format, len(columns), problems)
    patterns = dict(regex_patterns or {})
    if parse_mode == "regex":
        missing = [column for column in columns if column not in patterns]
        if missing:
            problems.append(
                f"response_format 'regex' needs a pattern for: {', '.join(missing)}"
            )

    by_name: dict[str, OutputField] = {}
    for item in declared:
        if item.name in by_name:
            problems.append(f"{item.name}: declared twice")
        elif item.name not in columns:
            problems.append(f"{item.name}: not an output column {list(columns)}")
        else:
            by_name[item.name] = item
    resolved = tuple(
        by_name.get(column) or OutputField(column, type="any", required=False)
        for column in columns
    )

    rules = []
    for item in resolved:
        compiled, field_problems = _compile_field(
            item, patterns.get(item.name), checked=checked
        )
        problems.extend(field_problems)
        rules.append(compiled)
    every_field = {item.name: item for item in resolved}
    for relation in relations:
        problems.extend(_relation_problems(relation, every_field, input_columns))
    if checked:
        problems.extend(policy_problems(policy))
    if problems:
        raise ContractError(problems)

    pattern_pairs = tuple(sorted(patterns.items()))
    return OutputContract(
        output_columns=columns,
        fields=resolved,
        relations=tuple(relations),
        policy=policy,
        parse_mode=parse_mode,
        regex_patterns=pattern_pairs,
        schema=schema,
        checked=checked,
        fingerprint=_fingerprint(
            columns,
            resolved,
            relations,
            policy,
            parse_mode,
            pattern_pairs,
            schema,
            checked,
        ),
        rules=tuple(rules),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _output_columns(
    output_columns: Sequence[str] | None,
    declared: list[OutputField],
    problems: list[str],
) -> tuple[str, ...]:
    if output_columns is None:
        columns: list[str] = []
        for item in declared:
            if item.name not in columns:
                columns.append(item.name)
    else:
        columns = list(output_columns)
    if not columns:
        problems.append("the contract names no output column")
    return tuple(columns)


def _parse_mode(response_format: str, column_count: int, problems: list[str]) -> str:
    """Resolve the parse mode exactly as ``create_response_parser`` does.

    "raw" with several columns reads JSON, because raw text can only fill one
    column; this mirrors the parser factory so the unchecked contract parses
    every answer the way today's pipeline does.
    """
    fmt = response_format.lower()
    if fmt == "raw":
        return "json" if column_count > 1 else "text"
    if fmt in ("json", "regex"):
        return fmt
    problems.append(
        f"unknown response_format {response_format!r}; use 'raw', 'json' or 'regex'"
    )
    return "text"


def _compile_field(
    item: OutputField, extraction: str | None, *, checked: bool
) -> tuple[FieldRules, list[str]]:
    """Check one declaration and compile it, collecting every problem.

    ``checked`` decides whether the extraction pattern must be proven safe:
    always under a checked contract, never under the unchecked one (parity
    with today's parser; see :func:`build_contract`).
    """
    name = item.name
    problems: list[str] = []
    if item.type not in FIELD_TYPES:
        problems.append(
            f"{name}: unknown type {item.type!r}; use one of {', '.join(FIELD_TYPES)}"
        )
    is_text = item.type == "text"
    numeric = item.type in ("integer", "number")

    if item.required and item.absent is not None:
        problems.append(f"{name}: a required field cannot have an absent marker")
    unknown = [step for step in item.normalise if step not in NORMALISATIONS]
    for step in unknown:
        problems.append(
            f"{name}: unknown normalisation {step!r}; use {', '.join(NORMALISATIONS)}"
        )
    if item.normalise and not is_text:
        problems.append(f"{name}: normalisations apply to text fields only")

    if (item.minimum is not None or item.maximum is not None) and not numeric:
        problems.append(
            f"{name}: minimum and maximum apply to integer and number fields only"
        )
    for label, bound in (("minimum", item.minimum), ("maximum", item.maximum)):
        if bound is not None and not _is_number(bound):
            problems.append(f"{name}: {label} {bound!r} is not a number")
    if (
        _is_number(item.minimum)
        and _is_number(item.maximum)
        and item.minimum > item.maximum  # type: ignore[operator]
    ):
        problems.append(
            f"{name}: minimum {item.minimum} is above the maximum {item.maximum}"
        )

    for label, length in (
        ("min_length", item.min_length),
        ("max_length", item.max_length),
    ):
        if length is not None and (
            isinstance(length, bool) or not isinstance(length, int) or length < 0
        ):
            problems.append(f"{name}: {label} must be a whole number of at least 0")
    if (item.min_length is not None or item.max_length is not None) and not is_text:
        problems.append(f"{name}: min_length and max_length apply to text fields only")
    if (
        isinstance(item.min_length, int)
        and isinstance(item.max_length, int)
        and item.min_length > item.max_length
    ):
        problems.append(
            f"{name}: min_length {item.min_length} is above max_length {item.max_length}"
        )

    pattern = None
    if item.pattern is not None:
        if not is_text:
            problems.append(f"{name}: pattern applies to text fields only")
        pattern = _compile_pattern(f"{name}: pattern", item.pattern, problems)
    extractor = None
    if extraction is not None:
        extractor = _compile_pattern(
            f"{name}: extraction pattern", extraction, problems, check_safety=checked
        )

    def key(value: Any) -> Any:
        if isinstance(value, str):
            return normalise(value, item.normalise) if is_text else value.strip()
        return value

    lookup: dict[Any, Any] = {}
    if item.allowed is not None:
        if not item.allowed:
            problems.append(f"{name}: the allowed values are empty")
        for value in item.allowed:
            if not _fits_type(item.type, value):
                problems.append(
                    f"{name}: allowed value {value!r} is not {type_phrase(item.type)}"
                )
                continue
            value_key = key(value)
            if not _hashable(value_key):
                problems.append(_unhashable(f"{name}: allowed value", value))
                continue
            if value_key in lookup:
                problems.append(
                    f"{name}: allowed values {lookup[value_key]!r} and {value!r} "
                    "are the same after normalisation"
                )
                continue
            lookup[value_key] = value
        allowed_keys = dict(lookup)
        alias_targets: dict[Any, Any] = {}
        for alias, target in item.aliases.items():
            if target not in item.allowed:
                problems.append(
                    f"{name}: alias {alias!r} maps to {target!r}, "
                    "which is not an allowed value"
                )
                continue
            alias_key = key(alias)
            if alias_key in allowed_keys and allowed_keys[alias_key] != target:
                problems.append(
                    f"{name}: alias {alias!r} maps to {target!r} but is also the "
                    f"allowed value {allowed_keys[alias_key]!r}"
                )
            elif alias_key in alias_targets and alias_targets[alias_key] != target:
                problems.append(
                    f"{name}: alias {alias!r} maps to two values, "
                    f"{alias_targets[alias_key]!r} and {target!r}"
                )
            else:
                alias_targets[alias_key] = target
                lookup[alias_key] = target
    elif item.aliases:
        problems.append(f"{name}: aliases need allowed values to map to")

    absent_key = None
    if item.absent is not None:
        absent_key = key(item.absent)
        if not _hashable(absent_key):
            problems.append(_unhashable(f"{name}: the absent marker", item.absent))
            absent_key = None
        elif absent_key in lookup:
            problems.append(
                f"{name}: the absent marker {item.absent!r} is also an allowed "
                "value or alias"
            )

    rules = FieldRules(
        declaration=item,
        lookup=types.MappingProxyType(lookup),
        absent_key=absent_key,
        pattern=pattern,
        extractor=extractor,
    )
    return rules, problems


def _compile_pattern(
    label: str, pattern: str, problems: list[str], *, check_safety: bool = True
) -> re.Pattern[str] | None:
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        problems.append(
            f"{label} {pattern!r} is not a valid regular expression: {error}"
        )
        return None
    if check_safety:
        unsafe = _unsafe_pattern_reason(pattern)
        if unsafe:
            problems.append(
                f"{label} {pattern!r} is not safe on long answers: {unsafe}"
            )
            return None
    return compiled


def _relation_problems(
    relation: Relation,
    fields: Mapping[str, OutputField],
    input_columns: Sequence[str] | None,
) -> list[str]:
    label = f"relation {relation.describe()}"
    problems: list[str] = []
    if relation.right is None and relation.input_column is None:
        return [
            f"relation on {relation.left} names neither an output field nor an "
            "input column"
        ]
    if relation.right is not None and relation.input_column is not None:
        return [
            f"relation on {relation.left} names both an output field and an "
            "input column"
        ]
    if relation.op not in COMPARISONS:
        problems.append(
            f"{label}: unknown comparison {relation.op!r}; use {' '.join(COMPARISONS)}"
        )
    left = fields.get(relation.left)
    if left is None:
        problems.append(f"{label}: {relation.left!r} is not an output column")
    if relation.right is not None:
        right = fields.get(relation.right)
        if right is None:
            problems.append(f"{label}: {relation.right!r} is not an output column")
        elif left is not None:
            problems.extend(_type_problems(label, relation.op, left.type, right.type))
    elif input_columns is None or relation.input_column not in input_columns:
        problems.append(f"{label}: {relation.input_column!r} is not an input column")
    elif (
        left is not None and left.type == "boolean" and relation.op not in ("==", "!=")
    ):
        problems.append(f"{label} orders true/false values")
    return problems


def _type_problems(label: str, op: str, left: str, right: str) -> list[str]:
    families = {"integer": "number", "number": "number"}
    left_family = families.get(left, left)
    right_family = families.get(right, right)
    if "any" in (left_family, right_family):
        return []
    if left_family != right_family:
        return [f"{label} compares {type_phrase(left)} with {type_phrase(right)}"]
    if left_family == "boolean" and op not in ("==", "!="):
        return [f"{label} orders true/false values"]
    return []


def policy_problems(policy: RunPolicy) -> list[str]:
    """Every reason `policy` is unusable, or an empty list.

    Public because the ledger validates the policy it is given by the same
    rule the contract builder uses; one rule, one place.
    """
    problems = []
    if policy.on_failure not in ("reject", "deliver_flagged"):
        problems.append(
            f"policy on_failure {policy.on_failure!r}: use 'reject' or 'deliver_flagged'"
        )
    if policy.partial_rows not in ("whole_row", "keep_valid_fields"):
        problems.append(
            f"policy partial_rows {policy.partial_rows!r}: use 'whole_row' or "
            "'keep_valid_fields'"
        )
    fraction = policy.early_stop_fraction
    if fraction is not None and not (_is_number(fraction) and 0 <= fraction < 1):
        problems.append(
            f"policy early_stop_fraction {fraction!r}: use a share from 0 up to "
            "(not including) 1, or None to turn early stop off"
        )
    window = policy.early_stop_window
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        problems.append(f"policy early_stop_window {window!r}: use at least 1 row")
    return problems


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _fits_type(field_type: str, value: Any) -> bool:
    if field_type == "text":
        return isinstance(value, str)
    if field_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "number":
        return _is_number(value)
    if field_type == "boolean":
        return isinstance(value, bool)
    return True


def _hashable(value: Any) -> bool:
    """Whether ``value`` can be a lookup key (a list or a dict cannot)."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _unhashable(label: str, value: Any) -> str:
    return (
        f"{label} {value!r} is a {type(value).__name__}, which cannot be looked "
        "up; use a single value such as text, a number or true/false"
    )


# ---------------------------------------------------------------------------
# Pattern safety
# ---------------------------------------------------------------------------


# The analysis gives up, and refuses the pattern, past these sizes. Each is a
# count, never a time, so the decision is the same on every machine.
_POSITION_LIMIT = 2_000  # character positions, counted repeats written out
_EDGE_LIMIT = 200_000  # "can follow" links between positions
_SEARCH_LIMIT = 200_000  # states the ambiguity search may visit
# A counted repeat ``x{m,n}`` with more optional copies than this is analysed
# as ``x{m,}``. That only adds paths, so the proof stays sound, and small.
_OPTIONAL_COPY_LIMIT = 16

_EXPONENTIAL = (
    "part of it can repeat in more than one way over the same text, like "
    "(a+)+ or (a|a)*, which takes exponential time; rewrite it so the text "
    "splits into repetitions only one way, for example (\\w+,)*\\w+ instead "
    "of (\\w+,?)+"
)
_POLYNOMIAL = (
    "two repeats can take turns over the same characters, like \\d*\\d* or "
    ".*a.*b, which takes polynomial time; merge them into one repeat, or "
    "separate them with a required character that neither can match, for "
    "example [^,]*,[^,]* instead of .*,.*"
)
_BACK_REFERENCE = "back-references can backtrack without bound; remove them"
_LOOKAROUND = (
    "a lookahead or lookbehind that contains a repeat runs again at every "
    "character where it is reached; put such a lookahead at the very start "
    "of the pattern, right after ^, or remove its repeat"
)
_TOO_LARGE = (
    "it is too large to prove safe; shorten its counted repeats ({n} and "
    "{m,n}) or split it into simpler rules"
)

_CATEGORY_SOURCES = {
    "CATEGORY_DIGIT": "\\d",
    "CATEGORY_NOT_DIGIT": "\\D",
    "CATEGORY_SPACE": "\\s",
    "CATEGORY_NOT_SPACE": "\\S",
    "CATEGORY_WORD": "\\w",
    "CATEGORY_NOT_WORD": "\\W",
}

# Characters outside the Basic Multilingual Plane that the overlap test also
# tries: emoji, mathematical digits and letters, a cased script, CJK.
_ASTRAL_SAMPLES = "\U0001f600\U00010000\U0001d7ce\U0001d400\U00010400\U00010428"
_ASTRAL_SAMPLES += "\U00020000\U000e0001\U0010ffff"


def _unsafe_pattern_reason(pattern: str) -> str | None:
    """Why ``pattern`` could take more than linear time, or ``None``.

    Python's ``re`` backtracks and has no timeout, and a timeout would make
    a verdict depend on machine speed, which breaks the judge's determinism.
    So safety is proven before the run, from the pattern alone.

    The rule: two open-ended repeats may follow each other (with optional
    items between them), or one may repeat inside another, only when no
    stretch of text can be shared between them, that is when their
    characters are disjoint or a required character neither can match sits
    between them. ``\\d*\\d*``, ``.*a.*b`` and ``(a+)+`` break it;
    ``^[a-z]+(?: [a-z]+)*$`` and ``(\\w+,)*\\w+`` keep it.

    It is checked exactly on the pattern's position automaton (Glushkov):
    one state per character position, one edge per way a character can
    follow another. A backtracking matcher's work on a failing text is the
    number of paths it can take over it, and that grows

    * exponentially when some state can return to itself by two different
      paths over the same text (``(a+)+``: :data:`_EXPONENTIAL`);
    * polynomially when two looping states can trade the same text, that is
      when a text loops on the first, leads from the first to the second, and
      loops on the second (``\\d*\\d*``: :data:`_POLYNOMIAL`);
    * linearly otherwise.

    Both conditions are searched for in product automata, whose steps need
    one character that every component's class matches; classes are compared
    with ``re`` itself on every Basic Multilingual Plane character plus astral
    samples, so flags such as ``(?i)`` mean what they mean when the pattern
    runs. Anchors and lookarounds are skipped, atomic groups and possessive
    repeats are read as plain ones, and long counted repeats as open-ended:
    each only adds paths, so the check may refuse a safe pattern but never
    accepts an unsafe one. Anything it cannot read, or that is too large to
    search within fixed counts, is refused with a message saying how to
    rewrite it.

    One match attempt is then linear. ``search`` makes one attempt per start
    of an unanchored pattern; :data:`MAX_PATTERN_INPUT` bounds that.
    """
    try:
        parsed = _sre_parse.parse(pattern)
    except re.error:  # pragma: no cover - compile already reported it
        return None
    flags = getattr(getattr(parsed, "state", None), "flags", 0)
    try:
        automaton = _Automaton()
        automaton.sequence(list(parsed), flags)
        automaton.check()
    except _UnprovableError as unsafe:
        return str(unsafe)
    return None


class _UnprovableError(Exception):
    """The pattern was not proven safe; the message tells the client why."""


@dataclass
class _Piece:
    """What the automaton construction knows about one sub-pattern.

    ``first`` and ``last`` are the positions a match of it can start and end
    on, and ``empty`` the number of ways it can match nothing (2 standing for
    "several"). They are multisets on purpose: a position listed twice is
    reached by two routes, and two routes over the same text are exactly the
    ambiguity the check looks for. Each position appears at most twice.
    """

    first: list[int]
    last: list[int]
    empty: int


class _Automaton:
    """The position automaton of one pattern, with every route kept.

    State ``i`` is the ``i``-th character position (a literal, a class, a
    dot), labelled by ``classes[i]``, a regular expression for the characters
    it matches. ``edges`` holds ``(p, q)`` once per route by which position
    ``q`` can directly follow position ``p``.
    """

    def __init__(self) -> None:
        self.classes: list[str] = []
        self.astral = ""
        self.edges: list[tuple[int, int]] = []
        self.repeat_depth = 0

    # -- construction ------------------------------------------------------

    def sequence(self, items: list[Any], flags: int) -> _Piece:
        piece = _Piece([], [], 1)
        for op, value in items:
            piece = self.concat(piece, self.item(_op_name(op), value, flags))
        return piece

    def item(self, name: str, value: Any, flags: int) -> _Piece:
        if name in ("LITERAL", "NOT_LITERAL", "ANY", "IN"):
            return self.position(name, value, flags)
        if name == "AT":
            return _Piece([], [], 1)
        if name == "SUBPATTERN":
            _group, add, remove, items = value
            return self.sequence(list(items), (flags | add) & ~remove)
        if name == "ATOMIC_GROUP":
            return self.sequence(list(value), flags)
        if name == "BRANCH":
            pieces = [self.sequence(list(branch), flags) for branch in value[1]]
            return _Piece(
                _cap([index for piece in pieces for index in piece.first]),
                _cap([index for piece in pieces for index in piece.last]),
                min(sum(piece.empty for piece in pieces), 2),
            )
        if name in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
            low, high, items = value
            return self.repeat(low, high, list(items), flags)
        if name in ("ASSERT", "ASSERT_NOT"):
            self.lookaround(list(value[1]), flags)
            return _Piece([], [], 1)
        if name in ("GROUPREF", "GROUPREF_EXISTS"):
            raise _UnprovableError(_BACK_REFERENCE)
        raise _UnprovableError(
            f"it uses {name.lower()}, which the safety check cannot read; "
            "rewrite it without that construct"
        )

    def position(self, name: str, value: Any, flags: int) -> _Piece:
        if len(self.classes) >= _POSITION_LIMIT:
            raise _UnprovableError(_TOO_LARGE)
        self.classes.append(_class_source(name, value, flags))
        self.astral += _astral_characters(name, value)
        index = len(self.classes) - 1
        return _Piece([index], [index], 0)

    def concat(self, left: _Piece, right: _Piece) -> _Piece:
        self.link(left.last, right.first)
        return _Piece(
            _cap(left.first + left.empty * right.first),
            _cap(right.last + right.empty * left.last),
            min(left.empty * right.empty, 2),
        )

    def repeat(self, low: int, high: int, items: list[Any], flags: int) -> _Piece:
        """``items{low,high}``, with every copy written out as its own positions.

        ``x{2,4}`` becomes ``x x (x (x)?)?``: nested, as ``re`` counts, so a
        given length has one route. ``x{2,}`` becomes ``x x+``.
        """
        open_ended = high - low > _OPTIONAL_COPY_LIMIT
        nested = 1 if high > 1 else 0
        self.repeat_depth += nested
        try:
            piece = _Piece([], [], 1)
            for _ in range(max(low - 1, 0) if open_ended else low):
                piece = self.concat(piece, self.sequence(items, flags))
            if open_ended:
                body = self.sequence(items, flags)
                self.link(body.last, body.first)
                extra = 1 if low == 0 else 0
                looped = _Piece(body.first, body.last, min(body.empty + extra, 2))
                return self.concat(piece, looped)
            tail: _Piece | None = None
            for _ in range(high - low):
                body = self.sequence(items, flags)
                inner = body if tail is None else self.concat(body, tail)
                tail = _Piece(inner.first, inner.last, min(inner.empty + 1, 2))
            return piece if tail is None else self.concat(piece, tail)
        finally:
            self.repeat_depth -= nested

    def lookaround(self, items: list[Any], flags: int) -> None:
        """Check a lookaround's body as a pattern of its own.

        The body runs again at every position the lookaround is reached at.
        One without a repeat costs a constant each time; one with a repeat
        costs up to the whole text, so it is allowed only where it is reached
        once per attempt: before any character position, outside repeats.
        """
        body = _Automaton()
        body.sequence(items, flags)
        body.check()
        if body.has_loop() and (self.classes or self.repeat_depth):
            raise _UnprovableError(_LOOKAROUND)

    def link(self, sources: list[int], targets: list[int]) -> None:
        if len(self.edges) + len(sources) * len(targets) > _EDGE_LIMIT:
            raise _UnprovableError(_TOO_LARGE)
        self.edges.extend((p, q) for p in sources for q in targets)

    # -- analysis ----------------------------------------------------------

    def successors(self) -> list[dict[int, int]]:
        """For each position, the positions that can follow it, with routes."""
        table: list[dict[int, int]] = [{} for _ in self.classes]
        for p, q in self.edges:
            table[p][q] = table[p].get(q, 0) + 1
        return table

    def has_loop(self) -> bool:
        successors = self.successors()
        return any(self._looping(successors, _components(successors)))

    @staticmethod
    def _looping(
        successors: list[dict[int, int]], component: dict[Any, int]
    ) -> list[bool]:
        sizes: dict[int, int] = {}
        for ident in component.values():
            sizes[ident] = sizes.get(ident, 0) + 1
        return [
            sizes[component[p]] > 1 or p in successors[p]
            for p in range(len(successors))
        ]

    def overlap(self, *positions: int) -> bool:
        """Whether one character matches the class of every given position."""
        sources = tuple(sorted({self.classes[p] for p in positions}))
        return _overlap(sources, self.astral)

    def check(self) -> None:
        """Raise :class:`_UnprovableError` unless every match attempt is linear."""
        successors = self.successors()
        component = _components(successors)
        looping = self._looping(successors, component)
        budget = [_SEARCH_LIMIT]
        if self._exponential(successors, component, looping, budget):
            raise _UnprovableError(_EXPONENTIAL)
        if self._polynomial(successors, component, looping, budget):
            raise _UnprovableError(_POLYNOMIAL)

    def _exponential(
        self,
        successors: list[dict[int, int]],
        component: dict[Any, int],
        looping: list[bool],
        budget: list[int],
    ) -> bool:
        """Can a state return to itself by two routes over the same text?

        Searched on pairs of states moving in step: a cycle through a pair
        ``(p, p)`` that passes a pair ``(p', q')`` with ``p' != q'``, or that
        takes two parallel routes between the same positions, is two
        different loops over one text.
        """
        groups: dict[int, list[int]] = {}
        for p in range(len(successors)):
            if looping[p]:
                groups.setdefault(component[p], []).append(p)
        for members in groups.values():
            inside = set(members)
            graph: dict[tuple[int, int], list[tuple[int, int]]] = {}
            doubled: list[tuple[tuple[int, int], tuple[int, int]]] = []
            frontier = [(p, p) for p in members]
            seen = set(frontier)
            while frontier:
                node = frontier.pop()
                a, b = node
                following = []
                for a2, routes in successors[a].items():
                    if a2 not in inside:
                        continue
                    for b2 in successors[b]:
                        if b2 not in inside or not self.overlap(a2, b2):
                            continue
                        following.append((a2, b2))
                        if a == b and a2 == b2 and routes > 1:
                            doubled.append((node, (a2, b2)))
                        if (a2, b2) not in seen:
                            _spend(budget)
                            seen.add((a2, b2))
                            frontier.append((a2, b2))
                graph[node] = following
            pair_component = _components(graph)
            mixed: dict[int, list[bool]] = {}
            for (a, b), ident in pair_component.items():
                mixed.setdefault(ident, []).append(a == b)
            for flags in mixed.values():
                if any(flags) and not all(flags):
                    return True
            for start, end in doubled:
                if pair_component[start] == pair_component[end]:
                    return True
        return False

    def _polynomial(
        self,
        successors: list[dict[int, int]],
        component: dict[Any, int],
        looping: list[bool],
        budget: list[int],
    ) -> bool:
        """Can two looping states trade the same text?

        For looping ``p`` and ``q`` in different loops, searched on triples
        of states moving in step from ``(p, p, q)``: reaching ``(p, q, q)``
        means one text loops on ``p``, leads from ``p`` to ``q``, and loops on
        ``q``. Repeating it k times splits the text between the two loops in
        k ways, and a chain of such pairs multiplies.
        """
        loops = [p for p in range(len(successors)) if looping[p]]
        predecessors: list[list[int]] = [[] for _ in successors]
        for p, table in enumerate(successors):
            for q in table:
                predecessors[q].append(p)
        for q in loops:
            leads_to_q = _reach(predecessors, q)
            for p in loops:
                if component[p] == component[q] or p not in leads_to_q:
                    continue
                start, target = (p, p, q), (p, q, q)
                frontier, seen = [start], {start}
                while frontier:
                    a, b, c = frontier.pop()
                    for a2 in successors[a]:
                        if component[a2] != component[p]:
                            continue
                        for c2 in successors[c]:
                            if component[c2] != component[q]:
                                continue
                            for b2 in successors[b]:
                                if b2 not in leads_to_q or not self.overlap(a2, b2, c2):
                                    continue
                                node = (a2, b2, c2)
                                if node == target:
                                    return True
                                if node not in seen:
                                    _spend(budget)
                                    seen.add(node)
                                    frontier.append(node)
        return False


def _op_name(op: Any) -> str:
    return getattr(op, "name", str(op))


def _cap(positions: list[int]) -> list[int]:
    """``positions`` with each kept at most twice: "two routes" is enough."""
    counts: dict[int, int] = {}
    kept = []
    for index in positions:
        if counts.get(index, 0) < 2:
            counts[index] = counts.get(index, 0) + 1
            kept.append(index)
    return kept


def _spend(budget: list[int]) -> None:
    budget[0] -= 1
    if budget[0] < 0:
        raise _UnprovableError(_TOO_LARGE)


def _reach(graph: list[list[int]], start: int) -> set[int]:
    """Every node reachable from ``start`` in ``graph``, ``start`` included."""
    seen = {start}
    frontier = [start]
    while frontier:
        for nxt in graph[frontier.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def _components(graph: Any) -> dict[Any, int]:
    """The strongly connected component of every node (Tarjan, iterative).

    ``graph`` maps each node to its successors: a list of dicts indexed by
    position, or a dict of lists keyed by product state. Iteration follows
    insertion order, so the numbering is the same in every process.
    """
    nodes = range(len(graph)) if isinstance(graph, list) else list(graph)
    index: dict[Any, int] = {}
    low: dict[Any, int] = {}
    stack: list[Any] = []
    on_stack: set[Any] = set()
    component: dict[Any, int] = {}
    count = 0
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(graph[root]))]
        while work:
            node, pending = work[-1]
            descended = False
            for nxt in pending:
                if nxt not in index:
                    index[nxt] = low[nxt] = len(index)
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(graph[nxt])))
                    descended = True
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component[member] = count
                    if member == node:
                        break
                count += 1
    return component


def _class_source(name: str, value: Any, flags: int) -> str:
    """A regular expression for the characters one position matches."""
    if name == "ANY":
        body = "[\\s\\S]" if flags & re.DOTALL else "[^\\n]"
    elif name == "LITERAL":
        body = f"[{_escaped(value)}]"
    elif name == "NOT_LITERAL":
        body = f"[^{_escaped(value)}]"
    else:
        parts: list[str] = []
        for member_op, member in value:
            member_name = _op_name(member_op)
            if member_name == "NEGATE":
                parts.insert(0, "^")
            elif member_name == "LITERAL":
                parts.append(_escaped(member))
            elif member_name == "RANGE":
                parts.append(f"{_escaped(member[0])}-{_escaped(member[1])}")
            elif _op_name(member) in _CATEGORY_SOURCES:
                parts.append(_CATEGORY_SOURCES[_op_name(member)])
            else:
                raise _UnprovableError(
                    f"its character class uses {member_name.lower()}, which the "
                    "safety check cannot read; rewrite the class"
                )
        body = "[" + "".join(parts) + "]"
    modes = ("i" if flags & re.IGNORECASE else "") + ("a" if flags & re.ASCII else "")
    return f"(?{modes}:{body})" if modes else body


def _escaped(code: int) -> str:
    return f"\\U{code:08x}"


def _astral_characters(name: str, value: Any) -> str:
    """The pattern's own characters beyond the BMP, so overlaps there count."""
    codes: list[int] = []
    if name in ("LITERAL", "NOT_LITERAL"):
        codes.append(value)
    elif name == "IN":
        for member_op, member in value:
            member_name = _op_name(member_op)
            if member_name == "LITERAL":
                codes.append(member)
            elif member_name == "RANGE":
                low, high = member
                codes.extend((low, min(low + 1, high), max(high - 1, low), high))
    found = ""
    for code in codes:
        if code > 0xFFFF:
            char = chr(code)
            found += char + char.lower() + char.upper()
    return found


@functools.lru_cache(maxsize=1)
def _alphabet() -> str:
    """Every BMP character, then the astral samples: what overlaps are tried on."""
    return "".join(map(chr, range(0x10000))) + _ASTRAL_SAMPLES


@functools.lru_cache(maxsize=8192)
def _overlap(sources: tuple[str, ...], astral: str) -> bool:
    """Whether one character matches every class in ``sources``."""
    probe = "".join(f"(?={source})" for source in sources[1:]) + sources[0]
    return re.search(probe, _alphabet() + astral) is not None


# ---------------------------------------------------------------------------
# Schema-derived contracts
# ---------------------------------------------------------------------------


def _fields_from_schema(schema: type) -> tuple[list[OutputField], list[str]]:
    """Read a Pydantic model into declarations.

    ``Literal`` and ``Enum`` become allowed values; ``ge``/``le`` bounds become
    minimum and maximum (``gt``/``lt`` move one inward on integers, and on
    numbers the boundary itself is caught when the judge runs the model's own
    validation); patterns and lengths carry over. Anything Ondine cannot read
    (a nested model, a list) becomes an "any" field: no declarative rule, but
    the model's validation still runs on it in the judge.
    """
    model_fields = getattr(schema, "model_fields", None)
    if not isinstance(model_fields, Mapping):
        return [], [f"schema {schema!r} is not a Pydantic model"]
    fields: list[OutputField] = []
    for name, info in model_fields.items():
        field_type, allowed, nullable = _schema_type(info.annotation)
        bounds: dict[str, Any] = {}
        for meta in info.metadata:
            _read_constraint(meta, field_type, bounds)
        fields.append(
            OutputField(
                name=name,
                type=field_type,
                required=bool(info.is_required()) and not nullable,
                allowed=allowed,
                **bounds,
            )
        )
    return fields, []


def _schema_type(annotation: Any) -> tuple[str, tuple[Any, ...] | None, bool]:
    nullable = False
    origin = typing.get_origin(annotation)
    if origin is Union or origin is getattr(types, "UnionType", None):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        nullable = len(members) < len(typing.get_args(annotation))
        if len(members) != 1:
            return "any", None, nullable
        annotation = members[0]
        origin = typing.get_origin(annotation)
    if origin is Literal:
        values = typing.get_args(annotation)
        return _type_of_values(values), tuple(values), nullable
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = tuple(member.value for member in annotation)
        return _type_of_values(values), values, nullable
    simple = {
        bool: "boolean",
        int: "integer",
        float: "number",
        Decimal: "number",
        str: "text",
    }
    return simple.get(annotation, "any"), None, nullable


def _type_of_values(values: Sequence[Any]) -> str:
    for field_type in ("boolean", "integer", "number", "text"):
        if values and all(_fits_type(field_type, value) for value in values):
            return field_type
    return "any"


def _read_constraint(meta: Any, field_type: str, bounds: dict[str, Any]) -> None:
    numeric = field_type in ("integer", "number")
    step = 1 if field_type == "integer" else 0
    if numeric:
        if getattr(meta, "ge", None) is not None:
            bounds["minimum"] = meta.ge
        if getattr(meta, "gt", None) is not None:
            bounds["minimum"] = meta.gt + step
        if getattr(meta, "le", None) is not None:
            bounds["maximum"] = meta.le
        if getattr(meta, "lt", None) is not None:
            bounds["maximum"] = meta.lt - step
    if field_type == "text":
        if getattr(meta, "min_length", None) is not None:
            bounds["min_length"] = meta.min_length
        if getattr(meta, "max_length", None) is not None:
            bounds["max_length"] = meta.max_length
        if isinstance(getattr(meta, "pattern", None), str):
            bounds["pattern"] = meta.pattern


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _fingerprint(
    columns: tuple[str, ...],
    fields: tuple[OutputField, ...],
    relations: Sequence[Relation],
    policy: RunPolicy,
    parse_mode: str,
    patterns: tuple[tuple[str, str], ...],
    schema: type | None,
    checked: bool,
) -> str:
    document = {
        "version": "contract-v1",
        "checked": checked,
        "columns": list(columns),
        "parse_mode": parse_mode,
        "regex_patterns": [list(pair) for pair in patterns],
        "fields": [_field_document(item) for item in fields],
        "relations": [
            [relation.left, relation.op, relation.right, relation.input_column]
            for relation in relations
        ],
        "policy": [
            policy.on_failure,
            policy.partial_rows,
            _tagged(policy.early_stop_fraction),
            policy.early_stop_window,
        ],
        "schema": _schema_document(schema),
    }
    canonical = json.dumps(document, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _field_document(item: OutputField) -> dict[str, Any]:
    return {
        "name": item.name,
        "type": item.type,
        "required": item.required,
        "absent": _tagged(item.absent),
        "allowed": None
        if item.allowed is None
        else [_tagged(value) for value in item.allowed],
        "aliases": sorted(
            [_tagged(alias), _tagged(target)] for alias, target in item.aliases.items()
        ),
        "minimum": _tagged(item.minimum),
        "maximum": _tagged(item.maximum),
        "min_length": item.min_length,
        "max_length": item.max_length,
        "pattern": item.pattern,
        "normalise": list(item.normalise),
    }


def _tagged(value: Any) -> str:
    """A rendering that keeps the type, so 1, 1.0, True and "1" all differ."""
    if isinstance(value, enum.Enum):
        value = value.value
    return f"{type(value).__name__}:{value!r}"


def _schema_document(schema: type | None) -> Any:
    if schema is None:
        return None
    try:
        layout: Any = schema.model_json_schema()  # type: ignore[attr-defined]
    except Exception:  # a type Pydantic cannot render still gets a stable print
        layout = sorted(getattr(schema, "model_fields", {}))
    decorators = getattr(schema, "__pydantic_decorators__", None)
    validators = []
    if decorators is not None:
        for group in ("field_validators", "model_validators", "validators"):
            validators.extend(sorted(getattr(decorators, group, {}) or {}))
    return {
        "name": f"{schema.__module__}.{schema.__qualname__}",
        "layout": json.loads(json.dumps(layout, sort_keys=True, default=_tagged)),
        "validators": validators,
    }

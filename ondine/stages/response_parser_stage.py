"""Response parsing stage for structured output extraction."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ondine.adapters.containers.result_container import ResultContainerImpl
from ondine.core.models import (
    CostEstimate,
    ResponseBatch,
    ValidationResult,
)
from ondine.harness.contract import OutputContract, build_contract
from ondine.harness.judge import ParsedAnswer, judge
from ondine.stages.llm_invocation_stage import model_answered
from ondine.stages.pipeline_stage import PipelineStage

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ondine.core.data_container import Row  # noqa: TC001
    from ondine.core.models import RowMetadata
    from ondine.harness.judge import Verdict

#: The attempt every verdict here belongs to. The model stage reports each
#: answer it gets as attempt 1 of its pass, and the auto-retry pass runs as a
#: sub-pipeline whose ledger is renumbered when it merges into the run's.
_ATTEMPT = 1

_NO_INPUTS: Mapping[str, Any] = {}


class ResponseParser(ABC):
    """Abstract base for response parsers (Strategy pattern)."""

    @abstractmethod
    def parse(self, response: str) -> dict[str, Any]:
        """Parse response into structured data."""
        pass


class RawTextParser(ResponseParser):
    """Parser that returns raw text."""

    def parse(self, response: str) -> dict[str, Any]:
        """Return response as-is, after cleaning chat format artifacts."""
        cleaned = response.strip()

        # Strip common chat format prefixes (assistant:, user:, system:)
        for prefix in ["assistant:", "user:", "system:"]:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
                break

        return {"output": cleaned}


class JSONParser(ResponseParser):
    """Parser that extracts JSON from response."""

    def __init__(self, strict: bool = False):
        """
        Initialize JSON parser.

        Args:
            strict: If True, fail on invalid JSON; if False, try to extract
        """
        self.strict = strict

    def parse(self, response: str) -> dict[str, Any]:
        """Parse JSON from response."""
        try:
            return json.loads(response.strip())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            if self.strict:
                raise

            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)  # type: ignore[no-any-return]
            if "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)  # type: ignore[no-any-return]
            # Return as raw text if can't parse
            return {"output": response.strip()}


class PydanticParser(ResponseParser):
    """
    Parser that validates responses against Pydantic models.

    Provides type-safe extraction with automatic validation.
    """

    def __init__(self, model: type[BaseModel], strict: bool = True):
        """
        Initialize Pydantic parser.

        Args:
            model: Pydantic model class for validation
            strict: If True, fail on validation errors
        """
        self.model = model
        self.strict = strict

    def parse(self, response: str) -> BaseModel:  # type: ignore[override]
        """Parse and validate response with Pydantic model."""
        try:
            # Try to parse as JSON first
            json_parser = JSONParser(strict=False)
            data = json_parser.parse(response)

            # Validate with Pydantic and return the model instance
            return self.model(**data)

        except ValidationError as e:
            if self.strict:
                raise ValueError(f"Pydantic validation failed: {e}")
            # Return raw data if validation fails
            return {"output": response.strip(), "validation_error": str(e)}  # type: ignore[return-value]


class RegexParser(ResponseParser):
    """
    Parser that extracts data using regex patterns.

    Useful for extracting specific fields from structured text.
    """

    def __init__(self, patterns: dict[str, str]):
        """
        Initialize regex parser.

        Args:
            patterns: Dict mapping field names to regex patterns
        """
        self.patterns = {key: re.compile(pattern) for key, pattern in patterns.items()}

    def parse(self, response: str) -> dict[str, Any]:
        """Extract fields using regex patterns."""
        result = {}

        for field_name, pattern in self.patterns.items():
            match = pattern.search(response)
            if match:
                # Use first group if groups exist, else full match
                if match.groups():
                    result[field_name] = match.group(1)
                else:
                    result[field_name] = match.group(0)
            else:
                result[field_name] = None

        return result


class ResponseParserStage(
    PipelineStage[tuple[list[ResponseBatch], list[str]], ResultContainerImpl]
):
    """
    Turn each LLM response into its row's output values, through the judge.

    For every response the stage asks the judge
    (:func:`ondine.harness.judge.judge`) for a verdict, writes the verdict's
    values, and reports the verdict to the run's row ledger. The judge is the
    only place that reads an answer and maps it onto the output columns; this
    stage only gathers what the judge needs:

    - **the answer**: the response text, which the judge reads the way the
      contract says; or, when a client parser is configured (``with_parser``,
      structured output's parser), that parser's output, which the stage
      still produces by calling ``parser.parse(text)`` as it always has;
    - **the row's inputs**, read only when the contract compares a field with
      an input column.

    Under the unchecked contract (nothing declared) the values, the error
    list and the loss counters are what this stage produced before the judge
    existed: an answer it cannot read is a lost row under this stage's name.

    Under a checked contract the values written are not yet the client's:
    the pipeline's delivery reads them, by row, and keeps or blanks each by
    the row's status. A failed field is written as None, so no later stage
    (context verification, the context store) can read a rejected answer;
    only under ``on_failure="deliver_flagged"``, the one policy that ships a
    failing value, is it written with the judge's candidate. A verdict that
    did not pass is reported with the response text, which the ledger keeps
    for the quarantine.

    A response the model never gave is written but not judged as an answer:
    a call the error policy skipped or filled with its default, or the
    placeholder the batch splitter leaves for a row its batch answer lacked.
    Those rows were already reported lost upstream, and a verdict on a
    placeholder would mark them delivered. If such a response cannot even be
    read, that is reported as a failure of this stage, as before.
    """

    def __init__(
        self,
        parser: ResponseParser | None = None,
        output_columns: list[str] | None = None,
        contract: OutputContract | None = None,
    ):
        """
        Initialize response parser stage.

        Args:
            parser: The client's own parser, whose output the judge maps. With
                a contract and no parser, the judge reads the text as the
                contract says. With neither, today's default:
                :class:`RawTextParser`.
            output_columns: Output column names.
            contract: The run's output contract, which the pipeline resolves.
                Without one the stage judges under the unchecked contract over
                its output columns.
        """
        super().__init__("ResponseParser")
        if parser is None and contract is None:
            parser = RawTextParser()
        self.parser = parser
        self.output_columns = output_columns or ["output"]
        self.contract = contract

    def process(
        self,
        input_data: tuple[list[ResponseBatch], list[str]] | list[ResponseBatch],
        context: Any,
    ) -> ResultContainerImpl:
        """Judge every response and write its values into a ResultContainer."""
        # Handle both tuple (batches, output_cols) and list [batches] for backward compatibility
        if isinstance(input_data, tuple):
            batches, output_cols = input_data
            # Use output_cols from input_data (overrides self.output_columns if provided)
            if not output_cols:
                output_cols = self.output_columns
        else:
            # Backward compatibility: input_data is just the list of batches
            batches = input_data
            output_cols = self.output_columns

        contract = self._contract_for(list(output_cols))
        inputs_of = _inputs_reader(contract, context)
        ledger = getattr(context, "ledger", None)
        if ledger is not None:
            # A context driven outside a pipeline learns its size from reports.
            ledger.widen(
                max(
                    (m.row_index for b in batches for m in b.metadata),
                    default=-1,
                )
                + 1
            )

        # Initialize result storage - use dict to maintain row_index ordering
        results: dict[int, Row] = {}

        for batch in batches:
            for response, metadata in zip(
                batch.responses, batch.metadata, strict=False
            ):
                row = metadata.row_index
                text = response.text if hasattr(response, "text") else str(response)
                verdict = judge(contract, self._answer(text), inputs_of(row))
                results[row] = {
                    "_row_index": row,
                    **_written_values(verdict, contract),
                }
                self._report(context, ledger, row, verdict, response, metadata, text)

        # Convert to list sorted by row_index
        sorted_results = [results[idx] for idx in sorted(results.keys())]

        # Create ResultContainer
        container = ResultContainerImpl(
            data=sorted_results,
            columns=["_row_index"] + list(output_cols),
        )

        self.logger.info(f"Parsed {len(results)} responses")

        return container

    def _contract_for(self, output_cols: list[str]) -> OutputContract:
        """The contract to judge under: the run's, or unchecked over ``output_cols``."""
        if self.contract is None:
            return build_contract(output_cols)
        if list(self.contract.output_columns) != output_cols:
            raise ValueError(
                f"the output contract covers {list(self.contract.output_columns)}, "
                f"but this stage writes {output_cols}"
            )
        return self.contract

    def _answer(self, text: str) -> Any:
        """What the judge reads: the text, or the client parser's reading of it."""
        if self.parser is None:
            return text
        try:
            return ParsedAnswer(self.parser.parse(text), text)
        # A client parser's failure is the row's, not the run's.
        except Exception as error:
            self.logger.debug(f"Failed to parse response: {error}")
            return ParsedAnswer(None, text, error=str(error))

    def _report(
        self,
        context: Any,
        ledger: Any,
        row: int,
        verdict: Verdict,
        response: Any,
        metadata: RowMetadata,
        text: str,
    ) -> None:
        """Report ``verdict`` to the ledger, or a lost row the way it always was.

        A verdict that did not pass goes with ``text``, the raw answer, which
        the ledger keeps only for a row that ends quarantined. A passing
        verdict goes without it: a delivered row's text is in the frame.
        """
        if ledger is not None and _model_answered(response, metadata):
            if verdict.passed:
                ledger.record_verdict(row, _ATTEMPT, verdict, stage=self.name)
            else:
                ledger.record_verdict(
                    row, _ATTEMPT, verdict, stage=self.name, raw_answer=text
                )
        elif not verdict.passed and hasattr(context, "record_failed_rows"):
            context.record_failed_rows(
                [row],
                stage=self.name,
                message="; ".join(reason.text for reason in verdict.reasons),
            )

    def validate_input(
        self, input_data: tuple[list[ResponseBatch], list[str]]
    ) -> ValidationResult:
        """Validate response batches."""
        result = ValidationResult(is_valid=True)

        batches, output_cols = input_data

        if not batches:
            result.add_error("No response batches provided")

        if not output_cols:
            result.add_error("No output columns specified")

        return result

    def estimate_cost(
        self, input_data: tuple[list[ResponseBatch], list[str]]
    ) -> CostEstimate:
        """Response parsing has no LLM cost."""
        return CostEstimate(
            total_cost=Decimal("0.0"),
            total_tokens=0,
            input_tokens=0,
            output_tokens=0,
            rows=sum(len(b.responses) for b in input_data[0]),
        )


def _written_values(verdict: Verdict, contract: OutputContract) -> dict[str, Any]:
    """The values this stage writes for one row: what delivery will read.

    The verdict's values: without a contract, today's parse output; under a
    checked contract, each passing field's value and None for a failed one,
    so a rejected answer never reaches context verification or the context
    store. Under ``on_failure="deliver_flagged"`` a failed field's candidate
    (``FieldVerdict.value``) is written instead, because that policy ships
    it, flagged, on an unverified row; delivery blanks it on any other row
    (decision D-C).
    """
    if not verdict.checked or contract.policy.on_failure != "deliver_flagged":
        return verdict.values
    return {
        item.name: None if item.outcome == "absent" else item.value
        for item in verdict.fields
    }


def _model_answered(response: Any, metadata: RowMetadata) -> bool:
    """Whether ``response`` is the model's answer for its row, so worth a verdict.

    Not when the error policy skipped the call or filled in its default (the
    model stage marks both, and reports them to the ledger as provider
    failures), nor for the placeholder the batch splitter leaves for a row its
    batch answer lacked (reported there as that stage's failure). The model
    stage's own reading (:func:`model_answered`) decides the first two, so
    both stages agree on what counts as an answer.
    """
    if (metadata.custom or {}).get("parse_error"):
        return False
    return model_answered(response)


def _inputs_reader(
    contract: OutputContract, context: Any
) -> Callable[[int], Mapping[str, Any]]:
    """``inputs_of(row)``: the input values the judge compares ``row`` against.

    Only a relation naming an input column reads them, so a contract without
    one (the unchecked contract among them) costs nothing here. Rows are
    numbered by position in the full input, which is what the context keeps
    as the loaded (or preprocessed) data, on resume too.
    """
    if not any(relation.input_column for relation in contract.relations):
        return lambda row: _NO_INPUTS
    full_input = context.full_input() if hasattr(context, "full_input") else None
    rows: list[Mapping[str, Any]] = list(full_input) if full_input is not None else []

    def inputs_of(row: int) -> Mapping[str, Any]:
        return rows[row] if 0 <= row < len(rows) else _NO_INPUTS

    return inputs_of

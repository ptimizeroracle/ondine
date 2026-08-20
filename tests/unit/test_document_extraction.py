"""Public-behavior tests for document-to-structured-data extraction."""

from __future__ import annotations

import json
from typing import get_type_hints

import pytest
from pydantic import BaseModel, Field, model_validator

from ondine.documents import (
    DocumentElement,
    DocumentPage,
    DocumentParser,
    DocumentSource,
    ExtractionCandidate,
    ExtractionTask,
    ExtractionTaskRunner,
    FieldEvidence,
    FieldStatus,
    IncompleteDocumentError,
    PageStatus,
    ParsedDocument,
    extract_document,
)


class Invoice(BaseModel):
    """Small schema that keeps the acceptance test focused on one field."""

    invoice_number: str


class InvoiceSummary(BaseModel):
    """Two-field schema used to verify identity-based reconciliation."""

    invoice_number: str
    currency: str


class InvoiceAmount(BaseModel):
    """Schema with a strict semantic type used to exercise validation failure."""

    total: float


class BalancedInvoice(BaseModel):
    """Schema whose validity depends on two fields together."""

    subtotal: float
    total: float

    @model_validator(mode="after")
    def total_covers_subtotal(self) -> BalancedInvoice:
        """Reject internally inconsistent totals after both fields parse."""
        if self.total < self.subtotal:
            raise ValueError("total cannot be lower than subtotal")
        return self


class AliasedInvoice(BaseModel):
    """Schema whose external alias differs from its Python field name."""

    invoice_number: str = Field(alias="invoiceNumber")


class Address(BaseModel):
    """Nested value used to verify self-contained task schemas."""

    city: str


class NestedInvoice(BaseModel):
    """Schema containing a Pydantic definition reference."""

    billing_address: Address


class _InvoiceParser:
    """Parser fake that preserves one source element on the first page."""

    def __init__(self, version: str = "1") -> None:
        self._version = version

    def parse(self, source: DocumentSource) -> ParsedDocument:
        element = DocumentElement(
            element_id="page-1-line-1",
            page_number=1,
            kind="text",
            text="Invoice number: INV-0042",
        )
        return ParsedDocument(
            document_id=source.document_id,
            parser_name="test-parser",
            parser_version=self._version,
            page_count=1,
            pages=(
                DocumentPage(
                    page_number=1,
                    status=PageStatus.PARSED,
                    elements=(element,),
                ),
            ),
        )


class _InvoiceTaskRunner:
    """Task-runner fake that returns a cited value for the requested field."""

    execution_fingerprint = "invoice-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        task = tasks[0]
        return (
            ExtractionCandidate(
                task_id=task.task_id,
                field_path=task.field_path,
                value="INV-0042",
                status=FieldStatus.FOUND,
                evidence=(
                    FieldEvidence(
                        document_id=document.document_id,
                        page_number=1,
                        element_ids=("page-1-line-1",),
                        quote="INV-0042",
                    ),
                ),
            ),
        )


class _PageGapParser:
    """Parser fake that repeats the production bug of dropping a blank page."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        return ParsedDocument(
            document_id=source.document_id,
            parser_name="gap-parser",
            parser_version="1",
            page_count=3,
            pages=(
                DocumentPage(page_number=1, status=PageStatus.PARSED),
                DocumentPage(page_number=3, status=PageStatus.PARSED),
            ),
        )


class _WrongDocumentParser(_InvoiceParser):
    """Parser fake that accidentally associates another document's graph."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        document = super().parse(source)
        return ParsedDocument(
            document_id="sha256:not-the-input-document",
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            page_count=document.page_count,
            pages=document.pages,
        )


class _ThreePageParser(_InvoiceParser):
    """Parser fake that explicitly represents a blank middle page."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        first_page = super().parse(source).pages[0]
        third_page_element = DocumentElement(
            element_id="page-3-line-1",
            page_number=3,
            kind="text",
            text="End of invoice",
        )
        return ParsedDocument(
            document_id=source.document_id,
            parser_name="test-parser",
            parser_version="1",
            page_count=3,
            pages=(
                first_page,
                DocumentPage(page_number=2, status=PageStatus.BLANK),
                DocumentPage(
                    page_number=3,
                    status=PageStatus.PARSED,
                    elements=(third_page_element,),
                ),
            ),
        )


class _MalformedManifestParser:
    """Parser fake with a caller-selected invalid physical-page manifest."""

    def __init__(
        self,
        pages: tuple[DocumentPage, ...],
        *,
        page_count: int = 1,
    ) -> None:
        self._pages = pages
        self._page_count = page_count

    def parse(self, source: DocumentSource) -> ParsedDocument:
        return ParsedDocument(
            document_id=source.document_id,
            parser_name="malformed-parser",
            parser_version="1",
            page_count=self._page_count,
            pages=self._pages,
        )


class _FailedPageParser(_ThreePageParser):
    """Parser fake whose middle page could not be decoded."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        document = super().parse(source)
        return ParsedDocument(
            document_id=document.document_id,
            parser_name=document.parser_name,
            parser_version=document.parser_version,
            page_count=document.page_count,
            pages=(
                document.pages[0],
                DocumentPage(
                    page_number=2,
                    status=PageStatus.FAILED,
                    error="simulated decoder failure",
                ),
                document.pages[2],
            ),
        )


class _MismatchedElementPageParser(_InvoiceParser):
    """Parser fake whose element claims a different physical page."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        element = DocumentElement(
            element_id="misplaced-line",
            page_number=2,
            kind="text",
            text="Invoice number: INV-0042",
        )
        return ParsedDocument(
            document_id=source.document_id,
            parser_name="mismatched-element-parser",
            parser_version="1",
            page_count=1,
            pages=(
                DocumentPage(
                    page_number=1,
                    status=PageStatus.PARSED,
                    elements=(element,),
                ),
            ),
        )


class _DuplicateElementParser(_InvoiceParser):
    """Parser fake that assigns one source identity to two page elements."""

    def parse(self, source: DocumentSource) -> ParsedDocument:
        elements = tuple(
            DocumentElement(
                element_id="duplicate-line",
                page_number=1,
                kind="text",
                text=text,
            )
            for text in ("first", "second")
        )
        return ParsedDocument(
            document_id=source.document_id,
            parser_name="duplicate-element-parser",
            parser_version="1",
            page_count=1,
            pages=(
                DocumentPage(
                    page_number=1,
                    status=PageStatus.PARSED,
                    elements=elements,
                ),
            ),
        )


class _MissingTaskRunner:
    """Runner fake that loses every planned task without aborting the run."""

    execution_fingerprint = "missing-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        return ()


class _AbsentTaskRunner:
    """Runner fake that examined a required field and found it absent."""

    execution_fingerprint = "absent-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        task = tasks[0]
        return (
            ExtractionCandidate(
                task_id=task.task_id,
                field_path=task.field_path,
                value=None,
                status=FieldStatus.ABSENT_IN_DOCUMENT,
            ),
        )


class _RecordingTaskRunner(_InvoiceTaskRunner):
    """Successful runner fake that records durable task identities."""

    def __init__(self, execution_fingerprint: str = "recording-runner:v1") -> None:
        self.execution_fingerprint = execution_fingerprint
        self.task_ids: list[str] = []

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        self.task_ids.extend(task.task_id for task in tasks)
        return super().run(document, tasks)


class _DuplicateTaskRunner:
    """Runner fake that returns two incompatible values for one task ID."""

    execution_fingerprint = "duplicate-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        task = tasks[0]
        return tuple(
            ExtractionCandidate(
                task_id=task.task_id,
                field_path=task.field_path,
                value=value,
                status=FieldStatus.FOUND,
            )
            for value in ("INV-0042", "INV-9999")
        )


class _InvalidEvidenceRunner:
    """Runner fake that cites a page absent from the parser manifest."""

    execution_fingerprint = "invalid-evidence-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        task = tasks[0]
        return (
            ExtractionCandidate(
                task_id=task.task_id,
                field_path=task.field_path,
                value="INV-0042",
                status=FieldStatus.FOUND,
                evidence=(
                    FieldEvidence(
                        document_id=document.document_id,
                        page_number=99,
                        quote="INV-0042",
                    ),
                ),
            ),
        )


class _ForgedQuoteRunner(_InvoiceTaskRunner):
    """Runner fake whose quoted evidence does not occur in the source page."""

    execution_fingerprint = "forged-quote-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        candidate = super().run(document, tasks)[0]
        return (
            ExtractionCandidate(
                task_id=candidate.task_id,
                field_path=candidate.field_path,
                value=candidate.value,
                status=candidate.status,
                evidence=(
                    FieldEvidence(
                        document_id=document.document_id,
                        page_number=1,
                        element_ids=("page-1-line-1",),
                        quote="INV-NOT-IN-THE-SOURCE",
                    ),
                ),
            ),
        )


class _UnknownTaskRunner(_InvoiceTaskRunner):
    """Runner fake that leaks a result belonging to another task batch."""

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        expected = super().run(document, tasks)
        leaked = ExtractionCandidate(
            task_id="task:unknown",
            field_path=("invoice_number",),
            value="INV-LEAKED",
            status=FieldStatus.FOUND,
            evidence=expected[0].evidence,
        )
        return (*expected, leaked)


class _ReverseTaskRunner:
    """Runner fake that returns valid task results in reverse order."""

    execution_fingerprint = "reverse-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        values = {"invoice_number": "INV-0042", "currency": "EUR"}
        evidence = (
            FieldEvidence(
                document_id=document.document_id,
                page_number=1,
                element_ids=("page-1-line-1",),
            ),
        )
        return tuple(
            ExtractionCandidate(
                task_id=task.task_id,
                field_path=task.field_path,
                value=values[task.field_path[0]],
                status=FieldStatus.FOUND,
                evidence=evidence,
            )
            for task in reversed(tasks)
        )


class _InvalidValueRunner:
    """Runner fake that cites a value which cannot satisfy the field schema."""

    execution_fingerprint = "invalid-value-runner:v1"

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        task = tasks[0]
        return (
            ExtractionCandidate(
                task_id=task.task_id,
                field_path=task.field_path,
                value="not-a-decimal",
                status=FieldStatus.FOUND,
                evidence=(
                    FieldEvidence(
                        document_id=document.document_id,
                        page_number=1,
                        element_ids=("page-1-line-1",),
                    ),
                ),
            ),
        )


class _ValueTaskRunner:
    """Runner fake for schemas whose fields need caller-selected values."""

    def __init__(
        self,
        values: dict[str, object],
        *,
        execution_fingerprint: str = "value-runner:v1",
    ) -> None:
        self._values = values
        self.execution_fingerprint = execution_fingerprint
        self.task_ids: list[str] = []
        self.task_schemas: dict[str, dict[str, object]] = {}

    def run(
        self,
        document: ParsedDocument,
        tasks: tuple[ExtractionTask, ...],
    ) -> tuple[ExtractionCandidate, ...]:
        candidates: list[ExtractionCandidate] = []
        for task in tasks:
            field_name = task.field_path[0]
            self.task_ids.append(task.task_id)
            self.task_schemas[field_name] = json.loads(task.schema_json)
            candidates.append(
                ExtractionCandidate(
                    task_id=task.task_id,
                    field_path=task.field_path,
                    value=self._values[field_name],
                    status=FieldStatus.FOUND,
                    evidence=(
                        FieldEvidence(
                            document_id=document.document_id,
                            page_number=1,
                            element_ids=("page-1-line-1",),
                        ),
                    ),
                )
            )
        return tuple(candidates)


def test_document_source_identity_is_always_derived_from_its_bytes():
    """Callers must not be able to forge an ID that aliases different content."""
    direct = DocumentSource(
        content=b"identity-bearing bytes",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    factory = DocumentSource.from_bytes(
        b"identity-bearing bytes",
        name="renamed.pdf",
        media_type="application/pdf",
    )

    assert direct.document_id == factory.document_id


def test_extract_document_returns_validated_data_with_field_evidence():
    """A plausible field value without its source evidence is not trustworthy."""
    source = DocumentSource.from_bytes(
        b"synthetic invoice",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_InvoiceTaskRunner(),
    )

    assert result.data == Invoice(invoice_number="INV-0042")
    assert result.evidence_for("invoice_number") == (
        FieldEvidence(
            document_id=source.document_id,
            page_number=1,
            element_ids=("page-1-line-1",),
            quote="INV-0042",
        ),
    )


def test_extract_document_rejects_a_page_manifest_with_a_gap():
    """A dropped blank page must be visible rather than silently disappearing."""
    source = DocumentSource.from_bytes(
        b"three physical pages",
        name="contract.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="missing page manifest entries: 2"):
        extract_document(
            source,
            schema=Invoice,
            parser=_PageGapParser(),
            task_runner=_InvoiceTaskRunner(),
        )


def test_extract_document_rejects_a_source_graph_for_different_bytes():
    """Concurrent parser output must never cross from one document to another."""
    source = DocumentSource.from_bytes(
        b"correct invoice bytes",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="parser returned document .* expected"):
        extract_document(
            source,
            schema=Invoice,
            parser=_WrongDocumentParser(),
            task_runner=_InvoiceTaskRunner(),
        )


def test_missing_task_output_returns_an_honest_partial_result():
    """Missing work must remain visible instead of crashing or looking complete."""
    source = DocumentSource.from_bytes(
        b"invoice whose model call was lost",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_MissingTaskRunner(),
    )

    assert result.success is True
    assert result.is_complete is False
    assert result.data is None
    assert result.partial_data == {"invoice_number": None}
    assert result.fields[0].status == FieldStatus.NOT_EXAMINED


def test_complete_extraction_converts_to_one_dataframe_row():
    """A document result should cross into Pandas without losing schema names."""
    source = DocumentSource.from_bytes(
        b"synthetic invoice",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_InvoiceTaskRunner(),
    )

    assert result.to_pandas().to_dict(orient="records") == [
        {"invoice_number": "INV-0042"}
    ]


def test_parser_version_change_invalidates_document_task_identity():
    """Resume must not reuse work generated from a different parser graph."""
    source = DocumentSource.from_bytes(
        b"same invoice bytes",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    runner = _RecordingTaskRunner()

    extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(version="1"),
        task_runner=runner,
    )
    extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(version="2"),
        task_runner=runner,
    )

    assert runner.task_ids[0] != runner.task_ids[1]


def test_duplicate_task_results_become_an_explicit_conflict():
    """Duplicate outputs must never resolve through accidental last-write-wins."""
    source = DocumentSource.from_bytes(
        b"invoice with duplicate provider output",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_DuplicateTaskRunner(),
    )

    assert result.data is None
    assert result.is_complete is False
    assert result.fields[0].status == FieldStatus.CONFLICTING


def test_evidence_for_an_unknown_page_cannot_produce_trusted_data():
    """A value citing a nonexistent page must remain visibly unresolved."""
    source = DocumentSource.from_bytes(
        b"invoice with impossible citation",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_InvalidEvidenceRunner(),
    )

    assert result.data is None
    assert result.is_complete is False
    assert result.fields[0].status == FieldStatus.CONFLICTING


def test_partial_extraction_dataframe_marks_unexamined_fields_as_missing():
    """Partial output must use None rather than an exception or sentinel value."""
    source = DocumentSource.from_bytes(
        b"invoice whose model call was lost",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_MissingTaskRunner(),
    )

    assert result.to_pandas().to_dict(orient="records") == [{"invoice_number": None}]


def test_require_complete_names_the_unresolved_fields():
    """Strict consumers need one diagnostic guard against partial documents."""
    source = DocumentSource.from_bytes(
        b"invoice whose model call was lost",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_MissingTaskRunner(),
    )

    with pytest.raises(IncompleteDocumentError, match="invoice_number"):
        result.require_complete()


def test_unknown_task_output_is_rejected_instead_of_ignored():
    """Cross-batch output must not disappear behind an otherwise valid result."""
    source = DocumentSource.from_bytes(
        b"invoice with leaked task output",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="unknown task result: task:unknown"):
        extract_document(
            source,
            schema=Invoice,
            parser=_InvoiceParser(),
            task_runner=_UnknownTaskRunner(),
        )


def test_identical_source_graphs_produce_the_same_task_identity():
    """Stable inputs need stable IDs so completed work can later be resumed."""
    source = DocumentSource.from_bytes(
        b"same invoice bytes",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    runner = _RecordingTaskRunner()

    for _ in range(2):
        extract_document(
            source,
            schema=Invoice,
            parser=_InvoiceParser(),
            task_runner=runner,
        )

    assert runner.task_ids[0] == runner.task_ids[1]


def test_reordered_task_results_realign_by_identity():
    """Provider order must not swap values between schema fields."""
    source = DocumentSource.from_bytes(
        b"two-field invoice",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=InvoiceSummary,
        parser=_InvoiceParser(),
        task_runner=_ReverseTaskRunner(),
    )

    assert result.data == InvoiceSummary(invoice_number="INV-0042", currency="EUR")


def test_explicit_blank_page_remains_in_the_result_manifest():
    """A blank physical page is accounted for even though it has no elements."""
    source = DocumentSource.from_bytes(
        b"three-page invoice",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_ThreePageParser(),
        task_runner=_InvoiceTaskRunner(),
    )

    assert [page.status for page in result.document.pages] == [
        PageStatus.PARSED,
        PageStatus.BLANK,
        PageStatus.PARSED,
    ]


def test_schema_invalid_candidate_becomes_an_incomplete_field():
    """A cited but ill-typed value must not abort or become trusted data."""
    source = DocumentSource.from_bytes(
        b"invoice with invalid amount",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=InvoiceAmount,
        parser=_InvoiceParser(),
        task_runner=_InvalidValueRunner(),
    )

    assert result.success is True
    assert result.data is None
    assert result.is_complete is False
    assert result.fields[0].status == FieldStatus.CONFLICTING


@pytest.mark.parametrize(
    ("pages", "message"),
    [
        (
            (
                DocumentPage(page_number=1, status=PageStatus.PARSED),
                DocumentPage(page_number=2, status=PageStatus.PARSED),
            ),
            "unexpected page manifest entries: 2",
        ),
        (
            (
                DocumentPage(page_number=1, status=PageStatus.PARSED),
                DocumentPage(page_number=1, status=PageStatus.BLANK),
            ),
            "duplicate page manifest entries: 1",
        ),
    ],
)
def test_extract_document_rejects_extra_or_duplicate_page_entries(
    pages: tuple[DocumentPage, ...],
    message: str,
):
    """Every physical page number must have exactly one manifest entry."""
    source = DocumentSource.from_bytes(
        b"one physical page",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match=message):
        extract_document(
            source,
            schema=Invoice,
            parser=_MalformedManifestParser(pages),
            task_runner=_InvoiceTaskRunner(),
        )


def test_failed_page_prevents_a_complete_document_result():
    """Valid fields cannot prove full coverage when one physical page failed."""
    source = DocumentSource.from_bytes(
        b"three-page invoice with a decoder failure",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_FailedPageParser(),
        task_runner=_InvoiceTaskRunner(),
    )

    assert result.data == Invoice(invoice_number="INV-0042")
    assert result.is_complete is False
    with pytest.raises(IncompleteDocumentError, match="failed pages: 2"):
        result.require_complete()


def test_element_page_must_match_its_manifest_entry():
    """Contradictory page ownership would make every later citation ambiguous."""
    source = DocumentSource.from_bytes(
        b"invoice with misplaced parser element",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="element 'misplaced-line'.*page 2.*page 1"):
        extract_document(
            source,
            schema=Invoice,
            parser=_MismatchedElementPageParser(),
            task_runner=_InvoiceTaskRunner(),
        )


def test_model_level_validation_failure_cannot_leak_raw_values():
    """Cross-field-invalid values must not survive as apparently found data."""
    source = DocumentSource.from_bytes(
        b"invoice whose total is internally inconsistent",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=BalancedInvoice,
        parser=_InvoiceParser(),
        task_runner=_ValueTaskRunner({"subtotal": 100.0, "total": 90.0}),
    )

    assert result.data is None
    assert {field.status for field in result.fields} == {FieldStatus.CONFLICTING}
    assert result.partial_data == {"subtotal": None, "total": None}


def test_aliased_field_keeps_a_meaningful_task_schema_and_validates():
    """Pydantic aliases must not erase schema details or block final validation."""
    source = DocumentSource.from_bytes(
        b"invoice using an external field alias",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    runner = _ValueTaskRunner({"invoice_number": "INV-0042"})

    result = extract_document(
        source,
        schema=AliasedInvoice,
        parser=_InvoiceParser(),
        task_runner=runner,
    )

    assert result.require_complete().invoice_number == "INV-0042"
    assert runner.task_schemas["invoice_number"]["type"] == "string"


def test_nested_field_task_schema_carries_its_referenced_definitions():
    """A runner must receive a self-contained schema, not a dangling $ref."""
    source = DocumentSource.from_bytes(
        b"invoice with nested billing address",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    runner = _ValueTaskRunner({"billing_address": {"city": "Paris"}})

    result = extract_document(
        source,
        schema=NestedInvoice,
        parser=_InvoiceParser(),
        task_runner=runner,
    )

    assert result.require_complete().billing_address.city == "Paris"
    assert "$defs" in runner.task_schemas["billing_address"]


def test_execution_contract_change_invalidates_task_identity():
    """Resume must not reuse output from a different model or prompt contract."""
    source = DocumentSource.from_bytes(
        b"same invoice and parser graph",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    runner_v1 = _ValueTaskRunner(
        {"invoice_number": "INV-0042"},
        execution_fingerprint="model-a:prompt-v1",
    )
    runner_v2 = _ValueTaskRunner(
        {"invoice_number": "INV-0042"},
        execution_fingerprint="model-b:prompt-v2",
    )

    for runner in (runner_v1, runner_v2):
        extract_document(
            source,
            schema=Invoice,
            parser=_InvoiceParser(),
            task_runner=runner,
        )

    assert runner_v1.task_ids[0] != runner_v2.task_ids[0]


def test_public_protocol_annotations_resolve_at_runtime():
    """Documentation and dependency-injection tools need concrete type hints."""
    parser_hints = get_type_hints(DocumentParser.parse)
    runner_hints = get_type_hints(ExtractionTaskRunner.run)

    assert parser_hints["source"] is DocumentSource
    assert parser_hints["return"] is ParsedDocument
    assert runner_hints["document"] is ParsedDocument


def test_source_graph_requires_at_least_one_physical_page():
    """A zero-page manifest cannot prove that any document was parsed."""
    source = DocumentSource.from_bytes(
        b"empty parser output",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="page_count must be at least 1"):
        extract_document(
            source,
            schema=Invoice,
            parser=_MalformedManifestParser((), page_count=0),
            task_runner=_InvoiceTaskRunner(),
        )


def test_duplicate_element_identity_is_rejected():
    """Two source elements cannot share one provenance identity on a page."""
    source = DocumentSource.from_bytes(
        b"invoice with duplicate element IDs",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    with pytest.raises(ValueError, match="duplicate element ids on page 1"):
        extract_document(
            source,
            schema=Invoice,
            parser=_DuplicateElementParser(),
            task_runner=_InvoiceTaskRunner(),
        )


def test_fabricated_evidence_quote_cannot_produce_trusted_data():
    """A quote must be present in the cited page rather than model-invented."""
    source = DocumentSource.from_bytes(
        b"invoice with fabricated evidence quote",
        name="invoice.pdf",
        media_type="application/pdf",
    )

    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_ForgedQuoteRunner(),
    )

    assert result.data is None
    assert result.fields[0].status == FieldStatus.CONFLICTING


def test_required_absent_field_is_named_by_require_complete():
    """A factual absence must remain distinct while strict diagnostics name it."""
    source = DocumentSource.from_bytes(
        b"invoice without an invoice number",
        name="invoice.pdf",
        media_type="application/pdf",
    )
    result = extract_document(
        source,
        schema=Invoice,
        parser=_InvoiceParser(),
        task_runner=_AbsentTaskRunner(),
    )

    assert result.fields[0].status == FieldStatus.ABSENT_IN_DOCUMENT
    with pytest.raises(IncompleteDocumentError, match="invoice_number"):
        result.require_complete()

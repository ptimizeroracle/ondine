"""Certified columns: how many delivered values are right, with an exact range.

Use it through the pipeline: ``pipeline.draw_sample(result)``,
``pipeline.certify(result, gold=...)`` against labels you trust, and
``pipeline.audit(result, model=..., max_cost=...)`` against a reference model,
and ``pipeline.export_review(result, folder)`` then ``pipeline.read_review(folder)``
for blind labels from people. See ``docs/guides/certified-columns.md``.
"""

from ondine.ruler.certificate import (
    Certificate,
    Estimate,
    FieldCertificate,
    LabellerAgreement,
)
from ondine.ruler.delivered import CertificationError
from ondine.ruler.reference import AuditEstimate, ReferenceAnswers
from ondine.ruler.sample import Sample
from ondine.ruler.sheet import ReviewExport, ReviewLabels, ReviewSheetError

__all__ = [
    "AuditEstimate",
    "Certificate",
    "CertificationError",
    "Estimate",
    "FieldCertificate",
    "LabellerAgreement",
    "ReferenceAnswers",
    "ReviewExport",
    "ReviewLabels",
    "ReviewSheetError",
    "Sample",
]

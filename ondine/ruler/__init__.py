"""Certified columns: how many delivered values are right, with an exact range.

Use it through the pipeline: ``pipeline.draw_sample(result)``,
``pipeline.certify(result, gold=...)`` against labels you trust, and
``pipeline.audit(result, model=..., max_cost=...)`` against a reference model. See ``docs/guides/certified-columns.md``.
"""

from ondine.ruler.certificate import Certificate, Estimate, FieldCertificate
from ondine.ruler.delivered import CertificationError
from ondine.ruler.reference import AuditEstimate, ReferenceAnswers
from ondine.ruler.sample import Sample

__all__ = [
    "AuditEstimate",
    "Certificate",
    "CertificationError",
    "Estimate",
    "FieldCertificate",
    "ReferenceAnswers",
    "Sample",
]

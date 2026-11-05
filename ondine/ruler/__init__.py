"""Certified columns: how many delivered values are right, with an exact range.

Use it through the pipeline: ``pipeline.draw_sample(result)`` and
``pipeline.certify(result, gold=...)``. See ``docs/guides/certified-columns.md``.
"""

from ondine.ruler.certificate import Certificate, Estimate, FieldCertificate
from ondine.ruler.delivered import CertificationError
from ondine.ruler.sample import Sample

__all__ = [
    "Certificate",
    "CertificationError",
    "Estimate",
    "FieldCertificate",
    "Sample",
]

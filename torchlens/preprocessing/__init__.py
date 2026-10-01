"""Preprocessing provenance + verification (tvscope lane; NeuroAI audience).

The verified-capture story for feature extraction: RESOLVE the preprocessing
authority the user's own loader ships (never a TorchLens-hosted recipe --
memo D1/D7), AUDIT what actually ran against it field-by-field (the verdict
oracle -- memo D3), and optionally DIAGNOSE the concrete tensor batch with
opt-in checks that may never claim a match (memo D4). Three separate records
carry three separate epistemic claims:

- :class:`torchlens.data_classes.trace.ResolvedPreprocessing` -- the
  authority provenance record (persisted on ``Trace.input_preprocessor``),
  with the derived closed ``status`` (authoritative / unverified_fallback /
  unknown).
- :class:`PreprocessingAudit` -- the field-level configuration comparison
  (match / mismatch / unknown-with-reason per field; strict mode refuses
  mismatch AND unknown with distinct typed codes).
- :class:`InputDiagnostics` -- opt-in tensor findings that contradict or
  fail-to-contradict, never verify.

Generic by design: nothing here is gated behind the ``neuro`` extra (memo
section 8, all three labs verbatim). Every spelling is DOCUMENTED-UNSTABLE
pending the naming sprint; import as ``import torchlens.preprocessing``.
"""

from __future__ import annotations

from ._audit import (
    AuditFinding,
    PreprocessingAudit,
    PreprocessingAuditError,
    audit,
)
from ._authorities import (
    AuthorityAdapter,
    register_authority_adapter,
    resolve,
)
from ._diagnostics import (
    DEFAULT_CHECKS,
    DiagnosticFinding,
    InputDiagnostics,
    diagnose,
)
from ._records import (
    COMPARABLE_FIELDS,
    STATUS_AUTHORITATIVE,
    STATUS_UNKNOWN,
    STATUS_UNVERIFIED_FALLBACK,
    DeclaredPreprocessing,
    Resolution,
    status_of,
    unknown_resolution,
)

__tl_layer__ = "L5"

__all__ = [
    "COMPARABLE_FIELDS",
    "DEFAULT_CHECKS",
    "STATUS_AUTHORITATIVE",
    "STATUS_UNKNOWN",
    "STATUS_UNVERIFIED_FALLBACK",
    "AuditFinding",
    "AuthorityAdapter",
    "DeclaredPreprocessing",
    "DiagnosticFinding",
    "InputDiagnostics",
    "PreprocessingAudit",
    "PreprocessingAuditError",
    "Resolution",
    "audit",
    "diagnose",
    "register_authority_adapter",
    "resolve",
    "status_of",
    "unknown_resolution",
]

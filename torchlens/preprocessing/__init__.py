"""Preprocessing provenance + verification (tvscope lane; NeuroAI audience).

The verified-capture story for feature extraction: RESOLVE the preprocessing
authority the user's own loader ships (never a TorchLens-hosted recipe --
memo D1/D7), AUDIT what actually ran against it field-by-field (the verdict
oracle -- memo D3), and optionally DIAGNOSE the concrete tensor batch with
opt-in checks that may never claim a match (memo D4). On request only,
:func:`check_card` compares what a capture recorded with the model's known card
(HF processor, torchvision weights, timm, open_clip) and warns on a mismatch;
:func:`attach_authority` records an authority a model cannot carry itself
(a torchvision model keeps no handle to its weights). Three separate records
carry three separate epistemic claims:

- :class:`torchlens._data_classes.trace.ResolvedPreprocessing` -- the
  authority provenance record (persisted on ``Trace.provenance.input_preprocessor``),
  with the derived closed ``status`` (authoritative / unverified_fallback /
  unknown).
- :class:`PreprocessingComparison` -- the field-level configuration comparison
  (match / mismatch / unknown-with-reason per field; strict mode refuses
  mismatch AND unknown with distinct typed codes).
- :class:`InputDiagnostics` -- opt-in tensor findings that contradict or
  fail-to-contradict, never verify.

A project teaches ``tl.trace`` its own raw input through :func:`register`, the
public door of the input detector registry (the same registry the built-in
Hugging Face text, image and multimodal routes use). A detector receives
``(model, payload, **kwargs)`` and returns a ``Trace`` or ``None``; it claims
the call by re-entering ``tl.trace`` with what ``forward`` takes, forwarding
``kwargs`` so the caller's options survive. :func:`unregister` removes it, and
both affect later traces only.

Generic by design: nothing here is gated behind the ``neuro`` extra (memo
section 8, all three labs verbatim). Every spelling is DOCUMENTED-UNSTABLE
pending the naming sprint; import as ``import torchlens.preprocessing``.
"""

from __future__ import annotations

from torchlens._autoroute.input import register, unregister

from ._attached import attach_authority
from ._audit import (
    AuditFinding,
    PreprocessingComparison,
    compare,
)
from ._authorities import (
    AuthorityAdapter,
    register_authority_adapter,
    resolve,
)
from ._card import check_card
from ._diagnostics import (
    DiagnosticFinding,
    InputDiagnostics,
    diagnose,
)
from ._imagenet_default import imagenet_default
from ._records import DeclaredPreprocessing, Resolution, unknown_resolution

__tl_layer__ = "L5"

__all__ = [
    "AuditFinding",
    "AuthorityAdapter",
    "DeclaredPreprocessing",
    "DiagnosticFinding",
    "InputDiagnostics",
    "PreprocessingComparison",
    "Resolution",
    "attach_authority",
    "compare",
    "check_card",
    "diagnose",
    "imagenet_default",
    "register",
    "register_authority_adapter",
    "resolve",
    "unknown_resolution",
    "unregister",
]

"""The decomposition-transparency belt (W1, weightsfree memo D3 / defect L1).

On the meta substrate torch implements many ops as PYTHON decompositions
(``torch/_refs``, ``_prims``, ``_meta_registrations``, ``_library``, ...).
TorchLens wraps at the torch-function level, so without this belt it records
the decomposition's inner allocator call INSTEAD of the user's op and its
in-place/barcode bookkeeping folds the outer user op into it — measured:
distilgpt2's 291 records collapsed to 24, a six-op chain became one
allocator record carrying the last op's shape (defect L1, found by all
three labs in round 1).

The fix is DO-NOT-RECORD, never normalize-after (D3): a wrapped call whose
first non-TorchLens caller frame sits inside torch decomposition machinery
is not recorded; the enclosing USER op stays the record. Suppressing the
frames restores byte-identical labels and shapes with ZERO maintained
mapping tables. Scoped to ADMITTED meta structure-only captures only,
memoized per code object, root set FEATURE-DETECTED against the running
torch (published via ``tl.compat.report()`` / ``tl.doctor()``), and
exception flow is never suppressed — the belt gates RECORDING only.

Disclosed table condition (D3): a direct USER call to a private
``torch._refs.*`` function inside ``forward`` is indistinguishable from a
decomposition frame and is likewise unrecorded.

Every spelling is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import CodeType
from typing import Final

import torch

__all__ = ["DECOMPOSITION_ROOTS", "caller_is_torch_decomposition"]

_TORCH_ROOT: Final[Path] = Path(torch.__file__).resolve().parent

#: Candidate torch-internal decomposition/meta-machinery roots. Each entry is
#: a path relative to the torch package root; only entries that EXIST in the
#: running torch survive feature detection (a renamed internal module drops
#: out of the belt visibly via compat.report(), never silently misclassifies).
_ROOT_CANDIDATES: Final[tuple[str, ...]] = (
    "_refs",
    "_decomp",
    "_prims",
    "_prims_common",
    "_subclasses",
    "_meta_registrations.py",
    "_library",
    "_ops.py",
    "_compile.py",
    "_dynamo",
)

DECOMPOSITION_ROOTS: Final[tuple[str, ...]] = tuple(
    # Directory roots carry a trailing separator so "_refs" never matches a
    # hypothetical "_refs_other" sibling; file roots match exactly.
    str(_TORCH_ROOT / candidate) + ("" if candidate.endswith(".py") else os.sep)
    for candidate in _ROOT_CANDIDATES
    if (_TORCH_ROOT / candidate).exists()
)
"""The feature-detected decomposition root set for the running torch."""

_TORCHLENS_ROOT: Final[str] = str(Path(__file__).resolve().parents[2]) + os.sep

# Per-code-object classification cache. Code objects are effectively
# immortal (module/function lifetime), so a plain dict is bounded by the
# amount of loaded code; the hot path pays one dict hit per frame.
_CODE_CLASSIFICATION: dict[CodeType, str] = {}

_KIND_TORCHLENS: Final[str] = "torchlens"
_KIND_DECOMPOSITION: Final[str] = "decomposition"
_KIND_USER: Final[str] = "user"


def _classify_code(code: CodeType) -> str:
    """Classify one code object's home: torchlens / decomposition / user."""

    cached = _CODE_CLASSIFICATION.get(code)
    if cached is not None:
        return cached
    filename = code.co_filename
    if filename.startswith("<"):
        # Synthetic frames (<string>, <lambda> wrappers) are transparent:
        # skip them like torchlens frames rather than guessing an owner.
        kind = _KIND_TORCHLENS
    else:
        resolved = str(Path(filename).resolve())
        if resolved.startswith(_TORCHLENS_ROOT):
            kind = _KIND_TORCHLENS
        elif any(resolved.startswith(root) for root in DECOMPOSITION_ROOTS):
            kind = _KIND_DECOMPOSITION
        else:
            kind = _KIND_USER
    _CODE_CLASSIFICATION[code] = kind
    return kind


def caller_is_torch_decomposition() -> bool:
    """Whether the current wrapped call was issued BY torch decomposition code.

    Walks the caller frames outward, skipping TorchLens's own machinery; the
    FIRST non-TorchLens frame decides. Called only on the admitted
    weights-free path (the default path never pays this walk).
    """

    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            kind = _classify_code(frame.f_code)
            if kind == _KIND_TORCHLENS:
                frame = frame.f_back
                continue
            return kind == _KIND_DECOMPOSITION
        return False
    finally:
        del frame

"""Pristine-torch context for wrap-state-independent validation oracles.

R75-1 (b9-fable round-2, probe-proven): once torch is wrapped, EVERY
in-process value oracle -- the phase-0 ground-truth forward, the per-op
replay (which calls the wrapper-closure ``canonical_capture_callable``),
and backward validation's "stock autograd" forward -- observes through the
same installed wrapper shells. A TorchLens-induced numeric distortion (a
planted 0.1% ``tanh`` distortion in the wrapper layer) therefore validated
CLEAN in a wrapped process and was caught only in a fresh one: the verdict
on the identical model and bug depended on whether any capture had run
earlier in the process.

The fix: the ground-truth oracles run with the wrappers REMOVED
(``unwrap_torch``) and reinstalled afterwards. ``unwrap_torch`` restores
each callable from the append-only ``_decorated_to_orig`` ledger, whose
orig side is recorded by the wrap machinery BEFORE the decorator sees the
function -- so a corrupted decorator cannot poison the restoration, which
is exactly what makes this a genuinely independent observation root.
Re-install is cheap (cached wrapper maps; no re-decoration).

Residual, disclosed (r4 census MED-2 scope correction): the per-op replay
still executes the RECORDED capture callable, so replay alone remains
wrap-state-correlated. With the phase-0 ground truth pristine, a
wrapper-layer distortion that REACHES the final output diverges from
ground truth and fails validation regardless of process wrap state --
that (and only that) is the class the probe proved closed. A distortion
MASKED downstream (a saturating clamp, a sign/argmax head) matches the
pristine final output AND matches its own corrupt replay, so it still
validates clean in-process; the cross-process mode-vs-wrapper
differential harness (``tools/differential_capture.py``) is the
independent root for that masked-interior class, and
``tests/test_validation_pristine_ground_truth.py`` pins the boundary.

Ledger integrity (r4 census MED-1): ``_decorated_to_orig`` is the SINGLE
shared root behind capture's orig grab, this oracle's restoration, and
sparse-runnable callable resolution -- a ledger entry whose "orig" is
itself a TorchLens wrapper (the orphaned prior-generation-snapshot
failure mode) would make the "pristine" forward run a wrapper and bless
coherently-corrupt captures. Wrappers are structurally identifiable
(``__tl_wrapper_name__`` is stamped at construction), so the oracle
scans the ledger before unwrapping and REFUSES on any wrapper-marked
orig value rather than blessing through a poisoned restoration.

If the wrappers cannot be removed (a capture is active on this thread --
only reachable by calling validation from inside a traced forward), the
caller must REFUSE to bless rather than silently validating through the
wrapper root; ``pristine_torch_oracle`` propagates the typed
``CaptureNotActiveError`` for the caller to convert into a fail-closed
verdict.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..backends.torch.wrappers import CompletenessWitnessMode, EscapeDetectorMode


@contextmanager
def pristine_torch_oracle() -> Iterator[bool]:
    """Run the enclosed oracle forward on pristine (unwrapped) torch.

    Yields
    ------
    bool
        ``True`` when the block runs on pristine torch: either torch was
        never wrapped in this process, or the wrappers were removed for the
        block and are reinstalled on exit (with the process-level escape
        detector and completeness-witness modes preserved -- teardown resets
        them, so they are snapshotted and re-applied).

    Raises
    ------
    torchlens.errors.CaptureNotActiveError
        If a capture is currently active (propagated from ``unwrap_torch``)
        or the unwrap ledger records a wrapper as an original callable
        (``code="pristine_ledger_poisoned"``). Callers convert either into
        a fail-closed (never-bless) verdict.
    """

    from .. import _state

    if not _state._is_decorated:
        yield True
        return

    # Ledger integrity (r4 census MED-1): refuse before unwrapping if any
    # recorded "orig" is itself a TorchLens wrapper -- restoring it would
    # run the pristine forward THROUGH a wrapper and bless coherently.
    poisoned = next(
        (
            orig
            for orig in _state._decorated_to_orig.values()
            if hasattr(orig, "__tl_wrapper_name__")
        ),
        None,
    )
    if poisoned is not None:
        from .._errors import CaptureNotActiveError

        raise CaptureNotActiveError(
            "the _decorated_to_orig unwrap ledger records a TorchLens wrapper "
            f"({getattr(poisoned, '__tl_wrapper_name__', '<unknown>')!s}) as an "
            "original callable, so the pristine-torch restoration cannot be "
            "trusted",
            code="pristine_ledger_poisoned",
            remedy=(
                "restart the process; if this recurs, a wrap-machinery defect is "
                "recording wrapped callables as originals -- report it"
            ),
        )

    # _state annotates the modes as bare str; wrap_torch validates against
    # the closed vocabularies, so the casts only restore the literal types.
    detector_mode = cast("EscapeDetectorMode", _state._escape_detector_mode)
    witness_mode = cast("CompletenessWitnessMode", _state._completeness_witness_mode)

    from ..backends.torch.wrappers import unwrap_torch, wrap_torch

    unwrap_torch()
    try:
        yield True
    finally:
        wrap_torch(
            escape_detector=detector_mode,
            completeness_witness=witness_mode,
        )

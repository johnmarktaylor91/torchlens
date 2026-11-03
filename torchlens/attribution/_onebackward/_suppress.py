"""Read-side backward suppression (M(reads) item 0b, decision D2).

Every one-backward read runs its ``autograd.grad`` call inside this context.
Without it, the read's own first backward is captured as a TorchLens backward
pass: per-op ``grad_fn_handle`` refs are nulled (538 -> 0 measured on gpt2,
destroying the read's own addressing), capture counters advance, and pinned
autograd refs grow without bound (~670 nodes leaked per target, 588 -> 21932
over 32 calls). The mechanism generalizes the shipped receptive-field probe
context (``receptive_field/_gradient.py::_probe_suppressed``): TorchLens
backward wrappers and gradient-payload hooks are gated off, protected capture
counters are snapshotted and asserted unchanged in ``finally``, and this
module adds the read-specific non-contamination tripwire on the pinned
autograd-ref count. User-installed PyTorch hooks still run inside the context
(documented boundary).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ...receptive_field._gradient import _probe_suppressed
from ._errors import ReadInternalError

__all__ = ["read_suppressed"]


def _pinned_ref_count(trace: Any) -> int | None:
    """Return the trace's pinned autograd-ref count without creating the list.

    Parameters
    ----------
    trace:
        Trace whose ``_backward_gradfn_refs`` strong-ref list is inspected.

    Returns
    -------
    int | None
        Number of pinned autograd node refs, or ``None`` when the trace has
        no ref list at all (loaded / preview traces).
    """

    refs = trace.__dict__.get("_backward_gradfn_refs")
    return None if refs is None else len(refs)


@contextmanager
def read_suppressed(trace: Any) -> Iterator[None]:
    """Run one read backward with TorchLens backward capture suppressed.

    Composes the shipped probe-suppression context (backward wrappers and
    gradient hooks off; ``num_backward_passes`` / ``has_gradients`` /
    backward-event counters snapshotted and asserted unchanged; state restored
    in ``finally``) with the read tripwire: the pinned autograd-ref count must
    be exactly unchanged, because ref growth is the measured leak axis of an
    unsuppressed read (D2). A failed tripwire is an internal contract breach,
    never a user error.

    Parameters
    ----------
    trace:
        Live trace whose autograd registry the read is about to consume.

    Yields
    ------
    None
        Control while backward capture is suppressed.

    Raises
    ------
    ReadInternalError
        If the read leaked pinned autograd refs despite suppression.
    """

    before = _pinned_ref_count(trace)
    inner_error: BaseException | None = None
    try:
        with _probe_suppressed(trace):
            yield
    except BaseException as exc:  # re-raised below after the tripwire check
        inner_error = exc
        raise
    finally:
        after = _pinned_ref_count(trace)
        if after != before:
            leak = ReadInternalError(
                "One-backward read suppression leaked pinned autograd refs "
                f"({before!r} -> {after!r}). This is a TorchLens contract "
                "breach, not a user error. Remedy: report this as a bug; "
                "re-capture the trace before further reads",
                code="read_suppression_leak",
                refs_before=before,
                refs_after=after,
            )
            if inner_error is None:
                raise leak
            raise leak from inner_error

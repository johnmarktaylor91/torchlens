"""Record consistency for a non-torch backend that declares backward capture.

``non_torch_backward_inert`` was written when only PyTorch recorded backward:
every non-torch trace had to carry no backward records. A backend whose spec
declares ``backward_capture`` now records them through the neutral backward
builder (``backends/_backward_neutral.py``), so for that backend the inert arms
are replaced by the consistency contract below; a backend that does not declare
backward capture still runs every inert arm.

The contract, for a declaring backend:

- every ``GradFn`` in ``_grad_fn_logs`` belongs to a recorded backward pass: it
  fired in one (a ``GradFnCall`` whose ``backward_pass_number`` is a key of
  ``_backward_pass_logs``) or it is one of a recorded pass's roots, and no call
  names a pass that was not recorded;
- the pass counts agree: the recorded pass numbers are exactly
  ``1..num_backward_passes`` (the density PyTorch's backward check holds);
- ``has_backward_pass`` is true exactly when a pass record exists, and no
  grad-fn record exists without a pass record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..errors._validation import MetadataInvariantError

__all__ = ("check_declared_backward_records",)


def _values(container: Any) -> list[Any]:
    """Return the records of a mapping-like container (``None`` means none)."""

    if not container:
        return []
    if isinstance(container, Mapping):
        return list(container.values())
    inner = getattr(container, "_dict", None)
    if isinstance(inner, Mapping):
        return list(inner.values())
    return list(container)


def _call_pass_numbers(grad_fn: Any) -> list[Any]:
    """Return the ``backward_pass_number`` of every call recorded on ``grad_fn``."""

    return [
        getattr(call, "backward_pass_number", None)
        for call in _values(getattr(grad_fn, "calls", None))
    ]


def _pass_roots(pass_logs: Mapping[Any, Any]) -> set[Any]:
    """Return every root grad-fn id any recorded pass names."""

    roots: set[Any] = set()
    for record in pass_logs.values():
        roots.update(getattr(record, "root_grad_fn_ids", None) or ())
    return roots


def _check_flag_and_counts(trace: Any, name: str, pass_logs: Mapping[Any, Any]) -> None:
    """Fail when ``has_backward_pass`` or the pass count disagrees with the records."""

    has_pass = bool(trace._autograd_has_backward_pass)
    if pass_logs and not has_pass:
        raise MetadataInvariantError(
            name,
            f"{len(pass_logs)} backward pass record(s) are present but has_backward_pass is false",
            code="non_torch_backward_inert",
        )
    if has_pass and not pass_logs:
        raise MetadataInvariantError(
            name,
            "has_backward_pass is true but no backward pass record is present",
            code="non_torch_backward_inert",
        )
    if trace._grad_fn_logs and not pass_logs:
        raise MetadataInvariantError(
            name,
            "grad_fn records are present but no backward pass record is present",
            code="non_torch_backward_inert",
        )
    num_passes = trace._num_backward_passes or 0
    expected = list(range(1, num_passes + 1)) if isinstance(num_passes, int) else None
    if sorted(pass_logs) != expected:
        raise MetadataInvariantError(
            name,
            f"num_backward_passes={num_passes!r} disagrees with the recorded backward pass "
            f"numbers {sorted(pass_logs)!r} (expected exactly 1..num_backward_passes)",
            code="non_torch_backward_inert",
        )


def _check_grad_fn_ownership(
    grad_fns: Iterable[tuple[Any, Any]], name: str, pass_logs: Mapping[Any, Any]
) -> None:
    """Fail on a grad-fn record that no recorded backward pass owns."""

    recorded = set(pass_logs)
    roots = _pass_roots(pass_logs)
    for object_id, grad_fn in grad_fns:
        label = getattr(grad_fn, "label", object_id)
        pass_numbers = _call_pass_numbers(grad_fn)
        unrecorded = sorted({str(number) for number in pass_numbers if number not in recorded})
        if unrecorded:
            raise MetadataInvariantError(
                name,
                f"grad_fn {label!r} has calls in backward pass(es) {unrecorded} that were "
                "not recorded",
                code="non_torch_backward_inert",
            )
        if not pass_numbers and object_id not in roots:
            raise MetadataInvariantError(
                name,
                f"grad_fn {label!r} is not owned by a recorded backward pass: it has no "
                "call and is no recorded pass's root",
                code="non_torch_backward_inert",
            )


def check_declared_backward_records(trace: Any, name: str) -> None:
    """Check the backward records of a backend that declares backward capture.

    Parameters
    ----------
    trace:
        Postprocessed non-torch trace whose backend declares ``backward_capture``.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If a grad-fn record is not owned by a recorded pass, the pass counts
        disagree, or ``has_backward_pass`` disagrees with the records.
    """

    pass_logs: Mapping[Any, Any] = trace._backward_pass_logs or {}
    _check_flag_and_counts(trace, name, pass_logs)
    _check_grad_fn_ownership((trace._grad_fn_logs or {}).items(), name, pass_logs)

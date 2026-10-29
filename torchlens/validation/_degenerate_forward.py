"""Type a perturbation failure whose saved output no perturbation could move.

A perturbation check fails when replaying an op with a perturbed parent gives
back its saved output unchanged. Usually that means the recorded parent does not
influence the op (a capture bug). Two degenerate forwards fail the same way with
no capture bug at all, because the saved value itself leaves nothing to move:

- every saved output element is non-finite (NaN or inf), so the perturbed replay
  is NaN wherever the saved value is and the comparison cannot see the parent;
- an in-forward ``torch.autograd.grad`` gradient is identically zero (for example
  a zero-initialised output layer), so the gradient stays zero under any
  perturbation of its inputs.

:func:`perturbation_insensitive_failure` records such a failure under its own
check, ``degenerate_forward``, instead of ``perturbation``. It relabels the
failure only: the decision stays ``perturbation_insensitive`` and the run still
fails. A partly non-finite output, or a finite forward that is really
insensitive, keeps the plain ``perturbation`` check.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from ..backends.torch._autograd_grad_boundary import is_autograd_grad_recorder
from .diagnostics import CHECK_DEGENERATE_FORWARD, CHECK_PERTURBATION, ValidationFailure

if TYPE_CHECKING:
    from ..data_classes.op import Op

DEGENERATE_NON_FINITE_OUTPUT = "non_finite_output"
DEGENERATE_ZERO_GRADIENT = "zero_gradient"


def _out_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor in a saved output payload.

    Parameters
    ----------
    value:
        An op's saved output: a tensor, or a tuple, list or dict holding tensors.

    Yields
    ------
    torch.Tensor
        Each tensor found, depth first.
    """

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _out_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _out_tensors(item)


def degenerate_forward_kind(layer: Op) -> str | None:
    """Return why ``layer``'s saved output cannot show a perturbation, if it cannot.

    Parameters
    ----------
    layer:
        The op whose perturbation check found the output unchanged.

    Returns
    -------
    str or None
        ``"non_finite_output"`` when every element of every saved output tensor
        is non-finite; ``"zero_gradient"`` when the op is an in-forward
        ``autograd.grad`` boundary whose saved gradients are all exactly zero;
        ``None`` otherwise, including an empty or partly finite output.
    """

    tensors = list(_out_tensors(getattr(layer, "out", None)))
    if not tensors or any(tensor.numel() == 0 for tensor in tensors):
        return None
    try:
        with torch.no_grad():
            if not any(bool(torch.isfinite(tensor).any()) for tensor in tensors):
                return DEGENERATE_NON_FINITE_OUTPUT
            if is_autograd_grad_recorder(getattr(layer, "func", None)) and not any(
                bool(tensor.any()) for tensor in tensors
            ):
                return DEGENERATE_ZERO_GRADIENT
    except (RuntimeError, TypeError):
        return None
    return None


def perturbation_insensitive_failure(
    layer: Op,
    op_label: str,
    perturbed_parents: list[str],
) -> ValidationFailure:
    """Build the failure record for an op whose output ignored a perturbed parent.

    Parameters
    ----------
    layer:
        The op whose perturbed replay reproduced its saved output.
    op_label:
        Label the failure is reported at.
    perturbed_parents:
        Labels of the parents that were perturbed.

    Returns
    -------
    ValidationFailure
        A ``degenerate_forward`` failure when :func:`degenerate_forward_kind`
        proves the saved output cannot move, else a ``perturbation`` failure.
    """

    kind = degenerate_forward_kind(layer)
    func_name = getattr(layer, "func_name", None)
    extra: dict[str, Any] = {"perturbed_parents": list(perturbed_parents)}
    if kind is None:
        return ValidationFailure(
            check=CHECK_PERTURBATION,
            op_label=op_label,
            func_name=func_name,
            message=(
                "output insensitive to perturbing parent(s) "
                f"{perturbed_parents}; the parent does not influence this op's value"
            ),
            extra=extra,
        )
    extra["degenerate_kind"] = kind
    reason = (
        "every saved output element is non-finite"
        if kind == DEGENERATE_NON_FINITE_OUTPUT
        else "the saved autograd.grad gradient is identically zero"
    )
    return ValidationFailure(
        check=CHECK_DEGENERATE_FORWARD,
        op_label=op_label,
        func_name=func_name,
        message=(
            f"degenerate forward: {reason}, so perturbing parent(s) {perturbed_parents} "
            "cannot change it and this edge cannot be checked"
        ),
        extra=extra,
    )

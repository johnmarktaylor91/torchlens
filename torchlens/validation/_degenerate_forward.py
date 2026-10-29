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

:func:`fail_if_final_outputs_non_finite` is the run-level guard: a forward whose
every final output element is non-finite can pass every edge check (a perturbed
parent turns NaN into finite values, and a replay from all-NaN arguments matches
the saved NaN whatever the func), so it would validate while proving nothing. It
fails such a run under the same check. It only ever turns a pass into a failure.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
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


def _all_non_finite(tensors: list[torch.Tensor]) -> bool:
    """Return whether every element of every tensor is non-finite (none empty).

    Parameters
    ----------
    tensors:
        Tensors to inspect.

    Returns
    -------
    bool
        ``False`` for an empty list, an empty tensor, or any finite element.
    """

    if not tensors or any(tensor.numel() == 0 for tensor in tensors):
        return False
    try:
        with torch.no_grad():
            return not any(bool(torch.isfinite(tensor).any()) for tensor in tensors)
    except (RuntimeError, TypeError):
        return False


def fail_if_final_outputs_non_finite(trace: Any, ground_truth_outputs: Sequence[Any]) -> bool:
    """Fail a passing run whose final model outputs are entirely non-finite.

    Parameters
    ----------
    trace:
        Live validation Trace whose replay validation passed.
    ground_truth_outputs:
        The model's final output tensors from the ground-truth forward.

    Returns
    -------
    bool
        ``True`` when the guard fired: a ``degenerate_forward`` failure is recorded
        at the first output layer and the trace's replay status becomes ``failed``.
        ``False`` (nothing changed) for a partly finite, empty or non-float output.
    """

    tensors = list(_out_tensors(list(ground_truth_outputs)))
    if not _all_non_finite(tensors):
        return False
    from .diagnostics import record_validation_failure

    output_layers = [str(label) for label in getattr(trace, "output_layers", []) or []]
    record_validation_failure(
        trace,
        ValidationFailure(
            check=CHECK_DEGENERATE_FORWARD,
            op_label=output_layers[0] if output_layers else None,
            message=(
                "degenerate forward: every element of every final model output is "
                "non-finite, so validation cannot check the forward"
            ),
            extra={
                "degenerate_kind": DEGENERATE_NON_FINITE_OUTPUT,
                "scope": "final_outputs",
                "output_layers": output_layers,
            },
        ),
    )
    status = trace.validation_replay_status
    trace._validation_replay_status = dataclasses.replace(
        status,
        state="failed",
        reason="degenerate_forward",
        message="Replay validation failed: every final model output is non-finite.",
        failed_node_count=status.failed_node_count + 1,
    )
    return True

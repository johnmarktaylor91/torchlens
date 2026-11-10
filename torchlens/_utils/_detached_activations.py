"""Guidance markers for retained activations whose autograd path TorchLens severed.

Re-exported from ``torchlens._utils._tensor_label``, where these helpers used to live.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from torch import Tensor
from torch.utils.weak import WeakIdKeyDictionary

from ._internal_read import internal_scalar_read

# Retained activations are TorchLens-owned tensors, but this diagnostic state stays in an
# identity-keyed weak registry instead of their ``TensorMeta``. Capture labels are session-scoped
# and retired between traces; the guidance must remain available for exactly as long as the
# retained payload itself remains alive.
_DETACHED_SAVED_ACTIVATIONS: WeakIdKeyDictionary = WeakIdKeyDictionary()

# This is intentionally a small allowlist. Each operation preserves an autograd path from a
# floating/complex tensor input to its tensor output when that input requires grad. A false
# positive is worse than missing guidance for an unusual loss-building operation.
_DETACHED_ACTIVATION_PROPAGATION_FUNCS = frozenset(
    {
        "__abs__",
        "__add__",
        "__getitem__",
        "__mul__",
        "__neg__",
        "__pow__",
        "__radd__",
        "__rmul__",
        "__rsub__",
        "__rtruediv__",
        "__sub__",
        "__truediv__",
        "abs",
        "absolute",
        "add",
        "clone",
        "contiguous",
        "div",
        "divide",
        "flatten",
        "mean",
        "mul",
        "nansum",
        "neg",
        "negative",
        "permute",
        "pow",
        "prod",
        "reshape",
        "reshape_as",
        "select",
        "squeeze",
        "sub",
        "sum",
        "t",
        "transpose",
        "true_divide",
        "unsqueeze",
        "view",
        "view_as",
    }
)
_DETACHED_ACTIVATION_ANY_INPUT_FUNCS = frozenset(
    {
        "__add__",
        "__mul__",
        "__pow__",
        "__radd__",
        "__rmul__",
        "__rsub__",
        "__rtruediv__",
        "__sub__",
        "__truediv__",
        "add",
        "div",
        "divide",
        "mul",
        "pow",
        "sub",
        "true_divide",
    }
)


def mark_detached_saved_activation(
    source: Tensor,
    retained: Tensor,
    label: str | None,
) -> None:
    """Mark a retained activation only when TorchLens severed its autograd path.

    Parameters
    ----------
    source:
        Live operation output before TorchLens retained a payload copy.
    retained:
        Tensor payload exposed by the resulting ``Op``.
    label:
        Best available operation label for diagnostic guidance.

    Returns
    -------
    None
        The weak identity registry is updated only for a proven TorchLens detach.
    """

    # These `requires_grad`/`grad_fn` reads are TorchLens-internal bookkeeping run for
    # every saved activation (including registered-buffer sources). Mark them as internal
    # scalar reads so the host-escape witness does not record them as host declared-state
    # facts.
    with internal_scalar_read():
        source_was_connected = bool(source.requires_grad and source.grad_fn is not None)
        retained_is_disconnected = not retained.requires_grad and retained.grad_fn is None
    if source_was_connected and retained_is_disconnected:
        _DETACHED_SAVED_ACTIVATIONS[retained] = label or "saved activation"


def has_detached_saved_activations() -> bool:
    """Return whether any guided-backward activation marker is still alive.

    Returns
    -------
    bool
        ``True`` when steady-state wrappers need to inspect tensor lineage.
    """

    return bool(_DETACHED_SAVED_ACTIVATIONS)


def propagate_detached_saved_activation(
    func_name: str,
    inputs: Iterable[Any],
    outputs: Iterable[Any],
) -> None:
    """Propagate guided-backward provenance through safe loss-building operations.

    Parameters
    ----------
    func_name:
        Decorated Torch callable name.
    inputs:
        Direct tensor arguments to the call.
    outputs:
        Direct tensor outputs from the call.

    Returns
    -------
    None
        Eligible disconnected outputs inherit the retained activation label weakly.
    """

    if not _DETACHED_SAVED_ACTIVATIONS or func_name not in _DETACHED_ACTIVATION_PROPAGATION_FUNCS:
        return
    tensor_inputs = [value for value in inputs if isinstance(value, Tensor)]
    if func_name in _DETACHED_ACTIVATION_ANY_INPUT_FUNCS:
        candidate_inputs = tensor_inputs
    else:
        candidate_inputs = tensor_inputs[:1]
    marked_labels = [
        label
        for value in candidate_inputs
        if (label := _DETACHED_SAVED_ACTIVATIONS.get(value)) is not None
    ]
    if not marked_labels:
        return
    label = marked_labels[0]
    for output in outputs:
        if not isinstance(output, Tensor):
            continue
        if output.requires_grad or output.grad_fn is not None:
            continue
        if output.is_floating_point() or output.is_complex():
            _DETACHED_SAVED_ACTIVATIONS[output] = label


def detached_saved_activation_label(tensor: Tensor) -> str | None:
    """Return the label for an exactly registered detached activation lineage.

    Parameters
    ----------
    tensor:
        Candidate autograd root tensor.

    Returns
    -------
    str | None
        Retained activation label for an identity match, otherwise ``None``.
    """

    return cast(str | None, _DETACHED_SAVED_ACTIVATIONS.get(tensor))

"""In-domain integer perturbations for native extension operators, failing closed otherwise.

A native extension operator (one a C++ extension registers outside the ``aten`` and
``prims`` namespaces, such as torch_scatter's ``gather_csr`` or torch_sparse's ``spmm_sum``)
often reads an integer index or pointer argument without bounds checks. The generic integer
draw keeps values inside the saved tensor's ``[min, max]`` range but breaks the structure a
CSR index pointer needs (non-decreasing, first entry and last entry fixed), so a perturbed
pointer made the kernel write out of range and abort the host process. Validation must never
be able to crash the user's process.

The rule: when the perturbed parent is an integer tensor in an argument slot of a native
extension operator, its perturbed value comes only from this module. A registered slot gets
an in-domain value (:func:`csr_pointer_shift_values` or :func:`index_left_shift_values`), a
slot the kernel reads only as data keeps the generic draw, and any other integer slot of such
an operator gets none: the caller records the edge ``unverified`` with
:data:`UNCHECKED_NATIVE_INDEX_REASON` instead of running the kernel. The
perturbation check itself is unchanged: an in-domain value still has to move the replayed
output, so a dead edge still fails. Bounds-checked ``aten``/``prims`` operators are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..backends.torch._foreign_data import CALL_OPERATOR_ATTR, recorded_custom_op_name

if TYPE_CHECKING:
    from .._data_classes.op import Op

UNCHECKED_NATIVE_INDEX_REASON = "unchecked_native_index_argument"
"""Unverified reason: no proven in-domain perturbation for an integer argument of a native
extension operator, so the perturbed call was not run."""

_CHECKED_NAMESPACES = frozenset({"aten", "prims"})
"""Operator namespaces whose kernels bounds-check their integer index arguments."""

_DEFAULT_STRATEGY = "default"
_FIRST_RETRY_STRATEGY = "step_up"
"""The first retry rung (``_shift_probe.VALUE_STEP_STRATEGIES[0]``): it gets the alternate
in-domain value; every later rung gets none, which ends the retry ladder."""


def csr_pointer_shift_values(values: torch.Tensor, alternate: bool) -> torch.Tensor | None:
    """Move interior CSR boundaries by one, keeping the pointer valid.

    Parameters
    ----------
    values:
        Saved index pointer, shape ``(..., n + 1)``, non-decreasing along the last dim.
    alternate:
        ``False`` moves every boundary that can move down by one; ``True`` moves every
        boundary that can move up by one.

    Returns
    -------
    torch.Tensor or None
        A pointer with the same first and last entries, still non-decreasing, so every
        segment stays inside ``[first, last]``; ``None`` when no interior boundary can
        move in that direction (or the saved pointer is not non-decreasing).
    """

    if values.dim() == 0 or values.shape[-1] < 3:
        return None
    if bool((values[..., 1:] < values[..., :-1]).any()):
        return None
    interior_slice = (Ellipsis, slice(1, -1))
    interior = values[interior_slice]
    if alternate:
        movable = interior < values[..., 2:]
        step = 1
    else:
        movable = values[..., :-2] < interior
        step = -1
    if not bool(movable.any()):
        return None
    shifted = values.clone()
    shifted[interior_slice] = interior + movable.to(values.dtype) * step
    return shifted


def index_left_shift_values(values: torch.Tensor, alternate: bool) -> torch.Tensor | None:
    """Shift index entries one position along the last dim, reusing saved values only.

    Parameters
    ----------
    values:
        Saved integer index tensor (sorted or not).
    alternate:
        ``False`` takes each entry from its right neighbour (the last entry stays);
        ``True`` takes each entry from its left neighbour (the first entry stays).

    Returns
    -------
    torch.Tensor or None
        Every entry is a value the saved tensor already held in the same row, so the
        result stays inside the operator's index domain, and a non-decreasing (or
        non-increasing) row stays sorted. ``None`` when the shift changes nothing.
    """

    if values.dim() == 0 or values.shape[-1] < 2:
        return None
    if alternate:
        shifted = torch.cat([values[..., :1], values[..., :-1]], dim=-1)
    else:
        shifted = torch.cat([values[..., 1:], values[..., -1:]], dim=-1)
    if torch.equal(shifted, values):
        return None
    return shifted


_Builder = Callable[[torch.Tensor, bool], torch.Tensor | None]

_CSR = csr_pointer_shift_values
_INDEX = index_left_shift_values
_VALUE = "value"
"""Slot marker: the kernel copies or reduces the slot's values as data, or its forward never
reads the slot (torch_sparse saves those for backward), so no draw can address memory and
the generic perturbation runs unchanged."""

_SCATTER_CSR = {0: _VALUE, 1: _CSR, 2: _VALUE}
_SCATTER_COO = {0: _VALUE, 1: _INDEX, 2: _VALUE}

_NATIVE_INDEX_SLOTS: dict[str, dict[int, _Builder | str]] = {
    # torch_scatter CSR family (src, indptr, out): indptr is a CSR pointer over src.
    **dict.fromkeys(
        ("gather_csr", "segment_sum_csr", "segment_mean_csr", "segment_min_csr"),
        _SCATTER_CSR,
    ),
    "segment_max_csr": _SCATTER_CSR,
    # torch_scatter COO family (src, index, out[, dim_size]): index sorted, in range.
    **dict.fromkeys(
        ("gather_coo", "segment_sum_coo", "segment_mean_coo", "segment_min_coo"),
        _SCATTER_COO,
    ),
    "segment_max_coo": _SCATTER_COO,
    # torch_scatter (src, index, dim, out, dim_size): index entries address the output dim.
    **dict.fromkeys(
        ("scatter_mul", "scatter_min", "scatter_max"), {0: _VALUE, 1: _INDEX, 3: _VALUE}
    ),
    # torch_sparse spmm_sum (row, rowptr, col, value, colptr, csr2csc, mat) and spmm_mean
    # (row, rowptr, col, value, rowcount, colptr, csr2csc, mat): the forward kernel reads
    # rowptr (a CSR pointer over the edges), col (rows of mat), value and mat; row,
    # rowcount, colptr and csr2csc are saved for backward only. spmm_min/spmm_max are
    # (rowptr, col, value, mat).
    "spmm_sum": {0: _VALUE, 1: _CSR, 2: _INDEX, 3: _VALUE, 4: _VALUE, 5: _VALUE, 6: _VALUE},
    "spmm_mean": {
        **dict.fromkeys((0, 3, 4, 5, 6, 7), _VALUE),
        1: _CSR,
        2: _INDEX,
    },
    **dict.fromkeys(("spmm_min", "spmm_max"), {0: _CSR, 1: _INDEX, 2: _VALUE, 3: _VALUE}),
}
"""Recorded operator name -> {positional slot: in-domain builder or ``_VALUE``}. An integer
slot with no entry has no proven-safe perturbation and fails closed."""


@dataclass(frozen=True)
class NativeIndexPerturbation:
    """Perturbed value for an integer slot of a native extension operator.

    ``values`` is ``None`` when no proven in-domain value exists; the caller must then
    record the edge unverified with ``reason`` rather than run the operator.
    """

    values: torch.Tensor | None
    reason: str = UNCHECKED_NATIVE_INDEX_REASON


def native_extension_operator_name(layer: Op) -> str | None:
    """Return the recorded name of a native extension operator, or ``None``.

    Parameters
    ----------
    layer:
        Op record being replayed.

    Returns
    -------
    str or None
        The operator's unqualified name when its replay callable calls a registered
        ``torch.ops`` operator outside the bounds-checked namespaces; ``None`` for
        every other op.
    """

    func = getattr(layer, "func", None)
    qualified = recorded_custom_op_name(getattr(func, CALL_OPERATOR_ATTR, func))
    if qualified is None:
        return None
    namespace, name = qualified.split("::")
    return None if namespace in _CHECKED_NAMESPACES else name


def native_index_perturbation(
    layer: Op,
    arg_type: str,
    key: Any,
    parent_values: torch.Tensor,
    perturb_strategy: str,
) -> NativeIndexPerturbation | None:
    """Return the only allowed perturbation of a native operator's integer argument.

    Parameters
    ----------
    layer:
        Op being replayed.
    arg_type:
        ``"args"`` or ``"kwargs"``: the container of the perturbed slot.
    key:
        Position or keyword of the perturbed slot.
    parent_values:
        Saved parent value at that slot.
    perturb_strategy:
        The replay ladder's strategy name (``"default"`` first, then retries).

    Returns
    -------
    NativeIndexPerturbation or None
        ``None`` when the rule does not apply (not an integer tensor, not a native
        extension operator, or a slot the table marks ``_VALUE``), so the generic
        perturbation runs. Otherwise the in-domain
        value for a registered slot (the alternate move on the first retry rung), or a
        result with ``values=None`` when no proven-safe value exists.
    """

    if not isinstance(parent_values, torch.Tensor) or not _is_integer_tensor(parent_values):
        return None
    name = native_extension_operator_name(layer)
    if name is None:
        return None
    builder = _NATIVE_INDEX_SLOTS.get(name, {}).get(key) if arg_type == "args" else None
    if builder == _VALUE:
        return None
    if builder is None or isinstance(builder, str) or parent_values.is_sparse:
        return NativeIndexPerturbation(None)
    if perturb_strategy == _DEFAULT_STRATEGY:
        return NativeIndexPerturbation(builder(parent_values, False))
    if perturb_strategy == _FIRST_RETRY_STRATEGY:
        return NativeIndexPerturbation(builder(parent_values, True))
    return NativeIndexPerturbation(None)


def _is_integer_tensor(values: torch.Tensor) -> bool:
    """Return whether a tensor holds integer (not bool, float or complex) values."""

    return not (values.dtype == torch.bool or values.is_floating_point() or values.is_complex())

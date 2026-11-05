"""Degenerate-domain exemptions: arguments whose saved call admits one valid value.

When an argument's valid domain, given the op's other saved arguments, holds
exactly one value (or every valid value yields the same output), no
perturbation can change it without making the call invalid or leaving the
output unchanged, so its value provably cannot influence the output beyond the
plain replay, which still runs and must match. Each check here is keyed on one
op family and the perturbed parent's argument slot, reads saved values with no
tolerance, and returns False for anything wider:

- the index-domain ops of ``_INDEX_DOMAIN_ARG_SPECS``: a domain of ``n <= 1``
  valid indices, or a saved index with zero in-domain entries;
- ``masked_scatter``: the mask when the saved source is empty, so the all-False
  mask is the only valid mask;
- ``ctc_loss``: ``target_lengths`` of a batch of one with concatenated 1-D
  targets, whose single length must equal the target count;
- ``conv_transpose{1,2,3}d``: ``output_padding`` when every stride and
  dilation is 1, so 0 is the only valid padding;
- ``argsort``: the input over a dim of size 1, whose indices are 0 for every
  input;
- ``__setitem__`` (posthoc, in ``_saved_value_proofs``): the value when a
  saved bool mask component selects nothing.

``exemptions.py`` binds the pre-execution checks in ``CUSTOM_EXEMPTION_CHECKS``
through ``_exemption_rows_ops``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member
from ._index_domain import (
    _index_domain_size,
    _parent_is_index_domain_arg,
    _saved_integer_index_tensor,
)

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

#: Index ops of the degenerate row that address the destination dim, where a
#: negative entry in ``[-n, 0)`` may name a real position. Counting it as
#: in-domain only narrows the zero-in-domain branch (ops that reject negative
#: indices never save one).
_WRAP_AWARE_INDEX_OPS = frozenset(
    {
        "index_add",
        "index_add_",
        "index_copy",
        "index_copy_",
        "index_fill",
        "index_fill_",
        "take_along_dim",
    }
)

MASKED_SCATTER_SPELLINGS = ("masked_scatter", "masked_scatter_", "maskedscatter")
CONV_TRANSPOSE_SPELLINGS = ("conv_transpose1d", "conv_transpose2d", "conv_transpose3d")


def _check_index_domain_degenerate(self: Trace, layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt an index parent ONLY when no in-domain perturbation exists.

    A domain of ``n <= 1`` valid values, or a saved index tensor with zero
    in-domain entries (e.g. an all-``ignore_index`` target), admits no valid
    alternate index at all -- value-irrelevance is forced by the domain
    constraint, not assumed. Any perturbable domain returns False so the
    strict in-domain rotation check runs.

    Parameters
    ----------
    self:
        Trace being validated (unused; signature parity with the registry).
    layer:
        Captured index-consuming op.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbed parent is a provably unperturbable index arg.
    """

    del self
    if len(layers_to_perturb) != 1:
        return False
    if not _parent_is_index_domain_arg(layer, layers_to_perturb[0]):
        return False
    saved_index = _saved_integer_index_tensor(layer)
    if saved_index is None:
        return False
    domain_size = _index_domain_size(layer)
    if domain_size is None:
        return False
    if domain_size < 2:
        return True
    low = -domain_size if getattr(layer, "func_name", None) in _WRAP_AWARE_INDEX_OPS else 0
    in_domain = (saved_index >= low) & (saved_index < domain_size)
    return not bool(in_domain.any())


def _sole_parent_slot(layer: Op, layers_to_perturb: list[str], position: int, kwarg: str) -> bool:
    """Return whether the one perturbed parent fills only ``position`` or ``kwarg``.

    A parent nested inside the argument (key ``(position, i)``) counts as the
    same slot. Any second perturbed parent, or the parent also sitting at any
    other positional or keyword slot, returns False so callers fail closed.
    """

    if len(layers_to_perturb) != 1:
        return False
    label = layers_to_perturb[0]
    positions = record_member(layer, "parent_arg_positions", None) or {}
    arg_keys = [key for key, p in (positions.get("args", {}) or {}).items() if p == label]
    kwarg_keys = [key for key, p in (positions.get("kwargs", {}) or {}).items() if p == label]
    if not arg_keys and not kwarg_keys:
        return False
    if any(key != kwarg for key in kwarg_keys):
        return False
    return all(
        key == position or (isinstance(key, tuple) and key[:1] == (position,)) for key in arg_keys
    )


def _saved_arg(layer: Op, position: int, kwarg: str, default: Any = None) -> Any:
    """Return the saved argument at ``position`` or keyword ``kwarg``, else ``default``."""

    args: tuple[Any, ...] = record_member(layer, "saved_args", None) or ()
    kwargs = record_member(layer, "saved_kwargs", None) or {}
    if kwarg in kwargs:
        return kwargs[kwarg]
    return args[position] if len(args) > position else default


def check_masked_scatter_empty_source_exempt(
    self: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Exempt the ``masked_scatter`` mask when the saved source is empty.

    ``masked_scatter(input, mask, source)`` needs at least as many source
    elements as True mask entries, so with an empty source the all-False mask
    is the only valid mask and the output is the input. The input and source
    parents, and any mask over a non-empty source, stay strict.

    Parameters
    ----------
    self:
        Trace being validated (unused; signature parity with the registry).
    layer:
        Captured ``masked_scatter`` op.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbed parent is the mask of an empty-source call.
    """

    del self
    if not _sole_parent_slot(layer, layers_to_perturb, 1, "mask"):
        return False
    mask = _saved_arg(layer, 1, "mask")
    source = _saved_arg(layer, 2, "source")
    if not (isinstance(mask, torch.Tensor) and mask.dtype == torch.bool):
        return False
    if not isinstance(source, torch.Tensor) or source.numel() != 0:
        return False
    return not bool(mask.any())


def check_ctc_single_target_length_exempt(
    self: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Exempt ``ctc_loss`` target lengths of a batch of one over concatenated targets.

    With batched ``log_probs`` of shape ``(T, N, C)`` and 1-D (concatenated)
    targets, torch requires ``sum(target_lengths) == targets.numel()``. With
    ``N == 1`` the single length therefore has exactly one valid value. Padded
    2-D targets, unbatched calls, input lengths and any batch of two or more
    (a sum-preserving redistribution is a valid alternate) stay strict.

    Parameters
    ----------
    self:
        Trace being validated (unused; signature parity with the registry).
    layer:
        Captured ``ctc_loss`` op.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbed parent is the forced single target length.
    """

    del self
    if not _sole_parent_slot(layer, layers_to_perturb, 3, "target_lengths"):
        return False
    log_probs = _saved_arg(layer, 0, "log_probs")
    targets = _saved_arg(layer, 1, "targets")
    lengths = _saved_arg(layer, 3, "target_lengths")
    if not (isinstance(log_probs, torch.Tensor) and log_probs.ndim == 3):
        return False
    if not (isinstance(targets, torch.Tensor) and targets.ndim == 1):
        return False
    if not isinstance(lengths, torch.Tensor) or lengths.is_floating_point():
        return False
    if lengths.numel() != 1 or log_probs.shape[1] != 1:
        return False
    return int(lengths.reshape(-1)[0]) == targets.numel()


def _int_entries(value: Any, default: int) -> list[int] | None:
    """Return a conv size argument as a list of ints, or None when unreadable."""

    if value is None:
        return [default]
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return None
        return [int(entry) for entry in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        entries: list[int] = []
        for entry in value:
            read = _int_entries(entry, default)
            if read is None or len(read) != 1:
                return None
            entries.extend(read)
        return entries
    if type(value) is int:
        return [value]
    return None


def check_conv_transpose_unit_output_padding_exempt(
    self: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Exempt a transposed conv's ``output_padding`` when stride and dilation are 1.

    torch requires each ``output_padding`` entry to be smaller than the stride
    or the dilation of its dim, so with every stride and dilation equal to 1
    the only valid padding is 0. Any stride or dilation above 1, a non-zero
    saved padding and every other parent stay strict.

    Parameters
    ----------
    self:
        Trace being validated (unused; signature parity with the registry).
    layer:
        Captured ``conv_transpose{1,2,3}d`` op, ``(input, weight, bias, stride,
        padding, output_padding, groups, dilation)``.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbed parent is the forced zero ``output_padding``.
    """

    del self
    if not _sole_parent_slot(layer, layers_to_perturb, 5, "output_padding"):
        return False
    stride = _int_entries(_saved_arg(layer, 3, "stride"), 1)
    dilation = _int_entries(_saved_arg(layer, 7, "dilation"), 1)
    padding = _int_entries(_saved_arg(layer, 5, "output_padding"), 0)
    if stride is None or dilation is None or not padding:
        return False
    if any(entry != 1 for entry in (*stride, *dilation)):
        return False
    return all(entry == 0 for entry in padding)


def check_argsort_singleton_dim_exempt(
    self: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Exempt the input of an ``argsort`` over a dim of size 1.

    Sorting one element returns index 0 whatever its value, so the input value
    cannot reach the output. The saved output must be all zero (it is, for any
    such call). A sorted dim of two or more and every other parent stay strict.

    Parameters
    ----------
    self:
        Trace being validated (unused; signature parity with the registry).
    layer:
        Captured ``argsort`` op, ``(input, dim=-1, descending=False)``.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbed parent is the input of a one-element sort.
    """

    del self
    if not _sole_parent_slot(layer, layers_to_perturb, 0, "input"):
        return False
    source = _saved_arg(layer, 0, "input")
    dim = _saved_arg(layer, 1, "dim", -1)
    out = getattr(layer, "out", None)
    if not isinstance(source, torch.Tensor) or type(dim) is not int:
        return False
    if not isinstance(out, torch.Tensor) or bool(out.ne(0).any()):
        return False
    if source.ndim == 0:
        return True
    if not -source.ndim <= dim < source.ndim:
        return False
    return int(source.shape[dim]) == 1

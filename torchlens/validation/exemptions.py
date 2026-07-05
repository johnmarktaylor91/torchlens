"""Validation exemption registries for perturbation and forward-replay checks.

Four registries control which operations are exempt from validation, and why:

1. ``SKIP_VALIDATION_ENTIRELY`` -- ops whose output is nondeterministic even
   with identical inputs and RNG state (e.g., ``empty_like`` returns
   uninitialized memory).  Both forward replay AND perturbation are skipped.

2. ``SKIP_PERTURBATION_ENTIRELY`` -- ops where ALL args are structural (shape,
   type template) so perturbation can never change the output.  Forward
   replay still runs to verify correctness.

3. ``STRUCTURAL_ARG_POSITIONS`` -- ops where SPECIFIC arg positions are
   structural (e.g., the index tensor in ``embedding``).  If the perturbed
   layer's tensor matches one of these positions, perturbation is skipped
   for that parent only.

4. ``CUSTOM_EXEMPTION_CHECKS`` -- ops requiring per-case logic that doesn't
   fit a simple position mapping (e.g., ``__getitem__`` tensor indexing,
   ``lstm`` hidden/cell states).

Additionally, ``posthoc_perturb_check`` handles dynamic exemptions that can
only be determined AFTER executing the function -- cases where perturbation
genuinely doesn't change the output for valid reasons (bool output, type
casting, special-value args like all-zeros making perturbation irrelevant).
"""

# Exemptions are tripwire-sensitive. Add a new exemption only when the predicate
# is proved from runtime evidence and a negative test shows it cannot mask the
# unintended value-sensitive case.

from typing import Any, Callable, Dict, List, Set, TYPE_CHECKING, Union

import torch

from ..data_classes.op import Op
from ..utils.tensor_utils import tensor_all_nan

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


# ---------------------------------------------------------------------------
# Registry 1: Skip ALL validation (forward replay + perturbation).
# These funcs produce nondeterministic output (e.g. uninitialized memory),
# so even forward replay would fail.
# ---------------------------------------------------------------------------
SKIP_VALIDATION_ENTIRELY: Set[str] = {
    "empty_like",
    "new",  # torch.Tensor.new() — uninitialized memory
    "new_empty",  # torch.Tensor.new_empty() — uninitialized memory
    "new_empty_strided",  # torch.Tensor.new_empty_strided() — uninitialized memory
    "newempty",  # torch.Tensor.new_empty() — uninitialized memory
    "newemptystrided",  # torch.Tensor.new_empty_strided() — uninitialized memory
}

# ---------------------------------------------------------------------------
# Registry 2: Skip perturbation only (forward replay still runs).
# All args are structural — output doesn't depend on input values.
# ---------------------------------------------------------------------------
SKIP_PERTURBATION_ENTIRELY: Set[str] = {
    "expand_as",
    "new_zeros",
    "new_ones",
    "zero_",
    "copy_",
    "clamp",
    "fill_",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
    "meshgrid",
    "broadcast_tensors",
    # torchvision C++ ops (PyCapsule): nms, roi_align, etc. Perturbed coordinates
    # can segfault these native extensions since they bypass Python exception handling.
    "_op",
    # In-place RNG ops: output is determined by RNG state, not input values.
    "exponential_",
}

# ---------------------------------------------------------------------------
# Registry 3: Specific arg positions that are structural (not value-sensitive).
# When the perturbed layer's tensor matches saved_args[pos], skip perturbation.
# ---------------------------------------------------------------------------
STRUCTURAL_ARG_POSITIONS: Dict[str, Set[int]] = {
    "cross_entropy": {1},  # target labels (LongTensor)
    "embedding": {1},  # index tensor — random indices cause CUDA OOB
    "gather": {2},  # index tensor
    "index_select": {2},  # index tensor
    "scatter_": {2},  # index tensor
    "maskedfill": {1},  # mask tensor; TorchLens canonical name for Tensor.masked_fill
    "masked_fill": {1},  # mask tensor
    "masked_fill_": {1},  # mask tensor
    "_pad_packed_sequence": {1},  # lengths tensor
    "type_as": {1},  # type template tensor (value irrelevant)
    "new_tensor": {0},  # source tensor is a dtype/device/layout factory
    "newtensor": {0},  # canonicalized torch.Tensor.new_tensor spelling
}


# ---------------------------------------------------------------------------
# Custom exemption check functions
# Signature: callable(self, layer, layers_to_perturb) -> bool
#   self = Trace instance
#   layer = Op being validated
#   layers_to_perturb = list of layer labels being perturbed
# ---------------------------------------------------------------------------


def _check_getitem_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt __getitem__ when the perturbed layer is a structural arg (index tensor,
    or any non-data arg)."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    args = layer.saved_args

    # Case 1: perturbed layer IS the tensor index — tensor indexing is structural
    if isinstance(args[1], torch.Tensor) and torch.equal(perturbed_tensor, args[1]):
        return True

    # Case 2: perturbed layer is NOT the data tensor — must be a structural arg
    # (slice, int index logged as tensor, etc.)
    if not torch.equal(perturbed_tensor, args[0]):
        return True

    return False


# In-place ops whose FIRST positional arg (``args[0]``) is the destination
# tensor written into. The RoPE/normalizing-flow/GNN "partition-write into an
# uninitialized buffer" idiom chains these into ``empty``/``empty_like``/
# ``new_empty`` allocations. Every entry writes its destination at args[0]; the
# canonicalized TorchLens spellings (no underscore) are included alongside.
INPLACE_DESTINATION_WRITE_FUNCS: Set[str] = {
    "__setitem__",
    "index_copy_",
    "indexcopy_",
    "index_copy",
    "indexcopy",
    "index_fill_",
    "indexfill_",
    "index_fill",
    "indexfill",
    "index_add_",
    "indexadd_",
    "index_add",
    "indexadd",
    "masked_scatter_",
    "maskedscatter_",
    "masked_scatter",
    "maskedscatter",
    "scatter_",
    "scatter",
}


def _uninitialized_value_origin(op: Any, source_trace: Any, depth: int = 0) -> bool:
    """Return whether an op's VALUE is itself uninitialized memory.

    True iff ``op`` is a DIRECT uninitialized-memory source
    (``empty``/``empty_like``/``new_empty``/...). The exemption it gates skips
    the perturbation-insensitivity check ONLY when the perturbed destination
    parent has no meaningful value to perturb -- i.e. it is literally
    uninitialized allocation memory.

    The walk does NOT chain through in-place destination writes. An in-place
    write (``index_copy_``/``__setitem__``/...) produces a tensor that contains
    REAL written data -- it is never "uninitialized memory" as a whole, even
    when its destination buffer was allocated by ``empty_like``. Following the
    chain back to the allocation was a tripwire hole (B1 review,
    ``TwoIndexCopyDim2``): the second write of an
    ``out = empty_like(x); out.index_copy_(...); out.index_copy_(...)`` idiom
    consumes the FIRST write's live data through its destination parent, so a
    wrong replay that drops that destination dependency must still fail the
    perturbation check. Only a parent that is itself an allocation op is exempt;
    the ``depth``/``source_trace`` parameters are retained for signature
    compatibility but are no longer used to recurse.
    """

    if op is None:
        return False
    return getattr(op, "func_name", None) in SKIP_VALIDATION_ENTIRELY


def _perturbed_parent_is_uninitialized_setitem_dest(
    layer: Op,
    layers_to_perturb: List[str],
) -> bool:
    """Return whether a perturbed in-place-write parent is an uninitialized dest.

    Returns True iff EVERY perturbed parent of this in-place destination-write op
    occupies its DESTINATION slot (``args[0]``) AND that perturbed parent is
    ITSELF a direct uninitialized-memory allocation
    (``empty``/``empty_like``/``new_empty``). Perturbing literally-uninitialized
    allocation memory has no semantic meaning -- the unwritten positions are
    garbage overwritten by a sibling/subsequent write and never meaningfully
    consumed -- so a "no output change" result is not a capture bug.

    The check is deliberately NARROW so it cannot mask a real dropped-dependency
    bug: if the perturbed parent is a genuine data tensor -- INCLUDING a prior
    in-place write whose buffer was allocated by ``empty_like`` but which now
    holds real written data -- or is NOT the destination, this returns False and
    the strict perturbation check stands. The exemption does NOT chain through
    intermediate writes (B1 review hole: a chained ``index_copy_`` destination
    holds live data and must stay strict).

    Parameters
    ----------
    layer:
        The in-place destination-write op being validated.
    layers_to_perturb:
        Parent layer labels currently being perturbed.

    Returns
    -------
    bool
        Whether the perturbation is an exempt uninitialized-destination case.
    """

    if not layers_to_perturb:
        return False

    # Resolve the source op of each perturbed parent from this op's saved source
    # trace so we can read its producing func_name without a Trace handle.
    source_trace = getattr(layer, "source_trace", None)
    if source_trace is None:
        return False

    # The destination tensor of an in-place write is the first positional arg.
    # The perturbed parent must occupy that args[0] slot to be the destination.
    dest_labels: set[str] = set()
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    for key, parent_label in parent_arg_positions.get("args", {}).items():
        # args[0] is the destination; a nested key like (0, ...) is NOT the
        # whole-destination tensor (it would index INTO the value/index args).
        if key == 0:
            dest_labels.add(parent_label)

    for perturbed_label in layers_to_perturb:
        if perturbed_label not in dest_labels:
            return False
        try:
            perturbed_op = source_trace[perturbed_label]
        except Exception:
            return False
        if not _uninitialized_value_origin(perturbed_op, source_trace):
            return False
    return True


def _check_setitem_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt __setitem__ when the perturbed layer is a bool mask arg."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    args = layer.saved_args

    # Case 1: saved_args[1] is a bool tensor and perturbed layer matches it (mask arg)
    if (
        _perturbed_parent_is_arg_position(layer, layers_to_perturb, 1)
        and len(args) > 1
        and isinstance(args[1], torch.Tensor)
        and args[1].dtype == torch.bool
    ):
        return True

    # Case 2: saved_args[1] is a tuple whose first element is a bool tensor
    if (
        _perturbed_parent_is_arg_position(layer, layers_to_perturb, 1)
        and len(args) > 1
        and type(args[1]) == tuple
        and isinstance(args[1][0], torch.Tensor)
        and args[1][0].dtype == torch.bool
    ):
        return True

    # Case 3: perturbed layer is the destination (args[0]) and it's all-zeros/all-ones.
    # __setitem__ overwrites the destination, so perturbing a "blank slate" destination
    # (e.g. new_zeros used in BART position embeddings) has no effect.
    if (
        _perturbed_parent_is_arg_position(layer, layers_to_perturb, 0)
        and torch.equal(perturbed_tensor, args[0])
        and _check_if_arg_is_special_val(args[0])
    ):
        return True

    # Case 4: perturbed layer is the destination, but the indexed destination
    # slice is fully overwritten by the replacement value.
    if _perturbed_parent_is_arg_position(
        layer, layers_to_perturb, 0
    ) and _setitem_destination_slice_is_fully_overwritten(perturbed_tensor, args):
        return True

    return False


def _setitem_destination_slice_is_fully_overwritten(
    perturbed_tensor: torch.Tensor | None,
    args: tuple[Any, ...],
) -> bool:
    """Return whether a ``__setitem__`` call overwrites the perturbed destination slice.

    Parameters
    ----------
    perturbed_tensor:
        Tensor selected for perturbation.
    args:
        Saved ``__setitem__`` positional arguments.

    Returns
    -------
    bool
        True when the perturbed tensor is the destination, the replacement is a
        tensor, and ``destination[index]`` has exactly the replacement shape.
    """

    if len(args) < 3:
        return False
    destination, index, replacement = args[:3]
    if not isinstance(perturbed_tensor, torch.Tensor):
        return False
    if not isinstance(destination, torch.Tensor) or not isinstance(replacement, torch.Tensor):
        return False
    if not torch.equal(perturbed_tensor, destination):
        return False
    try:
        selected = destination[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    return tuple(selected.shape) == tuple(replacement.shape)


def _check_index_put_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt ``index_put``/``index_put_`` when the destination is fully overwritten.

    The exact analogue of :func:`_check_setitem_exempt` Case 4 for the
    ``index_put`` family. ``index_put(input, indices, values, accumulate=False)``
    overwrites ``input[indices]`` with ``values`` when ``accumulate`` is False, so
    the destination's prior value at those positions is provably irrelevant. This
    exemption fires ONLY when the perturbed parent IS the destination (``args[0]``)
    and the written positions are fully overwritten; it must NOT exempt a perturbed
    VALUE or INDEX parent (those genuinely influence the output), and it must NOT
    exempt the accumulating case (where the prior destination value IS added in).

    The destination is identified by ARG POSITION (``parent_arg_positions["args"][0]
    == layers_to_perturb[0]``), not solely by tensor-content equality: a value-parent
    whose contents happen to equal the destination would otherwise be falsely
    exempted by the downstream ``torch.equal`` check. Requiring the perturbed parent
    to occupy arg slot 0 mirrors how the ``__mod__`` divisor exemption keys off
    ``parent_arg_positions``.
    """
    if not _perturbed_parent_is_arg_position(layer, layers_to_perturb, 0):
        return False
    perturbed_tensor = self[layers_to_perturb[0]].out
    return _index_put_destination_is_fully_overwritten(perturbed_tensor, layer)


def _perturbed_parent_is_arg_position(
    layer: Op,
    layers_to_perturb: List[str],
    position: int,
) -> bool:
    """Return whether the perturbed parent occupies positional arg ``position``.

    Reads ``layer.parent_arg_positions["args"]`` (arg index -> parent layer label)
    and checks the perturbed label is the parent registered at ``position``. Used to
    confirm a perturbed parent is the DESTINATION (slot 0) by structure rather than
    by tensor-content equality, which can collide when a value-parent's contents
    match the destination.
    """

    arg_positions = (getattr(layer, "parent_arg_positions", None) or {}).get("args", {})
    return arg_positions.get(position) == layers_to_perturb[0]


def _perturbed_parent_arg_positions(layer: Op, layers_to_perturb: List[str]) -> set[int]:
    """Return positional arg slots occupied by the perturbed parent.

    Parameters
    ----------
    layer:
        Captured op whose parent-argument map is being inspected.
    layers_to_perturb:
        Single perturbed parent label supplied by validation.

    Returns
    -------
    set[int]
        Positional arg indices whose parent label is the perturbed layer. Missing
        or ambiguous metadata returns an empty set so callers fail closed.
    """

    if len(layers_to_perturb) != 1:
        return set()
    arg_positions = (getattr(layer, "parent_arg_positions", None) or {}).get("args", {})
    return {
        position
        for position, parent_label in arg_positions.items()
        if parent_label == layers_to_perturb[0]
    }


def _index_put_destination_is_fully_overwritten(
    perturbed_tensor: torch.Tensor | None,
    layer: Op,
) -> bool:
    """Return whether an ``index_put`` call overwrites the perturbed destination.

    Parameters
    ----------
    perturbed_tensor:
        Tensor selected for perturbation.
    layer:
        Captured ``index_put``/``index_put_`` op.

    Returns
    -------
    bool
        True only when the perturbed tensor is the destination (``args[0]``), the
        call is non-accumulating, and the indexed positions cover the ENTIRE
        destination and are exactly written by the broadcast ``values`` (so the
        destination's prior value is wholly irrelevant). Returns False for any
        perturbed VALUE/INDEX parent, for the accumulating case, and for a
        partial overwrite (where un-indexed destination elements still flow
        through).
    """

    args = layer.saved_args
    if not isinstance(perturbed_tensor, torch.Tensor):
        return False
    if args is None or len(args) < 3:
        return False
    destination, indices, values = args[0], args[1], args[2]
    if not isinstance(destination, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    # Narrow: the perturbed parent must be the DESTINATION, never the values/index.
    if not torch.equal(perturbed_tensor, destination):
        return False
    # accumulate=True adds the value to the prior destination, so the prior value
    # is NOT irrelevant -- never exempt that case. accumulate may arrive as a
    # positional arg (index 3) or as a keyword.
    accumulate = False
    if len(args) > 3:
        accumulate = bool(args[3])
    elif "accumulate" in (layer.saved_kwargs or {}):
        accumulate = bool(layer.saved_kwargs["accumulate"])
    if accumulate:
        return False
    # index_put indices are an advanced-indexing tuple/list of LongTensors.
    if isinstance(indices, list):
        index = tuple(indices)
    elif isinstance(indices, tuple):
        index = indices
    else:
        index = (indices,)
    try:
        selected = destination[index]
    except (IndexError, TypeError, RuntimeError):
        return False
    # The written slice must broadcast the replacement exactly (every selected
    # element is overwritten, none left at its prior value).
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(selected.shape), tuple(values.shape))
    except RuntimeError:
        return False
    if tuple(broadcast_shape) != tuple(selected.shape):
        return False
    # The exemption is only sound when the WHOLE destination is overwritten: any
    # un-indexed element keeps its prior value and so still influences the output
    # (the partial-overwrite false-exemption guard). Require the indexed region to
    # cover every destination element, with no duplicate indices inflating the
    # count -- duplicates would match the numel without covering everything.
    if not _index_put_indices_are_unique(index):
        return False
    return int(selected.numel()) == int(destination.numel())


def _index_put_indices_are_unique(index: tuple[Any, ...]) -> bool:
    """Return whether advanced ``index_put`` indices address distinct positions.

    Duplicate indices would let ``selected.numel()`` reach ``destination.numel()``
    without actually covering every destination element, so the full-overwrite
    coverage check would be fooled. This conservatively requires each integer
    index tensor to hold unique values; a non-integer (e.g. boolean mask) or any
    structure it cannot verify returns False so the exemption is withheld.
    """

    for component in index:
        if not isinstance(component, torch.Tensor):
            return False
        if component.dtype == torch.bool:
            # A bool mask selects each True position once -> inherently unique.
            continue
        flattened = component.reshape(-1)
        if int(flattened.numel()) != int(torch.unique(flattened).numel()):
            return False
    return True


def _check_lstm_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt lstm when the perturbed layer is a hidden/cell state arg."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    args = layer.saved_args

    if len(args) < 2 or perturbed_tensor is None:
        return False
    hidden_arg = args[1]
    if isinstance(hidden_arg, torch.Tensor):
        return torch.equal(perturbed_tensor, hidden_arg)
    if isinstance(hidden_arg, (list, tuple)):
        return any(
            isinstance(hidden_tensor, torch.Tensor) and torch.equal(perturbed_tensor, hidden_tensor)
            for hidden_tensor in hidden_arg
        )
    return False


def _check_interpolate_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt interpolate when the perturbed layer is the scale_factor arg."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    kwargs = layer.saved_kwargs
    args = layer.saved_args

    # Path 1: scale_factor as kwarg
    if (
        "scale_factor" in kwargs
        and kwargs["scale_factor"] is not None
        and torch.equal(perturbed_tensor, torch.tensor(kwargs["scale_factor"]))
    ):
        return True

    # Path 2: scale_factor as positional arg 2
    if (
        len(args) >= 3
        and isinstance(args[2], torch.Tensor)
        and torch.equal(perturbed_tensor, args[2])
    ):
        return True

    return False


def _get_scatter_destination_dim_index(
    layer: Op,
) -> tuple[torch.Tensor, int, torch.Tensor] | None:
    """Return scatter destination, dim, and index tensors when they are replayable.

    Parameters
    ----------
    layer:
        Scatter operation being validated.

    Returns
    -------
    tuple[torch.Tensor, int, torch.Tensor] | None
        Destination tensor, scatter dimension, and index tensor, or ``None``
        when the call shape is unsupported or uses reduce semantics.
    """

    args = layer.saved_args
    kwargs = layer.saved_kwargs
    if len(args) < 1 or not isinstance(args[0], torch.Tensor):
        return None
    if kwargs.get("reduce") is not None:
        return None
    if len(args) > 4 and args[4] is not None:
        return None

    dest = args[0]
    dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
    index = kwargs.get("index", args[2] if len(args) > 2 else None)
    if not isinstance(dim, int) or not isinstance(index, torch.Tensor):
        return None
    if dim < 0:
        dim = dest.ndim + dim
    if dim < 0 or dim >= dest.ndim:
        return None
    return dest, dim, index


def _scatter_index_fully_overwrites_dim(dest: torch.Tensor, dim: int, index: torch.Tensor) -> bool:
    """Return whether scatter index covers every destination slot along ``dim``.

    Parameters
    ----------
    dest:
        Scatter destination tensor.
    dim:
        Normalized scatter dimension.
    index:
        Scatter index tensor.

    Returns
    -------
    bool
        True when every slice orthogonal to ``dim`` contains each valid
        destination index, making the destination's prior values irrelevant.
    """

    if index.ndim != dest.ndim or index.shape[dim] < dest.shape[dim]:
        return False
    n_positions = dest.shape[dim]
    if n_positions == 0:
        return False
    moved = index.detach().cpu().movedim(dim, -1).reshape(-1, index.shape[dim])
    required = set(range(n_positions))
    for row in moved:
        row_values = {int(value) for value in row.tolist() if 0 <= int(value) < n_positions}
        if not required.issubset(row_values):
            return False
    return True


def _check_scatter_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt scatter destination perturbation when scatter fully overwrites it."""

    perturbed_tensor = self[layers_to_perturb[0]].out
    scatter_components = _get_scatter_destination_dim_index(layer)
    if scatter_components is None:
        return False
    dest, dim, index = scatter_components
    if not torch.equal(perturbed_tensor, dest):
        return False
    return _scatter_index_fully_overwrites_dim(dest, dim, index)


def _check_one_arg_where_index_exempt(layer: Op) -> bool:
    """Return whether ``layer`` is the one-arg ``torch.where(condition)`` index form.

    One-arg ``torch.where(condition)`` is torch's alias for
    ``nonzero(condition, as_tuple=True)``: its output is a DISCRETE integer INDEX set, not a
    value select, so small value-perturbation legitimately cannot change it -- the same
    category as the topk/sort/max/min index exemption.

    NARROW + tripwire-safe: the 3-arg VALUE-select ``where`` must NOT match. In particular the
    mixed form ``torch.where(cond, input=a, other=b)`` is a genuine value select that TorchLens
    records as ``len(saved_args) == 1`` with ``input``/``other`` in ``saved_kwargs`` (and an
    integer output dtype when the branches are integer), so arg-count and dtype alone are NOT
    sufficient. This asserts NO branch appears in args OR kwargs, matching only the true one-arg
    forms: positional ``where(cond)`` -> ``saved_args == (cond,)``, ``saved_kwargs == {}``; and
    keyword ``where(condition=cond)`` -> ``saved_args == ()``, ``saved_kwargs == {'condition'}``.

    DUAL-LAB reviewed 2026-06-30 (Claude + Codex independently converged on this exact
    predicate; the kwarg guard is load-bearing, the int-dtype gate mirrors the topk/sort
    precedent). The 3-arg ``_check_where_exempt`` (``len(saved_args) >= 3``) path is untouched
    and stays fully armed.

    Parameters
    ----------
    layer:
        Operation whose perturbation-insensitivity is being excused.

    Returns
    -------
    bool
        Whether the op is a genuine one-arg ``where`` index form.
    """

    if getattr(layer, "func_name", None) != "where":
        return False
    if layer.dtype not in (torch.int, torch.long, torch.int32, torch.int64):
        return False
    saved_args = layer.saved_args or ()
    saved_kwargs = getattr(layer, "saved_kwargs", None) or {}
    if len(saved_args) == 1 and saved_kwargs == {}:
        return True
    if len(saved_args) == 0 and set(saved_kwargs) == {"condition"}:
        return True
    return False


def _check_where_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt ``where`` parents only when saved value semantics prove irrelevance.

    The condition parent is irrelevant when the saved true/false branches are
    equal after broadcasting. A branch parent is irrelevant only when the SAVED
    condition selects the opposite branch at every element in the full broadcast
    output shape. Branch identity is taken from ``parent_arg_positions`` and the
    saved condition is the only selectedness source; any missing metadata,
    non-tensor saved arg, duplicate branch parent, or broadcast failure returns
    False.
    """

    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if not perturbed_positions:
        return False
    args = layer.saved_args
    if args is None:
        return False
    if len(args) < 3:
        return False
    condition, true_branch, false_branch = args[:3]
    if not (
        isinstance(condition, torch.Tensor)
        and isinstance(true_branch, torch.Tensor)
        and isinstance(false_branch, torch.Tensor)
    ):
        return False
    if 0 in perturbed_positions:
        if perturbed_positions != {0}:
            return False
        try:
            true_values, false_values = torch.broadcast_tensors(true_branch, false_branch)
        except RuntimeError:
            return False
        return bool(torch.equal(true_values, false_values))
    if not perturbed_positions.issubset({1, 2}):
        return False
    if len(perturbed_positions) != 1:
        return False
    selectedness = _where_saved_condition_selectedness(condition, true_branch, false_branch)
    if selectedness is None:
        return False
    selected_true, total_elements = selectedness
    if 1 in perturbed_positions:
        return selected_true == 0
    if 2 in perturbed_positions:
        return selected_true == total_elements
    return False


def _where_saved_condition_selectedness(
    condition: torch.Tensor,
    true_branch: torch.Tensor,
    false_branch: torch.Tensor,
) -> tuple[int, int] | None:
    """Return selected true-branch count over the full saved ``where`` shape.

    Parameters
    ----------
    condition:
        Saved ``where`` condition tensor.
    true_branch:
        Saved true-branch tensor.
    false_branch:
        Saved false-branch tensor.

    Returns
    -------
    tuple[int, int] | None
        ``(true_selected_count, total_elements)`` after broadcasting the saved
        condition to ``broadcast_shapes(condition, true_branch, false_branch)``.
        Returns ``None`` on empty output or any broadcast/count failure.
    """

    try:
        condition_bool = condition.to(dtype=torch.bool)
        output_shape = torch.broadcast_shapes(
            tuple(condition_bool.shape),
            tuple(true_branch.shape),
            tuple(false_branch.shape),
        )
        broadcast_condition = torch.broadcast_to(condition_bool, output_shape)
        total_elements = int(broadcast_condition.numel())
        if total_elements == 0:
            return None
        true_selected = int(torch.count_nonzero(broadcast_condition).item())
    except (TypeError, RuntimeError):
        return None
    return true_selected, total_elements


def _masked_fill_saved_mask_selectedness(
    mask: torch.Tensor,
    input_tensor: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int] | None:
    """Return selected fill-value count over the full saved ``masked_fill`` shape.

    Parameters
    ----------
    mask:
        Saved boolean mask argument.
    input_tensor:
        Saved input/destination tensor.
    value:
        Saved scalar tensor fill value.

    Returns
    -------
    tuple[int, int] | None
        ``(fill_selected_count, total_elements)`` after broadcasting the saved
        mask over the full input/value output shape. Returns ``None`` if the
        saved data cannot prove exact selectedness.
    """

    try:
        mask_bool = mask.to(dtype=torch.bool)
        output_shape = torch.broadcast_shapes(
            tuple(mask_bool.shape),
            tuple(input_tensor.shape),
            tuple(value.shape),
        )
        broadcast_mask = torch.broadcast_to(mask_bool, output_shape)
        total_elements = int(broadcast_mask.numel())
        if total_elements == 0:
            return None
        fill_selected = int(torch.count_nonzero(broadcast_mask).item())
    except (TypeError, RuntimeError):
        return None
    return fill_selected, total_elements


def _check_masked_fill_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt ``masked_fill`` value parents only when entirely unselected.

    ``masked_fill(input, mask, value)`` is equivalent to
    ``where(mask, value, input)``. The input parent is irrelevant only when the
    saved mask is true at every output element; a tensor fill-value parent is
    irrelevant only when the saved mask is false at every output element. The
    mask itself remains handled by the structural-position registry.
    """

    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if not perturbed_positions:
        return False
    args = layer.saved_args
    if args is None or len(args) < 3:
        return False
    input_tensor, mask, value = args[:3]
    if not (
        isinstance(input_tensor, torch.Tensor)
        and isinstance(mask, torch.Tensor)
        and isinstance(value, torch.Tensor)
    ):
        return False
    if not perturbed_positions.issubset({0, 2}):
        return False
    if len(perturbed_positions) != 1:
        return False
    selectedness = _masked_fill_saved_mask_selectedness(mask, input_tensor, value)
    if selectedness is None:
        return False
    fill_selected, total_elements = selectedness
    if 0 in perturbed_positions:
        return fill_selected == total_elements
    if 2 in perturbed_positions:
        return fill_selected == 0
    return False


# ---------------------------------------------------------------------------
# Registry 4: Custom exemption checks keyed by func name.
# ---------------------------------------------------------------------------
CUSTOM_EXEMPTION_CHECKS: Dict[str, Callable[["Trace", Op, List[str]], bool]] = {
    "__getitem__": _check_getitem_exempt,
    "__setitem__": _check_setitem_exempt,
    "index_put": _check_index_put_exempt,
    "index_put_": _check_index_put_exempt,
    "lstm": _check_lstm_exempt,
    "interpolate": _check_interpolate_exempt,
    "scatter": _check_scatter_exempt,
    "scatter_": _check_scatter_exempt,
    "where": _check_where_exempt,
    "maskedfill": _check_masked_fill_exempt,
    "masked_fill": _check_masked_fill_exempt,
    "masked_fill_": _check_masked_fill_exempt,
}


# ---------------------------------------------------------------------------
# Structural position helper (used by core.py)
# ---------------------------------------------------------------------------


def perturbed_layer_at_structural_position(
    self: "Trace",
    layer: Op,
    layers_to_perturb: List[str],
    exempt_positions: Set[int],
) -> bool:
    """Check if the perturbed layer occupies a structural arg position.

    The decision is based on ``parent_arg_positions`` identity, not tensor value
    equality. Identical tensor values from another parent must not be enough to
    classify the perturbed parent as structural.
    """
    del self
    if len(layers_to_perturb) != 1:
        return False
    perturbed_label = layers_to_perturb[0]
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    for pos in exempt_positions:
        if parent_arg_positions.get("args", {}).get(pos) == perturbed_label:
            return True
    return False


def _binary_extrema_nonperturbed_arg_dominates(
    func_name: str,
    args: tuple[Any, ...],
    layer: Op,
    layers_to_perturb: List[str],
) -> bool:
    """Return whether a binary extrema output ignores the perturbed operand.

    Parameters
    ----------
    func_name:
        Captured extrema function name.
    args:
        Saved positional arguments.
    layer:
        Captured op being checked.
    layers_to_perturb:
        Parent labels currently being perturbed.

    Returns
    -------
    bool
        True when the non-perturbed tensor dominates the perturbed tensor at
        every output element, proving the perturbed operand cannot affect the
        selected extrema values.
    """

    if len(layers_to_perturb) != 1 or len(args) < 2:
        return False
    if not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
        return False
    perturbed_positions = _perturbed_parent_arg_positions(layer, layers_to_perturb)
    if perturbed_positions == {0}:
        perturbed, other = args[0], args[1]
    elif perturbed_positions == {1}:
        perturbed, other = args[1], args[0]
    else:
        return False
    try:
        if func_name in ("max", "maximum"):
            return bool(torch.all(other >= perturbed).item())
        if func_name in ("min", "minimum"):
            return bool(torch.all(other <= perturbed).item())
    except RuntimeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Posthoc perturbation check — excuses failures after execution.
# These handle genuinely dynamic/value-dependent cases that can't be
# determined before running the function.
# ---------------------------------------------------------------------------


def posthoc_perturb_check(
    self: "Trace",
    layer_to_validate_parents_for: Op,
    layers_to_perturb: List[str],
    verbose: bool = False,
) -> bool:
    """Post-hoc exemption check: called when perturbation did NOT change the output.

    This function runs AFTER execution, handling dynamic cases that cannot be
    determined pre-execution.  It checks a cascade of valid excuses:

    1. **Bool output** -- discrete output, perturbation may not flip it.
    2. **Discrete index outputs** (topk, sort) -- indices are order-dependent,
       not value-dependent.
    3. **Type casting** (``to()``) -- value irrelevant when casting type.
    4. **Full overwrite** (__setitem__ with same-shape replacement).
    5. **Small tensor coincidence** (__getitem__/unbind with numel < 20).
    6. **Redundant safety net** for *_like/meshgrid/broadcast_tensors.
    7. **All-inf/all-NaN output** -- extreme values.
    8. **Special-value arg loop** -- if ANY non-perturbed arg is all-zeros or
       all-ones, that single arg can explain output invariance (e.g.,
       multiplication by zero annihilates the other operand).  This correctly
       returns True on the first such special arg found.

    Returns True if there's a valid excuse (validation ops), False otherwise
    (validation fails with a printed message).
    """
    func_name = layer_to_validate_parents_for.func_name
    layer_label = layer_to_validate_parents_for.layer_label
    args = layer_to_validate_parents_for.saved_args

    # Bool output — discrete, perturbation may not change it
    if layer_to_validate_parents_for.dtype == torch.bool:
        return True

    # topk/sort/max/min indices — discrete output insensitive to value perturbation.
    # max(tensor, dim) and min(tensor, dim) return (values, indices); the indices
    # output is integer and may not change when values are perturbed.
    if func_name in (
        "topk",
        "sort",
        "max",
        "min",
    ) and layer_to_validate_parents_for.dtype in (
        torch.int,
        torch.long,
        torch.int32,
        torch.int64,
    ):
        return True

    # One-arg ``torch.where(condition)`` == ``nonzero(condition, as_tuple=True)``: a discrete
    # INDEX output, same category as the topk/sort/max/min index exemption above.
    if _check_one_arg_where_index_exempt(layer_to_validate_parents_for):
        return True

    # to() with tensor arg — type casting
    if func_name == "to" and len(args) > 1 and isinstance(args[1], torch.Tensor):
        return True

    # __setitem__ same-shape replacement — full overwrite
    if (
        func_name == "__setitem__"
        and _perturbed_parent_is_arg_position(layer_to_validate_parents_for, layers_to_perturb, 0)
        and len(args) > 2
        and isinstance(args[0], torch.Tensor)
        and isinstance(args[2], torch.Tensor)
        and args[0].shape == args[2].shape
    ):
        return True

    # __setitem__ non-tensor value — scalar set
    if (
        func_name == "__setitem__"
        and _perturbed_parent_is_arg_position(layer_to_validate_parents_for, layers_to_perturb, 0)
        and len(args) > 2
        and not isinstance(args[2], torch.Tensor)
    ):
        return True

    # __setitem__ into an UNINITIALIZED-MEMORY destination — perturbing
    # uninitialized memory is meaningless by construction.
    #
    # Idiom (RoPE/normalizing-flow interleave):
    #   out = torch.empty_like(x); out[..., 0::2] = a   # this op
    #   out[..., 1::2] = b                              # sibling __setitem__
    # The destination `out` (args[0]) is uninitialized memory from
    # empty/empty_like/new_empty. A strided/partial __setitem__ writes only some
    # positions; the UNWRITTEN positions retain garbage that is overwritten by a
    # sibling write and never meaningfully consumed. Perturbing that garbage is
    # not a real data dependency, so the perturbation check's sensitivity
    # contract does not apply.
    #
    # NARROW + tripwire-safe: exempt ONLY when the PERTURBED parent is ITSELF a
    # direct uninitialized-memory allocation (empty/empty_like/new_empty) AND it
    # occupies the destination slot. A REAL data tensor feeding the destination
    # -- INCLUDING a prior in-place write (index_copy_/__setitem__) whose buffer
    # was allocated by empty_like but which now holds real written data -- stays
    # strict, so a genuine dropped-dependency capture bug still fails. The
    # exemption deliberately does NOT chain through intermediate in-place writes:
    # the B1 adversarial review found that following the destination chain back to
    # the allocation masked a real perturbation failure on the second write of an
    # empty_like + chained index_copy_ idiom (the second write consumes the first
    # write's live data). (empty_like et al. are already nondeterministic-skipped
    # for their OWN replay; this extends that same correctness to their direct
    # consumption as the destination of one in-place index/scatter write.) Covers
    # the RoPE / normalizing-flow / torch_geometric "partition-write into an
    # uninitialized buffer" idiom where the perturbed destination is the bare
    # allocation.
    if (
        func_name in INPLACE_DESTINATION_WRITE_FUNCS
        and _perturbed_parent_is_uninitialized_setitem_dest(
            layer_to_validate_parents_for, layers_to_perturb
        )
    ):
        return True

    # __getitem__/unbind numel < 20 — small tensor coincidence
    if func_name in ["__getitem__", "unbind"] and layer_to_validate_parents_for.out.numel() < 20:
        return True

    # Redundant safety net: *_like ops, meshgrid, broadcast_tensors
    # (should be caught by SKIP_PERTURBATION_ENTIRELY / SKIP_VALIDATION_ENTIRELY,
    # but kept as belt-and-suspenders)
    if func_name in ["meshgrid", "broadcast_tensors"]:
        return True
    if func_name in [
        "full_like",
        "zeros_like",
        "ones_like",
        "empty_like",
        "rand_like",
        "randn_like",
    ]:
        return True

    # __getitem__ tensor index with < 20 unique values — coincidence
    if (
        func_name == "__getitem__"
        and isinstance(args[1], torch.Tensor)
        and len(args[1].unique()) < 20  # type: ignore[no-untyped-call]
    ):
        return True

    output_tensor = layer_to_validate_parents_for.out

    # max/min/maximum/minimum with multiple args — binary extrema are insensitive
    # only when the non-perturbed operand provably dominates elementwise.
    if func_name in ("max", "min", "maximum", "minimum") and len(args) > 1:
        if _binary_extrema_nonperturbed_arg_dominates(
            func_name, args, layer_to_validate_parents_for, layers_to_perturb
        ):
            return True

    # remainder/fmod divisor is locally irrelevant when the dividend is already
    # the result (e.g., 0 <= dividend < divisor elementwise).
    if func_name in ("remainder", "fmod", "__mod__") and len(args) > 1:
        dividend, divisor = args[:2]
        if isinstance(dividend, torch.Tensor) and isinstance(divisor, torch.Tensor):
            arg_positions = layer_to_validate_parents_for.parent_arg_positions.get("args", {})
            perturbed_label = layers_to_perturb[0]
            perturbed_is_divisor = arg_positions.get(1) == perturbed_label
            if perturbed_is_divisor and torch.equal(output_tensor, dividend):
                return True

    # max non-floating-point — discrete
    if func_name == "max" and not torch.is_floating_point(args[0]):
        return True

    # bernoulli with scalar p kwarg — self tensor is just a shape template,
    # output is determined entirely by p and RNG state, not self's values.
    if func_name == "bernoulli" and "p" in layer_to_validate_parents_for.saved_kwargs:
        return True

    # Constant output — function is structurally constant-valued
    # (e.g., softmax on a dimension with size 1 always produces all-ones).
    if output_tensor.numel() > 0:
        flat_output = output_tensor.reshape(-1)
        if torch.equal(flat_output, flat_output[0].expand_as(flat_output)):
            return True

    # All-inf / all-NaN output — extreme values
    num_inf = torch.isinf(output_tensor.abs()).int().sum()
    num_nan = torch.isnan(output_tensor.abs()).int().sum()
    if (num_inf == output_tensor.numel()) or (num_nan == output_tensor.numel()):
        return True

    # Special-value arg loop — all-zeros/all-ones in other args
    arg_type_dict = {
        "args": (enumerate, "saved_args"),
        "kwargs": (lambda x: x.items(), "saved_kwargs"),
    }

    for arg_type in ["args", "kwargs"]:
        iterfunc, fieldname = arg_type_dict[arg_type]
        for key, val in iterfunc(getattr(layer_to_validate_parents_for, fieldname)):  # type: ignore[operator]
            # Skip if it's the argument being perturbed
            if (
                key in layer_to_validate_parents_for.parent_arg_positions[arg_type]
                and layer_to_validate_parents_for.parent_arg_positions[arg_type][key]
                in layers_to_perturb
            ):
                continue
            if _check_if_arg_is_special_val(val):
                if verbose:
                    print(
                        f"Activations for layer {layer_label} do not change when "
                        f"values for {layers_to_perturb} are changed (out of parent "
                        f"layers {layer_to_validate_parents_for.parents}), but "
                        f"{arg_type[:-1]} {key} is all zeros or all-ones, so validation "
                        f"still succeeds..."
                    )
                return True

    # Check non-perturbed parent tensors directly — catches nested args
    # (e.g. einsum receives operands as a tuple, so the special-value loop
    # above can't see inside).
    for parent_label in layer_to_validate_parents_for.parents:
        if parent_label in layers_to_perturb:
            continue
        parent_tensor = self[parent_label].out
        if parent_tensor is not None and _check_if_arg_is_special_val(parent_tensor):
            if verbose:
                print(
                    f"Activations for layer {layer_label} do not change when "
                    f"values for {layers_to_perturb} are changed, but "
                    f"non-perturbed parent {parent_label} has special values "
                    f"(all-zeros or all-ones), so validation still succeeds..."
                )
            return True

    print(
        f"Activations for layer {layer_label} do not change when "
        f"values for {layers_to_perturb} are changed (out of parent "
        f"layers {layer_to_validate_parents_for.parents}), and the other "
        f'arguments are not "special" (all-ones or all-zeros) tensors.'
    )
    return False


def _check_if_arg_is_special_val(val: Union[torch.Tensor, Any]) -> bool:
    """Check if a value is all-zeros, all-ones, or empty (numel==0).

    These "special" values can make perturbation of OTHER args irrelevant:
    - All-zeros: multiplication by zero annihilates the other operand.
    - All-ones: identity element for multiplication, no-op for many ops.
    - Empty: no elements to be affected.

    Non-tensor values (scalars, etc.) are converted to tensors for the check.
    Returns False for values that can't be converted (strings, None, etc.).
    """
    if not isinstance(val, torch.Tensor):
        try:
            val = torch.tensor(val)
        except (TypeError, ValueError, RuntimeError):
            return False
    if torch.all(torch.eq(val, 0)) or torch.all(torch.eq(val, 1)) or (val.numel() == 0):
        return True
    return False

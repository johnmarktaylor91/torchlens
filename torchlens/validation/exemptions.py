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
# TODO: Audit PyTorch ops more exhaustively for additional exemptions.
# Current registries cover all cases encountered in the test suite as of 2026-03.
# When adding new model tests, if perturbation fails for a new function,
# add the exemption here (not in core.py).

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


def _check_setitem_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt __setitem__ when the perturbed layer is a bool mask arg."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    args = layer.saved_args

    # Case 1: saved_args[1] is a bool tensor and perturbed layer matches it (mask arg)
    if (
        isinstance(args[1], torch.Tensor)
        and args[1].dtype == torch.bool
        and torch.equal(perturbed_tensor, args[1])
    ):
        return True

    # Case 2: saved_args[1] is a tuple whose first element is a bool tensor
    if (
        type(args[1]) == tuple
        and isinstance(args[1][0], torch.Tensor)
        and args[1][0].dtype == torch.bool
        and torch.equal(perturbed_tensor, args[1][0])
    ):
        return True

    # Case 3: perturbed layer is the destination (args[0]) and it's all-zeros/all-ones.
    # __setitem__ overwrites the destination, so perturbing a "blank slate" destination
    # (e.g. new_zeros used in BART position embeddings) has no effect.
    if torch.equal(perturbed_tensor, args[0]) and _check_if_arg_is_special_val(args[0]):
        return True

    # Case 4: perturbed layer is the destination, but the indexed destination
    # slice is fully overwritten by the replacement value.
    if _setitem_destination_slice_is_fully_overwritten(perturbed_tensor, args):
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
    """
    perturbed_tensor = self[layers_to_perturb[0]].out
    return _index_put_destination_is_fully_overwritten(perturbed_tensor, layer)


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


def _get_scatter_destination_dim_index(layer: Op) -> tuple[torch.Tensor, int, torch.Tensor] | None:
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


def _check_where_exempt(self: "Trace", layer: Op, layers_to_perturb: List[str]) -> bool:
    """Exempt where when the perturbed condition selects between equal branches."""
    perturbed_tensor = self[layers_to_perturb[0]].out
    args = layer.saved_args
    if len(args) < 3:
        return False
    condition, true_branch, false_branch = args[:3]
    if not (
        isinstance(condition, torch.Tensor)
        and isinstance(true_branch, torch.Tensor)
        and isinstance(false_branch, torch.Tensor)
    ):
        return False
    if not torch.equal(perturbed_tensor, condition):
        return False
    true_values, false_values = torch.broadcast_tensors(true_branch, false_branch)
    return bool(torch.equal(true_values, false_values))


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
    """Check if the perturbed layer's tensor occupies a structural arg position.

    Compares the perturbed layer's out to saved_args at each
    exempt position using ``torch.equal``.  If there is a match, the perturbed
    layer controls structure (e.g., indices, masks) rather than values, so
    perturbation is meaningless.

    Note: uses ``torch.equal`` (exact elementwise match) which could
    theoretically false-positive if two different parents have identical
    tensor values, but this is exceedingly rare in practice.
    """
    perturbed_tensor = self[layers_to_perturb[0]].out
    for pos in exempt_positions:
        if pos >= len(layer.saved_args):
            continue
        arg_val = layer.saved_args[pos]
        if not isinstance(arg_val, torch.Tensor):
            continue
        if torch.equal(perturbed_tensor, arg_val):
            return True
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

    # to() with tensor arg — type casting
    if func_name == "to" and len(args) > 1 and isinstance(args[1], torch.Tensor):
        return True

    # __setitem__ same-shape replacement — full overwrite
    if (
        func_name == "__setitem__"
        and isinstance(args[2], torch.Tensor)
        and args[0].shape == args[2].shape
    ):
        return True

    # __setitem__ non-tensor value — scalar set
    if func_name == "__setitem__" and not isinstance(args[2], torch.Tensor):
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

    # max/min/maximum/minimum with multiple args — binary max/min is insensitive
    # to perturbation when one arg dominates (e.g., extreme negative vs normal value).
    if func_name in ("max", "min", "maximum", "minimum") and len(args) > 1:
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

    # Float32 precision exemption: when a non-perturbed parent's magnitude dwarfs
    # the perturbed parent's, the perturbation is swallowed by float32 arithmetic.
    # E.g., add(~51, ~659344): perturbing ~51 by ±5 doesn't change the sum at
    # float32 precision because 659344+51 and 659344+56 round to the same value.
    for parent_label in layer_to_validate_parents_for.parents:
        if parent_label in layers_to_perturb:
            continue
        parent_tensor = self[parent_label].out
        if parent_tensor is None:
            continue
        other_mag = parent_tensor.float().abs().max().item()
        for perturbed_label in layers_to_perturb:
            perturbed_tensor = self[perturbed_label].out
            if perturbed_tensor is None:
                continue
            perturbed_mag = perturbed_tensor.float().abs().max().item()
            if perturbed_mag > 0 and other_mag / perturbed_mag > 100:
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

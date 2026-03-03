# TODO: Audit PyTorch ops more exhaustively for additional exemptions.
# Current registries cover all cases encountered in the test suite as of 2026-03.
# When adding new model tests, if perturbation fails for a new function,
# add the exemption here (not in core.py).

from typing import Any, Dict, List, Set, Union

import torch

from ..data_classes.tensor_log import TensorLog
from ..helper_funcs import tensor_all_nan


# ---------------------------------------------------------------------------
# Registry 1: Skip ALL validation (forward replay + perturbation).
# These funcs produce nondeterministic output (e.g. uninitialized memory),
# so even forward replay would fail.
# ---------------------------------------------------------------------------
SKIP_VALIDATION_ENTIRELY: Set[str] = {
    "empty_like",
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
}

# ---------------------------------------------------------------------------
# Registry 3: Specific arg positions that are structural (not value-sensitive).
# When the perturbed layer's tensor matches creation_args[pos], skip perturbation.
# ---------------------------------------------------------------------------
STRUCTURAL_ARG_POSITIONS: Dict[str, Set[int]] = {
    "cross_entropy": {1},  # target labels (LongTensor)
    "index_select": {2},  # index tensor
    "scatter_": {2},  # index tensor
    "masked_fill_": {1},  # mask tensor
    "_pad_packed_sequence": {1},  # lengths tensor
    "type_as": {1},  # type template tensor (value irrelevant)
}


# ---------------------------------------------------------------------------
# Custom exemption check functions
# Signature: callable(self, layer, layers_to_perturb) -> bool
#   self = ModelLog instance
#   layer = TensorLog being validated
#   layers_to_perturb = list of layer labels being perturbed
# ---------------------------------------------------------------------------


def _check_getitem_exempt(self, layer: TensorLog, layers_to_perturb: List[str]) -> bool:
    """Exempt __getitem__ when the perturbed layer is a structural arg (index tensor,
    or any non-data arg)."""
    perturbed_tensor = self[layers_to_perturb[0]].tensor_contents
    args = layer.creation_args

    # Case 1: perturbed layer IS the tensor index — tensor indexing is structural
    if isinstance(args[1], torch.Tensor) and torch.equal(perturbed_tensor, args[1]):
        return True

    # Case 2: perturbed layer is NOT the data tensor — must be a structural arg
    # (slice, int index logged as tensor, etc.)
    if not torch.equal(perturbed_tensor, args[0]):
        return True

    return False


def _check_setitem_exempt(self, layer: TensorLog, layers_to_perturb: List[str]) -> bool:
    """Exempt __setitem__ when the perturbed layer is a bool mask arg."""
    perturbed_tensor = self[layers_to_perturb[0]].tensor_contents
    args = layer.creation_args

    # Case 1: creation_args[1] is a bool tensor and perturbed layer matches it (mask arg)
    if (
        isinstance(args[1], torch.Tensor)
        and args[1].dtype == torch.bool
        and torch.equal(perturbed_tensor, args[1])
    ):
        return True

    # Case 2: creation_args[1] is a tuple whose first element is a bool tensor
    if (
        type(args[1]) == tuple
        and isinstance(args[1][0], torch.Tensor)
        and args[1][0].dtype == torch.bool
        and torch.equal(perturbed_tensor, args[1][0])
    ):
        return True

    return False


def _check_lstm_exempt(self, layer: TensorLog, layers_to_perturb: List[str]) -> bool:
    """Exempt lstm when the perturbed layer is a hidden/cell state arg."""
    perturbed_tensor = self[layers_to_perturb[0]].tensor_contents
    args = layer.creation_args

    return (
        torch.equal(perturbed_tensor, args[1][0])  # hidden state h
        or torch.equal(perturbed_tensor, args[1][1])  # hidden state c
        or torch.equal(perturbed_tensor, args[2][0])  # cell state h
        or torch.equal(perturbed_tensor, args[2][1])  # cell state c
        or (
            isinstance(args[1], torch.Tensor)  # flat hidden state
            and torch.equal(perturbed_tensor, args[1])
        )
    )


def _check_interpolate_exempt(self, layer: TensorLog, layers_to_perturb: List[str]) -> bool:
    """Exempt interpolate when the perturbed layer is the scale_factor arg."""
    perturbed_tensor = self[layers_to_perturb[0]].tensor_contents
    kwargs = layer.creation_kwargs
    args = layer.creation_args

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


# ---------------------------------------------------------------------------
# Registry 4: Custom exemption checks keyed by func name.
# ---------------------------------------------------------------------------
CUSTOM_EXEMPTION_CHECKS: Dict[str, callable] = {
    "__getitem__": _check_getitem_exempt,
    "__setitem__": _check_setitem_exempt,
    "lstm": _check_lstm_exempt,
    "interpolate": _check_interpolate_exempt,
}


# ---------------------------------------------------------------------------
# Structural position helper (used by core.py)
# ---------------------------------------------------------------------------


def perturbed_layer_at_structural_position(
    self,
    layer: TensorLog,
    layers_to_perturb: List[str],
    exempt_positions: Set[int],
) -> bool:
    """Check if the perturbed layer's tensor matches any exempt arg position."""
    perturbed_tensor = self[layers_to_perturb[0]].tensor_contents
    for pos in exempt_positions:
        if pos >= len(layer.creation_args):
            continue
        arg_val = layer.creation_args[pos]
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
    self,
    layer_to_validate_parents_for: TensorLog,
    layers_to_perturb: List[str],
    verbose: bool = False,
) -> bool:
    """If perturbation didn't change the output, check whether there's a valid
    excuse (bool output, type casting, special-value args, etc.).

    Returns True if there's an excuse, False otherwise.
    """
    func_name = layer_to_validate_parents_for.func_applied_name
    layer_label = layer_to_validate_parents_for.layer_label
    args = layer_to_validate_parents_for.creation_args

    # Bool output — discrete, perturbation may not change it
    if layer_to_validate_parents_for.tensor_dtype == torch.bool:
        return True

    # topk/sort indices — discrete output insensitive to value perturbation
    if func_name in ("topk", "sort") and layer_to_validate_parents_for.tensor_dtype in (
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
    if (
        func_name in ["__getitem__", "unbind"]
        and layer_to_validate_parents_for.tensor_contents.numel() < 20
    ):
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
        and len(args[1].unique()) < 20
    ):
        return True

    # max/min with multiple args — binary max/min
    if func_name in ("max", "min") and len(args) > 1:
        return True

    # max non-floating-point — discrete
    if func_name == "max" and not torch.is_floating_point(args[0]):
        return True

    # All-inf / all-NaN output — extreme values
    output_tensor = layer_to_validate_parents_for.tensor_contents
    num_inf = torch.isinf(output_tensor.abs()).int().sum()
    num_nan = torch.isnan(output_tensor.abs()).int().sum()
    if (num_inf == output_tensor.numel()) or (num_nan == output_tensor.numel()):
        return True

    # Special-value arg loop — all-zeros/all-ones in other args
    arg_type_dict = {
        "args": (enumerate, "creation_args"),
        "kwargs": (lambda x: x.items(), "creation_kwargs"),
    }

    for arg_type in ["args", "kwargs"]:
        iterfunc, fieldname = arg_type_dict[arg_type]
        for key, val in iterfunc(getattr(layer_to_validate_parents_for, fieldname)):
            # Skip if it's the argument being perturbed
            if (
                key in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type]
                and layer_to_validate_parents_for.parent_layer_arg_locs[arg_type][key]
                in layers_to_perturb
            ):
                continue
            if _check_if_arg_is_special_val(val):
                if verbose:
                    print(
                        f"Activations for layer {layer_label} do not change when "
                        f"values for {layers_to_perturb} are changed (out of parent "
                        f"layers {layer_to_validate_parents_for.parent_layers}), but "
                        f"{arg_type[:-1]} {key} is all zeros or all-ones, so validation "
                        f"still succeeds..."
                    )
                return True

    print(
        f"Activations for layer {layer_label} do not change when "
        f"values for {layers_to_perturb} are changed (out of parent "
        f"layers {layer_to_validate_parents_for.parent_layers}), and the other "
        f'arguments are not "special" (all-ones or all-zeros) tensors.'
    )
    return False


def _check_if_arg_is_special_val(val: Union[torch.Tensor, Any]) -> bool:
    """Check if a value is all zeros, all ones, or empty."""
    if not isinstance(val, torch.Tensor):
        try:
            val = torch.tensor(val)
        except (TypeError, ValueError, RuntimeError):
            return False
    if torch.all(torch.eq(val, 0)) or torch.all(torch.eq(val, 1)) or (val.numel() == 0):
        return True
    return False

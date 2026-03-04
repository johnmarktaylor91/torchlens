"""Core validation logic for verifying saved activations.

Orchestrates forward replay and perturbation checks, delegating exemption
decisions to the registries and helpers in exemptions.py.
"""

from collections import defaultdict, deque
from typing import Optional, Any, Dict, List, Set, TYPE_CHECKING, Union

import torch

from ..data_classes.layer_pass_log import LayerPassLog

if TYPE_CHECKING:
    pass

from ..utils.rng import AutocastRestore, log_current_rng_states, set_rng_from_saved_states
from ..utils.collections import assign_to_sequence_or_dict
from ..utils.tensor_utils import tensor_nanequal, tensor_all_nan
from .exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    SKIP_PERTURBATION_ENTIRELY,
    STRUCTURAL_ARG_POSITIONS,
    CUSTOM_EXEMPTION_CHECKS,
    perturbed_layer_at_structural_position,
    posthoc_perturb_check,
)

MAX_PERTURB_ATTEMPTS = 100


def validate_saved_activations(
    self,
    ground_truth_output_tensors: List[torch.Tensor],
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Starting from outputs and internally terminated tensors, checks whether computing their values from the saved
    values of their input tensors yields their actually saved values, and whether computing their values from
    their parent tensors yields their saved values. Optionally validates all metadata invariants.

    Args:
        ground_truth_output_tensors: Ground truth output tensors from a fresh forward pass.
        verbose: Whether to print warning messages on validation failure.
        validate_metadata: Whether to run metadata invariant checks (default True).

    Returns:
        True if it passes the tests, False otherwise.
    """
    # First check that the ground truth output tensors are accurate:
    for i, output_layer_label in enumerate(self.output_layers):
        output_layer = self[output_layer_label]
        if not tensor_nanequal(
            output_layer.tensor_contents,
            ground_truth_output_tensors[i],
            allow_tolerance=False,
        ):
            print(
                f"The {i}th output layer, {output_layer_label}, does not match the ground truth output tensor."
            )
            return False

    # Validate the parents of each validated layer.
    validated_child_edges_for_each_layer: Dict[str, Set[str]] = defaultdict(set)
    validated_layers = set(self.output_layers + self.internally_terminated_layers)
    layers_to_validate_parents_for = deque(validated_layers)

    while len(layers_to_validate_parents_for) > 0:
        layer_to_validate_parents_for = layers_to_validate_parents_for.popleft()
        parent_layers_valid = validate_parents_of_saved_layer(
            self,
            layer_to_validate_parents_for,
            validated_layers,
            validated_child_edges_for_each_layer,
            layers_to_validate_parents_for,  # type: ignore[arg-type]
            verbose,
        )
        if not parent_layers_valid:
            return False

    if len(validated_layers) < len(self.layer_labels):
        unreached = set(self.layer_labels) - validated_layers
        print(
            f"All saved activations were accurate, but some layers were not reached (check that "
            f"child args logged accurately): {unreached}"
        )
        return False

    # Metadata invariant checks (after activation validation passes)
    if validate_metadata:
        from .invariants import check_metadata_invariants

        check_metadata_invariants(self)

    return True


def validate_parents_of_saved_layer(
    self,
    layer_to_validate_parents_for_label: str,
    validated_layers: Set[str],
    validated_child_edges_for_each_layer: Dict[str, Set[str]],
    layers_to_validate_parents_for: List[str],
    verbose: bool = False,
) -> bool:
    """Given a layer, checks that 1) all parent tensors appear properly in the saved arguments for that layer,
    2) that executing the function for that layer with the saved parent layer activations yields the
    ground truth activation values for that layer, and 3) that plugging in "perturbed" values for each
    child layer yields values different from the saved activations for that layer.

    Args:
        layer_to_validate_parents_for_label: Label of the layer whose parent edges are being validated.
        validated_layers: Set of layer labels already validated; mutated in-place to add newly validated layers.
        validated_child_edges_for_each_layer: Dict mapping each layer label to the set of its child edges
            that have been validated so far; mutated in-place as child edges are confirmed.
        layers_to_validate_parents_for: Work queue of layer labels still needing parent validation;
            mutated in-place to append newly discovered layers.
        verbose: Whether to print warning messages on validation failure.
    """
    layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]

    # Check that the arguments are logged correctly:
    if not _check_layer_arguments_logged_correctly(self, layer_to_validate_parents_for_label):
        print(
            f"Parent arguments for layer {layer_to_validate_parents_for_label} are not logged properly; "
            f"either a parent wasn't logged as an argument, or was logged an extra time"
        )
        return False

    # Check that executing the function based on the actual saved values of the parents yields the saved
    # values of the layer itself:

    if not _check_whether_func_on_saved_parents_yields_saved_tensor(
        self, layer_to_validate_parents_for_label, perturb=False
    ):
        return False

    # Check that executing the layer's function on the wrong version of the saved parent tensors
    # yields the wrong tensors, when each saved tensor is perturbed in turn:

    func_name = layer_to_validate_parents_for.func_applied_name
    for perturb_layer in layer_to_validate_parents_for.parent_layers:
        if func_name in SKIP_PERTURBATION_ENTIRELY:
            continue
        if not _check_whether_func_on_saved_parents_yields_saved_tensor(
            self,
            layer_to_validate_parents_for_label,
            perturb=True,
            layers_to_perturb=[perturb_layer],
            verbose=verbose,
        ):
            return False

    # Log that each parent layer has been validated for this source layer.

    for parent_layer_label in layer_to_validate_parents_for.parent_layers:
        parent_layer = self[parent_layer_label]
        validated_child_edges_for_each_layer[parent_layer_label].add(
            layer_to_validate_parents_for_label
        )
        if validated_child_edges_for_each_layer[parent_layer_label] == set(
            parent_layer.child_layers
        ):
            validated_layers.add(parent_layer_label)
            if (not parent_layer.is_input_layer) and not (
                parent_layer.is_buffer_layer and (parent_layer.buffer_parent is None)
            ):
                layers_to_validate_parents_for.append(parent_layer_label)

    return True


def _check_layer_arguments_logged_correctly(self, target_layer_label: str) -> bool:
    """Check whether the activations of the parent layers match the saved arguments of
    the target layer, and that the argument locations have been logged correctly.

    Args:
        target_layer_label: Layer to check

    Returns:
        True if arguments logged accurately, False otherwise
    """
    target_layer = self[target_layer_label]

    # Make sure that all parent layers appear in at least one argument and that no extra layers appear:
    parent_layers_in_args = set()
    for arg_type in ["args", "kwargs"]:
        parent_layers_in_args.update(list(target_layer.parent_layer_arg_locs[arg_type].values()))
    if parent_layers_in_args != set(target_layer.parent_layers):
        return False

    argtype_dict = {
        "args": (enumerate, "creation_args"),
        "kwargs": (lambda x: x.items(), "creation_kwargs"),
    }

    # Check for each parent layer that it is logged as a saved argument when it matches an argument, and
    # is not logged when it does not match a saved argument.

    for parent_layer_label in target_layer.parent_layers:
        parent_layer = self[parent_layer_label]
        for arg_type in ["args", "kwargs"]:
            iterfunc, argtype_field = argtype_dict[arg_type]
            for key, val in iterfunc(getattr(target_layer, argtype_field)):  # type: ignore[operator]
                validation_correct_for_arg_and_layer = _validate_layer_against_arg(
                    self, target_layer, parent_layer, arg_type, key, val
                )
                if not validation_correct_for_arg_and_layer:
                    return False
    return True


def _validate_layer_against_arg(self, target_layer, parent_layer, arg_type, key, val) -> bool:
    """Validate whether a parent layer is correctly logged for a specific argument of a target layer.

    Handles nested argument structures (lists, tuples, dicts) by recursing into them
    and delegating to ``_check_arglocs_correct_for_arg`` for each leaf value.

    Args:
        target_layer: The child layer whose argument log is being checked.
        parent_layer: The parent layer being tested against the argument.
        arg_type: Either ``"args"`` or ``"kwargs"``.
        key: The positional index (for args) or keyword string (for kwargs) of the argument.
        val: The saved argument value to inspect.

    Returns:
        True if the parent layer is correctly logged for all sub-positions of this argument,
        False if any position is inconsistently logged.
    """
    if type(val) in [list, tuple]:
        for v, subval in enumerate(val):
            argloc_key = (key, v)
            validation_correct_for_arg_and_layer = _check_arglocs_correct_for_arg(
                self, target_layer, parent_layer, arg_type, argloc_key, subval
            )
            if not validation_correct_for_arg_and_layer:
                return False

    elif type(val) == dict:
        for subkey, subval in val.items():
            argloc_key = (key, subkey)
            validation_correct_for_arg_and_layer = _check_arglocs_correct_for_arg(
                self, target_layer, parent_layer, arg_type, argloc_key, subval
            )
            if not validation_correct_for_arg_and_layer:
                return False
    else:
        argloc_key = key
        validation_correct_for_arg_and_layer = _check_arglocs_correct_for_arg(
            self, target_layer, parent_layer, arg_type, argloc_key, val
        )
        if not validation_correct_for_arg_and_layer:
            return False

    return True


def _check_arglocs_correct_for_arg(
    self,
    target_layer: LayerPassLog,
    parent_layer: LayerPassLog,
    arg_type: str,
    argloc_key: Union[str, tuple],
    saved_arg_val: Any,
) -> bool:
    """For a given layer and an argument to its child layer, checks that it is logged correctly:
    that is, that it's logged as an argument if it matches, and is not logged as an argument if it doesn't match.
    """
    target_layer_label = target_layer.layer_label
    parent_layer_label = parent_layer.layer_label
    if target_layer_label in parent_layer.children_tensor_versions:
        parent_activations = parent_layer.children_tensor_versions[target_layer_label]
    else:
        parent_activations = parent_layer.tensor_contents

    if isinstance(saved_arg_val, torch.Tensor):
        parent_layer_matches_arg = tensor_nanequal(
            saved_arg_val, parent_activations, allow_tolerance=False
        )
    else:
        parent_layer_matches_arg = False
    parent_layer_logged_as_arg = (argloc_key in target_layer.parent_layer_arg_locs[arg_type]) and (
        target_layer.parent_layer_arg_locs[arg_type][argloc_key] == parent_layer_label
    )

    if (
        parent_layer_matches_arg
        and (not parent_layer_logged_as_arg)
        and (parent_activations.numel() != 0)
        and (parent_activations.dtype != torch.bool)
        and (not tensor_all_nan(parent_activations))
        and (not torch.all(parent_activations == 0))
        and (not torch.all(torch.abs(parent_activations) == 1))
        and not any(
            [
                torch.equal(parent_activations, self[other_parent].tensor_contents)
                for other_parent in target_layer.parent_layers
                if other_parent != parent_layer_label
            ]
        )
    ):
        print(
            f"Parent {parent_layer_label} of {target_layer_label} has activations that match "
            f"{arg_type} {argloc_key} for {target_layer_label}, but is not logged as "
            f"such in parent_layer_arg_locs."
        )
        return False

    if (
        not parent_layer_matches_arg
        and parent_layer_logged_as_arg
        and parent_layer.func_applied_name in ["bernoulli_", "full"]
    ):  # in-place ops may modify tensor after logging, causing arg mismatch
        return True

    if (not parent_layer_matches_arg) and parent_layer_logged_as_arg:
        print(
            f"Parent {parent_layer_label} of {target_layer_label} is logged as {arg_type} {argloc_key} to "
            f"{target_layer_label}, but its saved activations don't match the saved argument."
        )
        return False

    return True


def _check_perturbation_exemptions(
    self,
    layer: LayerPassLog,
    layers_to_perturb: List[str],
) -> bool:
    """Check whether a perturbation check should be skipped for registry-based reasons.

    Returns True if the perturbation is exempt (caller should skip), False otherwise.
    """
    for perturbed_label in layers_to_perturb:
        p_entry = self[perturbed_label]
        if p_entry.tensor_contents is not None and p_entry.tensor_contents.numel() == 0:
            return True

    func_name = layer.func_applied_name
    if func_name in STRUCTURAL_ARG_POSITIONS:
        if perturbed_layer_at_structural_position(
            self, layer, layers_to_perturb, STRUCTURAL_ARG_POSITIONS[func_name]
        ):
            return True
    if func_name in CUSTOM_EXEMPTION_CHECKS:
        if CUSTOM_EXEMPTION_CHECKS[func_name](self, layer, layers_to_perturb):  # type: ignore[misc]
            return True

    return False


def _execute_func_with_restored_state(
    layer: LayerPassLog,
    input_args: Dict,
    layers_to_perturb: List[str],
    layer_label: str,
    verbose: bool,
) -> Any:
    """Execute a layer's function with restored RNG/autocast state.

    Returns the recomputed output tensor, or None if execution raised an
    exception (which is treated as an exempt perturbation).
    """
    layer_func = layer.func_applied
    current_rng_states = log_current_rng_states()
    set_rng_from_saved_states(layer.func_rng_states)
    try:
        with AutocastRestore(layer.func_autocast_state):
            recomputed_output = layer_func(*input_args["args"], **input_args["kwargs"])
    except Exception as e:
        if verbose:
            print(
                f"Perturbation of {layers_to_perturb} for layer "
                f"{layer_label} caused {type(e).__name__}: {e}"
            )
        set_rng_from_saved_states(current_rng_states)
        return None
    set_rng_from_saved_states(current_rng_states)

    if layer_func.__name__ in ("__setitem__", "zero_", "__delitem__"):
        recomputed_output = input_args["args"][0]

    if isinstance(recomputed_output, (list, tuple)):
        recomputed_output = recomputed_output[layer.iterable_output_index]

    return recomputed_output


def _check_whether_func_on_saved_parents_yields_saved_tensor(
    self,
    layer_to_validate_parents_for_label: str,
    perturb: bool = False,
    layers_to_perturb: Optional[List[str]] = None,
    verbose: bool = False,
) -> bool:
    """Checks whether executing the saved function for a layer on the saved value of its parent layers
    in fact yields the saved activations for that layer.

    Args:
        layer_to_validate_parents_for_label: label of the layer to check the saved activations
        perturb: whether to perturb the saved activations
        layers_to_perturb: layers for which to perturb the saved activations

    Returns:
        True if the activations match, False otherwise
    """
    if layers_to_perturb is None:
        layers_to_perturb = []

    layer = self[layer_to_validate_parents_for_label]

    # Input layers and buffer layers without parents have no function to replay
    if layer.func_applied is None:
        return True

    if layer.func_applied_name in SKIP_VALIDATION_ENTIRELY:
        return True

    if perturb and _check_perturbation_exemptions(self, layer, layers_to_perturb):
        return True

    input_args = _prepare_input_args_for_validating_layer(self, layer, layers_to_perturb)
    if input_args is None:
        return True  # Can't validate without saved function args (#131)

    recomputed_output = _execute_func_with_restored_state(
        layer, input_args, layers_to_perturb, layer_to_validate_parents_for_label, verbose
    )
    if recomputed_output is None:
        if not perturb:
            import warnings

            warnings.warn(
                f"Validation replay raised an exception for layer "
                f"{layer_to_validate_parents_for_label}; treating as failed validation."
            )
            return False
        return True  # perturbed execution raised — treat as exempt perturbation

    matches_saved = tensor_nanequal(recomputed_output, layer.tensor_contents, allow_tolerance=True)

    if not matches_saved and not perturb:
        parent_has_inplace_rng = any(
            self[p].func_applied_name in ["bernoulli_", "full"] for p in layer.parent_layers
        )
        if parent_has_inplace_rng:
            return True
        print(
            f"Saved activations for layer {layer_to_validate_parents_for_label} do not match the "
            f"values computed based on the parent layers {layer.parent_layers}."
        )
        return False

    if perturb and tensor_nanequal(recomputed_output, layer.tensor_contents, allow_tolerance=False):
        return posthoc_perturb_check(self, layer, layers_to_perturb, verbose)

    return True


def _prepare_input_args_for_validating_layer(
    self,
    layer_to_validate_parents_for: LayerPassLog,
    layers_to_perturb: List[str],
) -> Dict:
    """Prepares the input arguments for validating the saved activations of a layer.

    Args:
        layer_to_validate_parents_for: Layer being checked.
        layers_to_perturb: Layers for which to perturb the saved activations.

    Returns:
        Dict of input arguments.
    """
    if layer_to_validate_parents_for.creation_args is None:
        return None  # type: ignore[return-value]  # Can't validate without saved args (#131)
    input_args = {
        "args": list(layer_to_validate_parents_for.creation_args[:]),
        "kwargs": layer_to_validate_parents_for.creation_kwargs.copy(),
    }
    input_args = _copy_validation_args(input_args)

    # Swap in saved parent activations:

    for arg_type in ["args", "kwargs"]:
        for (
            key,
            parent_layer_arg,
        ) in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type].items():
            parent_layer = self[parent_layer_arg]
            if layer_to_validate_parents_for.layer_label in parent_layer.children_tensor_versions:
                parent_values = parent_layer.children_tensor_versions[
                    layer_to_validate_parents_for.layer_label
                ]
            else:
                parent_values = parent_layer.tensor_contents
            if parent_values is None:
                continue  # Skip validation for unsaved parents (#150)
            parent_values = parent_values.detach().clone()

            if parent_layer_arg in layers_to_perturb:
                parent_layer_func_values = _perturb_layer_activations(
                    parent_values, layer_to_validate_parents_for.tensor_contents
                )
            else:
                parent_layer_func_values = parent_values

            if type(key) != tuple:
                input_args[arg_type][key] = parent_layer_func_values
            else:
                input_args[arg_type][key[0]] = assign_to_sequence_or_dict(
                    input_args[arg_type][key[0]], key[1], parent_layer_func_values
                )

    return input_args


def _deep_clone_tensors(val: Any) -> Any:
    """Recursively clone all tensors in a nested structure."""
    if isinstance(val, torch.Tensor):
        return val.detach().clone()
    elif isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v) for v in val]
        return type(val)(cloned)
    elif isinstance(val, dict):
        return {k: _deep_clone_tensors(v) for k, v in val.items()}
    return val


def _copy_validation_args(input_args: Dict) -> Dict:
    """Deep-clone all tensors in the input argument dict to avoid in-place mutation during validation.

    Args:
        input_args: Dict with ``"args"`` (list) and ``"kwargs"`` (dict) keys holding
            the raw creation arguments for a layer.

    Returns:
        A new dict with the same structure but with every tensor replaced by a detached clone.
    """
    return {
        "args": [_deep_clone_tensors(v) for v in input_args["args"]],
        "kwargs": {k: _deep_clone_tensors(v) for k, v in input_args["kwargs"].items()},
    }


def _perturb_layer_activations(
    parent_activations: torch.Tensor, output_activations: torch.Tensor
) -> torch.Tensor:
    """
    Perturbs the values of a saved tensor.

    Args:
        parent_activations: Tensor of activation values for the parent tensor
        output_activations: Tensor of activation values for the tensor whose parents are being tested (the output)

    Returns:
        Perturbed version of saved tensor
    """
    device = parent_activations.device
    if parent_activations.numel() == 0:
        return parent_activations.detach().clone()

    if parent_activations.dtype in [
        torch.int,
        torch.long,
        torch.short,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        tensor_unique_vals = torch.unique(parent_activations)
        if len(tensor_unique_vals) > 1:
            perturbed_activations = parent_activations.detach().clone()
            for _ in range(MAX_PERTURB_ATTEMPTS):
                perturbed_activations = torch.randint(
                    parent_activations.min(),  # type: ignore[call-overload]
                    parent_activations.max() + 1,
                    size=parent_activations.shape,
                    device=device,
                ).type(parent_activations.dtype)
                if not torch.equal(perturbed_activations, parent_activations):
                    break
        else:
            perturbed_activations = parent_activations.detach().clone()
            for _ in range(MAX_PERTURB_ATTEMPTS):
                if torch.min(parent_activations) < 0:
                    perturbed_activations = torch.randint(
                        -10, 11, size=parent_activations.shape, device=device
                    ).type(parent_activations.dtype)
                else:
                    perturbed_activations = torch.randint(
                        0, 11, size=parent_activations.shape, device=device
                    ).type(parent_activations.dtype)
                if not torch.equal(perturbed_activations, parent_activations):
                    break

    elif parent_activations.dtype == torch.bool:
        perturbed_activations = parent_activations.detach().clone()
        for _ in range(MAX_PERTURB_ATTEMPTS):
            perturbed_activations = torch.randint(
                0, 2, size=parent_activations.shape, device=device
            ).bool()
            if not torch.equal(perturbed_activations, parent_activations):
                break
    else:
        output_std = output_activations.detach().float().abs().mean()
        output_std += torch.rand(output_std.shape, device=device) * 100
        output_std *= torch.rand(output_std.shape, device=device)
        output_std.requires_grad = False
        scale = output_std.to(device)
        if parent_activations.is_complex():
            perturbed_activations = torch.complex(
                torch.randn(parent_activations.shape, device=device) * scale,
                torch.randn(parent_activations.shape, device=device) * scale,
            ).type(parent_activations.dtype)
        else:
            perturbed_activations = (
                torch.randn_like(parent_activations.float(), device=device) * scale
            )
            perturbed_activations = perturbed_activations.type(parent_activations.dtype)

    return perturbed_activations

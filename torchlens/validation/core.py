"""Core validation logic for verifying saved activations.

Orchestrates a three-phase verification pipeline for every layer in the graph:

1. **Ground truth check** -- model outputs match a fresh forward pass.
2. **Forward replay** (BFS from outputs toward inputs) -- re-executing each
   layer's saved function on its saved parent activations reproduces the saved
   output tensor.
3. **Perturbation check** -- for each parent of a layer, substituting random
   "wrong" values into that parent slot changes the output, proving each
   parent genuinely influences the result.

Exemption decisions (which ops to skip, which args are structural) are
delegated to the registries in ``exemptions.py``.
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

# Maximum number of random perturbation attempts before giving up on finding
# a value different from the original.  Relevant for integer/bool tensors
# where the value space may be small (e.g., a single-element int tensor).
MAX_PERTURB_ATTEMPTS = 100


def validate_saved_activations(
    self,
    ground_truth_output_tensors: List[torch.Tensor],
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Run the full validation pipeline on a completed ModelLog.

    The BFS traversal starts from two kinds of seed layers:
    - **output layers** -- whose values are verified against ``ground_truth_output_tensors``.
    - **internally terminated layers** -- dead-end tensors with no children outside the model.

    From each seed the BFS walks *backward* through parent edges.  A parent
    layer is enqueued once ALL of its child edges have been individually
    validated (edge-counting completion), ensuring diamond-shaped subgraphs
    are handled correctly.

    After activation validation passes, optional metadata invariant checks
    (checks A-R in ``invariants.py``) run to verify structural/semantic
    consistency of the entire ModelLog.

    Args:
        ground_truth_output_tensors: Output tensors from a fresh forward pass,
            used to confirm the logged outputs are accurate before BFS begins.
        verbose: Whether to print warning messages on validation failure.
        validate_metadata: Whether to run metadata invariant checks (default True).

    Returns:
        True if all checks pass, False otherwise.
    """
    # Phase 0: verify logged outputs match a fresh forward pass (no tolerance).
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

    # BFS backward from outputs + internally terminated layers.
    # Edge-counting approach: a parent is enqueued only after ALL its child
    # edges are validated (validated_child_edges == set(child_layers)).
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

    # Completeness check: BFS must visit every layer in the graph.
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
    """Validate a single layer's parent edges: argument logging, forward replay,
    and perturbation for each parent.

    This is the inner loop of the BFS. For the layer identified by
    ``layer_to_validate_parents_for_label``, this function:

    1. Checks that the parent layer activations are correctly logged in the
       argument map (``parent_layer_arg_locs``).
    2. Re-executes the layer's function on saved parent values and confirms the
       output matches the saved tensor (forward replay, ``perturb=False``).
    3. For each parent, re-executes with that parent's activation *perturbed*
       and confirms the output *changes* (perturbation check, ``perturb=True``).
       Ops in ``SKIP_PERTURBATION_ENTIRELY`` skip this step.

    After all checks pass, each parent's validated-child-edge set is updated.
    When a parent has ALL its child edges validated, it is added to the BFS
    work queue (unless it is an input layer or a parentless buffer).

    Args:
        layer_to_validate_parents_for_label: Label of the layer whose parent edges are being validated.
        validated_layers: Set of layer labels already validated; mutated in-place to add newly validated layers.
        validated_child_edges_for_each_layer: Dict mapping each layer label to the set of its child edges
            that have been validated so far; mutated in-place as child edges are confirmed.
        layers_to_validate_parents_for: Work queue of layer labels still needing parent validation;
            mutated in-place to append newly discovered layers.
        verbose: Whether to print warning messages on validation failure.

    Returns:
        True if all parent edges are valid, False on the first failure.
    """
    layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]

    # Check that the arguments are logged correctly:
    if not _check_layer_arguments_logged_correctly(self, layer_to_validate_parents_for_label):
        print(
            f"Parent arguments for layer {layer_to_validate_parents_for_label} are not logged properly; "
            f"either a parent wasn't logged as an argument, or was logged an extra time"
        )
        return False

    # Forward replay: re-execute with correct parent values, expect same output.
    if not _check_whether_func_on_saved_parents_yields_saved_tensor(
        self, layer_to_validate_parents_for_label, perturb=False
    ):
        return False

    # Perturbation: for each parent, substitute random values and expect
    # the output to change, proving that parent genuinely influences this layer.

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

    # Record validated edges and enqueue parents whose ALL child edges are now validated.
    for parent_layer_label in layer_to_validate_parents_for.parent_layers:
        parent_layer = self[parent_layer_label]
        validated_child_edges_for_each_layer[parent_layer_label].add(
            layer_to_validate_parents_for_label
        )
        # Edge-counting completion: enqueue parent only when ALL its child
        # edges have been validated.  This correctly handles diamond graphs
        # where a parent feeds multiple children.
        if validated_child_edges_for_each_layer[parent_layer_label] == set(
            parent_layer.child_layers
        ):
            validated_layers.add(parent_layer_label)
            # Don't enqueue terminal seeds (inputs, parentless buffers) --
            # they have no parents to validate further.
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
    """Check bidirectional consistency between a parent's tensor and a child's arg slot.

    Validates two directions:
    - If the parent's activation matches the saved arg value AND the parent is
      not logged at that position, that is an error (unless the match is
      trivially coincidental -- e.g., empty tensor, bool tensor, all-NaN,
      all-zero, or all-abs-one, or another parent has identical values).
    - If the parent is logged at that position BUT its activation does not
      match the saved arg value, that is an error (with a special exemption
      for in-place RNG ops like ``bernoulli_`` and ``full`` that mutate after
      logging).

    Args:
        target_layer: The child layer whose argument log is being checked.
        parent_layer: The parent layer being tested against the argument.
        arg_type: Either ``"args"`` or ``"kwargs"``.
        argloc_key: The position key (int, str, or tuple for nested args).
        saved_arg_val: The saved argument value at that position.

    Returns:
        True if the logging is consistent, False if an inconsistency is found.
    """
    target_layer_label = target_layer.layer_label
    parent_layer_label = parent_layer.layer_label
    # children_tensor_versions stores per-child snapshots when an in-place
    # op modified the tensor between uses.  Fall back to tensor_contents
    # when no child-specific version exists.
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

    # Case 1: parent matches the arg value but is NOT logged at this position.
    # This is only an error if the match is non-trivially coincidental.
    # Exemptions for trivially coincidental matches:
    #   - empty tensor (numel==0)
    #   - bool tensor (many ops produce True/False identity matches)
    #   - all-NaN, all-zero, or all-abs-one tensors (special values)
    #   - another parent has identical tensor values (ambiguous attribution)
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

    # Case 2 exemption: in-place RNG ops (bernoulli_, full) mutate the tensor
    # AFTER it was logged as an arg, so the saved activation no longer matches
    # the creation_args snapshot.  This is expected and not a real mismatch.
    if (
        not parent_layer_matches_arg
        and parent_layer_logged_as_arg
        and parent_layer.func_applied_name in ["bernoulli_", "full"]
    ):
        return True

    # Case 3: parent is logged at this position but values don't match.
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

    Checks three exemption sources in order:
    1. Empty tensors (numel==0) -- perturbing an empty tensor is meaningless.
    2. Structural arg positions (``STRUCTURAL_ARG_POSITIONS``) -- the perturbed
       layer occupies a position that controls structure, not values (e.g.,
       index tensors for ``embedding``, ``index_select``).
    3. Custom exemption functions (``CUSTOM_EXEMPTION_CHECKS``) -- op-specific
       logic for complex cases like ``__getitem__``, ``lstm``, etc.

    Returns True if the perturbation is exempt (caller should skip), False otherwise.
    """
    # Empty tensors cannot be meaningfully perturbed.
    for perturbed_label in layers_to_perturb:
        p_entry = self[perturbed_label]
        if p_entry.tensor_contents is not None and p_entry.tensor_contents.numel() == 0:
            return True

    func_name = layer.func_applied_name

    # Registry 3: structural arg positions (e.g., index tensor for embedding).
    if func_name in STRUCTURAL_ARG_POSITIONS:
        if perturbed_layer_at_structural_position(
            self, layer, layers_to_perturb, STRUCTURAL_ARG_POSITIONS[func_name]
        ):
            return True

    # Registry 4: custom per-op exemption checks.
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
    """Execute a layer's function with restored RNG and autocast state.

    Restores the exact RNG state that was active when this layer originally
    ran, so that stochastic ops (dropout, etc.) reproduce identical results
    during forward replay.

    **Exception handling**: catches ALL exceptions and returns ``None``.
    The caller treats ``None`` as:
    - For ``perturb=True``: an exempt perturbation (perturbation caused an
      invalid input combination, e.g., out-of-range indices -- this is
      expected and not a validation failure).
    - For ``perturb=False``: a failed validation (the layer's own function
      can't be replayed, which IS a problem).

    Returns the recomputed output tensor, or None on exception.
    """
    layer_func = layer.func_applied

    # Save current RNG state so we can restore it after replay.
    current_rng_states = log_current_rng_states()
    # Restore the RNG state from when the layer originally executed.
    set_rng_from_saved_states(layer.func_rng_states)

    try:
        with AutocastRestore(layer.func_autocast_state):
            recomputed_output = layer_func(*input_args["args"], **input_args["kwargs"])
    except Exception as e:
        # Broad catch: perturbed values can trigger any exception (shape
        # mismatch, index OOB, dtype error, etc.).  Returning None lets the
        # caller decide whether this is acceptable.
        if verbose:
            print(
                f"Perturbation of {layers_to_perturb} for layer "
                f"{layer_label} caused {type(e).__name__}: {e}"
            )
        set_rng_from_saved_states(current_rng_states)
        return None

    # Restore the original RNG state so subsequent validation steps
    # don't inherit this layer's replayed state.
    set_rng_from_saved_states(current_rng_states)

    # In-place mutating ops (__setitem__, zero_, __delitem__) return None
    # from PyTorch but the "output" is the mutated first argument.
    if layer_func.__name__ in ("__setitem__", "zero_", "__delitem__"):
        recomputed_output = input_args["args"][0]

    # Multi-output functions (e.g., torch.max with dim) return a tuple;
    # select the specific output this layer represents.
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

    # Early exits for layers that cannot or should not be replayed.

    # Input layers and buffer layers without parents have no function to replay
    # (func_applied is None for model inputs and parentless buffers).
    if layer.func_applied is None:
        return True

    # Registry 1: skip ALL validation for nondeterministic ops (e.g., empty_like).
    if layer.func_applied_name in SKIP_VALIDATION_ENTIRELY:
        return True

    # Pre-execution perturbation exemptions (structural args, custom checks).
    if perturb and _check_perturbation_exemptions(self, layer, layers_to_perturb):
        return True

    input_args = _prepare_input_args_for_validating_layer(self, layer, layers_to_perturb)
    if input_args is None:
        return True  # Can't validate without saved function args (#131)

    recomputed_output = _execute_func_with_restored_state(
        layer, input_args, layers_to_perturb, layer_to_validate_parents_for_label, verbose
    )

    # None means execution raised an exception (see _execute_func_with_restored_state).
    if recomputed_output is None:
        if not perturb:
            # Non-perturbed replay failure is a real validation failure.
            import warnings

            warnings.warn(
                f"Validation replay raised an exception for layer "
                f"{layer_to_validate_parents_for_label}; treating as failed validation."
            )
            return False
        # Perturbed execution raised -- the perturbed values caused an invalid
        # input (e.g., wrong shape).  Treat as exempt.
        return True

    matches_saved = tensor_nanequal(recomputed_output, layer.tensor_contents, allow_tolerance=True)

    # Forward replay failure (non-perturbed): saved activations don't match.
    if not matches_saved and not perturb:
        # Exemption: parent is an in-place RNG op that may have mutated its
        # tensor after the child logged it as an arg.
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

    # Perturbation produced identical output -- run posthoc checks to see if
    # there's a valid excuse (bool output, special-value args, type cast, etc.).
    # Uses exact equality (no tolerance) since any change should be detectable.
    if perturb and tensor_nanequal(recomputed_output, layer.tensor_contents, allow_tolerance=False):
        return posthoc_perturb_check(self, layer, layers_to_perturb, verbose)

    return True


def _prepare_input_args_for_validating_layer(
    self,
    layer_to_validate_parents_for: LayerPassLog,
    layers_to_perturb: List[str],
) -> Dict:
    """Build the input argument dict for replaying a layer's function.

    Starts from the layer's saved ``creation_args`` / ``creation_kwargs``,
    deep-clones all tensors to prevent in-place mutation during replay, then
    swaps in each parent's saved activation (or a perturbed version) at the
    correct argument position.

    For nested argument positions (tuples, dicts nested inside args), the
    ``parent_layer_arg_locs`` key is a tuple ``(outer_key, inner_key)`` and
    ``assign_to_sequence_or_dict`` handles the nested assignment.

    Args:
        layer_to_validate_parents_for: Layer being checked.
        layers_to_perturb: Layers for which to perturb the saved activations.

    Returns:
        Dict with ``"args"`` and ``"kwargs"`` keys, or None if creation_args
        was not saved (can't validate without them).
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
    """Recursively clone all tensors in a nested structure of lists/tuples/dicts.

    Non-tensor leaves are returned as-is (shared reference).  Tensor leaves
    are detached and cloned so that in-place ops during validation replay
    don't corrupt the original saved data.

    Preserves container types: a tuple input produces a tuple output, not a list.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().clone()
    elif isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v) for v in val]
        # Preserve the original container type (list vs tuple vs namedtuple).
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
    """Generate a random perturbation of a saved tensor for validation.

    The perturbation strategy varies by dtype to produce meaningful
    "wrong" values while respecting type constraints:

    - **Integer types**: sample uniformly from [min, max+1) of the original
      tensor.  If the tensor is all-one-value (single unique value), widen
      the range to [-10, 11) or [0, 11).  Retries up to
      ``MAX_PERTURB_ATTEMPTS`` to avoid accidentally reproducing the original.
    - **Bool**: random 0/1, retried to ensure difference.
    - **Float/complex**: scaled random normal, where the scale is derived from
      the output tensor's mean absolute value (plus random jitter) to produce
      perturbations of comparable magnitude.

    Args:
        parent_activations: The original parent tensor to perturb.
        output_activations: The child layer's output tensor, used to calibrate
            the perturbation scale for float types.

    Returns:
        A new tensor of the same shape/dtype with perturbed values.
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
            # Multiple unique values: sample within the original range.
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
            # Single unique value: widen to a fixed range to guarantee a
            # different value can be produced.
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
        # Random bool, retried until different from original.
        perturbed_activations = parent_activations.detach().clone()
        for _ in range(MAX_PERTURB_ATTEMPTS):
            perturbed_activations = torch.randint(
                0, 2, size=parent_activations.shape, device=device
            ).bool()
            if not torch.equal(perturbed_activations, parent_activations):
                break
    else:
        # Float/complex: scaled random normal.  The scale is derived from the
        # output's mean absolute value plus random jitter, ensuring the
        # perturbation is large enough to be detectable but not astronomical.
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

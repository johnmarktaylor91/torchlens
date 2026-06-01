"""Core validation logic for verifying saved outs.

Orchestrates a three-phase verification pipeline for every layer in the graph:

1. **Ground truth check** -- model outputs match a fresh forward pass.
2. **Forward replay** (BFS from outputs toward inputs) -- re-executing each
   layer's saved function on its saved parent outs reproduces the saved
   output tensor.
3. **Perturbation check** -- for each parent of a layer, substituting random
   "wrong" values into that parent slot changes the output, proving each
   parent genuinely influences the result.

Exemption decisions (which ops to skip, which args are structural) are
delegated to the registries in ``exemptions.py``.
"""

from collections import defaultdict, deque
from typing import Optional, Any, Dict, List, Set, TYPE_CHECKING, Union, cast

import torch

from ..data_classes.op_log import Op

if TYPE_CHECKING:
    from ..data_classes.model_log import Trace

from ..utils.rng import execute_with_restored_rng_autocast
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

# Deep convolutional models can replay the same FP32 op with tiny differences
# after many accumulated reductions. Keep this local to validation replay so
# global tensor equality stays strict for graph bookkeeping and lower-tier tests.
DEEP_NUMERIC_REPLAY_FUNCS = frozenset(
    {
        "addmm",
        "baddbmm",
        "bmm",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "linear",
        "matmul",
        "mm",
    }
)
DEEP_NUMERIC_REPLAY_MIN_OPERATION_NUM = 100
DEEP_NUMERIC_REPLAY_RTOL = 1e-3
DEEP_NUMERIC_REPLAY_ATOL = 1e-4
DEEP_NUMERIC_REPLAY_OUTLIER_RTOL = 5e-2
DEEP_NUMERIC_REPLAY_OUTLIER_ATOL = 1e-2
DEEP_NUMERIC_REPLAY_MAX_OUTLIER_FRACTION = 1e-4
DEEP_NUMERIC_REPLAY_MAX_SCALED_DIFF = 5e-2
DEEP_NUMERIC_REPLAY_MAX_MEAN_SCALED_DIFF = 1e-3
GROUND_TRUTH_OUTPUT_RTOL = 1e-6
GROUND_TRUTH_OUTPUT_ATOL = 1e-8


def _raise_if_portable_bundle_log(self: Any) -> None:
    """Reject validation only when loaded logs lack replay callables.

    Parameters
    ----------
    self:
        Model log being validated.

    Raises
    ------
    TorchLensIOError
        If the log was loaded from a portable bundle without resolved
        ``func`` callables on computational nodes.
    """

    if not bool(getattr(self, "_loaded_from_bundle", False)):
        return
    unresolved = [
        getattr(layer, "layer_label", "<unknown>")
        for layer in getattr(self, "layer_list", [])
        if getattr(layer, "func", None) is None
        and getattr(layer, "func_name", "none") not in {"none", "input", "output", "buffer"}
    ]
    if unresolved:
        from .._io import TorchLensIOError

        raise TorchLensIOError(
            "validate_forward_pass requires resolved func callables; portable bundles "
            "drop them when functions cannot be represented. This bundle has unresolved "
            f"computational functions, e.g. {unresolved[:3]!r}."
        )


def _ground_truth_output_matches_saved(
    saved_output: torch.Tensor,
    ground_truth_output: torch.Tensor,
) -> bool:
    """Return whether a saved model output matches the direct forward output.

    The direct output check is exact first. For floating-point outputs, it then
    allows only sub-ULP wrapper noise, which covers models whose logged full
    forward produces numerically equivalent logits that differ at ~1e-11 scale.

    Parameters
    ----------
    saved_output:
        Output tensor saved by TorchLens logging.
    ground_truth_output:
        Output tensor from the direct model forward pass.

    Returns
    -------
    bool
        True if the outputs are exactly equal or differ only by the tight
        output-only floating-point tolerance.
    """
    if tensor_nanequal(saved_output, ground_truth_output, allow_tolerance=False):
        return True
    if saved_output.shape != ground_truth_output.shape:
        return False
    if saved_output.dtype != ground_truth_output.dtype:
        return False
    if not saved_output.is_floating_point():
        return False

    from .._state import pause_logging

    with pause_logging():
        if not torch.equal(saved_output.isnan(), ground_truth_output.isnan()):
            return False
        if not torch.equal(saved_output.isinf(), ground_truth_output.isinf()):
            return False
        saved_nonan = torch.nan_to_num(saved_output, 0.7234691827346)
        ground_truth_nonan = torch.nan_to_num(ground_truth_output, 0.7234691827346)
        return bool(
            torch.allclose(
                saved_nonan,
                ground_truth_nonan,
                rtol=GROUND_TRUTH_OUTPUT_RTOL,
                atol=GROUND_TRUTH_OUTPUT_ATOL,
            )
        )


def validate_saved_outs(
    self: "Trace",
    ground_truth_output_tensors: List[torch.Tensor],
    verbose: bool = False,
    validate_metadata: bool = True,
) -> bool:
    """Run the full validation pipeline on a completed Trace.

    The BFS traversal starts from two kinds of seed layers:
    - **output layers** -- whose values are verified against ``ground_truth_output_tensors``.
    - **internally terminated layers** -- dead-end tensors with no children outside the model.

    From each seed the BFS walks *backward* through parent edges.  A parent
    layer is enqueued once ALL of its child edges have been individually
    validated (edge-counting completion), ensuring diamond-shaped subgraphs
    are handled correctly.

    After out validation ops, optional metadata invariant checks
    (checks A-R in ``invariants.py``) run to verify structural/semantic
    consistency of the entire Trace.

    Args:
        ground_truth_output_tensors: Output tensors from a fresh forward pass,
            used to confirm the logged outputs are accurate before BFS begins.
        verbose: Whether to print warning messages on validation failure.
        validate_metadata: Whether to run metadata invariant checks (default True).

    Returns:
        True if all checks pass, False otherwise.
    """
    _raise_if_portable_bundle_log(self)

    # Phase 0: verify logged outputs match a fresh forward pass.
    output_label_counts: dict[str, int] = defaultdict(int)
    for i, output_layer_label in enumerate(self.output_layers):
        output_layer = _resolve_output_entry_for_index(
            self, output_layer_label, output_label_counts
        )
        if output_layer.out is None:
            print(f"The {i}th output layer, {output_layer_label}, has no saved out.")
            return False
        if not _ground_truth_output_matches_saved(output_layer.out, ground_truth_output_tensors[i]):
            print(
                f"The {i}th output layer, {output_layer_label}, does not match the ground truth output tensor."
            )
            return False

    # BFS backward from outputs + internally terminated layers.
    # Edge-counting approach: a parent is enqueued only after ALL its child
    # edges are validated (validated_child_edges == set(children)).
    validated_child_edges_for_each_layer: Dict[str, Set[str]] = defaultdict(set)
    validated_layers = {
        self[label].layer_label for label in self.output_layers + self.internal_sink_ops
    }
    layers_to_validate_parents_for = deque(validated_layers)

    while len(layers_to_validate_parents_for) > 0:
        layer_to_validate_parents_for = layers_to_validate_parents_for.popleft()
        parents_valid = validate_parents_of_saved_layer(
            self,
            layer_to_validate_parents_for,
            validated_layers,
            validated_child_edges_for_each_layer,
            layers_to_validate_parents_for,  # type: ignore[arg-type]
            verbose,
        )
        if not parents_valid:
            return False

    # Completeness check: BFS must visit every layer in the graph.
    expected_layers = {layer.layer_label for layer in self.layer_list}
    if len(validated_layers) < len(expected_layers):
        unreached = expected_layers - validated_layers
        print(
            f"All saved outs were accurate, but some layers were not reached (check that "
            f"child args logged accurately): {unreached}"
        )
        return False

    # Metadata invariant checks (after out validation ops)
    if validate_metadata:
        from .invariants import check_metadata_invariants

        check_metadata_invariants(self)

    return True


def validate_parents_of_saved_layer(
    self: "Trace",
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

    1. Checks that the parent layer outs are correctly logged in the
       argument map (``parent_arg_positions``).
    2. Re-executes the layer's function on saved parent values and confirms the
       output matches the saved tensor (forward replay, ``perturb=False``).
    3. For each parent, re-executes with that parent's out *perturbed*
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
    ops_to_validate = _validation_ops_for_entry(layer_to_validate_parents_for)

    # Check that the arguments are logged correctly:
    if not _check_layer_arguments_logged_correctly(self, layer_to_validate_parents_for_label):
        print(
            f"Parent arguments for layer {layer_to_validate_parents_for_label} are not logged properly; "
            f"either a parent wasn't logged as an argument, or was logged an extra time"
        )
        return False

    # Forward replay: re-execute with correct parent values, expect same output.
    ops_to_replay = _representative_ops_for_replay(self, ops_to_validate)
    for target_op in ops_to_replay:
        if not _check_whether_func_on_saved_parents_yields_saved_tensor(
            self, target_op.label, perturb=False
        ):
            return False

    # Perturbation: for each parent, substitute random values and expect
    # the output to change, proving that parent genuinely influences this layer.

    representative_parent_edges = _representative_parent_edges(self, ops_to_replay)
    for target_op, perturb_layer in representative_parent_edges:
        if target_op.func_name in SKIP_PERTURBATION_ENTIRELY:
            continue
        if not _check_whether_func_on_saved_parents_yields_saved_tensor(
            self,
            target_op.label,
            perturb=True,
            layers_to_perturb=[perturb_layer],
            verbose=verbose,
        ):
            return False

    # Record validated edges and enqueue parents whose ALL child edges are now validated.
    for parent_layer_label in layer_to_validate_parents_for.parents:
        parent_layer = self[parent_layer_label]
        validated_child_edges_for_each_layer[parent_layer_label].add(
            layer_to_validate_parents_for_label
        )
        # Enqueue a parent once at least one validated child proves a path to a
        # checked output or internal sink. Recurrent multi-pass layers can have
        # self/side child edges that are valid but not part of the current
        # representative validation frontier.
        if parent_layer_label not in validated_layers:
            validated_layers.add(parent_layer_label)
            # Don't enqueue terminal seeds (inputs, parentless buffers) --
            # they have no parents to validate further.
            if (not parent_layer.is_input) and not (
                parent_layer.is_buffer and (parent_layer.buffer_source is None)
            ):
                layers_to_validate_parents_for.append(parent_layer_label)

    return True


def _resolve_output_entry_for_index(
    self: "Trace", output_layer_label: str, output_label_counts: dict[str, int]
) -> Op:
    """Resolve an output label occurrence to a concrete output Op.

    Parameters
    ----------
    output_layer_label:
        Layer label from ``Trace.output_layers``.
    output_label_counts:
        Mutable per-label occurrence counts used to map duplicate output layer
        labels to their pass-specific output ops.

    Returns
    -------
    Op
        Concrete output op for this output occurrence.
    """

    output_entry = self[output_layer_label]
    ops = getattr(output_entry, "ops", None)
    if not hasattr(ops, "_list"):
        return cast(Op, output_entry)
    op_list = cast(list[Op], cast(Any, ops)._list)
    occurrence_index = output_label_counts[output_layer_label]
    output_label_counts[output_layer_label] += 1
    return op_list[occurrence_index]


def _representative_ops_for_replay(self: "Trace", ops_to_validate: List[Op]) -> List[Op]:
    """Return concrete child ops for expensive forward replay validation.

    Parameters
    ----------
    ops_to_validate:
        Concrete op passes for the child Layer currently being validated.

    Returns
    -------
    list of Op
        A stable subset that covers the Layer's distinct parent-layer edges.
    """

    if len(ops_to_validate) <= 1:
        return ops_to_validate

    representative_ops: dict[str, Op] = {}
    fallback_op = ops_to_validate[0]
    for target_op in ops_to_validate:
        if not target_op.parents:
            representative_ops.setdefault(target_op.label, target_op)
        for parent_label in target_op.parents:
            parent_layer_label = self[parent_label].layer_label
            representative_ops.setdefault(parent_layer_label, target_op)
    if not representative_ops:
        return [fallback_op]
    return list(dict.fromkeys(representative_ops.values()))


def _representative_parent_edges(self: "Trace", ops_to_validate: List[Op]) -> List[tuple[Op, str]]:
    """Return one concrete op edge for each parent Layer edge.

    Parameters
    ----------
    ops_to_validate:
        Concrete op passes for the child Layer currently being validated.

    Returns
    -------
    list of tuple of Op and str
        Pairs of child op and pass-qualified parent label to perturb.
    """

    representative_edges: dict[str, tuple[Op, str]] = {}
    for target_op in ops_to_validate:
        for parent_label in target_op.parents:
            parent_layer_label = self[parent_label].layer_label
            representative_edges.setdefault(parent_layer_label, (target_op, parent_label))
    return list(representative_edges.values())


def _validation_ops_for_entry(entry: Any) -> List[Op]:
    """Return pass-specific ops that should be used for validation.

    Parameters
    ----------
    entry:
        Trace entry resolved from a validation queue label. This is usually a
        ``Layer`` but may already be an ``Op`` for pass-qualified labels.

    Returns
    -------
    list of Op
        The concrete op passes whose saved args and outputs can be replayed.
    """

    ops = getattr(entry, "ops", None)
    if not hasattr(ops, "_list"):
        return [cast(Op, entry)]
    op_list = cast(list[Op], cast(Any, ops)._list)
    if len(op_list) == 0:
        return []
    return [op_list[0]]


def _check_layer_arguments_logged_correctly(self: "Trace", target_layer_label: str) -> bool:
    """Check whether the outs of the parent layers match the saved arguments of
    the target layer, and that the argument locations have been logged correctly.

    Args:
        target_layer_label: Layer to check

    Returns:
        True if arguments logged accurately, False otherwise
    """
    target_entry = self[target_layer_label]
    target_ops = _representative_ops_for_replay(self, _validation_ops_for_entry(target_entry))

    for target_layer in target_ops:
        # Genuine functionless ops have no torch function whose arguments could
        # be reconstructed: graph sources (inputs, buffers, internally generated
        # sources such as a torch.vmap-built attention mask) and GENUINE raw
        # forward-hook output replacements injected by the user. This skip is
        # deliberately narrow -- it must NOT swallow a real op that merely lost
        # its func, which would mask a capture bug.
        is_genuine_replacement = getattr(
            target_layer, "intervention_replaced", False
        ) and not getattr(target_layer, "is_internal_source", False)
        if target_layer.func is None and (
            target_layer.is_input
            or getattr(target_layer, "is_buffer", False)
            or getattr(target_layer, "is_internal_source", False)
            or is_genuine_replacement
        ):
            continue

        # Make sure that all parent layers appear in at least one argument and
        # that no extra layers appear:
        parents_in_args = set()
        for arg_type in ["args", "kwargs"]:
            parents_in_args.update(list(target_layer.parent_arg_positions[arg_type].values()))
        if parents_in_args != set(target_layer.parents):
            return False

        argtype_dict = {
            "args": (enumerate, "saved_args"),
            "kwargs": (lambda x: x.items(), "saved_kwargs"),
        }

        # Check for each parent layer that it is logged as a saved argument when it matches an argument,
        # and is not logged when it does not match a saved argument.

        for parent_layer_label in target_layer.parents:
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


def _validate_layer_against_arg(
    self: "Trace",
    target_layer: Op,
    parent_layer: Op,
    arg_type: str,
    key: Any,
    val: Any,
) -> bool:
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


def _parent_logged_for_any_arg_alias(target_layer: Op, parent_layer_labels: set[str]) -> bool:
    """Return whether any parent label alias is logged at any arg location.

    Parameters
    ----------
    target_layer:
        Child op whose parent-arg map is being inspected.
    parent_layer_labels:
        Equivalent labels for the same parent, usually pass-qualified op label
        and bare parent layer label.

    Returns
    -------
    bool
        True when any alias appears in the target arg-position map.
    """

    return any(
        logged_parent in parent_layer_labels
        for arg_type in ("args", "kwargs")
        for logged_parent in target_layer.parent_arg_positions[arg_type].values()
    )


def _check_arglocs_correct_for_arg(
    self: "Trace",
    target_layer: Op,
    parent_layer: Op,
    arg_type: str,
    argloc_key: str | tuple[Any, ...],
    saved_arg_val: Any,
) -> bool:
    """Check bidirectional consistency between a parent's tensor and a child's arg slot.

    Validates two directions:
    - If the parent's out matches the saved arg value AND the parent is
      not logged at that position, that is an error (unless the match is
      trivially coincidental -- e.g., empty tensor, bool tensor, all-NaN,
      all-zero, or all-abs-one, or another parent has identical values).
    - If the parent is logged at that position BUT its out does not
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
    target_op_label = getattr(target_layer, "label", target_layer_label)
    parent_layer_label = parent_layer.layer_label
    parent_arg_labels = {getattr(parent_layer, "label", parent_layer_label), parent_layer_label}
    # out_versions_by_child stores per-child snapshots when an in-place
    # op modified the tensor between uses.  Fall back to out
    # when no child-specific version exists.
    if target_op_label in parent_layer.out_versions_by_child:
        parent_outs = parent_layer.out_versions_by_child[target_op_label]
    elif target_layer_label in parent_layer.out_versions_by_child:
        parent_outs = parent_layer.out_versions_by_child[target_layer_label]
    else:
        parent_outs = parent_layer.out

    if isinstance(saved_arg_val, torch.Tensor):
        parent_layer_matches_arg = tensor_nanequal(
            saved_arg_val, parent_outs, allow_tolerance=False
        )
    else:
        parent_layer_matches_arg = False
    parent_layerged_as_arg = (
        argloc_key in target_layer.parent_arg_positions[arg_type]
        and target_layer.parent_arg_positions[arg_type][argloc_key] in parent_arg_labels
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
        and (not parent_layerged_as_arg)
        and (not _parent_logged_for_any_arg_alias(target_layer, parent_arg_labels))
        and (parent_outs.numel() != 0)
        and (parent_outs.dtype != torch.bool)
        and (not tensor_all_nan(parent_outs))
        and (not torch.all(parent_outs == 0))
        and (not torch.all(torch.abs(parent_outs) == 1))
        and not any(
            [
                torch.equal(parent_outs, self[other_parent].out)
                for other_parent in target_layer.parents
                if other_parent != parent_layer_label
            ]
        )
    ):
        print(
            f"Parent {parent_layer_label} of {target_layer_label} has outs that match "
            f"{arg_type} {argloc_key} for {target_layer_label}, but is not logged as "
            f"such in parent_arg_positions."
        )
        return False

    # Case 2 exemption: in-place RNG ops (bernoulli_, full) mutate the tensor
    # AFTER it was logged as an arg, so the saved out no longer matches
    # the saved_args snapshot.  This is expected and not a real mismatch.
    if (
        not parent_layer_matches_arg
        and parent_layerged_as_arg
        and parent_layer.func_name in ["bernoulli_", "full"]
    ):
        return True

    # Case 3: parent is logged at this position but values don't match.
    if (not parent_layer_matches_arg) and parent_layerged_as_arg:
        print(
            f"Parent {parent_layer_label} of {target_layer_label} is logged as {arg_type} {argloc_key} to "
            f"{target_layer_label}, but its saved outs don't match the saved argument."
        )
        return False

    return True


def _check_perturbation_exemptions(
    self: "Trace",
    layer: Op,
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
        if p_entry.out is not None and p_entry.out.numel() == 0:
            return True

    func_name = layer.func_name

    # Registry 3: structural arg positions (e.g., index tensor for embedding).
    if func_name in STRUCTURAL_ARG_POSITIONS:
        if perturbed_layer_at_structural_position(
            self, layer, layers_to_perturb, STRUCTURAL_ARG_POSITIONS[func_name]
        ):
            return True

    # Registry 4: custom per-op exemption checks.
    if func_name in CUSTOM_EXEMPTION_CHECKS:
        if CUSTOM_EXEMPTION_CHECKS[func_name](self, layer, layers_to_perturb):
            return True

    return False


def _execute_func_with_restored_state(
    layer: Op,
    input_args: dict[str, Any],
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
    layer_func = layer.func

    try:
        recomputed_output = execute_with_restored_rng_autocast(
            layer_func,
            tuple(input_args["args"]),
            dict(input_args["kwargs"]),
            rng_states=layer.func_rng_states,
            autocast_state=layer.func_autocast_state,
        )
    except Exception as e:
        # Broad catch: perturbed values can trigger any exception (shape
        # mismatch, index OOB, dtype error, etc.).  Returning None lets the
        # caller decide whether this is acceptable.
        if verbose:
            print(
                f"Perturbation of {layers_to_perturb} for layer "
                f"{layer_label} caused {type(e).__name__}: {e}"
            )
        return None

    # In-place mutating ops (__setitem__, zero_, __delitem__) return None
    # from PyTorch but the "output" is the mutated first argument.
    if layer_func.__name__ in ("__setitem__", "zero_", "__delitem__"):
        recomputed_output = input_args["args"][0]

    # Multi-output functions (e.g., torch.max with dim) return a tuple;
    # select the specific output this layer represents.
    if isinstance(recomputed_output, (list, tuple)):
        recomputed_output = recomputed_output[layer.multi_output_index]

    return recomputed_output


def _deep_numeric_replay_matches_saved(
    layer: Op,
    recomputed_output: torch.Tensor,
) -> bool:
    """Return whether a deep numeric replay matches within local relaxed tolerance.

    The standard validation tolerance remains the default for every layer. This
    fallback is intentionally narrow: it only applies to later convolution-like
    numeric ops, first tries a modest ``allclose`` relaxation, then allows a
    tiny fraction of stricter-check outliers only when the overall scaled error
    is still very small.

    Parameters
    ----------
    layer:
        Layer whose saved out is being replayed.
    recomputed_output:
        Output from re-executing ``layer.func`` on saved parent values.

    Returns
    -------
    bool
        True if this layer qualifies for the deep numeric replay tolerance and
        the recomputed output is close enough to the saved out.
    """
    saved_output = layer.out
    if saved_output is None:
        return False
    if layer.func_name not in DEEP_NUMERIC_REPLAY_FUNCS:
        return False
    if layer.step_index < DEEP_NUMERIC_REPLAY_MIN_OPERATION_NUM:
        return False
    if recomputed_output.shape != saved_output.shape:
        return False
    if recomputed_output.dtype != saved_output.dtype:
        return False
    if not recomputed_output.is_floating_point():
        return False

    from .._state import pause_logging

    with pause_logging():
        if not torch.equal(recomputed_output.isnan(), saved_output.isnan()):
            return False
        if not torch.equal(recomputed_output.isinf(), saved_output.isinf()):
            return False

        recomputed_nonan = torch.nan_to_num(recomputed_output, 0.7234691827346)
        saved_nonan = torch.nan_to_num(saved_output, 0.7234691827346)

        if torch.allclose(
            recomputed_nonan,
            saved_nonan,
            rtol=DEEP_NUMERIC_REPLAY_RTOL,
            atol=DEEP_NUMERIC_REPLAY_ATOL,
        ):
            return True

        close = torch.isclose(
            recomputed_nonan,
            saved_nonan,
            rtol=DEEP_NUMERIC_REPLAY_OUTLIER_RTOL,
            atol=DEEP_NUMERIC_REPLAY_OUTLIER_ATOL,
        )
        outlier_fraction = (~close).sum().item() / close.numel()
        if outlier_fraction > DEEP_NUMERIC_REPLAY_MAX_OUTLIER_FRACTION:
            return False

        diff = (recomputed_nonan - saved_nonan).abs()
        scale = torch.maximum(recomputed_nonan.abs(), saved_nonan.abs()) + 1e-12
        scaled_diff = diff / scale
        return bool(
            scaled_diff.max().item() <= DEEP_NUMERIC_REPLAY_MAX_SCALED_DIFF
            and scaled_diff.mean().item() <= DEEP_NUMERIC_REPLAY_MAX_MEAN_SCALED_DIFF
        )


def _check_whether_func_on_saved_parents_yields_saved_tensor(
    self: "Trace",
    layer_to_validate_parents_for_label: str,
    perturb: bool = False,
    layers_to_perturb: Optional[List[str]] = None,
    verbose: bool = False,
) -> bool:
    """Checks whether executing the saved function for a layer on the saved value of its parent layers
    in fact yields the saved outs for that layer.

    Args:
        layer_to_validate_parents_for_label: label of the layer to check the saved outs
        perturb: whether to perturb the saved outs
        layers_to_perturb: layers for which to perturb the saved outs

    Returns:
        True if the outs match, False otherwise
    """
    if layers_to_perturb is None:
        layers_to_perturb = []

    layer = self[layer_to_validate_parents_for_label]

    # Early exits for layers that cannot or should not be replayed.

    # Input layers and buffer layers without parents have no function to replay
    # (func is None for model inputs and parentless buffers).
    if layer.func is None:
        return True

    # Registry 1: skip ALL validation for nondeterministic ops (e.g., empty_like).
    if layer.func_name in SKIP_VALIDATION_ENTIRELY:
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

    matches_saved = tensor_nanequal(recomputed_output, layer.out, allow_tolerance=True)
    if not matches_saved and not perturb and isinstance(recomputed_output, torch.Tensor):
        matches_saved = _deep_numeric_replay_matches_saved(layer, recomputed_output)

    # Forward replay failure (non-perturbed): saved outs don't match.
    if not matches_saved and not perturb:
        # Exemption: parent is an in-place RNG op that may have mutated its
        # tensor after the child logged it as an arg.
        parent_has_inplace_rng = any(
            self[p].func_name in ["bernoulli_", "full"] for p in layer.parents
        )
        if parent_has_inplace_rng:
            return True
        print(
            f"Saved outs for layer {layer_to_validate_parents_for_label} do not match the "
            f"values computed based on the parent layers {layer.parents}."
        )
        return False

    # Perturbation produced identical output -- run posthoc checks to see if
    # there's a valid excuse (bool output, special-value args, type cast, etc.).
    # Uses exact equality (no tolerance) since any change should be detectable.
    if perturb and tensor_nanequal(recomputed_output, layer.out, allow_tolerance=False):
        return posthoc_perturb_check(self, layer, layers_to_perturb, verbose)

    return True


def _prepare_input_args_for_validating_layer(
    self: "Trace",
    layer_to_validate_parents_for: Op,
    layers_to_perturb: List[str],
) -> dict[str, Any] | None:
    """Build the input argument dict for replaying a layer's function.

    Starts from the layer's saved ``saved_args`` / ``saved_kwargs``,
    deep-clones all tensors to prevent in-place mutation during replay, then
    swaps in each parent's saved out (or a perturbed version) at the
    correct argument position.

    For nested argument positions (tuples, dicts nested inside args), the
    ``parent_arg_positions`` key is a tuple ``(outer_key, inner_key)`` and
    ``assign_to_sequence_or_dict`` handles the nested assignment.

    Args:
        layer_to_validate_parents_for: Layer being checked.
        layers_to_perturb: Layers for which to perturb the saved outs.

    Returns:
        Dict with ``"args"`` and ``"kwargs"`` keys, or None if saved_args
        was not saved (can't validate without them).
    """
    if layer_to_validate_parents_for.saved_args is None:
        return None  # Can't validate without saved args (#131)
    input_args = {
        "args": list(layer_to_validate_parents_for.saved_args[:]),
        "kwargs": layer_to_validate_parents_for.saved_kwargs.copy(),
    }
    input_args = _copy_validation_args(input_args)

    # Swap in saved parent outs:

    for arg_type in ["args", "kwargs"]:
        for (
            key,
            parent_layer_arg,
        ) in layer_to_validate_parents_for.parent_arg_positions[arg_type].items():
            parent_layer = self[parent_layer_arg]
            target_op_label = getattr(layer_to_validate_parents_for, "label", None)
            if target_op_label in parent_layer.out_versions_by_child:
                parent_values = parent_layer.out_versions_by_child[target_op_label]
            elif layer_to_validate_parents_for.layer_label in parent_layer.out_versions_by_child:
                parent_values = parent_layer.out_versions_by_child[
                    layer_to_validate_parents_for.layer_label
                ]
            else:
                parent_values = parent_layer.out
            if parent_values is None:
                continue  # Skip validation for unsaved parents (#150)
            parent_values = parent_values.detach().clone()

            if parent_layer_arg in layers_to_perturb:
                parent_layer_func_values = _perturb_layer_outs(
                    parent_values, layer_to_validate_parents_for.out
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


def _copy_validation_args(input_args: dict[str, Any]) -> dict[str, Any]:
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


def _perturb_layer_outs(parent_outs: torch.Tensor, output_outs: torch.Tensor) -> torch.Tensor:
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
        parent_outs: The original parent tensor to perturb.
        output_outs: The child layer's output tensor, used to calibrate
            the perturbation scale for float types.

    Returns:
        A new tensor of the same shape/dtype with perturbed values.
    """
    device = parent_outs.device
    if parent_outs.numel() == 0:
        return parent_outs.detach().clone()

    if parent_outs.dtype in [
        torch.int,
        torch.long,
        torch.short,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        tensor_unique_vals = torch.unique(parent_outs)
        if len(tensor_unique_vals) > 1:
            # Multiple unique values: sample within the original range.
            perturbed_outs = parent_outs.detach().clone()
            for _ in range(MAX_PERTURB_ATTEMPTS):
                perturbed_outs = torch.randint(
                    parent_outs.min(),  # type: ignore[call-overload]
                    parent_outs.max() + 1,
                    size=parent_outs.shape,
                    device=device,
                ).type(parent_outs.dtype)
                if not torch.equal(perturbed_outs, parent_outs):
                    break
        else:
            # Single unique value: widen to a fixed range to guarantee a
            # different value can be produced.
            perturbed_outs = parent_outs.detach().clone()
            for _ in range(MAX_PERTURB_ATTEMPTS):
                if torch.min(parent_outs) < 0:
                    perturbed_outs = torch.randint(
                        -10, 11, size=parent_outs.shape, device=device
                    ).type(parent_outs.dtype)
                else:
                    perturbed_outs = torch.randint(
                        0, 11, size=parent_outs.shape, device=device
                    ).type(parent_outs.dtype)
                if not torch.equal(perturbed_outs, parent_outs):
                    break

    elif parent_outs.dtype == torch.bool:
        # Random bool, retried until different from original.
        perturbed_outs = parent_outs.detach().clone()
        for _ in range(MAX_PERTURB_ATTEMPTS):
            perturbed_outs = torch.randint(0, 2, size=parent_outs.shape, device=device).bool()
            if not torch.equal(perturbed_outs, parent_outs):
                break
    else:
        # Float/complex: uniform random within the original value range.
        # Using the original range ensures perturbed values stay in the
        # valid domain for range-restricted functions (e.g., bernoulli
        # requires probabilities in [0,1]).  For typical tensors with wide
        # range this produces meaningfully different values.
        if parent_outs.is_complex():
            real = parent_outs.real.float()
            imag = parent_outs.imag.float()
            r_lo, r_hi = real.min().item(), real.max().item()
            i_lo, i_hi = imag.min().item(), imag.max().item()
            if r_lo == r_hi:
                r_lo, r_hi = r_lo - 1.0, r_hi + 1.0
            if i_lo == i_hi:
                i_lo, i_hi = i_lo - 1.0, i_hi + 1.0
            perturbed_outs = torch.complex(
                torch.rand(parent_outs.shape, device=device) * (r_hi - r_lo) + r_lo,
                torch.rand(parent_outs.shape, device=device) * (i_hi - i_lo) + i_lo,
            ).type(parent_outs.dtype)
        else:
            lo = parent_outs.float().min().item()
            hi = parent_outs.float().max().item()
            if hi - lo < max(1e-6, abs(lo) * 1e-6):
                # Near-constant tensor — range is too narrow for meaningful
                # perturbation at float32 precision.  Expand by ±10% of
                # magnitude (or ±1 near zero).  Conservative to avoid feeding
                # invalid values to C extensions (e.g. ROI align segfaults).
                expansion = max(1.0, abs(lo) * 0.1)
                lo, hi = lo - expansion, hi + expansion
            perturbed_outs = torch.rand_like(parent_outs.float(), device=device) * (hi - lo) + lo
            perturbed_outs = perturbed_outs.type(parent_outs.dtype)

    return perturbed_outs

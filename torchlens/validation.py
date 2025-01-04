from collections import defaultdict
from typing import Any, Dict, List, Set, TYPE_CHECKING, Union

import torch

from .tensor_log import TensorLogEntry

if TYPE_CHECKING:
    pass

from .helper_funcs import (
    log_current_rng_states,
    set_rng_from_saved_states,
    tuple_tolerant_assign,
    tensor_nanequal,
    tensor_all_nan,
)

FUNCS_NOT_TO_PERTURB_IN_VALIDATION = [
    "expand_as",
    "new_zeros",
    "new_ones",
    "zero_",
    "copy_",
    "clamp",
    "fill_",
    "zeros_like",
    "ones_like",
]


def validate_saved_activations(
        self, ground_truth_output_tensors: List[torch.Tensor], verbose: bool = False
) -> bool:
    """Starting from outputs and internally terminated tensors, checks whether computing their values from the saved
    values of their input tensors yields their actually saved values, and whether computing their values from
    their parent tensors yields their saved values.

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
    validated_child_edges_for_each_layer = defaultdict(set)
    validated_layers = set(self.output_layers + self.internally_terminated_layers)
    layers_to_validate_parents_for = list(validated_layers)

    while len(layers_to_validate_parents_for) > 0:
        layer_to_validate_parents_for = layers_to_validate_parents_for.pop(0)
        parent_layers_valid = validate_parents_of_saved_layer(
            self,
            layer_to_validate_parents_for,
            validated_layers,
            validated_child_edges_for_each_layer,
            layers_to_validate_parents_for,
            verbose,
        )
        if not parent_layers_valid:
            return False

    if len(validated_layers) < len(self.layer_labels):
        print(
            f"All saved activations were accurate, but some layers were not reached (check that "
            f"child args logged accurately): {set(self.layer_labels) - validated_layers}"
        )
        return False

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
        layer_to_validate_parents_for_label:
        validated_layers:
        validated_child_edges_for_each_layer:
        layers_to_validate_parents_for:
        verbose: whether to print warning messages
    """
    layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]

    # Check that the arguments are logged correctly:
    if not _check_layer_arguments_logged_correctly(
            self, layer_to_validate_parents_for_label
    ):
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

    for perturb_layer in layer_to_validate_parents_for.parent_layers:
        if (
                layer_to_validate_parents_for.func_applied_name
                in FUNCS_NOT_TO_PERTURB_IN_VALIDATION
        ):
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
            if ((not parent_layer.is_input_layer) and
                    not (parent_layer.is_buffer_layer and (parent_layer.buffer_parent is None))):
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
        parent_layers_in_args.update(
            list(target_layer.parent_layer_arg_locs[arg_type].values())
        )
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
            for key, val in iterfunc(getattr(target_layer, argtype_field)):
                validation_correct_for_arg_and_layer = (
                    _validate_layer_against_arg(
                        self, target_layer, parent_layer, arg_type, key, val
                    )
                )
                if not validation_correct_for_arg_and_layer:
                    return False
    return True


def _validate_layer_against_arg(
        self, target_layer, parent_layer, arg_type, key, val
):
    if type(val) in [list, tuple]:
        for v, subval in enumerate(val):
            argloc_key = (key, v)
            validation_correct_for_arg_and_layer = (
                _check_arglocs_correct_for_arg(
                    self, target_layer, parent_layer, arg_type, argloc_key, subval
                )
            )
            if not validation_correct_for_arg_and_layer:
                return False

    elif type(val) == dict:
        for subkey, subval in val.items():
            argloc_key = (key, subkey)
            validation_correct_for_arg_and_layer = (
                _check_arglocs_correct_for_arg(
                    self, target_layer, parent_layer, arg_type, argloc_key, subval
                )
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
        target_layer: TensorLogEntry,
        parent_layer: TensorLogEntry,
        arg_type: str,
        argloc_key: Union[str, tuple],
        saved_arg_val: Any,
):
    """For a given layer and an argument to its child layer, checks that it is logged correctly:
    that is, that it's logged as an argument if it matches, and is not logged as an argument if it doesn't match.
    """
    target_layer_label = target_layer.layer_label
    parent_layer_label = parent_layer.layer_label
    if target_layer_label in parent_layer.children_tensor_versions:
        parent_activations = parent_layer.children_tensor_versions[
            target_layer_label
        ]
    else:
        parent_activations = parent_layer.tensor_contents

    if type(saved_arg_val) == torch.Tensor:
        parent_layer_matches_arg = tensor_nanequal(
            saved_arg_val, parent_activations, allow_tolerance=False
        )
    else:
        parent_layer_matches_arg = False
    parent_layer_logged_as_arg = (
                                         argloc_key in target_layer.parent_layer_arg_locs[arg_type]
                                 ) and (
                                         target_layer.parent_layer_arg_locs[arg_type][argloc_key]
                                         == parent_layer_label
                                 )

    if (
            parent_layer_matches_arg
            and (not parent_layer_logged_as_arg)
            and (parent_activations.numel() != 0)
            and (parent_activations.dtype != torch.bool)
            and (not tensor_all_nan(parent_activations))
            and (parent_activations.abs().float().mean() != 0)
            and (parent_activations.abs().float().mean() != 1)
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

    if (not parent_layer_matches_arg) and parent_layer_logged_as_arg:
        print(
            f"Parent {parent_layer_label} of {target_layer_label} is logged as {arg_type} {argloc_key} to "
            f"{target_layer_label}, but its saved activations don't match the saved argument."
        )
        return False

    return True


def _check_whether_func_on_saved_parents_yields_saved_tensor(
        self,
        layer_to_validate_parents_for_label: str,
        perturb: bool = False,
        layers_to_perturb: List[str] = None,
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

    layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]

    if (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "__getitem__")
            and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1],
    )
    ):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "__getitem__")
            and not torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[0],
    )
    ):
        return True
    elif layer_to_validate_parents_for.func_applied_name == 'empty_like':
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "__setitem__")
            and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
            and (layer_to_validate_parents_for.creation_args[1].dtype == torch.bool)
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1],
    )
    ):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "cross_entropy")
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1],
    )
    ):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "__setitem__")
            and (type(layer_to_validate_parents_for.creation_args[1]) == tuple)
            and (
                    type(layer_to_validate_parents_for.creation_args[1][0]) == torch.Tensor
            )
            and (layer_to_validate_parents_for.creation_args[1][0].dtype == torch.bool)
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1][0],
    )
    ):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "index_select")
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[2],
    )
    ):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "lstm")
            and (torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1][0]) or
                 torch.equal(
                     self[layers_to_perturb[0]].tensor_contents,
                     layer_to_validate_parents_for.creation_args[1][1]) or
                 torch.equal(
                     self[layers_to_perturb[0]].tensor_contents,
                     layer_to_validate_parents_for.creation_args[2][0]) or
                 torch.equal(
                     self[layers_to_perturb[0]].tensor_contents,
                     layer_to_validate_parents_for.creation_args[2][1]) or
                 ((type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor) and
                  torch.equal(
                      self[layers_to_perturb[0]].tensor_contents,
                      layer_to_validate_parents_for.creation_args[1])
                 ))):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "_pad_packed_sequence")
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1]
    )):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "masked_fill_")
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[1]
    )):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "scatter_")
            and torch.equal(
        self[layers_to_perturb[0]].tensor_contents,
        layer_to_validate_parents_for.creation_args[2]
    )):
        return True
    elif (
            perturb
            and (layer_to_validate_parents_for.func_applied_name == "interpolate")
            and ((('scale_factor' in layer_to_validate_parents_for.creation_kwargs)
                  and (layer_to_validate_parents_for.creation_kwargs['scale_factor'] is not None)
                  and torch.equal(
                self[layers_to_perturb[0]].tensor_contents,
                torch.tensor(layer_to_validate_parents_for.creation_kwargs['scale_factor'])))
                 or ((len(layer_to_validate_parents_for.creation_args) >= 3)
                     and torch.equal(
                        self[layers_to_perturb[0]].tensor_contents,
                        layer_to_validate_parents_for.creation_args[2])))
    ):
        return True

    # Prepare input arguments: keep the ones that should just be kept, perturb those that should be perturbed

    input_args = _prepare_input_args_for_validating_layer(
        self, layer_to_validate_parents_for, layers_to_perturb
    )

    # set the saved rng value:
    layer_func = layer_to_validate_parents_for.func_applied
    current_rng_states = log_current_rng_states()
    set_rng_from_saved_states(layer_to_validate_parents_for.func_rng_states)
    try:
        recomputed_output = layer_func(*input_args["args"], **input_args["kwargs"])
    except:
        raise Exception(f"Invalid perturbed arguments for layer {layer_to_validate_parents_for_label}")
    set_rng_from_saved_states(current_rng_states)

    if layer_func.__name__ in [
        "__setitem__",
        "zero_",
        "__delitem__",
    ]:  # TODO: fix this
        recomputed_output = input_args["args"][0]

    if any([issubclass(type(recomputed_output), which_type) for which_type in [list, tuple]]):
        recomputed_output = recomputed_output[
            layer_to_validate_parents_for.iterable_output_index
        ]

    if (
            not (
                    tensor_nanequal(
                        recomputed_output,
                        layer_to_validate_parents_for.tensor_contents,
                        allow_tolerance=True,
                    )
            )
            and not perturb
    ):
        print(
            f"Saved activations for layer {layer_to_validate_parents_for_label} do not match the "
            f"values computed based on the parent layers {layer_to_validate_parents_for.parent_layers}."
        )
        return False

    if (
            tensor_nanequal(
                recomputed_output,
                layer_to_validate_parents_for.tensor_contents,
                allow_tolerance=False,
            )
            and perturb
    ):
        return _posthoc_perturb_check(
            self, layer_to_validate_parents_for, layers_to_perturb, verbose
        )

    return True


def _prepare_input_args_for_validating_layer(
        self,
        layer_to_validate_parents_for: TensorLogEntry,
        layers_to_perturb: List[str],
) -> Dict:
    """Prepares the input arguments for validating the saved activations of a layer.

    Args:
        layer_to_validate_parents_for: Layer being checked.
        layers_to_perturb: Layers for which to perturb the saved activations.

    Returns:
        Dict of input arguments.
    """
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
            if (
                    layer_to_validate_parents_for.layer_label
                    in parent_layer.children_tensor_versions
            ):
                parent_values = parent_layer.children_tensor_versions[
                    layer_to_validate_parents_for.layer_label
                ]
            else:
                parent_values = parent_layer.tensor_contents
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
                input_args[arg_type][key[0]] = tuple_tolerant_assign(
                    input_args[arg_type][key[0]], key[1], parent_layer_func_values
                )

    return input_args


def _copy_validation_args(input_args: Dict):
    new_args = []
    for i, val in enumerate(input_args["args"]):
        if type(val) == torch.Tensor:
            new_args.append(val.detach().clone())
        elif type(val) in [list, tuple, set]:
            new_iter = []
            for i2, val2 in enumerate(val):
                if type(val2) == torch.Tensor:
                    new_iter.append(val2.detach().clone())
                else:
                    new_iter.append(val2)
            new_args.append(type(val)(new_iter))
        else:
            new_args.append(val)
    input_args["args"] = new_args

    new_kwargs = {}
    for key, val in input_args["kwargs"].items():
        if type(val) == torch.Tensor:
            new_kwargs[key] = val.detach().clone()
        elif type(val) in [list, tuple, set]:
            new_iter = []
            for i2, val2 in enumerate(val):
                if type(val2) == torch.Tensor:
                    new_iter.append(val2.detach().clone())
                else:
                    new_iter.append(val2)
            new_kwargs[key] = type(val)(new_iter)
        else:
            new_kwargs[key] = val
    input_args["kwargs"] = new_kwargs
    return input_args


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
            while torch.equal(perturbed_activations, parent_activations):
                perturbed_activations = torch.randint(
                    parent_activations.min(),
                    parent_activations.max() + 1,
                    size=parent_activations.shape,
                    device=device,
                ).type(parent_activations.dtype)
        else:
            perturbed_activations = parent_activations.detach().clone()
            while torch.equal(perturbed_activations, parent_activations):
                if torch.min(parent_activations) < 0:
                    perturbed_activations = torch.randint(
                        -10, 11, size=parent_activations.shape, device=device
                    ).type(parent_activations.dtype)
                else:
                    perturbed_activations = torch.randint(
                        0, 11, size=parent_activations.shape, device=device
                    ).type(parent_activations.dtype)

    elif parent_activations.dtype == torch.bool:
        perturbed_activations = parent_activations.detach().clone()
        while torch.equal(perturbed_activations, parent_activations):
            perturbed_activations = torch.randint(
                0, 2, size=parent_activations.shape, device=device
            ).bool()
    else:
        mean_output_sqrt = output_activations.detach().float().abs().mean()
        mean_output_sqrt += torch.rand(mean_output_sqrt.shape) * 100
        mean_output_sqrt *= torch.rand(mean_output_sqrt.shape)
        mean_output_sqrt.requires_grad = False
        perturbed_activations = torch.randn_like(
            parent_activations.float(), device=device
        ) * mean_output_sqrt.to(device)
        perturbed_activations = perturbed_activations.type(parent_activations.dtype)

    return perturbed_activations


def _posthoc_perturb_check(
        self,
        layer_to_validate_parents_for: TensorLogEntry,
        layers_to_perturb: List[str],
        verbose: bool = False,
) -> bool:
    """If a layer fails the "perturbation check"--that is, if perturbing the values of parent
    layers doesn't change the values relative to the layer's saved values--checks whether one of the
    remaining arguments is a "special" tensor, such as all-ones or all-zeros, such that perturbing a tensor
    wouldn't necessarily change the output of the layer.

    Args:
        layer_to_validate_parents_for: layer being checked.
        layers_to_perturb: parent layers being perturbed

    Returns:
        True if there's an "excuse" for the perturbation failing, False otherwise.
    """
    # Check if the tensor is all nans or all infinite:
    if layer_to_validate_parents_for.tensor_dtype == torch.bool:
        return True
    elif (
            (layer_to_validate_parents_for.func_applied_name == "to")
            and (len(layer_to_validate_parents_for.creation_args) > 1)
            and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
    ):
        return True
    elif (
            (layer_to_validate_parents_for.func_applied_name == "__setitem__")
            and (type(layer_to_validate_parents_for.creation_args[2]) == torch.Tensor)
            and (
                    layer_to_validate_parents_for.creation_args[0].shape
                    == layer_to_validate_parents_for.creation_args[2].shape
            )
    ):
        return True
    elif (
            layer_to_validate_parents_for.func_applied_name in ["__getitem__", "unbind"]
    ) and (
            layer_to_validate_parents_for.tensor_contents.numel() < 20
    ):  # some elements can be the same by chance
        return True
    elif (
            (layer_to_validate_parents_for.func_applied_name == "__getitem__")
            and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
            and (layer_to_validate_parents_for.creation_args[1].unique() < 20)
    ):
        return True
    elif (layer_to_validate_parents_for.func_applied_name == "max") and len(
            layer_to_validate_parents_for.creation_args
    ) > 1:
        return True
    elif (
            layer_to_validate_parents_for.func_applied_name == "max"
    ) and not torch.is_floating_point(
        layer_to_validate_parents_for.creation_args[0]
    ):
        return True
    else:
        num_inf = (
            torch.isinf(layer_to_validate_parents_for.tensor_contents.abs())
            .int()
            .sum()
        )
        num_nan = (
            torch.isnan(layer_to_validate_parents_for.tensor_contents.abs())
            .int()
            .sum()
        )
        if (num_inf == layer_to_validate_parents_for.tensor_contents.numel()) or (
                num_nan == layer_to_validate_parents_for.tensor_contents.numel()
        ):
            return True

    arg_type_dict = {
        "args": (enumerate, "creation_args"),
        "kwargs": (lambda x: x.items(), "creation_kwargs"),
    }

    layer_to_validate_parents_for_label = layer_to_validate_parents_for.layer_label
    for arg_type in ["args", "kwargs"]:
        iterfunc, fieldname = arg_type_dict[arg_type]
        for key, val in iterfunc(getattr(layer_to_validate_parents_for, fieldname)):
            # Skip if it's the argument itself:
            if (
                    key in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type]
            ) and (
                    layer_to_validate_parents_for.parent_layer_arg_locs[arg_type][key]
            ) in layers_to_perturb:
                continue
            arg_is_special = _check_if_arg_is_special_val(val)
            if arg_is_special:
                if verbose:
                    print(
                        f"Activations for layer {layer_to_validate_parents_for_label} do not change when "
                        f"values for {layers_to_perturb} are changed (out of parent "
                        f"layers {layer_to_validate_parents_for.parent_layers}), but {arg_type[:-1]} {key} is "
                        f"all zeros or all-ones, so validation still succeeds..."
                    )
                return True

    print(
        f"Activations for layer {layer_to_validate_parents_for_label} do not change when "
        f"values for {layers_to_perturb} are changed (out of parent "
        f"layers {layer_to_validate_parents_for.parent_layers}), and the other "
        f'arguments are not "special" (all-ones or all-zeros) tensors.'
    )
    return False


def _check_if_arg_is_special_val(val: Union[torch.Tensor, Any]):
    # If it's one of the other arguments, check if it's all zeros or all ones:
    if type(val) != torch.Tensor:
        try:
            val = torch.Tensor(val)
        except:
            return True
    if torch.all(torch.eq(val, 0)) or torch.all(torch.eq(val, 1)) or (val.numel() == 0):
        return True
    else:
        return False

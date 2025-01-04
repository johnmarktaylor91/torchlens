import random
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .model_history import ModelHistory

from .tensor_log import TensorLogEntry


def _getitem_during_pass(self: "ModelHistory", ix) -> TensorLogEntry:
    """Fetches an item when the pass is unfinished, only based on its raw barcode.

    Args:
        ix: layer's barcode

    Returns:
        Tensor log entry object with info about specified layer.
    """
    if ix in self._raw_tensor_dict:
        return self._raw_tensor_dict[ix]
    else:
        raise ValueError(f"{ix} not found in the ModelHistory object.")


def _getitem_after_pass(self, ix):
    """
    Overloaded such that entries can be fetched either by their position in the tensor log, their layer label,
    or their module address. It should say so and tell them which labels are valid.
    """
    if ix in self.layer_dict_all_keys:
        return self.layer_dict_all_keys[ix]

    keys_with_substr = [
        key for key in self.layer_dict_all_keys if str(ix) in str(key)
    ]
    if len(keys_with_substr) == 1:
        return self.layer_dict_all_keys[keys_with_substr[0]]

    self._give_user_feedback_about_lookup_key(ix, "get_one_item")


def _give_user_feedback_about_lookup_key(self, key: Union[int, str], mode: str):
    """For __getitem__ and get_op_nums_from_user_labels, gives the user feedback about the user key
    they entered if it doesn't yield any matches.

    Args:
        key: Lookup key used by the user.
    """
    if (type(key) == int) and (
            key >= len(self.layer_list) or key < -len(self.layer_list)
    ):
        raise ValueError(
            f"You specified the layer with index {key}, but there are only {len(self.layer_list)} "
            f"layers; please specify an index in the range "
            f"-{len(self.layer_list)} - {len(self.layer_list) - 1}."
        )

    if key in self.module_addresses:
        module_num_passes = self.module_num_passes[key]
        raise ValueError(
            f"You specified output of module {key}, but it has {module_num_passes} passes; "
            f"please specify e.g. {key}:2 for the second pass of {key}."
        )

    if key.split(":")[0] in self.module_addresses:
        module, pass_num = key.split(":")
        module_num_passes = self.module_num_passes[module]
        raise ValueError(
            f"You specified module {module} pass {pass_num}, but {module} only has "
            f"{module_num_passes} passes; specify a lower number."
        )

    if key in self.layer_labels_no_pass:
        layer_num_passes = self.layer_num_passes[key]
        raise ValueError(
            f"You specified output of layer {key}, but it has {layer_num_passes} passes; "
            f"please specify e.g. {key}:2 for the second pass of {key}."
        )

    if key.split(":")[0] in self.layer_labels_no_pass:
        layer_label, pass_num = key.split(":")
        layer_num_passes = self.layer_num_passes[layer_label]
        raise ValueError(
            f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
            f"{layer_num_passes} passes. Specify a lower number."
        )

    raise ValueError(self._get_lookup_help_str(key, mode))


def _str_after_pass(self) -> str:
    """Readable summary of the model history after the pass is finished.

    Returns:
        String summarizing the model.
    """
    s = f"Log of {self.model_name} forward pass:"

    # General info

    s += f"\n\tRandom seed: {self.random_seed_used}"
    s += (
        f"\n\tTime elapsed: {np.round(self.elapsed_time_total, 3)}s "
        f"({np.round(self.elapsed_time_torchlens_logging, 3)}s spent logging)"
    )

    # Overall model structure

    s += "\n\tStructure:"
    if self.model_is_recurrent:
        s += f"\n\t\t- recurrent (at most {self.model_max_recurrent_loops} loops)"
    else:
        s += "\n\t\t- purely feedforward, no recurrence"

    if self.model_is_branching:
        s += "\n\t\t- with branching"
    else:
        s += "\n\t\t- no branching"

    if self.model_has_conditional_branching:
        s += "\n\t\t- with conditional (if-then) branching"
    else:
        s += "\n\t\t- no conditional (if-then) branching"

    if len(self.buffer_layers) > 0:
        s += f"\n\t\t- contains {len(self.buffer_layers)} buffer layers"

    s += f"\n\t\t- {len(self.module_addresses)} total modules"

    # Model tensors:

    s += "\n\tTensor info:"
    s += (
        f"\n\t\t- {self.num_tensors_total} total tensors ({self.tensor_fsize_total_nice}) "
        f"computed in forward pass."
    )
    s += f"\n\t\t- {self.num_tensors_saved} tensors ({self.tensor_fsize_saved_nice}) with saved activations."

    # Model parameters:

    s += (
        f"\n\tParameters: {self.total_param_layers} parameter operations ({self.total_params} params total; "
        f"{self.total_params_fsize_nice})"
    )

    # Print the module hierarchy.
    s += "\n\tModule Hierarchy:"
    s += _module_hierarchy_str(self)

    # Now print all layers.
    s += "\n\tLayers"
    if self._all_layers_saved:
        s += " (all have saved activations):"
    elif self.num_tensors_saved == 0:
        s += " (no layer activations are saved):"
    else:
        s += " (* means layer has saved activations):"
    for layer_ind, layer_barcode in enumerate(self.layer_labels):
        pass_num = self.layer_dict_main_keys[layer_barcode].pass_num
        total_passes = self.layer_dict_main_keys[layer_barcode].layer_passes_total
        if total_passes > 1:
            pass_str = f" ({pass_num}/{total_passes} passes)"
        else:
            pass_str = ""

        if self.layer_dict_main_keys[layer_barcode].has_saved_activations and (
                not self._all_layers_saved
        ):
            s += "\n\t\t* "
        else:
            s += "\n\t\t  "
        s += f"({layer_ind}) {layer_barcode} {pass_str}"

    return s


def _str_during_pass(self) -> str:
    """Readable summary of the model history during the pass, as a debugging aid.

    Returns:
        String summarizing the model.
    """
    s = f"Log of {self.model_name} forward pass (pass still ongoing):"
    s += f"\n\tRandom seed: {self.random_seed_used}"
    s += f"\n\tInput tensors: {self.input_layers}"
    s += f"\n\tOutput tensors: {self.output_layers}"
    s += f"\n\tInternally initialized tensors: {self.internally_initialized_layers}"
    s += f"\n\tInternally terminated tensors: {self.internally_terminated_layers}"
    s += f"\n\tInternally terminated boolean tensors: {self.internally_terminated_bool_layers}"
    s += f"\n\tBuffer tensors: {self.buffer_layers}"
    s += "\n\tRaw layer labels:"
    for layer in self._raw_tensor_labels_list:
        s += f"\n\t\t{layer}"
    return s


def pretty_print_list_w_line_breaks(lst, indent_chars: str, line_break_every=5):
    """
    Utility function to pretty print a list with line breaks, adding indent_chars every line.
    """
    s = f"\n{indent_chars}"
    for i, item in enumerate(lst):
        s += f"{item}"
        if i < len(lst) - 1:
            s += ", "
        if ((i + 1) % line_break_every == 0) and (i < len(lst) - 1):
            s += f"\n{indent_chars}"
    return s


def _get_lookup_help_str(self, layer_label: Union[int, str], mode: str) -> str:
    """Generates a help string to be used in error messages when indexing fails."""
    sample_layer1 = random.choice(self.layer_labels_w_pass)
    sample_layer2 = random.choice(self.layer_labels_no_pass)
    if len(self.module_addresses) > 0:
        sample_module1 = random.choice(self.module_addresses)
        sample_module2 = random.choice(self.module_passes)
    else:
        sample_module1 = "features.3"
        sample_module2 = "features.4:2"
    module_str = f"(e.g., {sample_module1}, {sample_module2})"
    if mode == "get_one_item":
        msg = (
            "e.g., 'pool' will grab the maxpool2d or avgpool2d layer, 'maxpool' will grab the 'maxpool2d' "
            "layer, etc., but there must be only one such matching layer"
        )
    elif mode == "query_multiple":
        msg = (
            "e.g., 'pool' will grab all maxpool2d or avgpool2d layers, 'maxpool' will grab all 'maxpool2d' "
            "layers, etc."
        )
    else:
        raise ValueError("mode must be either get_one_item or query_multiple")
    help_str = (
        f"Layer {layer_label} not recognized; please specify either "
        f"\n\n\t1) an integer giving the ordinal position of the layer "
        f"(e.g. 2 for 3rd layer, -4 for fourth-to-last), "
        f"\n\t2) the layer label (e.g., {sample_layer1}, {sample_layer2}), "
        f"\n\t3) the module address {module_str}"
        f"\n\t4) A substring of any desired layer label ({msg})."
        f"\n\n(Label meaning: conv2d_3_4:2 means the second pass of the third convolutional layer, "
        f"and fourth layer overall in the model.)"
    )
    return help_str


def _module_hierarchy_str(self):
    """
    Utility function to print the nested module hierarchy.
    """
    s = ""
    for module_pass in self.top_level_module_passes:
        module, pass_num = module_pass.split(":")
        s += f"\n\t\t{module}"
        if self.module_num_passes[module] > 1:
            s += f":{pass_num}"
        s += _module_hierarchy_str_helper(self, module_pass, 1)
    return s


def _module_hierarchy_str_helper(self, module_pass, level):
    """
    Helper function for _module_hierarchy_str.
    """
    s = ""
    any_grandchild_modules = any(
        [
            len(self.module_pass_children[submodule_pass]) > 0
            for submodule_pass in self.module_pass_children[module_pass]
        ]
    )
    if any_grandchild_modules or len(self.module_pass_children[module_pass]) == 0:
        for submodule_pass in self.module_pass_children[module_pass]:
            submodule, pass_num = submodule_pass.split(":")
            s += f"\n\t\t{'    ' * level}{submodule}"
            if self.module_num_passes[submodule] > 1:
                s += f":{pass_num}"
            s += _module_hierarchy_str_helper(self, submodule_pass, level + 1)
    else:
        submodule_list = []
        for submodule_pass in self.module_pass_children[module_pass]:
            submodule, pass_num = submodule_pass.split(":")
            if self.module_num_passes[submodule] == 1:
                submodule_list.append(submodule)
            else:
                submodule_list.append(submodule_pass)
        s += pretty_print_list_w_line_breaks(
            submodule_list, line_break_every=8, indent_chars=f"\t\t{'    ' * level}"
        )
    return s


def print_all_fields(self):
    """Print all data fields for ModelHistory."""
    fields_to_exclude = [
        "layer_list",
        "layer_dict_main_keys",
        "layer_dict_all_keys",
        "raw_tensor_dict",
        "decorated_to_orig_funcs_dict",
    ]

    for field in dir(self):
        attr = getattr(self, field)
        if not any(
                [field.startswith("_"), field in fields_to_exclude, callable(attr)]
        ):
            print(f"{field}: {attr}")


def to_pandas(self) -> pd.DataFrame:
    """Returns a pandas dataframe with info about each layer.

    Returns:
        Pandas dataframe with info about each layer.
    """
    fields_for_df = [
        "layer_label",
        "layer_label_w_pass",
        "layer_label_no_pass",
        "layer_label_short",
        "layer_label_w_pass_short",
        "layer_label_no_pass_short",
        "layer_type",
        "layer_type_num",
        "layer_total_num",
        "layer_passes_total",
        "pass_num",
        "operation_num",
        "tensor_shape",
        "tensor_dtype",
        "tensor_fsize",
        "tensor_fsize_nice",
        "func_applied_name",
        "func_time_elapsed",
        "function_is_inplace",
        "gradfunc",
        "is_input_layer",
        "is_output_layer",
        "is_buffer_layer",
        "is_part_of_iterable_output",
        "iterable_output_index",
        "parent_layers",
        "has_parents",
        "orig_ancestors",
        "child_layers",
        "has_children",
        "output_descendents",
        "sibling_layers",
        "has_siblings",
        "spouse_layers",
        "has_spouses",
        "initialized_inside_model",
        "min_distance_from_input",
        "max_distance_from_input",
        "min_distance_from_output",
        "max_distance_from_output",
        "computed_with_params",
        "num_params_total",
        "parent_param_shapes",
        "parent_params_fsize",
        "parent_params_fsize_nice",
        "modules_entered",
        "modules_exited",
        "is_submodule_input",
        "is_submodule_output",
        "containing_module_origin",
        "containing_modules_origin_nested",
    ]

    fields_to_change_type = {
        "layer_type_num": int,
        "layer_total_num": int,
        "layer_passes_total": int,
        "pass_num": int,
        "operation_num": int,
        "function_is_inplace": bool,
        "is_input_layer": bool,
        "is_output_layer": bool,
        "is_buffer_layer": bool,
        "is_part_of_iterable_output": bool,
        "has_parents": bool,
        "has_children": bool,
        "has_siblings": bool,
        "has_spouses": bool,
        "computed_with_params": bool,
        "num_params_total": int,
        "parent_params_fsize": int,
        "tensor_fsize": int,
        "is_submodule_input": bool,
        "is_submodule_output": bool,
    }

    model_df_dictlist = []
    for tensor_entry in self.layer_list:
        tensor_dict = {}
        for field_name in fields_for_df:
            tensor_dict[field_name] = getattr(tensor_entry, field_name)
        model_df_dictlist.append(tensor_dict)
    model_df = pd.DataFrame(model_df_dictlist)

    for field in fields_to_change_type:
        model_df[field] = model_df[field].astype(fields_to_change_type[field])

    return model_df

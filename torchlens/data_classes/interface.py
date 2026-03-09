"""ModelLog query and display methods: __getitem__, __str__, to_pandas, print_all_fields.

These functions are defined here to keep ModelLog's class body small. They
are bound to ModelLog as class attributes via assignment at the bottom of
model_log.py (the "method importation" pattern).

**__getitem__ lookup logic** (``_getitem_after_pass``):

The lookup cascade for string keys after the pass is finished:

1. Exact match in ``layer_dict_all_keys`` (all lookup keys for every
   LayerPassLog, including pass-qualified labels like ``"conv2d_1_1:1"``).
2. Exact match in ``layer_logs`` (no-pass labels -> LayerLog aggregate).
3. Exact match in ``_module_logs`` (module address or pass label ->
   ModuleLog or ModulePassLog).
4. Case-insensitive exact match against all of the above.
5. Substring match: if exactly one layer label contains the given string,
   return it.  If multiple match, raise ValueError listing them.
6. If nothing matches, raise KeyError with a help message.

For integer keys: direct index into ``layer_list`` (supports negative indexing).
For slice keys: returns a list slice of ``layer_list``.
"""

import random
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .model_log import ModelLog

from .layer_pass_log import LayerPassLog


def _getitem_during_pass(self: "ModelLog", ix) -> LayerPassLog:
    """Fetches an item when the pass is unfinished, only based on its raw barcode.

    Args:
        ix: layer's barcode

    Returns:
        Tensor log entry object with info about specified layer.
    """
    if ix in self._raw_layer_dict:
        return self._raw_layer_dict[ix]
    else:
        raise ValueError(f"{ix} not found in the ModelLog object.")


def _getitem_after_pass(self, ix):
    """Multi-key lookup for ModelLog entries after postprocessing.

    Lookup cascade:
    1. slice -> list slice of layer_list
    2. exact match in layer_dict_all_keys (LayerPassLog by any lookup key)
    3. exact match in layer_logs (LayerLog by no-pass label)
    4. exact match in _module_logs (ModuleLog/ModulePassLog by address)
    5. case-insensitive exact match against all of the above
    6. substring match: unique match returns the layer; ambiguous raises ValueError
    7. fallback: KeyError with contextual help message

    Args:
        ix: int (ordinal), slice, or str (label/address/substring).

    Returns:
        LayerPassLog, LayerLog, ModuleLog, or ModulePassLog.

    Raises:
        KeyError: No match found.
        ValueError: Ambiguous substring match or invalid index.
    """
    if isinstance(ix, slice):
        return self.layer_list[ix]  # #78: slice indexing support

    # Step 2: exact match against all lookup keys (pass-qualified, short forms, etc.)
    if ix in self.layer_dict_all_keys:
        return self.layer_dict_all_keys[ix]

    # Step 3: no-pass label -> aggregate LayerLog
    if isinstance(ix, str) and hasattr(self, "layer_logs") and ix in self.layer_logs:
        return self.layer_logs[ix]

    # Step 4: module address or pass label -> ModuleLog/ModulePassLog
    if isinstance(ix, str) and hasattr(self, "_module_logs") and ix in self._module_logs:
        return self._module_logs[ix]

    # Step 5: case-insensitive exact match (#23)
    if isinstance(ix, str):
        lower_ix = ix.lower()
        for key in self.layer_dict_all_keys:
            if str(key).lower() == lower_ix:
                return self.layer_dict_all_keys[key]
        if hasattr(self, "layer_logs"):
            for key in self.layer_logs:
                if str(key).lower() == lower_ix:
                    return self.layer_logs[key]
        if hasattr(self, "_module_logs"):
            for key in self._module_logs._dict:
                if key.lower() == lower_ix:
                    return self._module_logs[key]

    # Step 6: substring match (case-insensitive)
    if not isinstance(ix, int):
        keys_with_substr = [
            key for key in self.layer_dict_all_keys if str(ix).lower() in str(key).lower()
        ]
        if len(keys_with_substr) == 1:
            return self.layer_dict_all_keys[keys_with_substr[0]]
        elif len(keys_with_substr) > 1:
            matches_str = ", ".join(str(k) for k in keys_with_substr[:10])
            suffix = (
                f" (and {len(keys_with_substr) - 10} more)" if len(keys_with_substr) > 10 else ""
            )
            raise ValueError(
                f"Ambiguous lookup: '{ix}' matches {len(keys_with_substr)} layers: "
                f"{matches_str}{suffix}. Please use a more specific key."
            )

    # Step 7: nothing matched — give a helpful error
    _give_user_feedback_about_lookup_key(self, ix, "get_one_item")
    raise KeyError(ix)


def _give_user_feedback_about_lookup_key(self, key: Union[int, str], mode: str):
    """For __getitem__ and get_op_nums_from_user_labels, gives the user feedback about the user key
    they entered if it doesn't yield any matches.

    Args:
        key: Lookup key used by the user.
        mode: Either "get_one_item" (raise error if key not found) or
            "query_multiple" (return empty list if no matches).
    """
    if (type(key) == int) and (key >= len(self.layer_list) or key < -len(self.layer_list)):
        raise ValueError(
            f"You specified the layer with index {key}, but there are only {len(self.layer_list)} "
            f"layers; please specify an index in the range "
            f"-{len(self.layer_list)} - {len(self.layer_list) - 1}."
        )

    if type(key) != str:
        raise ValueError(_get_lookup_help_str(self, key, mode))

    if hasattr(self, "_module_logs") and key.rsplit(":", 1)[0] in self._module_logs:
        module, pass_num = key.rsplit(":", 1)
        module_num_passes = self._module_logs[module].num_passes
        raise ValueError(
            f"You specified module {module} pass {pass_num}, but {module} only has "
            f"{module_num_passes} passes; specify a lower number."
        )

    if key in self.layer_labels_no_pass:
        layer_num_passes = self.layer_num_passes.get(key, 1)
        if layer_num_passes > 1:
            raise ValueError(
                f"You specified output of layer {key}, but it has {layer_num_passes} passes; "
                f"please specify e.g. {key}:2 for the second pass of {key}."
            )

    if key.rsplit(":", 1)[0] in self.layer_labels_no_pass:
        layer_label, pass_num = key.rsplit(":", 1)
        layer_num_passes = self.layer_num_passes.get(layer_label, "unknown")
        raise ValueError(
            f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
            f"{layer_num_passes} passes. Specify a lower number."
        )

    raise ValueError(_get_lookup_help_str(self, key, mode))


def _str_after_pass(self) -> str:
    """Readable summary of the model history after the pass is finished.

    Returns:
        String summarizing the model.
    """
    s = f"Log of {self.model_name} forward pass:"

    # General info

    s += f"\n\tRandom seed: {self.random_seed_used}"
    s += (
        f"\n\tTime elapsed: {np.round(self.time_total, 3)}s "
        f"({np.round(self.time_logging, 3)}s spent logging)"
    )

    # Overall model structure

    s += "\n\tStructure:"
    if self.is_recurrent:
        s += f"\n\t\t- recurrent (at most {self.max_recurrent_loops} loops)"
    else:
        s += "\n\t\t- purely feedforward, no recurrence"

    if self.is_branching:
        s += "\n\t\t- with branching"
    else:
        s += "\n\t\t- no branching"

    if self.has_conditional_branching:
        s += "\n\t\t- with conditional (if-then) branching"
    else:
        s += "\n\t\t- no conditional (if-then) branching"

    if len(self.buffer_layers) > 0:
        s += f"\n\t\t- contains {len(self.buffer_layers)} buffer layers"

    s += f"\n\t\t- {max(0, len(self.modules) - 1)} total modules"  # -1 to exclude root "self"

    # Model tensors:

    s += "\n\tTensor info:"
    s += (
        f"\n\t\t- {self.num_tensors_total} total tensors ({self.total_activation_memory_str}) "
        f"computed in forward pass."
    )
    s += f"\n\t\t- {self.num_tensors_saved} tensors ({self.saved_activation_memory_str}) with saved activations."

    # Model parameters:

    s += (
        f"\n\tParameters: {self.total_param_layers} parameter operations ({self.total_params} params total; "
        f"{self.total_params_memory_str})"
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
        total_passes = self.layer_dict_main_keys[layer_barcode].num_passes
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
    for layer in self._raw_layer_labels_list:
        s += f"\n\t\t{layer}"
    return s


def _format_list_with_line_breaks(lst, indent_chars: str, line_break_every=5) -> str:
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
    if not self.layer_labels_w_pass:
        sample_layer1 = "conv2d_1_1:1"
        sample_layer2 = "conv2d_1_1"
    else:
        sample_layer1 = random.choice(self.layer_labels_w_pass)
        sample_layer2 = random.choice(self.layer_labels_no_pass)
    module_addrs = [ml.address for ml in self.modules if ml.address != "self"]
    if len(module_addrs) > 0:
        sample_module1 = random.choice(module_addrs)
        all_pass_labels = [
            pl for ml in self.modules for pl in ml.pass_labels if ml.address != "self"
        ]
        sample_module2 = random.choice(all_pass_labels) if all_pass_labels else "features.4:2"
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


def _module_hierarchy_str(self) -> str:
    """Build a tree-formatted string of the module call hierarchy.

    Starts from the root module ("self") pass 1 and recursively descends
    through call_children.  Leaf-heavy subtrees (where no child has
    grandchildren) are printed on a single line for compactness.
    """
    s = ""
    root_pass = self.modules["self"].passes.get(1)
    if root_pass is None:
        return s
    for module_pass in root_pass.call_children:
        module, pass_num = module_pass.split(":")
        s += f"\n\t\t{module}"
        if self.modules[module].num_passes > 1:
            s += f":{pass_num}"
        s += _module_hierarchy_str_recursive(self, module_pass, 1)
    return s


def _module_hierarchy_str_recursive(self, module_pass, level) -> str:
    """Recursively format child modules at the given indentation level.

    If any child has grandchildren (deeper nesting), each child gets its
    own line with recursive expansion.  Otherwise, all children are
    printed compactly on one line with ``_format_list_with_line_breaks``.
    """
    s = ""
    module_pass_log = self.modules[module_pass]
    children = module_pass_log.call_children
    any_grandchild_modules = any(
        [len(self.modules[child_pass_label].call_children) > 0 for child_pass_label in children]
    )
    if any_grandchild_modules or len(children) == 0:
        for submodule_pass in children:
            submodule, pass_num = submodule_pass.split(":")
            s += f"\n\t\t{'    ' * level}{submodule}"
            if self.modules[submodule].num_passes > 1:
                s += f":{pass_num}"
            s += _module_hierarchy_str_recursive(self, submodule_pass, level + 1)
    else:
        submodule_list = []
        for submodule_pass in children:
            submodule, pass_num = submodule_pass.split(":")
            if self.modules[submodule].num_passes == 1:
                submodule_list.append(submodule)
            else:
                submodule_list.append(submodule_pass)
        s += _format_list_with_line_breaks(
            submodule_list, line_break_every=8, indent_chars=f"\t\t{'    ' * level}"
        )
    return s


def print_all_fields(self) -> None:
    """Print all data fields for ModelLog."""
    fields_to_exclude = [
        "layer_list",
        "layer_dict_main_keys",
        "layer_dict_all_keys",
        "raw_layer_dict",
        "decorated_to_orig_funcs_dict",
    ]

    for field in dir(self):
        attr = getattr(self, field)
        if not any([field.startswith("_"), field in fields_to_exclude, callable(attr)]):
            print(f"{field}: {attr}")


def to_pandas(self) -> pd.DataFrame:
    """Returns a pandas dataframe with info about each layer.

    Returns:
        Pandas dataframe with info about each layer.

    Raises:
        RuntimeError: If called before the forward pass is complete.
    """
    if not self._pass_finished:
        raise RuntimeError(
            "to_pandas() cannot be called before the forward pass is complete. "
            "Please wait until log_forward_pass has returned."
        )
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
        "num_passes",
        "pass_num",
        "operation_num",
        "tensor_shape",
        "tensor_dtype",
        "tensor_memory",
        "tensor_memory_str",
        "func_name",
        "func_time",
        "func_is_inplace",
        "grad_fn_name",
        "is_input_layer",
        "is_output_layer",
        "is_buffer_layer",
        "is_part_of_iterable_output",
        "iterable_output_index",
        "parent_layers",
        "has_parents",
        "root_ancestors",
        "child_layers",
        "has_children",
        "output_descendants",
        "sibling_layers",
        "has_siblings",
        "co_parent_layers",
        "has_co_parents",
        "is_internally_initialized",
        "min_distance_from_input",
        "max_distance_from_input",
        "min_distance_from_output",
        "max_distance_from_output",
        "uses_params",
        "num_params_total",
        "parent_param_shapes",
        "params_memory",
        "params_memory_str",
        "modules_entered",
        "modules_exited",
        "is_submodule_input",
        "is_submodule_output",
        "containing_module",
        "containing_modules",
    ]

    fields_to_change_type = {
        "layer_type_num": int,
        "layer_total_num": int,
        "num_passes": int,
        "pass_num": int,
        "operation_num": int,
        "func_is_inplace": bool,
        "is_input_layer": bool,
        "is_output_layer": bool,
        "is_buffer_layer": bool,
        "is_part_of_iterable_output": bool,
        "has_parents": bool,
        "has_children": bool,
        "has_siblings": bool,
        "has_co_parents": bool,
        "uses_params": bool,
        "num_params_total": int,
        "params_memory": int,
        "tensor_memory": int,
        "is_submodule_input": bool,
        "is_submodule_output": bool,
    }

    model_df_dictlist = []
    for layer_entry in self.layer_list:
        layer_dict = {}
        for field_name in fields_for_df:
            layer_dict[field_name] = getattr(layer_entry, field_name)
        model_df_dictlist.append(layer_dict)
    model_df = pd.DataFrame(model_df_dictlist)

    for field in fields_to_change_type:
        model_df[field] = model_df[field].astype(fields_to_change_type[field])

    return model_df

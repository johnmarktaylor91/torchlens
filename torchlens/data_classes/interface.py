"""ModelLog query and display helpers used by ``ModelLog`` methods.

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

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from .model_log import ModelLog

from ._lookup_keys import _give_user_feedback_about_lookup_key
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


def _format_conditional_branch_stack(conditional_branch_stack: List[Tuple[int, str]]) -> str:
    """Render a compact string form for a conditional branch stack.

    Args:
        conditional_branch_stack: Outer-to-inner ``(cond_id, branch_kind)`` pairs.

    Returns:
        Compact string form, or an empty string when the stack is empty.
    """
    return ",".join(
        f"cond_{conditional_id}:{branch_kind}"
        for conditional_id, branch_kind in conditional_branch_stack
    )

"""Trace query and display helpers used by ``Trace`` custom_methods.

**__getitem__ lookup logic** (``_getitem_after_pass``):

The lookup cascade for string keys after the pass is finished:

1. Exact match in ``layer_logs`` (no-pass labels -> Layer aggregate).
2. Exact match in ``layer_dict_all_keys`` (all lookup keys for every
   Op, including pass-qualified labels like ``"conv2d_1_1:1"``).
3. Exact match in ``_module_logs`` (module address or pass label ->
   Module or ModuleCall).
4. Case-insensitive exact match against all of the above.
5. Substring match: if exactly one layer label contains the given string,
   return it.  If multiple match, raise ValueError listing them.
6. If nothing matches, raise KeyError with a help message.

For integer keys: direct index into ``layer_list`` (supports negative indexing).
For slice keys: returns a list slice of ``layer_list``.
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..capture.projections import LiveOpView
    from .trace import Trace

from .._errors import AmbiguousOpLookupError, InvalidArgumentError
from ..capture.projections import LiveOpView
from ..intervention.errors import SiteAmbiguityError
from ..intervention.selectors import BaseSelector
from ..intervention.types import FrozenTargetSpec, TargetSpec
from ._lookup_keys import _give_user_feedback_about_lookup_key
from .op import Op


def _ambiguous_lookup_match_labels(self: "Trace", key: str) -> list[str]:
    """Return final Op labels that share an ambiguous lookup key.

    Parameters
    ----------
    self:
        Trace containing the ambiguity registry.
    key:
        Ambiguous lookup key.

    Returns
    -------
    list[str]
        Final pass-qualified Op labels for matching entries.
    """

    ambiguous_lookup_keys = getattr(self, "_ambiguous_lookup_keys", {})
    raw_indices = ambiguous_lookup_keys.get(key, [])
    matches: list[str] = []
    for op in self.layer_list:
        if op.raw_index in raw_indices:
            matches.append(op.label)
    return matches


def _raise_ambiguous_lookup_key(self: "Trace", requested_key: str, stored_key: str) -> None:
    """Raise the strict ambiguous lookup error for a colliding alias key.

    Parameters
    ----------
    self:
        Trace containing the ambiguity registry.
    requested_key:
        User-supplied lookup key.
    stored_key:
        Canonical key stored in the ambiguity registry.
    """

    matches = _ambiguous_lookup_match_labels(self, stored_key)
    raise AmbiguousOpLookupError(
        f"Ambiguous lookup key {requested_key!r} matches {len(matches)} ops: "
        f"{', '.join(matches[:10])}{'...' if len(matches) > 10 else ''}. "
        "Use an exact raw/op label or a more specific key."
    )


def _getitem_during_pass(self: "Trace", ix: Any) -> Op | LiveOpView:
    """Fetches an item when the pass is unfinished, only based on its raw barcode.

    Args:
        ix: layer's barcode

    Returns:
        Tensor log entry object with info about specified layer.
    """
    capture_events = getattr(self, "capture_events", None)
    if capture_events is not None and ix in capture_events.live_index.by_raw_label:
        return LiveOpView(self, capture_events.live_index.require_event(ix))
    if (capture_events is None or not getattr(capture_events, "op_events", ())) and (
        ix in self._raw_graph_ws.raw_layer_dict
    ):
        return self._raw_graph_ws.raw_layer_dict[ix]
    raise InvalidArgumentError(
        f"{ix!r} is not a known raw label during this forward pass; final labels are not yet built",
        code="op_lookup_not_found",
        remedy="use a raw label seen this forward pass, or look up after trace() returns",
        key=repr(ix),
    )


def _getitem_after_pass(self: "Trace", ix: Any) -> Any:
    """Universal lookup for Trace entries after postprocessing.

    Lookup cascade:
    1. slice -> list slice of layer_list
    2. int -> Op by 0-based ordinal position
    3. string -> ops, module_calls, layers, modules, params, buffers, grad_fns
    4. alternate Op lookup keys
    5. case-insensitive exact match against alternate Op lookup keys
    6. substring match against alternate Op lookup keys
    7. fallback: KeyError with contextual help message

    Args:
        ix: int (ordinal), slice, or str (label/address/substring).

    Returns:
        Op, Layer, Module, or ModuleCall.

    Raises:
        KeyError: No match found.
        ValueError: Ambiguous substring match or invalid index.
    """
    if isinstance(ix, BaseSelector | TargetSpec | FrozenTargetSpec):
        from ..intervention.resolver import resolve_sites

        table = resolve_sites(self, ix, max_fanout=1)
        if len(table) != 1:
            raise SiteAmbiguityError(
                f"site {ix!r} matched {len(table)} sites, but Trace.__getitem__ requires one."
            )
        return table.first()

    if isinstance(ix, slice):
        return self.layer_list[ix]  # #78: slice indexing support

    if isinstance(ix, int):
        try:
            return self.ops[ix]
        except IndexError:
            _give_user_feedback_about_lookup_key(self, ix, "get_one_item")
            raise

    if isinstance(ix, str):
        if ix in self.layer_logs:
            return self.layer_logs[ix]

        ambiguous_lookup_keys = getattr(self, "_ambiguous_lookup_keys", {})
        if ix in ambiguous_lookup_keys:
            _raise_ambiguous_lookup_key(self, ix, ix)

        if ix in self.layer_dict_all_keys:
            return self.layer_dict_all_keys[ix]

        for accessor in (
            self.ops,
            self.module_calls,
            self.layers,
            self.modules,
            self.params,
            self.buffers,
            self.grad_fns,
        ):
            try:
                return accessor[ix]
            except (AttributeError, KeyError, ValueError, TypeError):
                pass

        lower_ix = ix.lower()
        for accessor in (
            self.ops,
            self.module_calls,
            self.layers,
            self.modules,
            self.params,
            self.buffers,
            self.grad_fns,
        ):
            try:
                for key in accessor.keys():
                    if str(key).lower() == lower_ix:
                        return accessor[key]
            except (AttributeError, KeyError, ValueError, TypeError):
                pass

        for key in self.layer_dict_all_keys:
            if str(key).lower() == lower_ix:
                if key in ambiguous_lookup_keys:
                    _raise_ambiguous_lookup_key(self, ix, key)
                return self.layer_dict_all_keys[key]

        keys_with_substr = [
            key for key in self.layer_dict_all_keys if str(ix).lower() in str(key).lower()
        ]
        entries_with_substr = {
            self.layer_dict_all_keys[key].raw_index: self.layer_dict_all_keys[key]
            for key in keys_with_substr
        }
        if len(entries_with_substr) == 1:
            return next(iter(entries_with_substr.values()))
        elif len(entries_with_substr) > 1:
            matches = [entry.layer_label for entry in entries_with_substr.values()]
            matches_str = ", ".join(str(k) for k in matches[:10])
            suffix = (
                f" (and {len(entries_with_substr) - 10} more)"
                if len(entries_with_substr) > 10
                else ""
            )
            raise AmbiguousOpLookupError(
                f"Ambiguous lookup: '{ix}' matches {len(entries_with_substr)} layers: "
                f"{matches_str}{suffix}. Please use a more specific key."
            )

    # Step 7: nothing matched — give a helpful error
    _give_user_feedback_about_lookup_key(self, ix, "get_one_item")
    raise KeyError(ix)


def _str_after_pass(self: "Trace") -> str:
    """Readable summary of the model history after the pass is finished.

    Returns:
        String summarizing the model.
    """
    s = f"Log of {self.model_class_name} forward pass:"

    # General info

    s += f"\n\tRandom seed: {self.random_seed}"
    s += f"\n\tTime elapsed: {self.capture_duration} ({self.overhead_duration} spent logging)"

    # Overall model structure

    s += "\n\tStructure:"
    if self.is_recurrent:
        s += f"\n\t\t- recurrent (at most {self.max_layer_op_count} loops)"
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
        f"\n\t\t- {self.num_tensors} total tensors ({self.total_activation_memory}) "
        f"computed in forward pass."
    )
    s += f"\n\t\t- {self.num_saved_ops} tensors ({self.saved_activation_memory}) with saved outs."
    nonfinite = self.first_nonfinite()
    if not nonfinite.startswith("No non-finite"):
        s += f"\n\t\t- NaN/Inf: {nonfinite}"

    # Model parameters:

    s += (
        f"\n\tParameters: {self.num_layers_with_params} parameter operations ({self.num_params} params total; "
        f"{self.total_param_memory})"
    )
    s += "\n\tFLOP convention: MACs are reported as FLOPs // 2."

    # Print the module hierarchy.
    s += "\n\tModule Hierarchy:"
    s += _module_hierarchy_str(self)

    # Now print all layers.
    s += "\n\tLayers"
    if self._layers_saved:
        s += " (all have saved outs):"
    elif self.num_saved_ops == 0:
        s += " (no layer outs are saved):"
    else:
        s += " (* means layer has saved outs):"
    for layer_ind, layer_entry in enumerate(self.layer_list):
        layer_barcode = layer_entry.layer_label
        pass_index = layer_entry.pass_index
        num_passes = layer_entry.num_passes
        if num_passes > 1:
            # One pass/op vocabulary (A10): a multi-pass row is "pass k/N",
            # never "k/N ops" -- "op" is reserved for the executed-op
            # denominator (Trace.num_ops).
            pass_str = f" (pass {pass_index}/{num_passes})"
        else:
            pass_str = ""

        if layer_entry.has_saved_activation and (not self._layers_saved):
            s += "\n\t\t* "
        else:
            s += "\n\t\t  "
        s += f"({layer_ind}) {layer_barcode} {pass_str}"

    return s


def _str_during_pass(self: "Trace") -> str:
    """Readable summary of the model history during the pass, as a debugging aid.

    Returns:
        String summarizing the model.
    """
    s = f"Log of {self.model_class_name} forward pass (pass still ongoing):"
    s += f"\n\tRandom seed: {self.random_seed}"
    s += f"\n\tInput tensors: {self.input_layers}"
    s += f"\n\tOutput tensors: {self.output_layers}"
    s += f"\n\tInternally initialized tensors: {self.internal_source_ops}"
    s += f"\n\tInternally terminated tensors: {self.internal_sink_ops}"
    s += f"\n\tInternally terminated boolean tensors: {self.internally_terminated_bool_ops}"
    s += f"\n\tBuffer tensors: {self.buffer_layers}"
    s += "\n\tRaw layer labels:"
    capture_events = getattr(self, "capture_events", None)
    labels = (
        [event.label_raw for event in capture_events.op_events]
        if capture_events is not None
        else self._raw_graph_ws.raw_layer_labels_list
    )
    for layer in labels:
        s += f"\n\t\t{layer}"
    return s


def _format_list_with_line_breaks(
    lst: list[Any], indent_chars: str, line_break_every: int = 5
) -> str:
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


def _module_hierarchy_str(self: "Trace") -> str:
    """Build a tree-formatted string of the module call hierarchy.

    Starts from the root module ("self") pass 1 and recursively descends
    through call_children.  Leaf-heavy subtrees (where no child has
    grandchildren) are printed on a single line for compactness.
    """
    s = ""
    root_module = cast(Any, self.modules["self"])
    root_pass = root_module.ops.get(1)
    if root_pass is None:
        return s
    for module_pass in root_pass.call_children:
        module, call_index = module_pass.rsplit(":", 1)
        s += f"\n\t\t{module}"
        if cast(Any, self.modules[module]).num_calls > 1:
            s += f":{call_index}"
        s += _module_hierarchy_str_recursive(self, module_pass, 1)
    return s


def _module_hierarchy_str_recursive(
    self: "Trace",
    module_pass: str,
    level: int,
    _in_progress: set[str] | None = None,
) -> str:
    """Recursively format child modules at the given indentation level.

    If any child has grandchildren (deeper nesting), each child gets its
    own line with recursive expansion.  Otherwise, all children are
    printed compactly on one line with ``_format_list_with_line_breaks``.

    Bounded display walk (r-b4 R27-5): a malformed/cyclic rehydrated
    ``call_children`` relationship renders a ``<cycle>`` marker, and nesting
    past 200 levels renders ``<max-depth>``, instead of crashing the display
    path with a raw ``RecursionError``.
    """
    if _in_progress is None:
        _in_progress = set()
    if level > 200:
        return f"\n\t\t{'    ' * level}<max-depth>"
    if module_pass in _in_progress:
        return f"\n\t\t{'    ' * level}<cycle>"
    _in_progress.add(module_pass)
    try:
        return _module_hierarchy_str_children(self, module_pass, level, _in_progress)
    finally:
        _in_progress.discard(module_pass)


def _module_hierarchy_str_children(
    self: "Trace",
    module_pass: str,
    level: int,
    _in_progress: set[str],
) -> str:
    """Format one guarded module call's children (body of the above)."""
    s = ""
    module_call_log = self.module_calls[module_pass]
    children = module_call_log.call_children
    any_grandchild_modules = any(
        len(self.module_calls[child_call_label].call_children) > 0 for child_call_label in children
    )
    if any_grandchild_modules or len(children) == 0:
        for submodule_pass in children:
            submodule, call_index = submodule_pass.rsplit(":", 1)
            s += f"\n\t\t{'    ' * level}{submodule}"
            if cast(Any, self.modules[submodule]).num_calls > 1:
                s += f":{call_index}"
            s += _module_hierarchy_str_recursive(self, submodule_pass, level + 1, _in_progress)
    else:
        submodule_list = []
        for submodule_pass in children:
            submodule, call_index = submodule_pass.rsplit(":", 1)
            if cast(Any, self.modules[submodule]).num_calls == 1:
                submodule_list.append(submodule)
            else:
                submodule_list.append(submodule_pass)
        s += _format_list_with_line_breaks(
            submodule_list, line_break_every=8, indent_chars=f"\t\t{'    ' * level}"
        )
    return s


def _format_conditional_branch_stack(conditional_branch_stack: list[tuple[int, str]]) -> str:
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

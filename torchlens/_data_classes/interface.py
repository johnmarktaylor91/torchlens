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

import warnings
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .._capture.projections import LiveOpView
    from .trace import Trace

from .._capture.projections import LiveOpView
from .._errors import AmbiguousOpLookupError, InvalidArgumentError
from .._pass_tokens import PASS_TOKEN_ALL, split_pass_token
from .._selection.errors import AmbiguousSiteError
from .._selection.selectors import BaseSelector
from .._selection.target import FrozenTargetSpec, TargetSpec
from .._utils._display_errors import ambiguous_substring_message, candidate_keys, lookup_message
from . import _doors
from ._lookup_keys import _give_user_feedback_about_lookup_key
from ._lookup_kinds import named_record
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
    for op in self._layer_list:
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
        lookup_message(
            f"Ambiguous lookup key {requested_key!r} matches {len(matches)} ops.",
            candidate_keys(matches, self._layer_dict_all_keys),
            "Use an exact op label or a more specific key.",
            query=requested_key,
        )
    )


def _getitem_during_pass(self: "Trace", ix: Any) -> Op | LiveOpView:
    """Fetches an item when the pass is unfinished, only based on its raw barcode.

    Args:
        ix: layer's barcode

    Returns:
        Tensor log entry object with info about specified layer.
    """
    # Instance-dict read: ``capture_events`` is a session field that is absent on
    # most torch captures, and ``getattr`` with a default paid the Trace miss
    # path (a formatted AttributeError) about eleven times per layer.
    capture_events = self.__dict__.get("capture_events")
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


def _names_one_thing(selector: Any) -> bool:
    """Return whether a selector spelling addresses exactly one site.

    A target spec, a frozen target spec, an exact-label selector and a facet
    selector each name one thing; every other selector names a set.
    """

    from .._selection.selectors import FacetSelector

    if isinstance(selector, TargetSpec | FrozenTargetSpec | FacetSelector):
        return True
    if getattr(selector, "selector_kind", None) != "label":
        return False
    token = split_pass_token(str(getattr(selector, "selector_value", "")))
    return token is None or token[1] != PASS_TOKEN_ALL


def _getitem_selector(self: "Trace", ix: Any) -> Any:
    """Resolve ``trace[selector]``: one record for a one-thing spelling, else a Selection.

    A selector that names a set (``tl.func``, ``tl.in_module``, ``tl.where``,
    ``tl.label_contains`` and their compositions) returns a ``tl.Selection``
    of every matching site, never capped; its ``repr`` and ``len`` report the
    count, and an empty result is a legal empty Selection. A spelling that
    names one thing (an exact label, a target spec, a facet selector) must
    match exactly one site and returns that record.

    Raises
    ------
    AmbiguousSiteError
        ``code="site_spelling_not_unique"`` when a one-thing spelling matches
        several sites.
    """

    from .._selection.module_alias_guard import refuse_trace_alias_spellings
    from .._selection.resolver import _guard_episode_step_query, _normalize_query, _resolve_sites
    from .._selection.selection import _selection_from_selector
    from ..intervention.errors import MultiMatchWarning

    if not _names_one_thing(ix):
        normalized = _normalize_query(ix)
        _guard_episode_step_query(self, normalized)
        refuse_trace_alias_spellings(self, normalized)
        return _selection_from_selector(normalized)._resolve(self)
    with warnings.catch_warnings():
        # The count is reported by the typed refusal below, not by a fan-out warning.
        warnings.simplefilter("ignore", MultiMatchWarning)
        table = _resolve_sites(self, ix, max_fanout=max(1, len(self._layer_list)))
    if len(table) != 1:
        raise AmbiguousSiteError(
            f"{ix!r} names one site but matched {len(table)} on this trace.",
            code="site_spelling_not_unique",
            remedy=(
                "address one site exactly (a pass-qualified label), or use a set-naming "
                "selector (tl.func, tl.in_module, tl.label_contains) to get every match "
                "as a Selection"
            ),
            matched=len(table),
        )
    return table[0]


def _getitem_after_pass(self: "Trace", ix: Any) -> Any:
    """Universal lookup for Trace entries after postprocessing.

    Lookup cascade:
    1. slice -> list slice of layer_list
    2. int -> Op by 0-based ordinal position
    3. ``label:N`` / ``label:last`` / ``label:all`` -> that pass's Op / Selection of every pass
    4. bare layer label -> its Layer, single- or multi-pass
    5. the kind the key names: module address -> Module, ``addr:N`` -> ModuleCall,
       parameter address -> Param, buffer address -> Buffer (``addr:N`` -> version Op)
    6. accessors, alternate Op lookup keys, case-insensitive, then substring
    7. fallback: KeyError with contextual help message

    Args:
        ix: int (ordinal), slice, or str (label/address/substring).

    Returns:
        Op, Layer, Selection, Module, ModuleCall, Param, Buffer or GradFn.

    Raises:
        KeyError: No match found.
        AmbiguousOpLookupError: A substring matching several layers.
        ValueError: Invalid index.
    """
    from .._selection.selection import Selection, Selector

    if isinstance(ix, Selector | Selection):
        return ix._resolve(self)

    if isinstance(ix, BaseSelector | TargetSpec | FrozenTargetSpec):
        return _getitem_selector(self, ix)

    if isinstance(ix, slice):
        return self._layer_list[ix]  # #78: slice indexing support

    if isinstance(ix, int):
        try:
            return self._layer_list[ix]  # every row, like the slice form above
        except IndexError:
            _give_user_feedback_about_lookup_key(self, ix, "get_one_item")
            raise

    if isinstance(ix, str):
        from .._trace_selector_helpers import canonical_module_site_key

        ix = canonical_module_site_key(self, ix) or ix
        if (record := _layer_label_lookup(self, ix)) is not None:
            return record

        ambiguous_lookup_keys = getattr(self, "_ambiguous_lookup_keys", {})
        if ix in ambiguous_lookup_keys:
            _raise_ambiguous_lookup_key(self, ix, ix)

        if (record := named_record(self, ix)) is not None:  # the kind the key names
            return record

        if ix in self._layer_dict_all_keys:
            return self._layer_dict_all_keys[ix]

        accessors = (
            self.ops,
            self.modules,
            self.module_calls,
            self.params,
            self.buffers,
            self.autograd.grad_fns,
        )
        for accessor in accessors:
            try:
                return accessor[ix]
            except (AttributeError, KeyError, ValueError, TypeError):
                pass

        lower_ix = ix.lower()
        for accessor in accessors:
            try:
                # ``ops.keys()`` lists the counted ops only; a
                # case-insensitive label still reaches every row.
                keys = (
                    [op.label for op in self._layer_list]
                    if accessor is self.ops
                    else accessor.keys()
                )
                for key in keys:
                    if str(key).lower() == lower_ix:
                        return accessor[key]
            except (AttributeError, KeyError, ValueError, TypeError):
                pass

        for key in self._layer_dict_all_keys:
            if str(key).lower() == lower_ix:
                if key in ambiguous_lookup_keys:
                    _raise_ambiguous_lookup_key(self, ix, key)
                return self._layer_dict_all_keys[key]

        keys_with_substr = [
            key for key in self._layer_dict_all_keys if str(ix).lower() in str(key).lower()
        ]
        entries_with_substr = {
            self._layer_dict_all_keys[key].raw_index: self._layer_dict_all_keys[key]
            for key in keys_with_substr
        }
        if len(entries_with_substr) == 1:
            return next(iter(entries_with_substr.values()))
        elif len(entries_with_substr) > 1:
            raise _ambiguous_substring_error(ix, list(entries_with_substr.values()))

    # Step 7: nothing matched — give a helpful error
    _give_user_feedback_about_lookup_key(self, ix, "get_one_item")
    raise KeyError(ix)


def _ambiguous_substring_error(ix: Any, records: list[Any]) -> AmbiguousOpLookupError:
    """The substring refusal card."""

    return AmbiguousOpLookupError(ambiguous_substring_message(str(ix), records))


def _layer_label_lookup(self: "Trace", ix: str) -> Any:
    """Resolve a layer label, bare or pass-tokened, to its record; ``None`` when ``ix`` names no layer.

    A bare layer label names the Layer, whether it ran once or several times;
    a single-pass Layer passes single-op reads (``.out``, ``.shape``) through to
    its Op, and a multi-pass one refuses them naming ``label:N`` and
    ``label:last``. ``label:N`` and ``label:last`` return that pass's Op and
    ``label:all`` the Selection of every pass. An op key that is not a layer
    label (a raw or alternate Op spelling) returns its Op.
    """

    from .._selection.selectors import LabelSelector

    token = split_pass_token(ix)
    base = ix if token is None else token[0]
    ops = self.ops.resolve_all(base)
    if not ops:
        return None
    if token is None:
        named = {op.layer_label for op in ops if ix in {op.layer_label, op.short_layer_label}}
        if len(named) == 1:
            layer = self._layer_logs.get(next(iter(named)))
            if layer is not None:
                return layer
        if len(ops) == 1:
            return ops[0]
    return _getitem_selector(self, LabelSelector(ix))


def _str_after_pass(self: "Trace") -> str:
    """Bounded Trace card; the module tree remains in ``trace.summary()``."""

    return _doors.report()._trace_summary._trace_text(self)


def _str_during_pass(self: "Trace") -> str:
    """The same bounded card while capture is in progress."""

    return _doors.report()._trace_summary._trace_text(self)


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

    Bounded display walk: a malformed/cyclic rehydrated
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

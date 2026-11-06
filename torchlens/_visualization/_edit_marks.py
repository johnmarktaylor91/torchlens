"""Edit marks: how a forward picture shows an intervention.

An edited node keeps its fill, shape and dash and gets a purple ring plus a
last label row naming the edit (``Δ zero_ablate``, ``Δ scale(0.5)``); the
header gets one line naming the edits (``1 edit: Δ zero_ablate on
linear_1_1``). Nodes changed downstream of the edit are tagged with an SVG
class only; the translucent highlighter under them is drawn after layout
(:mod:`._edit_highlighter`) and is off unless ``vis_show_cone=True``.

The edit text rule: the helper name alone when the edit takes no arguments,
short literal arguments written in, and tensors, callables and traces as a
short stand-in (``v``, the function's name or ``fn``, the trace's name or
``trace``). Several edits at one node take one row each in firing order, at
most three, then ``+N more``. The text comes from the op's fire records when
any fired, else from the staged recipe entries that resolve to the site.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from .._data_classes._module_edit_sites import REPLACEMENT_FUNC_NAME, replaced_site
from .._data_classes._trace_missing_attrs import is_live_trace
from .._selection.resolver import _resolve_sites
from .._utils._multipass_access import get_multipass_attr
from .._vocab.node_spec import EDITED_NODE_CLASS, INTERVENTION_SITE_COLOR
from ..intervention._edit_repr import _is_signature_default, render_edit_repr
from ..intervention._helpers import builtin_helper_constructor
from ._render_names import _SPEC_SLOT_BOX_OPS
from ._render_utils import html_escape
from ._typography import EDIT_GAP_SIZE, format_pt
from .node_spec import (
    _forward_site,
    _with_svg_class,
    intervention_graph_override,
    intervention_sites_for_log,
)
from .themes import _relative_luminance

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .._vocab.node_spec import CollapsedNodeSpecFn, NodeSpec
    from ..intervention._types import Edit, FireRecord
    from .themes import VisualizationTheme

__all__ = [
    "DELTA",
    "DOWNSTREAM_NODE_CLASS",
    "EDITED_NODE_CLASS",
    "EDIT_GAP_ROW",
    "EDIT_RING_PENWIDTH",
    "EditMark",
    "EditRow",
    "edit_ink",
    "edit_marks_by_site",
    "edit_marks_for_node",
    "edit_row_html",
    "edit_rows",
    "edit_tooltip",
    "header_edit_line",
    "make_intervention_collapsed_spec_fn",
    "short_edit_text",
]

#: The Greek capital delta every edit mark starts with.
DELTA = "Δ"
#: SVG class on a node the edit changed downstream (drawn only with the cone on).
DOWNSTREAM_NODE_CLASS = "tl_downstream"
#: Ring width of the edited node, in points.
EDIT_RING_PENWIDTH = 2.5
#: The dark skin's edit purple (JMT kept this tint, 29 Oct 2026).
_DARK_EDIT_INK = "#C084FC"
#: Graphviz's node label size, which TorchLens op labels use (``_typography`` base size).
_OP_LABEL_SIZE = 14.0
#: Edit text size over delta size: JMT picked an 11 pt delta over 10 pt text.
_TEXT_TO_DELTA = 10.0 / 11.0
#: Edit rows drawn at one node before ``+N more``.
_MAX_NODE_EDIT_ROWS = 3
#: Edits named in the header before it says ``N edits (marked)``.
_MAX_HEADER_EDITS = 3
#: Longest string argument written in; longer ones are cut with an ellipsis.
_MAX_LITERAL_CHARS = 16
#: Keyword arguments that never appear in the edit text.
_SILENT_KWARGS = frozenset({"allow_shape_change"})
#: Hook names the value-replacement door gives its generated hooks.
_SET_HOOK_MARKERS = ("_set_value_hook", "_value_hook")


@dataclass(frozen=True)
class EditMark:
    """One edit at one site, as the picture names it.

    Attributes
    ----------
    short:
        The edit text after the delta, for example ``scale(0.5)``.
    full:
        The edit's full public spelling, for the tooltip.
    detail:
        Engine and timing (``replay, post``), or ``staged`` when it has not fired.
    """

    short: str
    full: str
    detail: str


class EditRow(str):
    """A node label row drawn as the edit line (bold, in the edit purple).

    It is an ordinary string to node callbacks (``"Δ zero_ablate"``), so a
    ``node_spec_fn`` can read, drop or reorder it like any other row; the label
    builder recognises the type and draws it in ``color`` after a 4 pt gap.
    The delta is drawn at ``size`` (the op label's size) and the edit text one
    step smaller, in the 11 : 10 ratio of JMT's pick.
    """

    color: str
    size: float

    def __new__(cls, text: str, color: str, size: float = _OP_LABEL_SIZE) -> EditRow:
        """Return a row with ``text`` drawn in ``color``, its delta at ``size`` points."""

        row = super().__new__(cls, text)
        row.color = color
        row.size = size
        return row

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as ``EditRow(text, color, size)``."""

        return (EditRow, (str(self), self.color, self.size))


def edit_ink(theme: VisualizationTheme | None) -> str:
    """Return the edit purple for a skin: ``#7B1FA2``, or ``#C084FC`` on a dark page."""

    if theme is None:
        return INTERVENTION_SITE_COLOR

    page = str(theme.graph.get("bgcolor", "white"))
    luminance = _relative_luminance(page) if page.startswith("#") else None
    if luminance is not None and luminance < 0.2:
        return _DARK_EDIT_INK
    return INTERVENTION_SITE_COLOR


def _literal(value: Any) -> str | None:
    """Return a short literal's text, or ``None`` when the value needs a stand-in."""

    if value is None or isinstance(value, bool | int):
        return repr(value)
    if isinstance(value, float):
        return format(value, "g")
    if isinstance(value, str):
        if len(value) <= _MAX_LITERAL_CHARS:
            return value
        return value[: _MAX_LITERAL_CHARS - 1] + "…"
    if isinstance(value, tuple | list) and 0 < len(value) <= 3:
        parts = [_literal(item) for item in value]
        if all(part is not None for part in parts):
            return f"({', '.join(str(part) for part in parts)})"
    return None


def _stand_in(value: Any) -> str:
    """Return a short stand-in for one edit argument.

    Literals are written in; a tensor is ``v``; a trace or anything else with a
    short ``name`` shows that name (else ``trace``); a persisted trace identity
    shows its label; a callable shows its name (else ``fn``); anything else ``…``.
    """

    literal = _literal(value)
    if literal is None and isinstance(value, torch.Tensor):
        literal = "v"
    if literal is None:
        name = getattr(value, "name", None)
        if isinstance(name, str) and name and len(name) <= _MAX_LITERAL_CHARS:
            literal = name
        elif is_live_trace(value):
            literal = "trace"
        elif isinstance(value, Mapping) and isinstance(value.get("label"), str):
            literal = _literal(value["label"])
        elif callable(value):
            literal = _callable_name(value)
    return literal or "…"


def _callable_name(value: Any) -> str:
    """Return a callable's plain name, or ``fn`` for a lambda or a nested function."""

    name = getattr(value, "__name__", None)
    if isinstance(name, str) and name.isidentifier() and len(name) <= _MAX_LITERAL_CHARS:
        return name
    return "fn"


def _shown_kwargs(edit: Edit, constructor: Callable[..., Any] | None) -> list[str]:
    """Return ``key=value`` parts for the edit's keyword arguments that are not defaults."""

    order = _parameter_order(constructor)
    parts: list[str] = []
    for key, value in sorted(edit.kwargs, key=lambda item: order.get(item[0], len(order))):
        if key in _SILENT_KWARGS or value is None or not _takes_keyword(constructor, key):
            continue
        if constructor is not None and _is_signature_default(constructor, key, value):
            continue
        parts.append(f"{key}={_stand_in(value)}")
    return parts


def _takes_keyword(constructor: Callable[..., Any] | None, key: str) -> bool:
    """Return whether ``constructor`` takes ``key`` (a stored element mask it does not take)."""

    if constructor is None:
        return True
    try:
        parameters = inspect.signature(constructor).parameters
    except (TypeError, ValueError):
        return True
    variadic = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    return variadic or key in parameters


def _parameter_order(constructor: Callable[..., Any] | None) -> dict[str, int]:
    """Return each parameter's position in the constructor's signature (empty if unknown)."""

    if constructor is None:
        return {}
    try:
        names = inspect.signature(constructor).parameters
    except (TypeError, ValueError):
        return {}
    return {name: index for index, name in enumerate(names)}


def short_edit_text(edit: Edit) -> str:
    """Return the edit text after the delta, for example ``zero_ablate`` or ``scale(0.5)``."""

    constructor = builtin_helper_constructor(edit.name)
    parts = [_stand_in(value) for value in edit.args]
    parts.extend(_shown_kwargs(edit, constructor))
    if not parts:
        return edit.name
    return f"{edit.name}({', '.join(parts)})"


def _hook_mark(hook: Any, *, full: str | None, detail: str) -> EditMark:
    """Return the mark for a plain hook callable or a value replacement."""

    from ..intervention._types import Edit

    if isinstance(hook, Edit):
        return _helper_mark(hook, detail=detail)
    return EditMark(short=f"hook({_callable_name(hook)})", full=full or repr(hook), detail=detail)


def _helper_mark(edit: Edit, *, detail: str) -> EditMark:
    """Return the mark for a helper-built edit."""

    return EditMark(short=short_edit_text(edit), full=render_edit_repr(edit), detail=detail)


def _value_mark(value: Any, *, detail: str) -> EditMark:
    """Return the mark for a ``set`` value replacement."""

    shown = _callable_name(value) if callable(value) else _stand_in(value)
    return EditMark(short=f"set({shown})", full=f"set({shown})", detail=detail)


def _fire_mark(record: FireRecord) -> EditMark:
    """Return the mark for one fire record (``FireRecord.edit`` first, then its name)."""

    detail = ", ".join(str(part) for part in (record.engine, record.timing) if part)
    if record.edit is not None:
        return _helper_mark(record.edit, detail=detail)
    name = str(record.edit_name or "hook")
    if any(marker in name for marker in _SET_HOOK_MARKERS):
        return EditMark(short="set(v)", full="set(v)", detail=detail)
    leaf = name.rsplit(".", 1)[-1]
    shown = leaf if leaf.isidentifier() and len(leaf) <= _MAX_LITERAL_CHARS else "fn"
    return EditMark(short=f"hook({shown})", full=name, detail=detail)


def _op_key(op: Any) -> str:
    """Return the pass-qualified label of an op (``relu_1_1:2``), else its layer label."""

    label = get_multipass_attr(op, "label", None, multipass=None)
    if isinstance(label, str):
        return label
    return str(getattr(op, "layer_label", ""))


def _staged_entries(spec: Any) -> list[tuple[Any, Callable[[], EditMark]]]:
    """Return ``(site_target, mark builder)`` pairs for a recipe in firing order.

    Value replacements fire before hooks (``intervention.hooks`` normalizes
    ``target_value_specs`` first, then ``hook_specs``), then the recipe-level
    helper, value or hook applied to ``targets``.
    """

    entries: list[tuple[Any, Callable[[], EditMark]]] = []
    for value_spec in getattr(spec, "target_value_specs", ()) or ():
        entries.append(
            (value_spec.site_target, partial(_value_mark, value_spec.value, detail="staged"))
        )
    for hook_spec in getattr(spec, "hook_specs", ()) or ():
        if hook_spec.helper is not None:
            build = partial(_helper_mark, hook_spec.helper, detail="staged")
        else:
            build = partial(_hook_mark, hook_spec.hook, full=None, detail="staged")
        entries.append((hook_spec.site_target, build))
    recipe_level = _recipe_level_mark(spec)
    if recipe_level is not None:
        for target in getattr(spec, "targets", ()) or ():
            entries.append((target, recipe_level))
    return entries


def _recipe_level_mark(spec: Any) -> Callable[[], EditMark] | None:
    """Return the mark builder for a recipe's own helper, hook or value, if any."""

    helper = getattr(spec, "helper", None)
    hook = getattr(spec, "hook", None)
    value = getattr(spec, "value", None)
    if hook is not None:
        return partial(_hook_mark, hook, full=None, detail="staged")
    if helper is not None:
        return partial(_helper_mark, helper, detail="staged")
    if value is not None:
        return partial(_value_mark, value, detail="staged")
    return None


def _staged_marks(trace: Trace) -> dict[str, list[EditMark]]:
    """Return staged edit marks keyed by pass-qualified site label."""

    spec = trace._intervention_spec
    if spec is None:
        return {}

    marks: dict[str, list[EditMark]] = {}
    fanout = max(1, len(trace._layer_list or ()))
    for target, build in _staged_entries(spec):
        try:
            table = _resolve_sites(trace, target, max_fanout=fanout)
        except (LookupError, ValueError, TypeError):
            # A target that resolves to no site here names no node to mark.
            continue
        mark = build()
        for site in table:
            # Key by the site's own pass, as ``intervention_sites_for_log`` does,
            # so a backward edit on one pass of a multi-pass layer names that pass.
            marks.setdefault(_op_key(_forward_site(site)), []).append(mark)
    return marks


def edit_marks_by_site(trace: Trace, sites: Sequence[Op]) -> dict[str, tuple[EditMark, ...]]:
    """Return each site's edit marks in firing order, keyed by pass-qualified label.

    An op whose ``edits`` (its ``FireRecord`` list) is non-empty is
    named from those records, ordered by their timestamps; a site that has not
    fired is named from the staged recipe entries that resolve to it. A site
    with neither is marked ``edit``.

    Parameters
    ----------
    trace:
        Trace whose intervention is drawn.
    sites:
        Distinct intervention sites (``node_spec.intervention_sites_for_log``).
    """

    staged: dict[str, list[EditMark]] | None = None
    result: dict[str, tuple[EditMark, ...]] = {}
    for site in sites:
        key = _op_key(site)
        records = list(getattr(site, "edits", None) or ())
        if records:
            ordered = sorted(
                enumerate(records),
                key=lambda item: (getattr(item[1], "timestamp", None) or 0.0, item[0]),
            )
            result[key] = tuple(_fire_mark(record) for _, record in ordered)
            continue
        if staged is None:
            staged = _staged_marks(trace)
        result[key] = tuple(staged.get(key, ())) or (
            EditMark(short="edit", full="edit", detail="staged"),
        )
    return result


def _unique(marks: Iterable[EditMark]) -> list[EditMark]:
    """Return marks with repeated texts dropped, first occurrence kept."""

    seen: set[str] = set()
    unique: list[EditMark] = []
    for mark in marks:
        if mark.short not in seen:
            seen.add(mark.short)
            unique.append(mark)
    return unique


def edit_rows(marks: Iterable[EditMark], color: str, size: float = _OP_LABEL_SIZE) -> list[EditRow]:
    """Return a node's edit rows: one per edit, at most three, then ``+N more``."""

    unique = _unique(marks)
    rows = [EditRow(f"{DELTA} {mark.short}", color, size) for mark in unique[:_MAX_NODE_EDIT_ROWS]]
    hidden = len(unique) - _MAX_NODE_EDIT_ROWS
    if hidden > 0:
        rows.append(EditRow(f"+{hidden} more", color, size))
    return rows


def edit_tooltip(marks: Iterable[EditMark]) -> str:
    """Return the edited node's tooltip: every edit's full spelling, engine and timing."""

    parts = []
    for mark in _unique(marks):
        suffix = f" ({mark.detail})" if mark.detail else ""
        parts.append(f"{DELTA} {mark.short}: {mark.full}{suffix}")
    return "; ".join(parts)


def edit_row_html(row: str, color: str, size: float = _OP_LABEL_SIZE) -> str:
    """Return the HTML of one edit row, bold in the edit purple.

    The delta takes the op label's size and the text the 10/11 step below it
    (11 pt over 10 pt on the 11 pt mockup labels JMT chose from).
    """

    delta_pt = format_pt(round(size, 1))
    text_pt = format_pt(round(size * _TEXT_TO_DELTA, 1))
    text = str(row)
    if text.startswith(f"{DELTA} "):
        body = html_escape(text[len(DELTA) + 1 :])
        inner = (
            f'<FONT POINT-SIZE="{delta_pt}">{DELTA}</FONT>'
            # A non-breaking space: SVG drops a text run's leading plain space.
            f'<FONT POINT-SIZE="{text_pt}">&#160;{body}</FONT>'
        )
    else:
        inner = f'<FONT POINT-SIZE="{text_pt}">{html_escape(text)}</FONT>'
    return f'<FONT COLOR="{color}"><B>{inner}</B></FONT>'


#: The 4 pt blank row above a node's first edit row (no rule).
EDIT_GAP_ROW = (
    f'<TR><TD CELLPADDING="0"><FONT POINT-SIZE="{format_pt(EDIT_GAP_SIZE)}"> </FONT></TD></TR>'
)


def header_edit_line(
    trace: Trace,
    sites: Sequence[Op],
    color: str,
    *,
    vis_mode: str = "unrolled",
) -> str | None:
    """Return the header's edit line as HTML, or ``None`` when nothing is edited.

    ``1 edit: Δ zero_ablate on linear_1_1``; several edits are comma separated
    up to three, then the line reads ``N edits (marked)``; one edit on several
    sites reads ``Δ zero_ablate on 12 layers``. A module edit is drawn at its
    own ``interventionreplacement`` op; the line names the site the user
    edited instead (:func:`_header_site_names`).

    Graphviz writes the coloured token as its own SVG text run and places the
    next run by its own estimate of the token's width, which falls short of the
    drawn ``Δ`` (and more so in bold). The space after the token therefore sits
    INSIDE the coloured run as two non-breaking spaces: the gap becomes part of
    the next run's position, which survives rasterizers that strip a run's edge
    spaces, and the second space absorbs the estimate's shortfall. A leading
    space on the next run, or a bold token, drew ``zero_ablateon``. The SVG
    output joins the line's runs into one text, which a browser lays out in
    order, and there the gap collapses to one space (:mod:`._svg_text_runs`);
    PNG and PDF keep Graphviz's run positions and the padding.
    """

    if not sites:
        return None
    marks = edit_marks_by_site(trace, sites)
    site_names = _header_site_names(trace, sites, vis_mode)
    grouped: dict[str, list[str]] = {}
    total = 0
    for site in sites:
        site_label = site_names[_op_key(site)]
        for mark in _unique(marks.get(_op_key(site), ())):
            total += 1
            labels = grouped.setdefault(mark.short, [])
            if site_label not in labels:
                labels.append(site_label)
    noun = "edit" if total == 1 else "edits"
    if len(grouped) > _MAX_HEADER_EDITS:
        return html_escape(f"{total} {noun} (marked)")
    items = []
    for short, labels in grouped.items():
        token = f'<FONT COLOR="{color}">{DELTA} {html_escape(short)}&#160;&#160;</FONT>'
        where = html_escape(labels[0]) if len(labels) == 1 else f"{len(labels)} layers"
        items.append(f"{token}on {where}")
    # Non-breaking: SVG drops a plain space at the edge of a text run.
    return f"{total} {noun}:&#160;" + ",&#160;".join(items)


def _header_site_names(trace: Trace, sites: Sequence[Op], vis_mode: str) -> dict[str, str]:
    """Return the name the header gives each site, keyed by pass-qualified label.

    A module edit's value sits in its own ``interventionreplacement`` op, the
    site the trace resolves. The header names the module address the user
    wrote for a ``tl.module`` target (``transformer.h.5.mlp``), and the op the
    replacement stands in for otherwise; any other site keeps its own label.
    """

    addresses = _module_target_addresses(trace)
    shown = {
        _op_key(site): (
            replaced_site(site, trace)
            if getattr(site, "func_name", None) == REPLACEMENT_FUNC_NAME
            else site
        )
        for site in sites
    }
    edited_passes = Counter(str(getattr(op, "layer_label", "") or "") for op in shown.values())
    return {
        key: addresses.get(key) or _site_display_label(op, vis_mode, edited_passes)
        for key, op in shown.items()
    }


def _module_target_addresses(trace: Trace) -> dict[str, str]:
    """Return the address of the ``tl.module`` target behind each site it resolves to."""

    spec = trace._intervention_spec
    if spec is None:
        return {}
    fanout = max(1, len(trace._layer_list or ()))
    addresses: dict[str, str] = {}
    for target, _ in _staged_entries(spec):
        if getattr(target, "selector_kind", None) != "module":
            continue
        try:
            table = _resolve_sites(trace, target, max_fanout=fanout)
        except (LookupError, ValueError, TypeError):
            continue
        for site in table:
            addresses.setdefault(_op_key(_forward_site(site)), str(target.selector_value))
    return addresses


def _site_display_label(site: Any, vis_mode: str, edited_passes: Mapping[str, int]) -> str:
    """Return the name the header gives a site: the layer label, plus ``:N`` for one pass.

    A single-pass op reads ``linear_1_1`` (its pass label is ``linear_1_1:1``);
    an unrolled multi-pass op keeps its ``:N``. Rolled, the layer's one node
    stands for every pass, so the bare layer label is used only when every
    pass is edited; an edit on some passes keeps the ``:N`` that says which.
    """

    layer_label = str(getattr(site, "layer_label", "") or "")
    passes = getattr(site, "num_passes", 1)
    if not layer_label or not isinstance(passes, int) or passes <= 1:
        return layer_label or _op_key(site)
    if vis_mode == "rolled" and edited_passes.get(layer_label, 0) >= passes:
        return layer_label
    return _op_key(site)


def edit_marks_for_node(
    marks: Mapping[str, tuple[EditMark, ...]],
    node: Any,
    site_keys_by_layer: Mapping[str, Sequence[str]],
) -> tuple[EditMark, ...]:
    """Return the marks a drawn node shows: its own pass, or every edited pass when rolled."""

    key = _op_key(node)
    if key in marks:
        return marks[key]
    layer = str(getattr(node, "layer_label", ""))
    collected: list[EditMark] = []
    for site_key in site_keys_by_layer.get(layer, ()):
        collected.extend(marks.get(site_key, ()))
    return tuple(collected)


def make_intervention_collapsed_spec_fn(
    trace: Trace, request: Any, theme: VisualizationTheme | None = None
) -> CollapsedNodeSpecFn | None:
    """Build a collapsed-box callback that marks a box holding an edited op.

    A folded module box that contains an intervention site gets the same mark
    as the edited op node would: the edit purple ring, the edit rows
    (``Δ zero_ablate``, one per distinct edit) as its last label rows, the
    edits' full spellings in its tooltip and the ``tl_edited`` SVG class. The
    box's member ops come from the box-ops slot the renderer sets around the
    call. ``request`` supplies ``graph_overrides`` (the intervention style keys)
    and the user's ``collapsed_node_spec_fn``, which runs after the mark.

    Returns
    -------
    CollapsedNodeSpecFn | None
        Combined callback, or the user's callback when nothing is edited.
    """

    user_collapsed_spec_fn = request.collapsed_node_spec_fn
    site_ops = intervention_sites_for_log(trace)
    marks = edit_marks_by_site(trace, site_ops) if site_ops else {}
    if not marks:
        return user_collapsed_spec_fn
    style = _site_style(request.graph_overrides, theme)
    aliases = _site_aliases(site_ops)

    def intervention_collapsed_spec_fn(module: Any, default_spec: NodeSpec) -> NodeSpec:
        """Mark a box holding an edited op, then run any user callback."""

        found = {aliases[label] for label in _SPEC_SLOT_BOX_OPS.get() or () if label in aliases}
        box_marks = [mark for key, site in marks.items() if key in found for mark in site]
        spec = _marked_spec(default_spec, box_marks, style) if box_marks else default_spec
        if user_collapsed_spec_fn is None:
            return spec
        user_result = user_collapsed_spec_fn(module, spec)
        return spec if user_result is None else user_result

    return intervention_collapsed_spec_fn


def _site_style(
    graph_overrides: dict[str, Any] | None, theme: VisualizationTheme | None
) -> tuple[str, float, float]:
    """Return the edit mark's ``(colour, ring width, label size)`` for a draw."""

    color = intervention_graph_override(graph_overrides, "intervention_site_color", edit_ink(theme))
    width = intervention_graph_override(
        graph_overrides, "intervention_site_penwidth", EDIT_RING_PENWIDTH
    )
    size = theme.typography.base_size if theme is not None else _OP_LABEL_SIZE
    return str(color), float(width), float(size)


def _marked_spec(
    spec: NodeSpec, marks: Sequence[EditMark], style: tuple[str, float, float]
) -> NodeSpec:
    """Return ``spec`` with the edit ring, edit rows, tooltip and ``tl_edited`` class."""

    color, width, size = style
    tooltip = edit_tooltip(marks)
    return spec.replace(
        color=color,
        penwidth=width,
        lines=[*spec.lines, *edit_rows(marks, color, size)],
        tooltip=f"{spec.tooltip}; {tooltip}" if spec.tooltip else tooltip,
        extra_attrs=_with_svg_class(spec.extra_attrs, EDITED_NODE_CLASS),
    )


def _site_aliases(site_ops: list[Op]) -> dict[str, str]:
    """Map each spelling a box member list may use for a site to its pass-qualified key.

    A site is keyed ``relu_3:1`` (its pass label); a single-pass site may also
    be listed by its bare layer label.
    """

    aliases: dict[str, str] = {}
    for site in site_ops:
        layer_label = str(getattr(site, "layer_label", "") or "")
        pass_label = get_multipass_attr(site, "label", None, multipass=None)
        key = pass_label if isinstance(pass_label, str) else layer_label
        aliases[key] = key
        if layer_label and getattr(site, "num_passes", 1) == 1:
            aliases.setdefault(layer_label, key)
    return aliases

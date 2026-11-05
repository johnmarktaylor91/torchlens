"""Value-change marks for the interventions view: which ops an edit changed.

``trace.draw(view="interventions")`` answers "what did the edit change?". The
edited node carries the surgery mark family's purple outline; this module adds
the second half of the answer: every other op whose saved value differs from
the clean run gets a light purple outline and a row with the largest absolute
change. Ops upstream of the edit, and downstream ops whose value came out the
same, stay plain.

The clean values come from the unedited trace the edited one was forked from
(``trace.parent_trace``). When that trace is gone, or is itself edited, there
is nothing honest to compare against: the view says so in one caption line and
marks only the edit sites.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .._errors import CaptureError
from .._utils._multipass_access import get_multipass_attr, is_multipass_layer
from .._vocab.node_spec import NodeSpec, NodeSpecFn
from ._render_names import _SPEC_SLOT_RENDERED_NODE

if TYPE_CHECKING:
    from .._data_classes.layer import Layer
    from .._data_classes.trace import Trace

__all__ = [
    "CHANGED_INK",
    "CHANGED_NODE_CLASS",
    "ChangeReport",
    "change_census_rows",
    "change_key_rows",
    "change_report",
    "make_change_spec_fn",
]

#: Light purple (the edit ink's family: purple 300 beside the edit's purple 700).
CHANGED_INK = "#BA68C8"
#: Outline width of a changed op: lighter than the edited op's 3.0.
CHANGED_PENWIDTH = 2.0
#: SVG class on a node whose saved value the edit changed (legend role ``changed value``).
CHANGED_NODE_CLASS = "tl_changed"


@dataclass(frozen=True)
class ChangeReport:
    """Per-op change against the clean run, or why there is none.

    Attributes
    ----------
    by_pass:
        Pass-qualified op label to its largest absolute change; ``None`` when
        the value changed but has no numeric difference (a bool or int tensor
        of another shape, say).
    by_layer:
        Aggregate layer label to the largest change over its passes (rolled nodes).
    unavailable:
        Why no clean values could be compared, or ``None`` when they were.
    """

    by_pass: Mapping[str, float | None] = field(default_factory=dict)
    by_layer: Mapping[str, float | None] = field(default_factory=dict)
    unavailable: str | None = None

    @property
    def changed_count(self) -> int:
        """Number of changed ops (each pass counted once)."""

        return len(self.by_pass)


def _clean_reference(
    trace: Trace, is_edited: Callable[[Trace], bool]
) -> tuple[Trace | None, str | None]:
    """Return the unedited trace to compare against, or why there is none."""

    parent = trace.parent_trace
    if parent is None:
        return None, (
            "changed ops not shown: the unedited trace is not available "
            "(draw a fork while the trace it came from is still alive)"
        )
    if is_edited(parent):
        return None, "changed ops not shown: the trace this one was forked from is itself edited"
    return parent, None


def _saved_value(trace: Trace, label: str) -> Any:
    """Return one op's saved value, or ``None`` when it is not readable."""

    try:
        return getattr(trace[label], "out", None)
    except (CaptureError, KeyError):
        # A value that was never saved, was evicted, or a label the other
        # trace does not have: nothing to compare, so the op stays plain.
        return None


def _value_change(edited: Any, clean: Any) -> tuple[bool, float | None]:
    """Return ``(changed, largest absolute change)`` for two saved values.

    NaN in both values at one position counts as unchanged there; NaN in one
    only counts as an infinite change.
    """

    if not isinstance(edited, torch.Tensor) or not isinstance(clean, torch.Tensor):
        return False, None
    if edited.shape != clean.shape or edited.dtype != clean.dtype:
        return True, None
    if edited.numel() == 0 or torch.equal(edited, clean):
        return False, None
    wide = torch.complex128 if edited.is_complex() else torch.float64
    diff = (edited.detach().to(wide) - clean.detach().to(wide)).abs()
    both_nan = torch.isnan(edited) & torch.isnan(clean)
    diff = diff.masked_fill(both_nan, 0.0).nan_to_num(nan=math.inf)
    largest = float(diff.max())
    return largest > 0.0, largest


def change_report(
    trace: Trace, edited_labels: Collection[str], is_edited: Callable[[Trace], bool]
) -> ChangeReport:
    """Compare every saved value of ``trace`` against the clean run.

    Parameters
    ----------
    trace:
        The edited trace being drawn.
    edited_labels:
        Pass-qualified and aggregate labels of the edit sites; they keep the
        edit's own outline and are never marked as changed.
    is_edited:
        Whether a trace carries edits; an edited parent is no clean run.

    Returns
    -------
    ChangeReport
        Changed ops with their largest absolute change, or the reason the
        clean values are unavailable.
    """

    clean, reason = _clean_reference(trace, is_edited)
    if clean is None:
        return ChangeReport(unavailable=reason)
    clean_labels = set(clean.op_labels)
    by_pass: dict[str, float | None] = {}
    by_layer: dict[str, float | None] = {}
    for label in trace.op_labels:
        if label not in clean_labels or label in edited_labels:
            continue
        changed, largest = _value_change(_saved_value(trace, label), _saved_value(clean, label))
        if not changed:
            continue
        layer_label = str(getattr(trace[label], "layer_label", label))
        if layer_label in edited_labels:
            continue
        by_pass[label] = largest
        previous = by_layer.get(layer_label, 0.0)
        by_layer[layer_label] = (
            None if largest is None or previous is None else max(previous, largest)
        )
    return ChangeReport(by_pass=by_pass, by_layer=by_layer)


def _change_row(largest: float | None) -> str:
    """Return the node row stating one op's change."""

    if largest is None:
        return "value changed"
    return f"changed by up to {largest:.3g}"


def make_change_spec_fn(report: ChangeReport) -> NodeSpecFn | None:
    """Build the node callback that outlines changed ops, or ``None`` with nothing to mark."""

    if not report.by_pass:
        return None

    def change_spec_fn(layer_log: Layer, default_spec: NodeSpec) -> NodeSpec:
        """Outline one changed op in light purple and print its largest change."""

        rendered = _SPEC_SLOT_RENDERED_NODE.get()
        target = layer_log if rendered is None else rendered
        pass_label = get_multipass_attr(target, "label", None, multipass=None)
        layer_label = str(getattr(layer_log, "layer_label", ""))
        if isinstance(pass_label, str) and pass_label in report.by_pass:
            largest = report.by_pass[pass_label]
        elif (
            not isinstance(pass_label, str) or not is_multipass_layer(layer_log)
        ) and layer_label in report.by_layer:
            largest = report.by_layer[layer_label]
        else:
            return default_spec
        classes = str(default_spec.extra_attrs.get("class", "")).split()
        return default_spec.replace(
            color=CHANGED_INK,
            penwidth=CHANGED_PENWIDTH,
            lines=[*default_spec.lines, _change_row(largest)],
            extra_attrs={
                **default_spec.extra_attrs,
                "class": " ".join([*classes, CHANGED_NODE_CLASS]),
            },
        )

    return change_spec_fn


def change_key_rows(report: ChangeReport, heuristic_count: int) -> tuple[str, ...]:
    """The caption's outline key, and why changed ops may be missing."""

    if report.unavailable is not None:
        key, extra = "purple outline: edited", (report.unavailable,)
    elif report.changed_count:
        key, extra = "purple outline: edited; light purple: changed by the edit", ()
    else:
        key, extra = "purple outline: edited", ("no other op's saved value changed",)
    if heuristic_count:
        key += "; dashed outline: edit site inferred, not recorded"
    return (key, *extra)


def change_census_rows(report: ChangeReport) -> tuple[str, ...]:
    """The census record of changed ops: every one, largest change first."""

    if report.unavailable is not None:
        return (report.unavailable,)
    ranked = sorted(
        report.by_pass.items(),
        key=lambda item: math.inf if item[1] is None else item[1],
        reverse=True,
    )
    rows = [f"ops whose saved value changed: {len(ranked)}"]
    for label, change in ranked:
        text = "changed (no numeric difference)" if change is None else f"{change:.6g}"
        rows.append(f"  {label}: max |change| {text}")
    return tuple(rows)

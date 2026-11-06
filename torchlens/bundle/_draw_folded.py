"""``Bundle.draw(graph="diff", depth=N)``: one member's model graph shaded by its change.

The folded diff draws the compared member through its own ``Trace.draw`` (so
``depth`` folds modules into boxes exactly as it does there) with a fill channel
whose value at every op is the Bundle's per-site distance from the baseline and
whose value at every folded module box is the module's most changed op, the
fold ``rank_changes(by="module")`` uses. Every drawn op and box gets its value:
nothing is truncated to a top-k.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from .._utils._record_group_table import record_member
from ._deltas import _resolve_member_name, site_deltas

if TYPE_CHECKING:
    import torch

    from .._data_classes.trace import Trace
    from ..viz._drawing import Drawing
    from . import Bundle
    from ._deltas import SiteDeltas


class ChangeOverlay:
    """The ``color_by`` source of a folded diff: per-op change and per-module maximum.

    Called on a drawn op it returns the compared member's distance from the
    baseline at that op's site (``0.0`` where the site is unchanged or has no
    comparison). :meth:`module_value` gives a folded module box the largest
    distance among the ops it holds, per call (``addr:N``) or over every call.
    """

    def __init__(self, deltas: SiteDeltas, member_name: str, member: Trace, name: str) -> None:
        """Hold the per-site distances and fold them onto ``member``'s modules."""

        self.__name__ = name
        self._deltas = deltas
        self._member_name = member_name
        self._modules = module_changes(member, deltas, member_name)

    def __call__(self, node: Any) -> float | None:
        """Return one drawn op's distance from the baseline, else ``None`` for a non-op node."""

        if getattr(node, "layer_label", None) is None:
            return None
        return self._deltas.value(node, self._member_name)

    def module_value(self, address: str, call_index: str | None = None) -> float | None:
        """Return a module box's most changed op (its own call ``addr:N`` first), else ``None``."""

        if call_index is not None and f"{address}:{call_index}" in self._modules:
            return self._modules[f"{address}:{call_index}"]
        return self._modules.get(address)


def module_changes(member: Trace, deltas: SiteDeltas, member_name: str) -> dict[str, float]:
    """Fold per-op distances into each enclosing module: its most changed op.

    Keys are module call labels (``transformer.h.5:1``) and bare addresses
    (``transformer.h.5``, the maximum over every call), as ``Trace.draw``'s
    module boxes look them up.
    """

    folded: dict[str, float] = {}
    for op in member._layer_list:
        value = deltas.value(op, member_name)
        keys = [str(label) for label in record_member(op, "modules", ()) or ()]
        keys += [key.rsplit(":", 1)[0] for key in keys]
        atomic = record_member(op, "atomic_module_address", None)
        if atomic:
            keys.append(str(atomic))
        for key in keys:
            folded[key] = max(value, folded.get(key, value))
    return folded


def draw_folded_diff(bundle: Bundle, path: str | None, options: dict[str, Any]) -> Drawing:
    """Draw the compared member folded to ``options["depth"]``, shaded by change from the baseline.

    ``members`` is ``(baseline, compared)`` (default: the first two members);
    ``metric`` and ``on`` pick the per-site distance as ``graph="diff"`` does.
    ``path``, ``file_format``, ``direction``, ``theme``, ``show_legend``,
    ``open_viewer`` and ``graph_overrides`` go to the member's ``Trace.draw``.
    """

    from .._visualization._encoding_request import EncodingChannelRequest

    baseline_name, member_name = _diff_pair(bundle, options["members"])
    bundle._require_comparable("delta_map", pairs=[(baseline_name, member_name)])
    metric = options["metric"]
    deltas = site_deltas(bundle, metric, baseline_name, options["on"], [baseline_name, member_name])
    member = bundle.members[member_name]
    label = f"{_metric_name(metric)} change from {baseline_name}"
    overlay = ChangeOverlay(deltas, member_name, member, label)
    theme = options["theme"]
    return member.draw(
        path,
        depth=options["depth"],
        direction=options["direction"],
        theme=theme if theme is not None else "torchlens",
        show_legend=options["show_legend"],
        file_format=options["file_format"],
        open_viewer=options["open_viewer"],
        graph_overrides=options["graph_overrides"],
        color_by=EncodingChannelRequest(source=overlay, display_name=label),
    )


def _diff_pair(bundle: Bundle, members: tuple[str | Trace, str | Trace] | None) -> tuple[str, str]:
    """Resolve ``(baseline, compared)`` member names; the default is the first two members."""

    names = bundle.names
    if members is None and len(names) < 2:
        raise InvalidArgumentError(
            "Bundle.draw(graph='diff') requires at least two bundle members",
            code="bundle_diff_members_invalid",
            remedy="diff a bundle with at least two members, or name both in members=",
            argument="members",
        )
    first, second = members if members is not None else (names[0], names[1])
    baseline_name = _resolve_member_name(bundle, first)
    member_name = _resolve_member_name(bundle, second)
    if baseline_name == member_name:
        raise InvalidArgumentError(
            "Bundle.draw(graph='diff') requires two distinct members",
            code="bundle_diff_members_invalid",
            remedy="pass two different member names in members=",
            argument="members",
        )
    return baseline_name, member_name


def _metric_name(metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> str:
    """Return a metric's display name: the string itself, else the callable's name."""

    if isinstance(metric, str):
        return metric
    return getattr(metric, "__name__", type(metric).__name__)


__all__ = ("ChangeOverlay", "draw_folded_diff", "module_changes")

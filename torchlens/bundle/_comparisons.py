"""Named per-node comparison fields stored on a bundle's supergraph.

``Bundle.delta_map`` recomputes tensor distances on every call; these helpers
compute once and stamp the values on each ``SupergraphNode.comparisons`` under
a queryable name, so later reads (queries, renders) hit the stored field.
Session-level state on the built supergraph — never serialized. Spellings are
DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from . import Bundle


def _bundle_store_comparison(
    self: Bundle,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    *,
    name: str | None = None,
    baseline: str | Trace | None = None,
    on: Literal["out", "grad"] = "out",
) -> str:
    """Compute one per-node comparison and stamp it on the supergraph.

    Runs the same computation as :meth:`Bundle.delta_map` once, then stores
    the per-member values on every contributing ``SupergraphNode`` under the
    returned name, so later queries read instead of recomputing.

    Parameters
    ----------
    self:
        Bundle whose supergraph receives the stored comparison.
    metric:
        Metric name from ``torchlens.intervention._metrics`` or a callable.
    name:
        Comparison name to store under. Defaults to
        ``"{metric}:{on}@{baseline}"`` for string metrics; a callable metric
        has no derivable stable name and requires an explicit one.
    baseline:
        Baseline member name or log, resolved like :meth:`Bundle.delta_map`.
    on:
        Tensor field to compare.

    Returns
    -------
    str
        The name the comparison was stored under.

    Raises
    ------
    InvalidArgumentError
        ``comparison_name_required`` when ``metric`` is a callable and no
        explicit ``name`` was supplied.
    """

    from ._deltas import _delta_map_values, _gated_baseline

    baseline_name = _gated_baseline(self, "delta_map", baseline)
    if name is None:
        if callable(metric):
            raise InvalidArgumentError(
                "a callable metric has no derivable stable comparison name",
                code="comparison_name_required",
                remedy="pass an explicit name= for callable metrics",
                argument="name",
            )
        name = f"{metric}:{on}@{baseline_name}"
    values = _delta_map_values(self, metric, baseline_name, on)
    supergraph = self.supergraph
    for node in supergraph.nodes.values():
        node.comparisons.pop(name, None)
    for node_label, member_values in values.items():
        supergraph.nodes[node_label].comparisons[name] = dict(member_values)
    return name


def _bundle_stored_comparison(self: Bundle, name: str) -> dict[str, dict[str, float]]:
    """Return one stored per-node comparison without recomputing.

    Parameters
    ----------
    self:
        Bundle whose supergraph holds the stored comparisons.
    name:
        Comparison name returned by :meth:`Bundle.store_comparison`.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of supergraph node name to member-name distance values, in
        supergraph topological order — the same shape ``delta_map`` returns.

    Raises
    ------
    InvalidArgumentError
        ``comparison_unknown`` when no comparison was stored under ``name``.
    """

    supergraph = self.supergraph
    result: dict[str, dict[str, float]] = {}
    available: set[str] = set()
    for node_label in supergraph.topological_order:
        node = supergraph.nodes[node_label]
        available.update(node.comparisons)
        values = node.comparisons.get(name)
        if values is not None:
            result[node_label] = dict(values)
    if not result:
        raise InvalidArgumentError(
            f"no comparison stored under {name!r}"
            + (f"; stored names: {sorted(available)}" if available else "; none stored yet"),
            code="comparison_unknown",
            remedy="store it first with bundle.store_comparison(...)",
            argument="name",
        )
    return result


def _bundle_stored_comparison_names(self: Bundle) -> tuple[str, ...]:
    """Return every comparison name stored on this bundle's supergraph.

    Returns
    -------
    tuple[str, ...]
        Sorted stored comparison names.
    """

    names: set[str] = set()
    for node in self.supergraph.nodes.values():
        names.update(node.comparisons)
    return tuple(sorted(names))


def _register_comparison_helpers(registry: dict[str, Callable[..., Any]]) -> None:
    """Register the stored-comparison helpers on the dynamic Bundle registry.

    Parameters
    ----------
    registry:
        Mutable name-to-callable mapping used by ``Bundle.__getattr__``.
    """

    registry["store_comparison"] = _bundle_store_comparison
    registry["stored_comparison"] = _bundle_stored_comparison
    registry["stored_comparison_names"] = _bundle_stored_comparison_names

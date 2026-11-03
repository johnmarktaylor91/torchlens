"""Named per-node comparison fields stored on a bundle's supergraph.

``Bundle.compare`` recomputes tensor distances on every call; these helpers
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
from .._selection.resolver import _resolve_sites
from .._selection.selectors import label as _exact_label
from ..intervention.types import Relationship
from ._aligned_sites import aligned_site_labels, resolve_aligned_site
from ._deltas import _distance_value

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

    Runs the per-node baseline comparison once, then stores
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
        Baseline member name or log; defaults to the bundle baseline.
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
        supergraph topological order.

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


def rank_changes(
    self: Bundle,
    baseline: str | Trace | None = None,
    *,
    k: int = 10,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
) -> list[tuple[str, float]]:
    """Rank sites by out distance from a baseline (the SITE axis)."""

    baseline_name = self._baseline_or_raise(baseline)
    # SITE AXIS (ledger memo D3f): most_changed answers "WHERE did the
    # bundle diverge" — per site it AVERAGES the metric from the baseline
    # to every other member. The member axis ("WHICH candidate mattered")
    # lives on the effect table, where most_changed is unambiguous.
    # top_k caps the RETURNED rows after every site is scored (a display
    # cap, never a compute bound or a coverage claim).
    # Operand scoping (A-GATE item 3): most_changed compares baseline
    # against every other member, never other members against each other,
    # so only baseline pairs must pass the comparison gate. Site reads
    # across all members still run the at() model-floor gate.
    self._require_comparable(
        "most_changed",
        pairs=[(baseline_name, name) for name in self._members if name != baseline_name],
    )
    baseline_log = self._members[baseline_name]
    scored: list[tuple[str, float]] = []
    baseline_sites = list(baseline_log._layer_list)
    if baseline_sites:  # the at() model-floor gate, once instead of per site
        names = list(self._members)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        self._require_comparable("at", pairs=pairs)
    aligned = aligned_site_labels(self._ensure_supergraph(), baseline_name)
    for site in baseline_sites:
        members = resolve_aligned_site(self._members, str(site.layer_label), aligned)
        if members is None:
            continue
        outs = {
            name: getattr(member, "out", None)
            for name, member in members.items()
            if getattr(member, "has_saved_activation", False)
        }
        base = outs.get(baseline_name)
        if not isinstance(base, torch.Tensor):
            continue
        distances: list[float] = []
        for name, out in outs.items():
            if name == baseline_name or not isinstance(out, torch.Tensor):
                continue
            label = str(site.layer_label)
            distances.append(_distance_value(base, out, metric, site=label))
        if distances:
            scored.append((str(site.layer_label), sum(distances) / len(distances)))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:k]


def compare_members(
    self: Bundle,
    a: str,
    b: str,
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
    on: Literal["out", "grad"] = "out",
) -> list[tuple[str, float]]:
    """Per-site differences between two members (gated)."""

    self._require_comparable("diff", pairs=[(a, b)])
    return self._diff_members(a, b, metric=metric, on=on)


def diff_members(
    self: Bundle,
    left_name: str,
    right_name: str,
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
    on: Literal["out", "grad"] = "out",
) -> list[tuple[str, float]]:
    """Per-site differences between two members, paired through the structural alignment."""

    left_log = self._members[left_name]
    right_log = self._members[right_name]
    # _distance_value holds the shared per-site rules: scalar fallback
    # only when both operands are scalar-like, identical payloads read
    # 0.0, different shapes refuse (typed) naming the site.
    rows: list[tuple[str, float]] = []
    aligned = aligned_site_labels(self._ensure_supergraph(), left_name)
    for left_site in left_log._layer_list:
        label = str(left_site.layer_label)
        right_label = aligned.get(label, {}).get(right_name, label)
        try:  # exact label: a bare string would be a substring selector
            right_sites = _resolve_sites(right_log, _exact_label(right_label), max_fanout=1)
        except Exception:  # noqa: BLE001 - missing sites are not common sites
            continue
        right_site = right_sites.first()
        left_out = getattr(left_site, on, None)
        right_out = getattr(right_site, on, None)
        if not isinstance(left_out, torch.Tensor) or not isinstance(
            right_out,
            torch.Tensor,
        ):
            continue
        rows.append((label, _distance_value(left_out, right_out, metric, site=label)))
    return rows


def comparability_table(self: Bundle) -> dict[tuple[str, str], Relationship]:
    """Pairwise relationships for all member pairs."""

    matrix: dict[tuple[str, str], Relationship] = {}
    for left in self._members:
        for right in self._members:
            matrix[(left, right)] = self.comparability(left, right)
    return matrix


def relationship_between(left: Trace, right: Trace) -> Relationship:
    """Derive relationship evidence for two model logs (identity ranks need LIVE evidence)."""

    if left is right:
        return Relationship.SAME_OBJECT

    left_class = left._source_class_qualname
    right_class = right._source_class_qualname
    left_weight = weight_fingerprint(left)
    right_weight = weight_fingerprint(right)

    left_model = weak_model(left)
    right_model = weak_model(right)
    if (
        left_model is not None
        and left_model is right_model
        and left_class is not None
        and left_class == right_class
    ):
        if left_weight is not None and left_weight == right_weight:
            return Relationship.SAME_OBJECT
        return Relationship.SAME_MODEL_OBJECT_AT_CAPTURE

    left_graph = left._provenance_fingerprint
    right_graph = right._provenance_fingerprint
    left_input = left._provenance_input_signature_hash
    right_input = right._provenance_input_signature_hash
    if left_graph is not None and left_graph == right_graph:
        if left_input is not None and left_input == right_input:
            return Relationship.SHARED_GRAPH_SAME_INPUT
        return Relationship.SHARED_GRAPH_DIFFERENT_INPUT

    if left_weight is not None and left_weight == right_weight:
        return Relationship.SAME_PARAM_SHAPES
    if left_class is not None and left_class == right_class:
        return Relationship.SHARED_ARCHITECTURE
    if left_class is not None and right_class is not None and left_class != right_class:
        return Relationship.DIFF_MODEL
    return Relationship.UNKNOWN


def weight_fingerprint(log: Trace) -> str | None:
    """Return the strongest available weight fingerprint."""

    return log._provenance_param_hash_full or log._provenance_param_hash_quick


def weak_model(log: Trace) -> Any | None:
    """Resolve a captured weak model reference."""

    ref = getattr(log, "_source_model_ref", None)
    if ref is None:
        return None
    try:
        return ref()
    except TypeError:
        return None

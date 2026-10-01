"""Member-pairwise delta and comparison helpers for :class:`Bundle`.

Split from ``bundle/__init__.py`` along the module-level helper seam (R43
file-size ceiling): every callable here is either a dynamic Bundle method
served by ``Bundle.__getattr__`` (``delta_map``, ``norm_delta``,
``output_delta``, ``compare``, ``aligned_pairs``, ``show_diff``) or a shared
metric/lookup primitive those methods and ``Bundle`` internals call
(``_distance_value``, ``_metric_label``, ``_tensor_field``, ...). Behavior is
unchanged; ``torchlens.bundle`` re-exports every name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..intervention._metrics import is_scalar_like, relative_l1_scalar, resolve_metric
from ..intervention.errors import BundleMemberError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from . import Bundle


def _metric_label(metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> str:
    """Return a stable label for a metric specifier.

    Parameters
    ----------
    metric:
        Metric name or callable.

    Returns
    -------
    str
        Human-readable metric label.
    """

    return metric if isinstance(metric, str) else getattr(metric, "__name__", "callable")


def _tensor_field(layer: Any, field: Literal["out", "grad"]) -> torch.Tensor | None:
    """Return a tensor field from a layer-like object.

    Parameters
    ----------
    layer:
        Layer or Op-like object.
    field:
        Tensor field to read.

    Returns
    -------
    torch.Tensor | None
        Tensor value when available.
    """

    value = getattr(layer, field, None)
    return value if isinstance(value, torch.Tensor) else None


def _distance_value(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """Return a Python float distance between two tensors.

    Parameters
    ----------
    reference:
        Reference tensor.
    candidate:
        Compared tensor.
    metric:
        Metric name or callable.

    Returns
    -------
    float
        Scalar distance.
    """

    metric_fn = resolve_metric(metric)
    value = (
        relative_l1_scalar(reference, candidate)
        if is_scalar_like(reference) and is_scalar_like(candidate)
        else metric_fn(reference, candidate)
    )
    return float(value.detach().item())


def _resolve_member_name(bundle: Bundle, member: str | Trace | None) -> str:
    """Resolve a member name or Trace reference within a bundle.

    Parameters
    ----------
    bundle:
        Bundle being queried.
    member:
        Member name, Trace reference, or ``None``.

    Returns
    -------
    str
        Resolved member name.
    """

    if member is None:
        return next(iter(bundle.names))
    if isinstance(member, str):
        if member not in bundle:
            raise KeyError(f"Unknown bundle member {member!r}.")
        return member
    for name, log in bundle.members.items():
        if log is member:
            return name
    raise KeyError("Trace is not a member of this Bundle.")


def _bundle_delta_map(
    self: Bundle,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    *,
    baseline: str | Trace | None = None,
    on: Literal["out", "grad"] = "out",
) -> dict[str, dict[str, float]]:
    """Return per-node tensor deltas from a baseline trace.

    Parameters
    ----------
    metric:
        Metric name from ``torchlens.intervention._metrics`` or a callable.
    baseline:
        Baseline member name or log. Defaults to the configured baseline, then
        the first member.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of supergraph node name to member-name distance values. Members
        without a tensor at a node are omitted for that node.
    """

    baseline_name = (
        self._baseline_or_raise(baseline)
        if baseline is not None or self.baseline_name is not None
        else next(iter(self.names))
    )
    result: dict[str, dict[str, float]] = {}
    supergraph = self.supergraph
    for graph_node_label in supergraph.topological_order:
        node = supergraph.nodes[graph_node_label]
        reference_layer = node.layer_refs.get(baseline_name)
        if reference_layer is None:
            continue
        reference = _tensor_field(reference_layer, on)
        if reference is None:
            continue
        values: dict[str, float] = {}
        for member_name in self.names:
            layer = node.layer_refs.get(member_name)
            candidate = _tensor_field(layer, on) if layer is not None else None
            if candidate is None:
                continue
            values[member_name] = (
                0.0
                if member_name == baseline_name
                else _distance_value(
                    reference,
                    candidate,
                    metric,
                )
            )
        if values:
            result[graph_node_label] = values
    return result


def _bundle_norm_delta(
    self: Bundle,
    *,
    baseline: str | Trace | None = None,
    on: Literal["out", "grad"] = "out",
) -> dict[str, dict[str, float]]:
    """Return relative L2 deltas for every comparable bundle node.

    Parameters
    ----------
    baseline:
        Baseline member name or log.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-node relative L2 distances keyed by member name.
    """

    return _bundle_delta_map(self, "relative_l2", baseline=baseline, on=on)


def _output_layer_pairs(
    target_log: Trace,
    candidate_log: Trace,
) -> list[tuple[Any, Any]]:
    """Return paired output layers by output index.

    Parameters
    ----------
    target_log:
        Reference model log.
    candidate_log:
        Compared model log.

    Returns
    -------
    list[tuple[Any, Any]]
        Paired output layer-like objects.
    """

    target_labels = list(getattr(target_log, "output_layers", []) or [])
    candidate_labels = list(getattr(candidate_log, "output_layers", []) or [])
    if target_labels and candidate_labels:
        # grind-r5 b7 R23 (sol HIGH): a silent shortest-prefix zip reported
        # only the surviving outputs' deltas, so a member that LOST an output
        # compared clean. Arity mismatch is a structural divergence and must
        # refuse, never truncate.
        if len(target_labels) != len(candidate_labels):
            raise BundleMemberError(
                f"output comparison refused: the target trace has "
                f"{len(target_labels)} output layers but the member has "
                f"{len(candidate_labels)} ({target_labels!r} vs {candidate_labels!r}); "
                "the graphs are structurally divergent, so a per-output delta "
                "would silently ignore the missing/extra outputs."
            )
        pairs: list[tuple[Any, Any]] = []
        for target_label, candidate_label in zip(target_labels, candidate_labels, strict=True):
            try:
                pairs.append((target_log[target_label], candidate_log[candidate_label]))
            except (KeyError, IndexError):
                continue
        return pairs
    target_layers = list(getattr(target_log, "layer_list", []))
    candidate_layers = list(getattr(candidate_log, "layer_list", []))
    return [(target_layers[-1], candidate_layers[-1])] if target_layers and candidate_layers else []


def _bundle_output_delta(
    self: Bundle,
    target: str | Trace,
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    on: Literal["out", "grad"] = "out",
) -> dict[str, dict[str, float]]:
    """Return output divergence for every member versus a target trace.

    Parameters
    ----------
    target:
        Target member name or ``Trace`` reference.
    metric:
        Metric name or callable.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, dict[str, float]]
        Member-keyed output distance mapping.
    """

    target_name = _resolve_member_name(self, target)
    target_log = self[target_name]
    result: dict[str, dict[str, float]] = {}
    for member_name, member_log in self.members.items():
        output_values: dict[str, float] = {}
        for output_index, (target_layer, member_layer) in enumerate(
            _output_layer_pairs(target_log, member_log)
        ):
            reference = _tensor_field(target_layer, on)
            candidate = _tensor_field(member_layer, on)
            if reference is None or candidate is None:
                continue
            label = str(getattr(target_layer, "layer_label", f"output_{output_index}"))
            output_values[label] = (
                0.0 if member_name == target_name else _distance_value(reference, candidate, metric)
            )
        result[member_name] = output_values
    return result


def _bundle_motif_occurrences(self: Bundle) -> dict[str, list[tuple[str, str]]]:
    """Return repeated operation-equivalence motifs across bundle traces.

    Parameters
    ----------
    self:
        Bundle being inspected.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        Operation-equivalence key to ``(member_name, layer_label)`` occurrences.
    """

    motifs: dict[str, list[tuple[str, str]]] = {}
    for member_name, member in self.members.items():
        for layer in getattr(member, "layer_list", []):
            key = getattr(layer, "equivalence_class", None)
            if not key:
                continue
            motifs.setdefault(str(key), []).append((member_name, str(layer.layer_label)))
    return {key: rows for key, rows in motifs.items() if len(rows) > 1}


def _bundle_compare(
    self: Bundle,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    *,
    baseline: str | Trace | None = None,
    on: Literal["out", "grad"] = "out",
) -> dict[str, Any]:
    """Return a unified bundle comparison payload.

    Parameters
    ----------
    metric:
        Metric name or callable.
    baseline:
        Baseline member name or log. Defaults like :meth:`delta_map`.
    on:
        Tensor field to compare.

    Returns
    -------
    dict[str, Any]
        Uniform payload with metric metadata, node deltas, output deltas, and
        repeated motif occurrences.
    """

    baseline_name = (
        self._baseline_or_raise(baseline)
        if baseline is not None or self.baseline_name is not None
        else next(iter(self.names))
    )
    return {
        "baseline": baseline_name,
        "metric": _metric_label(metric),
        "on": on,
        "nodes": _bundle_delta_map(self, metric, baseline=baseline_name, on=on),
        "outputs": _bundle_output_delta(self, baseline_name, metric=metric, on=on),
        "motifs": _bundle_motif_occurrences(self),
    }


def _alignment_score(left: Any, right: Any, left_index: int, right_index: int) -> float:
    """Return a conservative cross-architecture alignment score.

    Parameters
    ----------
    left:
        Left layer-like object.
    right:
        Right layer-like object.
    left_index:
        Topological index of ``left``.
    right_index:
        Topological index of ``right``.

    Returns
    -------
    float
        Heuristic score in ``[0, 1]``.
    """

    score = 0.0
    if getattr(left, "module", None) == getattr(right, "module", None):
        score += 0.35
    if getattr(left, "func_name", None) == getattr(right, "func_name", None):
        score += 0.35
    if getattr(left, "shape", None) == getattr(right, "shape", None):
        score += 0.2
    distance = abs(left_index - right_index)
    score += max(0.0, 0.1 - (distance * 0.01))
    return min(score, 1.0)


def _bundle_aligned_pairs(
    self: Bundle,
    left: str | Trace | None = None,
    right: str | Trace | None = None,
    *,
    min_score: float = 0.45,
) -> list[tuple[Any, Any]]:
    """Return best-match layer pairs across two bundle members.

    Alignment rules are intentionally conservative:

    1. Prefer exact module path and operation name matches.
    2. Use tensor shape and topological proximity to break ties.
    3. Pair each right-side layer at most once.

    Parameters
    ----------
    left:
        Left member name or log. Defaults to the first bundle member.
    right:
        Right member name or log. Defaults to the second bundle member.
    min_score:
        Minimum heuristic score required to emit a pair.

    Returns
    -------
    list[tuple[Any, Any]]
        Paired layer-like objects, ordered by the left trace.
    """

    names = self.names
    if len(names) < 2 and (left is None or right is None):
        raise ValueError("aligned_pairs requires at least two bundle members.")
    left_name = _resolve_member_name(self, left if left is not None else names[0])
    right_name = _resolve_member_name(self, right if right is not None else names[1])
    left_layers = list(getattr(self[left_name], "layer_list", []))
    right_layers = list(getattr(self[right_name], "layer_list", []))
    available_right = set(range(len(right_layers)))
    pairs: list[tuple[Any, Any]] = []
    for left_index, left_layer in enumerate(left_layers):
        best_index: int | None = None
        best_score = 0.0
        for right_index in available_right:
            score = _alignment_score(left_layer, right_layers[right_index], left_index, right_index)
            if score > best_score:
                best_index = right_index
                best_score = score
        if best_index is not None and best_score >= min_score:
            available_right.remove(best_index)
            pairs.append((left_layer, right_layers[best_index]))
    return pairs


def _bundle_show_diff(
    self: Bundle,
    *,
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "relative_l2",
    layout: Literal["paired"] = "paired",
    **kwargs: Any,
) -> str:
    """Render a two-column bundle diff for a clean/intervention pair.

    Examples
    --------
    >>> trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    >>> ablated = trace.fork("ablated")
    >>> ablated.do(tl.module("layer1.0.relu"), tl.zero_ablate())
    >>> bundle = tl.bundle({"clean": trace, "ablated": ablated}, baseline="clean")
    >>> bundle.show_diff(vis_outpath="bundle_diff_clean_vs_zero_relu")

    Parameters
    ----------
    metric:
        Metric forwarded to ``tl.viz.bundle_diff``.
    layout:
        Layout strategy forwarded to ``tl.viz.bundle_diff``.
    **kwargs:
        Additional renderer options forwarded unchanged.

    Returns
    -------
    str
        Graphviz DOT source for the rendered diff.
    """

    from ..visualization.bundle_diff import bundle_diff

    return bundle_diff(self, metric=metric, layout=layout, **kwargs)

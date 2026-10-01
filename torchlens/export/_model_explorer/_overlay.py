"""Node-data overlay writer over the existing overlay vocabulary (memo D8).

A thin serializer, not a second resolver: values come from the SAME closed
overlay vocabulary ``Trace.draw`` uses (``torchlens/visualization/
overlays.py``) plus its ``field:<name>`` / callable escape hatch, so every
proposed new stat is a one-liner, never a new preset vocabulary. Missing,
nonfinite, and rolled-varying values are OMITTED and counted by reason --
never a zero heatmap. Export never runs backward and never widens the save
policy: value-derived providers refuse typed when the capture retained no
facts to serialize.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from ..._errors import InvalidArgumentError
from ...errors._base import TorchLensError
from ...visualization.overlays import builtin_overlay_value, normalize_overlay_name
from ._errors import ModelExplorerExportError
from ._ids import PROXY_ID_PREFIX

__tl_layer__ = "L8"

#: The standard packet (memo D8): measurement providers always; value-derived
#: providers only by explicit request, because retention is capture policy.
STANDARD_OVERLAYS: tuple[str, ...] = ("time", "flops", "bytes", "nan")

#: Provider display names carry unit and aggregation (memo D8).
_PROVIDER_NAMES = {
    "time": "time (ms per op)",
    "flops": "flops (forward FLOPs per op)",
    "bytes": "bytes (activation B per op)",
    "nan": "nonfinite (1=nonfinite per op)",
    "magnitude": "magnitude (mean abs per op)",
    "grad_norm": "grad_norm (L2 per op)",
    "intervention": "intervention (fire count per op)",
    "bundle_delta": "bundle_delta (norm delta per op)",
}

#: Providers whose values derive from activation/gradient VALUES: dropped by
#: the public privacy profile BY CONSTRUCTION (memo D14) and refused on
#: captures that retained no values.
_VALUE_DERIVED = frozenset({"magnitude", "grad_norm", "nan"})

#: TL's colorblind-safe sequential ramp anchors (light theme), shared with
#: the draw() encoding channel.
_SEQUENTIAL_GRADIENT = ({"stop": 0, "bgColor": "#FFFFFF"}, {"stop": 1, "bgColor": "#0072B2"})


def build_node_data(
    log: Any,
    payload: dict[str, Any],
    source: str | Callable[[Any], Any],
    *,
    privacy_profile: str = "local",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one vendor-format node-data provider for every graph in a payload.

    Parameters
    ----------
    log:
        The exported trace (value source).
    payload:
        The collection payload produced by ``to_model_explorer_dict`` (node
        ids and ``torchlens_label`` attrs drive the join).
    source:
        Overlay preset name, ``"field:<record field>"``, or a callable
        ``node -> value``.
    privacy_profile:
        ``"public"`` refuses value-derived providers by construction.

    Returns
    -------
    tuple[dict, dict]
        The ``ModelNodeData``-shaped dict (graphsData by graph id) and a
        coverage disclosure (provider name, per-reason omission counts).
    """

    provider, slug, resolver = _resolve_source(log, source, privacy_profile)
    graphs_data: dict[str, Any] = {}
    coverage: dict[str, Any] = {
        "provider": provider,
        "provider_slug": slug,
        "resolved": 0,
        "omitted_missing": 0,
        "omitted_nonfinite": 0,
        "omitted_rolled_varying": 0,
        "omitted_unjoined": 0,
        "proxies_excluded": 0,
    }
    for graph in payload.get("graphs") or []:
        results = _graph_results(log, graph, resolver, coverage)
        if results:
            graphs_data[str(graph.get("id", ""))] = {
                "name": provider,
                "results": results,
                "gradient": [dict(item) for item in _SEQUENTIAL_GRADIENT],
            }
    if coverage["resolved"] == 0:
        raise ModelExplorerExportError(
            f"Node-data provider {provider!r} resolved zero values on this capture: "
            "the required facts were not retained at capture time, and export never "
            "widens the save policy or runs backward",
            code="model_explorer_value_not_retained",
            remedy=(
                "re-capture with the needed retention (a save= selection covering the "
                "target ops, or backward_ready/grad capture for gradient providers), "
                "or drop this provider"
            ),
            provider=provider,
        )
    return graphs_data, coverage


def _resolve_source(
    log: Any, source: str | Callable[[Any], Any], privacy_profile: str
) -> tuple[str, str, Callable[[Any], Any]]:
    """Resolve a provider source into (display name, slug, node resolver)."""

    if callable(source) and not isinstance(source, str):
        name = f"callable:{getattr(source, '__name__', 'anonymous')}"
        return name, _slug(name), _callable_resolver(source)
    if isinstance(source, str) and source.startswith("field:"):
        field_name = source[len("field:") :]
        return (
            f"field:{field_name} (record field per op)",
            _slug(f"field-{field_name}"),
            _field_resolver(field_name),
        )
    canonical = normalize_overlay_name(str(source))
    if privacy_profile == "public" and canonical in _VALUE_DERIVED:
        raise InvalidArgumentError(
            f"Overlay provider {canonical!r} is value-derived and the public privacy "
            "profile drops value-derived data by construction",
            code="model_explorer_overlay_public_conflict",
            remedy="export with privacy_profile='local', or drop this provider",
            argument="overlays",
        )
    if getattr(log, "structure_only", False) and canonical in _VALUE_DERIVED:
        raise ModelExplorerExportError(
            f"Overlay provider {canonical!r} needs activation/gradient values, and a "
            "structure-only capture retained none",
            code="model_explorer_value_not_retained",
            remedy="re-capture without structure_only=True, or drop this provider",
            provider=canonical,
        )
    provider = _PROVIDER_NAMES.get(canonical, f"{canonical} (per op)")
    return provider, _slug(canonical), _builtin_resolver(canonical)


def _builtin_resolver(canonical: str) -> Callable[[Any], Any]:
    """Adapt one closed-vocabulary overlay preset into a node resolver."""

    def resolve(node: Any) -> Any:
        """Return the preset's numeric value for one op node (ms for time)."""
        value = builtin_overlay_value(node, canonical)
        if canonical == "time" and value is not None:
            return float(value) * 1000.0
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    return resolve


def _field_resolver(field_name: str) -> Callable[[Any], Any]:
    """Adapt a record-field source into a node resolver."""

    def resolve(node: Any) -> Any:
        """Read the record field off one op node; absent reads as None."""
        return getattr(node, field_name, None)

    return resolve


def _callable_resolver(source: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a user callable so its exceptions surface typed, chained."""

    def resolve(node: Any) -> Any:
        """Call the user callable on one op node, chaining failures typed."""
        try:
            return source(node)
        except Exception as exc:
            raise ModelExplorerExportError(
                f"Overlay callable {getattr(source, '__name__', 'anonymous')!r} raised "
                f"on node {getattr(node, 'label', '<unknown>')!r}",
                code="model_explorer_overlay_callable_error",
                remedy="fix the callable to return a numeric value or None per node",
            ) from exc

    return resolve


def _graph_results(
    log: Any,
    graph: dict[str, Any],
    resolver: Callable[[Any], Any],
    coverage: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve one graph's node values, counting omissions by reason."""

    results: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes") or []:
        node_id = str(node.get("id", ""))
        if node_id.startswith(PROXY_ID_PREFIX):
            coverage["proxies_excluded"] += 1
            continue
        record = _join_record(log, node)
        if record is None:
            coverage["omitted_unjoined"] += 1
            continue
        value = resolver(record)
        if value is None:
            key = "omitted_rolled_varying" if _is_rolled_aggregate(record) else "omitted_missing"
            coverage[key] += 1
            continue
        numeric = _numeric_value(value, node_id)
        if numeric is None:
            coverage["omitted_nonfinite"] += 1
            continue
        results[node_id] = {"value": numeric}
        coverage["resolved"] += 1
    return results


def _join_record(log: Any, node: dict[str, Any]) -> Any | None:
    """Join one payload node back to its trace record via torchlens_label."""

    label = next(
        (
            str(attr.get("value", ""))
            for attr in node.get("attrs") or []
            if attr.get("key") == "torchlens_label"
        ),
        "",
    )
    if not label:
        return None
    try:
        return log[label]
    except (LookupError, TorchLensError):
        return None


def _is_rolled_aggregate(record: Any) -> bool:
    """Return whether a joined record is a multi-pass rolled aggregate."""

    ops = getattr(record, "ops", None)
    try:
        return ops is not None and len(ops) > 1
    except TypeError:
        return False


def _numeric_value(value: Any, node_id: str) -> float | int | None:
    """Coerce one resolved value to a finite number, or ``None`` to omit.

    Non-numeric values refuse typed: a provider emitting strings would
    silently render as a broken heatmap.
    """

    if isinstance(value, bool):
        return 1 if value else 0
    if not isinstance(value, (int, float)):
        raise ModelExplorerExportError(
            f"Overlay value {value!r} on node {node_id!r} is not numeric",
            code="model_explorer_overlay_value_invalid",
            remedy="providers must resolve to numbers or None (omit); fix the source",
        )
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _slug(name: str) -> str:
    """Return a filesystem-safe provider slug."""

    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-").lower() or "provider"

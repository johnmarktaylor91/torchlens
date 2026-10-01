"""Bundle value diff for Model Explorer's split pane (memo D15).

With the universal D2 id rule, two captures of the same architecture have
IDENTICAL id sets, so Model Explorer's own machinery does the matching: the
app's "Match node id" sync dropdown costs one documented step and the embed
self-activates config sync when ``syncNavigationData`` is present.
``bundle_delta`` node-data providers on both panes upgrade the viewer's
id-presence-only diff to numeric deltas. ``mappingEntries`` are emitted ONLY
for legacy-id nodes (with ``disableMappingFallback=true`` so accidental id
equality cannot lie); shape/dtype mismatch stays structural; one-sided nodes
stay native presence differences; missing values are coverage, never zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..._errors import InvalidArgumentError
from ...errors._base import TorchLensError
from ...utils.display import atomic_write_text
from ._collection import to_model_explorer_dict
from ._ids import LEGACY_ID_PREFIX
from ._validate import validate_model_explorer_payload

__tl_layer__ = "L8"

_DELTA_PROVIDER = "bundle_delta (L2 norm of out difference per op)"


def model_explorer_diff(
    members: Any,
    out_dir: str | Path,
    *,
    label: str | None = None,
    privacy_profile: str = "local",
) -> Path:
    """Export a two-capture value-diff product for the split-pane viewer.

    Parameters
    ----------
    members:
        A TorchLens ``Bundle`` with exactly two members, or a two-item
        sequence of ``Trace`` objects (subject, reference).
    out_dir:
        Destination directory for the paired collections, per-pane
        ``bundle_delta`` node data, sync config, manifest, and README.
    label:
        Optional base label for the two collections.
    privacy_profile:
        Forwarded to both exports.

    Returns
    -------
    Path
        The written directory.
    """

    left, right, names = _resolve_pair(members)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = str(label or getattr(left, "model_class_name", None) or "model")
    payloads = {}
    for side, trace, name in (("left", left, names[0]), ("right", right, names[1])):
        payload = to_model_explorer_dict(
            trace, label=f"{base} [{name}]", privacy_profile=privacy_profile
        )
        validate_model_explorer_payload(payload, strict=True)
        atomic_write_text(destination / f"{side}.json", json.dumps(payload, indent=2))
        payloads[side] = payload
    deltas, coverage = _pairwise_deltas(left, right, payloads["left"], payloads["right"])
    for side in ("left", "right"):
        atomic_write_text(
            destination / f"nodedata.{side}.bundle_delta.json",
            json.dumps(_delta_node_data(payloads[side], deltas), indent=2),
        )
    sync = _sync_navigation(payloads["left"], payloads["right"])
    atomic_write_text(
        destination / "embed-config.json", json.dumps({"syncNavigationData": sync}, indent=2)
    )
    if sync.get("mappingEntries"):
        atomic_write_text(destination / "sync-navigation.json", json.dumps(sync, indent=2))
    manifest = {
        "schema": "torchlens.model_explorer_diff_manifest.v1",
        "members": list(names),
        "delta_provider": _DELTA_PROVIDER,
        "coverage": coverage,
    }
    atomic_write_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    atomic_write_text(destination / "README.txt", _readme(names))
    return destination


def _resolve_pair(members: Any) -> tuple[Any, Any, tuple[str, str]]:
    """Resolve a Bundle or a two-item sequence into (left, right, names)."""

    names = getattr(members, "names", None)
    if names is not None:
        member_names = [str(name) for name in names]
        if len(member_names) != 2:
            raise InvalidArgumentError(
                f"Model Explorer value diff needs exactly two members; this bundle has "
                f"{len(member_names)}",
                code="model_explorer_diff_pair_invalid",
                remedy="pass a two-member Bundle or a (subject, reference) trace pair",
                argument="members",
            )
        return (
            members[member_names[0]],
            members[member_names[1]],
            (
                member_names[0],
                member_names[1],
            ),
        )
    pair = list(members) if isinstance(members, (tuple, list)) else []
    if len(pair) != 2:
        raise InvalidArgumentError(
            "Model Explorer value diff needs exactly two captures",
            code="model_explorer_diff_pair_invalid",
            remedy="pass a two-member Bundle or a (subject, reference) trace pair",
            argument="members",
        )
    return pair[0], pair[1], ("subject", "reference")


def _pairwise_deltas(
    left: Any, right: Any, left_payload: dict[str, Any], right_payload: dict[str, Any]
) -> tuple[dict[str, float], dict[str, int]]:
    """Compute per-node L2 deltas over the id-aligned node intersection."""

    left_labels = _labels_by_id(left_payload)
    right_labels = _labels_by_id(right_payload)
    coverage = {
        "aligned": 0,
        "one_sided": 0,
        "structural_mismatch": 0,
        "value_missing": 0,
    }
    deltas: dict[str, float] = {}
    for node_id, left_label in left_labels.items():
        right_label = right_labels.get(node_id)
        if right_label is None:
            coverage["one_sided"] += 1
            continue
        left_out = _saved_out(left, left_label)
        right_out = _saved_out(right, right_label)
        if left_out is None or right_out is None:
            coverage["value_missing"] += 1
            continue
        if tuple(left_out.shape) != tuple(right_out.shape) or left_out.dtype != right_out.dtype:
            coverage["structural_mismatch"] += 1
            continue
        difference = (left_out.detach().float() - right_out.detach().float()).norm()
        deltas[node_id] = float(difference.item())
        coverage["aligned"] += 1
    coverage["one_sided"] += sum(1 for node_id in right_labels if node_id not in left_labels)
    return deltas, coverage


def _labels_by_id(payload: dict[str, Any]) -> dict[str, str]:
    """Map the execution graph's node ids to their torchlens labels."""

    graphs = payload.get("graphs") or []
    if not graphs:
        return {}
    labels: dict[str, str] = {}
    for node in graphs[0].get("nodes") or []:
        label = next(
            (
                str(attr.get("value", ""))
                for attr in node.get("attrs") or []
                if attr.get("key") == "torchlens_label"
            ),
            "",
        )
        if label:
            labels[str(node.get("id", ""))] = label
    return labels


def _saved_out(trace: Any, label: str) -> Any | None:
    """Return one op's saved output tensor, or ``None`` when not retained."""

    try:
        record = trace[label]
        out = record.out
    except (AttributeError, LookupError, TorchLensError):
        return None
    return out if isinstance(out, torch.Tensor) else None


def _delta_node_data(payload: dict[str, Any], deltas: dict[str, float]) -> dict[str, Any]:
    """Build the per-pane vendor node-data dict for the delta provider."""

    graphs = payload.get("graphs") or []
    if not graphs:
        return {}
    graph_id = str(graphs[0].get("id", ""))
    results = {node_id: {"value": value} for node_id, value in deltas.items()}
    return {
        graph_id: {
            "name": _DELTA_PROVIDER,
            "results": results,
            "gradient": [
                {"stop": 0, "bgColor": "#FFFFFF"},
                {"stop": 1, "bgColor": "#D55E00"},
            ],
        }
    }


def _sync_navigation(left_payload: dict[str, Any], right_payload: dict[str, Any]) -> dict[str, Any]:
    """Build the sync-navigation config: mappings ONLY for legacy-id nodes."""

    sync: dict[str, Any] = {"type": "sync_navigation"}
    left_legacy = _legacy_ids(left_payload)
    right_legacy = _legacy_ids(right_payload)
    entries = [
        {"leftNodeIds": [node_id], "rightNodeIds": [node_id]}
        for node_id in sorted(left_legacy & right_legacy)
    ]
    if entries:
        sync["mappingEntries"] = entries
        sync["disableMappingFallback"] = True
    return sync


def _legacy_ids(payload: dict[str, Any]) -> set[str]:
    """Return the legacy-prefixed node ids of a payload's execution graph."""

    graphs = payload.get("graphs") or []
    if not graphs:
        return set()
    return {
        str(node.get("id", ""))
        for node in graphs[0].get("nodes") or []
        if str(node.get("id", "")).startswith(LEGACY_ID_PREFIX)
    }


def _readme(names: tuple[str, str]) -> str:
    """Write the split-pane instructions for the app target."""

    return (
        "TorchLens Model Explorer value diff\n"
        "===================================\n\n"
        f"left.json  = {names[0]}\n"
        f"right.json = {names[1]}\n\n"
        "App (ai-edge-model-explorer):\n"
        "  1. Open left.json, then 'Add to split pane' -> right.json.\n"
        "  2. In the sync-navigation dropdown pick 'Match node id'\n"
        "     (ids are aligned by construction across the two captures).\n"
        "  3. Upload nodedata.left.bundle_delta.json to the left pane and\n"
        "     nodedata.right.bundle_delta.json to the right pane to see\n"
        "     numeric value deltas as heatmaps.\n\n"
        "Embed (ai-edge-model-explorer-visualizer):\n"
        "  Pass embed-config.json's syncNavigationData through the visualizer\n"
        "  config; sync self-activates. sync-navigation.json exists only when\n"
        "  legacy-id nodes needed explicit mappingEntries.\n"
    )

"""Structured overview records and dump views (agent memo 3.3-3.4).

``overview(mode="manifest")`` is the torch-free bounded digest read from
validated manifest JSON only -- the only safe first look at an artifact the
caller did not produce (it never unpickles). ``overview(mode="folded")`` is
the DEFAULT structural orientation view: the recurrence fold plus capture
honesty, counts, coverage, audit, and next operations. The two modes return
DIFFERENT schema strings, never one schema with conditionally absent fields.
Neither wraps ``trace.summary()`` prose: records are structured, volatile
provenance is isolated in one block a diffing agent skips by key, and human
text renders FROM the record.
"""

from __future__ import annotations

from typing import Any

from ._fold import FOLD_KEY_VERSION, fold_class_rows, fold_trace

#: Schema ids -- the two overview modes are distinct schemas by contract.
OVERVIEW_MANIFEST_SCHEMA = "torchlens.agent.overview_manifest.v1"
OVERVIEW_FOLDED_SCHEMA = "torchlens.agent.overview_folded.v1"
DUMP_SCHEMA = "torchlens.agent.dump.v1"

#: Closed dump-view vocabulary.
DUMP_VIEWS = ("overview", "graph", "full")

#: Ceiling on labels listed in one anomaly disclosure.
_ANOMALY_LABELS_MAX = 8


def manifest_overview(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Build the torch-free manifest-mode overview record.

    Parameters
    ----------
    manifest:
        Parsed ``manifest.json`` mapping, or ``None`` when absent/unparseable
        (disclosed, never guessed).

    Returns
    -------
    dict[str, Any]
        Bounded ~12-field digest of manifest-declared facts only.
    """

    if manifest is None:
        return {
            "manifest_readable": False,
            "note": (
                "No parseable manifest.json; this is not a .tlspec directory "
                "artifact or it is corrupt. Nothing was unpickled."
            ),
        }
    from ._artifacts import declared_bytes_and_count

    declared_bytes, declared_count = declared_bytes_and_count(manifest)
    backward = manifest.get("backward_summary") or {}
    sites = manifest.get("sites")
    return {
        "manifest_readable": True,
        "kind": manifest.get("kind", "unknown"),
        "tlspec_version": manifest.get("tlspec_version"),
        "save_level": manifest.get("save_level"),
        "model_signature": manifest.get("model_signature"),
        "n_layers": manifest.get("n_layers"),
        "n_sites": len(sites) if isinstance(sites, list) else None,
        "declared_payload_bytes": declared_bytes,
        "declared_payload_count": declared_count,
        "has_backward_pass": bool(backward.get("has_backward_pass", False)),
        "gradient_blob_count": backward.get("gradient_blob_count", 0),
        # Volatile provenance isolated in ONE block a diffing agent skips.
        "provenance": {
            "created_at": manifest.get("created_at"),
            "torchlens_version_at_save": manifest.get("torchlens_version"),
            "torch_version_at_save": manifest.get("torch_version"),
            "platform": manifest.get("platform"),
        },
    }


def _coverage_block(log: Any) -> dict[str, Any]:
    """Payload/gradient coverage counts an agent needs before value work."""

    ops = list(getattr(log, "layer_list", []) or [])
    from ..report._agent_json import _payload_state

    states = [_payload_state(op) for op in ops]
    saved_grads = sum(
        1 for op in ops if getattr(op, "grad", None) is not None or getattr(op, "grad_ref", None)
    )
    return {
        "ops_total": len(ops),
        "payload_present": states.count("present"),
        "payload_lazy": states.count("lazy"),
        "payload_unsaved": states.count("unsaved"),
        "grads_available": saved_grads,
    }


def _audit_block(log: Any) -> dict[str, Any]:
    """The always-present audit block: health verdict + non-finite facts."""

    from ..report._agent_json import _health_summary

    health = _health_summary(log)
    coverage = getattr(log, "nonfinite_coverage", None)
    nonfinite = {
        "labels": health["nonfinite_labels"][:_ANOMALY_LABELS_MAX],
        "n_labels": len(health["nonfinite_labels"]),
        "basis": str(getattr(coverage, "basis", "unknown")) if coverage is not None else "unknown",
    }
    return {"health": health, "nonfinite": nonfinite}


def _intervention_block(log: Any) -> dict[str, Any] | None:
    """Bounded structured intervention evidence, or ``None`` when clean."""

    audit = list(getattr(log, "intervention_audit", []) or [])
    if not audit:
        return None
    last = audit[-1]
    return {
        "n_records": len(audit),
        "last_record": {
            str(key): str(value)[:200]
            for key, value in (last.items() if isinstance(last, dict) else [])
        }
        or str(last)[:200],
    }


def _anomalies_block(log: Any, capture: dict[str, Any]) -> list[dict[str, Any]]:
    """Notable anomalies an agent must not miss, each one structured row."""

    anomalies: list[dict[str, Any]] = []
    if not capture.get("capture_verified", True):
        anomalies.append(
            {
                "kind": "capture_unverified",
                "detail": str(capture.get("capture_verification_reason")),
            }
        )
    if capture.get("structure_only"):
        anomalies.append(
            {
                "kind": "structure_only",
                "detail": "shapes/dtypes are hypotheses, not measurements",
            }
        )
    if capture.get("poisoned"):
        anomalies.append(
            {"kind": "poisoned", "detail": "diverged sparse run; values are not model-faithful"}
        )
    nonfinite = list(getattr(log, "nonfinite_ops", []) or [])
    if nonfinite:
        anomalies.append(
            {
                "kind": "nonfinite_ops",
                "detail": f"{len(nonfinite)} op(s) held NaN/Inf",
                "labels": [str(label) for label in nonfinite[:_ANOMALY_LABELS_MAX]],
            }
        )
    return anomalies


def _next_operations(has_payloads: bool) -> dict[str, str]:
    """Concrete next tool calls (agent-surface spellings, always runnable)."""

    steps = {
        "sites": 'call_tool("torchlens_query_sites", {"path": <path>})',
        "graph_rows": 'call_tool("torchlens_dump", {"path": <path>, "view": "graph"})',
        "report": 'call_tool("torchlens_explain", {"path": <path>, "max_tokens": 2000})',
    }
    if has_payloads:
        steps["values"] = 'call_tool("torchlens_payload_stats", {"path": <path>})'
    return steps


def folded_overview(log: Any) -> dict[str, Any]:
    """Build the default folded structural orientation record.

    Parameters
    ----------
    log:
        Loaded ``Trace``.

    Returns
    -------
    dict[str, Any]
        Folded overview data block (fold classes, capture honesty, counts,
        coverage, audit, interventions, anomalies, next operations).
    """

    from ..report._agent_json import build_agent_json

    skeleton = build_agent_json(log, max_ops=1)
    capture = skeleton["capture"]
    fold = fold_trace(log)
    coverage = _coverage_block(log)
    data: dict[str, Any] = {
        "fold_key": FOLD_KEY_VERSION,
        "classes": fold_class_rows(fold),
        "n_classes": len(fold.classes),
        "capture": capture,
        "counts": skeleton["counts"],
        "memory": skeleton["memory"],
        "inputs": skeleton["inputs"],
        "outputs": skeleton["outputs"],
        "coverage": coverage,
        "audit": _audit_block(log),
        "logged_values": skeleton["logged_values"],
        "interventions": _intervention_block(log),
        "anomalies": _anomalies_block(log, capture),
        "next_operations": _next_operations(
            coverage["payload_present"] + coverage["payload_lazy"] > 0
        ),
    }
    return data


def dump_view(
    log: Any,
    *,
    view: str,
    max_rows: int,
    offset: int = 0,
    class_id: str | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Build one dump view's data block plus row-paging facts.

    Parameters
    ----------
    log:
        Loaded ``Trace``.
    view:
        One of :data:`DUMP_VIEWS`.
    max_rows:
        Row cap for the ``graph`` view (paged, execution-ordered).
    offset:
        Row offset for the ``graph`` view.
    class_id:
        Optional fold-class filter for drill-down (``graph`` view only).

    Returns
    -------
    tuple[dict, int, int]
        Data block, rows included, rows omitted.

    Raises
    ------
    InvalidArgumentError
        ``agent_view_invalid`` on an unknown view or class id.
    """

    from .._errors import InvalidArgumentError
    from ..report._agent_json import _op_entry, build_agent_json

    if view not in DUMP_VIEWS:
        raise InvalidArgumentError(
            f"view={view!r} is not a dump view",
            code="agent_view_invalid",
            remedy=f"pass one of {', '.join(DUMP_VIEWS)}",
        )
    if view == "overview":
        return folded_overview(log), 0, 0
    if view == "full":
        return build_agent_json(log), 0, 0
    ops = list(getattr(log, "layer_list", []) or [])
    if class_id is not None:
        fold = fold_trace(log)
        by_id = {cls.class_id: set(cls.members) for cls in fold.classes}
        if class_id not in by_id:
            raise InvalidArgumentError(
                f"class_id={class_id!r} names no fold class",
                code="agent_view_invalid",
                remedy='list classes via overview mode="folded" first',
                known_classes=len(by_id),
            )
        members = by_id[class_id]
        ops = [op for op in ops if str(getattr(op, "label", "")) in members]
    total = len(ops)
    page = ops[offset : offset + max_rows]
    rows = [_op_entry(op) for op in page]
    data = {
        "rows": rows,
        "rows_total": total,
        "offset": offset,
        "class_id": class_id,
    }
    return data, len(rows), max(0, total - offset - len(rows))

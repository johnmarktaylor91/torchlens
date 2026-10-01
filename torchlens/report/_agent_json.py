"""Machine-readable trace dump for agent consumers (``Trace.to_agent_json``).

DOCUMENTED-UNSTABLE spelling (naming ratification deferred to the UI/naming
sprint). The dump describes the SAME public surface a human drives -- it is a
navigation aid, never a parallel API: every record points back at the live
spelling (``trace[label]``, ``tl.func(...)``, ``tl.report.explain(...)``) an
agent should call next.

Honesty contract (report/AGENTS.md): capture outcome/verification facts are
carried verbatim from the same source ``explain()`` uses; a rescued, ceilinged,
or structure-only capture must stay visible in the dump.
"""

from __future__ import annotations

from typing import Any

from .._capture_honesty import (
    capture_advisories,
    capture_verification,
    episode_facts,
    poison_facts,
    refuse_presenter_subject,
)
from ._explain import _safe_len

#: Schema identifier for the agent trace dump.
AGENT_TRACE_SCHEMA = "torchlens.agent_trace.v1"

#: Static self-description embedded in every dump so an agent can navigate the
#: trace without reading prose docs first. ``next_steps`` is built per trace
#: (:func:`_guide_next_steps`) so every listed spelling is executable by the
#: dump's reader in its current state -- an entry that needs objects the
#: reader does not hold (a live ``model``/``x``) teaches a crash.
_GUIDE: dict[str, Any] = {
    "purpose": (
        "Structural dump of one captured forward pass. Use it to discover "
        "layer labels, graph edges, module structure, and capture-honesty "
        "facts, then drive the live object with the spellings below."
    ),
    "payloads": (
        "Tensor values are never inlined here. Read a saved activation as "
        "trace[<layer_label>].out; ops with saved=false raise on payload "
        "reads -- re-capture with a wider save= predicate (e.g. "
        "tl.trace(model, x, save=tl.func('relu')))."
    ),
    "navigation": {
        "ops": (
            "Execution-ordered operation records. 'label' is the unique "
            "pass-qualified id (layer_label:pass); 'layer_label' addresses "
            "the rolled layer via trace[layer_label]. 'parents'/'children' "
            "hold layer_labels of adjacent ops in the dataflow graph."
        ),
        "modules": (
            "Module hierarchy rows keyed by dotted address; 'address_parent'"
            " / 'address_children' give the containment tree."
        ),
        "capture": (
            "Settled capture facts. capture_verified=false means parts of "
            "the forward may be missing or unattributed -- treat every count "
            "as a lower bound. structure_only=true means shapes/dtypes are "
            "hypotheses, not measurements. poisoned=true means this is a "
            "diverged sparse run kept for inspection; its values are NOT "
            "model-faithful. An 'episode' block means one wrapped multi-step "
            "generation run; fidelity_basis='forced' is the non-verifying "
            "teacher-forcing mode."
        ),
        "truncation": (
            "Non-null when max_ops dropped op rows; counts disclose exactly what was omitted."
        ),
    },
}


def _guide_next_steps(log: Any) -> dict[str, str]:
    """Build the per-trace executable next-step menu.

    Every entry must run for a reader holding ONLY ``trace`` (plus an
    installed torchlens): no entry may require the live ``model``/``x``
    objects, and state-dependent entries appear only when this trace can
    honor them.

    Parameters
    ----------
    log:
        Completed trace the dump describes.

    Returns
    -------
    dict[str, str]
        Ordered next-step spellings.
    """

    steps: dict[str, str] = {
        "summary": "trace.summary()",
        "plain_language_report": "tl.report.explain(trace)",
        "budgeted_report": "tl.report.explain(trace, max_tokens=500)",
        "health_audit": "trace.audit()",
        "resource_profile": "trace.profile(level='module')",
        "inventory": "trace.bill_of_materials()",
        "nonfinite_evidence": "trace.first_nonfinite()",
        "nonfinite_coverage": "trace.nonfinite_coverage",
    }
    if int(getattr(log, "num_saved_ops", 0) or 0) > 0:
        steps["one_activation"] = "trace[<layer_label>].out"
    steps["receptive_field"] = "trace[<layer_label>].receptive_field"
    steps["draw_graph"] = "trace.draw()"
    if _pandas_available():
        steps["op_table"] = "trace.to_pandas()"
        if _first_op_site_key(log) is not None:
            steps["sites_table"] = "trace.sites_table()"
        if getattr(log, "decoded_output", None) is not None:
            steps["output_table"] = "trace.output_table(top_n=5)"
    logged = (getattr(log, "annotations", {}) or {}).get("logged_values")
    if isinstance(logged, dict) and logged:
        steps["logged_values"] = "trace.logged_values"
    steps["environment_diagnosis"] = "tl.utils.doctor()"
    return steps


def _pandas_available() -> bool:
    """Return whether pandas is importable in the reader's environment."""

    from importlib.util import find_spec

    try:
        return find_spec("pandas") is not None
    except (ImportError, ValueError):
        return False


def _first_op_site_key(log: Any) -> str | None:
    """Return the first op's site key, or ``None`` on keyless artifacts."""

    for op in getattr(log, "layer_list", []) or []:
        return getattr(op, "site_key", None)
    return None


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-primitive projection of one logged value, bounded.

    Parameters
    ----------
    value:
        User-recorded value from ``log_value``.

    Returns
    -------
    Any
        The value itself when it is a JSON primitive, otherwise its ``repr``
        truncated to 500 characters (disclosed with a trailing ellipsis).
    """

    if value is None or isinstance(value, bool | int | float | str):
        return value
    rendered = repr(value)
    if len(rendered) > 500:
        rendered = rendered[:497] + "..."
    return rendered


def _quantity_int(value: Any) -> int | None:
    """Coerce a TorchLens quantity (Flops/Bytes) to a plain integer.

    Parameters
    ----------
    value:
        Quantity-like object or ``None``.

    Returns
    -------
    int | None
        Integer value, or ``None`` when unavailable or non-numeric.
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_shape(shape: Any) -> list[int] | None:
    """Return a JSON-safe copy of a recorded output shape.

    Parameters
    ----------
    shape:
        Recorded shape tuple, or ``None``.

    Returns
    -------
    list[int] | None
        Plain list of dimension sizes, or ``None`` when unrecorded.
    """

    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _memory_scopes(log: Any) -> dict[str, Any]:
    """Both payload-scope byte figures, named (C02; sumfam D8)."""

    from ._factcore import _memory

    memory = _memory(log)
    return {
        "at_capture_bytes": memory.at_capture_bytes,
        "at_capture_saved_ops": memory.at_capture_saved_ops,
        "retained_now_bytes": memory.retained_now_bytes,
        "retained_now_present_ops": memory.retained_now_present_ops,
        "retained_now_lazy_ops": memory.retained_now_lazy_ops,
        "scope_note": memory.scope_note,
    }


def _factcore_counts(log: Any, *, module_row_count: int) -> dict[str, Any]:
    """The counts block from the ONE numbers core (F09; sumfam D3/D17).

    Both grains print under their names -- ``operations`` counts every
    tracked tensor row (the historical meaning) while ``compute_ops`` and
    ``alias_rows`` name the identity-partition split.
    """

    try:
        from ._factcore import factcore

        core = factcore(log)
    except Exception:  # noqa: BLE001 -- foreign/partial logs keep the raw fallback
        return {
            "layers": _safe_len(getattr(log, "layer_labels", None)),
            "operations": int(getattr(log, "num_ops", 0) or 0),
            "tensors_total": int(getattr(log, "num_tensors", 0) or 0),
            "tensors_saved": int(getattr(log, "num_saved_ops", 0) or 0),
            "parameters": int(getattr(log, "num_params", 0) or 0),
            "modules": module_row_count,
        }
    return {
        "layers": core.counts.layers,
        "operations": core.counts.tracked_tensor_rows,
        "compute_ops": core.counts.compute_ops,
        "alias_rows": core.counts.alias_rows,
        "tensors_total": int(getattr(log, "num_tensors", 0) or 0),
        "tensors_saved": core.memory.at_capture_saved_ops,
        "parameters": core.params.total or 0,
        "modules": module_row_count,
    }


def _health_summary(log: Any) -> dict[str, Any]:
    """The three-state health verdict + coverage (C02; sumfam D5/D9).

    D4 (F09 item 13): a machine dump never implicitly pays for a payload
    scan -- this serves the basis in hand and reads NOT-CHECKED otherwise
    (the explicit scan spelling is ``tl.report.health_facts(trace)``).
    """

    from ._health import health_facts

    facts = health_facts(log, allow_scan=False)
    return {
        "verdict": facts.verdict,
        "basis": facts.basis,
        "checked": facts.checked,
        "nonfinite_labels": list(facts.nonfinite_labels),
        "alias_nonfinite_labels": list(facts.alias_nonfinite_labels),
        "unexamined": facts.unexamined,
        "unchecked": facts.unchecked,
    }


def _payload_state(op: Any) -> str:
    """Return the op's CURRENT payload state: present | lazy | unsaved.

    Capture-time ``saved`` used to mean two different things (agent memo P0
    part 4): "a payload is resident" and "a lazy ref could materialize one".
    This is the resident-now answer, read through the private slot accessor
    so the probe itself never triggers a materializing ``op.out`` read. A
    row with ``saved=true`` and ``payload_state="unsaved"`` is a
    payload-stripped artifact -- saved at capture, no bytes here.

    Parameters
    ----------
    op:
        Operation record.

    Returns
    -------
    str
        ``"present"`` (resident tensor), ``"lazy"`` (materializable blob
        ref), or ``"unsaved"`` (no payload here).
    """

    slot = getattr(op, "_slot", None)
    if callable(slot):
        if slot("out") is not None:
            return "present"
        if slot("out_ref") is not None:
            return "lazy"
        return "unsaved"
    if getattr(op, "out_ref", None) is not None:
        return "lazy"
    return "present" if getattr(op, "has_saved_activation", False) else "unsaved"


def _site_key_or_none(op: Any) -> str | None:
    """Return the op's structural site key, or ``None`` when unavailable.

    ``Op.site_key`` is a plain persisted field (``str | None``); legacy
    keyless artifacts carry ``None``, which the dump discloses as ``null``.

    Parameters
    ----------
    op:
        Operation record.

    Returns
    -------
    str | None
        ``site_key_v1`` string, or ``None`` on keyless artifacts.
    """

    key = getattr(op, "site_key", None)
    return str(key) if key is not None else None


def _op_entry(op: Any) -> dict[str, Any]:
    """Build one JSON-safe operation row.

    Parameters
    ----------
    op:
        Operation record from ``trace.layer_list``.

    Returns
    -------
    dict[str, Any]
        Flat JSON-serializable record for the op.
    """

    device_ref = getattr(op, "device_ref", None)
    dtype = getattr(op, "dtype", None)
    call_stack = [str(m) for m in getattr(op, "module_call_stack", ()) or ()]
    # FF1 (agent memo 3.5): `module_address` is the ATOMIC-module address and
    # is null on 90-91% of real-transformer rows, so every module-grain
    # rollup collapsed into one None bucket. `module_containment` is the
    # call-stack-derived innermost containing module -- populated on every
    # op that ran inside any module.
    containment = call_stack[-1].rsplit(":", 1)[0] if call_stack else None
    return {
        "payload_state": _payload_state(op),
        "label": str(getattr(op, "label", getattr(op, "layer_label", "unknown"))),
        "layer_label": str(getattr(op, "layer_label", "unknown")),
        "pass_index": int(getattr(op, "pass_index", 1) or 1),
        "num_passes": int(getattr(op, "num_passes", 1) or 1),
        "func_name": str(getattr(op, "func_name", "unknown")),
        "shape": _json_shape(getattr(op, "shape", None)),
        "dtype": str(dtype) if dtype is not None else None,
        "device": str(getattr(device_ref, "name", device_ref)) if device_ref else None,
        "parents": [str(p) for p in getattr(op, "parents", ()) or ()],
        "children": [str(c) for c in getattr(op, "children", ()) or ()],
        "module_call_stack": call_stack,
        "module_address": getattr(op, "atomic_module_address", None),
        "module_containment": containment,
        "saved": bool(getattr(op, "has_saved_activation", False)),
        "site_key": _site_key_or_none(op),
        "num_params": int(getattr(op, "num_params", 0) or 0),
        "flops_forward": _quantity_int(getattr(op, "flops_forward", None)),
    }


def _module_entry(address: str, module: Any) -> dict[str, Any]:
    """Build one JSON-safe module hierarchy row.

    Parameters
    ----------
    address:
        Dotted module address key from ``trace.modules``.
    module:
        Module record.

    Returns
    -------
    dict[str, Any]
        Flat JSON-serializable record for the module.
    """

    parent = getattr(module, "address_parent", None)
    return {
        "address": str(address),
        "class_name": str(getattr(module, "class_name", type(module).__name__)),
        "num_calls": _safe_len(getattr(module, "calls", None)),
        "address_parent": str(parent) if parent is not None else None,
        "address_children": [str(c) for c in getattr(module, "address_children", ()) or ()],
        "num_params": int(getattr(module, "num_params", 0) or 0),
    }


def _op_labels(accessor: Any) -> list[str]:
    """Return pass-qualified labels from an op accessor or label sequence.

    Parameters
    ----------
    accessor:
        ``TraceOpAccessor`` (yields op records) or plain label sequence.

    Returns
    -------
    list[str]
        Pass-qualified op labels.
    """

    if accessor is None:
        return []
    labels: list[str] = []
    for item in accessor:
        labels.append(str(getattr(item, "label", item)))
    return labels


def build_agent_json(log: Any, *, max_ops: int | None = None) -> dict[str, Any]:
    """Build the self-describing machine-readable dump of a finished trace.

    Parameters
    ----------
    log:
        Completed ``Trace``.
    max_ops:
        Optional cap on emitted op rows (execution order, first ``max_ops``
        kept). Omission is disclosed in the ``truncation`` block, never
        silent.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dump under the ``torchlens.agent_trace.v1`` schema.

    Raises
    ------
    ValueError
        If ``max_ops`` is not a positive integer.
    """

    refuse_presenter_subject(log, "Trace.to_agent_json")
    if max_ops is not None and (
        isinstance(max_ops, bool) or not isinstance(max_ops, int) or max_ops < 1
    ):
        raise ValueError(
            "max_ops must be a positive integer (the cap on emitted op rows); "
            "omit it to dump every op."
        )

    ops = list(getattr(log, "layer_list", []) or [])
    truncation: dict[str, Any] | None = None
    if max_ops is not None and len(ops) > max_ops:
        truncation = {
            "ops_included": max_ops,
            "ops_omitted": len(ops) - max_ops,
            "policy": "first max_ops rows in execution order",
            "note": (
                "Op rows were dropped to honor max_ops; counts above remain "
                "the full-capture truth. Re-dump without max_ops for the "
                "complete graph."
            ),
        }
        ops = ops[:max_ops]

    capture = {
        **capture_verification(log),
        "backend": str(getattr(log, "backend", "unknown")),
        "model_class": str(getattr(log, "model_class_name", type(log).__name__)),
        "structure_only": bool(getattr(log, "structure_only", False)),
        "grouping": str(getattr(log, "grouping", "unknown")),
        "has_backward_pass": bool(getattr(log, "has_backward_pass", False)),
        "device_summary": getattr(log, "backend_runtime_device_summary", None),
        # Poison, episode, and advisory facts are honesty disclosures: a
        # diverged sparse run or a forced-tokens episode must never dump as an
        # ordinary clean forward (WT1 A-V row 23).
        **poison_facts(log),
    }
    episode = episode_facts(log)
    if episode is not None:
        capture["episode"] = episode
    advisories = capture_advisories(log)
    if advisories:
        capture["advisories"] = advisories

    modules_map = getattr(log, "modules", {}) or {}
    module_rows = [
        _module_entry(address, module)
        for address, module in modules_map.items()
        if isinstance(address, str)
    ]

    guide = dict(_GUIDE)
    guide["next_steps"] = _guide_next_steps(log)

    logged_values = {
        str(name): _json_safe_value(value)
        for name, value in (
            (getattr(log, "annotations", {}) or {}).get("logged_values", {}) or {}
        ).items()
    }

    return {
        "schema": AGENT_TRACE_SCHEMA,
        "schema_stability": (
            "documented-unstable: field names may be renamed by the naming "
            "ratification sprint; branch on 'schema' before hard-coding."
        ),
        "guide": guide,
        "capture": capture,
        "logged_values": logged_values,
        # ONE machine core (sumfam D17/item 14): counts read the same
        # FactCore sections explain-json reads; the parity gate pins every
        # overlapping field so the two vocabularies cannot drift apart.
        "counts": _factcore_counts(log, module_row_count=len(module_rows)),
        # Payload-scope law (C02; sumfam D8): every byte figure names its
        # scope -- at_capture (immutable capture fact) vs retained_now
        # (what THIS object holds; zero on a payload-stripped artifact).
        "memory": _memory_scopes(log),
        "health": _health_summary(log),
        "inputs": _op_labels(getattr(log, "input_ops", None)),
        "outputs": _op_labels(getattr(log, "output_ops", None)),
        "layer_labels": [str(label) for label in getattr(log, "layer_labels", []) or []],
        "ops": [_op_entry(op) for op in ops],
        "modules": module_rows,
        "truncation": truncation,
    }

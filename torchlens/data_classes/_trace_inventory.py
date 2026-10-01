"""Trace-level inventory builders: structural sites table and bill of materials.

Both surfaces are read-only rollups of already-captured facts — they mint no
new claims, and every number is derived from record fields the trace already
carries. Spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    import pandas as pd

    from .trace import Trace


def _require_pandas() -> Any:
    """Import pandas with the TorchLens tabular-extra error message.

    Returns
    -------
    Any
        Imported pandas module.
    """

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from e
    return pd


def build_sites_table(trace: Trace) -> pd.DataFrame:
    """Build the tabular view of a trace's structural sites.

    One row per distinct L1 ``site_key`` in first-occurrence execution order,
    aggregating the ops that share the site (reused-module calls share keys
    across instances by design).

    Parameters
    ----------
    trace:
        Completed trace.

    Returns
    -------
    pandas.DataFrame
        Columns ``site_key``, ``module_site``, ``layer_type``,
        ``output_slot``, ``call_ordinal``, ``n_ops``, ``labels``,
        ``layer_labels``, ``passes``, and ``shapes``.

    Raises
    ------
    InvalidArgumentError
        ``site_key_unavailable`` when ops exist but none carries a site key
        (legacy pre-site-key artifact) — consistent with the L1 accessors,
        never a silently empty table.
    """

    from ._site_key import parse_site_key

    pd = _require_pandas()
    ops = list(trace.ops)
    keyed = [op for op in ops if getattr(op, "site_key", None) is not None]
    if ops and not keyed:
        raise InvalidArgumentError(
            "This trace's ops carry no site keys: it was captured/saved "
            "before site_key_v1 existed, so a sites table would be "
            "silently empty.",
            code="site_key_unavailable",
            remedy="re-capture with a current TorchLens to mint site keys",
        )

    sites: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for op in keyed:
        key = op.site_key
        entry = sites.get(key)
        if entry is None:
            module_site, layer_type, output_slot, call_ordinal = parse_site_key(key)
            entry = sites[key] = {
                "site_key": key,
                "module_site": "/".join(module_site),
                "layer_type": layer_type,
                "output_slot": output_slot,
                "call_ordinal": call_ordinal,
                "n_ops": 0,
                "labels": [],
                "layer_labels": [],
                "passes": [],
                "shapes": [],
            }
        entry["n_ops"] += 1
        entry["labels"].append(op.label)
        layer_label = getattr(op, "layer_label", None)
        if layer_label is not None and layer_label not in entry["layer_labels"]:
            entry["layer_labels"].append(layer_label)
        entry["passes"].append(getattr(op, "pass_index", 1))
        shape = getattr(op, "shape", None)
        if shape not in entry["shapes"]:
            entry["shapes"].append(shape)

    rows = []
    for entry in sites.values():
        rows.append(
            {
                **entry,
                "labels": tuple(entry["labels"]),
                "layer_labels": tuple(entry["layer_labels"]),
                "passes": tuple(entry["passes"]),
                "shapes": tuple(entry["shapes"]),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "site_key",
            "module_site",
            "layer_type",
            "output_slot",
            "call_ordinal",
            "n_ops",
            "labels",
            "layer_labels",
            "passes",
            "shapes",
        ],
    )


def build_bill_of_materials(trace: Trace) -> dict[str, Any]:
    """Build the inventory of what one trace actually contains.

    Every figure is read from fields the trace already carries; sections
    report presence honestly (an empty backward section means no backward
    facts were captured, not that none happened).

    Parameters
    ----------
    trace:
        Completed trace.

    Returns
    -------
    dict[str, Any]
        Nested sections: ``capture``, ``graph``, ``parameters``,
        ``buffers``, ``activations``, ``backward``, and ``annotations``.
    """

    from ..quantities import Bytes

    ops = list(trace.ops)
    saved_ops = [op for op in ops if getattr(op, "has_saved_activation", False)]
    saved_bytes = Bytes(
        sum(
            int(memory)
            for op in saved_ops
            if (memory := getattr(op, "activation_memory", None)) is not None
        )
    )
    outcome = getattr(trace, "outcome", None)
    capture_duration = getattr(trace, "capture_duration", None)
    from .._capture_honesty import capture_advisories, poison_facts
    from ..report._factcore import _memory

    memory_facts = _memory(trace)

    logged_values = (getattr(trace, "annotations", {}) or {}).get("logged_values", {})
    return {
        "capture": {
            "backend": getattr(trace, "backend", None),
            "model_class_name": getattr(trace, "model_class_name", None),
            "outcome_status": getattr(getattr(outcome, "status", None), "value", None),
            "capture_verified": getattr(trace, "capture_verified", None),
            "structure_only": getattr(trace, "structure_only", False),
            "grouping": getattr(trace, "grouping", None),
            "save_mode": getattr(trace, "save_mode", None),
            "capture_duration": capture_duration,
            **poison_facts(trace),
            "advisories": capture_advisories(trace),
        },
        "graph": {
            "num_ops": len(ops),
            "num_layers": getattr(trace, "num_layers", None),
            "num_modules": getattr(trace, "num_modules", None),
            "num_module_calls": getattr(trace, "num_module_calls", None),
            "num_inputs": len(getattr(trace, "input_ops", ()) or ()),
            "num_outputs": len(getattr(trace, "output_ops", ()) or ()),
            "is_recurrent": getattr(trace, "is_recurrent", False),
            "has_conditionals": getattr(trace, "has_conditionals", False),
            "num_conditionals": getattr(trace, "num_conditionals", 0),
        },
        "parameters": {
            "num_params": getattr(trace, "num_params", None),
            "num_params_trainable": getattr(trace, "num_params_trainable", None),
            "num_params_frozen": getattr(trace, "num_params_frozen", None),
            "num_param_tensors": len(getattr(trace, "params", {}) or {}),
            "param_memory": getattr(trace, "total_param_memory", None),
        },
        "buffers": {
            "num_buffer_tensors": len(getattr(trace, "buffers", {}) or {}),
        },
        # Payload-scope law (C02; sumfam D8): BOM answers "what does THIS
        # object retain?", so its byte figures are retained_now -- the
        # at_capture facts print beside them, never under the same name.
        # (Before-pin: three surfaces claimed 64 MB of data an artifact did
        # not contain.)
        "activations": {
            "payload_scope": "retained_now",
            "num_saved_at_capture": len(saved_ops),
            "at_capture_memory": saved_bytes,
            "retained_now_memory": Bytes(memory_facts.retained_now_bytes),
            "num_present_now": memory_facts.retained_now_present_ops,
            "num_lazy_now": memory_facts.retained_now_lazy_ops,
            "total_activation_memory": getattr(trace, "total_activation_memory", None),
        },
        "backward": {
            "num_grad_fn_records": len(getattr(trace, "grad_fn_logs", {}) or {}),
            "num_backward_passes": len(getattr(trace, "backward_pass_logs", {}) or {}),
        },
        "annotations": tuple(sorted(getattr(trace, "annotations", {}) or {})),
        # log_value read-back (sumfam D20): logged values are inventory, not
        # just an opaque annotations key name.
        "logged_values": dict(logged_values) if isinstance(logged_values, dict) else {},
    }

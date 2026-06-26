"""Trace-summary exporter for menagerie retrace-required metadata."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

import torchlens as tl

from menagerie.catalog import CATALOG_DB, CatalogRow, ensure_catalog, load_rows
from menagerie.identity import recipe_revision_sha256
from menagerie.op_taxonomy import (
    OP_CATEGORIES,
    OP_TAXONOMY_VERSION,
    classify_activation_type,
    classify_norm_type,
    classify_op,
)
from menagerie.recipe import build_input_for_row, instantiate_model
from menagerie.structural_digest import _trace_for_digest

TRACE_SUMMARY_VERSION = "1.0.0"
TRACE_SUMMARY_DB = Path(__file__).resolve().parent / "data" / "trace_summary.db"
BYTES_PER_MB = 1_048_576

TRACE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "trace_summary_version",
    "op_taxonomy_version",
    "recipe_revision_sha256",
    "torchlens_version",
    "n_params_source",
    "n_compute_ops",
    "n_unique_op_types",
    "n_inplace_ops",
    "has_custom_op",
    "op_type_histogram",
    "dominant_op_type",
    "top_op_types_json",
    "pct_conv",
    "pct_linear",
    "pct_attention",
    "pct_norm",
    "pct_elementwise",
    "pct_reduction",
    "pct_reshape",
    "pct_embedding",
    "pct_pooling",
    "graph_depth",
    "graph_max_width",
    "branching_factor",
    "max_fan_out",
    "max_fan_in",
    "is_branching",
    "is_recurrent",
    "max_recurrence_iters",
    "n_recurrent_layers",
    "has_conditional_branching",
    "n_conditionals",
    "is_dynamic_graph",
    "n_modules",
    "n_module_calls",
    "module_max_depth",
    "n_top_level_modules",
    "n_unique_module_types",
    "module_type_histogram",
    "top_module_types_json",
    "top_level_block_sequence_json",
    "model_class_name",
    "model_class_qualname",
    "n_params",
    "n_params_trainable",
    "n_params_frozen",
    "n_param_tensors",
    "param_memory_bytes",
    "primary_param_dtype",
    "param_dtype_set_json",
    "quantized_param_tensor_count",
    "has_frozen_params",
    "n_buffers",
    "buffer_memory_bytes",
    "buffer_overwrite_count",
    "total_flops_forward",
    "total_macs_forward",
    "total_flops_backward",
    "total_macs_backward",
    "flops_coverage_pct",
    "n_unknown_flops_ops",
    "flops_by_op_type",
    "macs_by_op_type",
    "activation_memory_bytes",
    "forward_peak_memory_bytes",
    "largest_activation_bytes",
    "param_memory_mb",
    "activation_memory_mb",
    "forward_peak_memory_mb",
    "output_shape",
    "n_output_tensors",
    "output_container_kind",
    "has_attention",
    "has_conv",
    "has_embedding",
    "has_residual",
    "has_self_attention",
    "has_cross_attention",
    "norm_type",
    "activation_fn_type",
    "pooling_type",
    "structural_barcode_json",
    "graph_shape_hash",
    "structural_fingerprint_hash",
)

_SQL_TYPES: dict[str, str] = {
    "stable_id": "TEXT PRIMARY KEY",
    "trace_summary_version": "TEXT NOT NULL",
    "op_taxonomy_version": "TEXT NOT NULL",
    "recipe_revision_sha256": "TEXT",
    "torchlens_version": "TEXT",
    "n_params_source": "TEXT",
    "branching_factor": "REAL",
    "flops_coverage_pct": "REAL",
    "pct_conv": "REAL",
    "pct_linear": "REAL",
    "pct_attention": "REAL",
    "pct_norm": "REAL",
    "pct_elementwise": "REAL",
    "pct_reduction": "REAL",
    "pct_reshape": "REAL",
    "pct_embedding": "REAL",
    "pct_pooling": "REAL",
    "param_memory_mb": "REAL",
    "activation_memory_mb": "REAL",
    "forward_peak_memory_mb": "REAL",
    "op_type_histogram": "TEXT",
    "top_op_types_json": "TEXT",
    "module_type_histogram": "TEXT",
    "top_module_types_json": "TEXT",
    "top_level_block_sequence_json": "TEXT",
    "param_dtype_set_json": "TEXT",
    "flops_by_op_type": "TEXT",
    "macs_by_op_type": "TEXT",
    "structural_barcode_json": "TEXT",
    "dominant_op_type": "TEXT",
    "model_class_name": "TEXT",
    "model_class_qualname": "TEXT",
    "primary_param_dtype": "TEXT",
    "output_shape": "TEXT",
    "output_container_kind": "TEXT",
    "norm_type": "TEXT",
    "activation_fn_type": "TEXT",
    "pooling_type": "TEXT",
    "graph_shape_hash": "TEXT",
    "structural_fingerprint_hash": "TEXT",
}

_BOOL_COLUMNS = {
    "has_custom_op",
    "is_branching",
    "is_recurrent",
    "has_conditional_branching",
    "is_dynamic_graph",
    "has_frozen_params",
    "has_attention",
    "has_conv",
    "has_embedding",
    "has_residual",
    "has_self_attention",
    "has_cross_attention",
}

_JSON_COLUMNS = {
    "op_type_histogram",
    "top_op_types_json",
    "module_type_histogram",
    "top_module_types_json",
    "top_level_block_sequence_json",
    "param_dtype_set_json",
    "flops_by_op_type",
    "macs_by_op_type",
    "structural_barcode_json",
}


def _json_dumps(value: Any) -> str:
    """Serialize JSON values with deterministic key order.

    Parameters
    ----------
    value:
        JSON-serializable value.

    Returns
    -------
    str
        Compact JSON string.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _int_or_none(value: Any) -> int | None:
    """Convert a numeric-like value to int while preserving nulls.

    Parameters
    ----------
    value:
        Numeric value or ``None``.

    Returns
    -------
    int | None
        Integer value, or ``None``.
    """

    if value is None:
        return None
    return int(value)


def _percent(numerator: int, denominator: int) -> float:
    """Return a schema percentage in 0-100 units.

    Parameters
    ----------
    numerator:
        Count in the numerator.
    denominator:
        Count in the denominator.

    Returns
    -------
    float
        Percentage in 0-100 units.
    """

    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def _top_items(counter: Counter[str], limit: int = 5) -> list[str]:
    """Return top counter keys with deterministic tie-breaking.

    Parameters
    ----------
    counter:
        Counter to rank.
    limit:
        Maximum number of keys to return.

    Returns
    -------
    list[str]
        Top keys sorted by descending count, then name.
    """

    return [
        key for key, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _module_class_name(trace: Any, op: Any) -> str | None:
    """Return the best owning module class name for an Op.

    Parameters
    ----------
    trace:
        TorchLens trace.
    op:
        TorchLens Op.

    Returns
    -------
    str | None
        Owning module class name when TorchLens recorded one.
    """

    atomic_module = getattr(op, "atomic_module", None)
    if atomic_module is not None:
        return str(getattr(atomic_module, "class_name", "") or "") or None
    module_address = getattr(op, "module", None)
    if isinstance(module_address, str) and module_address in trace.modules:
        return str(getattr(trace.modules[module_address], "class_name", "") or "") or None
    return None


def _compute_ops(trace: Any) -> list[Any]:
    """Return compute Ops from a TorchLens trace.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    list[Any]
        Non-boundary operation records in trace order.
    """

    return list(trace.compute_ops)


def _op_histograms(trace: Any, compute_ops: Sequence[Any]) -> tuple[Counter[str], Counter[str]]:
    """Build raw op-name and taxonomy-category histograms.

    Parameters
    ----------
    trace:
        TorchLens trace.
    compute_ops:
        Compute Ops from the trace.

    Returns
    -------
    tuple[Counter[str], Counter[str]]
        Raw function-name counts and taxonomy-category counts.
    """

    op_type_histogram: Counter[str] = Counter()
    category_histogram: Counter[str] = Counter()
    for op in compute_ops:
        func_name = str(getattr(op, "func_name", None) or getattr(op, "layer_type", "other"))
        module_class_name = _module_class_name(trace, op)
        op_type_histogram[func_name] += 1
        category_histogram[classify_op(func_name, module_class_name)] += 1
    return op_type_histogram, category_histogram


def _dominant_category(category_histogram: Counter[str]) -> str:
    """Return the most frequent taxonomy category.

    Parameters
    ----------
    category_histogram:
        Category counts.

    Returns
    -------
    str
        Dominant category, or ``"other"`` for an empty trace.
    """

    if not category_histogram:
        return "other"
    return sorted(category_histogram.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _module_histogram(trace: Any) -> Counter[str]:
    """Return a full module-class histogram.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    Counter[str]
        Module class-name counts.
    """

    return Counter(str(getattr(module, "class_name", "") or "unknown") for module in trace.modules)


def _top_level_modules(trace: Any) -> list[str]:
    """Return ordered top-level module class names.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    list[str]
        Class names of modules directly under the root module.
    """

    root = getattr(trace, "root_module", None)
    child_addresses = list(getattr(root, "address_children", []) or [])
    result: list[str] = []
    for address in child_addresses:
        if address in trace.modules:
            result.append(str(getattr(trace.modules[address], "class_name", "") or "unknown"))
    return result


def _graph_depth_and_width(compute_ops: Sequence[Any]) -> tuple[int, int]:
    """Compute graph depth and maximum width from Op input distances.

    Parameters
    ----------
    compute_ops:
        Compute Ops from a TorchLens trace.

    Returns
    -------
    tuple[int, int]
        ``(graph_depth, graph_max_width)``.
    """

    distances = [
        int(distance)
        for op in compute_ops
        if (distance := getattr(op, "max_distance_from_input", None)) is not None
    ]
    if not distances:
        return 0, 0
    return max(distances), max(Counter(distances).values(), default=0)


def _module_max_depth(compute_ops: Sequence[Any]) -> int:
    """Compute maximum module nesting depth from Op module-call depth.

    Parameters
    ----------
    compute_ops:
        Compute Ops from a TorchLens trace.

    Returns
    -------
    int
        Maximum module nesting depth.
    """

    return max((int(getattr(op, "module_call_depth", 0) or 0) for op in compute_ops), default=0)


def _flops_coverage(compute_ops: Sequence[Any]) -> tuple[float, int]:
    """Compute FLOPs coverage over compute Ops.

    Parameters
    ----------
    compute_ops:
        Compute Ops from a TorchLens trace.

    Returns
    -------
    tuple[float, int]
        ``(coverage_pct, unknown_count)``.
    """

    unknown = sum(getattr(op, "flops_forward", None) is None for op in compute_ops)
    return _percent(len(compute_ops) - unknown, len(compute_ops)), unknown


def _by_op_type_forward_totals(callable_dict: Any) -> dict[str, int]:
    """Convert TorchLens FLOPs/MACs-by-type data to schema JSON.

    Parameters
    ----------
    callable_dict:
        Dict-like object returned by a TorchLens by-op-type property.

    Returns
    -------
    dict[str, int]
        Mapping from op type to forward total.
    """

    raw = callable_dict() if callable(callable_dict) else callable_dict
    result: dict[str, int] = {}
    for key, value in dict(raw).items():
        if isinstance(value, Mapping):
            result[str(key)] = int(value.get("forward", 0) or 0)
        else:
            result[str(key)] = int(value or 0)
    return dict(sorted(result.items()))


def _param_dtype_summary(trace: Any) -> tuple[str, list[str]]:
    """Return primary parameter dtype and full dtype set.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    tuple[str, list[str]]
        Most common dtype by scalar count and sorted dtype set.
    """

    dtype_counts: Counter[str] = Counter()
    for param in trace.params:
        dtype_counts[str(getattr(param, "dtype", ""))] += int(getattr(param, "num_params", 0) or 0)
    if not dtype_counts:
        return "", []
    primary = sorted(dtype_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return primary, sorted(dtype_counts)


def _buffer_memory_bytes(trace: Any) -> int:
    """Return final registered-buffer memory in bytes.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    int
        Sum of final buffer activation-memory records.
    """

    return sum(int(getattr(buffer, "activation_memory", 0) or 0) for buffer in trace.buffers)


def _largest_activation_bytes(compute_ops: Sequence[Any]) -> int:
    """Return largest compute-op activation size in bytes.

    Parameters
    ----------
    compute_ops:
        Compute Ops from a TorchLens trace.

    Returns
    -------
    int
        Maximum activation-memory value.
    """

    return max((int(getattr(op, "activation_memory", 0) or 0) for op in compute_ops), default=0)


def _output_summary(trace: Any) -> tuple[str, int, str]:
    """Return output shape, tensor count, and output container kind.

    Parameters
    ----------
    trace:
        TorchLens trace.

    Returns
    -------
    tuple[str, int, str]
        Schema output summary fields.
    """

    output_ops = list(trace.output_ops)
    shapes = [getattr(op, "shape", None) for op in output_ops]
    shape_strings = [str(tuple(shape)) if shape is not None else "" for shape in shapes]
    output_shape = shape_strings[0] if len(shape_strings) == 1 else _json_dumps(shape_strings)
    specs = [getattr(op, "container_spec", None) for op in output_ops]
    first_spec = next((spec for spec in specs if spec is not None), None)
    if first_spec is None:
        output_container_kind = "tensor" if len(output_ops) <= 1 else "multi_tensor"
    else:
        output_container_kind = str(
            getattr(first_spec, "type_qualname", None)
            or getattr(first_spec, "type_module", None)
            or type(first_spec).__name__
        )
    return output_shape, len(output_ops), output_container_kind


def _aggregate_subtypes(
    trace: Any,
    compute_ops: Sequence[Any],
    classifier_name: str,
) -> str:
    """Aggregate norm or activation subtypes across compute Ops.

    Parameters
    ----------
    trace:
        TorchLens trace.
    compute_ops:
        Compute Ops from the trace.
    classifier_name:
        ``"norm"`` or ``"activation"``.

    Returns
    -------
    str
        Concrete subtype, ``"mixed"``, or ``"none"``.
    """

    subtypes: set[str] = set()
    for op in compute_ops:
        func_name = str(getattr(op, "func_name", "") or "")
        module_class_name = _module_class_name(trace, op)
        if classifier_name == "norm":
            subtype = classify_norm_type(func_name, module_class_name)
        else:
            subtype = classify_activation_type(func_name, module_class_name)
        if subtype != "none":
            subtypes.add(subtype)
    if not subtypes:
        return "none"
    if len(subtypes) > 1:
        return "mixed"
    return next(iter(subtypes))


def _pooling_type(trace: Any, compute_ops: Sequence[Any]) -> str:
    """Aggregate pooling subtype across compute Ops.

    Parameters
    ----------
    trace:
        TorchLens trace.
    compute_ops:
        Compute Ops from the trace.

    Returns
    -------
    str
        Pooling subtype, ``"mixed"``, or ``"none"``.
    """

    subtypes: set[str] = set()
    for op in compute_ops:
        func_name = str(getattr(op, "func_name", "") or "").lower()
        module_name = str(_module_class_name(trace, op) or "").lower()
        if classify_op(func_name, module_name) != "pooling":
            continue
        combined = f"{func_name} {module_name}"
        if "max" in combined:
            subtypes.add("max_pool")
        elif "avg" in combined or "average" in combined:
            subtypes.add("avg_pool")
        elif "lp" in combined:
            subtypes.add("lp_pool")
        else:
            subtypes.add("pooling")
    if not subtypes:
        return "none"
    if len(subtypes) > 1:
        return "mixed"
    return next(iter(subtypes))


def _has_custom_op(compute_ops: Sequence[Any]) -> bool:
    """Return whether any op appears outside the standard torch namespace.

    Parameters
    ----------
    compute_ops:
        Compute Ops from a TorchLens trace.

    Returns
    -------
    bool
        Whether a custom callable was recorded.
    """

    standard_prefixes = ("torch", "builtins", "operator", "_operator")
    for op in compute_ops:
        func = getattr(op, "func", None)
        module_name = str(getattr(func, "__module__", "") or "")
        qualname = str(getattr(op, "func_qualname", "") or "")
        if module_name and not module_name.startswith(standard_prefixes):
            return True
        if qualname and "." in qualname and not qualname.startswith(standard_prefixes):
            return True
    return False


def _has_residual(trace: Any, compute_ops: Sequence[Any]) -> bool:
    """Detect residual-style add ops from graph topology.

    Parameters
    ----------
    trace:
        TorchLens trace.
    compute_ops:
        Compute Ops from the trace.

    Returns
    -------
    bool
        Whether an add-like op combines skip-connected parents.
    """

    add_names = {"add", "add_", "torch.add", "__add__", "__iadd__"}
    for op in compute_ops:
        func_name = str(getattr(op, "func_name", "") or "").lower()
        if func_name not in add_names or len(getattr(op, "parents", []) or []) < 2:
            continue
        parents = [trace.ops[label] for label in op.parents if label in trace.ops]
        for left_index, left_parent in enumerate(parents):
            left_ancestors = set(getattr(left_parent, "input_ancestors", set()) or set())
            left_distance = getattr(left_parent, "max_distance_from_input", None)
            for right_parent in parents[left_index + 1 :]:
                right_ancestors = set(getattr(right_parent, "input_ancestors", set()) or set())
                if left_parent.label in right_ancestors or right_parent.label in left_ancestors:
                    return True
                right_distance = getattr(right_parent, "max_distance_from_input", None)
                if left_distance is not None and right_distance is not None:
                    if abs(int(left_distance) - int(right_distance)) > 1:
                        return True
    return False


def _attention_motifs(trace: Any, compute_ops: Sequence[Any]) -> tuple[bool, bool]:
    """Return self-attention and cross-attention motif flags.

    Parameters
    ----------
    trace:
        TorchLens trace.
    compute_ops:
        Compute Ops from the trace.

    Returns
    -------
    tuple[bool, bool]
        ``(has_self_attention, has_cross_attention)``.
    """

    has_self_attention = False
    has_cross_attention = False
    for op in compute_ops:
        func_name = str(getattr(op, "func_name", "") or "")
        module_class_name = _module_class_name(trace, op)
        if classify_op(func_name, module_class_name) != "attention":
            continue
        combined = f"{func_name} {module_class_name or ''}".lower()
        if "cross" in combined or "encoder_decoder" in combined:
            has_cross_attention = True
        else:
            has_self_attention = True
    return has_self_attention, has_cross_attention


def _structural_barcode(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact structural barcode blob.

    Parameters
    ----------
    row:
        Partially computed summary row.

    Returns
    -------
    dict[str, Any]
        Structural barcode object.
    """

    return {
        "n_ops": row["n_compute_ops"],
        "graph_depth": row["graph_depth"],
        "max_width": row["graph_max_width"],
        "is_recurrent": row["is_recurrent"],
        "is_branching": row["is_branching"],
        "is_dynamic_graph": row["is_dynamic_graph"],
        "has_attention": row["has_attention"],
        "has_conv": row["has_conv"],
        "has_residual": row["has_residual"],
        "norm_type": row["norm_type"],
        "activation_fn": row["activation_fn_type"],
        "n_params": row["n_params"],
        "n_modules": row["n_modules"],
        "top_op": row["dominant_op_type"],
    }


def static_n_params(model: Any) -> int:
    """Count parameters from ``named_parameters`` without tracing.

    Parameters
    ----------
    model:
        Constructed model.

    Returns
    -------
    int
        Static scalar parameter count.
    """

    if not hasattr(model, "named_parameters"):
        return 0
    return sum(int(parameter.numel()) for _name, parameter in model.named_parameters())


def summarize_model(
    stable_id: str,
    model: Any,
    example_input: Any,
    recipe_sha256: str = "",
    *,
    compute_identity_hashes: bool = True,
) -> dict[str, Any]:
    """Retrace a model and return the complete trace-summary record.

    Parameters
    ----------
    stable_id:
        Durable menagerie model id.
    model:
        Constructed model.
    example_input:
        Example input accepted by the model.
    recipe_sha256:
        Recipe revision hash for provenance.
    compute_identity_hashes:
        Whether to compute both structural-digest hashes through their public
        entry points. Tests may disable this when hash values are not under test.

    Returns
    -------
    dict[str, Any]
        Summary row keyed by ``TRACE_SUMMARY_COLUMNS``.
    """

    trace = _trace_for_digest(model, example_input)
    return summarize_trace(
        stable_id,
        trace,
        recipe_sha256,
        compute_identity_hashes=compute_identity_hashes,
    )


def summarize_trace(
    stable_id: str,
    trace: Any,
    recipe_sha256: str = "",
    *,
    compute_identity_hashes: bool = False,
) -> dict[str, Any]:
    """Build the complete trace-summary record from an ALREADY-CAPTURED trace.

    This is the single-trace sibling of :func:`summarize_model`. It derives every
    structural metadata field -- including both identity hashes -- from one trace
    object, so the menagerie validation observer can produce the metadata row off
    the trace it already built for forward-replay validation, instead of
    re-instantiating the model and re-tracing it two-to-three more times.

    Both identity hashes are deterministic functions of the trace
    (``Trace.graph_shape_hash`` and ``Trace._raw_event_shape_hash``); reading them
    here is byte-identical to the public ``architecture_distinctness_hash`` /
    ``structural_fingerprint`` entry points WHENEVER the supplied trace was built
    with the same capture configuration those entry points use (CPU,
    ``inference_only=True``). Bucket A's field-equivalence audit
    (``menagerie.trace_equivalence_audit``) is the gate that proves this holds for
    the validation-config trace before this path is wired into the observer.

    Parameters
    ----------
    stable_id:
        Durable menagerie model id.
    trace:
        Completed TorchLens trace.
    recipe_sha256:
        Recipe revision hash for provenance.
    compute_identity_hashes:
        Retained for signature parity with :func:`summarize_model`. The two
        identity hashes are always read off the supplied trace; this flag only
        controls whether a missing hash raises (parity with the public hash entry
        points) instead of recording an empty string. Defaults to ``False`` so the
        observer path is non-raising.

    Returns
    -------
    dict[str, Any]
        Summary row keyed by ``TRACE_SUMMARY_COLUMNS``.
    """

    compute_ops = _compute_ops(trace)
    op_type_histogram, category_histogram = _op_histograms(trace, compute_ops)
    module_type_histogram = _module_histogram(trace)
    top_level_blocks = _top_level_modules(trace)
    graph_depth, graph_max_width = _graph_depth_and_width(compute_ops)
    flops_coverage_pct, n_unknown_flops_ops = _flops_coverage(compute_ops)
    primary_param_dtype, param_dtype_set = _param_dtype_summary(trace)
    output_shape, n_output_tensors, output_container_kind = _output_summary(trace)
    has_self_attention, has_cross_attention = _attention_motifs(trace, compute_ops)
    param_memory_bytes = int(getattr(trace, "total_param_memory", 0) or 0)
    activation_memory_bytes = int(getattr(trace, "total_activation_memory", 0) or 0)
    forward_peak_memory_bytes = int(getattr(trace, "forward_peak_memory", 0) or 0)

    row: dict[str, Any] = {
        "stable_id": stable_id,
        "trace_summary_version": TRACE_SUMMARY_VERSION,
        "op_taxonomy_version": OP_TAXONOMY_VERSION,
        "recipe_revision_sha256": recipe_sha256,
        "torchlens_version": str(getattr(tl, "__version__", "")),
        "n_params_source": "traced",
        "n_compute_ops": int(getattr(trace, "num_compute_ops", 0) or 0),
        "n_unique_op_types": len(op_type_histogram),
        "n_inplace_ops": sum(bool(getattr(op, "is_inplace", False)) for op in compute_ops),
        "has_custom_op": _has_custom_op(compute_ops),
        "op_type_histogram": dict(sorted(op_type_histogram.items())),
        "dominant_op_type": _dominant_category(category_histogram),
        "top_op_types_json": _top_items(op_type_histogram),
        **{
            f"pct_{category}": _percent(category_histogram.get(category, 0), len(compute_ops))
            for category in (
                "conv",
                "linear",
                "attention",
                "norm",
                "elementwise",
                "reduction",
                "reshape",
                "embedding",
                "pooling",
            )
        },
        "graph_depth": graph_depth,
        "graph_max_width": graph_max_width,
        "branching_factor": float(getattr(trace, "branching_factor", 0.0) or 0.0),
        "max_fan_out": int(getattr(trace, "max_out_degree", 0) or 0),
        "max_fan_in": int(getattr(trace, "max_in_degree", 0) or 0),
        "is_branching": bool(getattr(trace, "is_branching", False)),
        "is_recurrent": bool(getattr(trace, "is_recurrent", False)),
        "max_recurrence_iters": int(getattr(trace, "max_layer_op_count", 0) or 0),
        "n_recurrent_layers": len(getattr(trace, "recurrent_layers", []) or []),
        "has_conditional_branching": bool(getattr(trace, "has_conditional_branching", False)),
        "n_conditionals": int(getattr(trace, "num_conditionals", 0) or 0),
        "is_dynamic_graph": bool(getattr(trace, "is_dynamic_graph", False)),
        "n_modules": int(getattr(trace, "num_modules", 0) or 0),
        "n_module_calls": int(getattr(trace, "num_module_calls", 0) or 0),
        "module_max_depth": _module_max_depth(compute_ops),
        "n_top_level_modules": len(top_level_blocks),
        "n_unique_module_types": len(module_type_histogram),
        "module_type_histogram": dict(sorted(module_type_histogram.items())),
        "top_module_types_json": _top_items(module_type_histogram),
        "top_level_block_sequence_json": top_level_blocks,
        "model_class_name": str(getattr(trace, "model_class_name", "") or ""),
        "model_class_qualname": str(getattr(trace, "model_class_qualname", "") or ""),
        "n_params": int(getattr(trace, "num_params", 0) or 0),
        "n_params_trainable": int(getattr(trace, "num_params_trainable", 0) or 0),
        "n_params_frozen": int(getattr(trace, "num_params_frozen", 0) or 0),
        "n_param_tensors": int(getattr(trace, "num_param_tensors", 0) or 0),
        "param_memory_bytes": param_memory_bytes,
        "primary_param_dtype": primary_param_dtype,
        "param_dtype_set_json": param_dtype_set,
        "quantized_param_tensor_count": sum(
            bool(getattr(param, "is_quantized", False)) for param in trace.params
        ),
        "has_frozen_params": bool(getattr(trace, "has_frozen_params", False)),
        "n_buffers": len(getattr(trace, "buffers", []) or []),
        "buffer_memory_bytes": _buffer_memory_bytes(trace),
        "buffer_overwrite_count": int(getattr(trace, "num_buffer_write_ops", 0) or 0),
        "total_flops_forward": int(getattr(trace, "total_flops_forward", 0) or 0),
        "total_macs_forward": int(getattr(trace, "total_macs_forward", 0) or 0),
        "total_flops_backward": _int_or_none(getattr(trace, "total_flops_backward", None)),
        "total_macs_backward": _int_or_none(getattr(trace, "total_macs_backward", None)),
        "flops_coverage_pct": flops_coverage_pct,
        "n_unknown_flops_ops": n_unknown_flops_ops,
        "flops_by_op_type": _by_op_type_forward_totals(getattr(trace, "flops_by_op_type", {})),
        "macs_by_op_type": _by_op_type_forward_totals(getattr(trace, "macs_by_op_type", {})),
        "activation_memory_bytes": activation_memory_bytes,
        "forward_peak_memory_bytes": forward_peak_memory_bytes,
        "largest_activation_bytes": _largest_activation_bytes(compute_ops),
        "param_memory_mb": param_memory_bytes / BYTES_PER_MB,
        "activation_memory_mb": activation_memory_bytes / BYTES_PER_MB,
        "forward_peak_memory_mb": forward_peak_memory_bytes / BYTES_PER_MB,
        "output_shape": output_shape,
        "n_output_tensors": n_output_tensors,
        "output_container_kind": output_container_kind,
        "has_attention": category_histogram.get("attention", 0) > 0,
        "has_conv": category_histogram.get("conv", 0) > 0,
        "has_embedding": category_histogram.get("embedding", 0) > 0,
        "has_residual": _has_residual(trace, compute_ops),
        "has_self_attention": has_self_attention,
        "has_cross_attention": has_cross_attention,
        "norm_type": _aggregate_subtypes(trace, compute_ops, "norm"),
        "activation_fn_type": _aggregate_subtypes(trace, compute_ops, "activation"),
        "pooling_type": _pooling_type(trace, compute_ops),
    }
    row["structural_barcode_json"] = _structural_barcode(row)
    # Both hashes are deterministic functions of THIS trace -- read them off the
    # single captured trace rather than re-tracing (kills the two extra metadata
    # re-traces). ``compute_identity_hashes`` only controls whether a genuinely
    # missing hash raises (public-entry-point parity) or records an empty string.
    graph_shape_hash = getattr(trace, "graph_shape_hash", None)
    raw_event_hash = getattr(trace, "_raw_event_shape_hash", None)
    if compute_identity_hashes:
        if not graph_shape_hash:
            raise RuntimeError("TorchLens trace did not expose graph_shape_hash")
        if not raw_event_hash:
            raise RuntimeError("TorchLens trace did not expose _raw_event_shape_hash")
    row["graph_shape_hash"] = str(graph_shape_hash or "")
    row["structural_fingerprint_hash"] = str(raw_event_hash or "")
    return {column: row.get(column) for column in TRACE_SUMMARY_COLUMNS}


def static_summary_row(stable_id: str, model: Any, recipe_sha256: str = "") -> dict[str, Any]:
    """Build a static-count fallback row without running a forward pass.

    Parameters
    ----------
    stable_id:
        Durable menagerie model id.
    model:
        Constructed model.
    recipe_sha256:
        Recipe revision hash for provenance.

    Returns
    -------
    dict[str, Any]
        Sparse summary row with ``n_params_source == "static_count"``.
    """

    row: dict[str, Any] = {column: None for column in TRACE_SUMMARY_COLUMNS}
    row.update(
        {
            "stable_id": stable_id,
            "trace_summary_version": TRACE_SUMMARY_VERSION,
            "op_taxonomy_version": OP_TAXONOMY_VERSION,
            "recipe_revision_sha256": recipe_sha256,
            "torchlens_version": str(getattr(tl, "__version__", "")),
            "n_params": static_n_params(model),
            "n_params_source": "static_count",
        }
    )
    return row


def _sql_type(column: str) -> str:
    """Return the SQLite type declaration for a summary column.

    Parameters
    ----------
    column:
        Summary column name.

    Returns
    -------
    str
        SQLite column type declaration.
    """

    if column in _SQL_TYPES:
        return _SQL_TYPES[column]
    if column in _BOOL_COLUMNS:
        return "INTEGER"
    return "INTEGER"


def ensure_store(db_path: Path = TRACE_SUMMARY_DB) -> Path:
    """Ensure the trace-summary SQLite table exists.

    Parameters
    ----------
    db_path:
        SQLite store path.

    Returns
    -------
    Path
        Store path.
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)
    column_defs = ",\n                ".join(
        f"{column} {_sql_type(column)}" for column in TRACE_SUMMARY_COLUMNS
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS trace_summaries (
                {column_defs}
            )
            """
        )
    return db_path


def _sqlite_value(column: str, value: Any) -> Any:
    """Convert a Python summary value for SQLite persistence.

    Parameters
    ----------
    column:
        Summary column name.
    value:
        Python value.

    Returns
    -------
    Any
        SQLite-compatible value.
    """

    if value is None:
        return None
    if column in _BOOL_COLUMNS:
        return int(bool(value))
    if column in _JSON_COLUMNS:
        return _json_dumps(value)
    return value


def persist_summary(row: Mapping[str, Any], db_path: Path = TRACE_SUMMARY_DB) -> None:
    """Persist one summary row to SQLite.

    Parameters
    ----------
    row:
        Summary row.
    db_path:
        SQLite store path.
    """

    ensure_store(db_path)
    placeholders = ", ".join("?" for _column in TRACE_SUMMARY_COLUMNS)
    columns_sql = ", ".join(TRACE_SUMMARY_COLUMNS)
    values = [_sqlite_value(column, row.get(column)) for column in TRACE_SUMMARY_COLUMNS]
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            f"""
            INSERT OR REPLACE INTO trace_summaries ({columns_sql})
            VALUES ({placeholders})
            """,
            values,
        )


def load_summary(stable_id: str, db_path: Path = TRACE_SUMMARY_DB) -> dict[str, Any] | None:
    """Load one persisted summary row.

    Parameters
    ----------
    stable_id:
        Durable menagerie model id.
    db_path:
        SQLite store path.

    Returns
    -------
    dict[str, Any] | None
        Persisted row, or ``None`` when absent.
    """

    ensure_store(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM trace_summaries WHERE stable_id = ?",
            (stable_id,),
        ).fetchone()
    return None if row is None else dict(row)


def export_catalog_row(row: CatalogRow, db_path: Path = TRACE_SUMMARY_DB) -> dict[str, Any]:
    """Build and persist a summary for one catalog row.

    Parameters
    ----------
    row:
        Menagerie catalog row.
    db_path:
        SQLite store path.

    Returns
    -------
    dict[str, Any]
        Persisted summary row.
    """

    recipe_sha256 = row.recipe_revision_sha256 or recipe_revision_sha256(row)
    model = instantiate_model(row)
    try:
        example_input = build_input_for_row(row)
        summary = summarize_model(row.stable_id, model, example_input, recipe_sha256)
    except Exception:
        summary = static_summary_row(row.stable_id, model, recipe_sha256)
    persist_summary(summary, db_path)
    return summary


def _rows_for_stable_ids(stable_ids: Sequence[str], catalog_db: Path) -> list[CatalogRow]:
    """Load catalog rows for stable ids.

    Parameters
    ----------
    stable_ids:
        Stable ids to export.
    catalog_db:
        Catalog SQLite path.

    Returns
    -------
    list[CatalogRow]
        Matching rows in requested stable-id order.
    """

    ensure_catalog(catalog_db)
    rows_by_id = {row.stable_id: row for row in load_rows(db_path=catalog_db)}
    missing = [stable_id for stable_id in stable_ids if stable_id not in rows_by_id]
    if missing:
        raise LookupError(f"stable_id values not found in catalog: {', '.join(missing)}")
    return [rows_by_id[stable_id] for stable_id in stable_ids]


def export_catalog(
    *,
    stable_ids: Sequence[str] | None = None,
    all_catalog: bool = False,
    limit: int | None = None,
    catalog_db: Path = CATALOG_DB,
    db_path: Path = TRACE_SUMMARY_DB,
) -> list[dict[str, Any]]:
    """Export summaries for selected catalog rows.

    Parameters
    ----------
    stable_ids:
        Explicit stable ids to export.
    all_catalog:
        Whether to run over the full catalog.
    limit:
        Optional row limit for catalog mode.
    catalog_db:
        Catalog SQLite path.
    db_path:
        Trace-summary SQLite path.

    Returns
    -------
    list[dict[str, Any]]
        Exported rows.
    """

    if stable_ids:
        rows = _rows_for_stable_ids(stable_ids, catalog_db)
    elif all_catalog:
        rows = load_rows(limit=limit, db_path=catalog_db)
    else:
        raise ValueError("pass --stable-ids or --all-catalog")
    if limit is not None and stable_ids:
        rows = rows[:limit]
    return [export_catalog_row(row, db_path) for row in rows]


def build_parser() -> argparse.ArgumentParser:
    """Build the trace-summary CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        CLI parser.
    """

    parser = argparse.ArgumentParser(description="export menagerie trace summaries")
    parser.add_argument("--stable-ids", nargs="*", help="stable ids to retrace and export")
    parser.add_argument("--all-catalog", action="store_true", help="export every catalog row")
    parser.add_argument("--limit", type=int, help="limit rows exported")
    parser.add_argument("--catalog-db", type=Path, default=CATALOG_DB)
    parser.add_argument("--db", type=Path, default=TRACE_SUMMARY_DB)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the trace-summary CLI.

    Parameters
    ----------
    argv:
        Optional CLI argument list.

    Returns
    -------
    int
        Process exit status.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    stable_ids = args.stable_ids if args.stable_ids else None
    try:
        rows = export_catalog(
            stable_ids=stable_ids,
            all_catalog=bool(args.all_catalog),
            limit=args.limit,
            catalog_db=args.catalog_db,
            db_path=args.db,
        )
    except Exception as error:
        print(f"trace_summary export failed: {error}", file=sys.stderr)
        return 1
    for row in rows:
        source = row.get("n_params_source") or ""
        n_params = row.get("n_params")
        print(f"{row['stable_id']}\tn_params={n_params}\tn_params_source={source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Node labels, kinds, and curated attrs (memo D3, D5, D6).

Labels are short op types so Model Explorer's identical-group ("twin")
detection works -- TorchLens' globally unique labels killed it on every v2
artifact (0 twin classes measured). The unique spelling stays searchable as
the ``torchlens_label`` attr. Attrs are the curated 15-key ordered set,
short, omit-when-absent -- attr keys and values ARE Model Explorer's
on-edge label vocabulary rendered at ~7.5px, and the 550-field firehose
stays in csv/parquet where it lives today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...utils.display import format_flops, human_readable_size

__tl_layer__ = "L8"


@dataclass(frozen=True)
class AttrContext:
    """Trace-level facts consulted while building per-node attrs.

    ``nonfinite_labels`` holds the trace's pass-qualified nonfinite op
    labels; ``nonfinite_basis`` the coverage basis (an unexamined op gets NO
    nonfinite row -- an attr must never assert an all-clear it cannot
    prove). ``public`` applies the D14 public privacy profile BY
    CONSTRUCTION: source paths, code lines, and value-derived attrs drop
    regardless of what the capture retained.
    """

    nonfinite_labels: frozenset[str] = frozenset()
    nonfinite_basis: str | None = None
    public: bool = False
    include_source: bool = False


def node_kind(entry: Any) -> str:
    """Return the semantic kind attr (the v2 six-bucket type, memo D1)."""

    if getattr(entry, "is_input", False):
        return "input"
    if getattr(entry, "is_output", False):
        return "output"
    if getattr(entry, "is_buffer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool", False):
        return "bool"
    if int(getattr(entry, "num_params", 0) or 0) > 0:
        return "parameterized"
    return "operation"


def node_label(entry: Any) -> str:
    """Return the short display label: func_name, layer_type, boundary kind.

    Boundary and buffer records carry the literal placeholder ``"none"`` as
    their func_name; it is treated as absent, never rendered.
    """

    func_name = _real_func_name(entry)
    if func_name:
        return func_name
    layer_type = str(getattr(entry, "layer_type", "") or "")
    if layer_type and layer_type.lower() != "none":
        return layer_type
    return node_kind(entry)


def _real_func_name(entry: Any) -> str | None:
    """Return the recorded func name, mapping the ``"none"`` placeholder out."""

    func_name = str(getattr(entry, "func_name", "") or "")
    if not func_name or func_name.lower() == "none":
        return None
    return func_name


def curated_attrs(entry: Any, context: AttrContext) -> list[dict[str, str]]:
    """Build the ordered curated attr rows for one unrolled node (D6)."""

    rows: list[tuple[str, str | None]] = [
        ("kind", node_kind(entry)),
        ("op", _real_func_name(entry)),
        ("shape", shape_string(getattr(entry, "shape", None))),
        ("dtype", dtype_string(entry)),
        ("device", _device_string(entry)),
        ("module", _module_call_string(entry)),
        ("pass", _pass_string(entry)),
        ("act_bytes", _bytes_string(getattr(entry, "activation_memory", None))),
        ("params", _params_string(entry)),
        ("flops", _flops_string(getattr(entry, "flops_forward", None))),
        ("time", _duration_string(getattr(entry, "func_duration", None))),
        ("saved", _saved_string(entry)),
        ("nonfinite", _nonfinite_string(entry, context)),
        ("torchlens_label", str(getattr(entry, "label", "") or "") or None),
        ("site_key", str(getattr(entry, "site_key", "") or "") or None),
    ]
    if context.include_source and not context.public:
        rows.append(("source", _source_string(entry)))
    return [{"key": key, "value": value} for key, value in rows if value is not None]


def curated_rolled_attrs(layer: Any, ops: list[Any], context: AttrContext) -> list[dict[str, str]]:
    """Build attr rows for one rolled multi-pass node, honestly aggregated.

    Values that vary across passes are omitted rather than averaged; exact
    cross-pass totals carry a ``total_`` key so the aggregation is disclosed
    in the key itself (the encoding-channel honesty rule, applied to attrs).
    """

    # Essential complexity (CC>10 named): the rolled attr table IS the
    # honesty policy -- each row's varying/aggregate/single-valued decision
    # belongs beside the others, not scattered across helpers.
    num_passes = len(ops) or int(getattr(layer, "num_passes", 1) or 1)
    rows: list[tuple[str, str | None]] = [
        ("kind", node_kind(layer)),
        ("op", _real_func_name(layer)),
        ("passes", str(num_passes) if num_passes > 1 else None),
        ("shape", _rolled_shape_string(layer, ops)),
        ("dtype", _single_valued(dtype_string(op) for op in ops) if ops else None),
        ("params", _params_string(layer)),
        ("total_flops", _flops_string(_sum_quantity(ops, "flops_forward"))),
        ("total_time", _duration_string(_sum_quantity(ops, "func_duration"))),
        ("torchlens_label", str(getattr(layer, "layer_label", "") or "") or None),
    ]
    site_keys = {str(getattr(op, "site_key", "") or "") for op in ops}
    if len(site_keys) == 1 and next(iter(site_keys)):
        rows.append(("site_key", next(iter(site_keys))))
    return [{"key": key, "value": value} for key, value in rows if value is not None]


def output_metadata(entry: Any) -> list[dict[str, Any]]:
    """Build ``outputsMetadata`` slot ``"0"`` with keys ``shape``/``dtype``.

    The vendor's shipped bundle resolves its "Tensor shape" edge-label mode
    by reading metadata key ``shape`` -- NOT ``tensor_shape`` (memo D5,
    pinned by the contract harness).
    """

    attrs = []
    shape_value = shape_string(getattr(entry, "shape", None))
    if shape_value is not None:
        attrs.append({"key": "shape", "value": shape_value})
    dtype_value = dtype_string(entry)
    if dtype_value is not None:
        attrs.append({"key": "dtype", "value": dtype_value})
    if not attrs:
        return []
    return [{"id": "0", "attrs": attrs}]


def shape_string(shape: Any) -> str | None:
    """Format a captured shape tuple compactly (``2x4``; ``scalar`` for 0-d)."""

    if shape is None:
        return None
    dims = list(shape)
    if not dims:
        return "scalar"
    return "x".join(str(dim) for dim in dims)


def dtype_string(entry: Any) -> str | None:
    """Return the short dtype spelling (``float32``), if recorded."""

    dtype_ref = getattr(entry, "dtype_ref", None)
    name = getattr(dtype_ref, "name", None)
    if not name:
        dtype = getattr(entry, "dtype", None)
        name = str(dtype) if dtype is not None else None
    if not name:
        return None
    return str(name).removeprefix("torch.")


def _device_string(entry: Any) -> str | None:
    """Return the output device spelling, if recorded."""

    device = getattr(entry, "output_device", None)
    return str(device) if device else None


def _module_call_string(entry: Any) -> str | None:
    """Return the canonical (innermost) module call, if inside a module."""

    stack = getattr(entry, "module_call_stack", ()) or ()
    return str(stack[-1]) if stack else None


def _pass_string(entry: Any) -> str | None:
    """Return ``n/N`` for multi-pass ops; omitted for single-pass ops."""

    num_passes = int(getattr(entry, "num_passes", 1) or 1)
    if num_passes <= 1:
        return None
    return f"{int(getattr(entry, 'pass_index', 1) or 1)}/{num_passes}"


def _bytes_string(value: Any) -> str | None:
    """Format a byte quantity, omitting unknowns."""

    if value is None:
        return None
    return human_readable_size(int(value))


def _params_string(entry: Any) -> str | None:
    """Format the combined param count / param memory row."""

    num_params = int(getattr(entry, "num_params", 0) or 0)
    if num_params <= 0:
        return None
    param_memory = getattr(entry, "param_memory", None)
    if param_memory is None:
        return str(num_params)
    return f"{num_params} ({human_readable_size(int(param_memory))})"


def _flops_string(value: Any) -> str | None:
    """Format a forward-FLOPs quantity, omitting unknowns and zeros."""

    if value is None or int(value) <= 0:
        return None
    return format_flops(int(value))


def _duration_string(value: Any) -> str | None:
    """Format a duration in milliseconds, omitting unknowns and zeros."""

    if value is None or float(value) <= 0.0:
        return None
    return f"{float(value) * 1000.0:.3g} ms"


def _saved_string(entry: Any) -> str | None:
    """Return the saved-activation status row."""

    saved = getattr(entry, "has_saved_activation", None)
    if saved is None:
        return None
    return "yes" if saved else "no"


def _nonfinite_string(entry: Any, context: AttrContext) -> str | None:
    """Return the nonfinite status, only where the record actually checked.

    ``yes`` when the op is in the trace's nonfinite record; ``no`` only when
    the coverage basis provably examined this op; otherwise omitted -- an
    unexamined op must never carry a false all-clear. Public profiles drop
    the row entirely (value-derived, memo D14).
    """

    if context.public:
        return None
    label = str(getattr(entry, "label", "") or "")
    if label and label in context.nonfinite_labels:
        return "yes"
    if context.nonfinite_basis == "capture_time":
        return "no"
    if context.nonfinite_basis == "saved_payloads" and getattr(
        entry, "has_saved_activation", False
    ):
        return "no"
    return None


def _source_string(entry: Any) -> str | None:
    """Return ``basename:line`` of the captured call site (opt-in row)."""

    context_rows = getattr(entry, "code_context", None) or ()
    for location in context_rows:
        file_name = getattr(location, "file", None)
        line = getattr(location, "line_number", None)
        if line is None:
            line = getattr(location, "call_line", None)
        if file_name and line is not None:
            return f"{str(file_name).rsplit('/', 1)[-1]}:{line}"
    return None


def _rolled_shape_string(layer: Any, ops: list[Any]) -> str | None:
    """Return the derived across-pass shape summary for a rolled node."""

    summary = getattr(layer, "shape_summary", None)
    if summary:
        return str(summary)
    return _single_valued(shape_string(getattr(op, "shape", None)) for op in ops)


def _single_valued(values: Any) -> str | None:
    """Return the single distinct value, or ``None`` when values vary."""

    seen = {value for value in values if value is not None}
    if len(seen) == 1:
        return next(iter(seen))
    return None


def _sum_quantity(ops: list[Any], field_name: str) -> float | None:
    """Sum a numeric per-pass quantity; ``None`` when nothing is recorded."""

    total = 0.0
    any_recorded = False
    for op in ops:
        value = getattr(op, field_name, None)
        if value is not None:
            total += float(value)
            any_recorded = True
    return total if any_recorded else None

"""Formatting helpers for TorchLens visualization node labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from torchlens._data_classes.layer import Layer
from torchlens._data_classes.op import Op

from .._capture.flops import append_attention_note
from .._errors import InvalidArgumentError
from .._quantities import Duration
from .._record_type import record_type
from .._utils._multipass_access import get_multipass_attr, is_multipass_layer
from .._utils._record_group_table import record_member
from ._pass_marks import pass_rows

PARAM_SEPARATOR: Final[str] = " · "

_KWARG_ORDER: Final[dict[str, tuple[str, ...]]] = {
    "linear": ("in_features", "out_features", "bias"),
    "conv1d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ),
    "conv2d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ),
    "conv3d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ),
    "convolution": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "padding_mode",
    ),
    "convtranspose1d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "output_padding",
        "groups",
        "bias",
        "dilation",
        "padding_mode",
    ),
    "convtranspose2d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "output_padding",
        "groups",
        "bias",
        "dilation",
        "padding_mode",
    ),
    "convtranspose3d": (
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "output_padding",
        "groups",
        "bias",
        "dilation",
        "padding_mode",
    ),
    "layernorm": ("normalized_shape", "eps", "elementwise_affine", "bias"),
    "batchnorm": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "batchnorm1d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "batchnorm2d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "batchnorm3d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "instancenorm": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "instancenorm1d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "instancenorm2d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "instancenorm3d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "groupnorm": ("num_groups", "num_channels", "eps", "affine"),
    "embedding": (
        "num_embeddings",
        "embedding_dim",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    ),
    "maxpool1d": ("kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"),
    "maxpool2d": ("kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"),
    "maxpool3d": ("kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"),
    "avgpool1d": ("kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"),
    "avgpool2d": ("kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"),
    "avgpool3d": ("kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"),
    "adaptiveavgpool1d": ("output_size",),
    "adaptiveavgpool2d": ("output_size",),
    "adaptiveavgpool3d": ("output_size",),
    "adaptivemaxpool1d": ("output_size", "return_indices"),
    "adaptivemaxpool2d": ("output_size", "return_indices"),
    "adaptivemaxpool3d": ("output_size", "return_indices"),
    "multiheadattention": (
        "embed_dim",
        "num_heads",
        "dropout",
        "bias",
        "add_bias_kv",
        "add_zero_attn",
        "kdim",
        "vdim",
        "batch_first",
    ),
    "scaleddotproductattention": ("num_heads", "embed_dim", "dropout_p", "is_causal"),
    "dropout": ("p", "inplace"),
    "dropout1d": ("p", "inplace"),
    "dropout2d": ("p", "inplace"),
    "dropout3d": ("p", "inplace"),
}


def format_shape(shape: Any) -> str:
    """Render a shape in Python tuple notation.

    Parameters
    ----------
    shape:
        Shape-like object, such as ``torch.Size``, tuple, list, or any iterable of
        dimensions.

    Returns
    -------
    str
        Python tuple notation: ``(d1, d2)``, ``(d,)``, or ``()``.
    """

    dims = _shape_tuple(shape)
    if len(dims) == 0:
        return "()"
    if len(dims) == 1:
        return f"({dims[0]},)"
    return f"({', '.join(str(dim) for dim in dims)})"


#: Most per-pass values a rolled scalar node's value row lists; beyond it the
#: row keeps the first and last pass around ``...``.
MAX_LISTED_SCALAR_VALUES: Final[int] = 4


def format_output_shape(shape: Any) -> str:
    """Render a node's output shape for its label's shape row.

    A 0-d tensor reads ``scalar``: ``()`` is the honest Python literal but
    reads as "empty" on a drawing. Every other shape is :func:`format_shape`,
    which kwarg values and parameter brackets keep using unchanged.

    Parameters
    ----------
    shape:
        The node's output shape.

    Returns
    -------
    str
        ``"scalar"`` for an empty shape, else the tuple notation.

    Examples
    --------
    >>> format_output_shape(())
    'scalar'
    >>> format_output_shape((1,))
    '(1,)'
    """

    if shape is not None and len(_shape_tuple(shape)) == 0:
        return "scalar"
    return format_shape(shape)


def scalar_value_line(layer_log: Any) -> str | None:
    """Return a 0-d node's value row (``= 2.31``), or ``None``.

    Only saved activations are read (never recomputed), one ``.item()`` per
    pass. A rolled node lists its passes in order (``= 0.87, 1.02, 0.95``), at
    most :data:`MAX_LISTED_SCALAR_VALUES`, else first, ``...``, last. A
    terminal boolean already prints its outcome on its first row, so it gets
    none; a missing or unreadable value claims nothing.

    Parameters
    ----------
    layer_log:
        The drawn Op or rolled Layer.

    Returns
    -------
    str | None
        The value row, or ``None`` when there is nothing honest to print.
    """

    if layer_log.shape is None or len(_shape_tuple(layer_log.shape)) != 0:
        return None
    if record_member(layer_log, "is_terminal_bool", False):
        return None
    source = getattr(layer_log, "original", layer_log)  # a drawn wrapper names its record
    ops = getattr(getattr(source, "ops", None), "values", None)
    passes = sorted(ops(), key=lambda op: op.pass_number) if callable(ops) else [source]
    texts = [_saved_scalar_text(op) for op in passes]
    if not texts or None in texts:
        return None
    if len(texts) > MAX_LISTED_SCALAR_VALUES:
        texts = [texts[0], "...", texts[-1]]
    return "= " + ", ".join(str(text) for text in texts)


def shape_rows(layer_log: Any, *, with_memory: bool = False) -> list[str]:
    """Return a node's shape row, plus the value row a 0-d node carries.

    Parameters
    ----------
    layer_log:
        The drawn Op or rolled Layer.
    with_memory:
        Append the activation memory to the shape row (``scalar, 4 B``), as
        the default label does.

    Returns
    -------
    list[str]
        The shape row, then the value row when :func:`scalar_value_line` has one.
    """

    shape_row = format_output_shape(layer_log.shape)
    if with_memory:
        shape_row = f"{shape_row}, {format_memory(layer_log.cost.activation_memory)}"
    value_row = scalar_value_line(layer_log)
    return [shape_row] if value_row is None else [shape_row, value_row]


def _saved_scalar_text(op: Any) -> str | None:
    """Return one pass's saved 0-d value as label text, or ``None``."""

    out = getattr(op, "out", None)
    saved = getattr(op, "has_saved_activation", False)
    if out is None or not saved or getattr(out, "ndim", None) != 0:
        return None
    try:
        value = out.item()
    except (RuntimeError, ValueError):  # meta or fake tensors hold no value to print
        return None
    if isinstance(value, bool):
        return str(value).upper()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def format_dtype(dtype: Any) -> str | None:
    """Format an output dtype for a node label row.

    Parameters
    ----------
    dtype:
        Backend dtype object or string (``torch.float16``,
        ``mlx.core.float32``), or ``None``.

    Returns
    -------
    str | None
        The dtype name without its framework namespace (``"float16"``), or
        ``None`` when there is no dtype to show.
    """

    if dtype is None:
        return None
    text = str(dtype).strip()
    if not text:
        return None
    return text.rsplit(".", 1)[-1]


def _shape_with_trainability(shape: Any, trainable: bool | None) -> str:
    """Render a shape with frozen params marked by square brackets.

    Parameters
    ----------
    shape:
        Shape-like object to render.
    trainable:
        Whether the parameter is trainable. ``None`` defaults to trainable notation.

    Returns
    -------
    str
        Shape text using ``(...)`` for trainable or unknown params and ``[...]`` for frozen
        params.
    """

    shape_text = format_shape(shape)
    if trainable is not False:
        return shape_text
    assert shape_text.startswith("(") and shape_text.endswith(")")
    return f"[{shape_text[1:-1]}]"


def saved_for_backward_line(layer_log: Any, vis_mode: str) -> str | None:
    """Build the saved-for-backward disclosure row for one node, if any.

    The row appears only when the captured measurement PROVES autograd
    retained tensors at this op (``num_autograd_tensors > 0``); an op that
    saved nothing, or whose backward graph was never built, gets no row --
    an absent row makes no claim. On rolled multi-pass layers the stored
    fields are already cross-pass sums, disclosed in the row text.

    Parameters
    ----------
    layer_log:
        Op or Layer to annotate.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    str | None
        Plain-text label row, or ``None`` when no retention was measured.
    """

    count = record_member(layer_log, "num_autograd_tensors", None)
    if count is None or count <= 0:
        return None
    suffix = ""
    if vis_mode == "rolled" and getattr(layer_log, "num_passes", 1) > 1:
        suffix = " (total across passes)"
    noun = "tensor" if count == 1 else "tensors"
    memory = record_member(layer_log, "autograd_memory", None)
    if memory is not None:
        return f"saved for backward: {count:,} {noun}, {format_memory(memory)}{suffix}"
    return f"saved for backward: {count:,} {noun}{suffix}"


def format_memory(bytes_or_quantity: Any) -> str:
    """Render memory in TorchLens' human-readable style.

    Parameters
    ----------
    bytes_or_quantity:
        Numeric byte count, preformatted memory string, or object with a useful
        ``str`` representation.

    Returns
    -------
    str
        Human-readable memory such as ``"156 KB"``: decimal units, each 1000
        times the one before (B, KB, MB, GB, TB), so the name matches the divisor.
    """

    if isinstance(bytes_or_quantity, bool) or not isinstance(bytes_or_quantity, int | float):
        return str(bytes_or_quantity)
    return decimal_size(float(bytes_or_quantity))


#: Unit names for the drawing's sizes, each 1000 times the one before. Drawings
#: keep the released names (``"60 KB"``) with the decimal divisor those names
#: mean; card and repr byte quantities print IEC units (``KiB``, divisor 1024).
_DECIMAL_UNITS: Final = ("B", "KB", "MB", "GB", "TB", "PB")


def decimal_size(size: float) -> str:
    """Return a byte count for a drawing, for example ``"60 KB"`` or ``"1.9 KB"``.

    The divisor is 1000, the one the unit names mean. Whole bytes print without
    a decimal (``"48 B"``); larger values print one decimal under the largest
    unit the rounded value reaches, dropped when it is zero (``"588 KB"``, not
    ``"588.0 KB"``), so ``999_999`` bytes is ``"1 MB"``, never ``"1000 KB"``.
    """

    if size < 0:
        raise ValueError("size must be non-negative.")
    if size < 1000:
        return f"{int(size)} B"
    value = size
    unit = _DECIMAL_UNITS[0]
    for unit in _DECIMAL_UNITS[1:]:
        value /= 1000.0
        if round(value, 1) < 1000 or unit == _DECIMAL_UNITS[-1]:
            break
    text = f"{value:,.1f}"
    return f"{text.removesuffix('.0')} {unit}"


#: Counts at or above this print compact (``12.3K``); smaller ones print in full.
COMPACT_COUNT_FROM = 10_000
_COUNT_UNITS: Final = ((1e9, "B"), (1e6, "M"), (1e3, "K"))


def compact_count(count: int, full_below: int = COMPACT_COUNT_FROM) -> str:
    """Return a count for a drawing: in full below ``full_below``, else 3 significant figures.

    A full count carries thousands separators: ``9999`` is ``"9,999"``; ``147968`` is ``"148K"``, ``85798656`` is
    ``"85.8M"`` and ``1_500_000_000`` is ``"1.5B"``. The exact count stays in
    the data and the node tooltip. The header's parameter count passes
    ``full_below=1000`` (``1024`` is ``"1.02K"``).
    """

    if count < full_below:
        return f"{count:,}"
    for index, (divisor, suffix) in enumerate(_COUNT_UNITS):
        if count >= divisor or index == len(_COUNT_UNITS) - 1:
            text = f"{count / divisor:.3g}"
            if float(text) >= 1000 and index > 0:  # 999.96K rounds up into the next unit
                divisor, suffix = _COUNT_UNITS[index - 1]
                text = f"{count / divisor:.3g}"
            return f"{text}{suffix}"
    return f"{count:,}"


#: Parameter counts print compact from this size in every drawn label (the
#: header's parameter line, collapsed module boxes, segment chips), so one
#: drawing never shows ``3.77K params`` beside ``3,600 params``.
PARAM_COMPACT_FROM = 1000


def param_count(count: int) -> str:
    """Return a drawn parameter count: in full below 1,000, else 3 significant figures.

    ``999`` stays ``"999"``; ``3600`` is ``"3.6K"`` and ``85798656`` is ``"85.8M"``.
    """

    return compact_count(count, full_below=PARAM_COMPACT_FROM)


def header_count(count: int) -> str:
    """Return a header count: thousands separators below 10,000, compact from there."""

    return f"{count:,}" if count < COMPACT_COUNT_FROM else compact_count(count)


def collapsed_param_detail(num_params: int, num_frozen: int) -> str:
    """Return a collapsed module box's parameter row, for example ``"85.8M params (all trainable)"``."""

    if num_params == 0:
        return "0 parameters"
    if num_frozen == 0:
        return f"{param_count(num_params)} params (all trainable)"
    if num_frozen == num_params:
        return f"{param_count(num_params)} params (all frozen)"
    trainable = param_count(num_params - num_frozen)
    return f"{param_count(num_params)} params ({trainable} trainable, {param_count(num_frozen)} frozen)"


def format_collapsed_module_contents(num_layers: int, num_buffer_layers: int) -> str:
    """Format the operation and buffer counts for a collapsed module.

    Parameters
    ----------
    num_layers:
        Total tensor-layer count already computed for the module.
    num_buffer_layers:
        Number of buffer tensor layers within ``num_layers``.

    Returns
    -------
    str
        An honest collapsed-module summary such as ``"2 ops + 6 buffers"``.
    """

    num_ops = max(0, num_layers - num_buffer_layers)
    parts: list[str] = []
    if num_ops:
        parts.append(f"{compact_count(num_ops)} {'op' if num_ops == 1 else 'ops'}")
    if num_buffer_layers:
        noun = "buffer" if num_buffer_layers == 1 else "buffers"
        parts.append(f"{compact_count(num_buffer_layers)} {noun}")
    return " + ".join(parts)


def format_module_kwargs(module: Any, suppressed_keys: frozenset[str] = frozenset()) -> str | None:
    """Render captured module/function kwargs in Python keyword syntax.

    Parameters
    ----------
    module:
        Rendered layer-like object. TorchLens stores visualization kwargs in
        ``func_config`` on ``Layer`` and ``Op`` records.
    suppressed_keys:
        Constructor-arg keys the checked-suppression prepass proved
        redundant against this trace's captured shapes
        (``visualization._arg_suppression``). The default (empty) renders
        every arg — render paths without the trace-bearing prepass
        (detached/standalone records) degrade to all-visible.

    Returns
    -------
    str | None
        Comma-separated ``name=value`` entries, or ``None`` when none are available.
    """

    config = record_member(module, "func_config", None)
    if not isinstance(config, Mapping) or len(config) == 0:
        return None

    ordered_keys = [
        key for key in _ordered_kwarg_keys(module, config) if key not in suppressed_keys
    ]
    parts = [f"{key}={_format_value(config[key])}" for key in ordered_keys]
    return ", ".join(parts) if parts else None


def format_param_list(params: Any) -> str | None:
    """Render a parameter list for a visualization node.

    Parameters
    ----------
    params:
        Parameter logs, shape tuples, or a layer-like object exposing ``_param_logs``
        and ``param_shapes``.

    Returns
    -------
    str | None
        ``"params: weight (3072, 768) · bias (3072,)"`` style text, or ``None``.
    """

    param_items = _param_items(params)
    if not param_items:
        return None

    parts: list[str] = []
    for item in param_items:
        name = getattr(item, "name", None)
        shape = getattr(item, "shape", item)
        trainable = getattr(
            item,
            "is_trainable",
            getattr(item, "trainable", getattr(item, "requires_grad", None)),
        )
        shape_text = _shape_with_trainability(shape, trainable)
        if name:
            parts.append(f"{name} {shape_text}")
        else:
            parts.append(shape_text)
    return "params: " + PARAM_SEPARATOR.join(parts)


def format_module_path(address: Any) -> str | None:
    """Render a module path row.

    Parameters
    ----------
    address:
        Module address, optionally carrying the legacy ``<br/>@`` prefix.

    Returns
    -------
    str | None
        ``"@path.to.module"`` (no space after ``@``), or ``None``.
    """

    if address is None:
        return None
    text = str(address).replace("<br/>", "").strip()
    if not text:
        return None
    if text.startswith("@"):
        text = text[1:].strip()
    if not text:
        return None
    return f"@{text}"


def _shape_tuple(shape: Any) -> tuple[Any, ...]:
    """Convert a shape-like object to a tuple."""

    if shape is None:
        return ()
    if isinstance(shape, str):
        return (shape,)
    if isinstance(shape, Sequence):
        return tuple(shape)
    if isinstance(shape, Iterable):
        return tuple(shape)
    return (shape,)


def _ordered_kwarg_keys(module: Any, config: Mapping[str, Any]) -> list[str]:
    """Return config keys in declaration-style order for known layer types."""

    normalized_type = str(getattr(module, "layer_type", "")).lower().replace("_", "")
    order = _KWARG_ORDER.get(normalized_type, ())
    keys = [key for key in order if key in config]
    keys.extend(key for key in config if key not in keys)
    return keys


def _format_value(value: Any) -> str:
    """Render a kwarg value, using tuple notation for shape-like values."""

    if isinstance(value, tuple | list):
        return format_shape(value)
    return str(value)


def _param_items(params: Any) -> list[Any]:
    """Extract parameter-like items from a node or iterable."""

    from torchlens._visualization._render_common import BoundaryNode, FocusNode  # deferred: cycle

    barcodes = getattr(params, "_param_barcodes", None)
    if barcodes is not None and len(barcodes) == 0:
        return []

    source = params.original if isinstance(params, FocusNode) else params
    param_logs = source._param_logs if isinstance(source, (Op, Layer, BoundaryNode)) else None
    if param_logs:
        return list(param_logs)

    param_shapes = getattr(params, "param_shapes", None)
    if param_shapes:
        return list(param_shapes)

    if isinstance(params, Mapping):
        return list(params.values())
    if isinstance(params, Iterable) and not isinstance(params, str):
        return list(params)
    return []


def compute_selected_node_lines(
    layer_log: Any,
    node_address: str,
    vis_mode: str,
    node_label_fields: list[str],
    pass_mark: str = "",
) -> list[str]:
    """Build node-label rows from an explicit field picker.

    Parameters
    ----------
    layer_log:
        Op or Layer to render.
    node_address:
        Existing address suffix from TorchLens node address logic.
    vis_mode:
        ``"unrolled"`` or ``"rolled"``.
    node_label_fields:
        Requested field names.
    pass_mark:
        The default title's pass mark (``:2``, `` (x3)``) the ``"label"`` row keeps.

    Returns
    -------
    list[str]
        Selected label rows.

    Raises
    ------
    ValueError
        If an unknown field is requested.
    """

    rows: list[str] = []
    for field_name in node_label_fields:
        if field_name in {"label", "name"}:
            rows.append(f"{getattr(layer_log, 'layer_label', '')}{pass_mark}")
        elif field_name in {"type", "op", "operation"}:
            rows.append(str(getattr(layer_log, "func_name", None) or record_type(layer_log)))
        elif field_name == "shape":
            rows.extend(shape_rows(layer_log))
        elif field_name == "shape_summary":
            # L1's across-pass summary: row only when the field is set
            # (same skip-when-absent semantics as "params").
            summary = getattr(layer_log, "shape_summary", None)
            if isinstance(summary, str) and summary:
                rows.append(summary)
        elif field_name == "dtype":
            # Output dtype, framework prefix dropped ("float16", Keras
            # show_dtype style); skipped when the node has no tensor dtype.
            dtype_text = format_dtype(getattr(layer_log, "dtype", None))
            if dtype_text is not None:
                rows.append(dtype_text)
        elif field_name in {"memory", "bytes"}:
            rows.append(str(record_member(layer_log, "activation_memory", "")))
        elif field_name == "module":
            rows.append(format_module_path(node_address) or "@root")
        elif field_name == "params":
            param_line = format_param_list(layer_log)
            if param_line is not None:
                rows.append(param_line)
        elif field_name == "pass":
            # The real 1-based pass (unrolled) or the pass count (rolled); no row
            # on a single-pass node, where a "1" on every node is noise.
            rows.extend(pass_rows(layer_log, vis_mode))
        elif field_name == "flops":
            rows.append(str(record_member(layer_log, "flops_forward", 0) or 0))
            append_attention_note(rows, layer_log)
        elif field_name == "time":
            if is_multipass_layer(layer_log):
                # Rolled recurrent node: ``func_duration`` is per-pass and would
                # leak the multi-pass ValueError tripwire. Report the aggregate
                # total across passes (same choice as the rolled summary builder's
                # ``total_func_duration``), an honest total rather than a crash.
                duration = float(record_member(layer_log, "total_func_duration", 0.0) or 0.0)
            else:
                duration = float(get_multipass_attr(layer_log, "func_duration", 0.0) or 0.0)
            rows.append(str(Duration(duration)))
        else:
            raise InvalidArgumentError(
                f"Unsupported node label field: {field_name!r}",
                code="node_label_field_invalid",
                remedy="pass documented node label field names",
                field=str(field_name),
            )
    if rows:
        return rows
    # Empty selection (every requested row skipped) falls back to the
    # default label rows; lazy import avoids a module cycle with the
    # node renderer that calls this picker.
    from ._render_nodes import compute_default_node_lines  # deferred: cycle _visualization

    return list(compute_default_node_lines(layer_log, node_address, vis_mode))

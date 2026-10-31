"""Formatting helpers for TorchLens visualization node labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from .._errors import InvalidArgumentError
from .._quantities import Duration
from .._utils._multipass_access import get_multipass_attr, is_multipass_layer
from .._utils.display import human_readable_size
from ..capture.flops import append_attention_note

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

    count = getattr(layer_log, "num_autograd_tensors", None)
    if count is None or count <= 0:
        return None
    suffix = ""
    if vis_mode == "rolled" and getattr(layer_log, "num_passes", 1) > 1:
        suffix = " (total across passes)"
    noun = "tensor" if count == 1 else "tensors"
    memory = getattr(layer_log, "autograd_memory", None)
    if memory is not None:
        return f"saved for backward: {count} {noun}, {format_memory(memory)}{suffix}"
    return f"saved for backward: {count} {noun}{suffix}"


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
        Human-readable memory such as ``"156.0 KB"``.
    """

    if isinstance(bytes_or_quantity, int | float):
        return human_readable_size(float(bytes_or_quantity))
    return str(bytes_or_quantity)


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

    config = getattr(module, "func_config", None)
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

    barcodes = getattr(params, "_param_barcodes", None)
    if barcodes is not None and len(barcodes) == 0:
        return []

    param_logs = getattr(params, "_param_logs", None)
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
            rows.append(str(getattr(layer_log, "layer_label", "")))
        elif field_name in {"type", "op", "operation"}:
            rows.append(str(getattr(layer_log, "func_name", None) or layer_log.layer_type))
        elif field_name == "shape":
            rows.append(format_shape(layer_log.shape))
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
            rows.append(str(getattr(layer_log, "activation_memory", "")))
        elif field_name == "module":
            rows.append(format_module_path(node_address) or "@root")
        elif field_name == "params":
            param_line = format_param_list(layer_log)
            if param_line is not None:
                rows.append(param_line)
        elif field_name == "pass":
            if vis_mode == "unrolled":
                # Per-pass leaf node: show the op's real 1-based recurrent pass
                # (``pass_index``). The old ``call_index`` default read 1 for every
                # pass because Ops carry no ``call_index`` -- a field literally
                # named "pass" that always says 1 is silent wrongness.
                rows.append(str(get_multipass_attr(layer_log, "pass_index", 1, multipass=1)))
            else:
                rows.append(str(get_multipass_attr(layer_log, "num_passes", 1)))
        elif field_name == "flops":
            rows.append(str(getattr(layer_log, "flops_forward", 0) or 0))
            append_attention_note(rows, layer_log)
        elif field_name == "time":
            if is_multipass_layer(layer_log):
                # Rolled recurrent node: ``func_duration`` is per-pass and would
                # leak the multi-pass ValueError tripwire. Report the aggregate
                # total across passes (same choice as the rolled summary builder's
                # ``total_func_duration``), an honest total rather than a crash.
                duration = float(getattr(layer_log, "total_func_duration", 0.0) or 0.0)
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
    from ._render_nodes import compute_default_node_lines

    return list(compute_default_node_lines(layer_log, node_address, vis_mode))

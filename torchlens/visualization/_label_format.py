"""Formatting helpers for TorchLens visualization node labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from ..utils.display import human_readable_size

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


def format_module_kwargs(module: Any) -> str | None:
    """Render captured module/function kwargs in Python keyword syntax.

    Parameters
    ----------
    module:
        Rendered layer-like object. TorchLens stores visualization kwargs in
        ``func_config`` on ``Layer`` and ``Op`` records.

    Returns
    -------
    str | None
        Comma-separated ``name=value`` entries, or ``None`` when none are available.
    """

    config = getattr(module, "func_config", None)
    if not isinstance(config, Mapping) or len(config) == 0:
        return None

    ordered_keys = _ordered_kwarg_keys(module, config)
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
        shape_text = format_shape(shape)
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
        ``"@ path.to.module"`` with one space after ``@``, or ``None``.
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
    return f"@ {text}"


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

    if getattr(params, "num_param_tensors", None) == 0:
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

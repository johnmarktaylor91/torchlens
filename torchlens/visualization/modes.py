"""Node-mode presets for TorchLens graph visualization."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from .._literals import VisNodeModeLiteral
from ..utils.display import human_readable_size
from .node_spec import NodeSpec

if TYPE_CHECKING:
    from ..data_classes.layer_log import LayerLog
    from ..data_classes.module_log import ModuleLog

NodeModeFn = Callable[["LayerLog", NodeSpec], NodeSpec]
CollapsedNodeModeFn = Callable[["ModuleLog", NodeSpec], NodeSpec]

VISION_LAYER_TYPES: Final[frozenset[str]] = frozenset(
    {
        "conv1d",
        "conv2d",
        "conv3d",
        "convolution",
        "convtranspose1d",
        "convtranspose2d",
        "convtranspose3d",
        "maxpool1d",
        "maxpool2d",
        "maxpool3d",
        "avgpool1d",
        "avgpool2d",
        "avgpool3d",
        "adaptiveavgpool1d",
        "adaptiveavgpool2d",
        "adaptiveavgpool3d",
        "adaptivemaxpool1d",
        "adaptivemaxpool2d",
        "adaptivemaxpool3d",
        "upsample",
        "interpolate",
        "resize",
    }
)
ATTENTION_PROJECTION_ROLES: Final[dict[str, str]] = {
    "q_proj": "q",
    "k_proj": "k",
    "v_proj": "v",
    "out_proj": "out",
    "qkv_proj": "qkv",
}
DOMAIN_NODE_MODES: Final[frozenset[str]] = frozenset({"vision", "attention"})


def default_node_mode(layer_log: "LayerLog", spec: NodeSpec) -> NodeSpec:
    """Return the default node spec unchanged.

    Parameters
    ----------
    layer_log:
        Layer being rendered.
    spec:
        Current node spec.

    Returns
    -------
    NodeSpec
        The unchanged node spec.
    """

    del layer_log
    return spec


def profiling_node_mode(layer_log: "LayerLog", spec: NodeSpec) -> NodeSpec:
    """Append runtime, output storage, and source-call details when available.

    Parameters
    ----------
    layer_log:
        Layer being rendered.
    spec:
        Current node spec.

    Returns
    -------
    NodeSpec
        Node spec with profiling rows appended.
    """

    lines = list(spec.lines)
    runtime = _get_optional_attr(layer_log, "func_time")
    if isinstance(runtime, int | float):
        lines.append(f"t={runtime * 1000:.2f}ms")

    tensor_memory = _get_optional_attr(layer_log, "tensor_memory")
    if isinstance(tensor_memory, int | float):
        lines.append(f"out={_compact_size(float(tensor_memory))}")

    call_location = _first_call_location(layer_log)
    if call_location is not None:
        file_name = Path(call_location.file).name
        lines.append(f"call={file_name}:{call_location.line_number}")
        function_name = call_location.code_qualname or call_location.func_name
        if function_name:
            lines.append(f"fn={function_name}")
    else:
        func_name = _get_optional_attr(layer_log, "func_name")
        if isinstance(func_name, str) and func_name:
            lines.append(f"fn={func_name}")
    return spec.replace(lines=lines)


def vision_node_mode(layer_log: "LayerLog", spec: NodeSpec) -> NodeSpec:
    """Append input/output spatial shapes for vision-like layers.

    Parameters
    ----------
    layer_log:
        Layer being rendered.
    spec:
        Current node spec.

    Returns
    -------
    NodeSpec
        Node spec with an IO-shape row for spatial layers.
    """

    if _normalized_layer_type(layer_log) not in VISION_LAYER_TYPES:
        return spec

    input_shape = _first_input_shape(layer_log)
    output_shape = _get_optional_attr(layer_log, "tensor_shape")
    if not isinstance(input_shape, tuple) or not isinstance(output_shape, tuple):
        return spec

    lines = list(spec.lines)
    lines.append(f"in={_format_shape(input_shape)} out={_format_shape(output_shape)}")
    return spec.replace(lines=lines)


def attention_node_mode(layer_log: "LayerLog", spec: NodeSpec) -> NodeSpec:
    """Append compact annotations for attention-related layers.

    Parameters
    ----------
    layer_log:
        Layer being rendered.
    spec:
        Current node spec.

    Returns
    -------
    NodeSpec
        Node spec with attention details when heuristics match.
    """

    lines = list(spec.lines)
    layer_type = _normalized_layer_type(layer_log)
    containing_mha = _is_inside_multihead_attention(layer_log)

    if "multiheadattention" in layer_type or layer_type == "scaleddotproductattention":
        attention_line = _format_attention_head_line(layer_log)
        if attention_line:
            lines.append(attention_line)
        dropout = _attention_dropout(layer_log)
        if dropout is not None and dropout != 0:
            lines.append(f"dropout={dropout:g}")

    role = _attention_projection_role(layer_log, containing_mha)
    if role is not None:
        lines.append(f"(role={role})")

    if "softmax" in layer_type and containing_mha:
        dim = _get_optional_attr(layer_log, "func_config")
        if isinstance(dim, dict) and dim.get("dim") == -1:
            lines.append("(attn-softmax)")
    return spec.replace(lines=lines) if lines != spec.lines else spec


def profiling_collapsed_node_mode(module_log: "ModuleLog", spec: NodeSpec) -> NodeSpec:
    """Append aggregate runtime and output storage for a collapsed module.

    Parameters
    ----------
    module_log:
        Collapsed module being rendered.
    spec:
        Current module node spec.

    Returns
    -------
    NodeSpec
        Module node spec with aggregate profiling rows when available.
    """

    model_log = getattr(module_log, "_source_model_log", None)
    if model_log is None:
        return spec

    runtime = 0.0
    saw_runtime = False
    output_bytes = 0.0
    saw_output = False
    for layer_label in getattr(module_log, "all_layers", []):
        layer_log = model_log[layer_label]
        layer_runtime = _get_optional_attr(layer_log, "func_time")
        if isinstance(layer_runtime, int | float):
            runtime += float(layer_runtime)
            saw_runtime = True
        tensor_memory = _get_optional_attr(layer_log, "tensor_memory")
        if isinstance(tensor_memory, int | float):
            output_bytes += float(tensor_memory)
            saw_output = True

    lines = list(spec.lines)
    if saw_runtime:
        lines.append(f"t={runtime * 1000:.2f}ms")
    if saw_output:
        lines.append(f"out={_compact_size(output_bytes)}")
    return spec.replace(lines=lines)


def identity_collapsed_node_mode(module_log: "ModuleLog", spec: NodeSpec) -> NodeSpec:
    """Return a collapsed module node spec unchanged.

    Parameters
    ----------
    module_log:
        Module being rendered.
    spec:
        Current module node spec.

    Returns
    -------
    NodeSpec
        The unchanged node spec.
    """

    del module_log
    return spec


MODE_REGISTRY: Final[dict[VisNodeModeLiteral, NodeModeFn]] = {
    "default": default_node_mode,
    "profiling": profiling_node_mode,
    "vision": vision_node_mode,
    "attention": attention_node_mode,
}
COLLAPSED_MODE_REGISTRY: Final[dict[VisNodeModeLiteral, CollapsedNodeModeFn]] = {
    "default": identity_collapsed_node_mode,
    "profiling": profiling_collapsed_node_mode,
    "vision": identity_collapsed_node_mode,
    "attention": identity_collapsed_node_mode,
}


def _get_optional_attr(obj: object, attr_name: str) -> Any:
    """Read an attribute, returning ``None`` when LayerLog delegation cannot satisfy it."""

    try:
        return getattr(obj, attr_name)
    except (AttributeError, ValueError):
        return None


def _compact_size(size: float) -> str:
    """Format bytes without the space used by the generic display helper."""

    return human_readable_size(size).replace(" ", "")


def _first_call_location(layer_log: "LayerLog") -> Any | None:
    """Return the first captured call-stack location for a layer."""

    call_stack = _get_optional_attr(layer_log, "func_call_stack")
    if isinstance(call_stack, list) and call_stack:
        return call_stack[0]
    return None


def _normalized_layer_type(layer_log: "LayerLog") -> str:
    """Return a lower-case layer type with underscores removed."""

    return str(layer_log.layer_type).lower().replace("_", "")


def _first_input_shape(layer_log: "LayerLog") -> tuple[int, ...] | None:
    """Infer the first tensor input shape from parent graph metadata or captured args."""

    model_log = _get_optional_attr(layer_log, "source_model_log")
    parent_layers = _get_optional_attr(layer_log, "parent_layers")
    if model_log is not None and isinstance(parent_layers, list):
        for parent_label in parent_layers:
            parent = model_log[parent_label]
            shape = _get_optional_attr(parent, "tensor_shape")
            if isinstance(shape, tuple):
                return shape

    captured_args = _get_optional_attr(layer_log, "captured_args")
    if isinstance(captured_args, list):
        for value in captured_args:
            if isinstance(value, torch.Tensor):
                return tuple(value.shape)
    return None


def _format_shape(shape: tuple[Any, ...]) -> str:
    """Format a tensor shape compactly for a preset row."""

    return "x".join(str(dim) for dim in shape) if shape else "x1"


def _is_inside_multihead_attention(layer_log: "LayerLog") -> bool:
    """Return whether the layer belongs to a recorded MultiheadAttention module."""

    model_log = _get_optional_attr(layer_log, "source_model_log")
    containing_modules = _get_optional_attr(layer_log, "containing_modules")
    if model_log is None or not isinstance(containing_modules, list):
        return False
    for module_pass in containing_modules:
        module_address = str(module_pass).rsplit(":", 1)[0]
        module_log = model_log.modules[module_address]
        if module_log.module_class_name == "MultiheadAttention":
            return True
    return False


def _format_attention_head_line(layer_log: "LayerLog") -> str:
    """Format head/embed/head-dim details for an attention operation."""

    shape = _get_optional_attr(layer_log, "tensor_shape")
    heads: int | None = None
    head_dim: int | None = None
    if isinstance(shape, tuple) and len(shape) >= 4:
        heads = int(shape[1])
        head_dim = int(shape[-1])
    config = _get_optional_attr(layer_log, "func_config")
    if isinstance(config, dict):
        heads = int(config["num_heads"]) if "num_heads" in config else heads
        embed_dim = config.get("embed_dim")
        if isinstance(embed_dim, int) and heads is not None and heads != 0:
            head_dim = embed_dim // heads
    embed = heads * head_dim if heads is not None and head_dim is not None else None
    parts: list[str] = []
    if heads is not None and embed is not None:
        parts.append(f"heads={heads} embed={embed}")
    if head_dim is not None:
        parts.append(f"head_dim={head_dim}")
    return " ".join(parts)


def _attention_dropout(layer_log: "LayerLog") -> float | None:
    """Return non-structural attention dropout captured in layer config."""

    config = _get_optional_attr(layer_log, "func_config")
    if not isinstance(config, dict):
        return None
    value = config.get("dropout", config.get("dropout_p"))
    return float(value) if isinstance(value, int | float) else None


def _attention_projection_role(layer_log: "LayerLog", containing_mha: bool) -> str | None:
    """Infer q/k/v/out projection role from module address and MHA context."""

    layer_type = _normalized_layer_type(layer_log)
    if "linear" not in layer_type:
        return None

    containing_modules = _get_optional_attr(layer_log, "containing_modules")
    if isinstance(containing_modules, list):
        for module_pass in containing_modules:
            segment = str(module_pass).rsplit(":", 1)[0].split(".")[-1]
            if segment in ATTENTION_PROJECTION_ROLES:
                return ATTENTION_PROJECTION_ROLES[segment]

    if not containing_mha:
        return None
    config = _get_optional_attr(layer_log, "func_config")
    if isinstance(config, dict) and config.get("out_features") == config.get("in_features"):
        return "out"
    return "qkv"

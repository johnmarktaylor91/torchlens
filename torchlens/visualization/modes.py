"""Node-mode presets for TorchLens graph visualization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch

from .._literals import VisNodeModeLiteral
from ..quantities import Duration
from ..utils._multipass_access import get_multipass_attr, is_multipass_layer
from ..utils.display import human_readable_size
from ._label_format import format_shape
from .node_spec import NodeSpec

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.module import Module
    from ..data_classes.op import Op

#: Mode presets receive the per-pass ``Op`` on unrolled nodes and the
#: aggregate ``Layer`` on rolled nodes, so per-pass rows (``t=``, ``call=``)
#: are exact where the node is per-pass and explicitly aggregated where it
#: is not. User ``node_spec_fn`` callbacks keep their parent-Layer contract.
NodeModeFn = Callable[["Op | Layer", NodeSpec], NodeSpec]
CollapsedNodeModeFn = Callable[["Module", NodeSpec, "CollapsedModeScope"], NodeSpec]


@dataclass(frozen=True)
class CollapsedModeScope:
    """Op inventory of exactly the calls one collapsed box represents.

    Parameters
    ----------
    op_labels:
        Pass-qualified op labels covered by the box: the single call for an
        unrolled per-call box, every call for a rolled box. Surfaced
        own-output ops rendered outside the box are excluded, matching the
        box's own tensor-count line.
    note:
        Disclosure appended to aggregate rows when the box title claims more
        than ``op_labels`` covers (a repeat-fold representative standing for
        a run of siblings), else ``None``.
    """

    op_labels: tuple[str, ...]
    note: str | None = None


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
# The former "vision"/"attention" node_style presets are removed; the style
# functions stay exported through torchlens.experimental.node_styles for use
# via node_spec_fn.


def default_node_mode(layer_log: Op | Layer, spec: NodeSpec) -> NodeSpec:
    """Return the default node spec unchanged.

    Parameters
    ----------
    layer_log:
        Per-pass Op or aggregate Layer being rendered.
    spec:
        Current node spec.

    Returns
    -------
    NodeSpec
        The unchanged node spec.
    """

    del layer_log
    return spec


def profiling_node_mode(layer_log: Op | Layer, spec: NodeSpec) -> NodeSpec:
    """Append runtime, output storage, and source-call details when available.

    Parameters
    ----------
    layer_log:
        Per-pass Op (unrolled nodes) or aggregate Layer (rolled nodes)
        being rendered.
    spec:
        Current node spec.

    Returns
    -------
    NodeSpec
        Node spec with profiling rows appended. On a rolled multi-pass
        layer the per-pass fields have no single value, so runtime and
        storage rows are exact cross-pass totals with the aggregation
        disclosed, and call/fn rows appear only when every pass agrees
        (a varying source is disclosed, never projected from one pass).
    """

    if is_multipass_layer(layer_log):
        multipass_layer = cast("Layer", layer_log)
        return spec.replace(lines=_multipass_profiling_lines(multipass_layer, list(spec.lines)))

    lines = list(spec.lines)
    runtime = _get_optional_attr(layer_log, "func_duration")
    if isinstance(runtime, int | float):
        lines.append(f"t={Duration(float(runtime)):.2f}")

    memory = _get_optional_attr(layer_log, "activation_memory")
    if isinstance(memory, int | float):
        lines.append(f"out={_compact_size(float(memory))}")

    call_location = _first_call_location(layer_log)
    if call_location is not None:
        file_name = Path(call_location.file).name
        lines.append(f"call={file_name}:{call_location.line_number}")
        function_name = call_location.func_qualname or call_location.func_name
        if function_name:
            lines.append(f"fn={function_name}")
    else:
        func_name = _get_optional_attr(layer_log, "func_name")
        if isinstance(func_name, str) and func_name:
            lines.append(f"fn={func_name}")
    return spec.replace(lines=lines)


def _multipass_profiling_lines(layer_log: Layer, lines: list[str]) -> list[str]:
    """Append honest aggregate profiling rows for a rolled multi-pass layer.

    Per-pass reads on the aggregate Layer trip the deliberate
    ``layer_pass_ambiguous`` refusal; the historical caller swallowed it and
    every profiling row silently vanished. Aggregate from the per-pass Ops
    instead: exact totals with the aggregation disclosed, unanimity-checked
    call/fn rows, and an explicit "varies across passes" disclosure where no
    single honest value exists.
    """

    ops = list(layer_log.ops.values())
    n_passes = len(ops)

    durations = [_get_optional_attr(op, "func_duration") for op in ops]
    timed = [float(d) for d in durations if isinstance(d, int | float)]
    if timed:
        lines.append(f"t={Duration(sum(timed)):.2f} (total across {n_passes} passes)")

    memories = [_get_optional_attr(op, "activation_memory") for op in ops]
    sized = [float(m) for m in memories if isinstance(m, int | float)]
    if sized:
        lines.append(f"out={_compact_size(sum(sized))} (total across {n_passes} passes)")

    locations = [_first_call_location(op) for op in ops]
    location_keys = {(loc.file, loc.line_number) if loc is not None else None for loc in locations}
    if location_keys == {None}:
        call_location = None
    elif len(location_keys) == 1:
        call_location = locations[0]
    else:
        lines.append("call=varies across passes")
        call_location = None

    if call_location is not None:
        file_name = Path(call_location.file).name
        lines.append(f"call={file_name}:{call_location.line_number}")
        function_name = call_location.func_qualname or call_location.func_name
        if function_name:
            lines.append(f"fn={function_name}")
        return lines

    func_names = {_get_optional_attr(op, "func_name") for op in ops}
    if len(func_names) == 1:
        func_name = next(iter(func_names))
        if isinstance(func_name, str) and func_name:
            lines.append(f"fn={func_name}")
    else:
        lines.append("fn=varies across passes")
    return lines


def vision_node_mode(layer_log: Op | Layer, spec: NodeSpec) -> NodeSpec:
    """Append input/output spatial shapes for vision-like layers.

    Parameters
    ----------
    layer_log:
        Per-pass Op or aggregate Layer being rendered.
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
    output_shape = _get_optional_attr(layer_log, "shape")
    if not isinstance(input_shape, tuple) or not isinstance(output_shape, tuple):
        return spec

    lines = list(spec.lines)
    lines.append(f"in={format_shape(input_shape)}, out={format_shape(output_shape)}")
    return spec.replace(lines=lines)


def attention_node_mode(layer_log: Op | Layer, spec: NodeSpec) -> NodeSpec:
    """Append compact annotations for attention-related layers.

    Parameters
    ----------
    layer_log:
        Per-pass Op or aggregate Layer being rendered.
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


def profiling_collapsed_node_mode(
    module_log: Module, spec: NodeSpec, scope: CollapsedModeScope
) -> NodeSpec:
    """Append aggregate runtime and output storage for a collapsed module.

    Parameters
    ----------
    module_log:
        Collapsed module being rendered.
    spec:
        Current module node spec.
    scope:
        Pass-qualified op inventory the box actually represents, with an
        optional scope-disclosure note. Aggregating per-pass Op records
        (never per-Layer reads) keeps recurrent layers fully counted: a
        per-Layer ``func_duration`` read refuses on multi-pass layers, and
        a Layer-level total would leak passes executed outside this box.

    Returns
    -------
    NodeSpec
        Module node spec with aggregate profiling rows when available.
    """

    trace = getattr(module_log, "_source_trace", None)
    if trace is None:
        return spec

    runtime = 0.0
    saw_runtime = False
    output_bytes = 0.0
    saw_output = False
    for op_label in scope.op_labels:
        op = trace.ops[op_label]
        op_runtime = _get_optional_attr(op, "func_duration")
        if isinstance(op_runtime, int | float):
            runtime += float(op_runtime)
            saw_runtime = True
        memory = _get_optional_attr(op, "activation_memory")
        if isinstance(memory, int | float):
            output_bytes += float(memory)
            saw_output = True

    suffix = f" ({scope.note})" if scope.note else ""
    lines = list(spec.lines)
    if saw_runtime:
        lines.append(f"t={Duration(runtime):.2f}{suffix}")
    if saw_output:
        lines.append(f"out={_compact_size(output_bytes)}{suffix}")
    return spec.replace(lines=lines)


def identity_collapsed_node_mode(
    module_log: Module, spec: NodeSpec, scope: CollapsedModeScope
) -> NodeSpec:
    """Return a collapsed module node spec unchanged.

    Parameters
    ----------
    module_log:
        Module being rendered.
    spec:
        Current module node spec.
    scope:
        Box op inventory (unused by the identity preset).

    Returns
    -------
    NodeSpec
        The unchanged node spec.
    """

    del module_log, scope
    return spec


MODE_REGISTRY: Final[dict[VisNodeModeLiteral, NodeModeFn]] = {
    "default": default_node_mode,
    "profiling": profiling_node_mode,
}
COLLAPSED_MODE_REGISTRY: Final[dict[VisNodeModeLiteral, CollapsedNodeModeFn]] = {
    "default": identity_collapsed_node_mode,
    "profiling": profiling_collapsed_node_mode,
}


def _get_optional_attr(obj: object, attr_name: str) -> Any:
    """Read an attribute for a label row, degrading honestly to ``None``.

    An absent attribute (detached or legacy record) returns ``None``. A
    per-pass read on an aggregate multi-pass Layer returns the EXPLICIT
    ``None`` marker (the row is omitted rather than projected from one
    pass); ``profiling_node_mode`` branches to per-Op aggregation before
    any such read. Every other ``ValueError`` propagates — the historical
    blanket ``except ValueError`` silently swallowed the deliberate
    ``layer_pass_ambiguous`` tripwire (the bug class
    ``utils._multipass_access`` documents).
    """

    return get_multipass_attr(obj, attr_name, default=None, multipass=None)


def _compact_size(size: float) -> str:
    """Format bytes without the space used by the generic display helper."""

    return human_readable_size(size).replace(" ", "")


def _first_call_location(layer_log: Op | Layer) -> Any | None:
    """Return the first captured call-stack location for a layer."""

    call_stack = _get_optional_attr(layer_log, "code_context")
    if isinstance(call_stack, list) and call_stack:
        return call_stack[0]
    return None


def _normalized_layer_type(layer_log: Op | Layer) -> str:
    """Return a lower-case layer type with underscores removed."""

    return str(layer_log.layer_type).lower().replace("_", "")


def _first_input_shape(layer_log: Op | Layer) -> tuple[int, ...] | None:
    """Infer the first tensor input shape from parent graph metadata or captured args."""

    trace = _get_optional_attr(layer_log, "source_trace")
    parents = _get_optional_attr(layer_log, "parents")
    if trace is not None and isinstance(parents, (list, tuple)):
        for parent_label in parents:
            parent = trace[parent_label]
            shape = _get_optional_attr(parent, "shape")
            if isinstance(shape, tuple):
                return shape

    saved_args = _get_optional_attr(layer_log, "saved_args")
    if isinstance(saved_args, list):
        for value in saved_args:
            if isinstance(value, torch.Tensor):
                return tuple(value.shape)
    return None


def _is_inside_multihead_attention(layer_log: Op | Layer) -> bool:
    """Return whether the layer belongs to a recorded MultiheadAttention module."""

    trace = _get_optional_attr(layer_log, "source_trace")
    modules = _get_optional_attr(layer_log, "modules")
    if trace is None or not isinstance(modules, (list, tuple)):
        return False
    for module_pass in modules:
        address = str(module_pass).rsplit(":", 1)[0]
        module_log = trace.modules[address]
        if module_log.class_name == "MultiheadAttention":
            return True
    return False


def _format_attention_head_line(layer_log: Op | Layer) -> str:
    """Format head/embed/head-dim details for an attention operation."""

    shape = _get_optional_attr(layer_log, "shape")
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


def _attention_dropout(layer_log: Op | Layer) -> float | None:
    """Return non-structural attention dropout captured in layer config."""

    config = _get_optional_attr(layer_log, "func_config")
    if not isinstance(config, dict):
        return None
    value = config.get("dropout", config.get("dropout_p"))
    return float(value) if isinstance(value, int | float) else None


def _attention_projection_role(layer_log: Op | Layer, containing_mha: bool) -> str | None:
    """Infer q/k/v/out projection role from module address and MHA context."""

    layer_type = _normalized_layer_type(layer_log)
    if "linear" not in layer_type:
        return None

    modules = _get_optional_attr(layer_log, "modules")
    if isinstance(modules, (list, tuple)):
        for module_pass in modules:
            segment = str(module_pass).rsplit(":", 1)[0].split(".")[-1]
            if segment in ATTENTION_PROJECTION_ROLES:
                return ATTENTION_PROJECTION_ROLES[segment]

    if not containing_mha:
        return None
    config = _get_optional_attr(layer_log, "func_config")
    if isinstance(config, dict) and config.get("out_features") == config.get("in_features"):
        return "out"
    return "qkv"

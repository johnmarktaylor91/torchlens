"""Field extraction for the netron projection (lane F14, netron memo D-16..D-18).

Everything here reads finished-``Trace`` records and returns small python
values; the wording contract (memo D-17) is enforced at this one seam:

- attribute names are chosen knowing netron sorts them alphabetically,
  case-insensitively -- names ARE the display order;
- the module-path attribute is never named ``module`` (netron already labels
  the ONNX domain "module" in the sidebar);
- observed duration, never latency; FLOPs are estimates; activation bytes
  are captured-tensor bytes; MISSING IS ABSENT, NEVER ZERO.

All attribute spellings are placeholders for the naming sprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._netron_records import NetronAttr

__tl_layer__ = "L8"

#: torch dtype name -> ONNX ``TensorProto.DataType`` integer. Unknown dtypes
#: OMIT the type and keep the exact dtype string as a property (memo D-05);
#: elemType 0 is UNDEFINED pretending to be information and is never emitted.
_ONNX_ELEM_TYPES: dict[str, int] = {
    "float32": 1,
    "uint8": 2,
    "int8": 3,
    "uint16": 4,
    "int16": 5,
    "int32": 6,
    "int64": 7,
    "bool": 9,
    "float16": 10,
    "float64": 11,
    "uint32": 12,
    "uint64": 13,
    "complex64": 14,
    "complex128": 15,
    "bfloat16": 16,
    "float8_e4m3fn": 17,
    "float8_e4m3fnuz": 18,
    "float8_e5m2": 19,
    "float8_e5m2fnuz": 20,
}


def dtype_string(dtype: Any) -> str:
    """Return the exact short dtype spelling (``torch.float32`` -> ``float32``)."""

    text = str(dtype)
    return text[6:] if text.startswith("torch.") else text


def onnx_elem_type(dtype: Any) -> int | None:
    """Map a captured dtype to its ONNX elemType; ``None`` = honest unknown."""

    if dtype is None:
        return None
    return _ONNX_ELEM_TYPES.get(dtype_string(dtype))


def _quantity(value: Any) -> float | None:
    """Coerce a TorchLens quantity (Duration/Flops/Bytes) to a float, or None."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_leaf_attrs(
    entry: Any,
    *,
    intervened: bool = False,
    rolled_members: list[Any] | None = None,
) -> list[NetronAttr]:
    """Build the curated ~8-fact attribute set for one leaf node (memo D-17).

    Parameters
    ----------
    entry:
        Layer-pass entry (or the first pass of a rolled group).
    intervened:
        Whether the intervention audit marks this node (compo row C2).
    rolled_members:
        All passes of a rolled group; duration and flops then sum across
        passes and per-pass facts (``pass``) are skipped.

    Returns
    -------
    list[NetronAttr]
        Attributes in their final alphabetical display order.
    """

    from ._netron_records import NetronAttr

    attrs: list[NetronAttr] = []
    dtype = getattr(entry, "dtype", None)
    if dtype is not None:
        attrs.append(NetronAttr("dtype", "s", dtype_string(dtype)))
    members = rolled_members or [entry]
    flops = [q for m in members if (q := _quantity(getattr(m, "flops_forward", None)))]
    if flops:
        attrs.append(NetronAttr("flops_estimated", "i", int(sum(flops))))
    if intervened:
        attrs.append(NetronAttr("intervened", "s", "true"))
    module_path = module_path_of(entry)
    if module_path:
        attrs.append(NetronAttr("module_path", "s", module_path))
    durations = [q for m in members if (q := _quantity(getattr(m, "func_duration", None)))]
    if durations:
        attrs.append(NetronAttr("observed_duration_us", "i", int(round(sum(durations) * 1e6))))
    memory = _quantity(getattr(entry, "activation_memory", None))
    if memory:
        attrs.append(NetronAttr("output_tensor_bytes", "i", int(memory)))
    params = getattr(entry, "num_params", None)
    if params:
        attrs.append(NetronAttr("params", "i", int(params)))
    if rolled_members is None and int(getattr(entry, "num_passes", 1) or 1) > 1:
        attrs.append(NetronAttr("pass", "i", int(getattr(entry, "pass_index", 0))))
    shape = getattr(entry, "shape", None)
    if shape is not None and all(
        isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
    ):
        attrs.append(NetronAttr("shape", "ints", [int(dim) for dim in shape]))
    return attrs


def rolled_variance_attrs(members: list[Any], *, feedback_sources: list[str]) -> list[NetronAttr]:
    """Disclose cross-pass variance for one rolled node (memo D-13).

    A rolled edge is typed only when every member pass agrees on shape and
    dtype; otherwise the variants ride these properties and the edge stays
    untyped -- asserting a false invariant is worse than the clutter.
    Recurrent feedback edges (which would draw as dataflow cycles the ONNX
    checker rejects) are disclosed by the ``recurrence`` attribute.
    """

    from ._netron_records import NetronAttr

    attrs: list[NetronAttr] = []
    dtypes: list[str] = []
    shapes: list[str] = []
    for member in members:
        dtype = dtype_string(getattr(member, "dtype", None))
        if dtype not in dtypes:
            dtypes.append(dtype)
        shape = "x".join(str(dim) for dim in (getattr(member, "shape", ()) or ()))
        if shape not in shapes:
            shapes.append(shape)
    if len(dtypes) > 1:
        attrs.append(NetronAttr("dtype_variants", "s", " | ".join(dtypes)))
    attrs.append(NetronAttr("passes", "i", len(members)))
    if feedback_sources:
        attrs.append(
            NetronAttr(
                "recurrence",
                "s",
                f"feedback from the previous pass ({len(members)} passes): "
                + ", ".join(feedback_sources),
            )
        )
    if len(shapes) > 1:
        attrs.append(NetronAttr("shape_variants", "s", " | ".join(shapes)))
    return attrs


def module_path_of(entry: Any) -> str:
    """Return the innermost containing module-call address for the sidebar."""

    stack = getattr(entry, "module_call_stack", None) or ()
    if not stack:
        return ""
    return _strip_single_pass(str(stack[-1]))


def _strip_single_pass(call_label: str) -> str:
    """Drop the gratuitous ``:1`` from single-call spellings (memo D-18)."""

    return call_label[:-2] if call_label.endswith(":1") else call_label


def node_doc(entry: Any) -> str:
    """Build the one-line, privacy-normalized docString (memo D-18).

    Source locations emit basename+line only, never absolute home or temp
    paths; the innermost recorded frame is the op's own call site.
    """

    contexts = getattr(entry, "code_context", None) or []
    location = ""
    for frame in reversed(list(contexts)):
        file_name = str(getattr(frame, "file", "") or "")
        line = getattr(frame, "line", None)
        if file_name:
            base = file_name.replace("\\", "/").rsplit("/", 1)[-1]
            location = f"{base}:{line}" if line else base
            break
    func = str(getattr(entry, "func_qualname", "") or getattr(entry, "func_name", ""))
    # torch's free functions carry an internal holder-class qualname that
    # reads as noise in the sidebar; the bare function name is the fact.
    if "_VariableFunctionsClass." in func:
        func = func.rsplit(".", 1)[-1]
    if func and location:
        return f"{func} called at {location}"
    return func or location


def port_names(entry: Any, inputs: list[str]) -> list[str] | None:
    """Derive netron ``input_names`` port labels with strict alignment (D-16).

    Names come from ``arg_names`` indexed by ``parent_arg_positions``; on any
    mismatch the hook is omitted -- a numeric port beats a wrong name.
    """

    positions = getattr(entry, "parent_arg_positions", None)
    arg_names = list(getattr(entry, "arg_names", None) or [])
    if not isinstance(positions, dict) or not arg_names or not inputs:
        return None
    label_to_name: dict[str, str] = {}
    for position, parent_label in (positions.get("args") or {}).items():
        if isinstance(position, int) and 0 <= position < len(arg_names):
            label_to_name[str(parent_label)] = str(arg_names[position])
    for kwarg_name, parent_label in (positions.get("kwargs") or {}).items():
        label_to_name[str(parent_label)] = str(kwarg_name)
    names: list[str] = []
    for value_name in inputs:
        base = value_name.rsplit(":", 1)[0]
        name = label_to_name.get(value_name) or label_to_name.get(base)
        if not name or "\n" in name:
            return None
        names.append(name)
    if len(set(names)) != len(names):
        return None
    return names

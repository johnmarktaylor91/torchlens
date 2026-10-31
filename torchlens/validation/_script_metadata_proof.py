"""Graph proof that a TorchScript boundary call reads a tensor argument for metadata only.

A ``jitscript`` boundary op (``backends/torch/_script_boundary.py``) lists every tensor
argument of the scripted call as a parent. When the scripted body reads an argument only
through its shape, rank, element count, dtype or device (a DeBERTa-style
``scaled_size_sqrt(query_layer, scale_factor)`` reads ``query_layer.size(-1)``), the call's
value cannot depend on that argument's elements, so perturbing them cannot change the
output: the edge is outside the perturbation contract (C2), the same class as arg 1 of
``view_as`` in ``STRUCTURAL_ARG_POSITIONS``. Membership is proved per scripted function
from its own TorchScript graph with nested scripted calls inlined (``function.inlined_graph``,
the same program, so a helper that hands an argument to another scripted helper is judged by
what that helper does with it), never from a name:

- only ops whose callable is the TorchScript boundary replay callable;
- only a graph input typed ``Tensor`` with at least one use, every use being input 0 of a
  node whose kind is in :data:`METADATA_READ_NODE_KINDS` (enumerated, no wildcard);
- every call slot the perturbed parent fills must map to such an input, so a parent that
  also feeds a value-bearing argument stays perturbation-tested;
- an unreadable graph, a schema/graph arity mismatch, a container slot, or any other use
  (a value op, a block or graph return, ``prim::PythonOp``, a call left un-inlined, ...)
  gives no proof and the perturbation check runs as before.

A proof is reported as its own exemption reason (:data:`SCRIPT_METADATA_ONLY_REASON`) with
the function, the proven arguments and the node kinds read; it is never a silent skip.
"""

from __future__ import annotations

from typing import Any

from ..backends.torch._script_boundary import script_function_of

SCRIPT_METADATA_ONLY_REASON = "torchscript_metadata_only_argument"
"""Exemption reason recorded when the graph proof holds (spelled as a literal at the
``core.py`` call site, where the exemption ledger's scanner reads it)."""

METADATA_READ_NODE_KINDS: frozenset[str] = frozenset(
    {
        "aten::size",  # x.size(), x.size(d), x.shape
        "aten::dim",  # x.dim(), x.ndim
        "aten::numel",  # x.numel()
        "prim::dtype",  # x.dtype
        "prim::device",  # x.device
    }
)
"""TorchScript node kinds whose output depends only on a tensor's metadata (input 0)."""


def _named_graph_inputs(function: Any) -> list[tuple[str, Any]] | None:
    """Pair the scripted function's schema argument names with its graph inputs.

    Parameters
    ----------
    function:
        A ``torch.jit.ScriptFunction``.

    Returns
    -------
    list[tuple[str, Any]] | None
        ``(name, graph input)`` pairs, or ``None`` when the graph or schema cannot be read
        or their arities differ.
    """

    try:
        names = [argument.name for argument in function.schema.arguments]
        inputs = list(function.inlined_graph.inputs())
    except (AttributeError, RuntimeError, TypeError):
        return None
    if len(names) != len(inputs):
        return None
    return list(zip(names, inputs))


def _metadata_read_kinds(value: Any) -> frozenset[str] | None:
    """Return the node kinds reading ``value`` when every use is a metadata read.

    Parameters
    ----------
    value:
        A TorchScript graph input.

    Returns
    -------
    frozenset[str] | None
        The metadata node kinds, or ``None`` when the input is not a tensor, has no use,
        or has any use that is not input 0 of a :data:`METADATA_READ_NODE_KINDS` node.
    """

    try:
        if value.type().kind() != "TensorType":
            return None
        kinds = set()
        for use in value.uses():
            kind = use.user.kind()
            if kind not in METADATA_READ_NODE_KINDS or use.offset != 0:
                return None
            kinds.add(kind)
    except (AttributeError, RuntimeError, TypeError):
        return None
    return frozenset(kinds) or None


def metadata_only_arguments(function: Any) -> dict[str, frozenset[str]]:
    """Map each argument the graph proves metadata-only to the node kinds reading it.

    Parameters
    ----------
    function:
        A ``torch.jit.ScriptFunction``.

    Returns
    -------
    dict[str, frozenset[str]]
        Proven argument names; empty when nothing is proven or the graph is unreadable.
    """

    pairs = _named_graph_inputs(function)
    if pairs is None:
        return {}
    proven = {}
    for name, value in pairs:
        kinds = _metadata_read_kinds(value)
        if kinds is not None:
            proven[name] = kinds
    return proven


def _perturbed_argument_names(layer: Any, perturbed_label: str, names: list[str]) -> set[str]:
    """Return the schema argument names of every call slot the perturbed parent fills.

    Parameters
    ----------
    layer:
        The boundary op.
    perturbed_label:
        Parent being perturbed.
    names:
        The scripted function's schema argument names, in order.

    Returns
    -------
    set[str]
        Argument names; empty when any slot is a container member or out of range, so
        no proof can apply.
    """

    positions = getattr(layer, "parent_arg_positions", {}) or {}
    slots: set[str] = set()
    for key, label in (positions.get("args", {}) or {}).items():
        if label != perturbed_label:
            continue
        if not isinstance(key, int) or not 0 <= key < len(names):
            return set()
        slots.add(names[key])
    for key, label in (positions.get("kwargs", {}) or {}).items():
        if label != perturbed_label:
            continue
        if not isinstance(key, str) or key not in names:
            return set()
        slots.add(key)
    return slots


def script_metadata_only_proof(layer: Any, layers_to_perturb: list[str]) -> str | None:
    """Return the proof that the perturbed parent reaches the scripted call as metadata only.

    Parameters
    ----------
    layer:
        Op whose parent is being perturbed.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    str | None
        The recorded justification when the proof holds; ``None`` otherwise (the
        perturbation check then runs).
    """

    function = script_function_of(getattr(layer, "func", None))
    if function is None or len(layers_to_perturb) != 1:
        return None
    pairs = _named_graph_inputs(function)
    if pairs is None:
        return None
    slots = _perturbed_argument_names(layer, layers_to_perturb[0], [name for name, _ in pairs])
    proven = metadata_only_arguments(function)
    if not slots or not slots <= proven.keys():
        return None
    kinds = sorted(set().union(*(proven[name] for name in slots)))
    return (
        f"TorchScript function {function.qualified_name} reads argument(s) "
        f"{', '.join(sorted(slots))} only through {', '.join(kinds)} (graph proof), so the "
        "call's value cannot depend on their elements"
    )

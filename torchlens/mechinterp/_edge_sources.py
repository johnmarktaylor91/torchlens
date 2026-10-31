"""Dataflow resolution behind the component edge graph (internal).

Two questions, both answered from the captured graph and never from an
architecture table:

* Where does a receiver read the residual stream? Each receiver module
  (an attention module with the head vocabulary, an MLP, the unembedding)
  reads it through exactly one normalization; the receiver's input is the
  tensor ENTERING that norm (the pre-norm residual, which is additive).
* Which writers make up that tensor? The spine walk (``_walk``) from the
  pre-norm input lists the operands that wrote into the stream; an operand
  that is itself a full-width two-tensor addition (the parallel-block
  ``mlp_out + attn_out`` merge on GPT-NeoX) is expanded into its operands.

Each leaf writer is then classified by its module chain: an embedding
(the ``input`` node), an attention module with the head vocabulary (its
heads plus its verified output-bias constant), or an MLP. Anything else
means the stream is not additive under this decomposition, and the graph
refuses typed rather than fold an unknown term into a neighbour.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from .._semantic.facets import AbsenceReason
from .._semantic.recipes.lm_head import _external_parents, _norm_input_spec
from ._errors import refuse
from ._residual import _deepest_value_ancestor, _module_stack
from ._walk import (
    _is_qualifying_add,
    _normalize,
    _recorded_shape,
    _tensor_parent_labels,
    walk_spine,
)

__all__ = ["LeafWriter", "classify_leaf", "pre_norm_input", "writer_leaves"]

#: Class-name fragments that mark a normalization module (LayerNorm, RMSNorm,
#: GroupNorm, ...); matched case-insensitively on the module record's class.
_NORM_MARKERS = ("norm",)

#: Class-name fragments that mark an MLP block (the kit's writer vocabulary).
_MLP_MARKERS = ("mlp", "feedforward", "ffn")


@dataclass(frozen=True)
class LeafWriter:
    """One literal writer into the residual stream, classified.

    Parameters
    ----------
    op_label:
        Pass-qualified label of the operand op that entered the addition.
    kind:
        ``"embedding"``, ``"attention"`` or ``"mlp"``.
    module_address:
        The owning embedding, attention or MLP module address.
    """

    op_label: str
    kind: str
    module_address: str


def _class_name(trace: Any, address: str) -> str:
    """Return a module record's lower-cased class name, or ``""``."""

    try:
        return str(getattr(trace.modules[address], "class_name", "")).lower()
    except (KeyError, ValueError):
        return ""


def _is_norm_module(trace: Any, address: str | None) -> bool:
    """Return whether a module address names a normalization module."""

    if address is None:
        return False
    name = _class_name(trace, address)
    return any(marker in name for marker in _NORM_MARKERS)


def pre_norm_input(trace: Any, module: Any, *, receiver: str) -> tuple[Any, Any]:
    """Return ``(norm_record, input_op)``: the pre-norm residual a module reads.

    The receiver module's external parents (ops outside it that its interior
    consumes) must include exactly one norm output; the input is the op the
    norm itself consumes (the lm-head recipe's pass-exact resolution).
    Masks and rotary tables are other external parents and are ignored.
    """

    externals = _external_parents(trace, module)
    address = str(getattr(module, "address", ""))
    if isinstance(externals, AbsenceReason):
        refuse(
            code="mi_edge_graph_nonadditive",
            message=f"The input of receiver {receiver!r} ({address}) could not be resolved: "
            f"{externals.detail}.",
            remedy="capture with default options (layers_to_save='all') so every op inside "
            "the receiver module is recorded",
            receiver=receiver,
            module_address=address,
        )
    norms = []
    for op in externals.values():
        stack = _module_stack(op)
        if stack and _is_norm_module(trace, stack[-1]):
            norms.append(stack[-1])
    distinct = sorted(set(norms))
    if len(distinct) != 1:
        refuse(
            code="mi_edge_graph_nonadditive",
            message=f"Receiver {receiver!r} ({address}) reads {len(distinct)} normalized "
            f"inputs ({distinct}); the component graph needs each receiver to read the "
            "residual stream through exactly one pre-norm, so the stream is not additive "
            "under this decomposition (post-norm blocks read an already renormalized state).",
            remedy="use a pre-norm additive-residual model (GPT-2, GPT-NeoX/Pythia, "
            "Llama-class); for other models decompose one state with "
            "torchlens.mechinterp.residual_decomposition(trace, target=...)",
            receiver=receiver,
            module_address=address,
            norms=distinct,
        )
    norm = trace.modules[distinct[0]]
    spec = _norm_input_spec(norm)
    if isinstance(spec, AbsenceReason):
        refuse(
            code="mi_edge_graph_nonadditive",
            message=f"The input entering norm {distinct[0]!r} of receiver {receiver!r} could "
            f"not be resolved: {spec.detail}.",
            remedy="capture with default options (layers_to_save='all')",
            receiver=receiver,
            module_address=address,
        )
    return norm, spec.home


def _expand(trace: Any, label: str, hidden_shape: tuple[int, ...]) -> list[str]:
    """Expand a writer that is itself a full-width addition into its operands."""

    op = trace.ops[label]
    normalized = _normalize(trace, op, hidden_shape, {}, structure_only=False)
    if not _is_qualifying_add(normalized, hidden_shape):
        return [str(getattr(op, "label", label))]
    leaves: list[str] = []
    for parent in _tensor_parent_labels(normalized):
        leaves.extend(_expand(trace, parent, hidden_shape))
    return leaves


def writer_leaves(trace: Any, input_op: Any) -> list[str]:
    """Return the literal writer op labels whose sum is ``input_op``'s value.

    A pre-norm input that is not an addition (the first GPT-NeoX block reads
    the embedding directly) is a single writer.
    """

    hidden_shape = _recorded_shape(input_op)
    if hidden_shape is None:
        refuse(
            code="mi_target_unresolvable",
            message=f"Receiver input {getattr(input_op, 'label', '?')!r} has no recorded shape.",
            remedy="capture with default options and retry",
        )
    start = _normalize(trace, input_op, hidden_shape, {}, structure_only=False)
    if not _is_qualifying_add(start, hidden_shape):
        return _expand(trace, str(getattr(input_op, "label", "")), hidden_shape)
    walk = walk_spine(trace, input_op)
    leaves: list[str] = []
    for node in walk.nodes:
        for label in node.writer_labels:
            leaves.extend(_expand(trace, label, hidden_shape))
    return leaves


def _refuse_nonadditive_writer(label: str, where: str, func_name: str) -> NoReturn:
    """Refuse a residual writer that is not a graph component (typed)."""

    refuse(
        code="mi_edge_graph_nonadditive",
        message=f"Residual writer {label!r} (inside {where}, func {func_name!r}) is not an "
        "embedding, an attention head or an MLP, so the residual stream is not additive "
        "under the component-graph decomposition (a renormalized or rescaled stream, or an "
        "unknown writer).",
        remedy="use a pre-norm additive-residual model (GPT-2, GPT-NeoX/Pythia, "
        "Llama-class); for other models decompose one state with "
        "torchlens.mechinterp.residual_decomposition(trace, target=...)",
        writer_label=label,
        module_address=where,
    )


def classify_leaf(trace: Any, label: str, attention_addresses: frozenset[str]) -> LeafWriter:
    """Classify one leaf writer by its deepest value-identical ancestor's modules.

    The innermost classifiable module wins. A norm output writing into the
    stream renormalizes it, so it refuses before any enclosing block's class
    name can claim the write.
    """

    op = trace.ops[label]
    ancestor, _verification = _deepest_value_ancestor(trace, op)
    stack = _module_stack(ancestor)
    func_name = str(getattr(ancestor, "func_name", "?"))
    if stack and _is_norm_module(trace, stack[-1]):
        _refuse_nonadditive_writer(label, stack[-1], func_name)
    for address in reversed(stack):
        if address in attention_addresses:
            return LeafWriter(op_label=label, kind="attention", module_address=address)
        name = _class_name(trace, address)
        if "embedding" in name:
            return LeafWriter(op_label=label, kind="embedding", module_address=address)
        if any(marker in name for marker in _MLP_MARKERS):
            return LeafWriter(op_label=label, kind="mlp", module_address=address)
        if "attention" in name or "attn" in name:
            refuse(
                code="mi_head_geometry_unavailable",
                message=f"Attention module {address!r} writes into the residual stream but "
                "exposes no per-head result view, so its heads cannot be graph nodes.",
                remedy="load the model with attn_implementation='eager' (or capture fused SDPA "
                "with capture=tl.options.CaptureOptions(save_args=True)) and capture "
                "with layers_to_save='all'",
                module_address=address,
            )
    _refuse_nonadditive_writer(label, stack[-1] if stack else "<top level>", func_name)

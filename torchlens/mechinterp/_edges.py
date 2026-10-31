"""The component edge graph: heads and MLPs as nodes, q/k/v split receivers.

The circuit-discovery vocabulary of the MIB circuit track and EAP-IG, read
off the captured graph. Nodes are the residual writers: ``input`` (the
embedding sum), ``a{L}.h{H}`` (one head's output, without the attention
output bias), ``b{L}`` (layer L's attention output bias, its own node so
the decomposition never drops it) and ``m{L}`` (one MLP's output, bias
included). Receivers are the residual readers: ``a{L}.h{H}<q|k|v>``,
``m{L}`` and ``logits``. An edge ``sender->receiver`` exists when the sender
wrote into the residual state the receiver reads.

Exactness: a receiver's input is the tensor ENTERING its pre-norm (never the
norm's output; the norm is not linearised), and on an additive pre-norm
residual stream that tensor is exactly the sum of its incoming edges'
contributions. Each edge's contribution to its receiver's input is the
sender's output, per batch row and position. The builder checks that
identity for every receiver before it returns (identities run IN the API)
and refuses typed when the stream is not additive under this decomposition.

Which receivers read which writers comes from the dataflow, never from an
architecture table: on a parallel-block model (GPT-NeoX/Pythia) MLP L reads
the same state as attention L, so it receives no edge from layer L's heads.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import torch

from .._semantic.tolerances import within_reconstruction_tolerance
from ._anchors import resolve_lm_head
from ._edge_sources import LeafWriter, classify_leaf, pre_norm_input, writer_leaves
from ._errors import refuse
from ._heads import _attention_modules, attention_head_contributions
from ._residual import _require_payloads

__all__ = [
    "ComponentEdge",
    "ComponentGraph",
    "ComponentSender",
    "ComponentReceiver",
    "component_graph",
]

NodeKind = Literal["input", "head", "bias", "mlp"]
ReceiverKind = Literal["q", "k", "v", "mlp", "logits"]

#: The three per-head attention receivers, in vocabulary order.
_QKV: tuple[ReceiverKind, ...] = ("q", "k", "v")


@dataclass(frozen=True, eq=False)
class ComponentSender:
    """One residual writer (a graph source node).

    Parameters
    ----------
    name:
        Vocabulary name: ``input``, ``a{L}.h{H}``, ``b{L}`` or ``m{L}``.
    kind:
        ``"input"``, ``"head"``, ``"bias"`` or ``"mlp"``.
    layer_index:
        Transformer block index (integer); ``None`` where it does not apply.
    head:
        Head index (integer); ``None`` where it does not apply.
    module_address:
        The owning module (embedding, attention or MLP; the attention module
        for head and bias nodes).
    value:
        The node's output ``[batch, pos, d_model]``: what it adds to the
        residual stream, and so its contribution on every outgoing edge.
    """

    name: str
    kind: NodeKind
    layer_index: int | None
    head: int | None
    module_address: str | None
    value: torch.Tensor

    def __repr__(self) -> str:
        return f"ComponentNode({self.name!r}, kind={self.kind!r}, module={self.module_address!r})"


@dataclass(frozen=True, eq=False)
class ComponentReceiver:
    """One residual reader (a graph target).

    Parameters
    ----------
    name:
        Vocabulary name: ``a{L}.h{H}<q|k|v>``, ``m{L}`` or ``logits``.
    kind:
        ``"q"``, ``"k"``, ``"v"``, ``"mlp"`` or ``"logits"``.
    layer_index:
        Transformer block index (integer); ``None`` where it does not apply.
    head:
        Head index (integer); ``None`` where it does not apply.
    module_address:
        The reading module (attention, MLP, or the unembedding head).
    norm_address:
        The pre-norm the receiver reads the residual stream through.
    input_label:
        Pass-qualified label of the op whose output enters that norm.
    value:
        The captured receiver input ``[batch, pos, d_model]`` (pre-norm).
    senders:
        The source nodes whose outputs sum to ``value``, in node order.
    """

    name: str
    kind: ReceiverKind
    layer_index: int | None
    head: int | None
    module_address: str
    norm_address: str
    input_label: str
    value: torch.Tensor
    senders: tuple[ComponentSender, ...]
    _trace: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        return (
            f"ComponentReceiver({self.name!r}, kind={self.kind!r}, n_senders={len(self.senders)})"
        )


@dataclass(frozen=True, eq=False)
class ComponentEdge:
    """One edge ``sender->receiver`` of the component graph."""

    sender: ComponentSender
    receiver: ComponentReceiver

    @property
    def name(self) -> str:
        """Vocabulary name ``"<sender>-><receiver>"``."""

        return f"{self.sender.name}->{self.receiver.name}"

    @property
    def contribution(self) -> torch.Tensor:
        """The edge's contribution to the receiver's input, ``[batch, pos, d_model]``."""

        return self.sender.value

    def __selection__(self) -> Any:
        """Return the selection a patch of this edge writes (intervention doors).

        ``m{L}`` and ``logits`` receivers: the EDGE selection of the
        receiver's pre-norm input edge. ``a{L}.h{H}<q|k|v>`` receivers: the
        ACT selection of head ``H``'s role slice of the projection output.
        :func:`patch_edges` builds the matching value.
        """

        from ._edge_patch import receiver_selection

        return receiver_selection(self.receiver._trace, self.receiver)

    def __repr__(self) -> str:
        return f"ComponentEdge({self.name!r})"


class ComponentGraph:
    """The component edge graph of one trace (built by :func:`component_graph`).

    Read-only views keyed by vocabulary name, each in graph order:
    ``nodes`` (sources), ``receivers`` (targets) and ``edges``. Bias nodes
    ``b{L}`` and their edges are part of the graph; the MIB vocabulary count
    excludes them (``sum(e.sender.kind != "bias" for e in graph.edges.values())``).
    ``identity_receipt`` records the in-API check that every receiver's
    input equals the sum of its incoming contributions.
    """

    def __init__(
        self,
        nodes: tuple[ComponentSender, ...],
        receivers: tuple[ComponentReceiver, ...],
        identity_receipt: dict[str, Any],
    ) -> None:
        self.nodes: Mapping[str, ComponentSender] = MappingProxyType({n.name: n for n in nodes})
        self.receivers: Mapping[str, ComponentReceiver] = MappingProxyType(
            {r.name: r for r in receivers}
        )
        edges = (ComponentEdge(sender, r) for r in receivers for sender in r.senders)
        self.edges: Mapping[str, ComponentEdge] = MappingProxyType({e.name: e for e in edges})
        self.identity_receipt = identity_receipt
        # ``(op label, module address)`` of the embedding writes that sum to the
        # ``input`` node (EAP-IG's interpolation site); set by :func:`component_graph`.
        self._input_writes: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        return (
            f"ComponentGraph(n_nodes={len(self.nodes)}, n_receivers={len(self.receivers)}, "
            f"n_edges={len(self.edges)})"
        )


@dataclass
class _Layers:
    """Execution-ordered attention and MLP module addresses."""

    attention: list[str]
    mlp: list[str]


def _layer_order(leaves: list[LeafWriter]) -> _Layers:
    """Order attention and MLP modules by first appearance among the writers."""

    attention: list[str] = []
    mlp: list[str] = []
    for leaf in leaves:
        bucket = attention if leaf.kind == "attention" else mlp if leaf.kind == "mlp" else None
        if bucket is not None and leaf.module_address not in bucket:
            bucket.append(leaf.module_address)
    return _Layers(attention=attention, mlp=mlp)


def _head_nodes(
    trace: Any, address: str, layer: int, writer_values: list[torch.Tensor]
) -> list[ComponentSender]:
    """Return one attention layer's head nodes plus its verified bias node.

    The attention writer must be payload-identical to the layer's ``attn_out``
    (otherwise a post-projection scale or dropout sits between the heads and
    the stream), and the remainder ``attn_out - sum(heads)`` must be proven
    to be the output bias or zero.
    """

    stack = attention_head_contributions(trace, layers=address)
    attn_out = stack.target_value
    for value in writer_values:
        if attn_out is None or not torch.equal(value, attn_out):
            refuse(
                code="mi_edge_graph_nonadditive",
                message=f"The residual write of attention {address!r} is not its captured "
                "attn_out (a post-projection scale, norm or dropout sits between the heads "
                "and the stream), so per-head contributions do not add up to the write.",
                remedy="trace in eval mode; models that rescale or renormalize attention "
                "output before the residual add are outside the component graph",
                module_address=address,
            )
    nodes: list[ComponentSender] = []
    head = 0
    for row in stack.rows:
        kind = row.coordinate.kind
        if kind == "attention":
            nodes.append(
                ComponentSender(f"a{layer}.h{head}", "head", layer, head, address, row.value)
            )
            head += 1
        elif kind == "constant":
            nodes.append(ComponentSender(f"b{layer}", "bias", layer, None, address, row.value))
        else:
            refuse(
                code="mi_edge_graph_nonadditive",
                message=f"Attention {address!r} leaves an unproven remainder "
                f"({row.coordinate.provenance}); it is neither the output bias nor zero.",
                remedy="models whose attention output is not heads plus a constant bias are "
                "outside the component graph",
                module_address=address,
            )
    return nodes


def _require_mha(trace: Any, address: str) -> int:
    """Return the head count, refusing grouped-query attention (named default)."""

    facets = trace.modules[address].facets
    n_q, n_kv = int(facets["n_q_heads"]), int(facets["n_kv_heads"])
    if n_q != n_kv:
        refuse(
            code="mi_edge_graph_gqa_unsupported",
            message=f"Attention {address!r} uses grouped-query attention ({n_q} query heads, "
            f"{n_kv} key/value heads); the component graph's k and v receiver vocabulary "
            "for shared key/value heads is not defined yet.",
            remedy="use torchlens.mechinterp.attention_head_contributions(trace) for per-head "
            "writer rows on this model",
            module_address=address,
            n_q_heads=n_q,
            n_kv_heads=n_kv,
        )
    return n_q


def _identity(receivers: list[ComponentReceiver]) -> dict[str, Any]:
    """Check every receiver input against its contributions' sum (in-API gate)."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for receiver in receivers:
        if receiver.input_label in seen:
            continue
        seen.add(receiver.input_label)
        total = torch.zeros_like(receiver.value)
        magnitude = torch.zeros_like(receiver.value)
        for sender in receiver.senders:
            total = total + sender.value
            magnitude = magnitude + sender.value.detach().abs()
        residual = float((total - receiver.value).detach().abs().max())
        row = {
            "input_label": receiver.input_label,
            "first_receiver": receiver.name,
            "n_senders": len(receiver.senders),
            "max_abs_residual": residual,
            "max_abs_input": float(receiver.value.detach().abs().max()),
        }
        rows.append(row)
        if not within_reconstruction_tolerance(
            total,
            receiver.value,
            magnitude=magnitude + receiver.value.detach().abs(),
            reduction_length=len(receiver.senders),
        ):
            refuse(
                code="mi_edge_graph_open",
                message=f"The input of receiver {receiver.name!r} ({receiver.input_label}) "
                f"differs from the sum of its {len(receiver.senders)} incoming contributions "
                f"by up to {residual:.3e}; the decomposition does not close.",
                remedy="report the architecture: a writer the graph does not model reaches "
                "this receiver",
                receiver=receiver.name,
                max_abs_residual=residual,
            )
    return {
        "check": "receiver_input_equals_sum_of_contributions",
        "result": "within_reconstruction_tolerance",
        "per_input": rows,
    }


class _Builder:
    """One graph build: the writer classification and the node table it implies."""

    def __init__(self, trace: Any) -> None:
        self.trace = trace
        self.anchor = resolve_lm_head(trace)
        self.attention_addresses = frozenset(str(m.address) for m in _attention_modules(trace))
        labels = writer_leaves(trace, self.anchor.target_op)
        self.final_leaves = [self.classify(label) for label in labels]
        self.layers = _layer_order(self.final_leaves)
        self.payloads = _require_payloads(trace, labels, analysis="component_graph")
        self.input_node = self._input_node()
        self.nodes_of = self._component_nodes()
        self.order = [self.input_node]
        for layer in range(self.n_layers):
            for bucket in (self.layers.attention, self.layers.mlp):
                if layer < len(bucket):
                    self.order.extend(self.nodes_of[bucket[layer]])
        self.rank = {node.name: index for index, node in enumerate(self.order)}

    @property
    def n_layers(self) -> int:
        """Number of blocks (the longer of the attention and MLP module lists)."""

        return max(len(self.layers.attention), len(self.layers.mlp))

    def classify(self, label: str) -> LeafWriter:
        """Classify one leaf writer against this trace's attention modules."""

        return classify_leaf(self.trace, label, self.attention_addresses)

    def _writes_of(self, address: str) -> list[torch.Tensor]:
        """Return the payloads one module wrote into the final residual state."""

        return [self.payloads[f.op_label] for f in self.final_leaves if f.module_address == address]

    def _input_node(self) -> ComponentSender:
        """The ``input`` node: the sum of the embedding writes, in write order."""

        embedding = [self.payloads[f.op_label] for f in self.final_leaves if f.kind == "embedding"]
        if not embedding:
            refuse(
                code="mi_edge_graph_nonadditive",
                message="No embedding writes into the residual stream, so there is no input node.",
                remedy="trace a language model from token ids",
            )
        value = embedding[0]
        for addend in embedding[1:]:
            value = value + addend
        return ComponentSender("input", "input", None, None, None, value)

    def _component_nodes(self) -> dict[str, list[ComponentSender]]:
        """Map each attention and MLP module address to its graph nodes."""

        nodes_of: dict[str, list[ComponentSender]] = {}
        for layer, address in enumerate(self.layers.attention):
            _require_mha(self.trace, address)  # refuse before reading any head rows
            nodes_of[address] = _head_nodes(self.trace, address, layer, self._writes_of(address))
        for layer, address in enumerate(self.layers.mlp):
            writes = self._writes_of(address)
            if len(writes) != 1:
                refuse(
                    code="mi_edge_graph_nonadditive",
                    message=f"MLP {address!r} writes into the residual stream {len(writes)} times.",
                    remedy="models whose MLP writes more than once per forward are outside "
                    "the component graph",
                    module_address=address,
                )
            nodes_of[address] = [
                ComponentSender(f"m{layer}", "mlp", layer, None, address, writes[0])
            ]
        return nodes_of

    def senders_of(self, input_op: Any) -> tuple[ComponentSender, ...]:
        """Return the nodes whose outputs sum to ``input_op``'s value, in node order."""

        found: dict[str, ComponentSender] = {}
        for label in writer_leaves(self.trace, input_op):
            leaf = self.classify(label)
            if leaf.kind == "embedding":
                group: list[ComponentSender] | None = [self.input_node]
            else:
                group = self.nodes_of.get(leaf.module_address)
            if group is None:
                refuse(
                    code="mi_edge_graph_nonadditive",
                    message=f"Writer {label!r} ({leaf.module_address}) reaches a receiver but "
                    "not the final residual state.",
                    remedy="report the architecture (a side stream the graph does not model)",
                    writer_label=label,
                )
            found.update((node.name, node) for node in group)
        return tuple(sorted(found.values(), key=lambda node: self.rank[node.name]))

    def receivers_at(self, module: Any, name: str) -> tuple[str, str, torch.Tensor, Any]:
        """Return ``(norm_address, input_label, input_value, senders)`` for one module."""

        norm, input_op = pre_norm_input(self.trace, module, receiver=name)
        label = str(getattr(input_op, "label", ""))
        value = _require_payloads(self.trace, [label], analysis="component_graph")[label]
        return str(norm.address), label, value, self.senders_of(input_op)

    def attention_receivers(self, layer: int) -> list[ComponentReceiver]:
        """The ``a{L}.h{H}<q|k|v>`` receivers of one attention layer."""

        address = self.layers.attention[layer]
        n_heads = _require_mha(self.trace, address)
        module = self.trace.modules[address]
        norm_address, label, value, senders = self.receivers_at(module, f"a{layer}.h0<q>")
        return [
            ComponentReceiver(
                f"a{layer}.h{head}<{role}>",
                role,
                layer,
                head,
                address,
                norm_address,
                label,
                value,
                senders,
                _trace=self.trace,
            )
            for head in range(n_heads)
            for role in _QKV
        ]

    def mlp_receiver(self, layer: int) -> ComponentReceiver:
        """The ``m{L}`` receiver."""

        address = self.layers.mlp[layer]
        norm_address, label, value, senders = self.receivers_at(
            self.trace.modules[address], f"m{layer}"
        )
        return ComponentReceiver(
            f"m{layer}",
            "mlp",
            layer,
            None,
            address,
            norm_address,
            label,
            value,
            senders,
            _trace=self.trace,
        )

    def logits_receiver(self) -> ComponentReceiver:
        """The ``logits`` receiver: the residual state entering the final norm."""

        target = self.anchor.target_op
        label = str(getattr(target, "label", ""))
        value = _require_payloads(self.trace, [label], analysis="component_graph")[label]
        return ComponentReceiver(
            "logits",
            "logits",
            None,
            None,
            str(getattr(self.anchor.head, "address", "")),
            str(getattr(self.anchor.norm, "address", "")),
            label,
            value,
            self.senders_of(target),
            _trace=self.trace,
        )


def component_graph(trace: Any) -> ComponentGraph:
    """Build the component edge graph (heads and MLPs, q/k/v split) of a trace.

    Parameters
    ----------
    trace:
        A finished trace of a pre-norm additive-residual language model,
        captured with every op saved (``layers_to_save='all'``) and the
        builtin attention recipes' head views (eager attention).

    Returns
    -------
    ComponentGraph
        Nodes, receivers and edges keyed by vocabulary name, with the
        receiver-input identity already checked.
    """

    builder = _Builder(trace)
    receivers: list[ComponentReceiver] = []
    for layer in range(builder.n_layers):
        if layer < len(builder.layers.attention):
            receivers.extend(builder.attention_receivers(layer))
        if layer < len(builder.layers.mlp):
            receivers.append(builder.mlp_receiver(layer))
    receivers.append(builder.logits_receiver())
    graph = ComponentGraph(tuple(builder.order), tuple(receivers), _identity(receivers))
    graph._input_writes = tuple(
        (leaf.op_label, leaf.module_address)
        for leaf in builder.final_leaves
        if leaf.kind == "embedding"
    )
    return graph

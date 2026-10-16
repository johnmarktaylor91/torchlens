"""``build_graph``: an attribution graph from one capture, one backward read.

Every target node (each active feature's pre-activation and each target
logit) seeds one cotangent of a single one-backward read; the read returns
the gradient of every target at every MLP output a transcoder writes and at
the embedding output, under a frozen set that holds those MLP outputs and
the caller's frozen sites (attention patterns, normalisation scales) at
their captured values. An edge is that gradient contracted with the source
node's vector: a feature's activation times its decoder row, an error
node's unexplained MLP output, or a position's embedding output.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .._result import AttributionError
from .._site import SiteTarget
from ..onebackward import read, seed
from ..onebackward._accessor import ReadEdgeIndex, read_edge_index
from ..onebackward._read import _capture_outcome_status, _SitePayloads
from ..onebackward._targets import resolve_site_edge
from ._result import AttributionGraph, GraphNode
from ._transcoders import Reader, normalize_transcoders

__all__ = ["build_graph"]

_DOCS_POINTER = "See docs/reference/attribution.md, section Attribution graphs."


class _Unset:
    """Sentinel type for an omitted ``frozen=``."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _graph_refusal(message: str, *, code: str) -> AttributionError:
    """Build one coded attribution-graph refusal ending in its docs pointer."""

    return AttributionError(f"{message} {_DOCS_POINTER}", code=code)


@dataclass(frozen=True)
class _Sites:
    """The capture's addressed sites for one build."""

    index: ReadEdgeIndex
    payloads: _SitePayloads
    reads: dict[str, str]
    writes: dict[str, str]
    embedding: str
    tokens: torch.Tensor | None
    logits: str
    embedding_name: str
    logits_name: str


def _module(trace: Any, name: str, role: str) -> Any:
    """Return the module log ``name`` or refuse naming its role."""

    try:
        return trace.modules[name]
    except Exception as exc:  # noqa: BLE001 - any lookup failure is the same refusal
        raise _graph_refusal(
            f"The {role} module {name!r} is not a module of this capture. "
            "Remedy: name a module from dict(model.named_modules()) that ran once in "
            "the captured forward.",
            code="attribution_graph_site_not_found",
        ) from exc


def _one_op(module: Any, name: str, field: str, role: str) -> str:
    """Return the single op label of ``module.<field>`` or refuse."""

    ops = list(getattr(module, field, None) or ())
    if len(ops) != 1:
        raise _graph_refusal(
            f"The {role} module {name!r} has {len(ops)} {field.replace('_', ' ')} in this "
            "capture; the graph needs exactly one. Remedy: name a module that runs once "
            "and takes one tensor input (for an MLP, the module the transcoder replaces).",
            code="attribution_graph_site_not_found",
        )
    return str(ops[0])


def _edge_label(index: ReadEdgeIndex, label: str, role: str) -> str:
    """Return the pass-qualified edge label of an op label."""

    return resolve_site_edge(index, label, role=role).label


def _resolve_sites(
    trace: Any,
    readers: tuple[Reader, ...],
    embedding: str,
    logits_at: SiteTarget | None,
) -> _Sites:
    """Address every read, write, embedding and logit site on the capture."""

    index = read_edge_index(trace)
    reads = {
        reader.site: _edge_label(
            index,
            _one_op(_module(trace, reader.site, "transcoder"), reader.site, "input_ops", "MLP"),
            "target",
        )
        for reader in readers
    }
    writes: dict[str, str] = {}
    for reader in readers:
        for write, _decoder in reader.decoders:
            if write not in writes:
                module = _module(trace, write, "MLP output")
                op = _one_op(module, write, "output_ops", "MLP output")
                writes[write] = _edge_label(index, op, "input")
    embed_module = _module(trace, embedding, "embedding")
    embed_label = _edge_label(
        index, _one_op(embed_module, embedding, "output_ops", "embedding"), "input"
    )
    token_op = list(getattr(embed_module, "input_ops", None) or ())
    tokens = None
    if len(token_op) == 1:
        value = trace[token_op[0]].out
        if isinstance(value, torch.Tensor) and not value.is_floating_point():
            tokens = value.detach().reshape(-1)
    if logits_at is None:
        logits_label = _edge_label(index, str(trace.output_layers[0]), "target")
    else:
        if logits_at.unit is not None or logits_at.call is not None:
            raise _graph_refusal(
                f"logits_at={logits_at!r} selects a unit or a call; the graph reads the "
                "whole logit vector of a module that runs once. Remedy: pass "
                f"logits_at=tl.attribution.SiteTarget({logits_at.layer!r}).",
                code="attribution_graph_target_invalid",
            )
        module = _module(trace, logits_at.layer, "logits")
        op = _one_op(module, logits_at.layer, "output_ops", "logits")
        logits_label = _edge_label(index, op, "target")
    return _Sites(
        index,
        _SitePayloads(trace),
        reads,
        writes,
        embed_label,
        tokens,
        logits_label,
        embedding,
        "output" if logits_at is None else logits_at.layer,
    )


def _payload(sites: _Sites, label: str, role: str) -> torch.Tensor:
    """Return a site's captured value as ``(positions, width)``."""

    value = sites.payloads.payload(label)
    if value is None or value.dim() != 3 or value.shape[0] != 1:
        shape = None if value is None else tuple(value.shape)
        raise _graph_refusal(
            f"The {role} site {label!r} holds {shape}; the graph reads one prompt, "
            "shaped (1, positions, width), with every activation saved. Remedy: "
            "capture one sequence with the default save mode, for example "
            "tl.trace(model, tokens[None]).",
            code="attribution_graph_input_unsupported",
        )
    return value.detach()[0]


@dataclass(frozen=True)
class _Features:
    """The active features of every transcoder."""

    nodes: list[GraphNode]
    reader_index: list[int]
    positions: list[int]
    features: list[int]
    activations: list[torch.Tensor]
    pre_activations: list[torch.Tensor]


def _active_features(readers: tuple[Reader, ...], sites: _Sites) -> _Features:
    """Encode each transcoder's captured input; keep the nonzero activations."""

    found = _Features([], [], [], [], [], [])
    for number, reader in enumerate(readers):
        inputs = _payload(sites, sites.reads[reader.site], "MLP input")
        with torch.no_grad():
            pre = inputs @ reader.encoder.to(inputs.dtype) + reader.encoder_bias.to(inputs.dtype)
            acts = reader.transcoder.encode(inputs)
        if not isinstance(acts, torch.Tensor) or acts.shape != pre.shape:
            raise _graph_refusal(
                f"encode() of the transcoder for {reader.site!r} returned "
                f"{getattr(acts, 'shape', type(acts).__name__)}; expected {tuple(pre.shape)}. "
                "Remedy: make encode map (positions, d_in) to (positions, n_features).",
                code="attribution_graph_transcoder_invalid",
            )
        for position, feature in torch.nonzero(acts.detach(), as_tuple=False).tolist():
            found.nodes.append(GraphNode("feature", reader.site, position, feature=feature))
            found.reader_index.append(number)
            found.positions.append(position)
            found.features.append(feature)
            found.activations.append(acts[position, feature].detach().to(inputs.dtype))
            found.pre_activations.append(pre[position, feature])
    return found


def _salient(
    probabilities: torch.Tensor, max_n_logits: int, desired_logit_prob: float
) -> torch.Tensor:
    """Return the smallest top set reaching ``desired_logit_prob``, capped."""

    top_p, top_idx = torch.topk(probabilities, min(max_n_logits, probabilities.numel()))
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    return top_idx[:cutoff]


def _logit_targets(
    logits: torch.Tensor,
    target: Sequence[int] | None,
    max_n_logits: int,
    desired_logit_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the target ids and their probabilities at the last position."""

    probabilities = torch.softmax(logits[-1], dim=-1)
    if target is None:
        if max_n_logits < 1 or not 0.0 < desired_logit_prob <= 1.0:
            raise _graph_refusal(
                f"max_n_logits={max_n_logits!r} and desired_logit_prob="
                f"{desired_logit_prob!r} select no logits. Remedy: pass max_n_logits=10, "
                "desired_logit_prob=0.95 (the defaults).",
                code="attribution_graph_target_invalid",
            )
        ids = _salient(probabilities, max_n_logits, desired_logit_prob)
    else:
        given = list(target)
        vocab = probabilities.numel()
        if not given or not all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < vocab for i in given
        ):
            raise _graph_refusal(
                f"target={target!r} is not a non-empty list of token ids below {vocab}. "
                "Remedy: pass target=None for the salient logits, or target=[token_id, ...].",
                code="attribution_graph_target_invalid",
            )
        ids = torch.tensor(given, dtype=torch.long)
    return ids, probabilities[ids]


def _cotangent(
    shape: tuple[int, ...], dtype: torch.dtype, position: int, vector: torch.Tensor
) -> Any:
    """Return a ``(1, positions, width)`` cotangent nonzero at one position."""

    cot = torch.zeros(shape, dtype=dtype)
    cot[0, position] = vector.to(dtype)
    return cot


def _seeds(
    readers: tuple[Reader, ...],
    sites: _Sites,
    found: _Features,
    logit_ids: torch.Tensor,
    demeaned_logits: bool,
) -> tuple[list[Any], torch.Tensor]:
    """Return one seed per target node and the logit nodes' values."""

    seeds: list[Any] = []
    for number, position, feature in zip(
        found.reader_index, found.positions, found.features, strict=True
    ):
        reader = readers[number]
        label = sites.reads[reader.site]
        inputs = _payload(sites, label, "MLP input")
        cot = _cotangent((1, *inputs.shape), inputs.dtype, position, reader.encoder[:, feature])
        seeds.append(seed(label, cotangent=cot))
    logits = _payload(sites, sites.logits, "logits")
    last = logits.shape[0] - 1
    vocab = logits.shape[1]
    values = []
    for token in logit_ids.tolist():
        direction = torch.zeros(vocab, dtype=logits.dtype)
        direction[token] = 1.0
        if demeaned_logits:
            direction = direction - 1.0 / vocab
        cot = _cotangent((1, *logits.shape), logits.dtype, last, direction)
        seeds.append(seed(sites.logits, cotangent=cot))
        values.append((logits[last] * direction).sum())
    return seeds, torch.stack(values) if values else torch.zeros(0, dtype=logits.dtype)


def _within(labels: list[str]) -> Any:
    """Return the union selection of ``labels`` for ``read(within=...)``."""

    from ...selection_values import _lift_within

    return functools.reduce(operator.or_, (_lift_within(label, "read") for label in labels))


def _frozen_label(trace: Any, index: ReadEdgeIndex, name: str) -> str:
    """Return the edge label of a frozen module's output, or of an op label."""

    try:
        trace.modules[name]
    except Exception:  # noqa: BLE001 - not a module name: try it as an op label
        if name in index.edges or f"{name}:1" in index.edges:
            return _edge_label(index, name, "frozen")
    module = _module(trace, name, "frozen")
    return _edge_label(index, _one_op(module, name, "output_ops", "frozen"), "frozen")


def _frozen_labels(trace: Any, sites: _Sites, frozen: Sequence[str]) -> list[str]:
    """Return the frozen set: every written MLP output plus the caller's sites."""

    labels = list(sites.writes.values())
    for name in frozen:
        if not isinstance(name, str):
            raise _graph_refusal(
                f"frozen= holds {name!r}; it takes module names. Remedy: pass "
                "frozen=['blocks.0.attn.pattern', ...], the modules whose outputs are held "
                "at their captured values.",
                code="attribution_graph_site_not_found",
            )
        label = _frozen_label(trace, sites.index, name)
        if label not in labels:
            labels.append(label)
    return labels


#: Targets per one-backward read; each read returns every target's gradient
#: at every source site, so the chunk bounds the read's result size.
_TARGETS_PER_READ = 256


def _gradients(
    trace: Any,
    sites: _Sites,
    seeds: list[Any],
    frozen: list[str],
) -> torch.Tensor:
    """Run the reads; return ``(n_targets, n_sources, positions, width)`` gradients.

    Sources are the written MLP outputs in order, then the embedding output;
    a target that does not reach a source (``status='unreachable'``) has a
    zero gradient there, since no path means no direct effect.
    """

    sources = [*sites.writes.values(), sites.embedding]
    shapes = []
    for label in sources:
        shapes.append(tuple(_payload(sites, label, "source").shape))
    if len(set(shapes)) != 1:
        raise _graph_refusal(
            f"The MLP outputs and the embedding output have shapes {shapes}; the graph "
            "needs one residual width. Remedy: name MLPs that write the residual stream "
            "and the embedding that starts it.",
            code="attribution_graph_input_unsupported",
        )
    n_pos, width = shapes[0]
    dtype = _payload(sites, sites.embedding, "embedding").dtype
    by_layer = {
        sites.index.edges[label].layer_label: number for number, label in enumerate(sources)
    }
    grads = torch.zeros(len(seeds), len(sources), n_pos, width, dtype=dtype)
    per_target = len(sources) * n_pos * width * grads.element_size()
    for start in range(0, len(seeds), _TARGETS_PER_READ):
        chunk = seeds[start : start + _TARGETS_PER_READ]
        table = read(
            trace,
            target=chunk,
            within=_within(sources),
            frozen=frozen,
            method="grad",
            reduce=None,
            result_byte_budget=2 * per_target * len(chunk),
        )
        for row in table.rows():
            number = by_layer.get(row.address[0])
            if number is None or row.status != "ok" or not isinstance(row.value, torch.Tensor):
                continue
            grads[start + int(str(row.target_id)[1:]), number] = row.value.detach()[0].to(dtype)
    return grads


def _feature_vectors(
    readers: tuple[Reader, ...], found: _Features, write: str
) -> list[tuple[int, int, torch.Tensor]]:
    """Return ``(node, position, activation * decoder row)`` for features writing ``write``."""

    out = []
    for node, (number, position, feature, act) in enumerate(
        zip(found.reader_index, found.positions, found.features, found.activations, strict=True)
    ):
        for site, decoder in readers[number].decoders:
            if site == write:
                out.append((node, position, act * decoder[feature].to(act.dtype)))
    return out


def _reconstruction(
    readers: tuple[Reader, ...], found: _Features, write: str, shape: torch.Size, dtype: torch.dtype
) -> torch.Tensor:
    """Return what the active features and decoder biases put into ``write``."""

    recon = torch.zeros(shape, dtype=dtype)
    for _node, position, vector in _feature_vectors(readers, found, write):
        recon[position] = recon[position] + vector.to(dtype)
    for reader in readers:
        for site, bias in reader.decoder_biases:
            if site == write:
                recon = recon + bias.to(dtype)
    return recon


def _decoder_bias(
    readers: tuple[Reader, ...], write: str, dtype: torch.dtype
) -> torch.Tensor | None:
    """Return the summed decoder bias written into ``write``."""

    total = None
    for reader in readers:
        for site, bias in reader.decoder_biases:
            if site == write:
                total = bias.to(dtype) if total is None else total + bias.to(dtype)
    return total


def build_graph(  # noqa: PLR0913 -- the graph door: two subjects, then what, where, how
    trace: Any,
    transcoders: Any,
    *,
    embedding: str,
    target: Sequence[int] | None = None,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    demeaned_logits: bool = True,
    logits_at: SiteTarget | None = None,
    frozen: Sequence[str] | _Unset = _UNSET,
) -> AttributionGraph:
    """Build the attribution graph of one prompt's capture.

    Documented-unstable. Nodes are the active features of every transcoder,
    one error node per transcoder-written MLP output and position, one token
    node per position, and one logit node per target logit at the last
    position. Edges are direct linear effects along the frozen graph; see
    :class:`AttributionGraph` for the node order and the accounting.

    Parameters
    ----------
    trace
        A finished capture of one sequence, ``tl.trace(model, tokens[None])``,
        with activations saved.
    transcoders
        Mapping from MLP module name to transcoder: an object with
        ``encode``, ``W_enc``, ``b_enc``, ``W_dec`` and ``b_dec``, the same
        object ``tl.splice_module(transcoder, input="in")`` would splice in.
        A cross-layer transcoder gives ``W_dec`` and ``b_dec`` as mappings
        from MLP module names to tensors.
    embedding
        Module name whose output rows are the token vectors.
    target
        ``None`` picks the salient logits (``max_n_logits`` and
        ``desired_logit_prob``); a list of token ids names them.
    max_n_logits
        Cap on the salient logits.
    desired_logit_prob
        Keep the most probable logits until their probabilities sum to this.
    demeaned_logits
        ``True`` (default) makes each logit node its logit minus the mean
        logit; ``False`` makes it the raw logit.
    logits_at
        ``SiteTarget`` naming the module whose output holds the logits;
        ``None`` reads the model output.
    frozen
        Module names whose outputs are held at their captured values, such
        as attention-pattern and normalisation-scale modules, or op labels
        for values computed inside a module (``"rsqrt_1_5"``). Required:
        pass ``frozen=()`` for a model with neither. Every MLP output a
        transcoder writes is always held.

    Returns
    -------
    AttributionGraph
        The graph, its node accounting and its disclosures.

    Raises
    ------
    AttributionError
        Coded ``attribution_graph_*`` refusals for an omitted ``frozen=``, an
        unknown or repeated site, an unreadable transcoder, a capture that is
        not one saved sequence, or an invalid logit target.
    """

    if isinstance(frozen, _Unset):
        raise _graph_refusal(
            "build_graph needs frozen=: the modules held at their captured values "
            "(attention patterns and normalisation scales), so every edge is a direct "
            "linear effect. Remedy: pass frozen=['blocks.0.attn.pattern', ...], or "
            "frozen=() when the model has neither.",
            code="attribution_graph_frozen_required",
        )
    readers = normalize_transcoders(transcoders)
    sites = _resolve_sites(trace, readers, embedding, logits_at)
    found = _active_features(readers, sites)
    logits = _payload(sites, sites.logits, "logits")
    logit_ids, probabilities = _logit_targets(logits, target, max_n_logits, desired_logit_prob)
    seeds, logit_values = _seeds(readers, sites, found, logit_ids, demeaned_logits)
    frozen_labels = _frozen_labels(trace, sites, frozen)
    grads = _gradients(trace, sites, seeds, frozen_labels)
    embed_out = _payload(sites, sites.embedding, "embedding")
    return _assemble(
        readers,
        sites,
        found,
        logit_ids,
        probabilities,
        logit_values,
        grads,
        embed_out,
        demeaned_logits,
        frozen_labels,
        _capture_outcome_status(trace),
    )


def _assemble(  # noqa: PLR0913 -- one packer for the build's computed parts
    readers: tuple[Reader, ...],
    sites: _Sites,
    found: _Features,
    logit_ids: torch.Tensor,
    probabilities: torch.Tensor,
    logit_values: torch.Tensor,
    grads: torch.Tensor,
    embed_out: torch.Tensor,
    demeaned_logits: bool,
    frozen: list[str],
    capture_outcome: str | None,
) -> AttributionGraph:
    """Contract the read's gradients with every source vector; pack the graph."""

    dtype = embed_out.dtype
    n_pos = embed_out.shape[0]
    write_sites = list(sites.writes)
    nodes = list(found.nodes)
    error_start = len(nodes)
    nodes += [GraphNode("error", write, p) for write in write_sites for p in range(n_pos)]
    token_start = len(nodes)
    token_ids = sites.tokens.tolist() if sites.tokens is not None else [None] * n_pos
    nodes += [
        GraphNode("token", _site_name(sites, "embedding"), p, token=token_ids[p])
        for p in range(n_pos)
    ]
    logit_start = len(nodes)
    logit_site = _site_name(sites, "logits")
    nodes += [GraphNode("logit", logit_site, n_pos - 1, token=int(t)) for t in logit_ids.tolist()]
    n_nodes = len(nodes)
    adjacency = torch.zeros(n_nodes, n_nodes, dtype=dtype)
    decoder_bias = torch.full((n_nodes,), float("nan"), dtype=dtype)
    encoder_bias = torch.full((n_nodes,), float("nan"), dtype=dtype)
    values = torch.full((n_nodes,), float("nan"), dtype=dtype)
    rows = torch.tensor(
        list(range(len(found.nodes))) + list(range(logit_start, n_nodes)), dtype=torch.long
    )
    incoming = torch.zeros(len(rows), n_nodes, dtype=dtype)
    dec_total = torch.zeros(len(rows), dtype=dtype)
    error_vectors = []
    for w_number, write in enumerate(write_sites):
        grad = grads[:, w_number]
        out = _payload(sites, sites.writes[write], "MLP output").to(dtype)
        error = out - _reconstruction(readers, found, write, out.shape, dtype)
        error_vectors.append(error)
        pairs = _feature_vectors(readers, found, write)
        if pairs:
            node_idx = torch.tensor([node for node, _p, _v in pairs], dtype=torch.long)
            pos_idx = torch.tensor([position for _n, position, _v in pairs], dtype=torch.long)
            vectors = torch.stack([vector.to(dtype) for _n, _p, vector in pairs])
            incoming.index_add_(1, node_idx, (grad[:, pos_idx] * vectors).sum(dim=-1))
        start = error_start + w_number * n_pos
        incoming[:, start : start + n_pos] = (grad * error).sum(dim=-1)
        bias = _decoder_bias(readers, write, dtype)
        if bias is not None:
            dec_total = dec_total + (grad @ bias).sum(dim=-1)
    incoming[:, token_start:logit_start] = (grads[:, -1] * embed_out).sum(dim=-1)
    adjacency[rows] = incoming
    decoder_bias[rows] = dec_total
    for node in range(len(found.nodes)):
        reader = readers[found.reader_index[node]]
        encoder_bias[node] = reader.encoder_bias[found.features[node]].to(dtype)
        values[node] = found.pre_activations[node]
    encoder_bias[logit_start:] = 0.0
    values[logit_start:] = logit_values.to(dtype)
    other = values - adjacency.sum(dim=1) - encoder_bias - decoder_bias
    return AttributionGraph(
        nodes=tuple(nodes),
        adjacency_matrix=adjacency,
        node_values=values,
        constants={"encoder_bias": encoder_bias, "decoder_bias": decoder_bias, "other": other},
        active_features=torch.tensor(
            list(zip(found.reader_index, found.positions, found.features, strict=True)),
            dtype=torch.long,
        ).reshape(-1, 3),
        activation_values=(
            torch.stack(found.activations) if found.activations else torch.zeros(0, dtype=dtype)
        ),
        transcoder_sites=tuple(reader.site for reader in readers),
        error_sites=tuple(write_sites),
        error_vectors=torch.stack(error_vectors),
        input_tokens=(
            sites.tokens if sites.tokens is not None else torch.zeros(0, dtype=torch.long)
        ),
        logit_tokens=logit_ids,
        logit_probabilities=probabilities.detach(),
        demeaned_logits=demeaned_logits,
        frozen=tuple(frozen),
        capture_outcome=capture_outcome,
        disclosures=_disclosures(capture_outcome, sites),
    )


def _site_name(sites: _Sites, role: str) -> str:
    """Return the module name behind the embedding or logits role."""

    return sites.embedding_name if role == "embedding" else sites.logits_name


def _disclosures(capture_outcome: str | None, sites: _Sites) -> tuple[str, ...]:
    """Return the graph's plain-language disclosures."""

    notes = []
    if capture_outcome != "COMPLETE":
        notes.append(
            f"the capture's completeness verdict is {capture_outcome!r}, not 'COMPLETE': "
            "edges are built on what was captured and may miss what was not"
        )
    if sites.tokens is None:
        notes.append("the embedding input is not integer token ids; token nodes carry no id")
    return tuple(notes)

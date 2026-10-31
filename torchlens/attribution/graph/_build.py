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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._result import AttributionError
from .._site import SiteTarget
from ..onebackward import read, seed
from ..onebackward._accessor import ReadEdgeIndex, read_edge_index
from ..onebackward._read import _capture_outcome_status, _SitePayloads
from ..onebackward._targets import resolve_site_edge
from ._read_sites import check_upstream, facet_labels, is_module, op_read_label
from ._result import AttributionGraph, GraphNode
from ._transcoders import Reader, normalize_transcoders

if TYPE_CHECKING:
    from ...intervention.selectors import FacetSelector

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


def _read_labels(
    trace: Any, index: ReadEdgeIndex, readers: tuple[Reader, ...]
) -> tuple[dict[str, str], list[Reader]]:
    """Return each transcoder's read edge label and the readers keyed by op labels."""

    op_readers = [reader for reader in readers if not is_module(trace, reader.site)]
    op_sites = {reader.site for reader in op_readers}
    reads: dict[str, str] = {}
    for reader in readers:
        if reader.site in op_sites:
            reads[reader.site] = op_read_label(index, reader)
            continue
        module = _module(trace, reader.site, "transcoder")
        reads[reader.site] = _edge_label(
            index, _one_op(module, reader.site, "input_ops", "MLP"), "target"
        )
    return reads, op_readers


def _resolve_sites(
    trace: Any,
    readers: tuple[Reader, ...],
    embedding: str,
    logits_at: SiteTarget | None,
) -> _Sites:
    """Address every read, write, embedding and logit site on the capture.

    A transcoder key is a module name or an op label (``_read_sites``).
    """

    index = read_edge_index(trace)
    reads, op_readers = _read_labels(trace, index, readers)
    writes: dict[str, str] = {}
    for reader in readers:
        for write, _decoder in reader.decoders:
            if write not in writes:
                module = _module(trace, write, "MLP output")
                op = _one_op(module, write, "output_ops", "MLP output")
                writes[write] = _edge_label(index, op, "input")
    for reader in op_readers:
        check_upstream(trace, index, reader, reads[reader.site], writes)
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
                f"logits_at=tl.attribution.SiteTarget({logits_at.module_address!r}).",
                code="attribution_graph_target_invalid",
            )
        module = _module(trace, logits_at.module_address, "logits")
        op = _one_op(module, logits_at.module_address, "output_ops", "logits")
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
        "output" if logits_at is None else logits_at.module_address,
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
            found.nodes.append(GraphNode("feature", reader.site, position, feature_index=feature))
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
                f"max_num_logits={max_n_logits!r} and logit_probability_coverage="
                f"{desired_logit_prob!r} select no logits. Remedy: pass max_num_logits=10, "
                "logit_probability_coverage=0.95 (the defaults).",
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


@dataclass(frozen=True)
class _SeedSpec:
    """One target node's seed, built into a cotangent only when its chunk runs."""

    label: str
    shape: tuple[int, ...]
    position: int
    vector: torch.Tensor


def _seed_specs(
    readers: tuple[Reader, ...],
    sites: _Sites,
    found: _Features,
    logit_ids: torch.Tensor,
    demeaned_logits: bool,
) -> tuple[list[_SeedSpec], torch.Tensor]:
    """Return one seed spec per target node and the logit nodes' values."""

    specs: list[_SeedSpec] = []
    for number, position, feature in zip(
        found.reader_index, found.positions, found.features, strict=True
    ):
        reader = readers[number]
        label = sites.reads[reader.site]
        inputs = _payload(sites, label, "MLP input")
        vector = reader.encoder[:, feature].to(inputs.dtype)
        specs.append(_SeedSpec(label, (1, *inputs.shape), position, vector))
    logits = _payload(sites, sites.logits, "logits")
    last = logits.shape[0] - 1
    vocab = logits.shape[1]
    values = []
    for token in logit_ids.tolist():
        direction = torch.zeros(vocab, dtype=logits.dtype)
        direction[token] = 1.0
        if demeaned_logits:
            direction = direction - 1.0 / vocab
        specs.append(_SeedSpec(sites.logits, (1, *logits.shape), last, direction))
        values.append((logits[last] * direction).sum())
    return specs, torch.stack(values) if values else torch.zeros(0, dtype=logits.dtype)


def _seed(spec: _SeedSpec) -> Any:
    """Return the one-backward seed of one spec."""

    cot = _cotangent(spec.shape, spec.vector.dtype, spec.position, spec.vector)
    return seed(spec.label, cotangent=cot)


def _within(labels: list[str]) -> Any:
    """Return the union selection of ``labels`` for ``read(within=...)``."""

    from ..._selection_values import _lift_within

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


def _frozen_labels(trace: Any, sites: _Sites, frozen: Any) -> list[str]:
    """Return the frozen set: every written MLP output plus the caller's sites."""

    from ...intervention.selectors import FacetSelector

    labels = list(sites.writes.values())
    for name in [frozen] if isinstance(frozen, FacetSelector) else frozen:
        if isinstance(name, FacetSelector):
            named = facet_labels(trace, sites.index, name)
        elif isinstance(name, str):
            named = [_frozen_label(trace, sites.index, name)]
        else:
            raise _graph_refusal(
                f"frozen= holds {name!r}; it takes module names, op labels and facet "
                "selectors. Remedy: pass frozen=['blocks.0.attn.pattern', ...], the modules "
                "whose outputs are held at their captured values, or "
                "frozen=[tl.facet('pattern'), ...].",
                code="attribution_graph_site_not_found",
            )
        labels.extend(label for label in named if label not in labels)
    return labels


#: Targets per one-backward read; each read returns every target's gradient
#: at every source site, so the chunk bounds the read's result size and the
#: builder's gradient buffer.
_TARGETS_PER_READ = 256


@dataclass
class _BuildCounters:
    """Deterministic counters of one build (internal; filled when passed in).

    Attributes
    ----------
    reads
        One-backward reads run (one per chunk of targets).
    gradient_buffer_peak_bytes
        Largest gradient buffer the builder held at once:
        ``chunk x sources x positions x width x element size``.
    """

    reads: int = 0
    gradient_buffer_peak_bytes: int = 0


def _sources(sites: _Sites) -> tuple[list[str], int, int, torch.dtype]:
    """Return the source edge labels, positions, width and dtype."""

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
    return sources, n_pos, width, _payload(sites, sites.embedding, "embedding").dtype


def _gradient_chunks(
    trace: Any,
    sites: _Sites,
    specs: list[_SeedSpec],
    frozen: list[str],
    counters: _BuildCounters,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(start, (chunk, n_sources, positions, width))`` gradients per read.

    Sources are the written MLP outputs in order, then the embedding output;
    a target that does not reach a source (``status='unreachable'``) has a
    zero gradient there, since no path means no direct effect. Each chunk's
    buffer is released by the caller before the next read runs.
    """

    sources, n_pos, width, dtype = _sources(sites)
    by_layer = {
        sites.index.edges[label].layer_label: number for number, label in enumerate(sources)
    }
    for start in range(0, len(specs), _TARGETS_PER_READ):
        chunk = [_seed(spec) for spec in specs[start : start + _TARGETS_PER_READ]]
        grads = torch.zeros(len(chunk), len(sources), n_pos, width, dtype=dtype)
        per_target = len(sources) * n_pos * width * grads.element_size()
        counters.reads += 1
        counters.gradient_buffer_peak_bytes = max(
            counters.gradient_buffer_peak_bytes, grads.numel() * grads.element_size()
        )
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
            grads[int(str(row.target_id)[1:]), number] = row.value.detach()[0].to(dtype)
        del table
        yield start, grads


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
    embedding_module: str,
    target: Sequence[int] | None = None,
    max_num_logits: int = 10,
    logit_probability_coverage: float = 0.95,
    demeaned_logits: bool = True,
    logits_at: SiteTarget | None = None,
    frozen: Sequence[str | FacetSelector] | FacetSelector | _Unset = _UNSET,
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
        Mapping from read site to transcoder. A read site is the MLP module
        name (the transcoder reads the module's call input) or an op label
        (it reads that op's output, for a read point inside a module, such
        as a norm's value before its gain); an op-label key needs ``W_dec``
        and ``b_dec`` as mappings naming the MLP outputs it writes, and must
        run before them. The transcoder is an object with
        ``encode``, ``W_enc``, ``b_enc``, ``W_dec`` and ``b_dec``, the same
        object ``tl.splice_module(transcoder, input_route="in")`` would splice in.
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
        as attention-pattern and normalisation-scale modules, op labels
        for values computed inside a module (``"rsqrt_1_5"``), or facet
        selectors such as ``tl.facet("pattern")``, which hold the op behind
        that facet in every module that serves it. Required: pass
        ``frozen=()`` for a model with neither. Every MLP output a
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
    return _build(
        trace,
        transcoders,
        embedding=embedding_module,
        target=target,
        max_n_logits=max_num_logits,
        desired_logit_prob=logit_probability_coverage,
        demeaned_logits=demeaned_logits,
        logits_at=logits_at,
        frozen=frozen,
        counters=_BuildCounters(),
    )


def _build(  # noqa: PLR0913 -- build_graph's keywords plus the internal counters
    trace: Any,
    transcoders: Any,
    *,
    embedding: str,
    target: Sequence[int] | None,
    max_n_logits: int,
    desired_logit_prob: float,
    demeaned_logits: bool,
    logits_at: SiteTarget | None,
    frozen: Sequence[str | FacetSelector] | FacetSelector,
    counters: _BuildCounters,
) -> AttributionGraph:
    """Build the graph; fill ``counters`` (the body of :func:`build_graph`)."""

    readers = normalize_transcoders(transcoders)
    sites = _resolve_sites(trace, readers, embedding, logits_at)
    found = _active_features(readers, sites)
    logits = _payload(sites, sites.logits, "logits")
    logit_ids, probabilities = _logit_targets(logits, target, max_n_logits, desired_logit_prob)
    specs, logit_values = _seed_specs(readers, sites, found, logit_ids, demeaned_logits)
    frozen_labels = _frozen_labels(trace, sites, frozen)
    chunks = _gradient_chunks(trace, sites, specs, frozen_labels, counters)
    embed_out = _payload(sites, sites.embedding, "embedding")
    nodes, starts = _node_layout(sites, found, logit_ids, embed_out.shape[0])
    writes = _write_sources(readers, sites, found, embed_out.dtype)
    adjacency, dec_total = _edges(writes, found, nodes, starts, chunks, embed_out)
    return _pack(
        readers,
        sites,
        found,
        writes,
        nodes,
        starts,
        logit_ids,
        probabilities,
        logit_values,
        adjacency,
        dec_total,
        demeaned_logits,
        frozen_labels,
        _capture_outcome_status(trace),
    )


@dataclass(frozen=True)
class _Starts:
    """Where each node kind starts in the node order."""

    error: int
    token: int
    logit: int


def _node_layout(
    sites: _Sites, found: _Features, logit_ids: torch.Tensor, n_pos: int
) -> tuple[list[GraphNode], _Starts]:
    """Return the node order: features, errors, tokens, logits."""

    nodes = list(found.nodes)
    error_start = len(nodes)
    nodes += [GraphNode("error", write, p) for write in sites.writes for p in range(n_pos)]
    token_start = len(nodes)
    token_ids = sites.tokens.tolist() if sites.tokens is not None else [None] * n_pos
    nodes += [
        GraphNode("token", _site_name(sites, "embedding"), p, token_id=token_ids[p])
        for p in range(n_pos)
    ]
    logit_start = len(nodes)
    logit_site = _site_name(sites, "logits")
    nodes += [
        GraphNode("logit", logit_site, n_pos - 1, token_id=int(t)) for t in logit_ids.tolist()
    ]
    return nodes, _Starts(error_start, token_start, logit_start)


@dataclass(frozen=True)
class _WriteSources:
    """What one written MLP output contributes as sources."""

    node_idx: torch.Tensor
    pos_idx: torch.Tensor
    vectors: torch.Tensor | None
    error: torch.Tensor
    bias: torch.Tensor | None


def _write_sources(
    readers: tuple[Reader, ...], sites: _Sites, found: _Features, dtype: torch.dtype
) -> list[_WriteSources]:
    """Return each written MLP output's feature vectors, error vector and bias."""

    out = []
    for write in sites.writes:
        payload = _payload(sites, sites.writes[write], "MLP output").to(dtype)
        error = payload - _reconstruction(readers, found, write, payload.shape, dtype)
        pairs = _feature_vectors(readers, found, write)
        out.append(
            _WriteSources(
                torch.tensor([node for node, _p, _v in pairs], dtype=torch.long),
                torch.tensor([position for _n, position, _v in pairs], dtype=torch.long),
                torch.stack([vector.to(dtype) for _n, _p, vector in pairs]) if pairs else None,
                error,
                _decoder_bias(readers, write, dtype),
            )
        )
    return out


def _edges(  # noqa: PLR0913 -- the build's parts the contraction reads
    writes: list[_WriteSources],
    found: _Features,
    nodes: list[GraphNode],
    starts: _Starts,
    chunks: Iterator[tuple[int, torch.Tensor]],
    embed_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract each chunk's gradients with every source vector, chunk by chunk.

    Returns the adjacency (rows are targets) and each row's decoder-bias
    term. Only one chunk's gradients are alive at a time.
    """

    dtype = embed_out.dtype
    n_pos = embed_out.shape[0]
    n_nodes = len(nodes)
    rows = torch.tensor(
        list(range(len(found.nodes))) + list(range(starts.logit, n_nodes)), dtype=torch.long
    )
    adjacency = torch.zeros(n_nodes, n_nodes, dtype=dtype)
    dec_total = torch.zeros(len(rows), dtype=dtype)
    for start, grads in chunks:
        stop = start + grads.shape[0]
        incoming = torch.zeros(grads.shape[0], n_nodes, dtype=dtype)
        for w_number, source in enumerate(writes):
            grad = grads[:, w_number]
            if source.vectors is not None:
                incoming.index_add_(
                    1, source.node_idx, (grad[:, source.pos_idx] * source.vectors).sum(dim=-1)
                )
            first = starts.error + w_number * n_pos
            incoming[:, first : first + n_pos] = (grad * source.error).sum(dim=-1)
            if source.bias is not None:
                dec_total[start:stop] += (grad @ source.bias).sum(dim=-1)
        incoming[:, starts.token : starts.logit] = (grads[:, -1] * embed_out).sum(dim=-1)
        adjacency[rows[start:stop]] = incoming
        del grads, incoming
    return adjacency, dec_total


def _pack(  # noqa: PLR0913 -- one packer for the build's computed parts
    readers: tuple[Reader, ...],
    sites: _Sites,
    found: _Features,
    writes: list[_WriteSources],
    nodes: list[GraphNode],
    starts: _Starts,
    logit_ids: torch.Tensor,
    probabilities: torch.Tensor,
    logit_values: torch.Tensor,
    adjacency: torch.Tensor,
    dec_total: torch.Tensor,
    demeaned_logits: bool,
    frozen: list[str],
    capture_outcome: str | None,
) -> AttributionGraph:
    """Fill the node values and constants; pack the graph."""

    dtype = adjacency.dtype
    n_nodes = len(nodes)
    rows = torch.tensor(
        list(range(len(found.nodes))) + list(range(starts.logit, n_nodes)), dtype=torch.long
    )
    decoder_bias = torch.full((n_nodes,), float("nan"), dtype=dtype)
    encoder_bias = torch.full((n_nodes,), float("nan"), dtype=dtype)
    values = torch.full((n_nodes,), float("nan"), dtype=dtype)
    decoder_bias[rows] = dec_total
    for node in range(len(found.nodes)):
        reader = readers[found.reader_index[node]]
        encoder_bias[node] = reader.encoder_bias[found.features[node]].to(dtype)
        values[node] = found.pre_activations[node]
    encoder_bias[starts.logit :] = 0.0
    values[starts.logit :] = logit_values.to(dtype)
    other = values - adjacency.sum(dim=1) - encoder_bias - decoder_bias
    return AttributionGraph(
        nodes=tuple(nodes),
        adjacency_matrix=adjacency,
        node_values=values,
        constants={"encoder_bias": encoder_bias, "decoder_bias": decoder_bias, "other": other},
        active_feature_indices=torch.tensor(
            list(zip(found.reader_index, found.positions, found.features, strict=True)),
            dtype=torch.long,
        ).reshape(-1, 3),
        feature_activations=(
            torch.stack(found.activations) if found.activations else torch.zeros(0, dtype=dtype)
        ),
        transcoder_sites=tuple(reader.site for reader in readers),
        error_sites=tuple(sites.writes),
        error_vectors=torch.stack([source.error for source in writes]),
        input_token_ids=(
            sites.tokens if sites.tokens is not None else torch.zeros(0, dtype=torch.long)
        ),
        logit_token_ids=logit_ids,
        logit_probabilities=probabilities.detach(),
        has_demeaned_logits=demeaned_logits,
        frozen_sites=tuple(frozen),
        capture_status=capture_outcome,
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

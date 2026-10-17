"""Attribution graphs in Neuronpedia's published attribution-graph schema.

The schema is ``apps/webapp/app/api/graph/graph-schema.json`` in the
Neuronpedia repository (https://github.com/hijohnnylin/neuronpedia), JSON
Schema draft-07, title "Anthropic Attribution Graph", version 1.0.0; this
writer follows it as of commit 2f0bb10893b9c63c409dbc49db61f557883e0a2c
(the file's sha256 115c418bc7f6ad2a982f638c2b3fa260913a8f23e6789293f4cc293d614e59c1).
Neuronpedia's validator page, https://www.neuronpedia.org/graph/validator,
serves the same schema.

Node ids, node fields, link orientation and pruning follow circuit-tracer's
own writer (``circuit_tracer.utils.create_graph_files`` at v0.5.2), so a
graph opens in the same viewer the same way. Circuit-tracer is not imported:
its pruning rule is restated here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .._result import AttributionError

if TYPE_CHECKING:
    from ._result import AttributionGraph

__all__ = ["graph_to_neuronpedia"]

_DOCS_POINTER = "See docs/reference/attribution.md, section Attribution graphs."
_SCHEMA_VERSION = 1


def _refusal(message: str, *, code: str) -> AttributionError:
    """Build one coded export refusal ending in its docs pointer."""

    return AttributionError(f"{message} {_DOCS_POINTER}", code=code)


@dataclass(frozen=True)
class _Pruned:
    """What survives pruning: node and edge masks and every node's influence."""

    node_mask: torch.Tensor
    edge_mask: torch.Tensor
    influence: torch.Tensor


def _normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Row-normalised absolute matrix (rows summing to zero stay zero)."""

    absolute = matrix.abs()
    return absolute / absolute.sum(dim=1, keepdim=True).clamp(min=1e-10)


def _influence(matrix: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``weights @ (A + A^2 + ...)`` for the nilpotent adjacency of a graph.

    The series ends because the graph has no cycles; it never needs more
    steps than the longest path, which is below the node count.
    """

    current = weights @ matrix
    total = current
    for _ in range(matrix.shape[0]):
        if not bool(current.any()):
            return total
        current = current @ matrix
        total = total + current
    raise _refusal(
        "The graph's influence series did not end, so its adjacency has a cycle "
        "and is not an attribution graph built by tl.attribution.build_graph. "
        "Remedy: export a graph returned by tl.attribution.build_graph(...).",
        code="attribution_graph_export_unsupported",
    )


def _threshold(scores: torch.Tensor, share: float) -> torch.Tensor:
    """The smallest score kept when keeping ``share`` of the total score."""

    ordered = torch.sort(scores, descending=True).values
    cumulative = torch.cumsum(ordered, dim=0) / torch.sum(ordered)
    index = int(torch.searchsorted(cumulative, torch.tensor(share, dtype=cumulative.dtype)))
    return ordered[min(index, len(cumulative) - 1)]


def _drop_dead_ends(
    node_mask: torch.Tensor, edge_mask: torch.Tensor, n_features: int, n_inner: int
) -> None:
    """Drop feature and error nodes left without kept edges, until none change.

    Feature and error nodes need a kept outgoing edge; feature nodes also
    need a kept incoming edge. Token and logit nodes are always kept.
    """

    previous = node_mask.clone()
    node_mask[:n_inner] &= edge_mask[:, :n_inner].any(0)
    node_mask[:n_features] &= edge_mask[:n_features].any(1)
    while not torch.equal(node_mask, previous):
        previous.copy_(node_mask)
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[:n_inner] &= edge_mask[:, :n_inner].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)


def _prune(graph: AttributionGraph, node_threshold: float, edge_threshold: float) -> _Pruned:
    """Circuit-tracer's pruning rule on the graph's adjacency, in its dtype."""

    adjacency = graph.adjacency_matrix.detach().cpu()
    counts = graph.node_counts
    n_logits, n_tokens = counts["logit"], counts["token"]
    n_inner = adjacency.shape[0] - n_logits - n_tokens
    weights = torch.zeros(adjacency.shape[0], dtype=adjacency.dtype)
    weights[adjacency.shape[0] - n_logits :] = (
        graph.logit_probabilities.detach().cpu().to(adjacency.dtype)
    )
    node_influence = _influence(_normalize(adjacency), weights)
    if not bool(node_influence.sum() > 0):
        raise _refusal(
            "No node of the graph influences its logits (every edge into the logits, "
            "or every logit probability, is zero), so no node can be kept by its share "
            "of influence. Remedy: build the graph with targets the prompt reaches, "
            "tl.attribution.build_graph(trace, transcoders, embedding=..., target=[token_id]).",
            code="attribution_graph_export_unsupported",
        )
    node_mask = node_influence >= _threshold(node_influence, node_threshold)
    node_mask[n_inner:] = True
    pruned = adjacency.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    normalized = _normalize(pruned)
    edge_scores = normalized * (_influence(normalized, weights) + weights)[:, None]
    edge_mask = edge_scores >= _threshold(edge_scores.flatten(), edge_threshold)
    _drop_dead_ends(node_mask, edge_mask, counts["feature"], n_inner)
    ordered, order = torch.sort(node_influence, descending=True)
    cumulative = torch.zeros_like(node_influence)
    cumulative[order] = torch.cumsum(ordered, dim=0) / torch.sum(ordered)
    return _Pruned(node_mask, edge_mask, cumulative)


def _cantor(layer: int, feature: int) -> int:
    """Circuit-tracer's feature number: the Cantor pairing of layer and index."""

    return (layer + feature) * (layer + feature + 1) // 2 + feature


def _node(  # noqa: PLR0913 -- the schema's node fields
    node_id: str,
    feature: int,
    layer: str,
    position: int,
    feature_type: str,
    js_node_id: str,
    *,
    influence: float | None,
    activation: float | None = None,
    clerp: str = "",
    token_prob: float = 0.0,
    is_target_logit: bool = False,
) -> dict[str, Any]:
    """One node in the reference writer's field order."""

    return {
        "node_id": node_id,
        "feature": feature,
        "layer": layer,
        "ctx_idx": position,
        "feature_type": feature_type,
        "token_prob": token_prob,
        "is_target_logit": is_target_logit,
        "run_idx": 0,
        "reverse_ctx_idx": 0,
        "jsNodeId": js_node_id,
        "clerp": clerp,
        "influence": influence,
        "activation": activation,
    }


def _decode(tokenizer: Any, token_id: int) -> str:
    """One token id decoded alone."""

    return str(tokenizer.decode([int(token_id)]))


@dataclass(frozen=True)
class _Layout:
    """Where each node kind starts, and the logit layer's number."""

    error: int
    token: int
    logit: int
    logit_layer: str


def _layout(graph: AttributionGraph) -> _Layout:
    """Node-kind starts in the graph's node order."""

    counts = graph.node_counts
    error = counts["feature"]
    token = error + counts["error"]
    return _Layout(error, token, token + counts["token"], str(len(graph.error_sites) + 1))


def _schema_node(
    graph: AttributionGraph, layout: _Layout, index: int, influence: float, tokenizer: Any
) -> dict:
    """The schema node for graph node ``index``."""

    node = graph.nodes[index]
    position = node.position
    if index < layout.error:
        layer, _position, feature = (int(v) for v in graph.active_features[index].tolist())
        return _node(
            f"{layer}_{feature}_{position}",
            _cantor(layer, feature),
            str(layer),
            position,
            "cross layer transcoder",
            f"{layer}_{feature}-0",
            influence=influence,
            activation=float(graph.activation_values[index]),
        )
    if index < layout.token:
        layer = graph.error_sites.index(node.site)
        return _node(
            f"{layer}_-1_{position}",
            -1,
            str(layer),
            position,
            "mlp reconstruction error",
            f"{layer}_-1-{position}",
            influence=influence,
        )
    if index < layout.logit:
        return _node(
            f"E_{node.token}_{position}",
            position,
            "E",
            position,
            "embedding",
            f"E_{node.token}-{position}",
            influence=influence,
        )
    return _logit_node(graph, index - layout.logit, layout.logit_layer, tokenizer)


def _logit_node(graph: AttributionGraph, k: int, layer: str, tokenizer: Any) -> dict:
    """The schema node for the ``k``-th logit target (no influence, as the reference)."""

    vocab = int(graph.logit_tokens[k])
    position = graph.nodes[-1].position
    probability = float(graph.logit_probabilities[k])
    return _node(
        f"{layer}_{vocab}_{position}",
        vocab,
        layer,
        position,
        "logit",
        f"L_{vocab}-{position}",
        influence=None,
        clerp=f'Output "{_decode(tokenizer, vocab)}" (p={probability:.3f})',
        token_prob=probability,
        is_target_logit=k == 0,
    )


def _nodes_and_links(
    graph: AttributionGraph, pruned: _Pruned, tokenizer: Any
) -> tuple[list[dict], list[dict]]:
    """Kept nodes and links, dropping kept feature and error nodes without a kept link."""

    layout = _layout(graph)
    kept = pruned.node_mask.nonzero().flatten().tolist()
    influence = pruned.influence.tolist()
    nodes = {
        index: _schema_node(graph, layout, index, influence[index], tokenizer) for index in kept
    }
    targets, sources = pruned.edge_mask.nonzero(as_tuple=True)
    weights = graph.adjacency_matrix.detach().cpu()[targets, sources].tolist()
    links = [
        {"source": nodes[s]["node_id"], "target": nodes[t]["node_id"], "weight": weight}
        for s, t, weight in zip(sources.tolist(), targets.tolist(), weights, strict=True)
        if s in nodes and t in nodes
    ]
    linked = {link["source"] for link in links} | {link["target"] for link in links}
    used = [
        node
        for node in nodes.values()
        if node["node_id"] in linked or node["feature_type"] in ("embedding", "logit")
    ]
    return used, links


def _check_arguments(
    slug: Any, scan: Any, tokenizer: Any, node_threshold: Any, edge_threshold: Any
) -> None:
    """Refuse keyword values the export cannot write."""

    for name, value in (("slug", slug), ("scan", scan)):
        if not isinstance(value, str) or not value:
            raise _refusal(
                f"{name}={value!r} is not a non-empty string; the schema requires one. "
                f'Remedy: pass {name}="gemma-2-2b" (the scan is Neuronpedia\'s model id).',
                code="attribution_graph_export_argument_invalid",
            )
    if not callable(getattr(tokenizer, "decode", None)):
        raise _refusal(
            f"tokenizer={type(tokenizer).__name__} has no decode method, so token ids "
            "cannot become the schema's token strings. Remedy: pass the model's tokenizer, "
            "for example tokenizer=transformers.AutoTokenizer.from_pretrained(name).",
            code="attribution_graph_export_argument_invalid",
        )
    for name, value in (("node_threshold", node_threshold), ("edge_threshold", edge_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise _refusal(
                f"{name}={value!r} is not a share between 0 and 1. "
                "Remedy: pass node_threshold=0.8, edge_threshold=0.98 (the defaults).",
                code="attribution_graph_export_argument_invalid",
            )


def _check_graph(graph: AttributionGraph) -> None:
    """Refuse graphs the schema cannot express."""

    if any(graph.nodes[i].token is None for i in graph.node_indices("token")):
        raise _refusal(
            "The graph's token nodes carry no token ids (the embedding input was not "
            "integer ids), so the schema's prompt tokens and token node ids cannot be "
            "written. Remedy: build the graph from integer token ids, "
            'tl.attribution.build_graph(tl.trace(model, input_ids), transcoders, embedding="...").',
            code="attribution_graph_export_unsupported",
        )
    if len(graph.transcoder_sites) != len(graph.error_sites):
        raise _refusal(
            f"The graph has {len(graph.transcoder_sites)} transcoders but "
            f"{len(graph.error_sites)} MLP outputs with error nodes, so a feature's layer "
            "number and an error node's layer number would count different things. "
            "Remedy: build the graph with one transcoder per MLP it writes, "
            "tl.attribution.build_graph(trace, {mlp_name: transcoder, ...}, ...).",
            code="attribution_graph_export_unsupported",
        )
    if not bool(torch.isfinite(graph.adjacency_matrix).all()) or not bool(
        torch.isfinite(graph.activation_values).all()
    ):
        raise _refusal(
            "The graph's adjacency or feature activations hold NaN or infinite values, "
            "which JSON cannot carry. Remedy: rebuild the graph from a capture whose "
            "values are finite, tl.attribution.build_graph(tl.trace(model, input_ids), ...).",
            code="attribution_graph_export_unsupported",
        )


def graph_to_neuronpedia(  # noqa: PLR0913 -- the method's keywords, passed through
    graph: AttributionGraph,
    *,
    slug: str,
    scan: str,
    tokenizer: Any,
    prompt: str | None,
    node_threshold: float,
    edge_threshold: float,
    path: str | Path | None,
) -> dict[str, Any]:
    """Implement :meth:`AttributionGraph.to_neuronpedia`."""

    _check_arguments(slug, scan, tokenizer, node_threshold, edge_threshold)
    _check_graph(graph)
    prompt_tokens = [_decode(tokenizer, token) for token in graph.input_tokens.tolist()]
    if prompt is None:
        prompt = str(tokenizer.decode([int(t) for t in graph.input_tokens.tolist()]))
    pruned = _prune(graph, float(node_threshold), float(edge_threshold))
    nodes, links = _nodes_and_links(graph, pruned, tokenizer)
    exported = {
        "metadata": {
            "slug": slug,
            "scan": scan,
            "transcoder_list": [],
            "prompt_tokens": prompt_tokens,
            "prompt": prompt,
            "node_threshold": float(node_threshold),
            "schema_version": _SCHEMA_VERSION,
            "pruning_settings": {
                "node_threshold": float(node_threshold),
                "edge_threshold": float(edge_threshold),
            },
        },
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
            "clickedId": "",
            "sg_pos": "",
        },
        "nodes": nodes,
        "links": links,
    }
    if path is not None:
        Path(path).write_text(json.dumps(exported, allow_nan=False), encoding="utf-8")
    return exported

"""Attribution graphs exported in Neuronpedia's attribution-graph schema.

The structural contract (the library's ``check_contract``, which the
upload also runs before any request) is written by hand from Neuronpedia's
published schema (``apps/webapp/app/api/graph/graph-schema.json`` at commit
2f0bb10893b9c63c409dbc49db61f557883e0a2c): the required keys and value
types of the top level, ``metadata``, ``qParams``, nodes and links. The node
layout (ids, ``feature``, ``layer``, ``jsNodeId``, logit labels) and the
pruning rule are those of circuit-tracer's own writer, restated here in
numpy without importing it.

Pre-registered bars (float64 fixtures F1 and F3): every exported link
weight equals the graph's adjacency entry exactly, and every kept node's
influence equals an independent numpy computation of the influence rule
(closed form ``w @ ((I - A)^-1 - I)``, then the cumulative share) to 1e-12.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator

import numpy as np
import pytest
import torch

import torchlens as tl
from tests import _ag1_fixtures as fx
from torchlens.attribution.graph._neuronpedia import check_contract

INFLUENCE_TOL = 1e-12


class _Tokenizer:
    """Decodes every id to a bracketed string, as a tokenizer would a token."""

    def decode(self, ids: list[int]) -> str:
        return "".join(f"<{i}>" for i in ids)


TOKENIZER = _Tokenizer()


def _graph(name: str) -> tl.attribution.graph.AttributionGraph:
    n_layers = 3 if name == "F3" else 2
    model = fx.build_model(n_layers)
    transcoders = {
        "F1": fx.exact_transcoders,
        "F2": fx.lossy_transcoders,
        "F3": fx.cross_layer_transcoders,
    }[name](model)
    return tl.attribution.build_graph(
        tl.trace(model, torch.tensor([fx.TOKENS])),
        transcoders,
        embedding_module="embed",
        frozen=fx.frozen_names(n_layers),
        max_num_logits=2,
        logit_probability_coverage=0.99,
    )


@pytest.fixture(scope="module")
def graphs() -> Iterator[dict[str, tl.attribution.graph.AttributionGraph]]:
    built = {name: _graph(name) for name in ("F1", "F2", "F3")}
    try:
        yield built
    finally:
        built.clear()


def _export(graph, **kwargs) -> dict:
    return graph.to_neuronpedia(slug="fixture", model_id="tiny", tokenizer=TOKENIZER, **kwargs)


# ---------------------------------------------------------------- independent answer key


def _expected_ids(graph) -> list[str]:
    """Every graph node's schema id, from its GraphNode fields alone."""

    ids = []
    n_layers = len(graph.error_sites)
    for index, node in enumerate(graph.nodes):
        if node.kind == "feature":
            layer = graph.transcoder_sites.index(node.site)
            ids.append(f"{layer}_{node.feature_index}_{node.position}")
        elif node.kind == "error":
            ids.append(f"{graph.error_sites.index(node.site)}_-1_{node.position}")
        elif node.kind == "token":
            ids.append(f"E_{node.token_id}_{node.position}")
        else:
            ids.append(f"{n_layers + 1}_{node.token_id}_{node.position}")
        assert index == len(ids) - 1
    return ids


def _numpy_influence(graph) -> np.ndarray:
    """Circuit-tracer's node influence in closed form, then the cumulative share."""

    adjacency = graph.adjacency_matrix.numpy().astype(np.float64)
    absolute = np.abs(adjacency)
    normalized = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1e-10)
    n = adjacency.shape[0]
    weights = np.zeros(n)
    n_logits = len(graph.node_indices("logit"))
    weights[n - n_logits :] = graph.logit_probabilities.numpy()
    influence = weights @ (np.linalg.inv(np.eye(n) - normalized) - np.eye(n))
    order = np.argsort(-influence, kind="stable")
    shares = np.empty(n)
    shares[order] = np.cumsum(influence[order]) / influence.sum()
    return np.stack([influence, shares])


def _numpy_threshold(scores: np.ndarray, share: float) -> float:
    ordered = np.sort(scores)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    index = int(np.searchsorted(cumulative, share, side="left"))
    return float(ordered[min(index, len(ordered) - 1)])


def _numpy_kept(
    graph, node_influence_coverage=0.8, edge_influence_coverage=0.98
) -> tuple[set, set]:
    """Kept node ids and (source, target) id pairs by the pruning rule, in numpy."""

    influence = _numpy_influence(graph)[0]
    adjacency = graph.adjacency_matrix.numpy().astype(np.float64)
    n = adjacency.shape[0]
    n_features = len(graph.node_indices("feature"))
    n_inner = n - len(graph.node_indices("token")) - len(graph.node_indices("logit"))
    weights = np.zeros(n)
    weights[n - len(graph.node_indices("logit")) :] = graph.logit_probabilities.numpy()
    node_mask = influence >= _numpy_threshold(influence, node_influence_coverage)
    node_mask[n_inner:] = True
    pruned = adjacency * node_mask[:, None] * node_mask[None, :]
    absolute = np.abs(pruned)
    normalized = absolute / np.maximum(absolute.sum(axis=1, keepdims=True), 1e-10)
    reach = weights @ (np.linalg.inv(np.eye(n) - normalized) - np.eye(n)) + weights
    scores = normalized * reach[:, None]
    edge_mask = scores >= _numpy_threshold(scores.flatten(), edge_influence_coverage)
    while True:
        before = node_mask.copy()
        edge_mask &= node_mask[:, None] & node_mask[None, :]
        node_mask[:n_inner] &= edge_mask[:, :n_inner].any(axis=0)
        node_mask[:n_features] &= edge_mask[:n_features].any(axis=1)
        if (before == node_mask).all():
            break
    ids = _expected_ids(graph)
    targets, sources = np.nonzero(edge_mask & node_mask[:, None] & node_mask[None, :])
    links = {(ids[s], ids[t]) for s, t in zip(sources, targets, strict=True)}
    linked = {end for pair in links for end in pair}
    kinds = [node.kind for node in graph.nodes]
    nodes = {
        ids[i]
        for i in np.nonzero(node_mask)[0]
        if ids[i] in linked or kinds[i] in ("token", "logit")
    }
    return nodes, links


# ---------------------------------------------------------------- structural contract


def _check_contract(exported: dict) -> None:
    """The library's structural contract, written from the published schema."""

    check_contract(exported)


# ---------------------------------------------------------------- cases


@pytest.mark.parametrize("name", ["F1", "F2", "F3"])
def test_export_meets_the_structural_contract(graphs, name) -> None:
    exported = _export(graphs[name])
    _check_contract(exported)
    json.dumps(exported, allow_nan=False)
    assert exported["links"], "the default pruning keeps edges on every fixture"


@pytest.mark.parametrize("name", ["F1", "F3"])
def test_link_weights_equal_the_adjacency(graphs, name) -> None:
    graph = graphs[name]
    index = {node_id: i for i, node_id in enumerate(_expected_ids(graph))}
    exported = _export(graph)
    for link in exported["links"]:
        expected = graph.adjacency_matrix[index[link["target"]], index[link["source"]]].item()
        assert link["weight"] == expected


@pytest.mark.parametrize("name", ["F1", "F3"])
def test_influence_equals_an_independent_computation(graphs, name) -> None:
    graph = graphs[name]
    shares = _numpy_influence(graph)[1]
    index = {node_id: i for i, node_id in enumerate(_expected_ids(graph))}
    exported = _export(graph)
    checked = 0
    for node in exported["nodes"]:
        if node["feature_type"] == "logit":
            assert node["influence"] is None
            continue
        assert abs(node["influence"] - shares[index[node["node_id"]]]) <= INFLUENCE_TOL
        checked += 1
    assert checked >= len(graph.node_indices("token"))


@pytest.mark.parametrize("name", ["F1", "F3"])
def test_kept_nodes_and_links_follow_the_pruning_rule(graphs, name) -> None:
    graph = graphs[name]
    nodes, links = _numpy_kept(graph)
    exported = _export(graph)
    assert {node["node_id"] for node in exported["nodes"]} == nodes
    assert {(link["source"], link["target"]) for link in exported["links"]} == links
    assert any(node.startswith("0_") for node in nodes), "pruning keeps some layer-0 nodes"


def test_node_fields_follow_the_reference_layout(graphs) -> None:
    graph = graphs["F3"]
    exported = _export(graph)
    by_id = {node["node_id"]: node for node in exported["nodes"]}
    for i in graph.node_indices("feature"):
        node = graph.nodes[i]
        layer = graph.transcoder_sites.index(node.site)
        got = by_id.get(f"{layer}_{node.feature_index}_{node.position}")
        if got is None:
            continue
        assert got["feature"] == (layer + node.feature_index) * (
            layer + node.feature_index + 1
        ) // 2 + (node.feature_index)
        assert (got["layer"], got["ctx_idx"]) == (str(layer), node.position)
        assert got["feature_type"] == "cross layer transcoder"
        assert got["jsNodeId"] == f"{layer}_{node.feature_index}-0"
        assert got["activation"] == graph.feature_activations[i].item()
    for i in graph.node_indices("token"):
        node = graph.nodes[i]
        got = by_id[f"E_{node.token}_{node.position}"]
        assert (got["feature"], got["layer"], got["feature_type"]) == (
            node.position,
            "E",
            "embedding",
        )
        assert got["jsNodeId"] == f"E_{node.token}-{node.position}"
    for k, i in enumerate(graph.node_indices("logit")):
        node = graph.nodes[i]
        probability = graph.logit_probabilities[k].item()
        got = by_id[f"4_{node.token}_{fx.N_POS - 1}"]
        assert (got["layer"], got["feature_type"], got["jsNodeId"]) == (
            "4",
            "logit",
            f"L_{node.token}-{fx.N_POS - 1}",
        )
        assert got["clerp"] == f'Output "<{node.token}>" (p={probability:.3f})'
        assert (got["token_prob"], got["is_target_logit"]) == (probability, k == 0)
    errors = [
        node for node in exported["nodes"] if node["feature_type"] == "mlp reconstruction error"
    ]
    for got in errors:
        layer, _minus_one, position = got["node_id"].split("_")
        assert (got["feature"], got["jsNodeId"]) == (-1, f"{layer}_-1-{position}")
    meta = exported["metadata"]
    assert meta["prompt_tokens"] == [f"<{t}>" for t in fx.TOKENS]
    assert (meta["schema_version"], meta["transcoder_list"]) == (1, [])
    assert exported["qParams"] == {
        "pinnedIds": [],
        "supernodes": [],
        "linkType": "both",
        "clickedId": "",
        "sg_pos": "",
    }


def test_path_writes_the_returned_json(graphs, tmp_path) -> None:
    path = tmp_path / "graph.json"
    exported = _export(graphs["F2"], path=path)
    assert json.loads(path.read_text(encoding="utf-8")) == exported


def test_prompt_defaults_to_the_decoded_tokens(graphs) -> None:
    graph = graphs["F1"]
    assert _export(graph)["metadata"]["prompt"] == "".join(f"<{t}>" for t in fx.TOKENS)
    assert _export(graph, prompt="three tokens")["metadata"]["prompt"] == "three tokens"


def test_bos_position_nodes_are_exported(graphs) -> None:
    """Nodes at the first position are exported whenever pruning keeps them."""

    kept_first = 0
    for name in ("F1", "F2", "F3"):
        for node_threshold, edge_threshold in ((0.8, 0.98), (0.95, 0.99), (0.99, 0.999)):
            nodes, _links = _numpy_kept(graphs[name], node_threshold, edge_threshold)
            first = {
                node
                for node in nodes
                if not node.startswith("E_") and int(node.rsplit("_", 1)[1]) == 0
            }
            exported = _export(
                graphs[name],
                node_influence_coverage=node_threshold,
                edge_influence_coverage=edge_threshold,
            )
            got = {
                node["node_id"]
                for node in exported["nodes"]
                if node["ctx_idx"] == 0 and node["feature_type"] != "embedding"
            }
            assert got == first, (name, node_threshold, edge_threshold)
            kept_first += len(first)
    assert kept_first, "some fixture and thresholds keep feature or error nodes at position 0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"node_threshold": 1.5},
        {"edge_threshold": -0.1},
        {"slug": ""},
        {"tokenizer": object()},
    ],
    ids=["node_threshold", "edge_threshold", "slug", "tokenizer"],
)
def test_refuses_invalid_arguments(graphs, kwargs) -> None:
    settings = {"slug": "fixture", "scan": "tiny", "tokenizer": TOKENIZER, **kwargs}
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        graphs["F1"].to_neuronpedia(**settings)
    assert excinfo.value.fields["code"] == "attribution_graph_export_argument_invalid"
    assert "Remedy:" in str(excinfo.value)


def _without_token_ids(graph):
    nodes = tuple(
        dataclasses.replace(node, token=None) if node.kind == "token" else node
        for node in graph.nodes
    )
    return dataclasses.replace(graph, nodes=nodes)


def _with_nan(graph):
    adjacency = graph.adjacency_matrix.clone()
    adjacency[0, -1] = float("nan")
    return dataclasses.replace(graph, adjacency_matrix=adjacency)


def _with_cycle(graph):
    adjacency = graph.adjacency_matrix.clone()
    first, second = graph.node_indices("feature")[:2]
    adjacency[first, second] = adjacency[second, first] = 1.0
    return dataclasses.replace(graph, adjacency_matrix=adjacency)


@pytest.mark.parametrize(
    "change",
    [
        _without_token_ids,
        lambda g: dataclasses.replace(g, transcoder_sites=g.transcoder_sites[:1]),
        _with_nan,
        lambda g: dataclasses.replace(
            g, logit_probabilities=torch.zeros_like(g.logit_probabilities)
        ),
        _with_cycle,
    ],
    ids=["no_token_ids", "unaligned_layers", "non_finite", "no_influence", "cycle"],
)
def test_refuses_graphs_the_schema_cannot_express(graphs, change) -> None:
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        _export(change(graphs["F1"]))
    assert excinfo.value.fields["code"] == "attribution_graph_export_unsupported"
    assert "Remedy:" in str(excinfo.value)


def _broken(exported: dict, change: str) -> dict:
    broken = json.loads(json.dumps(exported))
    if change == "no_links":
        del broken["links"]
    elif change == "slug_type":
        broken["metadata"]["slug"] = 7
    elif change == "token_type":
        broken["metadata"]["prompt_tokens"][0] = 1
    elif change == "node_field":
        broken["nodes"][0]["ctx_idx"] = "0"
    elif change == "duplicate_id":
        broken["nodes"].append(dict(broken["nodes"][0]))
    elif change == "dangling_link":
        broken["links"][0]["target"] = "nowhere"
    elif change == "bool_weight":
        broken["links"][0]["weight"] = True
    return broken


def test_contract_check_refuses_broken_exports(graphs) -> None:
    exported = _export(graphs["F1"])
    check_contract(exported)
    changes = (
        "no_links",
        "slug_type",
        "token_type",
        "node_field",
        "duplicate_id",
        "dangling_link",
        "bool_weight",
    )
    for change in changes:
        with pytest.raises(tl.attribution.AttributionError) as excinfo:
            check_contract(_broken(exported, change))
        assert excinfo.value.fields["code"] == "attribution_graph_contract_invalid", change
        assert "Remedy:" in str(excinfo.value), change

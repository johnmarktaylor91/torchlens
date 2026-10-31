"""Cross-layer transcoder graphs on a tiny Hugging Face Gemma-2 (fixture F4).

A random ReLU cross-layer transcoder reads ``hook_resid_mid`` (an op label: the residual add
entering each ``pre_feedforward_layernorm``) and writes ``hook_mlp_out`` (each
``post_feedforward_layernorm`` output) of its own layer and every later one, with
``frozen=[tl.facet("pattern"), tl.facet("inv_rms")]``. The answer key is plain autograd on
the same model (``tests/_ag3_fixtures.py``), never TorchLens.

Bars, stated before the first run (float64 unless stated): every edge within
``1e-12 * max(1, |key|)``; node counts equal the key's; every node's accounting closes within
1e-10 and the remainder agrees with the key's within 1e-10; logit nodes equal the model's
demeaned pre-softcap logits within 1e-10; each error node equals the MLP output minus every
reader's write (decoder biases included) within 1e-12; the graph from the active-feature
subset of the transcoder equals the full transcoder's graph bitwise after the feature-index
remap. Float32 build: every edge within ``1e-5 * max |key|`` of the float64 key.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import torchlens as tl
from tests import _ag3_fixtures as fx
from tests.test_ag2_gemma2_sites import VMAP_BOUNDARY

transformers = pytest.importorskip("transformers")

EDGE_REL = 1e-12
ACCOUNT_TOL = 1e-10
ERROR_TOL = 1e-12
FLOAT32_FRAC = 1e-5
FROZEN = (tl.facet("pattern"), tl.facet("inv_rms"))


def _trace(model) -> tl.Trace:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.tensor([fx.TOKENS]))
    for warning in caught:
        if issubclass(warning.category, UserWarning) and "TorchLens" in str(warning.message):
            assert VMAP_BOUNDARY in str(warning.message), str(warning.message)
    return trace


def _graph(trace: tl.Trace, readers: list[fx.CrossLayerTranscoder]):
    return tl.attribution.build_graph(
        trace,
        fx.keyed(trace, readers),
        embedding_module="model.embed_tokens",
        target=fx.TARGETS,
        logits_at=tl.attribution.SiteTarget("lm_head"),
        frozen=list(FROZEN),
    )


def _identities(trace: tl.Trace, nodes) -> list[tuple]:
    """Return each node's identity in layer terms, independent of op labels."""

    layers = {fx.read_label(trace, layer): layer for layer in range(fx.N_LAYERS)}
    return [_identity(layers, node) for node in nodes]


def _identity(layers: dict[str, int], node) -> tuple:
    if node.kind == "feature":
        return ("feature", layers[node.site], node.position, node.feature_index)
    if node.kind == "error":
        return ("error", int(node.site.split(".")[2]), node.position)
    if node.kind == "token":
        return ("token", node.position)
    return ("logit", node.token)


class _Key:
    """Edges, node values and bias terms by plain autograd (see ``tests/_ag3_fixtures.py``)."""

    def __init__(self, model, readers: list[fx.CrossLayerTranscoder]) -> None:
        self.clean = fx.clean_forward(model)
        self.readers = readers
        self.acts = [r.encode(self.clean.reads[i]) for i, r in enumerate(readers)]
        self.active = [
            (layer, p, f)
            for layer in range(fx.N_LAYERS)
            for p, f in torch.nonzero(self.acts[layer] > 0).tolist()
        ]
        self.errors = [
            self.clean.mlp_out[layer] - fx.writes_at(readers, self.clean, layer)
            for layer in range(fx.N_LAYERS)
        ]
        n_pos = len(fx.TOKENS)
        dtype = self.clean.embed.dtype
        self.c = torch.ones(len(self.active), dtype=dtype, requires_grad=True)
        self.e = torch.ones(fx.N_LAYERS, n_pos, dtype=dtype, requires_grad=True)
        self.t = torch.ones(n_pos, dtype=dtype, requires_grad=True)
        self.s = torch.ones(fx.N_LAYERS, dtype=dtype, requires_grad=True)
        with fx.detached_norms_and_patterns():
            self.reads, self.z = fx.hooked_forward(model, self._write, self._embed)
        self.rows = self._rows()

    def _write(self, layer: int, out: torch.Tensor) -> torch.Tensor:
        write = fx.WRITE.format(layer)
        rows = [torch.zeros(fx.D_MODEL, dtype=out.dtype) for _ in fx.TOKENS]
        for k, (reader, p, f) in enumerate(self.active):
            if reader <= layer:
                vector = self.acts[reader][p, f] * self.readers[reader].W_dec[write][f]
                rows[p] = rows[p] + self.c[k] * vector
        bias = sum(r.b_dec[write] for r in self.readers[: layer + 1])
        value = (
            torch.stack(rows) + self.s[layer] * bias + self.e[layer][:, None] * self.errors[layer]
        )
        return value[None]

    def _embed(self, out: torch.Tensor) -> torch.Tensor:
        return out * self.t[None, :, None]

    def _targets(self) -> list[tuple[tuple, torch.Tensor, float, float]]:
        """Return ``(identity, live target, clean value, encoder bias)`` per target node.

        The live target (on the leaf forward) gives the gradients; the node value comes from
        the clean forward, because the leaf forward rebuilds each MLP output as writes plus
        error, which rounds differently in float64 and can move a float32 cast inside a later
        norm by one unit in the last place.
        """

        out = []
        for layer, p, f in self.active:
            reader = self.readers[layer]
            value = reader.pre(self.reads[layer][p])[f]
            clean = reader.pre(self.clean.reads[layer][p])[f]
            out.append((("feature", layer, p, f), value, float(clean), float(reader.b_enc[f])))
        z, zc = self.z[-1], self.clean.logits[-1]
        for token in fx.TARGETS:
            out.append((("logit", token), z[token] - z.mean(), float(zc[token] - zc.mean()), 0.0))
        return out

    def _rows(self) -> dict[tuple, dict]:
        rows = {}
        leaves = [self.c, self.e, self.t, self.s]
        for ident, value, clean, enc_bias in self._targets():
            grads = torch.autograd.grad(value, leaves, retain_graph=True, allow_unused=True)
            c, e, t, s = (
                torch.zeros_like(leaf) if g is None else g
                for g, leaf in zip(grads, leaves, strict=True)
            )
            edges = {("feature", *a): float(c[k]) for k, a in enumerate(self.active)}
            for layer in range(fx.N_LAYERS):
                for p in range(len(fx.TOKENS)):
                    edges[("error", layer, p)] = float(e[layer, p])
            for p in range(len(fx.TOKENS)):
                edges[("token", p)] = float(t[p])
            rows[ident] = {
                "edges": edges,
                "value": clean,
                "encoder_bias": enc_bias,
                "decoder_bias": float(s.sum()),
            }
        return rows

    def matrix(self, trace: tl.Trace, graph) -> torch.Tensor:
        """Return the key adjacency in the graph's node order (missing rows are zero)."""

        ids = _identities(trace, graph.nodes)
        key = torch.zeros(len(ids), len(ids), dtype=torch.float64)
        for i, row in enumerate(ids):
            if row in self.rows:
                edges = self.rows[row]["edges"]
                key[i] = torch.tensor([edges.get(col, 0.0) for col in ids], dtype=torch.float64)
        return key


@pytest.fixture(scope="module")
def f4():
    model = fx.build_model(torch.float64)
    readers = fx.layer_transcoders(torch.float64)
    trace = _trace(model)
    graph = _graph(trace, readers)
    yield model, readers, trace, graph, _Key(model, readers)
    trace.cleanup()


def _equal_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    mask = torch.isnan(a)
    return torch.equal(mask, torch.isnan(b)) and torch.equal(a[~mask], b[~mask])


class TestCrossLayerGemma2:
    """F4: edges, counts, accounting, subset exactness and error nodes against the key."""

    def test_edges_match_the_autograd_key(self, f4) -> None:
        _model, _readers, trace, graph, key = f4
        expected = key.matrix(trace, graph)
        assert expected.abs().max().item() > 0
        gap = (graph.adjacency_matrix - expected).abs()
        bound = EDGE_REL * expected.abs().clamp(min=1.0)
        assert bool((gap <= bound).all()), (gap.max().item(), (gap / bound).max().item())
        ids = _identities(trace, graph.nodes)
        cross = [
            (i, j)
            for i, row in enumerate(ids)
            for j, col in enumerate(ids)
            if row[0] == "feature" and col[0] == "feature" and row[1] - col[1] >= 2
        ]
        assert cross and max(abs(float(graph.adjacency_matrix[i, j])) for i, j in cross) > 1e-3

    def test_node_counts_match_the_key(self, f4) -> None:
        _model, _readers, trace, graph, key = f4
        n_pos = len(fx.TOKENS)
        assert graph.node_counts == {
            "feature": len(key.active),
            "error": fx.N_LAYERS * n_pos,
            "token": n_pos,
            "logit": len(fx.TARGETS),
        }
        features = [i for i in _identities(trace, graph.nodes) if i[0] == "feature"]
        assert features == [("feature", *a) for a in key.active]

    def test_accounting_closes_and_logits_match_the_model(self, f4) -> None:
        _model, _readers, trace, graph, key = f4
        c = graph.constants
        ids = _identities(trace, graph.nodes)
        for i, node in enumerate(graph.nodes):
            if node.kind not in {"feature", "logit"}:
                continue
            row = key.rows[ids[i]]
            edges = graph.adjacency_matrix[i].sum()
            total = edges + c["encoder_bias"][i] + c["decoder_bias"][i] + c["other"][i]
            assert abs(float(total - graph.node_values[i])) <= ACCOUNT_TOL
            assert abs(float(graph.node_values[i]) - row["value"]) <= ACCOUNT_TOL
            assert abs(float(c["decoder_bias"][i]) - row["decoder_bias"]) <= ACCOUNT_TOL
            key_other = (
                row["value"]
                - sum(row["edges"].values())
                - row["encoder_bias"]
                - row["decoder_bias"]
            )
            assert abs(float(c["other"][i]) - key_other) <= ACCOUNT_TOL
        z = key.clean.logits[-1]
        logits = [i for i, node in enumerate(graph.nodes) if node.kind == "logit"]
        expected = torch.stack([z[t] - z.mean() for t in fx.TARGETS])
        assert (graph.node_values[logits] - expected).abs().max().item() <= ACCOUNT_TOL

    def test_float32_edges_match_the_float64_key(self, f4) -> None:
        _model, _readers, trace64, graph64, key = f4
        model = fx.build_model(torch.float32)
        trace = _trace(model)
        graph = _graph(trace, fx.layer_transcoders(torch.float32))
        assert _identities(trace, graph.nodes) == _identities(trace64, graph64.nodes)
        expected = key.matrix(trace64, graph64)
        gap = (graph.adjacency_matrix.double() - expected).abs().max().item()
        assert gap <= FLOAT32_FRAC * expected.abs().max().item(), gap

    def test_active_subset_builds_the_same_graph(self, f4) -> None:
        _model, readers, trace, graph, _key = f4
        idx = [
            torch.unique(
                graph.active_feature_indices[graph.active_feature_indices[:, 0] == layer, 2]
            )
            for layer in range(fx.N_LAYERS)
        ]
        assert sum(len(i) for i in idx) < fx.N_LAYERS * fx.N_FEATURES
        small = _graph(trace, [fx.subset(r, i) for r, i in zip(readers, idx, strict=True)])
        remapped = small.active_feature_indices.clone()
        remapped[:, 2] = torch.stack(
            [idx[int(r)][int(f)] for r, _p, f in small.active_feature_indices.tolist()]
        )
        assert torch.equal(remapped, graph.active_feature_indices)
        layers = {fx.read_label(trace, layer): layer for layer in range(fx.N_LAYERS)}
        assert [
            (n.kind, n.site, n.position, n.token)
            if n.feature_index is None
            else (n.kind, n.site, n.position, int(idx[layers[n.site]][n.feature_index]), n.token)
            for n in small.nodes
        ] == [
            (n.kind, n.site, n.position, n.token)
            if n.feature_index is None
            else (n.kind, n.site, n.position, n.feature_index, n.token)
            for n in graph.nodes
        ]
        assert torch.equal(small.adjacency_matrix, graph.adjacency_matrix)
        assert _equal_nan(small.node_values, graph.node_values)
        assert set(small.constants) == set(graph.constants)
        for name, value in graph.constants.items():
            assert _equal_nan(small.constants[name], value), name
        assert torch.equal(small.feature_activations, graph.feature_activations)
        assert torch.equal(small.error_vectors, graph.error_vectors)

    def test_error_nodes_are_the_unexplained_mlp_output(self, f4) -> None:
        _model, readers, _trace_, graph, key = f4
        assert graph.error_sites == tuple(fx.WRITE.format(layer) for layer in range(fx.N_LAYERS))
        for layer in range(fx.N_LAYERS):
            expected = key.clean.mlp_out[layer] - fx.writes_at(readers, key.clean, layer)
            gap = (graph.error_vectors[layer] - expected).abs().max().item()
            assert gap <= ERROR_TOL, (layer, gap)
            assert expected.abs().max().item() > 1e-2

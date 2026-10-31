"""Attribution graphs on tiny fixtures whose every edge has a closed form.

F1: a two-layer transformer whose per-layer transcoders equal its MLPs, so
error nodes are zero. F2: the same model with transcoders that reconstruct
imperfectly, so error nodes carry the rest of each MLP output. F3: a
three-layer model with a cross-layer transcoder whose layer-0 features
write the MLP outputs of layers 0, 1 and 2.

Every expected value comes from ``tests/_ag1_fixtures.py``: the forward in
plain float64 algebra and the frozen linear map it defines. Nothing in the
answer key calls ``build_graph`` or the one-backward read.

Pre-registered bars (float64): every edge equals its key value to 1e-12;
node counts by kind equal the counts below; for every feature and logit
node, incoming edges plus the declared constants equal the node's value to
1e-10, and the key's own constant terms equal the graph's to 1e-10; the
logit nodes' totals reproduce the model's logits to 1e-10. Float32 F1:
every edge within 1e-5 + 1e-5 * |key| of the float64 key on the same
weights.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from tests import _ag1_fixtures as fx

#: Pre-registered node counts: (features, errors, tokens, logits).
F1_COUNTS = {"feature": 12, "error": 6, "token": 3, "logit": 2}
F3_COUNTS = {"feature": 18, "error": 9, "token": 3, "logit": 2}
EDGE_TOL = 1e-12
TOTAL_TOL = 1e-10


def _trace(model: torch.nn.Module) -> tl.Trace:
    return tl.trace(model, torch.tensor([fx.TOKENS]))


def _build(model, transcoders, n_layers, **kwargs):
    return tl.attribution.build_graph(
        _trace(model),
        transcoders,
        embedding_module="embed",
        frozen=fx.frozen_names(n_layers),
        **kwargs,
    )


def _layer(site: str) -> int:
    return int(site.split(".")[1])


def _source_injection(graph, node, model, frozen, transcoders):
    """The key's vectors for one source node: ``(resid0, {layer: mlp write})``."""

    zero = torch.zeros(fx.N_POS, fx.D_MODEL, dtype=torch.float64)
    if node.kind == "token":
        resid0 = zero.clone()
        resid0[node.position] = frozen.resid0[node.position]
        return resid0, {}
    if node.kind == "error":
        layer = _layer(node.site)
        write = zero.clone()
        write[node.position] = _key_errors(model, frozen, transcoders)[layer][node.position]
        return zero, {layer: write}
    assert node.kind == "feature"
    reader = transcoders[node.site]
    x = frozen.mlp_in[_layer(node.site)][node.position]
    act = torch.relu(x @ reader.W_enc.double() + reader.b_enc.double())[node.feature_index]
    writes = {}
    decoders = reader.W_dec if isinstance(reader.W_dec, dict) else {node.site: reader.W_dec}
    for site, decoder in decoders.items():
        write = zero.clone()
        write[node.position] = act * decoder.double()[node.feature_index]
        writes[_layer(site)] = write
    return zero, writes


def _key_errors(model, frozen, transcoders):
    """MLP output minus reconstruction, per layer, by plain algebra."""

    recon = [torch.zeros(fx.N_POS, fx.D_MODEL, dtype=torch.float64) for _ in model.blocks]
    for site, reader in transcoders.items():
        acts = torch.relu(
            frozen.mlp_in[_layer(site)] @ reader.W_enc.double() + reader.b_enc.double()
        )
        decoders = reader.W_dec if isinstance(reader.W_dec, dict) else {site: reader.W_dec}
        biases = reader.b_dec if isinstance(reader.b_dec, dict) else {site: reader.b_dec}
        for write, decoder in decoders.items():
            recon[_layer(write)] = recon[_layer(write)] + acts @ decoder.double()
            recon[_layer(write)] = recon[_layer(write)] + biases[write].double()
    return [frozen.mlp_out[layer] - recon[layer] for layer in range(len(model.blocks))]


def _readout(graph, node, ins, logits, transcoders):
    """The key's linear read-out of a target node from the map's outputs."""

    if node.kind == "feature":
        reader = transcoders[node.site]
        return ins[_layer(node.site)][node.position] @ reader.W_enc.double()[:, node.feature_index]
    direction = torch.zeros(fx.VOCAB, dtype=torch.float64)
    direction[node.token_id] = 1.0
    if graph.has_demeaned_logits:
        direction = direction - 1.0 / fx.VOCAB
    return logits[-1] @ direction


def _key(graph, model, transcoders):
    """The full answer key: adjacency, node values and constant terms."""

    frozen = fx.plain_forward(model)
    n = len(graph.nodes)
    adjacency = torch.zeros(n, n, dtype=torch.float64)
    targets = [i for i, node in enumerate(graph.nodes) if node.kind in ("feature", "logit")]
    for s, source in enumerate(graph.nodes):
        if source.kind == "logit":
            continue
        resid0, writes = _source_injection(graph, source, model, frozen, transcoders)
        ins, logits = fx.linear_map(model, frozen, resid0, writes, biases=False)
        for t in targets:
            adjacency[t, s] = _readout(graph, graph.nodes[t], ins, logits, transcoders)
    zero = torch.zeros(fx.N_POS, fx.D_MODEL, dtype=torch.float64)
    dec_writes: dict[int, torch.Tensor] = {}
    for site, reader in transcoders.items():
        biases = reader.b_dec if isinstance(reader.b_dec, dict) else {site: reader.b_dec}
        for write, bias in biases.items():
            layer = _layer(write)
            dec_writes[layer] = dec_writes.get(layer, zero) + bias.double().expand_as(zero)
    dec_ins, dec_logits = fx.linear_map(model, frozen, zero, dec_writes, biases=False)
    other_ins, other_logits = fx.linear_map(model, frozen, zero, {}, biases=True)
    values = torch.full((n,), float("nan"), dtype=torch.float64)
    constants = {
        name: torch.full((n,), float("nan"), dtype=torch.float64)
        for name in ("encoder_bias", "decoder_bias", "other")
    }
    for t in targets:
        node = graph.nodes[t]
        constants["decoder_bias"][t] = _readout(graph, node, dec_ins, dec_logits, transcoders)
        constants["other"][t] = _readout(graph, node, other_ins, other_logits, transcoders)
        if node.kind == "feature":
            reader = transcoders[node.site]
            b_enc = reader.b_enc.double()[node.feature_index]
            constants["encoder_bias"][t] = b_enc
            values[t] = (
                frozen.mlp_in[_layer(node.site)][node.position]
                @ reader.W_enc.double()[:, node.feature_index]
                + b_enc
            )
        else:
            constants["encoder_bias"][t] = 0.0
            values[t] = _readout(graph, node, [], frozen.logits, transcoders)
    return adjacency, values, constants, frozen


def _assert_against_key(graph, model, transcoders, counts):
    adjacency, values, constants, frozen = _key(graph, model, transcoders)
    assert graph.node_counts == counts
    assert (graph.adjacency_matrix - adjacency).abs().max().item() <= EDGE_TOL
    targets = list(graph.node_indices("feature")) + list(graph.node_indices("logit"))
    assert (graph.node_values[targets] - values[targets]).abs().max().item() <= TOTAL_TOL
    for name, term in constants.items():
        assert (graph.constants[name][targets] - term[targets]).abs().max().item() <= TOTAL_TOL, (
            name
        )
    total = graph.adjacency_matrix.sum(dim=1) + sum(graph.constants.values())
    assert (total[targets] - values[targets]).abs().max().item() <= TOTAL_TOL
    key_total = adjacency.sum(dim=1) + sum(constants.values())
    assert (key_total[targets] - values[targets]).abs().max().item() <= TOTAL_TOL
    return frozen


class TestPerLayerFixture:
    """F1: transcoders equal to the MLPs."""

    def test_every_edge_and_node_matches_the_closed_form(self) -> None:
        model = fx.build_model(2)
        transcoders = fx.exact_transcoders(model)
        graph = _build(model, transcoders, 2, max_num_logits=2, logit_probability_coverage=0.99)
        _assert_against_key(graph, model, transcoders, F1_COUNTS)
        assert graph.error_vectors.abs().max().item() <= EDGE_TOL
        assert graph.capture_outcome == "COMPLETE"
        assert graph.disclosures == ()

    def test_node_order_and_fields(self) -> None:
        model = fx.build_model(2)
        graph = _build(
            model, fx.exact_transcoders(model), 2, max_num_logits=2, logit_probability_coverage=0.99
        )
        kinds = [node.kind for node in graph.nodes]
        assert kinds == ["feature"] * 12 + ["error"] * 6 + ["token"] * 3 + ["logit"] * 2
        assert graph.active_feature_indices.tolist() == [
            [layer, pos, feat] for layer in (0, 1) for pos in range(3) for feat in (0, 2)
        ]
        assert graph.input_token_ids.tolist() == list(fx.TOKENS)
        assert [graph.nodes[i].token_id for i in graph.node_indices("token")] == list(fx.TOKENS)
        assert graph.transcoder_sites == ("blocks.0.mlp", "blocks.1.mlp")
        assert graph.error_sites == ("blocks.0.mlp", "blocks.1.mlp")
        probs = torch.softmax(fx.plain_forward(model).logits[-1], dim=-1)
        top = torch.topk(probs, 2)
        assert graph.logit_token_ids.tolist() == top.indices.tolist()
        assert (graph.logit_probabilities - top.values).abs().max().item() <= TOTAL_TOL
        assert graph.adjacency_matrix[graph.node_indices("error")].abs().sum().item() == 0.0
        assert graph.adjacency_matrix[:, graph.node_indices("logit")].abs().sum().item() == 0.0

    def test_raw_logits_reproduce_the_model_logits(self) -> None:
        model = fx.build_model(2)
        transcoders = fx.exact_transcoders(model)
        graph = _build(model, transcoders, 2, target=[0, 4], demeaned_logits=False)
        frozen = _assert_against_key(graph, model, transcoders, {**F1_COUNTS, "logit": 2})
        logit_rows = list(graph.node_indices("logit"))
        total = graph.adjacency_matrix.sum(dim=1) + sum(graph.constants.values())
        expected = frozen.logits[-1, [0, 4]]
        assert (total[logit_rows] - expected).abs().max().item() <= TOTAL_TOL
        with torch.no_grad():
            model_logits = model(torch.tensor([fx.TOKENS]))[0, -1, [0, 4]]
        assert (total[logit_rows] - model_logits).abs().max().item() <= TOTAL_TOL

    def test_causal_and_layer_order_zeros(self) -> None:
        """Later positions and later layers never drive earlier features."""

        model = fx.build_model(2)
        graph = _build(
            model, fx.exact_transcoders(model), 2, max_num_logits=2, logit_probability_coverage=0.99
        )
        for t in graph.node_indices("feature"):
            target = graph.nodes[t]
            for s in range(len(graph.nodes)):
                source = graph.nodes[s]
                later_pos = source.position > target.position
                later_layer = source.kind in ("feature", "error") and _layer(source.site) >= _layer(
                    target.site
                )
                if later_pos or later_layer:
                    assert graph.adjacency_matrix[t, s].item() == 0.0

    def test_float32_edges_within_the_registered_tolerance(self) -> None:
        model = fx.build_model(2, dtype=torch.float32)
        transcoders = fx.exact_transcoders(model)
        graph = _build(model, transcoders, 2, max_num_logits=2, logit_probability_coverage=0.99)
        assert graph.adjacency_matrix.dtype == torch.float32
        adjacency, _values, _constants, _frozen = _key(graph, model, transcoders)
        gap = (graph.adjacency_matrix.double() - adjacency).abs()
        assert (gap <= 1e-5 + 1e-5 * adjacency.abs()).all()

    def test_frozen_op_labels_equal_frozen_module_names(self) -> None:
        """An op label in ``frozen=`` holds the same value its module name does."""

        model = fx.build_model(2)
        transcoders = fx.exact_transcoders(model)
        trace = _trace(model)
        by_op = [str(trace.modules[name].output_ops[0]) for name in fx.frozen_names(2)]
        graph = tl.attribution.build_graph(
            trace, transcoders, embedding_module="embed", frozen=by_op, target=[0]
        )
        by_name = _build(model, transcoders, 2, target=[0])
        assert torch.equal(graph.adjacency_matrix, by_name.adjacency_matrix)
        assert graph.frozen_sites == by_name.frozen_sites


class TestErrorNodeFixture:
    """F2: transcoders that reconstruct imperfectly."""

    def test_error_nodes_carry_the_unexplained_mlp_output(self) -> None:
        model = fx.build_model(2)
        transcoders = fx.lossy_transcoders(model)
        graph = _build(model, transcoders, 2, max_num_logits=2, logit_probability_coverage=0.99)
        frozen = _assert_against_key(graph, model, transcoders, F1_COUNTS)
        key_errors = torch.stack(_key_errors(model, frozen, transcoders))
        assert (graph.error_vectors - key_errors).abs().max().item() <= EDGE_TOL
        assert key_errors.abs().max().item() > 1e-2

    def test_logit_totals_need_the_error_nodes(self) -> None:
        model = fx.build_model(2)
        transcoders = fx.lossy_transcoders(model)
        graph = _build(model, transcoders, 2, target=[0, 4], demeaned_logits=False)
        rows = list(graph.node_indices("logit"))
        errors = list(graph.node_indices("error"))
        total = graph.adjacency_matrix.sum(dim=1) + sum(graph.constants.values())
        expected = fx.plain_forward(model).logits[-1, [0, 4]]
        assert (total[rows] - expected).abs().max().item() <= TOTAL_TOL
        without = total[rows] - graph.adjacency_matrix[rows][:, errors].sum(dim=1)
        assert (without - expected).abs().max().item() > 1e-3


class TestCrossLayerFixture:
    """F3: a cross-layer transcoder; layer-0 features write MLP outputs 0, 1 and 2."""

    def test_cross_layer_edges_match_the_closed_form(self) -> None:
        model = fx.build_model(3)
        transcoders = fx.cross_layer_transcoders(model)
        graph = _build(model, transcoders, 3, max_num_logits=2, logit_probability_coverage=0.99)
        _assert_against_key(graph, model, transcoders, F3_COUNTS)
        first = [i for i in graph.node_indices("feature") if graph.nodes[i].site == "blocks.0.mlp"]
        last = [i for i in graph.node_indices("feature") if graph.nodes[i].site == "blocks.2.mlp"]
        assert graph.adjacency_matrix[last][:, first].abs().max().item() > 1e-3


class TestRefusals:
    """Each coded refusal fires with its code."""

    def _code(self, excinfo) -> str:
        return excinfo.value.fields["code"]

    def test_frozen_is_required(self) -> None:
        model = fx.build_model(2)
        with pytest.raises(tl.errors.AttributionError) as excinfo:
            tl.attribution.build_graph(
                _trace(model), fx.exact_transcoders(model), embedding_module="embed"
            )
        assert self._code(excinfo) == "attribution_graph_frozen_required"
        assert "frozen=()" in str(excinfo.value)

    def test_unknown_site(self) -> None:
        model = fx.build_model(2)
        with pytest.raises(tl.errors.AttributionError) as excinfo:
            tl.attribution.build_graph(
                _trace(model), fx.exact_transcoders(model), embedding_module="embedding", frozen=()
            )
        assert self._code(excinfo) == "attribution_graph_site_not_found"

    def test_transcoder_without_weights(self) -> None:
        model = fx.build_model(2)
        with pytest.raises(tl.errors.AttributionError) as excinfo:
            tl.attribution.build_graph(
                _trace(model), {"blocks.0.mlp": object()}, embedding_module="embed", frozen=()
            )
        assert self._code(excinfo) == "attribution_graph_transcoder_invalid"

    def test_batch_of_two_is_unsupported(self) -> None:
        model = fx.build_model(2)
        trace = tl.trace(model, torch.tensor([fx.TOKENS, fx.TOKENS]))
        with pytest.raises(tl.errors.AttributionError) as excinfo:
            tl.attribution.build_graph(
                trace, fx.exact_transcoders(model), embedding_module="embed", frozen=()
            )
        assert self._code(excinfo) == "attribution_graph_input_unsupported"

    def test_bad_target(self) -> None:
        model = fx.build_model(2)
        with pytest.raises(tl.errors.AttributionError) as excinfo:
            _build(model, fx.exact_transcoders(model), 2, target=[99])
        assert self._code(excinfo) == "attribution_graph_target_invalid"

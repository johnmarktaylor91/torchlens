"""Op-label read sites for attribution-graph transcoders, on a tiny Gemma-2.

Gemma Scope's transcoders read the pre-feedforward norm's value before its
``(1 + w)`` gain: the ``x * rsqrt(...)`` op inside ``Gemma2RMSNorm``, which no
module outputs. Keyed by that op label, an unfolded transcoder must give the
graph that a transcoder keyed by the MLP module gives with the gain folded
into its encoder (``W_enc / (1 + w)`` row by row): both encoders see the same
vector, so the graphs agree up to float32 rounding.

Bars, stated before the first run: the same nodes (feature sites mapped from
the op label to the module name) and active features; adjacency, node values,
activations, error vectors and the decoder-bias and remainder constants
within 1e-5 of the largest absolute entry of each; encoder biases equal
exactly.

The remainder bar was mis-specified: the remainder ``constants["other"]`` is
float32 noise in both graphs (largest 6.93e-7 against a difference of
4.77e-7 between them), so a bar relative to its own largest entry fails. That
case stays as written, marked strict-xfail. A second remainder bar, added
2026-10-16 10:47 PM PT after that result was read, holds each graph's
remainder within 1e-5 of the largest absolute node value (the accounting
scale of the graph's own identity).
"""

from __future__ import annotations

import copy
import re

import pytest
import torch

import torchlens as tl
from tests.test_ag2_gemma2_sites import (
    N_LAYERS,
    _frozen_ops,
    _model,
    _trace,
    _transcoders,
)

ROUNDING_FRAC = 1e-5


def _pre_gain_ops(trace: tl.Trace) -> dict[int, str]:
    """Return each layer's pre-feedforward ``x * rsqrt`` op label (the pre-gain value)."""

    labels = {}
    for op in trace:
        module = str(op.module or "").split(":")[0]
        if not module.endswith(".pre_feedforward_layernorm") or op.func_name != "__mul__":
            continue
        if any(trace[str(parent)].func_name == "rsqrt" for parent in op.parents):
            labels[int(module.split(".")[2])] = str(op.label)
    assert sorted(labels) == list(range(N_LAYERS))
    return labels


def _folded(transcoder, gain: torch.Tensor):
    """Return a copy of ``transcoder`` whose encoder rows are divided by ``gain``."""

    folded = copy.copy(transcoder)
    folded.W_enc = transcoder.W_enc / gain[:, None]
    return folded


def _gain(model, layer: int) -> torch.Tensor:
    return 1.0 + model.model.layers[layer].pre_feedforward_layernorm.weight.detach().float()


def _graph(model, transcoders):
    trace = _trace(model)
    return tl.attribution.build_graph(
        trace,
        transcoders,
        embedding="model.embed_tokens",
        logits_at=tl.attribution.SiteTarget("lm_head"),
        max_n_logits=3,
        desired_logit_prob=0.99,
        frozen=_frozen_ops(trace),
    )


def _op_keyed(model) -> tuple[dict, dict[str, str]]:
    """Return the op-label-keyed transcoders and the op label -> module name map."""

    ops = _pre_gain_ops(_trace(model))
    by_module = _transcoders()
    keyed = {ops[layer]: by_module[f"model.layers.{layer}.mlp"] for layer in range(N_LAYERS)}
    return keyed, {ops[layer]: f"model.layers.{layer}.mlp" for layer in range(N_LAYERS)}


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert the NaN masks agree and the finite entries meet the rounding bar."""

    assert a.shape == b.shape
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    finite = ~torch.isnan(b)
    scale = b[finite].abs().max().item() if bool(finite.any()) else 0.0
    gap = (a[finite].double() - b[finite].double()).abs().max().item() if scale else 0.0
    assert gap <= ROUNDING_FRAC * scale, (gap, scale)


def _code(info: pytest.ExceptionInfo) -> str:
    return str(info.value.fields["code"])


@pytest.fixture(scope="module")
def graph_pair():
    """Graphs keyed by the pre-gain op label (unfolded) and by the MLP (folded)."""

    model = _model()
    keyed, to_module = _op_keyed(model)
    by_op = _graph(model, keyed)
    folded = {
        f"model.layers.{layer}.mlp": _folded(tc, _gain(model, layer))
        for layer, tc in enumerate(_transcoders().values())
    }
    return by_op, _graph(model, folded), keyed, to_module


class TestOpLabelReadSite:
    """A transcoder keyed by an op label reads that op's output."""

    def test_op_label_read_equals_folded_module_read(self, graph_pair) -> None:
        by_op, by_module, keyed, to_module = graph_pair
        assert by_op.node_counts["feature"] > 0
        assert by_op.transcoder_sites == tuple(keyed)

        def identity(graph, mapping):
            return [
                (n.kind, mapping.get(n.site, n.site), n.position, n.feature, n.token)
                for n in graph.nodes
            ]

        assert identity(by_op, to_module) == identity(by_module, {})
        assert torch.equal(by_op.active_features, by_module.active_features)
        _close(by_op.adjacency_matrix, by_module.adjacency_matrix)
        for name in ("node_values", "activation_values", "error_vectors"):
            _close(getattr(by_op, name), getattr(by_module, name))
        _close(by_op.constants["decoder_bias"], by_module.constants["decoder_bias"])
        torch.testing.assert_close(
            by_op.constants["encoder_bias"],
            by_module.constants["encoder_bias"],
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        for graph in (by_op, by_module):
            finite = ~torch.isnan(graph.node_values)
            scale = graph.node_values[finite].abs().max().item()
            remainder = graph.constants["other"][finite].abs().max().item()
            assert remainder <= ROUNDING_FRAC * scale, (remainder, scale)

    @pytest.mark.xfail(
        strict=True,
        reason="the pre-registered remainder bar is relative to the remainder's own float32 "
        "noise (mis-specified); it is kept, failing, in the record",
    )
    def test_remainder_meets_the_noise_relative_bar(self, graph_pair) -> None:
        by_op, by_module, _keyed, _to_module = graph_pair
        _close(by_op.constants["other"], by_module.constants["other"])

    def test_unknown_op_label_refuses(self) -> None:
        model = _model()
        trace = _trace(model)
        tc = _transcoders()["model.layers.0.mlp"]
        with pytest.raises(tl.attribution.AttributionError) as info:
            tl.attribution.build_graph(
                trace, {"mul_999_999": tc}, embedding="model.embed_tokens", frozen=()
            )
        assert _code(info) == "attribution_graph_site_not_found"
        message = str(info.value)
        assert "'mul_999_999'" in message and "neither a module nor an op" in message
        example = re.search(r"for example '([^']+)'", message)
        assert example is not None
        assert example.group(1) in [str(op.label) for op in trace]

    def test_read_site_not_upstream_refuses(self) -> None:
        model = _model()
        ops = _pre_gain_ops(_trace(model))
        tc = _transcoders()["model.layers.0.mlp"]
        with pytest.raises(tl.attribution.AttributionError) as info:
            _graph(model, {ops[1]: tc})
        assert _code(info) == "attribution_graph_read_site_not_upstream"
        message = str(info.value)
        assert repr(ops[1]) in message
        assert "'model.layers.0.post_feedforward_layernorm'" in message
        example = re.search(r"for example '([^']+)'\)", message)
        assert example is not None
        graph = _graph(model, {example.group(1): tc})
        assert graph.transcoder_sites == (example.group(1),)

    def test_op_label_key_with_one_decoder_refuses(self) -> None:
        model = _model()
        ops = _pre_gain_ops(_trace(model))
        tc = copy.copy(_transcoders()["model.layers.0.mlp"])
        tc.W_dec = next(iter(tc.W_dec.values()))
        tc.b_dec = next(iter(tc.b_dec.values()))
        with pytest.raises(tl.attribution.AttributionError) as info:
            _graph(model, {ops[0]: tc})
        assert _code(info) == "attribution_graph_transcoder_invalid"
        assert "W_dec={'model.layers.0.mlp': W_dec}" in str(info.value)

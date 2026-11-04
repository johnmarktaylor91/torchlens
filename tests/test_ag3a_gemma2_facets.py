"""Facet-named frozen sets for attribution graphs on a tiny Hugging Face Gemma-2.

``Gemma2Attention`` serves the ``pattern`` facet (the eager softmax, after the
score softcap and the mask) and ``Gemma2RMSNorm`` serves ``inv_rms`` (its
``rsqrt`` op). ``frozen=[tl.facet("pattern"), tl.facet("inv_rms")]`` must
resolve to exactly the op labels AG-2 froze by hand and build the identical
graph: the same ops are held, so the bar is bitwise equality.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from tests.test_ag2_gemma2_sites import _frozen_ops, _model, _trace, _transcoders

NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)


def _home(trace: tl.Trace, module: str, facet: str) -> str:
    return str(trace.modules[module].facets[facet].spec.home_label)


def _held_by_hand(trace: tl.Trace, layers: int) -> list[str]:
    """AG-2's hand-found frozen set, checked against the config's structure.

    Found by op function and module, never by literal label: op labels count
    the ops before them, and transformers 4.x and 5.x build Gemma-2's masks
    with different op counts, so the same softmax is ``softmax_1_91`` on one
    and ``softmax_1_104`` on the other.
    """

    held = _frozen_ops(trace)
    funcs = {str(op.label): op.func_name for op in trace}
    kinds = [funcs[label] for label in held]
    assert held and len(set(held)) == len(held)
    assert kinds.count("softmax") == layers  # one eager pattern per attention
    assert kinds.count("rsqrt") == len(NORMS) * layers + 1  # four norms a layer, the final norm
    return held


def _graph(model, frozen):
    trace = _trace(model)
    return tl.attribution.build_graph(
        trace,
        _transcoders(),
        embedding_module="model.embed_tokens",
        logits_at=tl.attribution.SiteTarget("lm_head"),
        max_num_logits=3,
        logit_probability_coverage=0.99,
        frozen=frozen(trace) if callable(frozen) else frozen,
    )


def _equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))


class TestGemma2Facets:
    """``pattern`` and ``inv_rms`` name the ops an attribution graph holds."""

    def test_frozen_facets_build_the_same_graph(self) -> None:
        model = _model()
        by_facet = _graph(model, [tl.facet("pattern"), tl.facet("inv_rms")])
        layers = model.config.num_hidden_layers
        by_label = _graph(model, lambda trace: _held_by_hand(trace, layers))
        assert by_facet.nodes == by_label.nodes
        assert set(by_facet.frozen_sites) == set(by_label.frozen_sites)
        assert torch.equal(by_facet.active_feature_indices, by_label.active_feature_indices)
        for name in ("adjacency_matrix", "node_values", "feature_activations", "error_vectors"):
            _equal(getattr(by_facet, name), getattr(by_label, name))
        assert set(by_facet.constants) == set(by_label.constants)
        for name, value in by_facet.constants.items():
            _equal(value, by_label.constants[name])

    def test_facet_that_is_not_one_op_refuses(self) -> None:
        with pytest.raises(tl.errors.AttributionError) as info:
            _graph(_model(), [tl.facet("result")])
        assert info.value.fields["code"] == "attribution_graph_frozen_facet_unsupported"
        assert "tl.facet('result')" in str(info.value)

    def test_facet_no_module_serves_refuses(self) -> None:
        with pytest.raises(tl.errors.AttributionError) as info:
            _graph(_model(), [tl.facet("no_such_facet")])
        assert info.value.fields["code"] == "attribution_graph_site_not_found"
        assert "trace.modules[name].facets.keys()" in str(info.value)

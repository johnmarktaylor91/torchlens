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

#: The tiny Gemma-2's attention patterns (one per layer) and norm scales (four
#: per layer plus the final norm), as captured. Grouping rule rows 1 and 17: the
#: mask builder's one call line (masking_utils.py:522) runs twice, so its mask ops
#: are 2-pass layers; row 14: the rotary ``.float()`` pair on two source lines
#: (modeling_gemma2.py:127 and :131) is two single layers. Both shift these labels.
PATTERN_OPS = ("softmax_1_104:1", "softmax_2_198:1")
INV_RMS_OPS = (
    "rsqrt_1_54:1",
    "rsqrt_2_117:1",
    "rsqrt_3_127:1",
    "rsqrt_4_141:1",
    "rsqrt_5_151:1",
    "rsqrt_6_211:1",
    "rsqrt_7_221:1",
    "rsqrt_8_235:1",
    "rsqrt_9_245:1",
)
NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)


def _home(trace: tl.Trace, module: str, facet: str) -> str:
    return str(trace.modules[module].facets[facet].spec.home_label)


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

    def test_facets_resolve_to_the_frozen_op_labels(self) -> None:
        model = _model()
        assert list(model.config.layer_types) == ["sliding_attention", "full_attention"]
        assert model.config.attn_logit_softcapping == 50.0
        trace = _trace(model)
        patterns = tuple(_home(trace, f"model.layers.{i}.self_attn", "pattern") for i in (0, 1))
        norms = [f"model.layers.{i}.{norm}" for i in (0, 1) for norm in NORMS] + ["model.norm"]
        scales = tuple(_home(trace, name, "inv_rms") for name in norms)
        assert patterns == PATTERN_OPS
        assert sorted(scales) == sorted(INV_RMS_OPS)
        assert sorted(patterns + scales) == sorted(_frozen_ops(trace))
        graph = _graph(model, [tl.facet("pattern"), tl.facet("inv_rms")])
        held = set(graph.frozen_sites) - set(PATTERN_OPS) - set(INV_RMS_OPS)
        assert len(held) == 2  # the two written MLP outputs
        assert set(PATTERN_OPS + INV_RMS_OPS) <= set(graph.frozen_sites)

    def test_frozen_facets_build_the_same_graph(self) -> None:
        model = _model()
        by_facet = _graph(model, [tl.facet("pattern"), tl.facet("inv_rms")])
        by_label = _graph(model, list(PATTERN_OPS + INV_RMS_OPS))
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

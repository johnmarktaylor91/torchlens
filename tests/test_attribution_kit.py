"""L6 stage-4b attribution-kit behavior and detachability gates."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
from _source_corpus import module_ast
from torch import Tensor, nn

import torchlens as tl
from torchlens.attribution import occlusion
from torchlens.attribution._viz import attribution_color_by

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ATTRIBUTION_PREFIX = "torchlens.attribution"
_DOCUMENTED_UNSTABLE_TOKENS = {
    "alpha",
    "attribution_sum",
    "baseline",
    "bilinear_display_only",
    "caller",
    "derived",
    "feature_map",
    "grid_evidence",
    "grid_route",
    "layer_reason",
    "layer_source",
    "prefix_tokens",
    "reshape",
    "trace",
    "blur",
    "blur_kernel_size",
    "cmap",
    "completeness_residual",
    "grad_cam",
    "image",
    "integrated_gradients",
    "layer",
    "mean",
    "abs_mean",
    "abs_sum",
    "max",
    "num_steps",
    "native_map_resolution",
    "occluded_score",
    "occlusion",
    "original_score",
    "overlay",
    "reduce",
    "relu",
    "rendered_map_resolution",
    "score",
    "selection",
    "selection_digest",
    "source",
    "sum",
    "target",
    "target_delta",
    "upsampling",
    "zeros",
    "AttributionMetric",
    "SiteStash",
    "SiteTarget",
    "unit",
    "call",
    "integer_input_layers",
    "TokenAttributionPayload",
    "TokenAttribution",
    "absolute",
    "attribute",
    "auto",
    "average",
    "baseline_value",
    "baselines",
    "completeness_caveat",
    "converged",
    "deconvolution",
    "draw_bank",
    "fetch",
    "gaussian",
    "gradient_shap",
    "guided_backprop",
    "infidelity",
    "keep_special_tokens",
    "label_of",
    "leftovers",
    "mark_firing",
    "max_length",
    "max_passes",
    "mean_square",
    "method",
    "method_kwargs",
    "num_samples",
    "noise_bank",
    "noise_std",
    "noise_tunnel",
    "occlude_leaf",
    "occlusion_map",
    "pad_token",
    "path_evaluations_logical",
    "per_call",
    "per_chunk",
    "perturb",
    "physical_forward_calls",
    "radius",
    "register_site",
    "residual_of_means",
    "residual_rel",
    "sensitivity",
    "show",
    "site_census",
    "sites",
    "square_removal",
    "stash",
    "stdevs",
    "stdevs_resolved",
    "step_audit",
    "step_audit_seed",
    "step_batch_size",
    "steps_per_batch",
    "store_draws",
    "strides",
    "target_delta_abs",
    "text",
    "variance",
    "window",
    "deletion_curve",
    "insertion_curve",
    "PerturbationCurve",
    "CurveStep",
    "fractions",
    "scores",
    "area",
    "aopc",
    "steps",
    "features",
    "patch_size",
    "elements",
    "pixels",
    "patches",
    "tokens",
    "num_features",
    "index",
    "fraction",
    "inputs",
    "input_kwargs",
    "build_baseline_mask",
    "intervention",
    "randomization_test",
    "RandomizationTest",
    "RandomizationStep",
    "randomization",
    "cascading",
    "independent",
    "layers",
    "seed",
    "keep_values",
    "original_attribution",
    "randomized_module_addresses",
    "similarity",
    "similarities",
    "spearman",
    "spearman_abs",
    "ssim",
    "values",
    "causal_audit",
    "CausalAudit",
    "CausalAuditUnit",
    "CausalAuditSet",
    "claim",
    "k",
    "edit",
    "zero",
    "controls",
    "top",
    "bottom",
    "random",
    "num_random_draws",
    "per_unit",
    "site",
    "units",
    "sets",
    "rank_agreement",
    "set_of",
    "claimed_effect",
    "measured_effect",
    "kind",
    "draw",
    "component",
    "claim_kind",
    "constant_row",
    "rows_not_audited",
    "corrupted_input",
    "metric",
    "edges",
    "edge_table",
    "edge_patch",
    "example_reduction",
    "logits_site",
    "sets_not_measured",
    "reason",
    "road_curve",
    "ranking",
    "most_relevant_first",
    "least_relevant_first",
    "road",
    "AttributionGraph",
    "GraphNode",
    "feature_activations",
    "active_feature_indices",
    "adjacency_matrix",
    "build_graph",
    "capture_outcome",
    "constants",
    "decoder_bias",
    "demeaned_logits",
    "logit_probability_coverage",
    "disclosures",
    "embedding_module",
    "encoder_bias",
    "error",
    "error_sites",
    "error_vectors",
    "feature_index",
    "frozen",
    "input_token_ids",
    "logit",
    "logit_probabilities",
    "logit_token_ids",
    "logits_at",
    "max_num_logits",
    "node_counts",
    "node_indices",
    "node_values",
    "nodes",
    "other",
    "position",
    "token",
    "transcoder_sites",
    "transcoders",
    "edge_influence_coverage",
    "node_influence_coverage",
    "path",
    "prompt",
    "model_id",
    "slug",
    "to_neuronpedia",
    "tokenizer",
    "activation",
    "api_key",
    "influence",
    "upload_to_neuronpedia",
}


class _ImageScale(nn.Module):
    """Tiny spatial model whose occlusion deltas are easy to inspect."""

    def forward(self, value: Tensor) -> Tensor:
        """Double every input element."""

        return value * 2.0


def _output_sum(trace: Any) -> Tensor:
    """Return the single captured output tensor's sum."""

    return trace.output_ops[0].out.sum()


def test_occlusion_routes_named_baselines_through_selection_do() -> None:
    """Zeros, mean, and blur are explicit and produce disclosed scored deltas."""

    inputs = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    trace = tl.trace(
        _ImageScale(),
        inputs,
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        selection = tl.units(trace.input_ops[0].label, [(0, 0, 1, 1)])
        results = {
            baseline: occlusion(trace, selection, score=_output_sum, baseline=baseline)
            for baseline in ("zeros", "mean", "blur")
        }
        assert {result.extra["baseline"] for result in results.values()} == {
            "zeros",
            "mean",
            "blur",
        }
        assert results["zeros"].values.item() == pytest.approx(10.0)
        assert results["mean"].values.item() == pytest.approx(0.0)
        assert results["blur"].values.item() == pytest.approx(0.0)
        assert results["blur"].extra["blur_kernel_size"] == 3
        assert results["zeros"].extra["blur_kernel_size"] is None
        assert all(result.extra["selection_digest"] for result in results.values())
    finally:
        trace.cleanup()


def test_occlusion_rejects_an_implicit_or_invalid_baseline() -> None:
    """The baseline vocabulary is closed instead of silently choosing a fill."""

    inputs = torch.ones(1, 1, 3, 3)
    trace = tl.trace(
        _ImageScale(),
        inputs,
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        selection = trace.input_ops[0]
        with pytest.raises(ValueError, match="baseline must be"):
            occlusion(trace, selection, score=_output_sum, baseline="implicit")
    finally:
        trace.cleanup()


def _imported_modules(path: Path) -> set[str]:
    """Return absolute import targets in one Python source file."""

    tree = module_ast(path)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _overlay_model() -> nn.Sequential:
    """Return the small CNN used by the overlay bridge tests."""

    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 3),
    )


def test_overlay_paints_attributed_module_outputs_and_nothing_else() -> None:
    """Layer-scoped results color their module-output nodes; others stay None."""

    model = _overlay_model()
    inputs = torch.randn(2, 3, 8, 8)
    result = tl.attribution.layer_attribution(model, inputs, target=1, layer="2")
    trace = tl.trace(model, inputs)
    try:
        color_by = attribution_color_by(trace, result)
        expected = result.values.abs().sum().item()
        assert color_by(trace["conv2d_2_3"]) == pytest.approx(expected)
        assert color_by(trace["relu_1_2"]) is None
        assert color_by(trace[trace.input_layers[0]]) is None
        assert color_by(trace[trace.output_layers[0]]) is None
    finally:
        trace.cleanup()


def test_overlay_accepts_mappings_and_refuses_unknown_keys() -> None:
    """Mapping keys resolve by module name or layer label; misses refuse typed."""

    model = _overlay_model()
    inputs = torch.randn(2, 3, 8, 8)
    trace = tl.trace(model, inputs)
    try:
        color_by = attribution_color_by(trace, {"2": torch.tensor([[1.0, -3.0]]), "relu_1_2": 0.5})
        assert color_by(trace["conv2d_2_3"]) == pytest.approx(4.0)
        assert color_by(trace["relu_1_2"]) == pytest.approx(0.5)
        with pytest.raises(tl.errors.AttributionError, match="matches no module"):
            attribution_color_by(trace, {"not_a_layer": 1.0})
        with pytest.raises(tl.errors.AttributionError, match="no extra\\['layer'\\]"):
            attribution_color_by(
                trace,
                [tl.attribution.saliency(model, inputs, target=0)],
            )
    finally:
        trace.cleanup()

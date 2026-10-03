"""L6 stage-4b attribution-kit behavior and detachability gates."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from _source_corpus import module_ast, package_files
from torch import Tensor, nn

import torchlens as tl
from torchlens.attribution import occlusion

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ATTRIBUTION_PREFIX = "torchlens.attribution"
_DOCUMENTED_UNSTABLE_TOKENS = {
    "alpha",
    "attribution_sum",
    "baseline",
    "bilinear_display_only",
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
    "n_steps",
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
    "MetricResult",
    "SiteStash",
    "TokenAttributionPayload",
    "TokenAttributionResult",
    "absolute",
    "aggregation",
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
    "n_samples",
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
}


class _ImageScale(nn.Module):
    """Tiny spatial model whose occlusion deltas are easy to inspect."""

    def forward(self, value: Tensor) -> Tensor:
        """Double every input element."""

        return value * 2.0


def test_documented_unstable_attribution_surface_matches_glossary_index() -> None:
    """Every stage-4b spelling carries the exact no-shim glossary marker."""

    glossary = (_REPO_ROOT / "docs/reference/glossary.md").read_text(encoding="utf-8")
    indexed = glossary.split("<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:START -->", 1)[1].split(
        "<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:END -->", 1
    )[0]
    assert set(re.findall(r"`([^`]+)`", indexed)) == _DOCUMENTED_UNSTABLE_TOKENS
    surface_rows = [line for line in indexed.splitlines() if line.startswith("|")][2:]
    assert surface_rows
    assert all("unstable -- no deprecation shim owed" in row for row in surface_rows)


def _output_sum(trace: Any) -> Tensor:
    """Return the single captured output tensor's sum."""

    return trace.output_ops[0].out.sum()


def test_occlusion_routes_named_baselines_through_selection_do() -> None:
    """Zeros, mean, and blur are explicit and produce disclosed scored deltas."""

    inputs = torch.arange(1.0, 10.0).reshape(1, 1, 3, 3)
    trace = tl.trace(
        _ImageScale(),
        inputs,
        capture=tl.options.CaptureOptions(intervention_ready=True),
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
        capture=tl.options.CaptureOptions(intervention_ready=True),
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


@pytest.mark.heavy
def test_attribution_kit_lane_is_detachable_from_stage4a_and_capture_core() -> None:
    """No non-attribution package module or stage-4a file imports the shed lane."""

    package_offenders = {
        str(path.relative_to(_REPO_ROOT)): sorted(_imported_modules(path))
        for path in package_files()
        if "attribution" not in path.relative_to(_REPO_ROOT / "torchlens").parts
        and any(
            name == _ATTRIBUTION_PREFIX or name.startswith(f"{_ATTRIBUTION_PREFIX}.")
            for name in _imported_modules(path)
        )
    }
    stage4a_files = (
        "torchlens/selection.py",
        "torchlens/_selection_align.py",
        "tests/test_selection_algebra.py",
        "tests/test_selection_do.py",
        "tests/test_selection_align.py",
        "tests/test_selection_gallery.py",
        "tests/test_dna_canary.py",
    )
    stage4a_offenders = {
        relative: sorted(_imported_modules(_REPO_ROOT / relative))
        for relative in stage4a_files
        if any(
            name == _ATTRIBUTION_PREFIX or name.startswith(f"{_ATTRIBUTION_PREFIX}.")
            for name in _imported_modules(_REPO_ROOT / relative)
        )
    }
    assert not package_offenders
    assert not stage4a_offenders


def test_attribution_detachability_scanner_is_red_capable(tmp_path: Path) -> None:
    """The detachability scanner detects a planted direct import."""

    coupled = tmp_path / "coupled.py"
    coupled.write_text("from torchlens.attribution import occlusion\n", encoding="utf-8")
    assert _ATTRIBUTION_PREFIX in _imported_modules(coupled)


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


@pytest.mark.smoke
def test_overlay_paints_attributed_module_outputs_and_nothing_else() -> None:
    """Layer-scoped results color their module-output nodes; others stay None."""

    model = _overlay_model()
    inputs = torch.randn(2, 3, 8, 8)
    result = tl.attribution.layer_attribution(model, inputs, target=1, layer="2")
    trace = tl.trace(model, inputs)
    try:
        color_by = tl.attribution.overlay(trace, result)
        expected = result.values.abs().sum().item()
        assert color_by(trace["conv2d_2_3"]) == pytest.approx(expected)
        assert color_by(trace["relu_1_2"]) is None
        assert color_by(trace[trace.input_layers[0]]) is None
        assert color_by(trace[trace.output_layers[0]]) is None
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_overlay_accepts_mappings_reduces_and_refuses_unknown_keys() -> None:
    """Mapping keys resolve by module name or layer label; misses refuse typed."""

    model = _overlay_model()
    inputs = torch.randn(2, 3, 8, 8)
    trace = tl.trace(model, inputs)
    try:
        color_by = tl.attribution.overlay(
            trace, {"2": torch.tensor([[1.0, -3.0]]), "relu_1_2": 0.5}, reduce="max"
        )
        assert color_by(trace["conv2d_2_3"]) == pytest.approx(1.0)
        assert color_by(trace["relu_1_2"]) == pytest.approx(0.5)
        with pytest.raises(tl.attribution.AttributionError, match="matches no module"):
            tl.attribution.overlay(trace, {"not_a_layer": 1.0})
        with pytest.raises(tl.attribution.AttributionError, match="reduce must be"):
            tl.attribution.overlay(trace, {"2": 1.0}, reduce="median")
        with pytest.raises(tl.attribution.AttributionError, match="no extra\\['layer'\\]"):
            tl.attribution.overlay(
                trace,
                [tl.attribution.saliency(model, inputs, target=0)],
            )
    finally:
        trace.cleanup()

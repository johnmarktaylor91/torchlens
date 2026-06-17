"""Tests for the TorchLens 2.0 top-level API budget."""

from __future__ import annotations

import importlib
import warnings

import torchlens

TARGET_ALL = [
    "trace",
    "export",
    "fastlog",
    "facets",
    "record",
    "Recording",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "load",
    "save",
    "do",
    "push",
    "push_from",
    "replay",
    "replay_from",
    "rerun",
    "run",
    "bundle",
    "pluck",
    "peek",
    "extract",
    "extract_dataset",
    "batched_extract",
    "validate",
    "AmbiguousOpLookupError",
    "ReentrantTraceError",
    "Trace",
    "Layer",
    "Container",
    "Op",
    "ModelHistory",
    "Quantity",
    "Bytes",
    "Duration",
    "Flops",
    "Macs",
    "Bundle",
    "add",
    "label",
    "func",
    "func_transform",
    "followed_by",
    "grad_fn",
    "intervening",
    "without_op",
    "regex",
    "module",
    "output",
    "output_at",
    "input_at",
    "register_container",
    "preceded_by",
    "contains",
    "facet",
    "where",
    "in_module",
    "head",
    "clamp",
    "mean_ablate",
    "noise",
    "project_off",
    "project_onto",
    "replace_with",
    "resample_ablate",
    "scale",
    "splice_module",
    "span",
    "steer",
    "sweep",
    "swap_with",
    "zero_ablate",
    "when",
    "bwd_hook",
    "grad_clip",
    "grad_noise",
    "grad_clamp",
    "grad_scale",
    "grad_zero",
    "tap",
    "record_span",
    "log_forward_pass",
    "get_model_activations",
    "validate_model_activations",
    "validate_saved_activations",
    "render_graph",
    "render_model_graph",
    "draw_model_graph",
    "get_model_structure",
    "show_model_structure",
]

CANONICAL_SUBMODULES = [
    "torchlens.accessors",
    "torchlens.bridge",
    "torchlens.callbacks",
    "torchlens.compat",
    "torchlens.errors",
    "torchlens.examples",
    "torchlens.experimental",
    "torchlens.experimental.dagua",
    "torchlens.export",
    "torchlens.fastlog",
    "torchlens.grad",
    "torchlens.intervene",
    "torchlens.io",
    "torchlens.options",
    "torchlens.partial",
    "torchlens.report",
    "torchlens.semantic",
    "torchlens.stats",
    "torchlens.types",
    "torchlens.utils",
    "torchlens.validation",
    "torchlens.visualization",
    "torchlens.viz",
]


def test_all_size_exactly_89() -> None:
    """Top-level ``__all__`` should contain exactly the current API budget.

    Phase 1a budget was 40; backward-parity sprint added 6 (grad_clip, grad_noise,
    grad_clamp, grad_fn, intervening, label) = 46; post-backward
    megasprint P1 added `output` (multi-output module selector disambiguation
    per AD-7 / F-Multi) = 47; facets framework adds `facets` and B1 removes
    the duplicate `label` export = 47; v7 quantity types add 5 = 52; facets
    P2 adds `facet` and `head` selectors = 54; capture-unification P4 adds
    `followed_by` and `preceded_by` predicate-window selectors = 56.
    Capture-unification P5 adds `when`, `add`, and `replace_with` = 59.
    torch.func transform capture adds `func_transform` = 60.
    Backend-completion sharded payload hints add `JaxPayloadLoadHint` and
    `PayloadLoadHints` = 62. Container value-core adds `Container`,
    `output_at`, and `register_container` = 65. Container-completion P3 adds
    `input_at` = 66. Eclectic Unit G adds `sweep` = 67.
    Glossary-conform-v11 DO-NOW renames: adds `record`, `Recording`, `push`,
    `push_from`, `run`, `pluck`, `extract_dataset`, `without_op`, `regex`,
    `span`; removes `sites` = 76. Internal sprint 2 adds `export` and
    `AmbiguousOpLookupError` = 78. Tech-debt sprint adds
    `ReentrantTraceError` and ten paper-era compatibility shims = 89.
    """

    assert len(torchlens.__all__) == 89
    assert torchlens.__all__ == TARGET_ALL


def test_phase_b_exports_are_top_level_importable() -> None:
    """Phase B public names should resolve from the top-level namespace."""

    assert torchlens.export is importlib.import_module("torchlens.export")
    assert torchlens.AmbiguousOpLookupError.__name__ == "AmbiguousOpLookupError"


def test_all_target_names_importable() -> None:
    """Every budgeted top-level name should resolve as ``torchlens.X``."""

    for name in TARGET_ALL:
        assert hasattr(torchlens, name), name


def test_no_duplicates() -> None:
    """Top-level ``__all__`` should not contain duplicate names."""

    assert len(set(torchlens.__all__)) == len(torchlens.__all__)


def test_submodules_have_all() -> None:
    """Every canonical Phase 1a submodule should import and define ``__all__``."""

    for module_name in CANONICAL_SUBMODULES:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            submodule = importlib.import_module(module_name)
        assert hasattr(submodule, "__all__"), module_name
        assert len(submodule.__all__) >= 0


def test_attribution_submodule_namespace_is_exposed_without_top_level_pollution() -> None:
    """``tl.attribution`` should resolve without exporting attribution functions."""

    submodule = importlib.import_module("torchlens.attribution")

    assert torchlens.attribution is submodule
    assert hasattr(torchlens.attribution, "saliency")
    assert "attribution" not in torchlens.__all__
    assert "saliency" not in torchlens.__all__

"""Tests for the TorchLens 2.0 top-level API budget."""

from __future__ import annotations

import importlib
import warnings

import torchlens

TARGET_ALL = [
    "trace",
    "fastlog",
    "facets",
    "load",
    "save",
    "do",
    "replay",
    "replay_from",
    "rerun",
    "bundle",
    "peek",
    "extract",
    "batched_extract",
    "validate",
    "Trace",
    "Layer",
    "Op",
    "Quantity",
    "Bytes",
    "Duration",
    "Flops",
    "Macs",
    "Bundle",
    "label",
    "func",
    "grad_fn",
    "intervening",
    "module",
    "output",
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
    "resample_ablate",
    "scale",
    "splice_module",
    "steer",
    "swap_with",
    "zero_ablate",
    "bwd_hook",
    "grad_clip",
    "grad_noise",
    "grad_clamp",
    "grad_scale",
    "grad_zero",
    "tap",
    "record_span",
    "sites",
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


def test_all_size_exactly_54() -> None:
    """Top-level ``__all__`` should contain exactly the current API budget.

    Phase 1a budget was 40; backward-parity sprint added 6 (grad_clip, grad_noise,
    grad_clamp, grad_fn, intervening, label) = 46; post-backward
    megasprint P1 added `output` (multi-output module selector disambiguation
    per AD-7 / F-Multi) = 47; facets framework adds `facets` and B1 removes
    the duplicate `label` export = 47; v7 quantity types add 5 = 52; facets
    P2 adds `facet` and `head` selectors = 54.
    """

    assert len(torchlens.__all__) == 54
    assert torchlens.__all__ == TARGET_ALL


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

"""Tests for the TorchLens 2.0 top-level API budget."""

from __future__ import annotations

import importlib
import warnings

import torchlens

TARGET_ALL = [
    # L3 ATen profile facade (documented-unstable; exported at __all__ head).
    # Ratchet row repaired by L6 post-merge: the L1/L3 merge train landed the
    # export without this row, leaving main red on the frozen-surface test.
    "AtenOp",
    "trace",
    "release_model",
    "clear_capture_cache",
    "export",
    "hash",
    "assert_unchanged",
    "fastlog",
    "facets",
    "record",
    "Recording",
    "ActivationLookup",
    "CapturedRun",
    "JaxPayloadLoadHint",
    "PayloadLoadHints",
    "load",
    "save",
    "do",
    "push",
    "push_from",
    "run",
    "bundle",
    "pluck",
    "extract",
    "extract_dataset",
    "load_extraction",
    "validate",
    "decide_recording_of_batch",
    "record_kpi_in_graph",
    "register_tensor_connection",
    "show_bundle_graph",
    "summary",
    "options",
    "to_disk",
    "AmbiguousOpLookupError",
    "ReentrantTraceError",
    "Trace",
    "Layer",
    "Container",
    "Op",
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
    "grad_fn_label",
    "grad_input",
    "grad_output",
    "in_backward_pass",
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
    "merge_ranks",
    "merge_report",
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
    "Selection",
    "ResolvedSelection",
    "units",
    "params",
    "random_selection",
    "Edit",
    "patch_from",
    "top_k",
    "top_fraction",
    "threshold",
    "sign",
    "dead",
    "saturated",
    "low_variance",
    "neighborhood",
    "between",
    "changed",
    "top_changed",
    "stable_across_passes",
    "pass_variance",
    "subspace",
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
    "torchlens.hash",
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


def test_all_matches_frozen_target_ledger() -> None:
    """Top-level ``__all__`` should match the current frozen API ledger.

    DECLARED gate only (oracles wave-0 re-point, never fork): this ledger is
    the declared claim; the REACHABLE surface -- everything ``dir(tl)``
    actually serves, class layer included -- is owned and counted by
    tests/oracles/test_oracle_w0_surface.py, which cites this ledger. The
    historical name of this test carried a stale count (the fact-10
    four-values-for-one-quantity instance); counts now live in data, not
    names.

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
    `ReentrantTraceError` = 79. Capture unification and backward/public option
    follow-ups add `ActivationLookup`, `CapturedRun`, `decide_recording_of_batch`,
    `record_kpi_in_graph`, `register_tensor_connection`, `show_bundle_graph`,
    `options`, `to_disk`, `grad_input`, `grad_output`, and `in_backward_pass` = 90.
    The provisional structural-hash namespace and CI tripwire add `hash` and
    `assert_unchanged` = 92. Model-lifecycle release support adds
    `release_model` = 93. The predicate-interpreter consolidation exports
    `grad_fn_label` (its own selector kind after the label-kind collision fix) = 94.
    Merge-ranks rung C1 adds `merge_ranks` and `merge_report` (spec'd
    top-level entry points; machinery lives in `torchlens.merged`) = 96.
    The grind R39 cache remedy exports `clear_capture_cache` (the agreed
    user-facing half of the capture-cache bounds fix) = 97.
    The L6 selection algebra (feature megasprint, DOCUMENTED-UNSTABLE pending
    naming-session ratification) adds `Selection`, `ResolvedSelection`,
    `units`, `params`, and `random_selection` = 102; its stage 2 adds
    `Edit` (public edit-object type; HelperSpec is the deprecated alias)
    and `patch_from` = 104. The L6 producer wave adds the value-based and
    statistical selection producers (DOCUMENTED-UNSTABLE) `top_k`,
    `top_fraction`, `threshold`, `sign`, `dead`, `saturated`, and
    `low_variance` = 111. Three further 2026-08-19 waves land on top of that,
    all DOCUMENTED-UNSTABLE: the graph-structural producers `neighborhood` and
    `between` (executed-DAG n-hop region; source-to-sink influence sub-DAG);
    the comparative producers `changed`, `top_changed`,
    `stable_across_passes`, and `pass_variance`; and the semantic/appliance
    additions from the same sprint. The RUNTIME total is 114 -- asserted
    against TARGET_ALL below rather than re-derived here, because three
    concurrent lanes each computed an increment from 111 without knowing about
    the others and every hand-derived subtotal was wrong. The subspace
    producer wave adds `subspace` (direction/subspace support selection with
    mandatory basis provenance, DOCUMENTED-UNSTABLE) on top of that.
    Paper-era compatibility shims remain available through ``__getattr__`` but
    are not advertised in ``__all__``.
    """

    assert len(torchlens.__all__) == len(TARGET_ALL)
    assert torchlens.__all__ == TARGET_ALL


def test_phase_b_exports_are_top_level_importable() -> None:
    """Phase B public names should resolve from the top-level namespace."""

    assert torchlens.export is importlib.import_module("torchlens.export")
    assert torchlens.AmbiguousOpLookupError.__name__ == "AmbiguousOpLookupError"
    assert importlib.import_module("torchlens.facets") is importlib.import_module(
        "torchlens.semantic.facets"
    )


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
        assert all(isinstance(name, str) and name for name in submodule.__all__), module_name


def test_attribution_submodule_namespace_is_exposed_without_top_level_pollution() -> None:
    """``tl.attribution`` should resolve without exporting attribution functions."""

    submodule = importlib.import_module("torchlens.attribution")

    assert torchlens.attribution is submodule
    assert hasattr(torchlens.attribution, "saliency")
    assert "attribution" not in torchlens.__all__
    assert "saliency" not in torchlens.__all__

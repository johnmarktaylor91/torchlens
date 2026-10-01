"""Absence pins for the removed top-level deprecation shims.

The moved-name ``__getattr__`` table, the paper-era API shims, and the
deprecated top-level wrapper functions were deleted by the 2026-08-19
shim-removal lane. Contract now: the historical top-level spellings raise
``AttributeError`` and the canonical submodule spellings resolve.
"""

from __future__ import annotations

import importlib

import pytest

import torchlens

pytestmark = pytest.mark.smoke

#: (removed top-level name, canonical module, canonical attribute).
REMOVED_TOP_LEVEL_CASES = [
    ("ActivationPostfunc", "torchlens.types", "ActivationPostfunc"),
    ("Buffer", "torchlens.types", "Buffer"),
    ("FuncCallLocation", "torchlens.types", "FuncCallLocation"),
    ("GradientPostfunc", "torchlens.types", "GradientPostfunc"),
    ("GradFnAccessor", "torchlens.accessors", "GradFnAccessor"),
    ("GradFn", "torchlens.types", "GradFn"),
    ("GradFnCall", "torchlens.types", "GradFnCall"),
    ("LayerAccessor", "torchlens.accessors", "LayerAccessor"),
    ("MetadataInvariantError", "torchlens.errors", "MetadataInvariantError"),
    ("ModuleAccessor", "torchlens.accessors", "ModuleAccessor"),
    ("Module", "torchlens.types", "Module"),
    ("ModuleCall", "torchlens.types", "ModuleCall"),
    ("NodeSpec", "torchlens.experimental.dagua", "NodeSpec"),
    ("Param", "torchlens.types", "Param"),
    ("TraceState", "torchlens.io", "TraceState"),
    ("SaveLevel", "torchlens.types", "SaveLevel"),
    ("SiteTable", "torchlens.types", "SiteTable"),
    ("SpecCompat", "torchlens.types", "SpecCompat"),
    ("StreamingOptions", "torchlens.options", "StreamingOptions"),
    ("TargetManifestDiff", "torchlens.types", "TargetManifestDiff"),
    ("TensorLog", "torchlens.types", "TensorLog"),
    ("TensorSliceSpec", "torchlens.types", "TensorSliceSpec"),
    ("TorchLensPostfuncError", "torchlens.errors", "TorchLensPostfuncError"),
    ("TrainingModeConfigError", "torchlens.errors", "TrainingModeConfigError"),
    ("VisualizationOptions", "torchlens.options", "VisualizationOptions"),
    ("build_render_audit", "torchlens.experimental.dagua", "build_render_audit"),
    ("check_metadata_invariants", "torchlens.validation", "check_metadata_invariants"),
    ("check_spec_compat", "torchlens.validation", "check_spec_compat"),
    ("cleanup_tmp", "torchlens.io", "cleanup_tmp"),
    # get_model_metadata was itself a deprecated alias; canonical is log_model_metadata.
    ("get_model_metadata", "torchlens.io", "log_model_metadata"),
    ("list_logs", "torchlens.io", "list_logs"),
    ("log_model_metadata", "torchlens.io", "log_model_metadata"),
    ("trace_to_dagua_graph", "torchlens.experimental.dagua", "trace_to_dagua_graph"),
    ("preview_fastlog", "torchlens.fastlog", "preview"),
    ("rehydrate_nested", "torchlens.io", "rehydrate_nested"),
    ("render_lines_to_html", "torchlens.experimental.dagua", "render_lines_to_html"),
    ("render_trace_with_dagua", "torchlens.experimental.dagua", "render_trace_with_dagua"),
    ("reset_naming_counter", "torchlens.io", "reset_naming_counter"),
    ("resolve_sites", "torchlens.validation", "resolve_sites"),
    ("save_intervention", "torchlens.io", "save_intervention"),
    ("suppress_mutate_warnings", "torchlens.io", "suppress_mutate_warnings"),
    ("unwrap_torch", "torchlens.backends.torch.wrappers", "unwrap_torch"),
    (
        "validate_batch_of_models_and_inputs",
        "torchlens.validation",
        "validate_batch_of_models_and_inputs",
    ),
    ("wrap_torch", "torchlens.backends.torch.wrappers", "wrap_torch"),
    ("wrapped", "torchlens.backends.torch.wrappers", "wrapped"),
]

#: Removed top-level wrapper functions whose canonical spelling lives in a
#: submodule (validation/visualization/io).
REMOVED_WRAPPER_NAMES = [
    "validate_forward_pass",
    "validate_backward_pass",
    "validate_saved_outs",
    # "summary" left this ledger 2026-08-26 (megasprint lane A07, summary memo
    # A8): tl.summary is UN-removed as the first-class one-call front door.
    "show_model_graph",
    "draw_backward",
    "draw_combined",
    "load_intervention_spec",
]

#: Removed paper-era shim names.
REMOVED_PAPER_ERA_NAMES = [
    "log_forward_pass",
    "validate_model_activations",
    "validate_saved_activations",
    "render_graph",
    "render_model_graph",
    "draw_model_graph",
    "ModelHistory",
    "get_model_structure",
    "show_model_structure",
]


@pytest.mark.parametrize(("old_name", "module_name", "new_name"), REMOVED_TOP_LEVEL_CASES)
def test_moved_name_is_gone_and_canonical_resolves(
    old_name: str, module_name: str, new_name: str
) -> None:
    """The old top-level spelling refuses; the canonical spelling resolves."""

    with pytest.raises(AttributeError):
        getattr(torchlens, old_name)
    module = importlib.import_module(module_name)
    assert getattr(module, new_name) is not None


@pytest.mark.parametrize("name", REMOVED_WRAPPER_NAMES + REMOVED_PAPER_ERA_NAMES)
def test_removed_top_level_wrapper_is_gone(name: str) -> None:
    """Removed wrapper and paper-era spellings raise AttributeError."""

    with pytest.raises(AttributeError):
        getattr(torchlens, name)


def test_dir_lists_no_removed_names() -> None:
    """``dir(torchlens)`` no longer advertises any removed spelling."""

    visible = set(dir(torchlens))
    removed = (
        {case[0] for case in REMOVED_TOP_LEVEL_CASES}
        | set(REMOVED_WRAPPER_NAMES)
        | set(REMOVED_PAPER_ERA_NAMES)
    )
    still_visible = sorted(visible & removed)
    assert not still_visible, still_visible

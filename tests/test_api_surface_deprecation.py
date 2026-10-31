"""Absence pins for the removed top-level deprecation shims.

The moved-name ``__getattr__`` table, the paper-era API shims, and the
deprecated top-level wrapper functions were deleted by the 2026-08-19
shim-removal lane. Contract now: the historical top-level spellings raise
the typed ``facade_redirect`` ``FacadeTeachingError`` (an ``AttributeError``)
naming their canonical home, and the canonical submodule spellings resolve.
"""

from __future__ import annotations

import importlib

import pytest

import torchlens
from torchlens._errors import FacadeTeachingError

#: (removed top-level name, canonical module, canonical attribute).
REMOVED_TOP_LEVEL_CASES = [
    ("AtenOp", "torchlens.types", "AtenOp"),
    ("Container", "torchlens.types", "Container"),
    ("register_container", "torchlens.types", "register_container"),
    ("span", "torchlens.observers", "span"),
    ("tap", "torchlens.observers", "tap"),
    ("assert_unchanged", "torchlens.fingerprint", "assert_unchanged"),
    ("Quantity", "torchlens.types", "Quantity"),
    ("Bytes", "torchlens.types", "Bytes"),
    ("Duration", "torchlens.types", "Duration"),
    ("Flops", "torchlens.types", "Flops"),
    ("Macs", "torchlens.types", "Macs"),
    ("Buffer", "torchlens.types", "Buffer"),
    ("FuncCallLocation", "torchlens.types", "FuncCallLocation"),
    ("GradFnAccessor", "torchlens._data_classes.grad_fn", "GradFnAccessor"),
    ("GradFn", "torchlens.types", "GradFn"),
    ("GradFnCall", "torchlens.types", "GradFnCall"),
    ("LayerAccessor", "torchlens._data_classes._layer_accessors", "LayerAccessor"),
    ("MetadataInvariantError", "torchlens.errors", "MetadataInvariantError"),
    ("ModuleAccessor", "torchlens._data_classes.module", "ModuleAccessor"),
    ("Module", "torchlens.types", "Module"),
    ("ModuleCall", "torchlens.types", "ModuleCall"),
    ("NodeSpec", "torchlens.experimental.dagua", "NodeSpec"),
    ("Param", "torchlens.types", "Param"),
    ("TraceState", "torchlens.io", "TraceState"),
    ("SiteTable", "torchlens.intervention.resolver", "SiteTable"),
    ("SpecCompat", "torchlens.types", "SpecCompat"),
    ("StreamingOptions", "torchlens.options", "StorageOptions"),
    ("TargetManifestDiff", "torchlens.types", "TargetManifestDiff"),
    ("TorchLensPostfuncError", "torchlens.errors", "UserTransformError"),
    ("TrainingModeConfigError", "torchlens.errors", "TrainingModeConfigError"),
    ("VisualizationOptions", "torchlens.options", "VisualizationOptions"),
    ("check_metadata_invariants", "torchlens.validation", "check_metadata_invariants"),
    ("check_spec_compat", "torchlens.intervention", "check_spec_compat"),
    ("cleanup_tmp", "torchlens.io", "cleanup_tmp"),
    ("get_model_metadata", "torchlens", "trace"),
    ("list_logs", "torchlens.io", "list_traces"),
    ("log_model_metadata", "torchlens", "trace"),
    ("trace_to_dagua_graph", "torchlens.experimental.dagua", "trace_to_dagua_graph"),
    ("rehydrate_nested", "torchlens.io", "materialize_nested"),
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

#: Old top-level spellings whose later submodule home also left the public
#: surface: the redirect teaches the replacement workflow (``Trace.draw`` or
#: ``fastlog.dry_run``), not a renamed function.
REMOVED_SURFACE_NAMES = [
    "build_render_audit",
    "preview_fastlog",
    "render_lines_to_html",
    "render_trace_with_dagua",
]

#: Removed top-level wrapper functions whose canonical spelling lives in a
#: submodule (validation/visualization/io).
REMOVED_WRAPPER_NAMES = [
    "validate_forward_pass",
    "validate_backward_pass",
    "validate_saved_outs",
    # "summary" left this ledger 2026-08-26 (workstream A07, summary memo
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


@pytest.mark.smoke_cells(
    "test_moved_name_is_gone_and_canonical_resolves[NodeSpec-torchlens.experimental.dagua-NodeSpec]",
)
@pytest.mark.parametrize(("old_name", "module_name", "new_name"), REMOVED_TOP_LEVEL_CASES)
def test_moved_name_is_gone_and_canonical_resolves(
    old_name: str, module_name: str, new_name: str
) -> None:
    """The old top-level spelling refuses typed naming its home; the canonical resolves."""

    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(torchlens, old_name)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert module_name in str(excinfo.value)
    module = importlib.import_module(module_name)
    assert getattr(module, new_name) is not None


@pytest.mark.smoke_cells("test_removed_top_level_wrapper_is_gone[preview_fastlog]")
@pytest.mark.parametrize(
    "name", REMOVED_SURFACE_NAMES + REMOVED_WRAPPER_NAMES + REMOVED_PAPER_ERA_NAMES
)
def test_removed_top_level_wrapper_is_gone(name: str) -> None:
    """Removed wrapper and paper-era spellings raise the typed redirect error."""

    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(torchlens, name)
    assert excinfo.value.fields["code"] == "facade_redirect"


def test_dir_lists_no_removed_names() -> None:
    """``dir(torchlens)`` no longer advertises any removed spelling."""

    visible = set(dir(torchlens))
    removed = (
        {case[0] for case in REMOVED_TOP_LEVEL_CASES}
        | set(REMOVED_SURFACE_NAMES)
        | set(REMOVED_WRAPPER_NAMES)
        | set(REMOVED_PAPER_ERA_NAMES)
    )
    still_visible = sorted(visible & removed)
    assert not still_visible, still_visible

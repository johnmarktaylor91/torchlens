"""Absence pins for the removed top-level deprecation shims.

The moved-name ``__getattr__`` table, the paper-era API shims, and the
deprecated top-level wrapper functions were deleted by the 2026-08-19
shim-removal lane. Contract now: the historical top-level spellings raise
the typed ``facade_redirect`` ``FacadeTeachingError`` (an ``AttributeError``)
naming their canonical home, and the canonical submodule spellings resolve.
"""

from __future__ import annotations

import importlib

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
    ("NodeSpec", "torchlens.viz", "NodeSpec"),
    ("Param", "torchlens.types", "Param"),
    ("TraceState", "torchlens.io", "TraceState"),
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
    ("rehydrate_nested", "torchlens.io", "materialize_nested"),
    ("reset_naming_counter", "torchlens.io", "reset_naming_counter"),
    ("save_intervention", "torchlens.io", "save_intervention"),
    ("suppress_mutate_warnings", "torchlens.errors", "quiet"),
    ("unwrap_torch", "torchlens.backends.torch.wrappers", "unwrap_torch"),
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
    "resolve_sites",
    "SiteTable",
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
    "validate_batch_of_models_and_inputs",
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


#: Removed accessor classes have no public import; their redirect names the public
#: Trace attribute that returns them instead of the private module that defines them.
PUBLIC_ATTRIBUTE_REDIRECTS = {
    "GradFnAccessor": "torchlens.Trace.autograd",
    "LayerAccessor": "torchlens.Trace.layers",
    "ModuleAccessor": "torchlens.Trace.modules",
}


def test_removed_top_level_names_redirect_and_stay_hidden() -> None:
    """Every removed top-level spelling refuses typed and ``dir(torchlens)`` advertises none.

    One test over the four name tables (the per-name parametrization folded: its arcs are all
    covered by kept tests, but these tables are the data pin the migration guide cites per name).
    A moved name's redirect names its canonical module (or, for a removed accessor class, the
    public Trace attribute that returns it) and the canonical spelling resolves.
    """

    failures: list[str] = []
    for old_name, module_name, new_name in REMOVED_TOP_LEVEL_CASES:
        try:
            getattr(torchlens, old_name)
        except FacadeTeachingError as error:
            expected = PUBLIC_ATTRIBUTE_REDIRECTS.get(old_name, module_name)
            if error.fields["code"] != "facade_redirect" or expected not in str(error):
                failures.append(f"{old_name}: redirect does not name {expected}: {error}")
        else:
            failures.append(f"{old_name}: still resolves at the top level")
        if getattr(importlib.import_module(module_name), new_name, None) is None:
            failures.append(f"{old_name}: canonical {module_name}.{new_name} does not resolve")
    for name in REMOVED_SURFACE_NAMES + REMOVED_WRAPPER_NAMES + REMOVED_PAPER_ERA_NAMES:
        try:
            getattr(torchlens, name)
        except FacadeTeachingError as error:
            if error.fields["code"] != "facade_redirect":
                failures.append(f"{name}: wrong code {error.fields['code']!r}")
        else:
            failures.append(f"{name}: still resolves at the top level")
    removed = (
        {case[0] for case in REMOVED_TOP_LEVEL_CASES}
        | set(REMOVED_SURFACE_NAMES)
        | set(REMOVED_WRAPPER_NAMES)
        | set(REMOVED_PAPER_ERA_NAMES)
    )
    failures.extend(
        f"{name}: listed by dir(torchlens)" for name in sorted(set(dir(torchlens)) & removed)
    )
    assert not failures, "\n".join(failures)

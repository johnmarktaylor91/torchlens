"""TorchLens - extract outs and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace -- and no
torch import at all until first use (agent memo 3.11: tier-0 CLI verbs and
manifest preflights stay torch-free). Torch functions are wrapped lazily on
the first call to ``trace()`` and stay wrapped afterward. TorchLens 2.0 keeps
the top-level namespace intentionally small; historical spellings live in
their owning submodules.

For AI agents: ``trace.to_agent_json()``, ``tl.report.explain(trace,
max_tokens=N)``, ``torchlens.agent.guide()``, and the CLI
(``python -m torchlens --help``) are the machine-facing doors.
"""

from __future__ import annotations

import sys as _sys
import types as _types
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import TYPE_CHECKING as _TYPE_CHECKING, Any as _Any

if _TYPE_CHECKING:
    import torch as _torch
    from torch import nn as _nn

__version__ = "2.35.3"

if _TYPE_CHECKING:
    from .data_classes.trace import Trace
    from .intervention import Bundle

_LAZY_ATTRS = {
    # Import cold-start laziness (P4, rebaselined 2026-08-19): the former
    # eager import block (options/captured_run+ir/observers/quantities/errors
    # and their transitive chains) is fully deferred behind these rows -- the
    # marginal-import guard in tests/test_import_hygiene.py holds the line.
    "ActivationLookup": ("torchlens.captured_run", "ActivationLookup"),
    "AmbiguousOpLookupError": ("torchlens._errors", "AmbiguousOpLookupError"),
    "Bytes": ("torchlens.quantities", "Bytes"),
    "CapturedRun": ("torchlens.captured_run", "CapturedRun"),
    "Duration": ("torchlens.quantities", "Duration"),
    "Flops": ("torchlens.quantities", "Flops"),
    "Macs": ("torchlens.quantities", "Macs"),
    "Quantity": ("torchlens.quantities", "Quantity"),
    "ReentrantTraceError": ("torchlens._state", "ReentrantTraceError"),
    # Agent surface (F29): the read-only inspection core. Deliberately NOT in
    # __all__ (frozen root budget); reachable as tl.agent per the docs.
    "agent": ("torchlens.agent", None),
    # The docs teach tl.utils.doctor(); the row was missing so the taught
    # spelling raised AttributeError on a cold import (agent memo 3.10).
    "utils": ("torchlens.utils", None),
    "captured_run": ("torchlens.captured_run", None),
    "errors": ("torchlens.errors", None),
    "ir": ("torchlens.ir", None),
    "observers": ("torchlens.observers", None),
    "options": ("torchlens.options", None),
    "quantities": ("torchlens.quantities", None),
    "brainpipe": ("torchlens.brainpipe", None),
    "register_container": ("torchlens.ir.container", "register_container"),
    "span": ("torchlens.observers", "span"),
    "tap": ("torchlens.observers", "tap"),
    "to_disk": ("torchlens.options", "to_disk"),
    "AtenOp": ("torchlens.data_classes.aten_op", "AtenOp"),
    "Bundle": ("torchlens.intervention", "Bundle"),
    "Container": ("torchlens.data_classes.container", "Container"),
    "JaxPayloadLoadHint": ("torchlens._io", "JaxPayloadLoadHint"),
    "Layer": ("torchlens.data_classes.layer", "Layer"),
    "Op": ("torchlens.data_classes.op", "Op"),
    "PayloadLoadHints": ("torchlens._io", "PayloadLoadHints"),
    "Recording": ("torchlens.fastlog", "Recording"),
    "Trace": ("torchlens.data_classes.trace", "Trace"),
    "add": ("torchlens.intervention", "add"),
    "aggregate": ("torchlens.stats", "aggregate"),
    "assert_unchanged": ("torchlens.hash", "assert_unchanged"),
    "attribution": ("torchlens.attribution", None),
    # r7 R81 (sol b2): docs/semantic_io.md documents tl.autoroute.output.*
    # and the agent docs list autoroute among the lazy attrs, but the row
    # was missing -- the documented spelling resolved only after a separate
    # `import torchlens.autoroute` (import-order side effect).
    "autoroute": ("torchlens.autoroute", None),
    # Entry/facade repair (workstream A10; WT1 A-VI item 27, neuro memo item
    # 2): the integration and appliance namespaces were absent from this map,
    # so `tl.bridge` resolved only after an unrelated capture side effect
    # imported it and `tl.callbacks` never resolved at all. All four package
    # __init__ modules are import-inert by design (foreign dependencies stay
    # deferred behind their own facades).
    "bridge": ("torchlens.bridge", None),
    "callbacks": ("torchlens.callbacks", None),
    "neuro": ("torchlens.neuro", None),
    "notebook": ("torchlens.notebook", None),
    "bwd_hook": ("torchlens.intervention", "bwd_hook"),
    "clamp": ("torchlens.intervention", "clamp"),
    "compat": ("torchlens.compat", None),
    "contains": ("torchlens.intervention", "contains"),
    "decide_recording_of_batch": ("torchlens.user_funcs", "decide_recording_of_batch"),
    "debug": ("torchlens.debug", None),
    "data_classes": ("torchlens.data_classes", None),
    # Dataset extraction (D7/V5): the implementation module is lazy so the
    # manifest/resume machinery costs nothing until first use.
    "dataset_extraction": ("torchlens.dataset_extraction", None),
    "extract_dataset": ("torchlens.dataset_extraction", "extract_dataset"),
    # WT1 A-VI item 27: the reader half of the extraction workflow was only
    # reachable as torchlens.dataset_extraction.load_extraction while the
    # writer (extract_dataset) was top-level.
    "load_extraction": ("torchlens.dataset_extraction", "load_extraction"),
    "distributed": ("torchlens.distributed", None),
    "do": ("torchlens.intervention", "do"),
    "examples": ("torchlens.examples", None),
    "experimental": ("torchlens.experimental", None),
    "export": ("torchlens.export", None),
    "facets": ("torchlens.semantic", "facets"),
    "fastlog": ("torchlens.fastlog", None),
    "facet": ("torchlens.intervention", "facet"),
    "followed_by": ("torchlens.intervention", "followed_by"),
    "func": ("torchlens.intervention", "func"),
    "func_transform": ("torchlens.intervention", "func_transform"),
    "grad_clamp": ("torchlens.intervention", "grad_clamp"),
    "grad_clip": ("torchlens.intervention", "grad_clip"),
    "grad_fn": ("torchlens.intervention", "grad_fn"),
    "grad_fn_label": ("torchlens.intervention", "grad_fn_label"),
    "grad_input": ("torchlens.intervention", "grad_input"),
    "grad_noise": ("torchlens.intervention", "grad_noise"),
    "grad_output": ("torchlens.intervention", "grad_output"),
    "grad_scale": ("torchlens.intervention", "grad_scale"),
    "grad_zero": ("torchlens.intervention", "grad_zero"),
    "hash": ("torchlens.hash", None),
    "head": ("torchlens.intervention", "head"),
    "in_backward_pass": ("torchlens.intervention", "in_backward_pass"),
    "in_module": ("torchlens.intervention", "in_module"),
    "input_at": ("torchlens.intervention", "input_at"),
    "intervention": ("torchlens.intervention", None),
    "io": ("torchlens.io", None),
    "load": ("torchlens._io.bundle", "load"),
    "label": ("torchlens.intervention", "label"),
    "mean_ablate": ("torchlens.intervention", "mean_ablate"),
    "merge_ranks": ("torchlens.merged", "merge_ranks"),
    "merge_report": ("torchlens.merged", "merge_report"),
    "merged": ("torchlens.merged", None),
    "module": ("torchlens.intervention", "module"),
    "noise": ("torchlens.intervention", "noise"),
    "output": ("torchlens.intervention", "output"),
    "output_at": ("torchlens.intervention", "output_at"),
    "partial": ("torchlens.partial", None),
    "report": ("torchlens.report", None),
    "repgeom": ("torchlens.repgeom", None),
    "preceded_by": ("torchlens.intervention", "preceded_by"),
    "project_off": ("torchlens.intervention", "project_off"),
    "project_onto": ("torchlens.intervention", "project_onto"),
    "push": ("torchlens.intervention", "push"),
    "push_from": ("torchlens.intervention", "push_from"),
    "record": ("torchlens.fastlog", "record"),
    "record_kpi_in_graph": ("torchlens.user_funcs", "record_kpi_in_graph"),
    "receptive_field": ("torchlens.receptive_field", None),
    "regex": ("torchlens.intervention", "regex"),
    "register_tensor_connection": ("torchlens.user_funcs", "register_tensor_connection"),
    "clear_capture_cache": ("torchlens.user_funcs", "clear_capture_cache"),
    "release_model": ("torchlens.user_funcs", "release_model"),
    "replace_with": ("torchlens.intervention", "replace_with"),
    "run": ("torchlens.intervention", "run"),
    "save": ("torchlens._io.bundle", "save"),
    "scale": ("torchlens.intervention", "scale"),
    "show_bundle_graph": ("torchlens.user_funcs", "show_bundle_graph"),
    "summary": ("torchlens.user_funcs", "summary"),
    "splice_module": ("torchlens.intervention", "splice_module"),
    "stats": ("torchlens.stats", None),
    "steer": ("torchlens.intervention", "steer"),
    "sweep": ("torchlens.intervention.sweep", "sweep"),
    "swap_with": ("torchlens.intervention", "swap_with"),
    "trace": ("torchlens.user_funcs", "trace"),
    # Built-in activation transforms (transforms memo s7 home; the submodule
    # spelling tl.transforms is DOCUMENTED-UNSTABLE — the collision with the
    # input-side transform= slot is a named naming-sprint item).
    "transforms": ("torchlens.transforms", None),
    "user_funcs": ("torchlens.user_funcs", None),
    "validate": ("torchlens.validation.consolidated", "validate"),
    "validation": ("torchlens.validation", None),
    "visualization": ("torchlens.visualization", None),
    "types": ("torchlens.types", None),
    "accessors": ("torchlens.accessors", None),
    "backends": ("torchlens.backends", None),
    "viz": ("torchlens.viz", None),
    "when": ("torchlens.intervention", "when"),
    "where": ("torchlens.intervention", "where"),
    "without_op": ("torchlens.intervention", "without_op"),
    "zero_ablate": ("torchlens.intervention", "zero_ablate"),
    "Edit": ("torchlens.intervention", "Edit"),
    "patch_from": ("torchlens.intervention", "patch_from"),
    # L6 selection algebra (DOCUMENTED-UNSTABLE pending naming-session
    # ratification; provisional-name protocol).
    "Selection": ("torchlens.selection", "Selection"),
    "ResolvedSelection": ("torchlens.selection", "ResolvedSelection"),
    "units": ("torchlens.selection", "units"),
    "params": ("torchlens.selection", "params"),
    "random_selection": ("torchlens.selection", "random_selection"),
    # L6 value-based + statistical producers (DOCUMENTED-UNSTABLE pending
    # naming-session ratification; provisional-name protocol).
    "top_k": ("torchlens.selection_values", "top_k"),
    "top_fraction": ("torchlens.selection_values", "top_fraction"),
    "threshold": ("torchlens.selection_values", "threshold"),
    "sign": ("torchlens.selection_values", "sign"),
    "dead": ("torchlens.selection_values", "dead"),
    "saturated": ("torchlens.selection_values", "saturated"),
    "low_variance": ("torchlens.selection_values", "low_variance"),
    # L6 graph-structural producers (DOCUMENTED-UNSTABLE pending
    # naming-session ratification; provisional-name protocol).
    "neighborhood": ("torchlens.selection_graph", "neighborhood"),
    "between": ("torchlens.selection_graph", "between"),
    # L6 comparative producers: differential + cross-pass (DOCUMENTED-UNSTABLE
    # pending naming-session ratification; provisional-name protocol).
    "changed": ("torchlens.selection_compare", "changed"),
    "top_changed": ("torchlens.selection_compare", "top_changed"),
    "stable_across_passes": ("torchlens.selection_compare", "stable_across_passes"),
    "pass_variance": ("torchlens.selection_compare", "pass_variance"),
    # L6 subspace producer (DOCUMENTED-UNSTABLE pending naming-session
    # ratification; provisional-name protocol).
    "subspace": ("torchlens.selection_subspace", "subspace"),
}


def _resolve_top_level(name: str) -> _Any:
    """Resolve a top-level TorchLens attribute, honoring existing globals.

    Parameters
    ----------
    name:
        Top-level attribute name.

    Returns
    -------
    Any
        Existing global value or lazily resolved attribute.
    """

    if name in globals():
        return globals()[name]
    return __getattr__(name)


# Five-step facade tables (architecture memo 5.4; mechanism owned by
# torchlens.utils.facade). The redirect table is the teaching compensator for
# the 2026-08 shim-removal pass (docs/migration/v2.0_api_changes.md): every
# removed top-level spelling raises a typed ``facade_redirect`` AttributeError
# naming its canonical home instead of a bare "no attribute" (AUD-CODE 3.14).
# A redirect row wins over a real name (step 2 precedes step 4), so no row may
# name a LIVE top-level attribute; tests/test_w051_gate_facade_redirects.py
# pins that and that every canonical dotted path resolves. The refusal table
# stays empty at the root: no root name is deliberately refused today (the
# appliance namespaces, e.g. torchlens.neuro, carry their own refusal rows).
# Paper-era 1.x names (the pre-2.0 capture verb, the pre-2.0 log class, the
# paper-era validators, renderers and structure printers) ARE rows: teaching redirects for
# removed names are the table's purpose. The repo-wide removed-spelling lint
# (tests/test_removed_spelling_lint.py, group paper_era) admits exactly this
# file for that group through its audited _ALLOWED ledger (GATE-FIX row 1);
# docs/reference/deprecations.md teaches the same names in prose.
_REDIRECTS: dict[str, str] = {
    "ActivationPostfunc": "use torchlens.types.ActivationPostfunc",
    "batched_extract": "use torchlens.extract_dataset",
    "Buffer": "use torchlens.types.Buffer",
    "build_render_audit": "use torchlens.experimental.dagua.build_render_audit",
    "check_metadata_invariants": "use torchlens.validation.check_metadata_invariants",
    "check_spec_compat": "use torchlens.validation.check_spec_compat",
    "cleanup_tmp": "use torchlens.io.cleanup_tmp",
    "draw_backward": "use torchlens.visualization.draw_backward",
    "draw_combined": "use torchlens.visualization.draw_combined",
    "draw_model_graph": "use torchlens.visualization.show_model_graph, or trace.draw()",
    "FuncCallLocation": "use torchlens.types.FuncCallLocation",
    "get_model_metadata": "use torchlens.io.log_model_metadata -- get_model_metadata was renamed",
    "get_model_structure": "use torchlens.summary(model, x) for the module tree, or torchlens.trace(model, x).modules",
    "GradFn": "use torchlens.types.GradFn",
    "GradFnAccessor": "use torchlens.accessors.GradFnAccessor",
    "GradFnCall": "use torchlens.types.GradFnCall",
    "GradientPostfunc": "use torchlens.types.GradientPostfunc",
    "intervening": "use torchlens.without_op -- intervening was renamed",
    "LayerAccessor": "use torchlens.accessors.LayerAccessor",
    "list_logs": "use torchlens.io.list_logs",
    "load_intervention_spec": "use torchlens.io.load_intervention_spec",
    "log_forward_pass": "use torchlens.trace(model, x) -- log_forward_pass was the pre-2.0 capture verb",
    "log_model_metadata": "use torchlens.io.log_model_metadata",
    "MetadataInvariantError": "use torchlens.errors.MetadataInvariantError",
    "ModelHistory": "use torchlens.Trace -- ModelHistory was renamed ModelLog, then Trace",
    "ModelLog": "use torchlens.Trace -- ModelLog was renamed Trace",
    "Module": "use torchlens.types.Module",
    "ModuleAccessor": "use torchlens.accessors.ModuleAccessor",
    "ModuleCall": "use torchlens.types.ModuleCall",
    "ModuleInputSnapshot": "use torchlens.data_classes.ModuleInputSnapshot",
    "NodeSpec": "use torchlens.experimental.dagua.NodeSpec",
    "Param": "use torchlens.types.Param",
    "peek": "use torchlens.pluck",
    "PreHookEffect": "use torchlens.data_classes.PreHookEffect",
    "preview_fastlog": "use torchlens.fastlog.preview",
    "record_span": "use torchlens.span",
    "rehydrate_nested": "use torchlens.io.rehydrate_nested",
    "render_graph": "use torchlens.visualization.show_model_graph, or trace.draw()",
    "render_lines_to_html": "use torchlens.experimental.dagua.render_lines_to_html",
    "render_model_graph": "use torchlens.visualization.show_model_graph, or trace.draw()",
    "render_trace_with_dagua": "use torchlens.experimental.dagua.render_trace_with_dagua",
    "replay": "use torchlens.push",
    "replay_from": "use torchlens.push_from",
    "rerun": "use torchlens.run",
    "resample_ablate": "use torchlens.intervention.scramble_elements -- resample_ablate was renamed",
    "reset_naming_counter": "use torchlens.io.reset_naming_counter",
    "resolve_sites": "use torchlens.validation.resolve_sites",
    "save_intervention": "use torchlens.io.save_intervention",
    "SaveLevel": "use torchlens.types.SaveLevel",
    "show_model_graph": "use torchlens.visualization.show_model_graph",
    "show_model_structure": "use torchlens.summary(model, x) for the module tree, or torchlens.trace(model, x).modules",
    "SiteTable": "use torchlens.types.SiteTable",
    "SpecCompat": "use torchlens.types.SpecCompat",
    "StreamingOptions": "use torchlens.options.StreamingOptions",
    "suppress_mutate_warnings": "use torchlens.io.suppress_mutate_warnings",
    "TargetManifestDiff": "use torchlens.types.TargetManifestDiff",
    "TensorInputObservation": "use torchlens.data_classes.TensorInputObservation",
    "TensorLog": "use torchlens.types.TensorLog",
    "TensorSliceSpec": "use torchlens.types.TensorSliceSpec",
    "TorchLensPostfuncError": "use torchlens.errors.TorchLensPostfuncError",
    "trace_to_dagua_graph": "use torchlens.experimental.dagua.trace_to_dagua_graph",
    "TraceState": "use torchlens.io.TraceState",
    "TrainingModeConfigError": "use torchlens.errors.TrainingModeConfigError",
    "unwrap_torch": "use torchlens.backends.torch.wrappers.unwrap_torch",
    "validate_backward_pass": "use torchlens.validation.validate_backward_pass",
    "validate_batch_of_models_and_inputs": "use torchlens.validation.validate_batch_of_models_and_inputs",
    "validate_forward_pass": "use torchlens.validation.validate_forward_pass",
    "validate_model_activations": "use torchlens.validate(scope='forward')",
    "validate_saved_activations": "use torchlens.validate(scope='saved')",
    "validate_saved_outs": "use torchlens.validate(scope='saved')",
    "VisualizationOptions": "use torchlens.options.VisualizationOptions",
    "wrap_torch": "use torchlens.backends.torch.wrappers.wrap_torch",
    "wrapped": "use torchlens.backends.torch.wrappers.wrapped",
}
_REFUSALS: dict[str, str] = {}


def __getattr__(name: str) -> _Any:
    """Return lazy package attributes through the five-step facade order.

    The order (architecture memo 5.4, testable spec): (1) underscore-prefixed
    names raise plain ``AttributeError`` immediately; (2) redirect-table rows
    raise a typed teaching ``AttributeError`` naming the canonical spelling,
    with no dependency check; (3) refusal-table rows raise a typed
    ``AttributeError``; (4) real active names resolve through their per-name
    dependency gate; (5) everything else raises plain ``AttributeError``.
    Every typed error subclasses ``AttributeError`` so ``hasattr`` can never
    explode.

    Parameters
    ----------
    name:
        Attribute requested from the ``torchlens`` package.

    Returns
    -------
    Any
        The requested lazy object.

    Raises
    ------
    AttributeError
        Per the five-step contract above.
    """

    from .utils.facade import resolve_facade_attr

    return resolve_facade_attr(
        owner=__name__,
        name=name,
        module_globals=globals(),
        lazy_attrs=_LAZY_ATTRS,
        redirects=_REDIRECTS,
        refusals=_REFUSALS,
    )


def __dir__() -> list[str]:
    """Return visible top-level TorchLens attributes.

    Returns
    -------
    list[str]
        Sorted real public names (eager globals plus lazy facade rows);
        single-underscore implementation aliases are not advertised.
    """

    from .utils.facade import facade_dir

    return facade_dir(globals(), _LAZY_ATTRS)


def _did_you_mean_message(name: str, suggestions: list[str]) -> str:
    """Build a short suggestion suffix for lookup failures.

    Parameters
    ----------
    name:
        Lookup string supplied by the user.
    suggestions:
        Candidate layer labels.

    Returns
    -------
    str
        Human-readable lookup error.
    """

    if suggestions:
        suggestion_str = ", ".join(repr(item) for item in suggestions)
        return f"Layer {name!r} not found. Did you mean {suggestion_str}?"
    return f"Layer {name!r} not found."


def _out_from_log(trace: Trace, layer: str) -> _torch.Tensor:
    """Return a saved out from a layer lookup.

    Parameters
    ----------
    trace:
        Log containing saved outs.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.

    Returns
    -------
    torch.Tensor
        Saved layer out.

    Raises
    ------
    ValueError
        If the layer cannot be resolved or has no saved out.
    """

    try:
        layer_log = trace[layer]
    except (KeyError, ValueError) as exc:
        suggestions = trace.find_layers(layer) if hasattr(trace, "find_layers") else []
        raise ValueError(_did_you_mean_message(layer, suggestions)) from exc

    import torch

    out = getattr(layer_log, "out", None)
    if out is None:
        raise ValueError(f"Layer {layer!r} resolved but has no saved out.")
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Layer {layer!r} out is not a torch.Tensor.")
    return out


def _normalize_extract_layers(layers: _Iterable[str] | _Mapping[str, str]) -> dict[str, str]:
    """Normalize list or mapping layer specs to ``output_key -> lookup``.

    Parameters
    ----------
    layers:
        List of layer lookups or mapping from user label to layer lookup.

    Returns
    -------
    dict[str, str]
        Normalized extraction plan.
    """

    if isinstance(layers, _Mapping):
        return {str(label): str(pattern) for label, pattern in layers.items()}
    return {str(layer): str(layer) for layer in layers}


def _matching_saved_layer_labels(trace: Trace, pattern: str) -> list[str]:
    """Return saved layer labels matching an extraction pattern.

    Parameters
    ----------
    trace:
        Log containing candidate layer labels.
    pattern:
        Exact label or substring pattern.

    Returns
    -------
    list[str]
        Matching no-pass layer labels in execution order.
    """

    if pattern in trace.layer_dict_all_keys:
        return [pattern]
    if pattern in trace.layer_logs:
        return [pattern]
    lower_pattern = pattern.lower()
    matches = [
        label
        for label in trace.layer_labels
        if lower_pattern in label.lower() and label in trace.saved_ops
    ]
    if matches:
        return matches
    try:
        resolved = trace[pattern]
    except (KeyError, ValueError):
        return []
    label = getattr(resolved, "layer_label", pattern)
    return [str(label)]


def pluck(model: _nn.Module, x: _Any, layer: str, stop_after: _Any | None = None) -> _torch.Tensor:
    """Return the saved out for one layer.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.
        Resolves strictly to one result; ambiguous lookups raise ``ValueError``.
    stop_after:
        Experimental stop-early site. Currently validated for ``pluck`` and
        captured via the normal safe full-forward path.

    Returns
    -------
    torch.Tensor
        Saved out for the requested layer.

    Raises
    ------
    ValueError
        If ``layer`` does not resolve or did not produce a saved tensor.
    """

    from .experimental import _active_stop_after_site
    from .options import CaptureOptions

    _ = stop_after if stop_after is not None else _active_stop_after_site()
    trace = _resolve_top_level("trace")(
        model,
        x,
        capture=CaptureOptions(layers_to_save=[layer]),
    )
    return _out_from_log(trace, layer)


def _extract_layers_with_trace(
    model: _nn.Module,
    x: _Any,
    layers: _Iterable[str] | _Mapping[str, str],
) -> tuple[Trace, dict[str, _torch.Tensor], dict[str, _Any]]:
    """Run one selective capture and resolve the requested layers.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layers:
        Either a list of layer lookups or a mapping of ``user_label -> layer_lookup``.

    Returns
    -------
    tuple[Trace, dict[str, torch.Tensor], dict[str, Any]]
        The capture trace, the saved outs keyed as :func:`extract` keys them,
        and the resolved ``Layer`` views under the same keys.

    Raises
    ------
    ValueError
        If a lookup does not resolve or did not produce a saved tensor.
    """

    from .options import CaptureOptions as _LazyCaptureOptions

    layer_plan = _normalize_extract_layers(layers)
    trace = _resolve_top_level("trace")(
        model,
        x,
        capture=_LazyCaptureOptions(
            layers_to_save=list(layer_plan.values()),
        ),
    )
    outputs: dict[str, _torch.Tensor] = {}
    views: dict[str, _Any] = {}
    if isinstance(layers, _Mapping):
        for label, pattern in layer_plan.items():
            outputs[label] = _out_from_log(trace, pattern)
            views[label] = trace[pattern]
        return trace, outputs, views

    for pattern in layer_plan.values():
        matches = _matching_saved_layer_labels(trace, pattern)
        if not matches:
            suggestions = trace.find_layers(pattern)
            raise ValueError(_did_you_mean_message(pattern, suggestions))
        for match in matches:
            outputs[match] = _out_from_log(trace, match)
            views[match] = trace[match]
    return trace, outputs, views


def extract(
    model: _nn.Module,
    x: _Any,
    layers: _Iterable[str] | _Mapping[str, str],
) -> dict[str, _torch.Tensor]:
    """Return saved outs for many layers.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layers:
        Either a list of layer lookups or a mapping of ``user_label -> layer_lookup``.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from user labels to outs for mapping inputs, or from
        resolved layer labels to outs for list inputs.
    """

    _trace, outputs, _views = _extract_layers_with_trace(model, x, layers)
    return outputs


def bundle(*args: _Any, **kwargs: _Any) -> Bundle:
    """Construct a TorchLens Bundle.

    Parameters
    ----------
    *args, **kwargs:
        Forwarded to :class:`torchlens.intervention.bundle.Bundle`.

    Returns
    -------
    Bundle
        Constructed Bundle.
    """

    return _resolve_top_level("Bundle")(*args, **kwargs)


class _TorchLensModule(_types.ModuleType):
    """Protect top-level callables whose names collide with submodules."""

    def __setattr__(self, name: str, value: _Any) -> None:
        """Keep the public ``bundle`` constructor after its package is imported.

        Parameters
        ----------
        name:
            Attribute name being assigned by Python's import machinery or a caller.
        value:
            Value being assigned.
        """

        if (
            name == "bundle"
            and isinstance(value, _types.ModuleType)
            and value.__name__ == "torchlens.bundle"
            and callable(self.__dict__.get(name))
        ):
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _TorchLensModule


__all__ = [
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
    # F39 unhide (conflict-ledger row 8: the promotion claimed by five memos):
    # the streaming-statistics namespace and its dataloader aggregation door
    # were shipped, documented-unstable, and root-reachable but undeclared.
    "stats",
    "aggregate",
]

# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute -- the last of the three typing-import leaks the oracles
# panel found on the public surface (``Any`` and ``TYPE_CHECKING`` now enter
# underscore-aliased). Nothing reads the binding (the future feature is a
# compile-time flag), so unbind it; attribute access falls through to
# ``__getattr__`` and raises AttributeError like every other undeclared name.
# Gated by tests/oracles (reachable-surface walk + classification).
del annotations

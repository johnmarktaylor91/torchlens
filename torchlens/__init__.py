"""TorchLens - extract outs and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace -- and no
torch import at all until first use (agent memo 3.11: tier-0 CLI verbs and
manifest preflights stay torch-free). Torch functions are wrapped lazily on
the first call to ``trace()`` and stay wrapped afterward. TorchLens 2.0 keeps
the top-level namespace intentionally small; historical spellings live in
their owning submodules.

For AI agents: ``trace.to_agent_json()``, ``trace.explain(max_tokens=N)``,
``torchlens.agent.guide()``, and the CLI
(``python -m torchlens --help``) are the machine-facing doors.
"""

from __future__ import annotations

from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import TYPE_CHECKING as _TYPE_CHECKING, Any as _Any

if _TYPE_CHECKING:
    import torch as _torch
    from torch import nn as _nn

__version__ = "2.36.2"

if _TYPE_CHECKING:
    from ._data_classes.trace import Trace

_LAZY_ATTRS = {
    # Import cold-start laziness (P4, rebaselined 2026-08-19): the former
    # eager import block (options/captured_run+ir/observers/quantities/errors
    # and their transitive chains) is fully deferred behind these rows -- the
    # marginal-import guard in tests/test_import_hygiene.py holds the line.
    # Agent surface (F29): the read-only inspection core. Deliberately NOT in
    # __all__ (frozen root budget); reachable as tl.agent per the docs.
    "agent": ("torchlens.agent", None),
    "errors": ("torchlens.errors", None),
    "observers": ("torchlens.observers", None),
    "options": ("torchlens.options", None),
    "to_disk": ("torchlens.options", "to_disk"),
    "Bundle": ("torchlens.bundle", "Bundle"),
    "Layer": ("torchlens._data_classes.layer", "Layer"),
    "Op": ("torchlens._data_classes.op", "Op"),
    "Recording": ("torchlens.recording.types", "Recording"),
    "Trace": ("torchlens._data_classes.trace", "Trace"),
    "add": ("torchlens.intervention.predicates", "add"),
    "attribution": ("torchlens.attribution", None),
    # Entry/facade repair (workstream A10; WT1 A-VI item 27, neuro memo item
    # 2): the integration and appliance namespaces were absent from this map,
    # so `tl.bridge` resolved only after an unrelated capture side effect
    # imported it and `tl.callbacks` never resolved at all. All four package
    # __init__ modules are import-inert by design (foreign dependencies stay
    # deferred behind their own facades).
    "bridge": ("torchlens.bridge", None),
    "callbacks": ("torchlens.callbacks", None),
    "neuro": ("torchlens.neuro", None),
    "hook_grad": ("torchlens.intervention.helpers", "hook_grad"),
    "clamp": ("torchlens.intervention.helpers", "clamp"),
    "compat": ("torchlens.compat", None),
    "label_contains": ("torchlens.intervention.selectors", "label_contains"),
    # Dataset extraction (D7/V5): the implementation module is lazy so the
    # manifest/resume machinery costs nothing until first use.
    "dataset_extraction": ("torchlens.dataset_extraction", None),
    "extract_dataset": ("torchlens.dataset_extraction", "extract_dataset"),
    "distributed": ("torchlens.distributed", None),
    "examples": ("torchlens.examples", None),
    "experimental": ("torchlens.experimental", None),
    "export": ("torchlens.export", None),
    "facets": ("torchlens._semantic", "facets"),
    "recording": ("torchlens.recording", None),
    "facet": ("torchlens.intervention.selectors", "facet"),
    "followed_by": ("torchlens.intervention.selectors", "followed_by"),
    "func": ("torchlens.intervention.selectors", "func"),
    "func_transform": ("torchlens.intervention.selectors", "func_transform"),
    "clamp_grad": ("torchlens.intervention.helpers", "clamp_grad"),
    "clip_grad": ("torchlens.intervention.helpers", "clip_grad"),
    "grad_fn": ("torchlens.intervention.selectors", "grad_fn"),
    "grad_input": ("torchlens.intervention.selectors", "grad_input"),
    "add_grad_noise": ("torchlens.intervention.helpers", "add_grad_noise"),
    "grad_output": ("torchlens.intervention.selectors", "grad_output"),
    "scale_grad": ("torchlens.intervention.helpers", "scale_grad"),
    "zero_grad": ("torchlens.intervention.helpers", "zero_grad"),
    "fingerprint": ("torchlens.fingerprint", None),
    "head": ("torchlens.intervention.selectors", "head"),
    "in_backward_pass": ("torchlens.intervention.selectors", "in_backward_pass"),
    "in_module": ("torchlens.intervention.selectors", "in_module"),
    "model_input": ("torchlens.intervention.selectors", "model_input"),
    "intervention": ("torchlens.intervention", None),
    "io": ("torchlens.io", None),
    "load": ("torchlens._io.bundle", "load"),
    "label": ("torchlens.intervention.selectors", "label"),
    "mean_ablate": ("torchlens.intervention.helpers", "mean_ablate"),
    "merge_ranks": ("torchlens._merged", "merge_ranks"),
    "merge_report": ("torchlens._merged", "merge_report"),
    "module": ("torchlens.intervention.selectors", "module"),
    "add_noise": ("torchlens.intervention.helpers", "add_noise"),
    "model_output": ("torchlens.intervention.selectors", "model_output"),
    "debug": ("torchlens.debug", None),
    "report": ("torchlens.report", None),
    "repgeom": ("torchlens.repgeom", None),
    "preceded_by": ("torchlens.intervention.selectors", "preceded_by"),
    "project_out": ("torchlens.intervention.helpers", "project_out"),
    "project_onto": ("torchlens.intervention.helpers", "project_onto"),
    "record": ("torchlens.recording._record_one_shot", "record"),
    "receptive_field": ("torchlens.receptive_field", None),
    "label_matches": ("torchlens.intervention.selectors", "label_matches"),
    "clear_cache": ("torchlens._user_funcs", "clear_cache"),
    "release_model": ("torchlens._user_funcs", "release_model"),
    "replace_with": ("torchlens.intervention.predicates", "replace_with"),
    "scale": ("torchlens.intervention.helpers", "scale"),
    "summary": ("torchlens._user_funcs", "summary"),
    "splice_module": ("torchlens.intervention.helpers", "splice_module"),
    "stats": ("torchlens.stats", None),
    "steer": ("torchlens.intervention.helpers", "steer"),
    "sweep": ("torchlens.intervention.sweep", "sweep"),
    "trace": ("torchlens._user_funcs", "trace"),
    # Built-in activation transforms (transforms memo s7 home; the submodule
    # spelling tl.transforms is DOCUMENTED-UNSTABLE — the collision with the
    # input-side transform= slot is a named naming-sprint item).
    "transforms": ("torchlens.transforms", None),
    "validate": ("torchlens.validation.consolidated", "validate"),
    "validation": ("torchlens.validation", None),
    "types": ("torchlens.types", None),
    "appliances": ("torchlens.appliances", None),
    "backends": ("torchlens.backends", None),
    "viz": ("torchlens.viz", None),
    "site": ("torchlens.intervention.selectors", "site"),
    "at_step": ("torchlens.intervention.selectors", "at_step"),
    "when": ("torchlens.intervention.spec", "when"),
    "where": ("torchlens.intervention.selectors", "where"),
    "without_op": ("torchlens.intervention.selectors", "without_op"),
    "zero_ablate": ("torchlens.intervention.helpers", "zero_ablate"),
    "Edit": ("torchlens.intervention.types", "Edit"),
    "patch_from": ("torchlens.intervention.helpers", "patch_from"),
    "Recipe": ("torchlens.intervention.spec", "Recipe"),
    "positions": ("torchlens.intervention._position_edits", "positions"),
    # L6 selection algebra (DOCUMENTED-UNSTABLE pending naming-session
    # ratification; provisional-name protocol).
    "Selector": ("torchlens._selection", "Selector"),
    "Selection": ("torchlens._selection", "Selection"),
    "units": ("torchlens._selection", "units"),
    "param": ("torchlens._selection", "param"),
    "random_like": ("torchlens._selection", "random_like"),
    # L6 value-based + statistical producers (DOCUMENTED-UNSTABLE pending
    # naming-session ratification; provisional-name protocol).
    "top_k": ("torchlens._selection_values", "top_k"),
    "top_fraction": ("torchlens._selection_values", "top_fraction"),
    "threshold": ("torchlens._selection_values", "threshold"),
    "sign": ("torchlens._selection_values", "sign"),
    "dead": ("torchlens._selection_values", "dead"),
    "saturated": ("torchlens._selection_values", "saturated"),
    "low_variance": ("torchlens._selection_values", "low_variance"),
    # L6 graph-structural producers (DOCUMENTED-UNSTABLE pending
    # naming-session ratification; provisional-name protocol).
    "neighborhood": ("torchlens._selection_graph", "neighborhood"),
    "between": ("torchlens._selection_graph", "between"),
    # L6 comparative producers: differential + cross-pass (DOCUMENTED-UNSTABLE
    # pending naming-session ratification; provisional-name protocol).
    "changed": ("torchlens._selection_compare", "changed"),
    "top_changed": ("torchlens._selection_compare", "top_changed"),
    "stable_across_passes": ("torchlens._selection_compare", "stable_across_passes"),
    "pass_variance": ("torchlens._selection_compare", "pass_variance"),
    # L6 subspace producer (DOCUMENTED-UNSTABLE pending naming-session
    # ratification; provisional-name protocol).
    "subspace": ("torchlens._selection_subspace", "subspace"),
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
# torchlens._utils.facade). The redirect table is the teaching compensator for
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
    "aggregate": "use torchlens.stats.aggregate",
    "batched_extract": "use torchlens.extract_dataset",
    "AtenOp": "use torchlens.types.AtenOp",
    "Container": "use torchlens.types.Container",
    "register_container": "use torchlens.types.register_container",
    "span": "use torchlens.observers.span",
    "tap": "use torchlens.observers.tap",
    "assert_unchanged": "use torchlens.fingerprint.assert_unchanged",
    "Quantity": "use torchlens.types.Quantity",
    "Bytes": "use torchlens.types.Bytes",
    "Duration": "use torchlens.types.Duration",
    "Flops": "use torchlens.types.Flops",
    "Macs": "use torchlens.types.Macs",
    "brainpipe": "use torchlens.neuro (plan_extraction, ExtractionPlan, PlanSite, export_npz)",
    "Buffer": "use torchlens.types.Buffer",
    "build_render_audit": "removed from the public surface; use torchlens.Trace.draw (trace.draw(vis_renderer='dagua'))",
    "check_metadata_invariants": "use torchlens.validation.check_metadata_invariants",
    "check_spec_compat": "use torchlens.intervention.check_spec_compat",
    "cleanup_tmp": "use torchlens.io.cleanup_tmp",
    "draw_backward": "use torchlens.Trace.draw_backward (trace.draw_backward())",
    "draw_combined": "use torchlens.Trace.draw_combined (trace.draw_combined())",
    "draw_model_graph": "use torchlens.Trace.draw (trace.draw())",
    "FuncCallLocation": "use torchlens.types.FuncCallLocation",
    "get_model_metadata": "use torchlens.trace(model, x, save='none') -- a metadata-only capture is a trace that saves no payloads",
    "get_model_structure": "use torchlens.summary(model, x) for the module tree, or torchlens.trace(model, x).modules",
    "GradFn": "use torchlens.types.GradFn",
    "GradFnAccessor": "use torchlens._data_classes.grad_fn.GradFnAccessor",
    "GradFnCall": "use torchlens.types.GradFnCall",
    "intervening": "use torchlens.without_op -- intervening was renamed",
    "LayerAccessor": "use torchlens._data_classes._layer_accessors.LayerAccessor",
    "list_logs": "use torchlens.io.list_traces",
    "load_intervention_spec": "use torchlens.io.load_intervention_spec",
    "log_forward_pass": "use torchlens.trace(model, x) -- log_forward_pass was the pre-2.0 capture verb",
    "log_model_metadata": "use torchlens.trace(model, x, save='none') -- a metadata-only capture is a trace that saves no payloads",
    "mechinterp": "import torchlens.mechinterp, then use torchlens.mechinterp.edge_scores and its siblings -- a submodule, not a root name",
    "MetadataInvariantError": "use torchlens.errors.MetadataInvariantError",
    "ModelHistory": "use torchlens.Trace -- ModelHistory was renamed ModelLog, then Trace",
    "ModelLog": "use torchlens.Trace -- ModelLog was renamed Trace",
    "Module": "use torchlens.types.Module",
    "ModuleAccessor": "use torchlens._data_classes.module.ModuleAccessor",
    "ModuleCall": "use torchlens.types.ModuleCall",
    "ModuleInputSnapshot": "use torchlens.types.ModuleInputSnapshot",
    "NodeSpec": "use torchlens.experimental.dagua.NodeSpec",
    "Param": "use torchlens.types.Param",
    "peek": "use torchlens.pluck",
    "PreHookEffect": "use torchlens.types.PreHookEffect",
    "fastlog": "use torchlens.recording -- fastlog was renamed recording",
    "preview_fastlog": "removed from the public surface; use torchlens.recording.dry_run to see which sites a predicate keeps",
    "record_span": "use torchlens.observers.span",
    "rehydrate_nested": "use torchlens.io.materialize_nested",
    "render_graph": "use torchlens.Trace.draw (trace.draw())",
    "render_lines_to_html": "removed from the public surface; use torchlens.Trace.draw (trace.draw())",
    "render_model_graph": "use torchlens.Trace.draw (trace.draw())",
    "render_trace_with_dagua": "use torchlens.Trace.draw (trace.draw(vis_renderer='dagua'))",
    "replay": "use torchlens.Trace.do (do is the one intervention verb)",
    "replay_from": "use torchlens.Trace.do (do is the one intervention verb)",
    "rerun": "use torchlens.Trace.do with engine='rerun'",
    "resample_ablate": "use torchlens.intervention.scramble_elements -- resample_ablate was renamed",
    "reset_naming_counter": "use torchlens.io.reset_naming_counter",
    "resolve_sites": "use trace[selector] -- it returns every matching site as a torchlens.Selection",
    "load_extraction": "use torchlens.load(path) -- it detects a saved extraction directory",
    "save": "use torchlens.Trace.save (trace.save(path)) -- objects save themselves; torchlens.load(path) reads a saved artifact back",
    "save_intervention": "use torchlens.io.save_intervention",
    "semantic": "use torchlens.facets for facet functions and torchlens.mechinterp for the logit lens -- the semantic package is private",
    "show_model_graph": "use torchlens.Trace.draw (trace.draw())",
    "show_model_structure": "use torchlens.summary(model, x) for the module tree, or torchlens.trace(model, x).modules",
    "SiteTable": "use torchlens.Selection -- trace[selector] returns one",
    "SpecCompat": "use torchlens.types.SpecCompat",
    "StreamingOptions": "use torchlens.options.StorageOptions",
    "suppress_mutate_warnings": "use torchlens.io.suppress_mutate_warnings",
    "TargetManifestDiff": "use torchlens.types.TargetManifestDiff",
    "TensorInputObservation": "use torchlens.types.TensorInputObservation",
    "TorchLensPostfuncError": "use torchlens.errors.UserTransformError",
    "trace_to_dagua_graph": "use torchlens.experimental.dagua.trace_to_dagua_graph",
    "TraceState": "use torchlens.io.TraceState",
    "TrainingModeConfigError": "use torchlens.errors.TrainingModeConfigError",
    "transformer": "import torchlens.transformer, then use torchlens.transformer.AttentionView and its siblings -- a submodule, not a root name; the transformer renderers are torchlens.viz names (torchlens.viz.logit_lens, torchlens.viz.attention)",
    "tviz": "renamed; use torchlens.transformer (import torchlens.transformer) for the kit; the transformer renderers are torchlens.viz names (torchlens.viz.logit_lens, torchlens.viz.attention)",
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

    from ._utils.facade import resolve_facade_attr

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

    from ._utils.facade import facade_dir

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

    from ._errors import AmbiguousOpLookupError
    from ._trace_selector_helpers import canonical_module_site_key

    try:
        layer_log = trace[canonical_module_site_key(trace, layer) or layer]
    except AmbiguousOpLookupError:
        # An ambiguity names its candidates; a "not found" would hide them.
        raise
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

    from ._errors import AmbiguousOpLookupError
    from ._trace_selector_helpers import canonical_module_site_key

    if pattern in trace.layer_dict_all_keys:
        return [pattern]
    if pattern in trace.layer_logs:
        return [pattern]
    if canonical_module_site_key(trace, pattern) is not None:
        # A wrapper-relative or dot-flattened module name keys by the caller's spelling.
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
    except AmbiguousOpLookupError:
        raise
    except (KeyError, ValueError):
        return []
    label = getattr(resolved, "layer_label", pattern)
    return [str(label)]


def pluck(
    model: _nn.Module, inputs: _Any, layer: str, stop_after: _Any | None = None
) -> _torch.Tensor:
    """Return the saved out for one layer.

    Parameters
    ----------
    model:
        PyTorch model to run.
    inputs:
        Positional input argument or argument container for ``model.forward``.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.
        Resolves strictly to one result; ambiguous lookups raise ``ValueError``.
    stop_after:
        Stop-early site, passed to the capture as
        ``CaptureOptions(stop_after=...)``: the forward halts inclusively after
        the site, and an unsupported spelling refuses typed. ``None`` reads the
        ambient ``tl.experimental.stop_after`` site, if one is active.

    Returns
    -------
    torch.Tensor
        Saved out for the requested layer.

    Raises
    ------
    ValueError
        If ``layer`` does not resolve or did not produce a saved tensor.
    """

    from .options import CaptureOptions

    # an explicit site rides the capture options, where the trace
    # validates it; without one the trace reads the ambient site itself.
    capture = (
        CaptureOptions(layers_to_save=[layer])
        if stop_after is None
        else CaptureOptions(layers_to_save=[layer], stop_after=stop_after)
    )
    trace = _resolve_top_level("trace")(model, inputs, capture=capture)
    return _out_from_log(trace, layer)


def _extract_facet_plan(layers: _Any) -> dict[str, _Any]:
    """Return ``output_key -> facet selector`` for the facet selectors in an extract request.

    A list entry keys by the selector's spelling; a mapping entry by the user label.
    """

    from .intervention.selectors import FacetSelector

    items = layers.items() if isinstance(layers, _Mapping) else ((None, v) for v in layers)
    return {
        repr(value) if label is None else str(label): value
        for label, value in items
        if isinstance(value, FacetSelector)
    }


def _extract_with_facets(
    model: _nn.Module,
    x: _Any,
    layers: _Any,
    facet_plan: dict[str, _Any],
    *,
    detach_saved_activations: bool,
) -> tuple[Trace, dict[str, _torch.Tensor], dict[str, _Any]]:
    """Extract a request holding facet selectors: each facet's value, as ``trace[selector]`` reads it.

    One facet alone captures with ``save=selector`` (its homes); a mixed or
    several-facet request captures with every op saved. Other entries resolve
    as in a plain extract, from the same trace.
    """

    from .options import CaptureOptions as _LazyCaptureOptions

    capture = _LazyCaptureOptions(detach_saved_activations=detach_saved_activations)
    only = next(iter(facet_plan.values())) if len(facet_plan) == len(layers) == 1 else None
    if only is not None:
        trace = _resolve_top_level("trace")(model, x, capture=capture, save=only)
    else:
        trace = _resolve_top_level("trace")(model, x, capture=capture)
    outputs: dict[str, _torch.Tensor] = {}
    views: dict[str, _Any] = {}
    items = layers.items() if isinstance(layers, _Mapping) else ((None, v) for v in layers)
    for label, value in items:
        key = (
            str(label)
            if label is not None
            else (repr(value) if value in facet_plan.values() else str(value))
        )
        if key in facet_plan:
            facet = trace[facet_plan[key]]
            outputs[key], views[key] = facet.value, facet
        else:
            outputs[key] = _out_from_log(trace, str(value))
            views[key] = trace[str(value)]
    return trace, outputs, views


def _extract_layers_with_trace(
    model: _nn.Module,
    x: _Any,
    layers: _Iterable[str] | _Mapping[str, str],
    *,
    detach_saved_activations: bool = True,
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
    detach_saved_activations:
        Whether the returned outs are detached from the autograd graph.

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

    from ._trace_selector_helpers import canonical_module_site_key
    from .options import CaptureOptions as _LazyCaptureOptions

    facet_plan = _extract_facet_plan(layers)
    if facet_plan:
        return _extract_with_facets(
            model, x, layers, facet_plan, detach_saved_activations=detach_saved_activations
        )
    layer_plan = _normalize_extract_layers(layers)
    trace = _resolve_top_level("trace")(
        model,
        x,
        capture=_LazyCaptureOptions(
            layers_to_save=list(layer_plan.values()),
            detach_saved_activations=detach_saved_activations,
        ),
    )
    outputs: dict[str, _torch.Tensor] = {}
    views: dict[str, _Any] = {}
    if isinstance(layers, _Mapping):
        for label, pattern in layer_plan.items():
            outputs[label] = _out_from_log(trace, pattern)
            views[label] = trace[canonical_module_site_key(trace, pattern) or pattern]
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
    inputs: _Any,
    layers: _Iterable[str] | _Mapping[str, str],
    *,
    detach_saved_activations: bool = True,
) -> dict[str, _torch.Tensor]:
    """Return saved outs for many layers.

    Parameters
    ----------
    model:
        PyTorch model to run.
    inputs:
        Positional input argument or argument container for ``model.forward``.
    layers:
        Either a list of layer lookups or a mapping of ``user_label -> layer_lookup``.
    detach_saved_activations:
        DOCUMENTED-UNSTABLE. Defaults to ``True``: the outs are detached
        copies. ``False`` returns graph-connected outs (``grad_fn`` set
        wherever the model's forward carries autograd history), so a loss
        built from them backpropagates into the model's parameters.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from user labels to outs for mapping inputs, or from
        resolved layer labels to outs for list inputs.
    """

    _trace, outputs, _views = _extract_layers_with_trace(
        model, inputs, layers, detach_saved_activations=detach_saved_activations
    )
    return outputs


__all__ = [
    "trace",
    "release_model",
    "clear_cache",
    "export",
    "fingerprint",
    "recording",
    "facets",
    "record",
    "Recording",
    "load",
    "pluck",
    "extract",
    "extract_dataset",
    "validate",
    "summary",
    "options",
    "to_disk",
    "Trace",
    "Layer",
    "Op",
    "Bundle",
    "add",
    "label",
    "func",
    "func_transform",
    "followed_by",
    "grad_fn",
    "grad_input",
    "grad_output",
    "in_backward_pass",
    "without_op",
    "label_matches",
    "module",
    "model_output",
    "model_input",
    "preceded_by",
    "label_contains",
    "facet",
    "where",
    "in_module",
    "head",
    "clamp",
    "mean_ablate",
    "merge_ranks",
    "merge_report",
    "add_noise",
    "project_out",
    "project_onto",
    "replace_with",
    "scale",
    "splice_module",
    "steer",
    "sweep",
    "zero_ablate",
    "site",
    "at_step",
    "when",
    "hook_grad",
    "clip_grad",
    "add_grad_noise",
    "clamp_grad",
    "scale_grad",
    "zero_grad",
    "Selector",
    "Selection",
    "units",
    "param",
    "random_like",
    "Edit",
    "patch_from",
    "Recipe",
    "positions",
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
]

# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute -- the last of the three typing-import leaks the oracles
# panel found on the public surface (``Any`` and ``TYPE_CHECKING`` now enter
# underscore-aliased). Nothing reads the binding (the future feature is a
# compile-time flag), so unbind it; attribute access falls through to
# ``__getattr__`` and raises AttributeError like every other undeclared name.
# Gated by tests/oracles (reachable-surface walk + classification).
del annotations

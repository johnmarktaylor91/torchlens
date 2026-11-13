"""TorchLens - extract outs and metadata from PyTorch models.

Importing torchlens has **no side effects** on the torch namespace -- and no
torch import at all until first use (tier-0 CLI verbs and
manifest preflights stay torch-free). Torch functions are wrapped lazily on
the first call to ``trace()`` and stay wrapped afterward. TorchLens keeps the
top-level namespace intentionally small: other names live in their owning
submodules, and a removed top-level spelling raises an AttributeError that
names its new home.

For AI agents: start with ``torchlens.agent.guide()`` (or ``python -m torchlens guide``), a
cookbook of recipes; also ``trace.to_agent_json()``, ``trace.explain()`` and the CLI.
"""

from __future__ import annotations

import typing as _typing

if _typing.TYPE_CHECKING:
    from ._data_classes.trace import Trace

#: Released together with the io copy in _vocab/_package_version.py (test_version_coherence).
__version__ = "3.0.3"

_LAZY_ATTRS = {
    # Import cold-start laziness (rebaselined 2026-08-19): the former
    # eager import block (options/captured_run+ir/observers/quantities/errors
    # and their transitive chains) is fully deferred behind these rows -- the
    # marginal-import guard in tests/test_import_hygiene.py holds the line.
    # Agent surface: the read-only inspection core. Deliberately NOT in
    # __all__ (frozen root budget); reachable as tl.agent per the docs.
    "agent": ("torchlens.agent", None),
    "errors": ("torchlens.errors", None),
    "observers": ("torchlens.observers", None),
    "options": ("torchlens.options", None),
    "to_disk": ("torchlens.options", "to_disk"),
    "Bundle": ("torchlens.bundle", "Bundle"),
    "Layer": ("torchlens._data_classes.layer", "Layer"),
    "Op": ("torchlens._data_classes.op", "Op"),
    "Recording": ("torchlens.recording._types", "Recording"),
    "Trace": ("torchlens._data_classes.trace", "Trace"),
    "add": ("torchlens.intervention._predicates", "add"),
    "attribution": ("torchlens.attribution", None),
    # Entry/facade repair: the integration and appliance namespaces were absent from this map,
    # so `tl.bridge` resolved only after an unrelated capture side effect
    # imported it and `tl.callbacks` never resolved at all. All four package
    # __init__ modules are import-inert by design (foreign dependencies stay
    # deferred behind their own facades).
    "bridge": ("torchlens.bridge", None),
    "callbacks": ("torchlens.callbacks", None),
    "neuro": ("torchlens.neuro", None),
    "clamp": ("torchlens.intervention._helpers", "clamp"),
    "compat": ("torchlens.compat", None),
    "label_contains": ("torchlens._selection.selectors", "label_contains"),
    # Dataset extraction: the implementation module is lazy so the
    # manifest/resume machinery costs nothing until first use.
    "dataset_extraction": ("torchlens.dataset_extraction", None),
    "extract_dataset": ("torchlens.dataset_extraction", "extract_dataset"),
    "distributed": ("torchlens.distributed", None),
    "examples": ("torchlens.examples", None),
    "experimental": ("torchlens.experimental", None),
    "export": ("torchlens.export", None),
    "facets": ("torchlens._semantic", "facets"),
    "recording": ("torchlens.recording", None),
    "facet": ("torchlens._selection.selectors", "facet"),
    "followed_by": ("torchlens._selection.selectors", "followed_by"),
    "func": ("torchlens._selection.selectors", "func"),
    "func_transform": ("torchlens._selection.selectors", "func_transform"),
    "grad": ("torchlens._selection.selectors", "grad"),
    "clip_norm": ("torchlens.intervention._helpers", "clip_norm"),
    "grad_fn": ("torchlens._selection.selectors", "grad_fn"),
    "grad_input": ("torchlens._selection.selectors", "grad_input"),
    "grad_output": ("torchlens._selection.selectors", "grad_output"),
    "fingerprint": ("torchlens.fingerprint", None),
    "head": ("torchlens._selection.selectors", "head"),
    "in_backward_pass": ("torchlens._selection.selectors", "in_backward_pass"),
    "in_module": ("torchlens._selection.selectors", "in_module"),
    "model_input": ("torchlens._selection.selectors", "model_input"),
    "intervention": ("torchlens.intervention", None),
    "io": ("torchlens.io", None),
    "load": ("torchlens._io.bundle", "load"),
    # The extract front door (pluck, extract) lives with the extraction engine.
    "pluck": ("torchlens._extraction._front_door", "pluck"),
    "extract": ("torchlens._extraction._front_door", "extract"),
    "label": ("torchlens._selection.selectors", "label"),
    "mean_ablate": ("torchlens.intervention._helpers", "mean_ablate"),
    "merge_ranks": ("torchlens._merged", "merge_ranks"),
    "merge_report": ("torchlens._merged", "merge_report"),
    "module": ("torchlens._selection.selectors", "module"),
    "add_noise": ("torchlens.intervention._helpers", "add_noise"),
    "model_output": ("torchlens._selection.selectors", "model_output"),
    "debug": ("torchlens.debug", None),
    "report": ("torchlens.report", None),
    "repgeom": ("torchlens.repgeom", None),
    "preceded_by": ("torchlens._selection.selectors", "preceded_by"),
    "project_out": ("torchlens.intervention._helpers", "project_out"),
    "project_onto": ("torchlens.intervention._helpers", "project_onto"),
    "record": ("torchlens.recording._record_one_shot", "record"),
    "receptive_field": ("torchlens.receptive_field", None),
    "label_matches": ("torchlens._selection.selectors", "label_matches"),
    "all_sites": ("torchlens._selection.universal", "all_sites"),
    "clear_cache": ("torchlens._user_funcs", "clear_cache"),
    "release_model": ("torchlens._user_funcs", "release_model"),
    "replace_with": ("torchlens.intervention._predicates", "replace_with"),
    "scale": ("torchlens.intervention._helpers", "scale"),
    "summary": ("torchlens._user_funcs", "summary"),
    "splice_module": ("torchlens.intervention._helpers", "splice_module"),
    "stats": ("torchlens.stats", None),
    "steer": ("torchlens.intervention._helpers", "steer"),
    "sweep": ("torchlens.intervention._sweep", "sweep"),
    "trace": ("torchlens._user_funcs", "trace"),
    "draw": ("torchlens._user_funcs", "draw"),
    # Built-in activation transforms (transforms memo s7 home; the submodule
    # spelling tl.transforms is DOCUMENTED-UNSTABLE — the collision with the
    # input-side transform= slot is a named naming-sprint item).
    "transforms": ("torchlens.transforms", None),
    "validate": ("torchlens.validation._consolidated", "validate"),
    "validation": ("torchlens.validation", None),
    "types": ("torchlens.types", None),
    "appliances": ("torchlens.appliances", None),
    # The detector registries the glossary spells tl.postprocessing/preprocessing.*.
    "postprocessing": ("torchlens.postprocessing", None),
    "preprocessing": ("torchlens.preprocessing", None),
    "backends": ("torchlens.backends", None),
    "viz": ("torchlens.viz", None),
    "site": ("torchlens._selection.selectors", "site"),
    "at_step": ("torchlens._selection.selectors", "at_step"),
    "when": ("torchlens.intervention._spec", "when"),
    "where": ("torchlens._selection.selectors", "where"),
    "without_op": ("torchlens._selection.selectors", "without_op"),
    "zero_ablate": ("torchlens.intervention._helpers", "zero_ablate"),
    "Edit": ("torchlens.intervention._types", "Edit"),
    "patch_from": ("torchlens.intervention._helpers", "patch_from"),
    "Recipe": ("torchlens.intervention._spec", "Recipe"),
    "positions": ("torchlens.intervention._position_edits", "positions"),
    # L6 selection algebra (DOCUMENTED-UNSTABLE; provisional-name protocol).
    "Selector": ("torchlens._selection.selection", "Selector"),
    "Selection": ("torchlens._selection.selection", "Selection"),
    "units": ("torchlens._selection.selection", "units"),
    "param": ("torchlens._selection.selection", "param"),
    "random_like": ("torchlens._selection.selection", "random_like"),
    # L6 value-based + statistical producers (DOCUMENTED-UNSTABLE; provisional-name protocol).
    "top_k": ("torchlens._selection.values", "top_k"),
    "top_fraction": ("torchlens._selection.values", "top_fraction"),
    "threshold": ("torchlens._selection.values", "threshold"),
    "sign": ("torchlens._selection.values", "sign"),
    "dead": ("torchlens._selection.values", "dead"),
    "saturated": ("torchlens._selection.values", "saturated"),
    "low_variance": ("torchlens._selection.values", "low_variance"),
    # L6 graph-structural producers (DOCUMENTED-UNSTABLE; provisional-name protocol).
    "neighborhood": ("torchlens._selection.graph", "neighborhood"),
    "between": ("torchlens._selection.graph", "between"),
    # L6 differential + cross-pass producers (DOCUMENTED-UNSTABLE; provisional-name protocol).
    "changed": ("torchlens._selection.compare", "changed"),
    "top_changed": ("torchlens._selection.compare", "top_changed"),
    "stable_across_passes": ("torchlens._selection.compare", "stable_across_passes"),
    "pass_variance": ("torchlens._selection.compare", "pass_variance"),
    # L6 subspace producer (DOCUMENTED-UNSTABLE; provisional-name protocol).
    "subspace": ("torchlens._selection.subspace", "subspace"),
}


def _resolve_top_level(name: str) -> _typing.Any:
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


# Five-step facade tables (mechanism owned by
# torchlens._utils.facade). The redirect table is the teaching compensator for
# the 2026-08 shim-removal pass (docs/migration/v2.0_api_changes.md): every
# removed top-level spelling raises a typed ``facade_redirect`` AttributeError
# naming its canonical home instead of a bare "no attribute".
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
    "build_render_audit": "removed from the public surface; use torchlens.Trace.draw",
    "check_metadata_invariants": "use torchlens.validation.check_metadata_invariants",
    "check_spec_compat": "use torchlens.intervention.check_spec_compat",
    "cleanup_tmp": "use torchlens.io.cleanup_tmp",
    "draw_backward": "use torchlens.Trace.draw (trace.draw(graph='backward'))",
    "draw_combined": "use torchlens.Trace.draw (trace.draw(graph='combined'))",
    "do": "use torchlens.Trace.do (trace.do(site, edit, model=model, inputs=x)) -- do is a Trace method, not a root verb",
    "draw_model_graph": "use torchlens.Trace.draw (trace.draw())",
    "show_bundle_graph": "use torchlens.Bundle.draw (bundle.draw(path)): the merged bundle graph",
    "FuncCallLocation": "use torchlens.types.FuncCallLocation",
    "get_model_metadata": "use torchlens.trace(model, x, save='none') -- a metadata-only capture is a trace that saves no payloads",
    "get_model_structure": "use torchlens.summary(model, x) for the module tree, or torchlens.trace(model, x).modules",
    "GradFn": "use torchlens.types.GradFn",
    "GradFnAccessor": "use torchlens.Trace.autograd (trace.autograd.grad_fns returns a GradFnAccessor; the class has no public import)",
    "GradFnCall": "use torchlens.types.GradFnCall",
    "intervening": "use torchlens.without_op -- intervening was renamed",
    "LayerAccessor": "use torchlens.Trace.layers (trace.layers returns a LayerAccessor; the class has no public import)",
    "list_logs": "use torchlens.io.list_traces",
    "load_intervention_spec": "use torchlens.io.load_intervention_spec",
    "log_forward_pass": "use torchlens.trace(model, x) -- log_forward_pass was the pre-2.0 capture verb",
    "log_model_metadata": "use torchlens.trace(model, x, save='none') -- a metadata-only capture is a trace that saves no payloads",
    "mechinterp": "import torchlens.mechinterp, then use torchlens.mechinterp.edge_scores and its siblings -- a submodule, not a root name",
    "MetadataInvariantError": "use torchlens.errors.MetadataInvariantError",
    "ModelHistory": "use torchlens.Trace -- ModelHistory was renamed ModelLog, then Trace",
    "ModelLog": "use torchlens.Trace -- ModelLog was renamed Trace",
    "Module": "use torchlens.types.Module",
    "ModuleAccessor": "use torchlens.Trace.modules (trace.modules returns a ModuleAccessor; the class has no public import)",
    "ModuleCall": "use torchlens.types.ModuleCall",
    "ModuleInputSnapshot": "use torchlens.types.ModuleInputSnapshot",
    "NodeSpec": "use torchlens.viz.NodeSpec",
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
    "replay": "use torchlens.Trace.do (do is the one intervention verb)",
    "replay_from": "use torchlens.Trace.do (do is the one intervention verb)",
    "rerun": "use torchlens.Trace.do with engine='rerun'",
    "resample_ablate": "use torchlens.intervention.scramble_elements -- resample_ablate was renamed",
    "push": "use torchlens.Trace.do (trace.do(site, edit)) -- push folded into do, which applies and propagates the edit",
    "push_from": "use torchlens.Trace.do (trace.do(site, edit)) -- push_from folded into do, which applies and propagates the edit",
    "reset_naming_counter": "use torchlens.io.reset_naming_counter",
    "resolve_sites": "use torchlens.Trace indexing (trace[selector]) -- it returns every matching site as a torchlens.Selection",
    "load_extraction": "use torchlens.load(path) -- it detects a saved extraction directory",
    "run": "use torchlens.Trace.run (trace.run(model, x)) -- run is a Trace method, not a root verb",
    "save": "use torchlens.Trace.save (trace.save(path)) -- objects save themselves; torchlens.load(path) reads a saved artifact back",
    "save_intervention": "use torchlens.io.save_intervention",
    "semantic": "use torchlens.facets for facet functions and torchlens.mechinterp for the logit lens -- the semantic package is private",
    "show_model_graph": "use torchlens.Trace.draw (trace.draw())",
    "show_model_structure": "use torchlens.summary(model, x) for the module tree, or torchlens.trace(model, x).modules",
    "SiteTable": "use torchlens.Selection -- trace[selector] returns one",
    "SpecCompat": "use torchlens.types.SpecCompat",
    "StreamingOptions": "use torchlens.options.StorageOptions",
    "suppress_mutate_warnings": "use torchlens.errors.quiet() (or quiet(code, ...) for chosen codes)",
    "swap_with": "use torchlens.intervention._helpers.swap_with (swap_with(other_label)) -- an edit helper under torchlens.intervention._helpers, not a root verb",
    "TargetManifestDiff": "use torchlens.types.TargetManifestDiff",
    "TensorInputObservation": "use torchlens.types.TensorInputObservation",
    "TorchLensPostfuncError": "use torchlens.errors.UserTransformError",
    "TraceState": "use torchlens.io.TraceState",
    "TrainingModeConfigError": "use torchlens.errors.TrainingModeConfigError",
    "transformer": "import torchlens.transformer, then use torchlens.transformer.AttentionView and its siblings -- a submodule, not a root name; the transformer renderers are torchlens.viz names (torchlens.viz.logit_lens, torchlens.viz.attention)",
    "tviz": "renamed; use torchlens.transformer (import torchlens.transformer) for the kit; the transformer renderers are torchlens.viz names (torchlens.viz.logit_lens, torchlens.viz.attention)",
    "unwrap_torch": "use torchlens.backends.torch.wrappers.unwrap_torch",
    "validate_backward_pass": "use torchlens.validation.validate_backward_pass",
    "validate_batch_of_models_and_inputs": "removed: the batch CSV harness has no successor; use torchlens.validate once per model (torchlens.validate(model, x))",
    "validate_forward_pass": "use torchlens.validation.validate_forward_pass",
    "validate_model_activations": "use torchlens.validate(scope='forward')",
    "validate_saved_activations": "use torchlens.validate(scope='saved')",
    "validate_saved_outs": "use torchlens.validate(scope='saved')",
    "VisualizationOptions": "use torchlens.options.VisualizationOptions",
    "wrap_torch": "use torchlens.backends.torch.wrappers.wrap_torch",
    "wrapped": "use torchlens.backends.torch.wrappers.wrapped",
}
_REFUSALS: dict[str, str] = {}


def __getattr__(name: str) -> _typing.Any:
    """Return lazy package attributes through the five-step facade order.

    The order (testable spec): (1) underscore-prefixed
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


__all__ = [
    "trace",
    "draw",
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
    "grad",
    "grad_fn",
    "grad_input",
    "grad_output",
    "in_backward_pass",
    "without_op",
    "label_matches",
    "all_sites",
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
    "clip_norm",
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
    # unhide (conflict-ledger row 8: the promotion claimed by five memos):
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

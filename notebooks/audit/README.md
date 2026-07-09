# TorchLens Human-Facing Audit Notebooks

Coverage-optimized notebooks that exercise EVERY human-facing surface of TorchLens.
These are NOT beginner tutorials (see `notebooks/*.ipynb` for that) — they are a
developer ergonomics review: does every repr/summary/accessor/error message look clean?
Working through them in order should leave no user-experience surprises.

## How to review

The committed `.ipynb` are **output-stripped** by the repo's `nbstripout` git filter, so
executed output is reviewed via generated artifacts (not the notebooks on GitHub):

1. Regenerate + open the executed HTML: run the recipe in `CLAUDE.md` (writes
   `_exports/<notebook>.html`), then scroll each in a browser. _(`_exports/` is gitignored.)_
2. Flip through `visual/visual_audit.pdf` for pixel-level rendering nits
   (regenerate with `python visual/generate_visual_pack.py`).
3. Any cell marked **⚠️ GAP** is a surface that errored or felt awkward — those are the
   action items. GAP cells never crash; they *print* the awkwardness so the suite stays
   green end-to-end.

## Coverage — re-derived from the live surface (2026-07-06)

Coverage is computed programmatically against the code the notebooks ship with (each
setup cell pins `sys.path` to the repo root and asserts on `torchlens.__file__`).

- **`tl.__all__`: 79/79 names covered.** (Notebook 00 enumerates the live list and
  flags any `__all__` entry that fails `getattr`.)
- **Submodule families:** `tl.debug` 8/8 · `tl.export` 17/17 (12 dep-free formats
  executed, tracker formats enumerated) · `tl.attribution` 10/10 · `tl.viz` 11/11
  (incl. `feature_map_node_spec`) · `tl.report` 2/2 · `torchlens.io` 12 of 16 `__all__`
  names exercised (`inspect_tlspec`, `detect_tlspec_format`, `log_model_metadata` +
  its deprecated alias, `list_logs`, `save_intervention`, `load_intervention_spec`,
  `save`, `load`, `PayloadLoadHints`, `JaxPayloadLoadHint`; the remainder —
  `cleanup_tmp`, `rehydrate_nested`, `TraceState`, `suppress_mutate_warnings` — is
  judged plumbing) · `torchlens.errors` hierarchy enumerated + typed-catch
  demonstrated with the payload contract (nb 12) · `torchlens.backends` registry
  enumerated + `UnknownBackendError` surface (nb 16) · `tl.compat` 9/9 (all `from_*` shims invoked, guarded) ·
  `tl.bridge` all 16 adapters enumerated + import-exercised at runtime (captum, hf,
  profiler in depth) · collapse-v2 (`collapse=` float slider, `fold_runs=`,
  `collapse_plan`, `collapse_schedule`) 3/3 · `tl.validate` all 4 scopes ·
  capture-time kwargs (`storage=tl.to_disk`, `streaming`, `hooks`, `layers_to_save`,
  `chunk_size`, `activation_transform`/`grad_transform` (both verified INERT),
  `module_filter`, `random_seed`, `recipes`) demonstrated · vanilla dunder protocol
  (`len`/iteration/membership) on `Trace` (nb 01) and `Bundle` (nb 08).

Notable fixes verified since the June 2026 pass (dropped from GAP cells, kept in the
matrix notes): `Quantity` reprs now carry units; `Recording.n_records` counting;
`Bundle`/`Facet`/`MissingGradient` reprs; `project_onto` batch handling (explicit
`feature_axis`); public `op.container`; `export` in `tl.__all__`.

## Coverage matrix

All 17 notebooks run green. Per-notebook surfaces and the verified rough edges:

| Notebook | Surfaces covered | Green? | Notable rough edges flagged |
|---|---|---|---|
| `00_setup_and_first_capture` | `tl.trace`, `Trace.__repr__`/`__str__`, `summary()` levels, `layer_labels`/`op_labels`, `trace[label]`, `Op.out`, `tl.pluck`, programmatic `tl.__all__` enumeration, `tl.ReentrantTraceError` | yes | **`trace[layer_label]` return type is pass-count-dependent (Op vs Layer)**; `summary()` opens with a forensic fingerprint preamble; re-entrant failure prints a loud stdout banner before raising |
| `01_indexing_and_lookup` | `trace[int/label/substr/module_path]`, `.layers/.ops/.modules/.params/.buffers`, raw labels, `lookup_keys`, ambiguous-lookup error surface | yes | **`tl.AmbiguousOpLookupError` is exported but never raised** (plain `ValueError` comes out); empty accessors print bare `{}` |
| `02_activations_and_metadata` | `op.out`, `.shape`, `.dtype`, `.device_ref`, `.activation_memory`, `Bytes`/`Duration`/`Flops`/`Macs`, `saved_args`, `arg_expressions`, RNG fields, `code_context` | yes | `arg_expressions` population inconsistent; `code_context` is `None` for dynamic source; quantity reprs fixed since June |
| `03_the_data_model` | `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer` (incl. value tracking: `value_at`/`write_versions`), `GradFn`, `GradFnCall`, `show_call_tree`/`find_layers` | yes | `trace[label]` Op-vs-Layer trap; `total_` prefix dropped in Layer DataFrame columns; `Module.to_pandas` is per-layer; call accessors iterate as ints |
| `04_extraction_surfaces` | `tl.extract`, `tl.extract_dataset`, `Container`, `op.container`, `tl.register_container`, `to_pandas`, `output_table` | yes | `op.container` without `capture_container_structure=True` silently degenerate; `Container.summary()==repr()`; `output_table()` needs retained logits |
| `05_save_and_load` | `tl.save`/`tl.load`, `.tlspec` layout, save levels, round-trip, lazy load, `storage=tl.to_disk` streaming capture, intervention-spec persistence, `torchlens.io` inspection helpers | yes | re-save onto an existing bundle fails with an opaque error (no `overwrite=`); outs-less bundles read `.out` as silent `None`; **`tl.to_disk` not in `tl.__all__`**; `list_logs()` name mismatch + duplicate names |
| `06_backward_and_gradients` | backward trace, grad fields on `Op`, `GradFn`/`GradFnCall`, `tl.validate` scopes, `draw_backward` smoke | yes | post-backward substring lookup resolves to `GradFn` not `Op`; `save_grads` needs `backward_ready` too; draw returns DOT source, never the file path |
| `07_intervention` | `find_sites`, all selectors (incl. `tl.regex`), all action categories, `when`/`do`, `push`/`push_from`/`run`, capture-time `hooks=`/`attach_hooks`, `splice_module`, gradient actions | yes | **the entire gradient-action family is inert** (grad_zero/scale/clamp/clip/noise/bwd_hook — zero observable effect under every wiring); **`splice_module` post-transforms module OUTPUT vs documented input semantics**; `output_at`/`without_op` repr bugs; `tl.label` needs `_raw` suffix |
| `08_bundles_and_cross_trace` | `tl.bundle`/`Bundle`, Super* accessors, `bundle.at`, `compare_at`/`joint_metric`/`supergraph`, `sweep`, pairwise diff, `tl.viz.bundle_diff`, bundle save/load | yes | `viz.bundle_diff` carries a hardcoded stale caption; `relationship` blind to input identity; `diff_pair` vs `most_changed` disagree numerically; `sweep` rejects action helpers |
| `09_fastlog_record` | `tl.record`/`fastlog`, `Recording`, `to_trace`, `dry_run`, `repredicate`, `halt`, `save=` vs deprecated aliases | yes | `dry_run`/`repredicate` still `keep_op=`-only; `Recording.__repr__` is an ~85 KB event dump; no "record everything" spelling |
| `10_facets` | `tl.facets` registry (`snapshot`/`info`/`list`/`register` extensibility + per-trace `recipes=`), `tl.facet`/`tl.head` selectors, `FacetView` (`keys`/`menu`/`has`), `Facet.value`/`.grad`, namespace hygiene | yes | **`import torchlens.facets` fails** while `tl.facets` works; **no `__all__`** (dir() leaks typing names); `tl.head()` silent on non-attention models |
| `11_visualization_in_workflow` | `draw` option sweep, **collapse-v2** (string + float slider, `fold_runs=`, `collapse_plan()`, `collapse_schedule()`), customization callbacks (`node_spec_fn`/`skip_fn`/`collapse_fn`), presentation knobs (`for_paper`, `node_label_fields`, `vis_intervention_mode`), `Trace.annotate`, `tl.viz` primitives (all 11 names) | yes | option sweeps tripwired for byte-identical renders (`fold_runs`, `show_containers`, `node_mode='attention'` fallback); `viz.__all__` lists `batch_summary` (a module, not a callable); `node_overlay` rejects callables; `vis_outpath` double-extension footgun |
| `12_validation_stats_reporting` | `tl.validate` (all 4 scopes incl. structured `InterventionValidationReport`), `tl.report.explain` + `log_value`, `tl.tap`/`tl.span`, `summary` levels, **full `tl.debug` family** (8 diagnostics + result classes) | yes | **`summary(level='compute'/'cost')` CRASH on multi-pass models**; bool scopes vs structured intervention scope asymmetry; `verbose=True` prints nothing |
| `13_tabular_export` | `to_pandas` at all three levels, **all 17 `torchlens.export` formats** (12 executed, tracker formats enumerated), schema comparison, round-trips | yes | three disjoint DataFrame schemas; tracker exports fail opaquely when handed a path; `xarray` error names no offending op |
| `14_huggingface_guarded` | Guarded HF text + ViT trace, `tl.autoroute.input` registry, `tl.bridge.hf.trace_text`, attention q/k/v head facets | yes | `Facet.value` (not `.tensor`); SDPA facets need eager attention; `trace_image` path still undemonstrated |
| `15_attribution` | **`tl.attribution`** — all 8 methods, `AttributionResult`, `AttributionError` surface, target specs, `tl.viz.render_heatmap`/`causal_trace_heatmap` tie-in | yes | `tl.attribution` reachable but invisible (no `__all__`/docs pointer); each call re-runs the model (no `trace=` reuse) |
| `16_compat_and_bridges` | **`tl.compat`** (`report`/`CompatReport`, `from_fx`/`from_ilg`/`from_torchextractor`/`from_timm`/`from_huggingface`, `lovely`/`torchshow`) + **`tl.bridge`** roster (16 adapters enumerated; captum + profiler end-to-end) | yes | `compat.report` emits stray protobuf stderr noise; broken third-party deps crash raw through bridge imports; adapter modules leak plumbing names |
| `visual/generate_visual_pack.py` | All `draw()` options x model zoo; stapled `visual_audit.pdf` | yes | page 18 profiling doubled-unit nit; otherwise no broken pages |

## Adversarial review status

Two rounds of independent coverage + fresh-eyes UX critique (2026-07-06), every
substantive finding verified against live code and folded back. Round 1 surfaced three
HIGH findings the June pass missed (the inert gradient-action family, the
`summary(level='compute'/'cost')` crash on multi-pass models, the pass-count-dependent
`trace[label]` return type). Round 2 verified the fixes, added the `torchlens.errors`
and `torchlens.backends` surfaces, and drove an assert-not-print pass over every
"this kwarg does X" demo — which exposed two more inert capture kwargs
(`activation_transform`, `grad_transform`) and an intervention-spec round-trip that
silently loses its intervention.

## Workflow slicing (locked design)

Each notebook is one **user workflow**, not one class or submodule. The two notebooks
added in this pass follow the same rule: attribution is the "explain a prediction"
workflow; compat/bridges is the "coming from another tool" workflow. Full triaged
findings live in the maintainer's review report.

# TorchLens Human-Facing Audit Notebooks

Coverage-optimized notebooks that exercise EVERY human-facing surface of TorchLens.
These are NOT beginner tutorials (see `notebooks/*.ipynb` for that) — they are a
developer ergonomics review: does every repr/summary/accessor look clean?

## How to review

1. Open `_exports/<notebook>.html` in a browser and scroll through executed output.
2. Flip through `visual/visual_audit.pdf` to catch pixel-level rendering nits.
3. Any cell marked **⚠️ GAP** is a surface that errored or felt awkward — those are
   the action items.

## Coverage matrix

| Notebook | Surfaces covered | Runs green? | GAPs flagged |
|---|---|---|---|
| `00_setup_and_first_capture` | `tl.trace`, `Trace.__repr__`/`__str__`, `summary()` levels, `layer_labels`/`op_labels`, `trace[label]`, `Op.out`, `tl.peek` | yes | `summary()` levels appear identical on simple models; `trace[int]` returns Layer not Op (may be confusing) |
| `01_indexing_and_lookup` | `trace[int/label/substr/module_path]`, `.layers/.ops/.modules/.params/.buffers`, `AmbiguousOpLookupError`, raw labels | pending | — |
| `02_activations_and_metadata` | `op.out`, `.shape`, `.dtype`, `.device_ref`, `.activation_memory`, `Bytes`/`Duration`/`Flops`/`Macs`, `saved_args`, `arg_expressions`, RNG fields, `code_context` | pending | — |
| `03_the_data_model` | `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`, `GradFn`, `GradFnCall` — `__repr__` + `to_pandas()` + key fields | pending | — |
| `04_extraction_surfaces` | `tl.extract`, `tl.batched_extract`, `Container`, `Container.summary()`, `to_pandas`, `output_table` | pending | — |
| `05_save_and_load` | `tl.save`/`tl.load`, `.tlspec` bundle, save levels, round-trip, `PayloadLoadHints` | pending | — |
| `06_backward_and_gradients` | backward trace, grad fields on `Op`, `GradFn`/`GradFnCall`, `tl.validate` backward parity, `draw_backward` smoke | pending | — |
| `07_intervention` | `tl.sites`, selectors (`func`/`in_module`/`where`/`followed_by`/`output`/`&`/`\|`/`~`), actions (`zero_ablate`/`add`/`replace_with`/`scale`/`steer`/`mean_ablate`), `when`/`do`, `replay`/`replay_from`/`rerun`, `splice_module`, `bwd_hook`, `grad_*` | pending | — |
| `08_bundles_and_cross_trace` | `tl.bundle`/`Bundle`, members, Super* accessors, `bundle.at`, `sweep`, pairwise diff, bundle save/load | pending | — |
| `09_fastlog_record` | `tl.record`/`fastlog`, `Recording` repr, `to_trace`, `save=` predicate, `dry_run`, `halt`, module events | pending | — |
| `10_facets` | `tl.facets` namespace, `facet`/`head` selectors, registry discovery, attention facets (guarded), `facet.grad` / reconstruction | pending | — |
| `11_visualization_in_workflow` | `draw` (`vis_mode` unrolled/rolled), `node_mode` presets, `module` focus, `show_containers`, `code_panel`, `node_overlay`, `vis_theme`, `show_legend`, `draw_backward`/`draw_combined` | pending | — |
| `12_validation_stats_reporting` | `tl.validate` (fwd/bwd/saved-out), `tl.report.explain`, `tl.tap`/`tl.record_span`, `summary` levels, `output_table` | pending | — |
| `13_tabular_export` | `trace.to_pandas`, per-record `to_pandas`, `torchlens.export` csv/json/parquet, schema/round-trip | pending | — |
| `14_huggingface_guarded` _(optional)_ | Guarded HF text/vision trace, bridge helpers, `tl.autoroute` (verify path) | pending | — |
| `visual/generate_visual_pack.py` | All `draw()` options x model zoo; each render-path scenario; stapled `visual_audit.pdf` | pending | — |

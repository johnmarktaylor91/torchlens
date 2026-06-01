# TorchLens 2.0 Audit Notebook Series

This folder contains the complete user-facing audit series for checking the major TorchLens 2.0 features against the current implementation. The notebooks are runnable end-to-end on CPU with tiny local PyTorch models, with guarded optional Hugging Face sections.

| Notebook | Runnable | Complete | Known gaps | Focus |
|---|---|---|---|---|
| 00_setup_and_first_capture | Yes | Partial | `module_filter` no-op on the tiny fixture; see API Status. | Install/import sanity, `tl.trace`, the `Trace` object, `summary()`, and a first successful capture. |
| 01_indexing_and_lookup | Yes | Yes | None called out. | Pythonic indexing, label lookup, substring/module-path lookup, and `layers`/`ops`/`modules` accessors. |
| 02_the_data_model | Yes | Partial | Nested container paths, duplicate output labels, recurrent distances, and saved argument templates are current-build gaps. | Anatomy of `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`, `GradFn`, and `GradFnCall` records. |
| 03_activations_and_metadata | Yes | Partial | Recurrent non-boundary container/distance/input-summary fields are current-build gaps. | Saved activations, shapes/dtypes/memory, current quantity fields, args, RNG, and code-context metadata. |
| 04_visualization | Yes | Yes | Top-level graph wrappers are deprecated compatibility shims. | Forward graph drawing, modes, node specs, module focus, backward/combined graphs, and code-panel rendering. |
| 05_save_and_load | Yes | Yes | None called out. | `tl.save`/`tl.load`, portable `.tlspec` directory bundles, save levels, and round-trip field survival. |
| 06_backward_and_gradients | Yes | Yes | None called out. | `backward_ready`, gradient saving, `GradFn`/`GradFnCall` records, and backward validation. |
| 07_intervention_api | Yes | Yes | `grad_fn_label` helper is not exported. | `find_sites`, selectors, forks, hooks, `do`, `set`, `replay`, and `rerun`. |
| 08_fastlog | Yes | Yes | Some source events are inspected through `recording_trace`, not retained records. | Predicate-selected sparse recording, `CaptureSpec`, `dry_run`, `Recorder`, `halt`, module events, and backward grad records. |
| 09_bundles_and_cross_trace | Yes | Yes | `clear()` retains the baseline member by design. | `tl.bundle`, shared labels, Super* accessors, pairwise comparison, and bundle save/load. |
| 10_facets | Yes | Yes | Attention q/k/v head facets are guarded when unavailable. | `tl.facets` views, custom recipes, registry discovery, and built-in attention q/k/v head facets when available. |
| 11_huggingface_and_autorouting | Yes | Yes | Optional HF fixtures may skip with their real runtime reason. | Guarded Hugging Face text/image/multimodal tracing, bridge helpers, and `tl.autoroute.input` detector registration. |
| 12_validation_stats_reporting | Yes | Partial | Forced forward-failure diagnostics are not surfaced for the state-mutating probe. | `tl.validate` forward/backward parity, streaming `tl.aggregate` stats, and readable report/summary output. |
| 13_tabular_export | Yes | Partial | Ops/GradFn/GradFnCall accessor export falls back to per-record tables; `memory_str` may round-trip as `NaN`. | `to_pandas`, `to_csv`, `to_parquet`, and `to_json` workflows for traces, records, and accessors. |

## API Status

These notebooks execute against the current checkout, and runnable code is the source of truth for this audit series. When glossary/target names differ from the current build, the notebooks use the executable API and call out the gap instead of silently mixing names.

| Topic | Glossary/target name | Current build used here |
|---|---|---|
| Quantity classes | `tl.Quantity`, `tl.Bytes`, `tl.Duration`, `tl.Flops`, `tl.Macs` | Not exported yet; examples show numeric fields plus `*_str` display fields. |
| Activation memory naming | `activation_memory` | Partially migrated; current records still expose `memory` and `memory_str`. |
| Site lookup | top-level `tl.find_sites(...)` | `Trace.find_sites(...)` with top-level selector helpers such as `tl.func(...)`. |
| Tabular export | canonical package exporters | `torchlens.export.csv/json/parquet(log, path)` are primary; instance exporters are migration-only. |
| Module filtering | `module_filter` visibly narrows saved modules/layers | Current audit fixture shows no retained-layer delta even with a false predicate; notebook 00 labels this as a gap. |
| Container output metadata | Distinct nested output labels, paths, and multi-output markers | Notebooks 02/03 show duplicate/collapsed labels and `container_path=None` for nested members as current behavior. |
| Distance fields | Recurrent layers expose `min_distance_from_input/output` | Rolled recurrent non-boundary layers currently report `None` in notebooks 02/03. |
| Fastlog module/source controls | Module/source records retained directly | Module records are retained via `keep_module`; source/event-stream coverage is inspected through `Recording.recording_trace`. |
| Validation diagnostics | Forced parity failures produce a failure/diagnostic | The state-mutating probe in notebook 12 returns success, so the notebook labels this as a diagnostic gap. |
| Export accessors | `torchlens.export.*` accepts ops, grad-fns, and grad-fn-call accessors directly | Notebook 13 labels missing accessor `to_pandas()` for those surfaces and uses per-record fallback tables. |

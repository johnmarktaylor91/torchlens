# TorchLens 2.0 Audit Notebook Series

This folder contains the complete user-facing audit series for checking the major TorchLens 2.0 features against the current implementation. The notebooks are runnable end-to-end on CPU with tiny local PyTorch models, with guarded optional Hugging Face sections.

| Notebook | Status | Focus |
|---|---|---|
| 00_setup_and_first_capture | Complete | Install/import sanity, `tl.trace`, the `Trace` object, `summary()`, and a first successful capture. |
| 01_indexing_and_lookup | Complete | Pythonic indexing, label lookup, substring/module-path lookup, and `layers`/`ops`/`modules` accessors. |
| 02_the_data_model | Complete | Anatomy of `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`, `GradFn`, and `GradFnCall` records. |
| 03_activations_and_metadata | Complete | Saved activations, shapes/dtypes/memory, current quantity fields, args, RNG, and code-context metadata. |
| 04_visualization | Complete | Forward graph drawing, modes, node specs, module focus, backward/combined graphs, and code-panel rendering. |
| 05_save_and_load | Complete | `tl.save`/`tl.load`, portable `.tlspec` directory bundles, save levels, and round-trip field survival. |
| 06_backward_and_gradients | Complete | `backward_ready`, gradient saving, `GradFn`/`GradFnCall` records, and backward validation. |
| 07_intervention_api | Complete | `find_sites`, selectors, forks, hooks, `do`, `set`, `replay`, and `rerun`. |
| 08_fastlog | Complete | Predicate-selected sparse recording, `CaptureSpec`, `dry_run`, `Recorder`, and `halt`. |
| 09_bundles_and_cross_trace | Complete | `tl.bundle`, shared labels, Super* accessors, pairwise comparison, and bundle save/load. |
| 10_facets | Complete | `tl.facets` views, custom recipes, registry discovery, and built-in attention q/k/v head facets when available. |
| 11_huggingface_and_autorouting | Complete | Guarded Hugging Face text tracing, `tl.bridge.hf.trace_text`, and `tl.autoroute.input` detector registration. |
| 12_validation_stats_reporting | Complete | `tl.validate` forward/backward parity, streaming `tl.aggregate` stats, and readable report/summary output. |
| 13_tabular_export | Complete | `to_pandas`, `to_csv`, `to_parquet`, and `to_json` workflows for traces, records, and accessors. |

## API Status

These notebooks execute against the current checkout, and runnable code is the source of truth for this audit series. When glossary/target names differ from the current build, the notebooks use the executable API and call out the gap instead of silently mixing names.

| Topic | Glossary/target name | Current build used here |
|---|---|---|
| Quantity classes | `tl.Quantity`, `tl.Bytes`, `tl.Duration`, `tl.Flops`, `tl.Macs` | Not exported yet; examples show numeric fields plus `*_str` display fields. |
| Activation memory naming | `activation_memory` | Partially migrated; current records still expose `memory` and `memory_str`. |
| Site lookup | top-level `tl.find_sites(...)` | `Trace.find_sites(...)` with top-level selector helpers such as `tl.func(...)`. |
| Tabular export | canonical package exporters | `torchlens.export.csv/json/parquet(log, path)` are primary; instance exporters are migration-only. |

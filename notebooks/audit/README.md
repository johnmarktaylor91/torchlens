# TorchLens 2.0 Audit Notebook Series

This folder contains a user-facing audit series for checking the major TorchLens 2.0 features against the current implementation. Batch 1 is foundational and runnable end-to-end on CPU with tiny local PyTorch models.

| Notebook | Status | Focus |
|---|---|---|
| 00_setup_and_first_capture | Batch 1 | Install/import sanity, `tl.trace`, the `Trace` object, `summary()`, and a first successful capture. |
| 01_indexing_and_lookup | Batch 1 | Pythonic indexing, label lookup, substring/module-path lookup, and `layers`/`ops`/`modules` accessors. |
| 02_the_data_model | Batch 1 | Anatomy of `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`, `GradFn`, and `GradFnCall` records. |
| 03_activations_and_metadata | Batch 1 | Saved activations, shapes/dtypes/memory, current quantity fields, args, RNG, and code-context metadata. |
| 04_visualization | Batch 1 | Forward graph drawing, modes, node specs, module focus, backward/combined graphs, and code-panel rendering. |
| 05_save_and_load | Coming in later batches | Trace persistence and reload workflows. |
| 06_backward_and_gradients | Coming in later batches | Backward capture and gradient inspection. |
| 07_intervention_api | Coming in later batches | Intervention selectors, hooks, replay, and rerun. |
| 08_fastlog | Coming in later batches | Fast logging, predicates, and storage modes. |
| 09_bundles_and_cross_trace | Coming in later batches | Bundles, alignment, and cross-trace comparisons. |
| 10_facets | Coming in later batches | Semantic facets and built-in recipes. |
| 11_huggingface_and_autorouting | Coming in later batches | HuggingFace bridges and input autorouting. |
| 12_validation_stats_reporting | Coming in later batches | Validation, stats, reports, and diagnostics. |
| 13_tabular_export | Coming in later batches | DataFrame/table export workflows. |

> NOTE: These notebooks use the current code as the source of truth. Where the glossary names differ from executable code, the notebooks call that out and use the names that run.

# TorchLens Total Audit Regeneration Prompt

You are regenerating TorchLens Total Audit.

TorchLens is the substrate for capturing, inspecting, and intervening on
PyTorch model operations. Total Audit is the maintainer's sandbox to bulletproof
every public name. The notebooks should feel progressive, exhaustive,
refreshable, and friendly to a maintainer who is tinkering with the API surface.
Nothing should be hidden behind external planning documents.

## Folder Layout

- `notebooks/total_audit/00_install_and_smoke.ipynb`
- `notebooks/total_audit/01_basic_capture.ipynb`
- `notebooks/total_audit/02_layer_indexing.ipynb`
- `notebooks/total_audit/03_save_load_basics.ipynb`
- `notebooks/total_audit/05_visualization_basics.ipynb`
- `notebooks/total_audit/06_modellog_anatomy.ipynb`
- `notebooks/total_audit/07_layerlog_anatomy.ipynb`
- `notebooks/total_audit/08_layerpasslog_anatomy.ipynb`
- `notebooks/total_audit/09_other_log_types.ipynb`
- `notebooks/total_audit/10_bundle_anatomy.ipynb`
- `notebooks/total_audit/15_visualization_options.ipynb`
- `notebooks/total_audit/16_visualization_advanced.ipynb`
- `notebooks/total_audit/17_intervention_helpers.ipynb`
- `notebooks/total_audit/18_intervention_verbs.ipynb`
- `notebooks/total_audit/19_bundles_advanced.ipynb`
- `notebooks/total_audit/20_save_load_advanced.ipynb`
- `notebooks/total_audit/21_validation.ipynb`
- `notebooks/total_audit/23_export_formats.ipynb`
- `notebooks/total_audit/24_bridges.ipynb`
- `notebooks/total_audit/25_compat_truth_table.ipynb`
- `notebooks/total_audit/26_perf_and_scaling.ipynb`
- `notebooks/total_audit/27_taps_and_observers.ipynb`
- `notebooks/total_audit/28_sites_and_sweeps.ipynb`
- `notebooks/total_audit/29_edge_cases.ipynb`
- `notebooks/total_audit/_shared.py`
- `notebooks/total_audit/_coverage_manifest.json`
- `notebooks/total_audit/_coverage_matrix.md`
- `notebooks/total_audit/_coverage_exceptions.txt` when a real exception is
  needed
- `notebooks/total_audit/README.md`
- `notebooks/total_audit/REGEN_PROMPT.md`

## Per-Notebook Outline

`00_install_and_smoke` confirms imports, `torchlens.__all__`, and the first tiny
capture. Other notebooks assume this import path and seed policy work.

`01_basic_capture` captures a deterministic forward pass, displays labels, and
uses the activation onramp. Indexing and anatomy notebooks build on this.

`02_layer_indexing` explores exact labels, substring behavior, suggestions, and
saved activation lookups. Intervention notebooks depend on these site rules.

`03_save_load_basics` writes and reloads a portable artifact in a temporary
directory. Advanced save/load notebooks reuse this cleanup pattern.

`05_visualization_basics` checks graph display entry points while keeping output
small enough for review.

`06_modellog_anatomy` inspects model-level fields, layer dictionaries, summary
state, and traversal affordances.

`07_layerlog_anatomy` inspects per-layer aggregate records and their connection
to pass-qualified labels.

`08_layerpasslog_anatomy` inspects per-operation records, saved tensors, shapes,
parents, and canonical field order.

`09_other_log_types` covers module, parameter, buffer, and gradient-log types
through small introspection examples.

`10_bundle_anatomy` builds a small bundle and inspects members, baseline state,
and lazy comparison helpers.

`15_visualization_options` covers visualization option objects and small render
argument combinations.

`16_visualization_advanced` uses branched and dynamic models for advanced graph
setup without large image outputs.

`17_intervention_helpers` surveys selector and edit helper constructors on
clean/corrupt toy tensors.

`18_intervention_verbs` smoke-tests `do`, `replay`, `replay_from`, and `rerun`
entry points and records invalid-site expected failures.

`19_bundles_advanced` compares related logs and checks relationship-gated bundle
helper discovery.

`20_save_load_advanced` covers save levels, intervention-spec paths, and
temporary-artifact cleanup.

`21_validation` runs consolidated validation and legacy validation wrappers in
small deterministic settings.

`23_export_formats` covers export namespace discovery and small JSON/trace
friendly data paths.

`24_bridges` audits bridge namespace discovery and optional-extra errors without
network downloads.

`25_compat_truth_table` documents compatibility report setup and the distinction
between strict notebook coverage and deprecation alias protection.

`26_perf_and_scaling` uses batched extraction and sparse logging knobs on tiny
tensors.

`27_taps_and_observers` demonstrates `tap` and `record_span` setup around a tiny
capture.

`28_sites_and_sweeps` builds simple site collections and sweep inputs.

`29_edge_cases` runs dynamic, recurrent, and expected-failure paths that protect
against common audit regressions.

## Cell-Type Catalog

Every notebook should include at least five of these cell types, and most should
include all of them:

- Setup: imports, seeds, path handling, temp directory.
- Prose intro: what the notebook covers and what depends on it.
- Demonstration: a normal successful TorchLens use.
- Parameter cell: marked by visible text `Try changing this:`.
- "Try changing this" prompt: prose inviting small edits.
- Alternate-model cell: a second model family or input shape.
- Expected-failure cell: a `try`/`except` block for documented failure behavior.
- Reset helper: clears temp state so reruns are cheap.
- Inline visualization: compact printed or displayed state.
- Coverage marker: code-cell metadata `coverage_calls: [...]`.
- Cleanup cell: deletes temporary artifacts.

## Coverage Requirements

Run:

```bash
python scripts/generate_audit_coverage_manifest.py
python scripts/check_audit_coverage.py --strict
```

Every public item in `_coverage_manifest.json` must reach `called` in at least
one notebook through code-cell metadata `coverage_calls`. Compatibility aliases
are ignored by strict notebook coverage because deprecation tests are their
single source of truth, but they are included in `check_coverage_delta.py` for
deletion protection.

Use `coverage_expected_failure` only for real expected-failure demonstrations.
Use `_coverage_exceptions.txt` only for names that are genuinely not
user-facing, and include a rationale comment next to each exception.

## Quality Criteria

Every notebook must execute end-to-end with papermill. Outputs are checked in,
but they must stay bounded: each cell output at most 50 KB, each image output at
most 200 KB, and each notebook at most 1 MB. Seeds must be explicit. Avoid
timestamps and environment-specific absolute paths in outputs.

## Regeneration Procedure

1. Read the current source by running
   `python scripts/generate_audit_coverage_manifest.py`.
2. Run `python scripts/check_audit_coverage.py --strict`.
3. For each missing item, add a small runnable cell or a coverage marker to the
   most appropriate notebook above.
4. Execute changed notebooks with papermill. For a full refresh, run every
   numbered notebook in order.
5. Re-run the manifest generator, strict coverage checker, and
   `python scripts/check_coverage_delta.py`.
6. Run the project quality gates before handing off.
7. Commit only when the orchestrator has asked for a commit.

## Substrate Identity Guardrails

Do not add top-level names casually. `len(torchlens.__all__)` is expected to be
exactly 40. Keep appliances in subfolders behind extras, such as
`torchlens.viewer`, `torchlens.paper`, `torchlens.notebook`, `torchlens.llm`,
and `torchlens.neuro`. Total Audit should reveal API holes; it should not hide
them by expanding the public substrate without an explicit API decision.

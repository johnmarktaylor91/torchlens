# Architecture map

This page describes the package as it is on the main branch today: the layers, the path one
capture takes from `tl.trace()` to a saved file, where each public feature is implemented
and tested, and the tangles a maintainer meets first. The generated companion
[Where does X live](where.md) lists every package and module with its first docstring
line; this page names only what you need before editing. Testing is in
[testing.md](testing.md), the review rules in [review_checklist.md](review_checklist.md),
setup and gates in [CONTRIBUTING.md](https://github.com/johnmarktaylor91/torchlens/blob/main/CONTRIBUTING.md) at the repository root.

Two notes on currency. A consolidation into about 12 internal packages behind an
`import-linter` graph gate is planned, not landed; nothing below describes that target. Two
moves are pending in the next round: `torchlens.fastlog` becomes `torchlens.recording`, and
the `capture`, `user_funcs`, `captured_run`, `semantic` and `visualization` packages become
private; the generated index absorbs both on its next run.

## Size

| Measure | Value |
| --- | ---: |
| Top-level packages (plus `schemas/`, a data directory) | 54 |
| Nested packages | 22 |
| Root modules (`torchlens/*.py`) | 84 |
| Modules in all | 1,113 |
| Lines of source | about 602,000 |

Numbers come from `python tools/docs/where_index.py`.

## Layers as they are today

Lower layers never import higher ones at module import time. Higher layers import lower
ones freely. Imports placed inside functions cross the layers in both directions; see
[Known tangles](#known-tangles).

| Layer | Holds | Packages and modules |
| --- | --- | --- |
| 0. Foundation | Global toggle state, constants, errors, literals, torch compatibility probes, stateless helpers | `_state.py`, `constants.py`, `errors/`, `_errors.py`, `_literals.py`, `_vocab/`, `types.py`, `quantities.py`, `options.py`, `utils/` (`utils/_torch_compat.py` is the only home for fragile torch-private probes) |
| 1. Records | The product types and the internal record shapes | `data_classes/` (`Trace`, `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`), `ir/` (`OpEvent`, `OpRecord`, selectors, containers), `_trace_core/` (typed column store), `accessors/` |
| 2. Capture | Wrapping torch, running the forward pass, logging ops | `backends/` (registry and the torch backend in `backends/torch/`; previews for JAX, TensorFlow, MLX, Paddle, tinygrad), `capture/` (backend-neutral runner, predicates, halt, outcome), `_capture_*.py`, `_chunking.py`, `_input_*.py`, `_robustness.py`, `fastlog/` (sparse predicate recording behind `tl.record`), `distributed/`, `partial/` |
| 3. Postprocess | Turning the raw log into a finished graph | `postprocess/` (steps 0 to 20 plus fractional inserts, order derived from declared read and write contracts) |
| 4. Persistence and replay | Saving, loading, replaying | `_io/` (bundle writer and reader, `.tlspec` manifest, loaders), `io/` (public administration helpers), `schemas/`, `runnable.py` and the 18 `_runnable_*.py` root modules (sparse replay behind `Trace.run`), `captured_run.py`, `bundle/` |
| 5. Analysis and verbs | Everything that reads or edits a finished `Trace` | `validation/` (the tripwire), `intervention/`, `selection*.py`, `attribution/`, `mechinterp/`, `debug/`, `receptive_field/`, `repgeom/`, `stats/`, `semantic/` (facets), `transforms/`, `checks/`, `observe/`, `observability/`, `trackers/`, `snoop/`, `experiment/`, `merged/`, `neuro/`, `_extraction/` with `dataset_extraction.py`, `preprocessing/`, `conformance/`, `differential.py` |
| 6. Presentation and bridges | Rendering, reports, exports, external tools | `visualization/`, `viz/`, `tviz/`, `report/`, `export/`, `quickstart/`, `agent/`, `notebook/`, `bridge/`, `callbacks/`, `compat/`, `ecosystem/`, `examples/`, `autoroute/`, `kernel_telemetry.py` |
| Entry | The public facade | `__init__.py` (lazy attribute table `_LAZY_ATTRS`, `__all__`), `user_funcs.py` (`trace`, `summary`, `render`, validation wrappers) |

Torch stays clean on `import torchlens`. The first capture installs the wrappers through
model preparation, and they stay installed until `torchlens.backends.torch.wrappers.unwrap_torch()`.

## Data flow: one capture

```
tl.trace(model, x)                       __init__.py -> user_funcs.trace
  resolve backend                        backends/registry.py (BackendSpec; eager torch default)
  prepare model, install wrappers        backends/torch/model_prep.py -> wrappers.wrap_torch()
  run forward under the logging toggle   capture/trace.py (runner), _state.py (toggle)
    each torch call hits its wrapper     backends/torch/wrappers.py -> backends/torch/ops.py, _ops_*.py
    barcode nesting keeps bottom ops     backends/torch/_ops_*.py
    op records land on the live Trace    data_classes/ (Op), _trace_core/ (columns), ir/ (events)
  postprocess the raw log                data_classes/_trace_validation.py::_postprocess -> postprocess/
    graph cleanup, conditional
    attribution, loop grouping,
    labeling, finalization               postprocess/graph_traversal.py, control_flow.py,
                                         loop_detection.py, labeling.py, finalization.py
  return Trace                           data_classes/trace.py
tl.save(trace, path)                     _io/bundle.py (metadata plus safetensors blobs)
tl.load(path)                            _io/bundle.py (allowlist unpickler, manifest validation)
trace.run(inputs=x)                      _runnable_*.py (sparse replay) or the live refresh path
tl.validate(model, x)                    validation/consolidated.py -> validation/core.py (forward replay),
                                         validation/backward.py, _invariants_*.py
```

The sparse path shares the front half: `tl.record(model, x, save=...)` runs the same wrappers
but emits `OpEvent` rows through `capture/projections.py` into a `Recording`
(`fastlog/`); `Recording.to_trace()` materialises a `Trace` through the same postprocess.

Key types crossing layer boundaries: `Trace` (one captured forward pass), `Op` (one executed
call, the graph's node), `Layer` (ops grouped across recurrent passes), `Module` and
`ModuleCall`, `Param` and `Buffer`, `Recording` and `OpEvent` (the sparse path),
`CaptureOutcome` (the settled status every product carries), `RunResult` (what `Trace.run`
returns).

## Where each public feature lives

Public entries are the names in `torchlens.__all__` (116 names) or documented submodule
names. Test columns name the files or directories under `tests/` that hold the behaviour
tests; `tests/AGENTS.md` has the full area table.

| Feature | Public entry | Implementing package | Tests |
| --- | --- | --- | --- |
| Eager capture | `tl.trace` | `user_funcs.py`, `capture/`, `backends/torch/` | `test_toy_models.py`, `test_capture_*.py`, `capture_oracle/`, `classics_corpus/` |
| Backend registry and previews | `tl.trace(backend=...)`, `tl.backends` | `backends/` and `backends/<name>/` | `backends/`, `backend_conformance/`, `backend_parity/` |
| Sparse recording | `tl.record`, `Recording` | `fastlog/`, `capture/projections.py`, `ir/events.py` | `test_fastlog/`, `test_record_*.py` |
| Capture options and predicates | `tl.options.CaptureOptions`, `tl.func`, `tl.when`, `tl.in_module` | `options.py`, `intervention/` (selectors), `ir/predicate_registry.py` | `test_selection_*.py`, `test_capture_*.py` |
| Capture outcomes and partial captures | `Trace.outcome`, `tl.partial` | `capture/outcome.py`, `partial/` | `test_capture_outcomes*.py`, `test_postprocess_*.py` |
| Graph postprocess | (internal, runs inside `tl.trace`) | `postprocess/` | `test_postprocess_*.py`, `test_loop_*.py`, `test_conditional_*.py` |
| Product types and lookup | `tl.Trace`, `tl.Op`, `tl.Layer`, `trace[label]` | `data_classes/`, `_trace_core/`, `accessors/` | `test_metadata*.py`, `test_layer_log.py`, `godobject_oracle/` |
| Save and load | `tl.save`, `tl.load`, `tl.io` | `_io/`, `io/`, `schemas/` | `test_io_*.py`, `test_tlspec_*.py`, `release_goldens/` |
| Replay | `Trace.run`, `torchlens.runnable` | `_runnable_*.py`, `runnable.py`, `captured_run.py` | `test_runnable_*.py`, `test_run_*.py` |
| Validation | `tl.validate`, `tl.validate_forward_pass` | `validation/` | `test_validation*.py`, `test_backward*.py`, `validation_goldens/` |
| Backward capture | `CaptureOptions(backward_ready=True)`, `trace.draw_backward` | `backends/torch/backward.py`, `validation/backward.py` | `test_backward*.py` |
| Intervention | `tl.run`, `tl.when`, `Trace.fork`, `fork.do`, `tl.sweep` | `intervention/`, `_trace_state.py` | `test_intervention*.py`, `test_sites.py`, `test_site_sweep_unit_candidates.py` |
| Bundles and experiments | `tl.Bundle`, `torchlens.experiment` | `bundle/`, `intervention/` (Bundle), `experiment/` | `test_bundle_*.py` |
| Selection algebra | `tl.Selection`, `tl.units`, `tl.params`, `tl.top_k` | `selection.py`, `selection_values.py`, `selection_graph.py`, `selection_subspace.py`, `selection_compare.py` | `test_selection*.py` |
| Attribution | `tl.attribution` | `attribution/`, `attribution/graph/`, `attribution/onebackward/` | `test_attribution*.py` |
| Mechanistic interpretability kit | `tl.mechinterp` | `mechinterp/` | `semantic/`, `test_mi_*.py` |
| Facets | `tl.semantic`, `tl.facets` | `semantic/`, `semantic/recipes/` | `semantic/`, `test_semantic*.py` |
| Receptive and projective fields | `tl.receptive_field`, `op.receptive_field` | `receptive_field/`, `receptive_field/rules/` | `test_receptive*.py` |
| Representation geometry and statistics | `tl.repgeom`, `tl.stats` | `repgeom/`, `stats/` | `test_repgeom*.py`, `test_stats*.py` |
| Activation transforms | `tl.transforms` | `transforms/` | `test_transforms*.py`, `transforms_corpus/` |
| Debugging helpers | `tl.debug` | `debug/` | `test_debug*.py` |
| Training checks, observers, trackers | `torchlens.checks`, `tl.observers`, `torchlens.trackers` | `checks/`, `observers.py`, `trackers/`, `observe/`, `observability/` | `test_checks*.py`, `test_observ*.py`, `test_tracker*.py`, `test_train_mode/` |
| Live narration | `torchlens.snoop` | `snoop/` | `test_snoop*.py` |
| Multi-rank merge | `tl.merge_ranks`, `tl.merged` | `merged/`, `distributed/` | `test_merge*.py`, `test_distributed*.py` |
| Dataset extraction | `tl.dataset_extraction` | `dataset_extraction.py`, `_extraction/`, `_data_substrate/` | `test_dataset_extraction*.py` |
| Neuroscience bridges | `tl.neuro`, `torchlens.preprocessing` | `neuro/`, `preprocessing/`, `bridge/` | `test_neuro*.py` |
| Summary and reports | `tl.summary`, `trace.summary()`, `tl.report` | `report/`, `user_funcs.py` | `test_summary*.py`, `test_report*.py` |
| Graph rendering | `trace.draw`, `tl.visualization.show_model_graph` | `visualization/`, `visualization/renderers/`, `visualization/lenses/` | `visualization/`, `test_draw*.py`, `test_collapse*.py`, `test_theme*.py` |
| Activation plots and transformer pictures | `tl.viz`, `tl.tviz` | `viz/`, `tviz/` | `test_visual*.py`, `test_tviz*.py` |
| Static exports | `tl.export` | `export/`, `export/_model_explorer/` | `test_export*.py`, `test_modelexplorer_assets/` |
| Quickstart ladder | `torchlens.user_funcs.render`, `tl.summary(model, input_size=...)`, `torchlens.quickstart.InputSpec` | `quickstart/` | `test_quickstart*.py` |
| Agent surface | `trace.to_agent_json`, `tl.report.explain`, `tl.bridge.mcp` | `agent/`, `report/`, `bridge/` | `test_agent*.py`, `agent_surface_goldens/` |
| External tool bridges | `tl.bridge.<tool>`, `tl.callbacks` | `bridge/`, `bridge/lit/`, `callbacks/` | `test_bridges_*.py` |
| Environment report | `tl.utils.doctor()`, `tl.compat.report()` | `utils/__init__.py`, `compat/` | `test_doctor*.py`, `test_compat*.py` |
| Episode capture | `tl.options.EpisodeSpec` | `_episode_spec.py`, `capture/`, `bundle/` | `test_episode*.py` |
| Kernel telemetry | `torchlens.kernel_telemetry` | `kernel_telemetry.py` | `test_kernel_telemetry*.py` |

## Known tangles

**Deferred imports.** 544 of the modules sit in one import cycle that is held open by
about 1,300 imports placed inside function bodies. The root facade (`__init__.py`) and
`_io/__init__.py` are declared facades and defer on purpose; most of the rest exist because
a lower layer calls upward: `Trace._postprocess` imports `postprocess` inside the method,
`user_funcs.py` carries about 40 in-function imports, `data_classes/trace.py` about 45,
`backends/torch/wrappers.py` and `capture/trace.py` about 25 each. A module-level layer
diagram therefore understates the coupling; read the in-function imports of any module
before you move it. `_split_rebind.py` rebinds four split modules' functions back onto the
old module's globals, and `postprocess/_executor.py` reads its own package through
`sys.modules`, both so tests can keep patching private names.

**The god objects.** `Trace` is `data_classes/trace.py` (about 4,000 lines) plus 15
`data_classes/_trace_*.py` mixin modules; its behaviour for validation, visualization,
replay and selection is spread across them and across `_runnable_seam.py`. The torch
capture entry is `_trace_torch_model` in `user_funcs.py`, one function of about 1,300
lines outside the backend package; `torch_func_decorator` and its nested `wrapped_func` in
`backends/torch/wrappers.py` are about 900 and 760 lines; the forward runner
`run_and_log_inputs_through_model` in `capture/trace.py` about 860. `validation/core.py` is
5,200 lines and `utils/rng.py` 5,080. Changes to capture semantics usually touch three of
these at once.

**Five record shapes for one op.** An executed call is represented as `ir.events.OpEvent`
(the sparse event, 54 fields), `ir.op_record.OpRecord` (core plus facets with flat-name
properties), a row in the `_trace_core` column store, `data_classes.op.Op` (the public
node) and, grouped across passes, `data_classes.layer.Layer`. Six label vocabularies name
the same thing (`layer_label`, `op_label`, `raw_label`, `tensor_label`, `barcode`,
`node_label`), and `ConditionalEvent` is defined twice with different bodies
(`data_classes/trace.py`, `ir/events.py`). A field added to an op must be threaded through
every shape it should survive, and through `constants.py` field orders and the
`.tlspec` writer contract if it persists.

**Overlapping homes.** `io/` is the public administration namespace and `_io/` the
implementation. `viz/` plots activations, `visualization/` renders graphs, `tviz/` draws
transformer pictures. `checks/`, `validation/`, `debug/` and four root `_validation_*.py`
modules all verify something. `errors/` is the public exception package, `_errors.py` the
internal helpers, and `intervention/errors.py` the intervention catalogue. `facets.py` at
the root is an alias that replaces itself in `sys.modules` with `semantic/facets.py`. The
runnable replay is 18 `_runnable_*.py` root modules rather than a package. Module names in
`backends/torch/` (`belt.py`, `rescue.py`, `_runnable_seam.py`) do not say what they hold;
the generated index gives each one's first docstring line.

## Where to read next

- A module's one-line purpose and size: [where.md](where.md).
- Step order and contracts in postprocess: `torchlens/postprocess/_contracts.py`.
- The invariants a change must keep: `docs/agent-reference/critical-invariants.md`.
- The saved-file contract: `torchlens/schemas/writer_contract_v10.json` and
  `docs/reference/runnable_tlspec_contract.md`.
- The public surface, name by name: `docs/reference/glossary.md`.

# Architecture map

This page describes the package as it is on the main branch today: the layers, the path one
capture takes from `tl.trace()` to a saved file, where each public feature is implemented
and tested, and the tangles a maintainer meets first. The generated companion
[Where does X live](where.md) lists every package and module with its first docstring
line; this page names only what you need before editing. Testing is in
[testing.md](testing.md), the review rules in [review_checklist.md](review_checklist.md),
setup and gates in [CONTRIBUTING.md](https://github.com/johnmarktaylor91/torchlens/blob/main/CONTRIBUTING.md) at the repository root.

For the narrative version, read [How TorchLens works](../how-it-works.md) first: it follows one
capture from wrapping torch to a finished `Trace` and through lookups, gradients, edits, drawing,
saving and backends. This page is its maintainer companion and adds the boundaries, sizes and test
locations you need before changing the code.

Two notes on currency. A consolidation into about 12 internal packages is planned, not
landed; nothing below describes that target. Today's layer separations are already enforced
by the import-linter contracts in `.importlinter` (`python scripts/checks/run_all.py`). The
second-round moves have landed: the former fastlog package is `torchlens.recording`, and the
capture, user_funcs, captured_run, semantic and visualization packages are private
(`_capture/`, `_user_funcs.py`, `_captured_run.py`, `_semantic/`, `_visualization/`); the
generated index reflects them.

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
| 0. Foundation | Global toggle state, constants, errors, literals, torch compatibility probes, stateless helpers | `_state.py`, `_constants.py`, `errors/`, `_errors.py`, `_literals.py`, `_vocab/`, `types.py`, `_quantities.py`, `options.py`, `_utils/` (`_utils/_torch_compat.py` is the only home for fragile torch-private probes) |
| 1. Records | The product types and the internal record shapes | `_data_classes/` (`Trace`, `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`), `_ir/` (`OpEvent`, `OpRecord`, selectors, containers), `_trace_core/` (typed column store), `appliances/` |
| 2. Capture | Wrapping torch, running the forward pass, logging ops | `backends/` (registry and the torch backend in `backends/torch/`; backends for JAX, TensorFlow, MLX, Paddle, tinygrad), `_capture/` (backend-neutral runner, predicates, halt, outcome), `_capture_*.py`, `_chunking.py`, `_input_*.py`, `_robustness.py`, `recording/` (sparse predicate recording behind `tl.record`), `distributed/`, `_partial/` |
| 3. Postprocess | Turning the raw log into a finished graph | `_postprocess/` (steps 0 to 20 plus fractional inserts, order derived from declared read and write contracts) |
| 4. Persistence and replay | Saving, loading, replaying | `_io/` (bundle writer and reader, `.tlspec` manifest, loaders), `io/` (public administration helpers), `schemas/`, `_runnable/` (the frozen runnable vocabulary in its `__init__.py` and the sparse replay modules behind `Trace.run`), `captured_run.py`, `bundle/` |
| 5. Analysis and verbs | Everything that reads or edits a finished `Trace` | `validation/` (the tripwire), `intervention/`, `_selection*.py`, `attribution/`, `mechinterp/`, `debug/` (training checks in `debug/_checks/`), `_snoop/` (live narration), `receptive_field/`, `repgeom/`, `stats/`, `semantic/` (facets), `transforms/`, `_observe/`, `_observability/`, `trackers/`, `experiment/`, `_merged/`, `neuro/`, `_extraction/` with `dataset_extraction.py`, `preprocessing/`, `differential.py` |
| 6. Presentation and bridges | Rendering, reports, exports, external tools | `visualization/`, `viz/`, `transformer/`, `report/` (notebook cards in `report/_notebook/`), `export/`, `_quickstart/`, `agent/`, `bridge/`, `callbacks/`, `compat/`, `_ecosystem/`, `examples/`, `_autoroute/`, `_kernel_telemetry.py` |
| Entry | The public facade | `__init__.py` (lazy attribute table `_LAZY_ATTRS`, `__all__`), `user_funcs.py` (`trace`, `summary`, `render`, validation wrappers) |

Torch stays clean on `import torchlens`. The first capture installs the wrappers through
model preparation, and they stay installed until `torchlens.backends.torch.wrappers.unwrap_torch()`.

## Data flow: one capture

```
tl.trace(model, x)                       __init__.py -> user_funcs.trace
  resolve backend                        backends/registry.py (BackendSpec; eager torch default)
  prepare model, install wrappers        backends/torch/model_prep.py -> wrappers.wrap_torch()
  run forward under the logging toggle   _capture/trace.py (runner), _state.py (toggle)
    each torch call hits its wrapper     backends/torch/wrappers.py -> backends/torch/ops.py, _ops_*.py
    barcode nesting keeps bottom ops     backends/torch/_ops_*.py
    op records land on the live Trace    _data_classes/ (Op), _trace_core/ (columns), _ir/ (events)
  postprocess the raw log                _data_classes/trace.py::Trace._postprocess -> _postprocess/
    graph cleanup, conditional
    attribution, loop grouping,
    labeling, finalization               _postprocess/graph_traversal.py, control_flow.py,
                                         loop_detection.py, labeling.py, finalization.py
  return Trace                           _data_classes/trace.py
trace.save(path)                         _io/bundle.py (metadata plus safetensors blobs)
tl.load(path)                            _io/bundle.py (allowlist unpickler, manifest validation)
trace.run_transaction(inputs=x)                      _runnable_*.py (sparse replay) or the live refresh path
tl.validate(model, x)                    validation/_consolidated.py -> validation/_core.py (forward replay),
                                         validation/_backward.py, _invariants_*.py
```

The sparse path shares the front half: `tl.record(model, x, save=...)` runs the same wrappers
but emits `OpEvent` rows through `_capture/projections.py` into a `Recording`
(`recording/`); `Recording.to_trace()` materialises a `Trace` through the same postprocess.

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
| Eager capture | `tl.trace` | `_user_funcs.py`, `_capture/`, `backends/torch/` | `test_toy_models.py`, `test_capture_*.py`, `capture_oracle/`, `classics_corpus/` |
| Backend registry and the other backends | `tl.trace(backend=...)`, `tl.backends` | `backends/` and `backends/<name>/` | `backends/`, `backend_conformance/`, `backend_parity/` |
| Sparse recording | `tl.record`, `Recording` | `recording/`, `_capture/projections.py`, `_ir/events.py` | `test_fastlog/`, `test_record_*.py` |
| Capture options and predicates | `tl.options.CaptureOptions`, `tl.func`, `tl.when`, `tl.in_module` | `options.py`, `intervention/` (selectors) | `test_selection_*.py`, `test_capture_*.py` |
| Capture outcomes and partial captures | `Trace.outcome`, `tl.partial` | `_capture/outcome.py`, `_partial/` | `test_capture_outcomes*.py`, `test_postprocess_*.py` |
| Graph postprocess | (internal, runs inside `tl.trace`) | `_postprocess/` | `test_postprocess_*.py`, `test_loop_*.py`, `test_conditional_*.py` |
| Product types and lookup | `tl.Trace`, `tl.Op`, `tl.Layer`, `trace[label]` | `_data_classes/`, `_trace_core/`, `appliances/` | `test_metadata*.py`, `test_layer_log.py`, `godobject_oracle/` |
| Save and load | `Trace.save`, `tl.load`, `tl.io` | `_io/`, `io/`, `schemas/` | `test_io_*.py`, `test_tlspec_*.py`, `release_goldens/` |
| Replay | `Trace.run`, `torchlens._runnable` | `_runnable/`, `captured_run.py` | `test_runnable_*.py`, `test_run_*.py` |
| Validation | `tl.validate`, `torchlens.validation.validate_forward_pass` | `validation/` | `test_validation*.py`, `test_backward*.py`, `validation_goldens/` |
| Backward capture | `CaptureOptions(backward_ready=True)`, `trace.draw_backward` | `backends/torch/backward.py`, `validation/_backward.py` | `test_backward*.py` |
| Intervention | `tl.run`, `tl.when`, `Trace.fork`, `fork.do`, `tl.sweep` | `intervention/`, `_trace_state.py` | `test_intervention*.py` |
| Bundles and experiments | `tl.Bundle`, `torchlens.experiment` | `bundle/`, `intervention/` (Bundle), `experiment/` | `test_bundle_*.py` |
| Selection algebra | `tl.Selection`, `tl.units`, `tl.params`, `tl.top_k` | `_selection.py`, `_selection_values.py`, `_selection_graph.py`, `_selection_subspace.py`, `_selection_compare.py` | `test_selection*.py` |
| Attribution | `tl.attribution` | `attribution/`, `attribution/_graph/`, `attribution/_onebackward/` | `test_attribution*.py` |
| Mechanistic interpretability kit | `tl.mechinterp` | `mechinterp/` | `semantic/`, `test_mi_*.py` |
| Facets | `tl.facets`, `torchlens.mechinterp` | `_semantic/`, `_semantic/recipes/` | `semantic/`, `test_semantic*.py` |
| Receptive and projective fields | `tl.receptive_field`, `op.receptive_field` | `receptive_field/`, `receptive_field/_builtin_rules/` | `test_receptive*.py` |
| Representation geometry and statistics | `tl.repgeom`, `tl.stats` | `repgeom/`, `stats/` | `test_repgeom*.py`, `test_stats*.py` |
| Activation transforms | `tl.transforms` | `transforms/` | `test_transforms*.py`, `transforms_corpus/` |
| Debugging helpers | `tl.debug` | `debug/` | `test_debug*.py` |
| Training checks, observers, trackers | `tl.debug.ChecksSession`, `tl.observers`, `torchlens.trackers` | `debug/_checks/`, `observers.py`, `trackers/`, `_observe/`, `_observability/` | `test_checks*.py`, `test_observ*.py`, `test_tracker*.py`, `test_train_mode/` |
| Live narration | `torchlens._snoop` | `_snoop/` | `test_snoop*.py` |
| Multi-rank merge | `tl.merge_ranks`, `tl.merged` | `_merged/`, `distributed/` | `test_merge*.py`, `test_distributed*.py` |
| Dataset extraction | `tl.dataset_extraction` | `dataset_extraction.py`, `_extraction/`, `_data_substrate/` | `test_dataset_extraction*.py` |
| Neuroscience bridges | `tl.neuro`, `torchlens.preprocessing` | `neuro/`, `preprocessing/`, `bridge/` | `test_neuro*.py` |
| Summary and reports | `tl.summary`, `trace.summary()`, `tl.report` | `report/`, `user_funcs.py` | `test_summary*.py`, `test_report*.py` |
| Graph rendering | `trace.draw`, `torchlens._visualization.show_model_graph` | `_visualization/`, `_visualization/renderers/`, `_visualization/lenses/` | `visualization/`, `test_draw*.py`, `test_collapse*.py`, `test_theme*.py` |
| Activation plots and transformer pictures | `tl.viz`, `tl.tviz` | `viz/`, `transformer/` | `test_visual*.py`, `test_tviz*.py` |
| Static exports | `tl.export` | `export/`, `export/_model_explorer/` | `test_export*.py`, `test_modelexplorer_assets/` |
| Quickstart ladder | `torchlens._user_funcs.draw`, `tl.summary(model, input_shape=...)`, `torchlens._quickstart.InputSpec` | `_quickstart/` | `test_quickstart*.py` |
| Agent surface | `trace.to_agent_json`, `tl.report.explain`, `tl.bridge.mcp` | `agent/`, `report/`, `bridge/` | `test_agent*.py`, `agent_surface_goldens/` |
| External tool bridges | `tl.bridge.<tool>`, `tl.callbacks` | `bridge/`, `bridge/lit/`, `callbacks/` | `test_bridges_*.py` |
| Environment and model report | `tl.report.doctor()` | `report/_doctor.py`, `report/_environment.py`, `compat/_report.py` | `test_doctor*.py`, `test_compat*.py` |
| Episode capture | `tl.options.EpisodeOptions` | `_episode_spec.py`, `_capture/`, `bundle/` | `test_episode*.py` |
| Kernel telemetry | `torchlens._kernel_telemetry` | `_kernel_telemetry.py` | `test_kernel_telemetry*.py` |

## Known tangles

**Deferred imports.** 544 of the modules sit in one import cycle that is held open by
about 1,300 imports placed inside function bodies. The root facade (`__init__.py`) and
`_io/__init__.py` are declared facades and defer on purpose; most of the rest exist because
a lower layer calls upward: `Trace._postprocess` imports `postprocess` inside the method,
`user_funcs.py` carries about 40 in-function imports, `_data_classes/trace.py` about 45,
`backends/torch/wrappers.py` and `_capture/trace.py` about 25 each. A module-level layer
diagram therefore understates the coupling; read the in-function imports of any module
before you move it. The four split hubs (`backends/torch/ops.py`,
`backends/torch/completeness_witness.py`, `validation/_invariants.py`, `_runnable/execution.py`)
import their split modules at the bottom of the file, after their own definitions; a test
patches a private name on the module that defines it and on every module that binds it
(`docs/internal/agent-reference/known-gotchas.md`).

**The god objects.** `Trace` is `_data_classes/trace.py` (about 3,700 lines of thin methods and
the dataclass) over the `_data_classes/_trace_*.py` function modules and the destination
packages behind `_doors.py`; its behaviour for validation, visualization, replay and
selection lives in those destinations and in `_runnable/seam.py`. The torch
capture entry is `_trace_torch_model` in `user_funcs.py`, one function of about 1,300
lines outside the backend package; `torch_func_decorator` and its nested `wrapped_func` in
`backends/torch/wrappers.py` are about 900 and 760 lines; the forward runner
`run_and_log_inputs_through_model` in `_capture/trace.py` about 860. `validation/_core.py` is
5,200 lines and the host-nondeterminism monitor `_utils/rng/monitor.py` about 1,600. Changes to capture semantics usually touch three of
these at once.

**Five record shapes for one op.** An executed call is represented as `ir.events.OpEvent`
(the sparse event, 54 fields), `ir.op_record.OpRecord` (core plus facets with flat-name
properties), a row in the `_trace_core` column store, `data_classes.op.Op` (the public
node) and, grouped across passes, `data_classes.layer.Layer`. Six label vocabularies name
the same thing (`layer_label`, `op_label`, `raw_label`, `tensor_label`, `barcode`,
`node_label`), and `ConditionalEvent` is defined twice with different bodies
(`_data_classes/trace.py`, `_ir/events.py`). A field added to an op must be threaded through
every shape it should survive, and through `_constants.py` field orders and the
`.tlspec` writer contract if it persists.

**Overlapping homes.** `io/` is the public administration namespace and `_io/` the
implementation. `viz/` plots activations, `visualization/` renders graphs, `transformer/` draws
transformer pictures. `validation/`, `debug/` and four root `_validation_*.py`
modules all verify something. `errors/` is the public exception package, `_errors.py` the
internal helpers, and `intervention/errors.py` the intervention catalogue. `facets.py` at
the root is an alias that replaces itself in `sys.modules` with `_semantic/facets.py`.
Module names in `backends/torch/` (`belt.py`, `rescue.py`) do not say what they hold;
the generated index gives each one's first docstring line.

## Where to read next

- A module's one-line purpose and size: [where.md](where.md).
- Step order and contracts in postprocess: `torchlens/_postprocess/_pipeline.py`.
- The invariants a change must keep: `docs/internal/agent-reference/critical-invariants.md`.
- The saved-file contract: `torchlens/schemas/writer_contract_v12.json` and
  `docs/reference/runnable_tlspec_contract.md`.
- Every persisted key of the saved format, with its type, save level, policy and public
  accessor: [persisted_fields.md](persisted_fields.md) (generated).
- The public surface, name by name: `docs/reference/glossary.md`.

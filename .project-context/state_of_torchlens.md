# State of TorchLens 2.x

## What TorchLens Is
TorchLens records backend-resolved execution as a graph of tensor operations, modules,
parameters, buffers, activations, gradients, and source/context metadata. The stable default is
PyTorch eager capture, which lazily wraps PyTorch functions on first capture with persistent
wrappers gated by `_state._logging_enabled`. The main output is a `Trace`, whose layers are
represented as aggregate `Layer` objects over per-pass `Op` records. TorchLens 2.x also includes
portable `.tlspec` save/load, intervention bundles, torch-only sparse fastlog recording,
visualization helpers, optional bridges, and stub appliance namespaces.

## Public API Surface

`torchlens.__all__` contains the public capture, save/load, intervention, selector, observer,
validation, and core data-class names for the current 2.x surface.

### Capture and Extraction
- `trace` - backend-resolved capture returning `Trace`; `torchlens.user_funcs.trace`.
- `fastlog` - torch-only sparse predicate-recording namespace; `torchlens.fastlog`.
- `peek` - capture one layer activation by label/query; `torchlens.__init__.peek`.
- `extract` - extract one or more layer activations; `torchlens.__init__.extract`.
- `batched_extract` - batched extraction helper over an iterable; `torchlens.__init__.batched_extract`.

### Save and Load
- `save` - save `Trace`, `Bundle`, or related object to portable format; `torchlens._io.bundle.save`.
- `load` - load portable TorchLens artifact; `torchlens._io.bundle.load`.

### Validation
- `validate` - consolidated validator with `scope=`; `torchlens.validation.consolidated.validate`.

### Data Classes
- `Trace` - top-level capture container; `torchlens.data_classes.trace.Trace`.
- `Layer` - aggregate layer across one or more Op passes; `torchlens.data_classes.layer.Layer`.
- `Op` - one per-pass tensor operation record; `torchlens.data_classes.op.Op`.

### Intervention Containers and Execution
- `Bundle` - named collection of logs/specs with multi-trace helpers; `torchlens.intervention.bundle.Bundle`.
- `bundle` - convenience constructor for `Bundle`; `torchlens.__init__.bundle`.
- `do` - runtime intervention execution helper; `torchlens.intervention.runtime.do`.
- `replay` - replay an intervention/log/spec; `torchlens.intervention.replay.replay`.
- `replay_from` - replay from a source object; `torchlens.intervention.replay.replay_from`.
- `rerun` - rerun a model with interventions; `torchlens.intervention.rerun.rerun`.

### Selectors and Site Discovery
- `label` - selector by layer label; `torchlens.intervention.selectors.label`.
- `func` - selector by function name; `torchlens.intervention.selectors.func`.
- `module` - selector by module address/type; `torchlens.intervention.selectors.module`.
- `contains` - selector combinator for contained text/metadata; `torchlens.intervention.selectors.contains`.
- `where` - predicate selector; `torchlens.intervention.selectors.where`.
- `in_module` - selector constrained to module scope; `torchlens.intervention.selectors.in_module`.
- `sites` - build/inspect site collections; `torchlens.intervention.sites.sites`.

### Intervention Helper Transforms
- `clamp` - clamp activation values; `torchlens.intervention.helpers.clamp`.
- `mean_ablate` - mean-ablation helper; `torchlens.intervention.helpers.mean_ablate`.
- `noise` - noise injection helper; `torchlens.intervention.helpers.noise`.
- `project_off` - remove a projection direction; `torchlens.intervention.helpers.project_off`.
- `project_onto` - project onto a direction; `torchlens.intervention.helpers.project_onto`.
- `resample_ablate` - resample-ablation helper; `torchlens.intervention.helpers.resample_ablate`.
- `scale` - scale activation helper; `torchlens.intervention.helpers.scale`.
- `splice_module` - module-splicing helper; `torchlens.intervention.helpers.splice_module`.
- `steer` - steering-vector helper; `torchlens.intervention.helpers.steer`.
- `swap_with` - activation swap helper; `torchlens.intervention.helpers.swap_with`.
- `zero_ablate` - zero-ablation helper; `torchlens.intervention.helpers.zero_ablate`.
- `bwd_hook` - backward hook helper; `torchlens.intervention.helpers.bwd_hook`.
- `gradient_scale` - gradient scaling helper; `torchlens.intervention.helpers.gradient_scale`.
- `gradient_zero` - gradient-zeroing helper; `torchlens.intervention.helpers.gradient_zero`.

### Observers
- `tap` - create a tap observer for a site; `torchlens.observers.tap`.
- `record_span` - context manager for user span records during capture; `torchlens.observers.record_span`.

## Subpackage Map

| Path | What lives there | Why look there |
|------|------------------|----------------|
| `torchlens/__init__.py` | Top-level exports, shims, `peek`, `extract`, `batched_extract` | Public API shape and compatibility moves |
| `torchlens/_state.py` | Logging toggle, active log, wrapper maps, prepared model registry | Global state and decoration invariants |
| `torchlens/constants.py` | FIELD_ORDER tuples, torch function discovery | Field additions and wrapper coverage |
| `torchlens/user_funcs.py` | Capture, summary, graph display, validation entry points | Main user workflows |
| `torchlens/options.py` | Capture/save/vis/replay/intervention/streaming options | Flat kwarg to grouped option behavior |
| `torchlens/observers.py` | `tap`, `record_span`, active span storage | User instrumentation during capture |
| `torchlens/_io/` | Portable bundle internals, manifests, lazy refs, streaming writer | `.tlspec` and save/load implementation |
| `torchlens/io/` | Public I/O/admin helpers | `inspect_tlspec`, `detect_tlspec_format`, moved admin APIs |
| `torchlens/capture/` | Forward/backward runtime logging | Wrapper handoff and raw `Op` construction |
| `torchlens/decoration/` | Lazy wrapping and model preparation | Torch namespace lifecycle, module wrappers |
| `torchlens/postprocess/` | 20-step graph cleanup/finalization | Labels, loops, conditionals, modules, streaming finalization |
| `torchlens/data_classes/` | `Trace`, `Layer`, `Op`, module/param/buffer/grad records | User-visible capture data structures |
| `torchlens/validation/` | Forward/backward replay, invariants, `.tlspec` schema | Correctness checks |
| `torchlens/visualization/` | Graphviz, ELK, NodeSpec, overlays, bundle diff, fastlog preview | Rendering and visual customization |
| `torchlens/intervention/` | Bundle, sites, selectors, hooks, helpers, replay/rerun/save | Intervention API |
| `torchlens/multi_trace/` | Bundle supergraph, topology diff, node views, metrics | Cross-log comparisons |
| `torchlens/fastlog/` | Sparse predicate recording and storage | Low-overhead selected activation capture |
| `torchlens/bridge/` | Optional external-tool adapters | Captum, HF, SHAP, SAE Lens, profiler, LIT, etc. |
| `torchlens/compat/` | Migration helpers and compatibility report | Interop with torchextractor, FX, HF/timm, torchshow/lovely |
| `torchlens/export/` | Static SVG/HTML/tracing exports | Artifact export without full viewer extras |
| `torchlens/report/` | `explain(log)` and capture-time scalar logging | Human-readable log summaries |
| `torchlens/stats/` | Streaming stats and `aggregate()` | Dataset-scale activation summaries |
| `torchlens/callbacks/` | Lightning callback integration | Training-loop adapters |
| `torchlens/partial/` | Partial capture wrapper for failed forwards | Debugging failed captures |
| `torchlens/experimental/` | Unstable helpers, Dagua opt-in, node styles | Experimental APIs |
| `torchlens/viz/` | Convenience visual namespace | Bundle diff and heatmap helpers |
| `torchlens/accessors/` | Moved accessor aliases | Non-top-level public accessor imports |
| `torchlens/types.py` | Moved type aliases | Public types not in top-level `__all__` |

## Key Concepts

### Toggle Architecture
Torch wrapping is lazy. `decoration/model_prep.py:_ensure_model_prepared()` calls
`wrap_torch()` before a logging session. Once installed, wrappers stay in place and check
`_state._logging_enabled`; logging-off calls take the original PyTorch path. `active_logging()`
enables capture for one forward pass, and `pause_logging()` protects TorchLens internal tensor
ops from recursive logging.

### Trace, Layer, Op
`Op` is the per-operation/per-pass record built during capture. `Layer` is the postprocessed
aggregate for one final layer label and may contain multiple passes. `Trace` is the top-level
graph and owns lookup, summaries, rendering, save/load, validation, and
intervention convenience methods.

### Intervention Model
Intervention-ready captures retain enough metadata for site selection and replay. Selectors
resolve through `Trace.find_sites()` / `resolve_sites()` and `intervention/resolver.py`.
Hooks are normalized in `intervention/hooks.py`, executed through capture/live-hook paths, and
validated in `intervention/runtime.py`. `Bundle` collects named logs/specs and uses
`multi_trace/` for node views, topology comparisons, and diff metrics. Save levels live in
`intervention/save.py`.

### .tlspec Format
Portable artifacts are directory bundles with a public `manifest.json` schema and tensor blobs
stored as safetensors. `_io/tlspec.py` writes unified model-log, intervention, and bundle
manifests. `io.detect_tlspec_format()` distinguishes unified and legacy formats.
`validation.validate_tlspec()` validates unified manifests only; older 2.16 formats remain
loadable. Unified manifests currently support manifest schema v1 for torch writes and schema v2
for backend-aware validation/loading preflight. Schema v2 adds `backend`, `backend_runtime`,
nullable torch-specific fields, `payload_policy`, and audit-only non-torch payload refusal before
torch-shaped manifest parsing.

### Backend-Neutral Substrate
`Trace.backend` is the public backend tag resolved by `BackendSpec`. Trace identity also records
`module_identity_mode` (`torch_module`, future `pytree_module`, or `function_root`) and
`param_source` (`native-module`, future `pytree-derived`, or `none`). Op, Layer, and Param records
carry neutral `dtype_ref`, `device_ref`, `backend_address`, and `resolver_status` mirror fields
beside existing torch-shaped fields so torch accessors stay stable while non-torch builders can
populate backend-native metadata. This is a pickled object-state change, so `_io.TLSPEC_VERSION`
is 5; manifest schema v2 support is manifest-only and does not bump the pickled object-state
version again.

JAX M1 exposes leaf-level derived gradients through `trace.derived_grads` when capture receives
`tl.backends.jax.GradOptions`. These records come from a second pure `jax.value_and_grad` run and
are not backward capture; JAX traces keep `backward_capture=False`, and `trace.log_backward(...)`,
`trace.backward_passes`, `trace.saved_grad_ops`, and `op.grads` raise with guidance to
`trace.derived_grads`.

tinygrad M2 exposes a preview UOp-snapshot backend through explicit `backend="tinygrad"` on raw
callables with positional tinygrad `Tensor` inputs. It is pinned to the `tinygrad>=0.13,<0.14`
series and the runtime guard currently accepts `tinygrad==0.13.0`; live replay validation and
derived gradients require `DEV=PYTHON` realized-copy payloads. The module mode is
`function_root`, `Trace.param_source` is `"none"`, `.tlspec` save/load is audit-only, and
`tl.backends.tinygrad.GradOptions` populates leaf-level `trace.derived_grads` without enabling
true backward capture. Selective save predicates, module predicates, intervention, halt,
streaming, `save_grads=`, `backward_ready=True`, `tl.record(backend="tinygrad")`, mid-capture
realize/assign/replace/setitem mutation, and TinyJit execution raise typed backend errors with
workarounds.

### Fastlog vs Full Capture
Full capture returns a faithful `Trace` and runs postprocess. Fastlog returns a sparse
`Recording` of predicate-selected torch operation/module events. It can store to RAM, disk, or
both, but does not promise full graph invariants.

### Two-Pass Strategy
When `layers_to_save` is selective, TorchLens runs an exhaustive metadata pass and then a fast
activation-saving pass. Fast mode validates graph alignment and reuses labels, modules, loops,
and graph structure from the exhaustive pass.

### Train Mode
`train_mode=True` is the supported path for losses built from saved activations. It keeps
floating saved tensors graph-connected, preserves user `requires_grad` settings, and rejects
contradictory detach or disk-only settings.

### Appliance Subfolders and Extras
Appliance packages reserve user-facing namespaces by extra group. They should not import heavy
optional dependencies unless the corresponding namespace is imported.

## Shipped vs Stubbed Appliances

| Appliance | Extra | Docstring intent | Current state |
|-----------|-------|------------------|---------------|
| `torchlens.viewer` | `viewer` | Interactive HTML viewer, side panel, lazy expansion, search | Stub, `__all__ = []`, no dependencies |
| `torchlens.paper` | `paper` | Publication figures and 3D cuboid renderers | Stub, `__all__ = []`, no dependencies |
| `torchlens.notebook` | `notebook` | IPython/Jupyter notebook integration | Import gate for `IPython` and `jupyter_client`, no public objects |
| `torchlens.llm` | `llm` | Attention diagrams, logit lens, sequence selectors | Stub, `__all__ = []`, no dependencies |
| `torchlens.neuro` | `neuro` | RDM, CKA, Brain-Score, representation helpers | Import gate for `rsatoolbox` and `brainscore_core`, no public objects |

Extras in `pyproject.toml` also cover `viz`, `tabular`, `captum`, `sae`, `profiler`,
`lightning`, `wandb`, `hf`, `gradcam`, `shap`, `inseq`, `steering`, `repeng`, `dialz`,
`nnsight`, `lit`, `depyf`, `compat-shims`, `vision-shims`, `all`, `all-stretch`, `dev`,
`io`, and `test`.

## Notebook Galleries

| Location | Contents |
|----------|----------|
| `examples/5min/` | Seven quickstart notebooks plus README |
| `examples/50min/` | Nine longer workflow notebooks plus README |
| `examples/recipes/` | Five recipe notebooks plus README |
| `examples/intervention/` | Sixteen numbered Python intervention examples plus README |
| `examples/demos/` | Bundle-diff hero demo notebook and SVG artifact |
| `notebooks/audit/` | User-facing 2.x audit notebooks |
| `notebooks/` | Backward, fastlog, and training tutorials |

The total-audit notebooks are CI-gated through `scripts/generate_audit_coverage_manifest.py`
and `scripts/check_audit_coverage.py`.

## Release Pipeline
The release path uses semantic-release v9 configured in `pyproject.toml`. Version variables are
`pyproject.toml:project.version` and `torchlens/__init__.py:__version__`. A custom parser at
`scripts/no_major_parser.py` plus commit-msg and pre-push hooks prevent accidental major
releases; keep the 2.x family locked unless release work explicitly says otherwise. See
`feedback_version_bumps.md` in the maintainer memory path for the incident history if needed.

## Where To Look For X

| Question | Start here |
|----------|------------|
| Where is the logging toggle? | `torchlens/_state.py` |
| Where does wrapping happen? | `torchlens/decoration/torch_funcs.py`, `torchlens/decoration/model_prep.py` |
| Where is the main forward capture? | `torchlens/capture/trace.py` |
| Where are raw operation records built? | `torchlens/capture/output_tensors.py` |
| Where are inputs/buffers logged? | `torchlens/capture/source_tensors.py` |
| Where is backward capture? | `torchlens/capture/backward.py` |
| Where are labels assigned? | `torchlens/postprocess/labeling.py` |
| Where is loop detection? | `torchlens/postprocess/loop_detection.py` |
| Where is conditional branch attribution? | `torchlens/postprocess/control_flow.py`, `torchlens/postprocess/ast_branches.py` |
| Where are Layers and Modules built? | `torchlens/postprocess/finalization.py` |
| Where are field orders defined? | `torchlens/constants.py` |
| Where is `Trace.__getitem__` behavior? | `torchlens/data_classes/interface.py` and `trace.py` |
| Where is portable save/load? | `torchlens/_io/bundle.py`, `torchlens/_io/tlspec.py` |
| Where is manifest schema validation? | `torchlens/validation/__init__.py`, `torchlens/schemas/tlspec_manifest_v1.json`, `torchlens/schemas/tlspec_manifest_v2.json` |
| Where is Graphviz rendering? | `torchlens/visualization/rendering.py` |
| Where is ELK layout? | `torchlens/visualization/_elk_internal/layout.py` |
| Where is NodeSpec customization? | `torchlens/visualization/node_spec.py`, `torchlens/visualization/modes.py` |
| Where is bundle diff rendering? | `torchlens/visualization/bundle_diff.py` |
| Where is fastlog storage decided? | `torchlens/fastlog/_storage_resolver.py` |
| Where are bridge integrations? | `torchlens/bridge/` |
| Where is compat reporting? | `torchlens/compat/_report.py` |
| Where are observer taps/spans? | `torchlens/observers.py` |
| Where are release guards? | `scripts/no_major_parser.py`, `scripts/check_no_breaking_markers.py`, `pyproject.toml` |

## Pointers To Deeper Docs
- Root `AGENTS.md` and `CLAUDE.md` for agent behavior and common workflows.
- Per-subpackage `CLAUDE.md` files under `torchlens/` for implementation maps.
- `.project-context/architecture.md` for the older architecture narrative; verify against code.
- `.project-context/conventions.md` for naming, fields, tests, and commit conventions.
- `ROADMAP.md` and `LIMITATIONS.md` if present in the checkout for user-facing plans/limits.
- `notebooks/total_audit/README.md` and `_coverage_manifest.json` for audit coverage.

# TorchLens Architecture

## Module Map

High-altitude roles and load-bearing files per package. The complete per-module census
(every root module and package, one-line roles) is the "Top-level module inventory"
section below; this map only calls out structure worth knowing before editing.

### `torchlens/_state.py`
Global toggle and session state. Single source of truth for the `_logging_enabled` bool
checked by every wrapper, the active `Trace` reference, and capture-control mutable state.
Keeps its own imports minimal (stdlib plus `errors._base`) to stay near the bottom of the
dependency graph.

### `torchlens/user_funcs.py`
Public API entry points: `tl.trace()` lives here, along with `show_model_graph()`, the
authenticated capture cache, and admin helpers (`list_logs()`, `reset_naming_counter()`).
Orchestrates the two-pass strategy for selective saves via
`_run_model_and_save_specified_outs()`.

### `torchlens/constants.py`
`*_FIELD_ORDER` tuples (canonical field sets for the portable schema) and torch function
discovery sets.

### `torchlens/backends/torch/`
Eager torch capture — the wrapping and logging that used to live in `decoration/` plus the
per-op logging split:
- `wrappers.py` — lazy `wrap_torch()` (installed at first capture, stays until an explicit
  `unwrap_torch()`), DeviceContext handling.
- `model_prep.py` — permanent + per-session model preparation.
- `ops.py` + `_ops_*.py` — bottom-level op logging: exhaustive/fast dispatch, barcode
  nesting detection, argument/activation/retention handling.
- `sources.py` / `tensor_tracking.py` — source-node logging and parent-child links.
- `belt.py` / `rescue.py` — stale pre-wrap reference safety net (derived protocol-invisible
  belt and a wrap-time module-global sweep of every wrapped callable + disclosed mode-rescue
  rerun).
- `_completeness_*.py` / `completeness_witness.py` — capture-completeness witness.
- `collectives.py` — explicit `torch.distributed` collective boundary ops.

### `torchlens/capture/`
Backend-neutral capture orchestration: `trace.py` (forward-pass runner and session
setup/cleanup), `projections.py` (sparse predicate recording and `OpEvent` emission),
`predicates.py`/`stop.py` (capture decisions, halt/nonfinite), `outcome.py` (settled
capture outcomes), `arg_positions.py` (tensor extraction via static table → dynamic cache
→ BFS fallback), `salient_args.py`, `flops.py`.

### `torchlens/postprocess/`
Steps 0-20 with DERIVED order: `_contracts.py` holds each step's declared read/write
contract plus the direction authorities, and `_executor.py` derives the edges and runs
rank-keyed Kahn (import refuses if the derived order and registry disagree). Step bodies:
`graph_traversal.py` (outputs, ancestors, orphans, distances), `control_flow.py` +
`ast_branches.py` (conditional attribution), `loop_detection.py` +
`loop_grouping_adapter.py` (step-7 recurrence grouping through the shared backend-neutral
grouper), `labeling.py`, `finalization.py`.

### `torchlens/data_classes/`
The product types: `Trace` (`trace.py`, with its behavior split across `_trace_*.py`
modules), `Op`, `Layer`, `Module`/`ModuleCall`, `Param`, `Buffer`, accessor classes,
lookup (`interface.py`, `_lookup_keys.py`), and teardown (`cleanup.py`).

### `torchlens/validation/`
The tripwire: `core.py` (forward replay orchestration), `backward.py`,
`invariants.py` + the `_invariants_*.py` families (connectivity, topology, payloads,
conditionals, backward, buffers, equivalence, modules/params), `exemptions.py`
(narrow, contract-scoped exemptions only — see the validation-integrity rule).

### `torchlens/visualization/`
Graphviz rendering pipeline (`_render_dot.py` orchestrates; nodes/edges/leaf/common split
across `_render_*.py`), smart collapse v2 (`auto_collapse.py`, `collapse_plan.py`,
`collapse_optimizer.py`), `renderers/` (graphviz backend), and the dagua bridge — the
dagua renderer itself is explicit opt-in via `torchlens.experimental.dagua`.

### `torchlens/utils/`
Focused stateless helpers: `_torch_compat.py` (the ONLY home for fragile torch-private
probes and cross-version signatures), `rng.py`, `hashing.py`, tensor ops, argument
handling, introspection, collections, display.

## Data Flow

```
import torchlens                  # torch stays clean; wrapping is lazy
tl.trace(model, x)
  → backend resolution (BackendSpec registry; eager torch is the stable default)
  → first torch capture installs wrap_torch() through model preparation
    (wrappers stay installed until an explicit unwrap_torch())
  → model prep (backends/torch/model_prep.py): permanent stamps + per-session state
  → forward pass under the _logging_enabled toggle
    → each resolved torch call hits its wrapper (backends/torch/ops.py)
    → barcode nesting detection keeps only bottom-level ops
    → Op records accumulate on the live Trace
  → postprocess(trace): steps 0-20 in derived order
    (graph cleanup → conditional attribution → loop detection → labeling →
     finalization)
  → returns Trace
```

Key types flowing between modules:
- `Trace` — top-level container for one captured forward pass
- `Op` — one executed callable invocation (the dataflow graph's nodes)
- `Layer` — ops grouped across recurrent passes of the same layer
- `Module` / `ModuleCall` — per-module and per-call module metadata
- `Param` / `Buffer` — parameter and buffer records
- `Recording` / `OpEvent` — the sparse predicate-recording path (`tl.record`)

## Key Abstractions

### Toggle Architecture
Single `_logging_enabled` bool in `_state.py`. Wrappers check it on every call — when
False, one branch check, negligible overhead. No re-wrapping/un-wrapping per forward pass.

### Two-Pass Strategy
When a selective `save=` needs structure it cannot know up front, Pass 1 runs exhaustive
to discover full graph structure and Pass 2 runs fast, saving only the requested
activations. Counter alignment between passes is maintained via identical increment logic.

### Conditional Branch Attribution (Step 5)
Step 5 runs as six ordered phases:
1. 5a builds AST file indexes for source files referenced by terminal bool frames.
2. 5b classifies terminal scalar bools and records structural `ConditionalKey`s.
3. 5c materializes dense `Trace.conditional_records` IDs and rewrites bool metadata.
4. 5d runs the backward-only flood that marks branch-start parents.
5. 5e attributes ops and forward edges to branch arms, populating
   `conditional_arm_entry_edges` and `conditional_arm_children`.
6. 5f derives legacy THEN/ELIF/ELSE views and records `conditional_edge_call_indices`
   for rolled-mode divergence.

Primary branch metadata is cond-id-aware:
- `Trace.conditional_records` stores the canonical event records.
- `Trace.conditional_arm_entry_edges` stores arm-entry edges keyed by
  `(cond_id, branch_kind)`.
- `Trace.conditional_edge_call_indices` stores call indexes for rolled edges whose arm
  labels vary across passes.
- `conditional_arm_children` on `Op` / `Layer` stores per-node branch children.

Legacy `conditional_then_entry_edges`, `conditional_elif_entry_edges`,
`conditional_else_entry_edges`, `conditional_then_children`, `conditional_elif_children`,
and `conditional_else_children` are derived views computed from those primary structures.

### Barcode Nesting Detection
Random barcodes detect bottom-level vs wrapper functions. Barcode set on tensor before
call; if unchanged after → no nested torch calls → log it. If changed → nested call
already logged it. Lives in the `backends/torch/_ops_*.py` logging path.

### Operation Equivalence Types
Structural fingerprint (`equivalence_class`) built from the function name and argument
structure, with module-address information appended at op creation so identical ops in
different modules do not get loop-grouped together. Consumed by step-7 loop grouping.

### Layer Delegation
Single-pass layers delegate attribute access to their one pass. Multi-pass per-pass
fields raise **ValueError** (not AttributeError, to avoid Python's property/`__getattr__`
trap that would silently mask property bugs).

## Dependency Graph

```
_state.py           ← near-bottom: stdlib + errors._base only
constants.py        ← field orders + discovery sets, imported widely
utils/              ← leaf helpers, imported widely
backends/torch/     → logs ops into the live Trace (reads _state.py)
capture/            → backend-neutral orchestration; creates data_classes records
postprocess/        → mutates the Trace after forward (derived-order steps)
data_classes/       → the product types (Trace / Op / Layer / ...)
validation/         → replays and checks finished Traces
visualization/      → renders finished Traces
user_funcs.py       → public entry; orchestrates capture → postprocess → product
```

## Known Complexity

### Loop Detection (postprocess/loop_detection.py)
Step 7. The Trace side adapts through `loop_grouping_adapter.py` to the shared
backend-neutral grouper (the same one the eager previews use), so grouping logic must
never be duplicated in the Trace adapter.

### Exhaustive/Fast-Path Split (backends/torch/_ops_exhaustive.py and siblings)
Two parallel code paths that must maintain counter alignment. Fast path skips most
metadata but must match the exhaustive path's operation ordering exactly.

### Rendering Large Graphs (visualization/)
The Graphviz DOT pipeline is the shipped renderer; readability of large graphs is
governed by smart collapse v2 (`collapse="none"|"auto"|"max"|t`) rather than an
alternative layout engine. The dagua renderer (`torchlens.experimental.dagua`) is the
experimental opt-in alternative.

### Circular References (data_classes/)
Record types hold back-references to their owning `Trace` (`Trace` ↔ `Op`, `Trace` ↔
module records); `Param` pins its `nn.Parameter`. All rely on Python's cyclic GC;
explicit `cleanup()` is available.

## Conditional Attribution Limits

Fully attributed in eager Python `forward()`:
- `if` / `elif` / `else` chains
- Ternary `IfExp` (`x if cond else y`)

Classified only, not branch-attributed:
- `assert`
- standalone `bool(x)`
- comprehension filters
- `while`
- `match` guards

Documented false negatives:
- pure Python predicates such as `if self.training:` or `if python_bool:`
- `if tensor.item() > 0:`
- shape/metadata predicates such as `if x.shape[0] > 0:`
- functional conditionals such as `torch.where`

Unsupported / source-unavailable cases:
- Jupyter or REPL cells, `exec`, `eval`
- `torch.compile`, `torch.jit.script`, `torch.jit.trace`
- `nn.DataParallel` / `DistributedDataParallel`
- monkey-patched `forward` implementations

Deferred:
- dagua conditional-edge rendering
- while-loop body attribution

### DeviceContext Bypass (backends/torch/wrappers.py)
Python wrappers bypass C-level TorchFunctionMode dispatch. Factory functions need manual
device kwarg injection when `torch.device('meta')` context is active (HuggingFace use
case).

## Top-level module inventory

Complete inventory of `torchlens/` top-level entries (every root `*.py` module and every
package directory; `schemas/` included as the one non-package data directory). Roles are
one-liners derived from each module's docstring or a skim of its contents.

### Root modules

| module | role |
| --- | --- |
| `__init__.py` | Public package entry: lazy attribute surface (`_LAZY_ATTRS`), `__all__`, no torch side effects at import |
| `_capture_state_helpers.py` | Internal model state, cache, and input helpers for public trace capture |
| `_chunked_capture_helpers.py` | Internal chunked-forward helper functions for public trace capture |
| `_chunking.py` | Input chunking helpers for forward-only chunked capture |
| `_deprecations.py` | Shared helpers for additive public-API deprecations |
| `_distributed.py` | Detection of distributed (DTensor / device-mesh / TP / PP) model state, shared by capture entry and compat report |
| `_errors.py` | Shared internal exception types and actionable-message helpers built on `errors/_base` |
| `_fast_run.py` | Explicit guarded fast paths for repeated static-model execution (`run(fast=True)`) |
| `_input_coerce.py` | Duck-typed ergonomic input coercion for TorchLens entry points |
| `_input_walk.py` | Single-sourced model-input boundary traversal (normative dispatch for all input-tree walkers) |
| `_literals.py` | Shared `Literal` aliases for public option strings |
| `_robustness.py` | Tensor-variant detection and pre-flight guards for `trace` (meta/sparse/fake refusals) |
| `_runnable_attestation.py` | Numeric attestation and nondeterminism checks for runnable execution |
| `_runnable_call_arguments.py` | Sparse-call argument decoding and binding |
| `_runnable_call_outputs.py` | Sparse-call output binding and mutation checks |
| `_runnable_execution.py` | Transactional execution providers for the unified `Trace.run` surface |
| `_runnable_input_aliases.py` | Input alias topology and non-tensor tree contracts |
| `_runnable_input_metadata.py` | Input structure, literal, and metadata witness helpers |
| `_runnable_input_sites.py` | Live input sites and metadata contract checks |
| `_runnable_output_contracts.py` | Output reconstruction and post-execution contracts |
| `_runnable_path_faithfulness.py` | Path-faithfulness and state comparison helpers |
| `_runnable_providers.py` | Public provider entry points and run finalization |
| `_runnable_seam.py` | Narrow ownership seam between `Trace` and sparse runnable internals |
| `_runnable_state.py` | Non-executing state binding and allocation for sparse runnable traces |
| `_runnable_state_context.py` | State contracts and captured execution contexts |
| `_runnable_transaction.py` | Loaded-sparse transaction execution and allocation checks |
| `_runnable_verification.py` | Seed, RNG, attestation, and fork utilities for runnable runs |
| `_runnable_witness_contracts.py` | Control, shape, and host-escape witness checks |
| `_save_budget.py` | Running budget for retained activation bytes with typed refusal (`SaveBudgetExceededError`) |
| `_source_links.py` | Source-location link formatting helpers |
| `_split_rebind.py` | Compatibility helpers for behavior-preserving module decomposition |
| `_state.py` | Global state for toggle-gated decoration; single source of truth for capture-control mutable state |
| `_trace_selector_helpers.py` | Internal predicate and selector helpers for public trace capture |
| `_trace_state.py` | Run-state ownership for intervention execution (deliberately outside `intervention/`) |
| `_training_validation.py` | Shared validation helpers for training-compatible capture modes |
| `_transport.py` | Device/layout-aware host transport for tensor digest and codec paths |
| `_user_public_impls.py` | Private implementations backing public user-facing utility commands |
| `captured_run.py` | Shared public base types for TorchLens captured runs |
| `constants.py` | FIELD_ORDER tuples (canonical field sets) and torch function discovery sets |
| `facets.py` | Lazy alias stub: self-replaces in `sys.modules` with canonical `torchlens.semantic.facets` |
| `hash.py` | Provisional public structural-hash helpers |
| `observers.py` | User observer helpers: taps, scalar logs, record spans |
| `options.py` | Grouped option dataclasses (`CaptureOptions`, ...) for public TorchLens APIs |
| `quantities.py` | Numeric quantity types with unit-aware display |
| `runnable.py` | Frozen type contracts (enums and schema shapes) for sparse runnable `.tlspec` artifacts |
| `types.py` | Public type aliases and rarely used data classes |
| `user_funcs.py` | Public API entry points; contains every user-facing function |

### Packages

| module | role |
| --- | --- |
| `_io/` | Portable save/load implementation: scrubs a Trace to metadata plus safetensors blobs, writes/rehydrates directory bundles |
| `_trace_core/` | Private per-trace semantic store substrate (numpy-backed typed columns, per-trace interning) |
| `accessors/` | Accessor classes for TorchLens log collections |
| `attribution/` | Input-attribution methods for TorchLens |
| `autoroute/` | Auto-routing registries for model input and output handling (`autoroute.input`, `autoroute.output`) |
| `backends/` | Backend Protocol, public registry exports, and per-backend adapters |
| `bridge/` | External-tool bridge namespace (Captum, HF, SHAP, SAE Lens, LIT, profiler, ...) |
| `bundle/` | Single `Bundle` type for intervention-ready TorchLens model logs |
| `callbacks/` | Callback integration namespace with lazy Lightning support |
| `capture/` | Real-time tensor operation capture: source/output tensor logging, family tracking, forward-pass orchestration |
| `compat/` | Compatibility adapters and runtime support reports (`tl.compat.report`) |
| `data_classes/` | Core data structures representing a logged forward pass |
| `debug/` | Power-user debugging helpers for completed traces (`bisect_nan`, `hot_path`, ...) |
| `distributed/` | Distributed capture opt-in: arming, group lifetime identity, and evidence |
| `errors/` | Public TorchLens exception classes (base hierarchy, runnable errors, legacy path aliases) |
| `examples/` | Example-loading namespace for small TorchLens artifacts |
| `experimental/` | Experimental APIs with unstable naming and behavior |
| `export/` | Static export helpers for TorchLens logs |
| `fastlog/` | Lightweight predicate-recording namespace (`tl.record`) |
| `intervention/` | Import surface for intervention selectors, hooks, reruns, and bundles |
| `io/` | Public I/O and administrative helpers (log admin, intervention-spec save) delegating to `_io` and `user_funcs` |
| `ir/` | Internal backend-neutral IR for capture unification (op records, events, selectors, workspaces) |
| `merged/` | Cross-rank trace merging (rung C1): `merge_ranks`, `MergedTrace`, frozen merge vocabularies |
| `neuro/` | Extras-gated neuroscience namespace, import-inert, no public objects yet |
| `notebook/` | Extras-gated notebook namespace, import-inert, no public objects yet |
| `partial/` | Partial capture helpers for failed TorchLens forward ops |
| `postprocess/` | Postprocessing pipeline cleaning the model log after the forward pass |
| `receptive_field/` | Lazy public namespace for receptive-field analysis |
| `repgeom/` | Import-clean representation geometry helpers (NumPy + torch only), provisional |
| `report/` | Reporting helpers for TorchLens observer metadata |
| `schemas/` | JSON Schema documents for `.tlspec` manifests (v1/v2); data directory, not a Python package (no `__init__.py`) |
| `semantic/` | Semantic facet views: canonical home of facets (registry, recipes, patching) |
| `stats/` | Streaming statistics for out aggregation |
| `utils/` | Focused utility modules: RNG, tensor ops, argument handling, introspection, collections, hashing, display |
| `validation/` | Validation subpackage: saved outs, backward capture, and metadata invariants (the tripwire) |
| `visualization/` | Computational graph visualization via Graphviz (DOT rendering, smart collapse, dagua renderer dispatch) |
| `viz/` | Visualization convenience namespace: activation/tensor plots (montage, heatmaps, feature-map evolution) plus `bundle_diff` re-export |

### Dual homes (pending adjudication)

Four name pairs currently have two homes. Facts as of this writing (verified by reading);
the SF-33 adjudication decision will be recorded here when made -- this note does not
decide it.

- `torchlens/io` + `torchlens/_io`: `_io/` implements the portable save/load path
  (Trace scrub to metadata plus safetensors blobs, bundle write and rehydration);
  `io/` is the public-facing helper namespace (log administration such as `list_logs`
  and `reset_naming_counter`, intervention-spec save) that delegates to `_io` and
  `user_funcs`.
- `torchlens/viz` + `torchlens/visualization`: `visualization/` is the graph-rendering
  engine (Graphviz DOT emission, smart collapse, dagua renderer dispatch) behind `draw` and
  `show_model_graph`; `viz/` is a user-facing convenience namespace for
  activation/tensor plotting (montage, text tables, feature-map evolution, heatmaps,
  image scatter, line plots) and re-exports `bundle_diff` from `visualization/`.
- `torchlens/errors/` + `torchlens/_errors.py` (+ `intervention/errors.py`): `errors/`
  is the public exception package (`_base` hierarchy, runnable exception classes,
  legacy exception-path aliases); `_errors.py` holds shared internal exception types
  and actionable-message helpers built on `errors._base`; `intervention/errors.py`
  owns the intervention error catalog (severity tags, field-formatted messages) and
  imports from both.
- `torchlens/facets.py` + `torchlens/semantic/`: `semantic/` is the canonical
  implementation (`semantic/facets.py` plus `recipes` and `patching`); top-level
  `facets.py` is a lazy import-alias stub that replaces itself in `sys.modules` with
  `torchlens.semantic.facets` so the two module objects are identity-equal.

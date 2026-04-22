# TorchLens Architecture

## Module Map

### `torchlens/_state.py` (~208 lines)
Global toggle, session state, context managers. Single source of truth for `_logging_enabled`
bool checked by every decorated wrapper. Also stores pre-computed lookup tables, WeakSet of
prepared models, active ModelLog reference. **Must never import other torchlens modules**
(prevents circular deps).

### `torchlens/user_funcs.py` (~664 lines)
Public API: `log_forward_pass()`, `show_model_graph()`, `validate_forward_pass()`,
`get_model_metadata()`, `validate_batch_of_models_and_inputs()`. Orchestrates the two-pass
strategy when selective layers requested.

### `torchlens/constants.py` (~645 lines)
7 FIELD_ORDER tuples (canonical field sets for LayerPassLog, ModelLog, etc.), function
discovery sets (~90 IGNORED_FUNCS, ORIG_TORCH_FUNCS listing ~2000 functions to decorate).

### `torchlens/decoration/` (2 files, ~1,710 lines)
- `torch_funcs.py` — One-time decoration of ~2000 torch functions. Core interceptor with
  barcode nesting detection, in-place detection, DeviceContext bypass.
- `model_prep.py` — Two-phase model preparation (permanent `_prepare_model_once` + per-session
  `_prepare_model_session`). Module forward decorator with exhaustive/fast-path split.

### `torchlens/capture/` (7 files, ~4,960 lines)
Real-time tensor operation logging during forward pass.
- `trace.py` — Forward-pass orchestration, session setup/cleanup
- `output_tensors.py` — Core logging: builds LayerPassLog entries, exhaustive/fast dispatch
- `source_tensors.py` — Logs input and buffer tensors as source nodes
- `tensor_tracking.py` — Barcode system, parent-child links, backward hooks
- `arg_positions.py` — O(1) tensor extraction via 3-tier lookup (639 static entries)
- `salient_args.py` — Extracts significant function args for metadata
- `flops.py` — Per-operation FLOPs computation (~290 ops)

### `torchlens/postprocess/` (6 files, ~3,179 lines)
18-step pipeline. Order is critical — many steps depend on prior output.
- `graph_traversal.py` — Steps 1-4: output layers, ancestor marking, orphan removal, distance flood
- `control_flow.py` — Steps 5-7: six-phase conditional attribution (AST indexing, bool
  classification, event materialization, backward flood, forward arm attribution, derived
  views), module fixing, buffer cleanup
- `loop_detection.py` — Step 8: isomorphic subgraph expansion, layer assignment
- `labeling.py` — Steps 9-12: label generation, rename, trim/reorder, lookup keys
- `finalization.py` — Steps 13-18: undecorate, ParamLog, ModuleLog, LayerLog, mark complete

### `torchlens/data_classes/` (10 files, ~3,821 lines)
- `model_log.py` — ModelLog: top-level container, 70+ attrs
- `layer_pass_log.py` — LayerPassLog: per-pass entry (~85+ fields)
- `layer_log.py` — LayerLog: aggregate class grouping passes
- `buffer_log.py` — BufferLog(LayerPassLog): buffer-specific computed properties
- `module_log.py` — ModuleLog, ModulePassLog, ModuleAccessor
- `param_log.py` — ParamLog (lazy grad via `_param_ref`)
- `func_call_location.py` — Structured call stack frame with lazy properties
- `internal_types.py` — FuncExecutionContext, VisualizationOverrides
- `interface.py` — ModelLog query methods: `__getitem__`, `to_pandas()`, 7-step lookup cascade
- `cleanup.py` — Post-session teardown, cycle breaking

### `torchlens/validation/` (3 files, ~2,795 lines)
- `core.py` — BFS orchestration, forward replay, perturbation checks
- `exemptions.py` — 4 data-driven exemption registries + 16 posthoc checks
- `invariants.py` — 18 metadata invariant categories (A-R): structural + semantic

### `torchlens/visualization/` (3 files, ~2,777+ lines)
- `rendering.py` — Graphviz rendering: nodes, edges, module subgraphs, IF/THEN labels, override system
- `elk_layout.py` — ELK-based layout for large graphs, Worker thread, sfdp fallback
- `dagua_bridge.py` — ModelLog → DaguaGraph conversion for dagua renderer

### `torchlens/utils/` (7 files, ~950 lines)
Stateless helpers: arg handling, tensor ops (safe_copy, tensor_nanequal), RNG capture/restore,
barcode hashing, object introspection, display formatting, collection manipulation.

## Data Flow

```
import torchlens
  → decorate_all_once()       # wraps ~2000 torch functions permanently
  → patch_detached_references()  # patches `from torch import cos` style refs

log_forward_pass(model, input)
  → _prepare_model_once(model)   # permanent: tl_module_address, forward wrappers
  → _prepare_model_session(model) # per-call: requires_grad, buffers, session attrs
  → active_logging(model_log)    # enables _logging_enabled toggle
  →   model(input)               # forward pass — each torch op hits decorated wrapper
  →     torch_func_decorator     # barcode nesting → bottom-level ops logged
  →       log_function_output_tensors_exhaustive()  # builds LayerPassLog entry
  →       OR log_function_output_tensors_fast()     # reuses prior graph structure
  → postprocess(model_log)       # 18-step pipeline
  →   Steps 1-4: graph cleanup (outputs, ancestors, orphans, distances)
  →   Steps 5-7: control flow (Step 5a-5f conditional attribution, module fixing, buffer dedup)
  →   Step 8: loop detection (isomorphic subgraph expansion)
  →   Steps 9-12: labeling (raw→final labels, rename, reorder, lookup keys)
  →   Steps 13-18: finalization (undecorate, ParamLog, ModuleLog, LayerLog)
  → return ModelLog
```

Key types flowing between modules:
- `Dict[str, Dict]` — raw tensor dict during capture (`_raw_tensor_dict` on ModelLog)
- `LayerPassLog` — per-pass tensor operation entry (~85+ fields)
- `LayerLog` — aggregate grouping passes of the same layer
- `ModuleLog` / `ModulePassLog` — per-module metadata
- `ParamLog` — per-parameter metadata with lazy gradient access

## Key Abstractions

### Toggle Architecture
Single `_logging_enabled` bool in `_state.py`. Wrappers check it on every call — when False,
one branch check, negligible overhead. No re-wrapping/un-wrapping per forward pass.

### Two-Pass Strategy
When user requests specific layers (not "all"/"none"), Pass 1 runs exhaustive to discover full
graph structure, Pass 2 runs fast saving only requested activations. Counter alignment between
passes maintained via identical increment logic.

### Conditional Branch Attribution (Step 5)
Step 5 now runs as six ordered phases:
1. 5a builds AST file indexes for source files referenced by terminal bool frames.
2. 5b classifies terminal scalar bools and records structural `ConditionalKey`s.
3. 5c materializes dense `ModelLog.conditional_events` IDs and rewrites bool metadata.
4. 5d runs the backward-only flood that marks branch-start parents.
5. 5e attributes ops and forward edges to branch arms, populating
   `conditional_arm_edges` and `cond_branch_children_by_cond`.
6. 5f derives legacy THEN/ELIF/ELSE views and records `conditional_edge_passes` for
   rolled-mode divergence.

Primary branch metadata is cond-id-aware:
- `ModelLog.conditional_events` stores the canonical event records.
- `ModelLog.conditional_arm_edges` stores arm-entry edges keyed by `(cond_id, branch_kind)`.
- `ModelLog.conditional_edge_passes` stores pass numbers for rolled edges whose arm labels
  vary across passes.
- `cond_branch_children_by_cond` on `LayerPassLog` / `LayerLog` stores per-node branch children.

Legacy `conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges`,
`cond_branch_then_children`, `cond_branch_elif_children`, and `cond_branch_else_children`
are derived views computed from those primary structures.

### Barcode Nesting Detection
Random 8-char barcodes detect bottom-level vs wrapper functions. Barcode set on tensor before
call; if unchanged after → no nested torch calls → log it. If changed → nested call already
logged it.

### Operation Equivalence Types
Structural fingerprint: `{func_name}_{arg_hash}[_outindex{i}][_module{origin}]`. Used by
loop detection (Step 8) to group operations into layers.

### LayerLog Delegation
Single-pass layers: `__getattr__` delegates to `passes[1]`. Multi-pass per-pass fields:
raises **ValueError** (not AttributeError, to avoid Python's property/__getattr__ trap).

## Dependency Graph
```
_state.py          ← imported by everything (no outgoing torchlens imports)
constants.py       ← imported by capture/, postprocess/, data_classes/
utils/             ← imported by capture/, postprocess/, data_classes/, validation/
decoration/        → calls capture/ (via decorated wrappers)
                   → reads _state.py
capture/           → creates data_classes/ entries (LayerPassLog)
                   → reads _state.py, constants.py
postprocess/       → mutates data_classes/ entries
                   → reads constants.py
data_classes/      → references _state.py (TYPE_CHECKING only)
validation/        → reads data_classes/, calls original torch funcs
visualization/     → reads data_classes/ (LayerLog, ModelLog)
user_funcs.py      → orchestrates decoration/, capture/, postprocess/, validation/, visualization/
```

## Known Complexity

### Loop Detection (postprocess/loop_detection.py)
Most complex single module. BFS expansion of isomorphic subgraphs, iso group refinement with
direction-aware neighbor connectivity, adjacency union-find for layer assignment. Step 6's
module suffix mutation makes `_rebuild_pass_assignments` necessary (not defensive). ~826 lines.

### Exhaustive/Fast-Path Split (capture/output_tensors.py)
Two parallel code paths that must maintain counter alignment. Fast path skips most metadata
but must match exhaustive path's operation ordering exactly.

### ELK Layout (visualization/elk_layout.py)
Node.js subprocess with V8 heap sizing, Worker thread to prevent stack overflow, stress
algorithm with O(n^2) memory (NEVER use for >100k nodes), Kahn's topological sort for seeding.

### Circular References (data_classes/)
ModelLog ↔ LayerPassLog ↔ ModelLog cycles. ModuleLog ↔ ModelLog cycles. ParamLog pins
nn.Parameter. All rely on Python's cyclic GC. Explicit `cleanup()` available.

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
- ELK conditional rendering
- while-loop body attribution

### DeviceContext Bypass (decoration/torch_funcs.py)
Python wrappers bypass C-level TorchFunctionMode dispatch. Factory functions need manual
device kwarg injection when `torch.device('meta')` context is active (HuggingFace use case).

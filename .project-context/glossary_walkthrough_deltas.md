# Glossary Walkthrough Deltas

Running list of rename/design decisions made during JMT's glossary walkthrough
(2026-05-03+). Apply as a fixup pass after the walkthrough completes.

Format: scope | old | new | rationale

---

## FOUNDATIONAL — drop `Log` suffix across all sub-Trace classes

Class names lose the `Log` suffix uniformly. `Trace` already broke the `Log` pattern (was `ModelLog`); the rest follow. Symmetry is load-bearing — one convention, no special cases. Cascades through every other delta in this file (read every class name below as the new short form).

| Was | New |
|---|---|
| `OpLog` | `Op` |
| `LayerLog` | `Layer` |
| `ModuleLog` | `Module` |
| `ModuleCallLog` | `ModuleCall` |
| `ParamLog` | `Param` |
| `BufferLog` | `Buffer` |
| `GradFnLog` | `GradFn` |
| `GradFnCallLog` | `GradFnCall` |

`Trace` unchanged (already short form).

### Rationale

- **Symmetry is its own win.** Eight classes, one convention, zero special cases. Asymmetric (`Op` + `Layer` + `ModuleLog`) is a paper cut on every type hint forever.
- **`Log` telegraphs implementation, not concept.** Users care that they have an Op, not that they have an Op's log record.
- **Trace precedent.** `ModelLog` -> `Trace` rename caused zero confusion; users adapted instantly. Same expected here.
- **API reads cleaner.** `def fn(op: Op, layer: Layer) -> Module:` vs `def fn(op: OpLog, layer: LayerLog) -> ModuleLog:`.
- **Docs read more like English.** "Each Module in the trace..." vs "Each ModuleLog in the trace...".

### `Module` / `nn.Module` collision risk — accepted

The one real concern is `Module` shadowing `torch.nn.Module`. Concluded the collision is theoretical, not practical:

- Idiomatic torch references `nn.Module` qualified, not bare `Module` import.
- Idiomatic TorchLens uses `import torchlens as tl` and `tl.Module` qualified.
- Variable names (`module = trace.modules["x"]`) are lowercase — no class collision.
- Repr distinguishes: `<torchlens.Module 'encoder' calls=3>` vs `<torch.nn.modules.linear.Linear>`.
- `isinstance(x, Module)` ambiguity is real but rare; users doing TorchLens isinstance know which class they want.
- mypy/pylance flag any `from torchlens import Module` + `from torch.nn import Module` redefinition.

Mitigations baked into convention:

1. Docs always show `tl.Module` (qualified) in code samples; never bare `Module` import.
2. Glossary entry for `Module` opens with explicit disambiguation from `nn.Module` — first thing users read.
3. `__repr__` includes `torchlens.` namespace prefix on all sub-Trace classes.
4. README import section shows `import torchlens as tl` as canonical idiom.
5. Type hints in TorchLens-internal code use bare names; user-extension docs show `tl.X` qualified.

### Cascade scope

- All class definitions (`op_log.py` -> `op.py`, etc. — file renames in scope; verify import surface)
- All type hints in `torchlens/`, tests, notebooks, bridge adapters
- `FIELD_ORDER` constants in `constants.py` (constant names keyed on class — e.g. `OPLOG_FIELD_ORDER` -> `OP_FIELD_ORDER`)
- Glossary: every entry header and every cross-reference
- `__all__` in `torchlens/__init__.py`
- All docstrings referencing the old class names
- All `.project-context/` notes (audit notes, decision records, this file)
- Audit notes references throughout

### Backwards compatibility

Pre-2.0 marketing flag — no released user base depending on the long names yet. **Hard rename, no deprecation shim.** Land it in the rename sprint before public launch.

If a shim is wanted later for stragglers reading old blog posts / GH issue text: 8 one-line aliases (`OpLog = Op`) in a deprecated namespace module, removed in 3.0. Decide during the rename sprint; default is no shim.

### Apply notes

- Cascade through every other delta in this file mechanically: search-replace `OpLog` -> `Op`, `LayerLog` -> `Layer`, etc., in the deltas themselves before applying the field-level renames they describe.
- File renames (`op_log.py` -> `op.py` etc.) decided during the rename sprint, not locked here — naming is the lock; module-file organization is implementation detail.

---

## Trace — backward method locked as `Trace.backward()`

| Was | New | Where |
|---|---|---|
| (not yet on Trace; users call `tensor.backward()` directly) | `Trace.backward()` | Trace public method |

Torch parity wins over the four-character `back()` shorthand. Pairs cleanly with the existing `backward_*` field cluster (`backward_num_calls`, `backward_peak_memory`, `backward_duration`). Every torch user already knows the verb; zero cognitive translation.

`back()` rejected because:
- Navigation/history connotations (`browser.back()`, `stack.back()`) misread as "go to previous" not "run backprop"
- Breaks vocab consistency with `backward_*` fields
- Less searchable / less obvious to torch users

Audit follow-up: confirm the backward-parity sprint (currently in `.research/backward-parity-impl_P*`) lands the method on Trace under this name.

---

## Super* family — universal rule confirmed (all 8 sub-Trace classes have Super counterparts)

Verified 2026-05-17 against `torchlens/intervention/_super/`. All eight sub-Trace classes have a Super counterpart at Bundle level. Universal rule preserved (less confusing than an arbitrary subset).

| Sub-Trace class | Super counterpart | Source file |
|---|---|---|
| `Op` | `SuperOp` | `super_op.py:16` |
| `Layer` | `SuperLayer` (extends `SuperOp`) | `super_op.py:78` |
| `Module` | `SuperModule` | `super_logs.py:21` |
| `ModuleCall` | `SuperModuleCall` | `super_logs.py:134` |
| `Param` | `SuperParam` | `super_logs.py:61` |
| `Buffer` | `SuperBuffer` | `super_logs.py:57` |
| `GradFn` | `SuperGradFn` | `super_logs.py:109` |
| `GradFnCall` | `SuperGradFnCall` | `super_logs.py:138` |

Plus matching accessors for all 8 in the same files.

### Naming under the foundational Log-drop

Super* class names are already short-form (no `Log` suffix) — they were named correctly from the start. The Log-drop cascades only through their type parameters:

| Was | New |
|---|---|
| `Super["OpLog"]` | `Super["Op"]` |
| `Super["LayerLog"]` | `Super["Layer"]` |
| `Super["ModuleLog"]` | `Super["Module"]` |
| `Super["ModuleCallLog"]` | `Super["ModuleCall"]` |
| `Super["ParamLog"]` | `Super["Param"]` |
| `Super["BufferLog"]` | `Super["Buffer"]` |
| `Super["GradFnLog"]` | `Super["GradFn"]` |
| `Super["GradFnCallLog"]` | `Super["GradFnCall"]` |

Same cascade for `SuperAccessor["XLog", SuperX]` -> `SuperAccessor["X", SuperX]` across all 8 accessor declarations.

Reaffirms earlier "Revisit later" note (line 590, "Super\* family — universal via generic `Super[T]`"): universal rule + generic base, with per-kind extensions only where they earn it (tensor-stacking on SuperOp/SuperLayer, comparison hooks on SuperModule/SuperBuffer/SuperGradFn). That note stands; the universality is now empirically locked.

---

## FOUNDATIONAL — `pass` vs `call` vocabulary split locked

Two different concepts, two different words. Considered unifying to `call` for symmetry; rejected because it would conflate two semantically distinct ideas.

| Vocab | Concept | Captures | Lives on |
|---|---|---|---|
| **`call`** | Callable invocation | "Nth invocation of this `Callable`" | `ModuleCall.call_index`, `GradFnCall.call_index`, `Module.num_calls`, `GradFn.num_calls` |
| **`pass`** | Equivalence-class iteration | "Nth iteration through this recurrent graph position" | `Op.pass_index`, `Op.num_passes` (within parent Layer) |

### Why the split is honest

- **Different concepts.** "Call" counts invocations of a literal `Callable` (Module, GradFn — fixed identity). "Pass" counts iterations within a Layer's equivalence class (Op — emerges from recurrence).
- **torch.cos example.** If torch.cos is called 17 times across the trace, those 17 Ops partition into N Layers based on graph position. Each Op has `pass_index ∈ {1..k}` within its Layer, NOT 1..17 across the trace. Renaming to `call_index` would mislead users into reading it as the trace-wide count.
- **Heritage.** TorchLens has used "pass" for the loop/recurrence ontology since early days. The Layer-as-equivalence-class framing is core to TorchLens identity; "pass" specifically captures the iteration semantic.
- **Pythonic precision for Module/GradFn.** "Call" matches `__call__` literal mechanism, HF/torch community vocab, and the `Callable` ontology. Forcing "pass" onto these would be the awkward move.

### Why the friction is mild

- `:N` colon syntax uniform regardless: `layer["conv2d_1:2"]` and `trace.modules["encoder.block1:2"]` both mean "Nth invocation within parent context." Syntax is one rule; field names carry the semantic distinction underneath.
- Field-level access (`op.pass_index` vs `mc.call_index`) is power-user territory. One-time learning hit, not a recurring tax.
- `pass` overloading with `forward_pass` / `backward_pass` is shape-distinct (compound concept vs integer field) — rarely collides.

### Confirms existing locks

- Line 100-107 (`num_passes` on Op, scope-native): **stands**.
- Line 384 (`GradFnCall.call_index` locked, vocab-aligned to class name): **stands**.

### Implications elsewhere

- `forward_pass` / `backward_pass` reserved for whole-graph passes (`forward_duration`, `backward_duration`, etc.). These compound usages don't collide with `pass_index` as a field.
- No `Op.call_index` field — would mislead. The trace-wide "Nth call to this function" concept isn't currently exposed; if it ever is, name it `func_call_index` or similar to disambiguate from the equivalence-class semantic.
- `Module.calls` / `GradFn.calls` accessor names match their concept ("collection of Callable invocations"). `Layer.ops` stays `ops` (not `passes`), because the COLLECTION is of Ops; individual Op iteration count is the `pass_index` field on the member.

---

## FOUNDATIONAL — name / address / class vocabulary unified across all log classes

Reorganizes ~15 ad-hoc patterns into two coherent rules. Driven by JMT spotting the Module/Param asymmetry (Param has `name` for bare local segment of address; Module did not).

### Two rules

**Rule 1: primary subject uses bare form.**
- `address` — full registration path
- `name` — bare local segment (last `.`-separated piece)
- `class_name` — entity's class `__name__`
- `class_qualname` — entity's class full qualname
- `cls` — entity's class object

**Rule 2: secondary entity uses prefix-qualified form.**
- `<entity>_address`, `<entity>_name`, `<entity>_class_name`, `<entity>_class_qualname`, `<entity>_cls`

### User-overridable identifiers use `label`

- `trace_label` (Trace) — user-set, used for Bundle alignment (SUPERSEDES the earlier `trace_name` lock at line 603 of this file)
- `model_label` (Trace) — user-set, defaults to `model_class_name`
- `label` family on Op/Layer/ModuleCall/GradFn/GradFnCall — generic identifier (unchanged)

### Per-class table (current → unified)

**Trace** (model is secondary entity)

| Current | Unified | Change |
|---|---|---|
| `name` | `trace_label` | rename (supersedes earlier `trace_name` lock) |
| `model_name` (overloaded: class name AND user override) | `model_class_name` + `model_label` | split + rename |
| `model_class` | `model_class_qualname` | rename (qualname convention) |
| (missing) | `model_cls` (`@property`) | NEW — `type(_source_model_ref())` if alive |

**Module** (Module IS primary subject)

| Current | Unified | Change |
|---|---|---|
| `address` | `address` | unchanged |
| (missing) | `name` (`@property`) | NEW — `address.rsplit(".", 1)[-1]`; fills the Module/Param asymmetry |
| `class_name`, `class_qualname`, `cls` | same | unchanged |
| `all_addresses`, `address_parent`, `address_children`, `address_depth` | same | unchanged |

**ModuleCall**

| Current | Unified | Change |
|---|---|---|
| `address` (inherits) | same | unchanged |
| (missing) | `name` (`@property`) | NEW — parity with Module |
| `call_label`, `call_index` | same | unchanged (call_index locked at line 384) |

**Param** (Param is primary; module is secondary)

| Current | Unified | Change |
|---|---|---|
| `address`, `name` | same | unchanged |
| `all_addresses`, `module_address`, `all_module_addresses` | same | unchanged |
| (missing) | `module_name` (`@property`) | NEW — bare local name of owning module |
| `module_class_name`, `module_class_qualname` | same | unchanged |
| (missing) | `module_cls` (`@property`) | NEW — class object for secondary entity |

**Buffer** (DEFERRED for secondary-entity expansion)

| Current | Unified | Change |
|---|---|---|
| `buffer_address` (currently on Op) | `address` | rename (drop redundant `buffer_` prefix; Buffer extends Op) |
| (missing) | `name` (`@property`) | NEW — bare local name |
| (missing) | `module_address`, `module_name`, `module_class_name`, `module_class_qualname`, `module_cls` | **DEFERRED** — add if users ask. Less frequently introspected than Param. |

**Op** (function IS what Op describes; grad_fn is secondary)

| Current | Unified | Change |
|---|---|---|
| `func_name` | `func_name` | unchanged |
| (verify) | `func_qualname` | NEW or verify — parity with `func_name` |
| `arg_names` | `arg_names` | unchanged |
| `grad_fn_name` | `grad_fn_class_name` | rename — grad_fn IS a class; full consistency with Module's `class_name` pattern (qualified for secondary entity) |
| (missing) | `grad_fn_class_qualname` | NEW |
| (missing) | `grad_fn_cls` (`@property`) | NEW |
| `buffer_address` | `address` (when buffer-sourced) | normalize per Buffer change above |
| `layer_label` (+ variants) | same | label cluster unchanged |

**Layer** — inherits Op renames via delegation (`func_name`, `grad_fn_class_name`).

**GradFn** (grad_fn class IS primary subject)

| Current | Unified | Change |
|---|---|---|
| `name` | `class_name` | rename — matches Module's pattern (GradFn IS a class) |
| (missing) | `class_qualname` | NEW — parity |
| `cls` (`@property`) | `cls` | unchanged |
| `label` | `label` | unchanged |
| `grad_fn_type` | `grad_fn_type` | unchanged |
| `module_path` | (review) | **FLAG FOR REVIEW** — unclear semantics; resolve during rename sprint |

**GradFnCall** — inherits GradFn renames.

**FuncCallLocation** (source-tracing helper)

| Current | Unified | Change |
|---|---|---|
| `func_name` | `func_name` | unchanged |
| `code_qualname` | `func_qualname` | rename — parity with `func_name`. Same value as `function.__qualname__` 99% of the time; access-route honesty (`code_*` = via `frame.f_code`) doesn't warrant separate user-facing vocabulary. If bytecode-vs-source distinction ever matters in practice, document in docstring, not field name. |

### Summary of all renames in this section

- `Trace.name` → `Trace.trace_label`
- `Trace.model_name` (overloaded) → `Trace.model_class_name` + `Trace.model_label`
- `Trace.model_class` → `Trace.model_class_qualname`
- `Op.grad_fn_name` → `Op.grad_fn_class_name`
- `Op.buffer_address` → `Op.address` (for buffer-sourced ops)
- `GradFn.name` → `GradFn.class_name`
- `FuncCallLocation.code_qualname` → `FuncCallLocation.func_qualname`

### Summary of all additions

- `Module.name` (`@property`) — bare local segment of `address`
- `ModuleCall.name` (`@property`) — parity
- `Param.module_name` (`@property`) — parity
- `Param.module_cls` (`@property`) — parity
- `Op.func_qualname` (verify or NEW)
- `Op.grad_fn_class_qualname` (NEW)
- `Op.grad_fn_cls` (`@property`)
- `Trace.model_cls` (`@property`)
- `GradFn.class_qualname` (NEW)

### Deferred

- Buffer secondary-entity refs (`module_address`, `module_name`, `module_class_name`, `module_class_qualname`, `module_cls`) — add if users request
- `GradFn.module_path` semantics review — investigate and rename during sprint

### Execution

**One rename sprint**, not phased. Risk acknowledged: ~15 renames + ~10 additions touch FIELD_ORDER, constants, save/load schema, validation, repr, all tests, and external docs. Each rename needs precise grep + replace, plus FieldPolicy map updates. ~6 days of careful work; worth doing pre-launch because every doc and code sample references these forever.

---

## Trace / BackwardSidecar — `_backend_name` → `backend` (promoted to public, typed as Literal)

| Was | New | Where |
|---|---|---|
| `Trace._backend_name` (private) | `Trace.backend` (public) | Trace; drop underscore + drop `_name` suffix |
| `BackwardSidecar.backend_name` | `BackwardSidecar.backend` | `ir/backward.py:12`; parallel cascade |

Type annotation: `Literal["torch", "mlx"]` (add `"jax"` when the JAX backend lands).

### Why public

- Cross-backend Bundle work needs it (`if any(t.backend != "torch" for t in bundle): ...`)
- Intervention compatibility checks across backends
- User scripts inspecting captures need to know which backend ran
- Architectural endpoint (single capture substrate + backend-adapter pattern) treats backend as a first-class public concept

### Why `backend` (not `backend_name` or `capture_backend`)

- The value IS a backend identifier; `_name` suffix is redundant (parallel to PyTorch's `device`, not `device_name`)
- `capture_backend` would disambiguate from `backward_memory_backend`, but the longer name buys little — different prefixes already separate the concepts visually
- `Trace.backward_memory_backend` stays as a separate field (different concept: identifies the memory tracker, not the capture backend)

### Why `Literal` type

- Type checker catches typos (e.g., `trace.backend = "tourch"` → mypy error)
- IDE auto-complete shows valid values
- Valid set documented in the annotation itself
- Zero runtime cost — value is still a plain string, no enum import, identical `.tlspec` serialization
- Backward-compatible upgrade from current plain-`str` typing

### Caveat (future)

Literal is a closed set at the type-checker level. Third-party plugin backends would either need PR-extending the Literal or casting at the boundary. Not a concern for current torch/mlx scope; revisit if plugin backends become real.

### Cascade scope

- `Trace._backend_name` → `Trace.backend` (~4 call sites in `model_log.py`, `backends/mlx/backend.py`)
- `BackwardSidecar.backend_name` → `BackwardSidecar.backend` (`ir/backward.py:12`)
- FIELD_ORDER constants: `"_backend_name"` → `"backend"`
- FieldPolicy map at `model_log.py:708`
- Write sites at `backends/mlx/backend.py:377,547`: `trace._backend_name = self.name` → `trace.backend = self.name`
- Internal reads at `model_log.py:4161,4175`: `getattr(self, "_backend_name", "torch")` → `self.backend` (with default initialized in `__init__`)

### Default value

Initialize `self.backend: Literal["torch", "mlx"] = "torch"` in `Trace.__init__()` so reads never need `getattr` with fallback; backends override at capture time.

---

## Trace / all log classes — `run_state` → `state` (+ `RunState` → `TraceState`)

| Was | New | Where |
|---|---|---|
| `Trace.run_state` | `Trace.state` | field rename |
| `RunState` enum class | `TraceState` enum class | scoped to Trace |
| `_run_state.py` module | `_trace_state.py` | module rename |
| `replace_run_state_from()` method | `replace_state_from()` | method rename |
| `append_run_state_from()` method | `append_state_from()` | method rename |

### Rationale

- "Run" is vague — new users wonder "run of what?"
- "Intervention state" was too narrow: `RERUN_PROPAGATED` (rerun with new inputs) and `APPENDED` aren't intervention concepts
- Bare `state` is generic enough to cover the full lifecycle (PRISTINE, SPEC_STALE, REPLAY_PROPAGATED, RERUN_PROPAGATED, LIVE_CAPTURED, DIRECT_WRITE_DIRTY, APPENDED) without false-narrow framing
- Enum class `TraceState` is scoped (avoids generic `State` collision with anything else)

### Cascade scope

- `Trace.run_state` field (~10 call sites in `intervention/bundle.py`, `intervention/rerun.py`, `intervention/replay.py`, `visualization/_summary_internal/_builder.py`)
- `constants.py:117`: `"run_state"` → `"state"`
- `torchlens.io` re-export name
- `_run_state.py` module → `_trace_state.py`
- All references to `RunState` enum class

---

## Trace — `saved_*` Accessor family (8 new accessors for saved-data filtering)

JMT use case: index just the saved Ops/Layers/Modules without manual filtering. Replaces existing `List[str]` fields (`ops_with_saved_outs`, `ops_with_saved_grads`) with proper Accessors.

### Activation-saved accessors

| Accessor | Filters | Count field |
|---|---|---|
| `Trace.saved_ops` | Ops with `has_saved_activation = True` | `num_saved_ops` (existing) |
| `Trace.saved_layers` | Layers containing >= 1 saved Op | `num_saved_layers` (NEW) |
| `Trace.saved_module_calls` | ModuleCalls whose output Op has saved activation | `num_saved_module_calls` (NEW) |
| `Trace.saved_modules` | Modules with >= 1 saved ModuleCall | `num_saved_modules` (NEW) |

### Gradient-saved accessors

| Accessor | Filters | Count field |
|---|---|---|
| `Trace.saved_grad_ops` | Ops with `has_saved_grad = True` | `num_saved_grad_ops` (NEW) |
| `Trace.saved_grad_layers` | Layers containing >= 1 saved-grad Op | `num_saved_grad_layers` (NEW) |
| `Trace.saved_grad_module_calls` | ModuleCalls whose output Op has saved grad | `num_saved_grad_module_calls` (NEW) |
| `Trace.saved_grad_modules` | Modules with >= 1 saved-grad ModuleCall | `num_saved_grad_modules` (NEW) |

### Semantics rule

For each level, "saved" means "contains saved data at some level beneath":
- Op level: literally has the saved tensor
- Layer level: at least one of its Ops is saved (Layer = equivalence class of recurrent Ops)
- ModuleCall level: this call's output Op is saved (Module-output boundary; NOT any internal Op)
- Module level: at least one of its ModuleCalls is saved

### Full Accessor surface (each)

- Integer + label + substring + slice indexing
- `len()`, `iter()`, `to_pandas()`
- `keys()`, `values()`, `items()`

### Deprecate / replace

| Current (`List[str]`) | New (Accessor) |
|---|---|
| `Trace.ops_with_saved_outs` | `Trace.saved_ops` |
| `Trace.ops_with_saved_grads` | `Trace.saved_grad_ops` |

List form is still accessible via `list(trace.saved_ops)` or `trace.saved_ops.keys()`.

### `saved_grad_*` naming acknowledgment

`Trace.saved_grad_ops` reads slightly clunkily ("saved-grad ops"). Alternatives considered (`ops_with_saved_grads` verbose; `grad_saved_ops` parses better but breaks `saved_` leading prefix). Going with `saved_grad_*` for consistent leading `saved_` prefix and full symmetry across 8 accessors.

### Cascade scope

- 8 new Accessor implementations (or 4 if implemented as parametric filtered-Accessor over scope + predicate)
- 6 new `num_*` count fields (or `@property` derivations from `len(accessor)`)
- 2 field deprecations (`ops_with_saved_outs`, `ops_with_saved_grads`)
- `keep_unsaved_layers` capture-time flag stays (orthogonal to these read-side accessors)
- Other references: `validation/invariants.py`, `cleanup.py:464`, internal lookup paths

---

## Trace / all log classes — `io_format_version` → `tlspec_version`

| Was | New | Where |
|---|---|---|
| `Trace.io_format_version` | `Trace.tlspec_version` | field rename |
| same field on Layer, Op, ModuleCall, GradFnCall, FuncCallLocation, Param, model_log | `tlspec_version` | cascade |
| `IO_FORMAT_VERSION` constant | `TLSPEC_VERSION` | constant rename |
| `read_io_format_version()` helper | `read_tlspec_version()` | helper rename |

### Rationale

- "io" prefix reads cryptic to new users
- `tlspec_version` directly names the format users encounter (`.tlspec` file extension)
- Matches the existing manifest filename pattern (`tlspec_manifest_v1.json`)
- Unambiguous — won't be confused with `torchlens_version`, `torch_version`, or any API version
- Pairs cleanly with peer `torchlens_version` in manifest

### Keep `_version` fully spelled

Considered shortening to `_vers` / `_ver` / `_v`. Rejected because:
- Python ecosystem convention: `__version__` (PEP 396), `python_version`, `torch_version`, `pip_version` — fully spelled is universal
- 3-char savings doesn't earn breaking the idiom
- `vers` specifically is rare in Python attribute names (even `ver` is unusual)
- Short forms reserved for filenames (`v1`), CLI flags (`-V`), and git tags (`v2.16.0`) — different contexts

### Cascade scope

- ~15-20 call sites across `_io/`, `validation/`, all 7 log-class files
- FIELD_ORDER constants
- Save/load test fixtures
- External docs referencing format version

---

## Op — confirmed lock: I/O sentinels stay as Ops; add `trace.compute_ops` for filtering

`Op` continues to include graph I/O sentinels, internal source/sink sentinels, and buffer source entries — not just torch function calls. These special-case Ops participate in the equivalence/Layer machinery uniformly.

### Why keep I/O as Ops

1. **Loop equivalence requires it.** I/O tensors in recurrent blocks have structurally-equivalent positions across loop iterations; they must be Layer members. Splitting into separate types fragments the equivalence machinery.
2. **`trace_index` spans them uniformly.** Single linear sequence; separate types would need parallel index spaces.
3. **Accessor consistency.** `trace.ops[N]` returns the Nth graph member; no need for separate `trace.io[N]` / `trace.compute_ops[N]` lookups.
4. **Visualization treats them uniformly.** No special-case branches in rendering.
5. **Predicate filtering already discriminates.** Users who want compute-only filter via `is_input`/`is_output`/`is_internal_source`/etc.

### Naming-honesty caveat (acknowledged, not blocking)

"Op" technically means "operation," and an input tensor doesn't operate. Accepted: "Op" is the TorchLens graph-node shorthand. Alternatives (`Node`, `GraphMember`) lose specificity ("captured TorchLens graph entity") without gaining clarity.

### Documentation commitment

Glossary entry for `Op` must open with: "Op = graph node. Includes torch function calls AND graph I/O sentinels (`is_input`, `is_output`), internal source/sink sentinels (`is_internal_source`, `is_internal_sink`), and buffer source entries (`is_buffer_source`). All Op subtypes participate uniformly in the equivalence/Layer machinery."

### NEW: `Trace.compute_ops` and `Trace.compute_layers` Accessors

Filter to compute-only members (torch function calls, excluding I/O and sentinel kinds).

```python
def is_compute_op(op: Op) -> bool:
    return not (
        op.is_input
        or op.is_output
        or op.is_internal_source
        or op.is_internal_sink
        or op.is_buffer_source
    )
```

| Accessor | Filters | Count field |
|---|---|---|
| `Trace.compute_ops` | Ops where `is_compute_op(op) = True` | `num_compute_ops` (NEW @property) |
| `Trace.compute_layers` | Layers whose representative Op is compute (all members structurally identical, so first-Op test suffices) | `num_compute_layers` (NEW @property) |

Full Accessor surface (each, inherits from base):
- Integer + label + substring + slice indexing
- `len()`, `iter()`, `to_pandas()`
- `keys()`, `values()`, `items()`

### Companion predicates

Add `Op.is_compute_op` and `Layer.is_compute_layer` as `@property`:

```python
@property
def is_compute_op(self) -> bool:
    return not (
        self.is_input or self.is_output
        or self.is_internal_source or self.is_internal_sink
        or self.is_buffer_source
    )

# On Layer (single-Op or aggregate)
@property
def is_compute_layer(self) -> bool:
    return self.ops[0].is_compute_op  # representative suffices; all members structurally identical
```

User-facing convenience: `op.is_compute_op` / `layer.is_compute_layer` read cleaner than the 5-term filter.

### Module / ModuleCall parity — deferred

Could extend to `compute_modules` / `compute_module_calls` for full symmetry with the `saved_*` family. Less useful:
- Nearly every Module contains compute Ops (otherwise empty); the filter would be near-tautological
- ModuleCalls similarly

Defer; add only if users request.

### Cascade scope

- 2 new Accessor implementations (or parametric instances of the scoped-filtered Accessor)
- 2 new `num_*` count fields
- 2 new `is_compute_*` `@property` predicates (on Op and Layer)
- Glossary entry for `Op` updated per documentation commitment above

---

## Architectural lock: output sentinels stay cloned (each graph-boundary event is its own Op)

Considered reversing the early decision to clone output Ops from the final compute Op (alternative: alias the final compute Op as also-being-output). Decision: **keep cloning.**

### Unifying principle

**Each graph-boundary event is its own Op.** The graph is a sequence of events; every event gets a node.

| Event | Op kind |
|---|---|
| Function call (torch op) | compute Op |
| Entering the model | input sentinel Op |
| Leaving the model | output sentinel Op |
| Appearing mid-computation from outside | internal source Op |
| Being terminated mid-computation | internal sink Op |
| Entering from buffer state | buffer source Op |

The output sentinel represents the act of LEAVING the model (returning to the user) — a real graph event in the same way an input sentinel represents the act of ENTERING. These are graph-boundary events that deserve first-class node representation.

### Load-bearing benefits of cloning

1. **Multi-output models.** `(a, b, c)` return → three output Ops, each marking one return point. Aliasing would force ambiguous multi-flag semantics on compute ops.
2. **Recurrence/Layer equivalence.** Same-position output sentinels across loop iterations participate in Layer equivalence machinery uniformly. Aliasing would require dual-identity (compute-AND-output) on the same op, breaking partition cleanliness.
3. **Intervention spec stability across model variants.** Spec referencing `output_1` survives architecture churn (final layer changing from linear to conv). Aliasing would couple specs to underlying op type.
4. **Super[X] alignment.** SuperOp aligns Ops by label. Output sentinels align by output-identity regardless of underlying final-op type; aliasing would make architecturally-different traces un-alignable as outputs.
5. **Visualization uniformity.** Output nodes are explicit nodes; zero special-case rendering branches.
6. **Save/load consistency.** Output sentinels save like any other Op.
7. **find_sites / intervention targeting.** `find_sites(tl.output())` returns clean Op refs.

### Memory consideration — FLAG AS FOLLOW-UP TODO

If the current implementation creates output sentinels via literal `tensor.clone()`, that's ~50% memory overhead per saved output activation. This is a PERF issue, not an architectural one.

**TODO: verify sentinel ops use tensor-reference-sharing rather than deep-clone.**

- Investigate: do output/input/buffer-source sentinels currently use `tensor.clone()` (deep copy) or share underlying storage with the source op?
- If deep-clone: align with the existing TODO for `save_mode="reference"` (no-copy save mode). Sentinel ops are natural candidates for default reference-sharing since the user can't meaningfully mutate the sentinel's tensor differently from the source op's.
- If already reference-shared: confirm and document.

### Documentation commitment

Glossary entry for `Op` extended with the "graph-boundary event" framing — every Op is a graph event, sentinels included.

---

## GradFn — storage flip: `is_intervening` → `has_op` (resolves intervention-API lexical clash)

Storage field flips polarity from `is_intervening` (negative form) to `has_op` (positive form). Aligns with the Trace count `num_grad_fns_without_op` already locked at line 31.

### Vocabulary

| Field | Form | Storage |
|---|---|---|
| `GradFn.has_op` | `bool`; positive predicate (`True` = forward op captured for this grad_fn) | STORED |
| `GradFn.is_intervening` | DEPRECATED — `@property` returning `not self.has_op` (deprecation warning) | derived |
| `Trace.num_grad_fns_without_op` | already locked at line 31, pairs with `has_op = False` | unchanged |
| `Trace.num_grad_fns_with_op` | NEW optional — pairs with `has_op = True`, gives symmetric partition | NEW |

### Why drop "intervening"

Lexical clash with the intervention API is real. Both share the "interven-" root but refer to unrelated concepts:

- `is_intervening` (graph topology): a grad_fn sitting BETWEEN forward-corresponding grad_fns in the backward graph
- `intervention_*` family (user-applied modifications): `find_sites`, `state` (locked rename from `intervention_state`), replay, rerun, the whole intervention API surface

New users reading both reasonably wonder "are these related?" They're not. Eliminating "intervening" from GradFn vocab removes the false-association vector entirely.

### Why `has_op` (positive) over alternatives

| Alternative | Verdict |
|---|---|
| `is_uncaptured` | new vocab; misleading (PyTorch produced these; TorchLens just didn't have a matching forward op) |
| `is_orphan` | misnamed — "orphan" implies "left behind"; these are unmatched, not abandoned |
| `is_synthetic` | TorchLens doesn't construct these; PyTorch does |
| `has_op` (chosen) | zero new vocab; positive form reads naturally; symmetric with possible `num_grad_fns_with_op` |

### Cascade scope

- Storage flip at `grad_fn_log.py:143`: `is_intervening: bool = True` → `has_op: bool = False` (default inverts)
- FieldPolicy map: `"is_intervening": FieldPolicy.KEEP` → `"has_op": FieldPolicy.KEEP`
- Remove existing legacy `has_op` deprecation at `grad_fn_log.py:190` (it's no longer deprecated)
- Add NEW `is_intervening` deprecated `@property` returning `not self.has_op` with warning
- `validation/invariants.py:234`: invert predicate
- `intervention/resolver.py:622`: invert predicate
- `fastlog/types.py:134,609`: flip default value
- `backends/torch/backward.py:355`: invert assignment
- `visualization/rendering.py:1437,1592,1618`: invert predicates
- `constants.py:622`: rename field name in FIELD_ORDER tuple
- Legacy `__setstate__` handling at `grad_fn_log.py:174-175`: migrate serialized `is_intervening` to `has_op` (invert during load)

---

## Trace / LayerLog — `total_` prefix consistency on memory aggregate sums

Apply `total_` uniformly to all aggregate-sum memory fields. Resolves the inconsistency where `total_activation_memory` / `total_gradient_memory` have the prefix but `param_memory` / `autograd_saved_memory` (also aggregate sums) lack it.

### Trace renames

| Current | New |
|---|---|
| `param_memory` | `total_param_memory` (+ `_str`) |
| `autograd_saved_memory` | `total_autograd_saved_memory` (+ `_str`) |
| `total_out_memory` | `total_activation_memory` (+ `_str`) — cascades from already-locked "activation" vocab cluster |
| `saved_out_memory` | `saved_activation_memory` (+ `_str`) — same |

Already correct (no change):
- `total_activation_memory` / `saved_activation_memory` (after `out` → `activation` rename above)
- `total_gradient_memory` / `saved_gradient_memory` (locked at delta line 43)
- `total_param_gradient_memory` (locked at delta line 45)

Distinct pattern (NOT renamed):
- `forward_peak_memory` / `backward_peak_memory` — `peak_` denotes max, not sum; semantically different concept; keep as-is

### LayerLog renames

| Current | New |
|---|---|
| `param_memory` | `total_param_memory` (+ `_str`) |

Layer-scoped "total params" = sum of param memory attributed to this Layer's function position. The `total_` is honest at the Layer scope just as at the Trace scope.

### Not propagated

- OpLog and below: per-op fields are single-thing values, not aggregates. No `total_` needed.

### Why apply `total_` uniformly

- **Uniform recognition signal**: user sees `total_*` and immediately knows "aggregate sum." No mental check on whether the bare form is a sum or some other quantity.
- **Sets up future filter prefixes**: if any future field gets a `saved_X_memory` or `actually_used_X_memory` companion, the `total_` form pre-positions the contrast.
- **Removes surface inconsistency** without changing semantics.

### Why `autograd_saved_memory` keeps `_saved_` middle

- `_saved_` is load-bearing — identifies WHICH autograd memory (the save-for-backward subset)
- Dropping to `autograd_memory` would be ambiguous: could mean graph metadata, save-for-backward, gradient buffers, intermediate state
- PyTorch's autograd has multiple memory aspects; `_saved_` picks the specific one
- With `total_` prefix: `total_autograd_saved_memory` = "total memory of autograd-saved-for-backward tensors"

### Why `activation` (not `out`)

Cascades from locked "activation" vocabulary cluster:
- `save_raw_activations`, `activation_transform`, `has_saved_activation`, `detach_saved_activations`, `Op.save_activation()`, `saved_activation_memory`
- `total_out_memory` and `saved_out_memory` were the outliers; this completes the cluster

ML community standard: "activation" is canonical for tensors flowing through the network. "Out" is generic and could mean many things.

### Cascade scope

- `Trace.param_memory` → `Trace.total_param_memory` (`model_log.py:841,1131,2867,3644`)
- `Trace.autograd_saved_memory` → `Trace.total_autograd_saved_memory` (`model_log.py:831,1119,2343`)
- `Trace.total_out_memory` → `Trace.total_activation_memory` (`model_log.py:829,1117`)
- `Trace.saved_out_memory` → `Trace.saved_activation_memory` (`model_log.py:833,1121`)
- `LayerLog.param_memory` → `LayerLog.total_param_memory` (`layer_log.py:191,295,412`)
- All `_str` companion `@property` methods
- FIELD_ORDER constants in `constants.py`
- repr / summary references
- `interface.py:193` summary string (uses `saved_out_memory_str`)

---

## Feature removal (post-rename sprint): kill `keep_unsaved_layers=False` entirely

The `keep_unsaved_layers=False` mode (drop layers without saved activations from the trace during postprocess) is killed entirely after the rename sprint completes. The just-locked `saved_*` Accessors cover the only legitimate use case.

### Order of operations

1. **Rename sprint runs first.** During the sprint, `unlogged_ops` is NOT renamed to `dropped_ops` — the field is on the chopping block.
2. **Feature removal sprint follows.** Kill the drop logic + remove all associated fields.

### What gets killed

| Item | Status |
|---|---|
| `keep_unsaved_layers` constructor flag | REMOVED entirely (not deprecated; removed) |
| `Trace.unlogged_ops` (and the rename target `dropped_ops`) | REMOVED — field deleted, no replacement |
| `_unsaved_layers_lookup_keys` internal | REMOVED |
| Postprocess "drop unsaved layers" pass (`postprocess/__init__.py:140-146`) | REMOVED |
| Drop logic in `postprocess/labeling.py:585-591` | REMOVED |
| Reference handling in `backends/torch/sources.py:490` and `backends/torch/ops.py:1786` | REMOVED |
| FieldPolicy entries for `unlogged_ops` | REMOVED |
| FIELD_ORDER entries for `unlogged_ops` | REMOVED |

### What stays (unchanged)

- Save policy still controls which Ops have `has_saved_activation = True` — selective SAVE survives
- Op records are always retained in the trace regardless of save policy
- `trace.saved_ops` / `saved_layers` / `saved_module_calls` / `saved_modules` Accessors (just locked) provide the saved-subset views

### Why this is a clean win

1. **Uniform graph topology** across all save policies
2. **Clean Bundle/Super alignment** — traces with different save policies are now structurally identical
3. **No partial-graph bugs** — every Op exists; references never dangle
4. **Simpler postprocess** — fewer steps, less cleanup logic
5. **Stable intervention specs** — Op labels stay valid across save-policy variations
6. **Removes `dropped_ops` from the API surface** — no naming bikeshed, no docs needed

### Cost of keeping all Op records

Negligible — Op metadata is ~100-500 bytes per op. For a 1000-op model: ~500KB. For 100K ops (extreme): ~50MB. Activation tensors (GBs) dominate; Op record retention is rounding error.

### Migration / deprecation

- Pre-removal release: emit deprecation warning on `keep_unsaved_layers=False` use
- Removal release: constructor accepts `keep_unsaved_layers` parameter for one cycle as a silently-ignored no-op (for backward-compat), then removed entirely
- Docs: point users at `saved_*` Accessors for the equivalent capability

---

## Trace — additional Accessors for per-invocation parity

Completes the per-invocation Accessor pattern. Currently `Trace.ops` aggregates per-invocation Op records across all Layers, but the analogous ModuleCall and GradFnCall aggregations are missing.

### New Accessors

| Accessor | Filters | Count field |
|---|---|---|
| `Trace.module_calls` | All per-invocation ModuleCall records across all Modules | `num_module_calls` (NEW) |
| `Trace.grad_fn_calls` | All per-invocation GradFnCall records across all GradFns | `num_grad_fn_calls` (NEW) |

Full Accessor surface (each):
- Integer + label + substring + slice indexing
- `len()`, `iter()`, `to_pandas()`
- `keys()`, `values()`, `items()`

### Saved-variants for GradFnCalls

GradFnCalls store `grad_inputs` and `grad_outputs` (backward-pass gradient payloads). These can be saved or `None` depending on save policy.

| Accessor | Filters | Count field |
|---|---|---|
| `Trace.saved_grad_fn_calls` | GradFnCalls where `grad_inputs is not None OR grad_outputs is not None` | `num_saved_grad_fn_calls` (NEW) |
| `Trace.saved_grad_fns` | GradFns with >= 1 saved GradFnCall | `num_saved_grad_fns` (NEW) |

Companion predicate on GradFnCall:
```python
@property
def is_saved(self) -> bool:
    return self.grad_inputs is not None or self.grad_outputs is not None
```

User who wants "fully saved" filters further: `[c for c in trace.saved_grad_fn_calls if c.grad_inputs is not None and c.grad_outputs is not None]`.

### Why keep both `grad_inputs` AND `grad_outputs` on GradFnCall

Considered whether one is redundant. Decision: keep both. Reasons:
1. **DAG backward graphs (= every real network)**: at convergence points, `A.grad_inputs` = SUM of contributions from multiple downstream grad_fns' `grad_outputs`. The pieces and the sum are different tensors.
2. **Unmatched grad_fns** (`has_op = False`): no corresponding forward Op exists, so `op.grad` cannot substitute. The grad_fn_call data is the only source.
3. **Different debugging views**: "what grad arrived at this backward invocation" vs "what grad did this invocation produce" are distinct concerns.
4. **Memory cost addressed by save policy, not field reduction.** Users who want lighter capture configure backward save to populate only one side.

### Final `saved_*` family count

10 Accessors total, organized by save-type and scope:

| Scope | Forward activation | Forward gradient (at op) | Backward grad-flow (at grad_fn) |
|---|---|---|---|
| Op | `saved_ops` | `saved_grad_ops` | (n/a) |
| Layer | `saved_layers` | `saved_grad_layers` | (n/a) |
| ModuleCall | `saved_module_calls` | `saved_grad_module_calls` | (n/a) |
| Module | `saved_modules` | `saved_grad_modules` | (n/a) |
| GradFnCall | (n/a) | (n/a) | `saved_grad_fn_calls` |
| GradFn | (n/a) | (n/a) | `saved_grad_fns` |

Asymmetry is honest — backward-side log classes capture backward invocation payloads; forward-side classes use forward-op gradient predicates.

### Cascade

- 4 new Accessors total: `module_calls`, `grad_fn_calls`, `saved_grad_fn_calls`, `saved_grad_fns`
- 4 new `num_*` count fields
- 1 new `GradFnCall.is_saved` predicate
- 1 new `GradFn.has_saved_call` predicate (similar to `has_op`)

### TODO — revisit later: should Ops have option to save INPUT tensors too?

For symmetry and convenience. Currently Ops save OUTPUTS (activations). Saving inputs is mostly redundant with parent outputs (in a DAG, op B's input from A == op A's output). But:

- For Ops without a captured parent (graph-entry, buffer-source): inputs would otherwise not be auto-saved
- Convenience: `op.input` directly accessible without navigating to parent
- Parity with `grad_inputs`/`grad_outputs` on GradFnCall — could give Op `save_inputs` / `saved_inputs` / `has_saved_input` / `Trace.saved_input_ops` Accessor

Defer; revisit during the architectural endpoint work (capture-substrate unification). Mostly-redundant with parent activations means the symmetry argument is weaker than for GradFnCall.

---

## Trace — `flops_by_type` → `flops_by_op_type`

| Was | New | Where |
|---|---|---|
| `Trace.flops_by_type` (`@property`) | `Trace.flops_by_op_type` | `model_log.py:2897` |

### Rationale

At Trace level, "by_type" is ambiguous — multiple "type" concepts coexist (op types, pass type, model class type). The grouping key is `op.type` per the source code. Explicit `flops_by_op_type` matches the field source and disambiguates.

### Why `Op.type` stays bare (no rename)

Don't rename `Op.type` → `Op.op_type`. Stuttery (`op.op_type`). Bare `type` is fine on Op since the entity scope is implicit. Same pattern as `func_name`, `class_name`: subject-log fields use bare names; explicit qualification only when scope is ambiguous (Trace-level aggregations).

### Cascade scope

- `Trace.flops_by_type` → `Trace.flops_by_op_type` (`model_log.py:2897`)
- Docstring update mentioning the grouping key

---

## Trace — `trace_annotations` → `annotations`

| Was | New | Where |
|---|---|---|
| `Trace.trace_annotations` | `Trace.annotations` | drop redundant `trace_` prefix |

### Rationale

- `trace.trace_annotations` is stuttery
- Bare `annotations` is unambiguous within the Trace's own namespace (no other "annotations" concept on Trace)
- Plain `Dict[str, Any]` — user scratch space for arbitrary metadata; type unchanged

### Cascade scope

- `Trace.trace_annotations` → `Trace.annotations` (`model_log.py:1019,2362`)
- `user_funcs.py:324`: `trace.trace_annotations[...]` → `trace.annotations[...]`
- FieldPolicy at `model_log.py:750`: `"trace_annotations"` → `"annotations"`
- `constants.py:63` FIELD_ORDER
- Default fill at `model_log.py:169`

---

## Trace — `input_id` / `model_id` → `input_object_id` / `model_object_id`

| Was | New |
|---|---|
| `Trace.input_id` | `Trace.input_object_id` |
| `Trace.model_id` | `Trace.model_object_id` |

### Rationale

- "id" reads user-facing (could be a database key, slug, tag) but values are CPython `id()` (object identity / memory address)
- `input_object_id` and `model_object_id` honestly name the source: `id()` of the object
- Standard Python vocab — "object id" is the recognized term for `id()` return

### Within-process limitation (document)

`id()` values aren't stable across processes. After save+load in a different Python session, these stored ids are meaningless for cross-session relationship comparison. Useful only within-process for in-session trace comparison. Glossary note:

> Stored Python `id()` of the object at capture time. Useful for within-session relationship checks (do two traces share an input object?). NOT comparable across processes — different runs yield different ids for the same object.

### Cascade scope

- `Trace.input_id` → `Trace.input_object_id` (`model_log.py:162,741,1010,2479,2516`)
- `Trace.model_id` → `Trace.model_object_id` (`model_log.py:158,737,1006,2475`)
- `_state.py:244,297`: `_relationship_input_id` → `_relationship_input_object_id`
- `user_funcs.py:520`: `_input_id_for_relationship_evidence` → `_input_object_id_for_relationship_evidence`
- `user_funcs.py:1034,1047,1094`: call sites
- FieldPolicy and FIELD_ORDER

---

## Trace — `input_shape_hash` → `input_signature_hash`

| Was | New | Where |
|---|---|---|
| `Trace.input_shape_hash` | `Trace.input_signature_hash` | field rename |
| `_hash_input_shapes()` helper | `_hash_input_signatures()` | helper rename |

### Rationale

`_hash_input_shapes()` at `user_funcs.py:541-559` actually hashes shape + dtype + device — NOT just shape. The name `input_shape_hash` undersells the contents.

"Signature" is recognized ML/PyTorch vocabulary for the (shape, dtype, device) descriptor tuple. Pairs cleanly with "hash" suffix so users know it's a hash digest (vs the raw tuple).

### Cascade scope

- `Trace.input_shape_hash` → `Trace.input_signature_hash` (`model_log.py:163,742,1011`)
- `_state.py:94,235,245,257,277,289,298`: `_relationship_input_shape_hash` → `_relationship_input_signature_hash`; same for parameter names
- `user_funcs.py:541`: `_hash_input_shapes` → `_hash_input_signatures`
- `user_funcs.py:1035` call site
- FieldPolicy and FIELD_ORDER

---

## Trace — `ledger` → `state_history` (supersedes earlier rename)

Original field: `operation_history`. Earlier renamed to `ledger` at delta line 568. **Now superseded by `state_history`.**

| Was | New | Where |
|---|---|---|
| `Trace.ledger` (was `operation_history`) | `Trace.state_history` | append-only record of lifecycle operations |

### Why `state_history` wins

1. **Pairs with `Trace.state`** (locked in the `run_state` → `state` / `RunState` → `TraceState` decision earlier in this file) — coherent vocabulary cluster:
   - `state` = current lifecycle snapshot (PRISTINE, SPEC_STALE, REPLAY_PROPAGATED, etc.)
   - `state_history` = ordered history of lifecycle state changes
2. **Unambiguous scope** — "state history" can only mean meta-level lifecycle, not graph events (no collision with the locked "each graph-boundary event is its own Op" framing)
3. **No abstract-metaphor opacity** — `ledger` required users to infer "ledger of what"; `state_history` answers the question in the name
4. **No vocabulary overlap** — distinct from `operation_history` (Op collision), `events` (graph-event ambiguity), `lifecycle_events` (rejected for being too clinical)

### Each entry carries event metadata

Entries aren't just state values — each carries (timestamp, transition type, payload-as-needed). `state_history` is the right abstraction even when individual entries are rich because the field-level concept IS the chronological state-change history.

If a future event doesn't correspond to a state transition (metadata-only annotation), it can either:
- Be excluded from `state_history` and stored elsewhere (e.g., as a separate `notes` field)
- OR introduce a "no-op state transition" (current → current with annotation payload)

Defer this design question; for now `state_history` matches the audit-trail framing and append-only semantic.

### Cascade scope

- `Trace.ledger` (originally `operation_history`) → `Trace.state_history`
- FieldPolicy and FIELD_ORDER references
- Any glossary/docstring that mentions `ledger` or `operation_history`
- `interface.py`, `model_log.py` references
- **Update the earlier delta lock at line 568** — `state_history` supersedes `ledger`

---

## Trace — `train_mode` → `backward_ready` (supersedes audit `differentiable` lock)

| Was | New |
|---|---|
| `Trace.train_mode` (current code) | `Trace.backward_ready` |

Earlier audit (notebook_audit_notes.md:218-243, 772) locked `train_mode` → `differentiable`. **Now superseded by `backward_ready`** based on new context: the `_ready` family with `intervention_ready` is now a coherent cluster.

### Why supersede

The audit chose `differentiable` over `for_backward` because:
- `differentiable` is "capability claim" (mathematical, parallel to PyTorch's `requires_grad`)
- `for_backward=False` reads as a double negative

`backward_ready` solves both:
- "Ready for backward" is a capability claim, just framed differently
- `backward_ready=False` does NOT read as a double negative (unlike `for_backward=False`)

And adds:
- **Coherent `_ready` family** with `intervention_ready` — single suffix users learn once and infer across
- Concrete reads cleaner than abstract in API code
- Eliminates the `train_mode` vs PyTorch `model.training` conflation

### `_ready` family

| Field | Meaning |
|---|---|
| `intervention_ready` (existing, locked) | trace captured with intervention machinery prepared |
| `backward_ready` (new) | trace captured with backward machinery prepared |
| Future-extensible | `replay_ready`, `rerun_ready`, etc. as needed |

### Cascade scope

- Constructor param: `train_mode: bool = False` → `backward_ready: bool = False`
- Storage field: `self.train_mode` → `self.backward_ready`
- FieldPolicy at `model_log.py:746`: `"train_mode": FieldPolicy.DROP` → `"backward_ready": FieldPolicy.DROP`
- All call sites: `model_log.py:911,948,1015,2358,2394-2395,3790,3796,3802,3813`
- Validation function: `validate_train_mode_postfunc_output()` → `validate_backward_ready_postfunc_output()` (`op_log.py:241`)
- Glossary line 166 entry rewording
- Update audit notes (line 28, 62, 218-243, 772) to reflect supersede
- Backward-compat: accept `train_mode=` and `differentiable=` kwargs for one deprecation cycle (warnings), then remove

---

## Trace / Layer — remove `unsupported_ops` and `unsupported_op` (post-rename sprint)

Vestigial / always empty in current code. Per `model_log.py:1258-1268` docstring:

> The current exhaustive capture path records all observed tensor-producing ops, so this returns an empty list unless future capture metadata marks a layer with `unsupported_op=True`.

Nothing in the current capture path sets `Layer.unsupported_op = True`. Both fields are reserved-for-future placeholders with no concrete use case.

### Removal scope

| Item | Action |
|---|---|
| `Trace.unsupported_ops` (`@property`, `model_log.py:1258-1275`) | DELETE |
| `Layer.unsupported_op` (bool flag) | DELETE |
| FIELD_ORDER / FieldPolicy entries | DELETE |
| Tests referencing these | DELETE (verify scope first) |
| Glossary entries | DELETE |

### Why remove

1. **Dead code adds noise** — users discover the field, expect functionality, get empty list, get confused
2. **YAGNI** — if future capture path needs to mark unsupported ops, add field back with concrete semantics then
3. **Consistent with the `keep_unsaved_layers=False` removal** — cuts vestigial features that don't earn API surface

### Order

Same sprint cohort as `keep_unsaved_layers=False` removal — runs **after** the rename sprint, before launch.

---

## Trace — `capture_full_args` → `save_arg_templates` (supersedes line-1054 `capture_args_template` lock)

| Was | New |
|---|---|
| `Trace.capture_full_args` (current code) | `Trace.save_arg_templates` |

Earlier delta lock at line 1054 renamed `capture_full_args` → `capture_args_template`. **Now superseded** because `capture_args_template` reads ambiguously — the trailing noun "template" could be interpreted as the field's data type rather than a flag for action.

### Why supersede

1. **Plural `templates`** — `save_arg_templates` reads as plural target ("save the templates"), parallel to `values` in `save_arg_values`. The plural breaks the "is template the data?" misread.
2. **Pairs visibly with `save_arg_values`** (locked at delta line 88-89) — the orthogonal save modes (values vs templates) become structurally obvious in the name:

| Flag | What it saves | Cost |
|---|---|---|
| `save_arg_values` | actual tensor values per arg | heavy memory |
| `save_arg_templates` | structural templates (slot map, shape/dtype placeholders) per arg | lightweight |

3. **Verb-form consistency** — keeps the established `save_X` convention; doesn't break the verb-flag pattern (Direction A accepted earlier in this walkthrough).

### Cascade scope

- `Trace.capture_full_args` → `Trace.save_arg_templates`
- All `user_funcs.py` constructor/kwarg sites
- `options.py` references
- Glossary line 181 entry — rewording + new field name
- **Update line-1054 delta lock** to point to `save_arg_templates`
- CHANGELOG entries (verify user-visible release notes)
- Backward-compat: accept `capture_full_args=` kwarg for one cycle (warning), then remove

---

## Trace — `total_duration` → `capture_duration` (supersedes line-546 lock)

| Was (line-546 lock) | New |
|---|---|
| `Trace.total_duration` (was `duration` @property) | `Trace.capture_duration` (+ `_str`) |

Earlier delta lock at line 546 renamed `duration` → `total_duration` for parallel with `total_*` memory aggregates. **Now superseded** — `total` doesn't honestly describe what's measured (it's not a sum across many durations; it's the duration of the capture phase).

### Why `capture_duration`

1. **Names the actual scope** — all current durations on Trace measure the capture phase
2. **Frees `total_*` for honest "sum across many" semantics** — consistent with `total_activation_memory` etc. which ARE sums; `total_duration` would imply summing many durations, which it isn't
3. **Future-proof for other phase durations** — `capture_duration` reads as one phase; future `replay_duration` / `rerun_duration` slot in cleanly as sibling phases

### Sub-phase fields — keep bare (no `capture_` prefix)

| Field | Kept bare |
|---|---|
| `setup_duration` | inherently capture-scoped |
| `forward_duration` | same |
| `cleanup_duration` | same |
| `func_calls_duration` | same (subset of forward) |
| `overhead_duration` | same (total minus func calls) |

The `capture_duration` umbrella signals scope; sub-phases inherit it. Avoiding the verbose `capture_setup_duration` / `capture_forward_duration` / etc. — lower cascade risk and the verbosity doesn't earn its keep.

If a future feature adds non-capture-time durations (replay-phase, rerun-phase, etc.), those introduce their own prefixes at that point (`replay_forward_duration` etc.) without churning the existing capture-time fields.

### Per-operation durations (replay, rerun) live in `state_history` events

No separate Trace-level aggregate fields for replay or rerun durations. Each `state_history` event (per the just-locked `state_history` rename) carries its own timestamp and duration.

User queries:
```python
[e.duration for e in trace.state_history if e.kind == "replay"]
[e.duration for e in trace.state_history if e.kind == "rerun"]
```

Aggregate `@property` helpers (`most_recent_replay_duration`, `cumulative_rerun_duration`, etc.) can be added later IF users request — event-level data is sufficient for most queries.

### Cascade scope

- `Trace.duration` (`@property`) → `Trace.capture_duration` (`@property`)
- Update line-546 delta lock to reflect new target
- repr / summary references (`model_log.py:2847-2851`)
- Glossary entries
- Tests referencing `trace.duration` or `trace.total_duration`

---

## Trace — remove `Trace.load` classmethod (keep `tl.load` module function)

| Item | Action |
|---|---|
| `Trace.load(path)` classmethod (`model_log.py:2280-2295`) | REMOVE |
| `tl.load(path)` module function | KEEP — canonical entry point |
| `Trace.save(self, path)` instance method (`model_log.py:2267`) | KEEP — natural OOP, no footgun |
| `tl.save(trace, path)` module function | KEEP — module-level pair with `tl.load` |

### Why remove

Classmethod-on-instance footgun: `trace_instance.load(other_path)` is callable (Python permits classmethod-via-instance), returns a NEW trace, and does NOT modify the instance. Subtle bug surface — user reads as "load INTO this trace," but actually gets a new trace back while the original is unchanged.

### Why `Trace.save` stays

Instance method `trace.save(path)` is natural OOP — user has a trace, they save it. No footgun (instance methods on instances are unambiguous).

### Discoverability is fine via module-level

`tl.load(path)` is surfaced in:
- README import section / canonical examples
- `tl.__all__` (`__init__.py:805`)
- Module-level docs render prominently in modern doc systems (Sphinx, mkdocs)

No class-level discoverability loss in practice.

### Alternative considered: rename to `Trace.from_path` / `Trace.from_file`

Pythonic alt-constructor pattern (parallels `dict.fromkeys`, `datetime.fromisoformat`). Eliminates footgun via naming. Rejected in favor of simpler removal — `tl.load` covers it.

### Cascade scope

- Remove `Trace.load` classmethod (`model_log.py:2280-2295`)
- Update any docs/examples that reference `Trace.load(...)` → `tl.load(...)`
- Tests: spot-check for `Trace.load` callers
- Backward-compat: optional deprecated alias for one cycle emitting warning then removed

---

## Conditional — amendment to ConditionalArm storage (op_labels lists become @property)

Amends the locked Conditional design at delta line 282-317. The naming stays (`evaluation_*` / `execution_*` pair); the storage model changes to eliminate duplication.

### Issue

The locked design stores `evaluation_op_labels: list[str]` AND `execution_op_labels: list[str]` on ConditionalArm. AND each Op stores `in_conditionals: list[ConditionalRoleRef]` (the reverse mapping). **Data duplicated.**

`execution_op_labels` can be huge (entire transformer block body, etc.) — 30+ KB per arm for thousand-op bodies. Per deeply conditional model, multiplies fast.

### Fix: derive `_op_labels` from Op-side `in_conditionals` as `@property`

Op-side is the single source of truth. Arm derives on access.

```python
@dataclass
class ConditionalArm:
    kind: Literal["then", "elif", "else"]

    # Evaluation side (STORED — bounded)
    terminal_bool_op_label: str | None
    bool_value_at_run: bool | None
    condition_evaluated: bool
    evaluation_entry_edge: tuple[str, str] | None

    # Execution side (STORED — bounded)
    fired: bool
    execution_entry_edge: tuple[str, str] | None

    # DERIVED via @property — no storage
    @property
    def evaluation_op_labels(self) -> list[str]:
        return [
            op.label for op in self.trace.ops
            if any(
                ref.conditional_id == self.conditional_id
                and ref.arm_index == self.arm_index
                and ref.role == "evaluation"
                for ref in op.in_conditionals
            )
        ]

    @property
    def execution_op_labels(self) -> list[str]:
        return [
            op.label for op in self.trace.ops
            if any(
                ref.conditional_id == self.conditional_id
                and ref.arm_index == self.arm_index
                and ref.role == "body"
                for ref in op.in_conditionals
            )
        ]
```

### Why this works

- **Storage scales as O(N_arms × constants)**, not O(N_arms × N_body_ops). Big win for conditionally-large bodies.
- **No data duplication** — Op's `in_conditionals` is the single source of truth
- **Always fresh** — no stale-data risk if anything mutates
- **Visualization cost is the same** — rendering iterates ops anyway; the @property call is one scan per arm

### Naming kept as-is

`evaluation_*` / `execution_*` pair confirmed. Reasons:
- "Evaluation" used elsewhere in conditional vocab (`condition_evaluated`)
- Symmetric noun-form active pair reads cleanly
- Asymmetric `evaluation / body` would read worse
- Established within the locked design

### Caching note

If access pattern shows hot-path scans, add `@cached_property` on Trace (clear on graph mutation) or maintain reverse-index map at `Trace`-level for O(1) lookups. Don't over-engineer until profiling shows need.

### Cascade scope

- Update locked design at delta line 282-317 — move two list fields from stored to `@property`
- Implementation: Op-side `in_conditionals` is the canonical store; arm-side accessors derive
- Storage in serialization: dropped from `.tlspec` (FieldPolicy.DROP for these two), reconstructed on access

---

## Conditional — confirmed lock: `conditional_id` stays as `f"cond_{leading_terminal_bool_op_label}"`

Confirms the locked design at delta line 240-245. Considered alternatives (source-location IDs, sequential index, subgraph hash); none beat the current approach for TorchLens's domain.

### Why this is the right ID scheme

1. **Always present.** Every `if` chain has at least an `if` arm with a computed boolean. `if True:` is compile-time-optimized and never reaches captured graphs.
2. **Trace-unique.** Op labels are deterministic and unique within a trace (driven by `trace_index`); two different conditionals produce two different leading-bool labels.
3. **Loop-iteration-aware.** Uses the LAYER label (pass-stripped) — one conditional_id per source-code if-statement, with multiple firings as pass-indexed records. Matches user mental model.
4. **Stable across runs** of the same model + same-input-shape.
5. **Survives save/load.** Pure string derived from a stable label; no Python `id()` dependency.
6. **Universal across runtime contexts** — works for notebooks, REPL, dynamic code where source paths don't resolve cleanly.

### Why source-location IDs were rejected

Considered `f"cond_{source_file}_{source_line}"`:
- **Notebook source paths are unreliable** — cell-based, ephemeral, sometimes synthetic
- REPL/exec-based code has no canonical source location
- Lambda/dynamically-defined conditions break the source-file framing
- The leading-bool approach handles all these cases uniformly

### Acknowledged edge case (not blocking)

Cross-trace comparison with DIFFERENT input shapes: traces of input `(1, 512)` and `(1, 1024)` may produce different op-label indices, so conditional_ids of the SAME source-code conditional could differ across these traces.

This affects:
- Bundle alignment across dynamic-shape inputs (uncommon)
- Cross-input intervention spec portability (uncommon)

**Mitigation if it becomes load-bearing**: Conditional already stores `source_file` and `source_line` (per delta line 248-249). Bundle alignment code can fall back to source-location matching when conditional_id mismatch occurs. Don't change the canonical ID scheme; layer fallback logic in Bundle if needed.

### Lock

Current design at delta line 240-245 stands. No changes. TODO if Bundle dynamic-shape work demands it: add source-location secondary alignment key.

---

## Op — index family simplification: revert `trace_index` to `overall_index`, drop `compute_index`

Supersedes the index-family lock at delta line 515-539. Two changes:

### 1. Revert `trace_index` → `overall_index`

| Was (line-519 lock) | New |
|---|---|
| `Op.trace_index` | `Op.overall_index` |
| `Layer.trace_index` | `Layer.overall_index` |
| `GradFn.trace_index` | `GradFn.overall_index` |

`trace_index` reads ambiguously — could be misread as "index OF a trace" rather than "this op's index WITHIN the trace." `overall_index` honestly means "the overall position in the trace" (no FK-ish misread).

### 2. Drop `compute_index` entirely

| Was (line-519 lock) | Action |
|---|---|
| `Op.compute_index` (was `op_index`) | **REMOVED** — redundant with `overall_index` after semantic alignment below |

`compute_index` and `overall_index` (current code's `trace_index`) carried nearly identical data, differing only in their treatment of output ops. Both are invisible in output labels. Maintaining two fields with this narrow divergence is a footgun.

### 3. `overall_index` adopts Option β semantic (compute-only counter)

The unified `overall_index` follows what was previously `compute_index`'s semantic:

| Op kind | `overall_index` value |
|---|---|
| input | 0 |
| buffer | 0 |
| **output** | **0** (or sentinel; label uses only `type_index` anyway) |
| compute | 1, 2, 3, ... (sequential, **skip outputs**) |

Counter increments ONLY on compute ops. **Compute-op labels are guaranteed-sequential** — no "gaps" when outputs appear mid-trace.

### Concrete behavior comparison

For trace `input -> conv1 -> output_mid -> conv2 -> output_final`:

| Op | OLD (line-519 lock; `trace_index` includes outputs in counter) | NEW (Option β; counter skips outputs) |
|---|---|---|
| input | 0 | 0 |
| conv1 | 1 → `conv2d_1_1` | 1 → `conv2d_1_1` |
| output_mid | 2 → `output_1` | 0 → `output_1` |
| conv2 | 3 → `conv2d_2_3` (gap!) | 2 → `conv2d_2_2` (sequential) |
| output_final | 4 → `output_2` | 0 → `output_2` |

Mid-output gaps in compute-op labels are eliminated.

### Why this is a clear win

1. **One less ordinal field** — reduces the index footgun (4 stored → 3 stored fields on Op)
2. **Honest semantic** — `overall_index` actually IS the dense compute-op ordinal (matches what the previous lock at line 522 claimed but the code didn't deliver)
3. **No mid-output label gaps** — compute-op labels stay sequential regardless of where outputs appear
4. **Drops the `compute_index = None then set to num_ops` magic** in `labeling.py:149,246` — clean removal
5. **`num_ops` becomes unambiguous** — count of compute ops only (= max overall_index value)

### Final list of indices on Op (after this lock)

**Stored fields (3 ordinals):**

| Field | Value | Where used |
|---|---|---|
| `capture_index` (was `creation_index`) | 1-based raw capture-time counter; may have gaps after orphan removal | `_raw` labels; debug/internal |
| `type_index` | 1-based; Nth op of this type | Middle number in labels (`conv2d_1_5` → `1`) |
| `overall_index` (was `trace_index`, before that `overall_index`) | 0 for input/buffer/output; sequential 1+ for compute ops (skips outputs) | Last number in compute-op labels (`conv2d_1_5` → `5`); visualization rendering |
| `pass_index` | 1-based; iteration within parent Layer's equivalence class | After `:` in pass-qualified labels (`conv2d_1_5:2` → `2`) |

(Plus `pass_index` is the per-Layer recurrence count, separate concept; not strictly part of the ordinal family.)

**Removed:**
- `compute_index` (was `op_index`) — merged into `overall_index`

**NOT stored (transient):**

| Field | Value | Where used |
|---|---|---|
| `tensor_index` | 0-based position in `layer_list` (passed as `_add_lookup_keys_for_layer_entry` parameter) | Registered as integer lookup key so `trace[N]` works |

### Cascade scope

- Revert: `Op.trace_index` / `Layer.trace_index` / `GradFn.trace_index` → `overall_index` (all references)
- Remove: `Op.compute_index` field everywhere (`op_log.py:684`, `model_log.py:3611,3668`, `capture/projections.py:168,365`, `ir/predicate.py:32`, `ir/events.py:160`, `visualization/fastlog_preview.py:125`, `validation/invariants.py:635,639,1851`)
- Update `postprocess/labeling.py:70-93,145-153,246` — single counter for non-input/buffer/output ops; drop output's `compute_index = num_ops` magic
- Update `op_log.py:1926` repr: use `overall_index` instead of `compute_index` (semantically same value for compute ops after unification)
- `num_ops` semantic clarified: count of compute ops only (max `overall_index`)
- Backward-compat: `compute_index` as deprecated `@property` aliasing `overall_index` for one cycle
- Update earlier line-515-539 delta lock to reflect supersede

---

## Op — index family finalization: `overall_index` confirmed; add `ordinal_index` (0-based)

Finalizes the index-family discussion. Supersedes both the earlier line-519 lock AND the just-locked "rename to compute_index" proposal.

### Final field set on Op (5 stored, all 1-based except ordinal_index)

| Field | Base | Value | Visible in |
|---|---|---|---|
| `capture_index` (was `creation_index`) | 1-based | raw capture-time counter; may have gaps after orphan removal | `_raw` labels; debug |
| `type_index` | 1-based | Nth op of this type | Middle of labels (`conv2d_**1**_5`) |
| `overall_index` (was `op_index`/`compute_index`/`trace_index`/`overall_index` historically) | 1-based | 0 for inputs/buffers/outputs; sequential 1+ for compute ops only (skips outputs) | **Last number** in compute-op labels (`conv2d_1_**5**`) |
| `pass_index` | 1-based | iteration within parent Layer's equivalence class | After `:` in pass-qualified labels (`conv2d_1_5:**2**`) |
| **`ordinal_index`** (NEW; replaces transient `tensor_index`) | **0-based** | unique position in `trace.ops` | for `trace[N]` integer indexing |

### Final decisions

**1. `overall_index` is the chosen name** (not `compute_index`).

- More transparent to users — "overall" reads as "the overall position you see in labels"
- Maps cleanly to "the last number in `conv2d_1_5`"
- Boundary-op 0 value (inputs/buffers/outputs) is acceptable; documented in glossary
- New users grok it immediately; experts can read the glossary for precise semantic

**2. `compute_index` rejected as field name** despite being technically more honest about semantic. User-friendliness wins over strict-precision in naming when both names describe the same data.

**3. `overall_index` adopts Option β semantic** (skips outputs in counter; compute-op labels are guaranteed-sequential — no mid-output gaps).

**4. NEW: `ordinal_index` exposes the 0-based list position** as a stored field (replacing transient `tensor_index` parameter). Use case: round-trip `trace[op.ordinal_index] is op`. Critical for users who want to identify "what integer indexes this op via `trace[N]`."

**5. Drop `compute_index` and `trace_index` references entirely** — they were intermediate names in this walkthrough; final canonical names are `overall_index` (1-based label number) and `ordinal_index` (0-based lookup).

### Naming rationale recap

| Considered | Verdict |
|---|---|
| `trace_index` | rejected — could be misread as "index OF a trace" |
| `compute_index` | rejected — technically honest but less user-friendly; "compute" sounds narrower than the user's mental model for "the position in labels" |
| `step_index` | rejected — new vocab; doesn't clearly bridge |
| `overall_index` | CHOSEN — transparent, intuitive, historical continuity |
| `list_index` (for new 0-based field) | rejected — "list" collides with output-tuple / output-list concepts |
| `linear_index` | considered — new vocab |
| `sequence_index` | rejected — "sequence" overloaded in ML (token sequences) |
| `ordinal_index` | CHOSEN — "ordinal" precisely means "indicating position in sequence"; clean; no collisions |

### Cascade scope

**Rename/remove:**
- `Op.trace_index` (current code, from line-519 lock) → `Op.overall_index` (revert; same name as pre-line-519)
- `Op.compute_index` (current code, from line-519 lock) → **REMOVED entirely**
- Update label construction in `postprocess/labeling.py:70-95`: single counter for compute ops only (skip outputs and inputs/buffers); drop `compute_index = num_ops` magic at `:149,246`
- Update `op_log.py:1926` repr: use `overall_index` instead of `compute_index`

**Add:**
- `Op.ordinal_index` (NEW, stored, 0-based) — populate during `_add_lookup_keys_for_layer_entry` (replaces transient `tensor_index` parameter, which becomes the value stored on Op)
- Same field on Layer, Module, Param, Buffer, GradFn for parity (= position in respective accessor)

**`num_ops` semantic:**
- Clarified as count of compute ops only (= max `overall_index`)

**Backward-compat:**
- `Op.compute_index` deprecated `@property` aliasing `overall_index` for one cycle, then removed
- `Op.trace_index` deprecated `@property` aliasing `overall_index` for one cycle, then removed

**Cross-class cascade (for `ordinal_index`):**
- `Layer.ordinal_index`, `Module.ordinal_index`, `Param.ordinal_index`, `Buffer.ordinal_index`, `GradFn.ordinal_index`
- Each is 0-based position in the respective Accessor
- Round-trip property: `trace.layers[lyr.ordinal_index] is lyr` etc.

### Glossary entries

- `overall_index`: "1-based position counter for compute ops; 0 for inputs/buffers/outputs. The last number in compute-op labels like `conv2d_1_5`. Use `is_compute_op` to filter to ops where this is meaningful."
- `ordinal_index`: "0-based position in the parent accessor's ordered list. Use `trace[op.ordinal_index]` for round-trip lookup. Pythonic — works with negative indexing too (`op.ordinal_index - len(trace.ops)`)."

---

## Op — `capture_index` → `raw_index` (supersedes line-519 lock)

| Was (line-519 lock) | New |
|---|---|
| `Op.capture_index` (was `creation_index`) | `Op.raw_index` |

### Why

1. **Pairs with `_raw_` prefix convention** established in TorchLens (`_raw_label`, `_raw_to_final_layer_labels`, `_raw_layer_dict`)
2. **Honest about the pre-postprocess semantic** — the field's defining trait is "the order BEFORE postprocessing renumbered things"; gaps after orphan removal are a "raw" artifact
3. **Field is mostly internal/debug** — naming optimized for maintainers / power users who know the postprocess pipeline
4. **Cleans up raw label format** — current `{type}_{type_index}_{capture_index}_raw` has redundant `_raw`; with `raw_index` the suffix can be dropped (the index name carries the marker)

### Updated final Op index family (5 stored)

| Field | Base | Value | Visible in |
|---|---|---|---|
| **`raw_index`** (was `capture_index`/`creation_index`) | 1-based | raw capture order; gaps after orphan removal | Raw labels (debug); internal lookup |
| `type_index` | 1-based | Nth of this op type | Middle of labels: `conv2d_**1**_5` |
| `overall_index` | 1-based | 0 for inputs/buffers/outputs; sequential 1+ for compute ops only | Last number in compute-op labels: `conv2d_1_**5**` |
| `pass_index` | 1-based | iteration within parent Layer's recurrence | After `:` in pass-qualified labels: `conv2d_1_5:**2**` |
| `ordinal_index` | **0-based** | unique position in `trace.ops` | for `trace[N]` integer indexing |

### Raw label format

Option (cleaner): `{type}_{type_index}_{raw_index}` — drops redundant `_raw` suffix
Option (explicit): `{type}_{type_index}_{raw_index}_raw` — keeps `_raw` suffix as deliberate marker

Either works; defer to implementation taste during the rename sprint. Lean toward dropping the suffix since `raw_index` self-identifies.

### Cascade scope

- `Op.capture_index` → `Op.raw_index` (all references)
- `Layer.capture_index` → `Layer.raw_index` (if propagated)
- `_raw_label` construction in `capture/output_tensors.py:962` (per delta line 533): use `raw_index`
- All call sites in `op_log.py`, `model_log.py`, `postprocess/`, validation, tests
- FIELD_ORDER constants
- FieldPolicy maps
- Update line-519 lock to reflect supersede
- Glossary entry: "`raw_index`: 1-based position in the raw capture log, before postprocessing. May have gaps after orphan removal (orphan op records get removed; their `raw_index` values aren't reassigned to later ops). For most users, prefer `overall_index` (label number) or `ordinal_index` (for `trace[N]` lookup)."

### Backward-compat

- `Op.capture_index` deprecated `@property` aliasing `raw_index` for one cycle, then removed

---

## Op — `overall_index` → `step_index` (supersedes previous lock)

| Was (just-locked) | New |
|---|---|
| `Op.overall_index` | `Op.step_index` |

### Why supersede

`overall_index` was the chosen name for the compute-only label counter, but became misleading once `ordinal_index` was introduced. `ordinal_index` is the genuinely "overall" position (0-based, covers all ops uniquely), making `overall_index` a misnomer (it actually SKIPS boundary ops, returning 0 for inputs/buffers/outputs).

### Why `step_index`

1. **Honest about semantic** — boundary ops aren't computation steps; their `step_index = 0` reads naturally as "not a step in the forward computation"
2. **Pairs with `type_index`** — `type_index + step_index` reads as a clean pair in labels: "Op type X, step Y." Label `conv2d_1_5` = "1st conv2d type, 5th computation step."
3. **Distinct from `pass_index`** — step = sequence position in forward flow; pass = recurrence iteration. Different conceptual axes; no overlap.
4. **No collision with `ordinal_index`** — ordinal = position in list (0-based, unique); step = position in compute flow (1-based, compute-only). Both honest about their scope.
5. **Matches user mental model** — "step in the forward pass" is natural ML/CS vocabulary

### Mild caveat acknowledged

`step_index` is new vocabulary. Users will need to learn the meaning ("step in forward computation; boundary ops get 0"). Glossary entry handles this.

Mild collision with ML "training step" / RL "step" — but context (an Op-level field) resolves.

### Final Op index family (post all locks)

| Field | Base | Value | Visible in |
|---|---|---|---|
| `raw_index` | 1-based | raw capture order; may have gaps after orphan removal | `_raw` labels; debug |
| `type_index` | 1-based | Nth of this op type | Middle of labels: `conv2d_**1**_5` |
| **`step_index`** | 1-based | 0 for boundaries; sequential 1+ for compute ops only | **Last number in compute-op labels: `conv2d_1_**5**`** |
| `pass_index` | 1-based | iteration within parent Layer's recurrence | After `:` in pass-qualified labels: `conv2d_1_5:**2**` |
| `ordinal_index` | **0-based** | unique position in `trace.ops` (covers ALL ops uniquely) | for `trace[N]` integer indexing |

### Cascade scope

- `Op.overall_index` → `Op.step_index` (all references)
- Same field on Layer, GradFn analogs (per delta line 532-535)
- Label construction in `postprocess/labeling.py:91-93`: use `step_index`
- Glossary entry: "`step_index`: 1-based step number in the forward computation. Compute ops get sequential values (1, 2, 3, ...); inputs, buffers, and outputs get 0 (they're not computation steps). Appears as the last number in compute-op labels like `conv2d_1_5`."
- Update prior delta locks (line-519 etc.) to reflect final name
- Backward-compat: `overall_index`, `compute_index`, `trace_index` all deprecated `@property` aliases of `step_index` for one cycle, then removed

### Glossary cluster

- `raw_index`: raw capture position (pre-postprocess)
- `type_index`: Nth of this op type
- `step_index`: Nth computation step (compute-only counter; boundary ops get 0)
- `pass_index`: iteration within parent Layer's recurrence
- `ordinal_index`: 0-based position in `trace.ops` (unique across all ops, including boundaries)

Five indices on Op, each with clearly distinct semantic and use case. No more redundancy.

---

## Op — activation / gradient vocabulary: mixed convention (objects short, properties qualified)

Locks the activation/gradient naming pattern across Op (and parallel classes). Resolves the mild tension between PyTorch's short-form tensor idioms (`tensor.grad`) and TorchLens's compound property cluster (`activation_memory`, `gradient_memory`).

### The rule: tensor-attribute style is short; property/concept style is full

| Category | Form | Examples |
|---|---|---|
| **Tensor objects** | short | `Op.out` (activation tensor), `Op.grad` (gradient tensor) |
| **Capture config flags (verb form)** | full (matched within cluster) | `save_raw_activations`, `save_gradients` |
| **Compound property qualifiers** | full | `activation_memory`, `gradient_memory`, `activation_transform`, `gradient_transform` |
| **Predicates** | full (qualifier style) | `has_saved_activation`, `has_saved_gradient` |

### Per-field changes

**Tensor objects (KEEP short):**
- `Op.out` (already; current code is bare `out`)
- `Op.grad` (already; current code is bare `grad`)
- Same on Layer via delegation

**Memory fields (RENAME to qualified):**
- `Op.memory` → `Op.activation_memory`
- `Op.transformed_out_memory` → `Op.transformed_activation_memory`
- Same on Layer
- Pairs with locked `Trace.total_activation_memory` / `saved_activation_memory`

**Gradient property fields (full word):**
- `Op.gradient_memory` (NEW or rename if existing)
- `Op.transformed_gradient_memory` (NEW)
- `Op.gradient_transform` (per delta line 134-139 — pairs with `activation_transform`)

**Predicates (FULL word):**
- `has_saved_activation` (locked elsewhere; full word)
- `has_saved_gradient` (lock here; full word for symmetry)

**Capture config flags (FULL word, aligned cluster):**
- `save_raw_activations` (locked, plural)
- `save_gradients` (RENAME from `save_grads`; full + plural for parity)

### Shape/dtype on Op — keep bare

| Field | Action | Reasoning |
|---|---|---|
| `Op.shape` | KEEP bare | gradient has same shape as activation; bare is unambiguous |
| `Op.dtype` | KEEP bare | gradient has same dtype as activation (rare AMP exceptions documented per docstring); bare fine |
| `Op.transformed_out_shape` | rename → `Op.transformed_activation_shape` | qualified per the transformed cluster |
| `Op.transformed_out_dtype` | rename → `Op.transformed_activation_dtype` | qualified |

### Why this mixed convention works

1. **Access frequency justifies brevity for tensors** — `op.out`, `op.grad` are typed constantly; short wins.
2. **PyTorch precedent for tensor short-form** — `tensor.grad`, `.requires_grad`, `zero_grad()` — `grad` is its own term in PyTorch.
3. **Compound property names need disambiguation** — `op.activation_memory` vs `op.gradient_memory` reads as a clear pair; bare `op.memory` would be ambiguous.
4. **The rule is itself a single principle** — not two arbitrary conventions: "object access = short; property/concept = full."
5. **Symmetric predicates pair cleanly** — `has_saved_activation` / `has_saved_gradient` are visually parallel.

### Mild acknowledged asymmetry

- `out` vs `activation_*` — the tensor name (`out`) doesn't share the property stem (`activation_`)
- `grad` vs `gradient_*` — the tensor name (`grad`) IS a prefix of the property stem (`gradient_`)

So `grad`/`gradient_*` reads as "short form of same word"; `out`/`activation_*` are different words entirely. Documented: "Op.out is the activation tensor; activation-related properties use `activation_*` qualifier."

### Cascade scope

**Renames in current code:**
- `Op.memory` → `Op.activation_memory` (`op_log.py:724,1513`)
- `Op.transformed_out_memory` → `Op.transformed_activation_memory` (`op_log.py:725,1534,1557`)
- `Op.transformed_out_shape` → `Op.transformed_activation_shape`
- `Op.transformed_out_dtype` → `Op.transformed_activation_dtype`
- Trace flag: `save_grads` → `save_gradients` (parity with `save_raw_activations`)

**Additions:**
- `Op.has_saved_gradient` predicate (parallel to `has_saved_activation`)
- `Op.gradient_memory` (if not present)
- `Op.transformed_gradient_memory`

**Deprecation map updates** (reverse direction from current `op_log.py:1359-1374`):
- `"out": "out"` — KEEP
- `"transformed_out": "transformed_out"` — KEEP
- `"memory": "activation_memory"` — ADD deprecation
- `"transformed_out_memory": "transformed_activation_memory"` — ADD deprecation
- `"save_grads": "save_gradients"` — ADD deprecation
- (Remove the existing `"activation_X": "X"` entries since "activation_X" is the new canonical)

**Backward-compat aliases for one cycle:**
- `Op.memory` as deprecated `@property` aliasing `Op.activation_memory`
- `save_grads=` kwarg accepted with deprecation warning, then removed
- `transformed_out_memory` likewise

---

## Op — multi-output vocabulary redesign

Replaces the current three-field cluster (`is_part_of_iterable_output`, `multi_output_index`, `multi_output_role`) with a cleaner exhaustive design.

### Current cluster (to be replaced)

```python
is_part_of_iterable_output: bool
multi_output_index: int | None       # 0-based positional
multi_output_role: str | None        # structural identifier (dict key, field name, or str(index) fallback)
```

Issues:
- `is_part_of_iterable_output` is verbose; clear redundancy with multi_output_index
- `multi_output_role` name misleading — "role" suggests semantic role, but field carries structural identifier
- No explicit container-type info — user has to guess if it's a tuple, dict, namedtuple, etc.

### New cluster

```python
is_multi_output: bool             # @property (derived)
multi_output_type: type | None    # the container's Python class
multi_output_index: int | None    # 0-based positional (for tuple/list/namedtuple/dataclass/ordered-dict)
multi_output_name: str | None     # semantic name (for dict key, namedtuple field, dataclass attribute)
```

### By container type

| Container | `multi_output_type` | `multi_output_index` | `multi_output_name` |
|---|---|---|---|
| `(a, b, c)` tuple | `tuple` | `0`, `1`, `2` | `None` |
| `[a, b, c]` list | `list` | `0`, `1`, `2` | `None` |
| `MyTuple(values=v, indices=i)` namedtuple | `MyTuple` (class) | `0`, `1` | `"values"`, `"indices"` |
| `{"hidden": h, "attn": a}` dict | `dict` | `0`, `1` (insertion-order) | `"hidden"`, `"attn"` |
| `MyDC(hidden=h)` dataclass | `MyDC` | `0` (field-order) | `"hidden"` |
| single tensor | `None` | `None` | `None` |

### Why this design

1. **Type-clean fields** — `multi_output_index` is always int (when not None); `multi_output_name` is always str (when not None). No union-type footgun.
2. **Lossless info preservation** — namedtuple/dataclass keep BOTH int and str access paths
3. **Explicit container type** — `multi_output_type` removes guessing; user knows the structural shape
4. **`is_multi_output` predicate stays simple** — `@property`: True iff `multi_output_type is not None`
5. **Direct access patterns work** — `outputs[op.multi_output_index]` for positional; `outputs[op.multi_output_name]` for named
6. **No nested-container forward-compat** — native PyTorch returns are flat; deferred to YAGNI

### Native PyTorch returns covered

- Single tensor: most `torch.*` functions → all `None`
- Flat tuples: `torch.unbind`, `torch.split`, `torch.chunk`, `torch.tensor_split` → `multi_output_type=tuple`, `multi_output_index=N`, `multi_output_name=None`
- Namedtuples: `torch.max/min/sort/topk/median/mode/kthvalue/cummax/cummin/svd/lu_unpack/...` → `multi_output_type=torch.return_types.X`, both index AND name populated

### Naming decisions made

- `multi_output_type` over `multi_output_class` — matches Python's `type()` builtin idiom; mild TorchLens overload (`Op.type`, `type_index`) tolerated since context distinguishes
- `multi_output_name` over `multi_output_key` — more general (covers dict keys + namedtuple fields + dataclass attrs); "name" doesn't imply lookup mechanism the way "key" does; pairs naturally with `multi_output_index` as a noun pair; matches TorchLens `_name` cluster
- Kept `is_multi_output` as `@property` (derived) — convenience predicate, no storage cost

### Cascade scope

- `Op.multi_output_index` — KEEP (already exists at `op_log.py:496`)
- `Op.multi_output_role` → `Op.multi_output_name` (rename)
- `Op.is_part_of_iterable_output` → `Op.is_multi_output` (rename, become `@property`)
- NEW: `Op.multi_output_type` (stored field)
- Same fields on Layer via delegation (`layer_log.py:161-163,259-261`)
- Function: `multi_output_role_from_path` → `multi_output_name_from_path`
- Validation reference at `validation/core.py:641` (`recomputed_output[layer.multi_output_index]`) — still works
- FIELD_ORDER, FieldPolicy maps
- Backward-compat: deprecated `@property` aliases for `multi_output_role`, `is_part_of_iterable_output`

### Glossary entries

- `is_multi_output`: True when this Op came from a multi-output container (tuple, list, dict, namedtuple, dataclass, etc.).
- `multi_output_type`: The Python class of the container that produced this Op (`tuple`, `list`, `dict`, namedtuple class, dataclass, etc.). `None` if not from a multi-output.
- `multi_output_index`: 0-based positional access for ordered containers (tuple/list/namedtuple/dataclass; insertion-order for dicts). `None` for non-multi-output.
- `multi_output_name`: Semantic name (dict key, namedtuple field, dataclass attribute). `None` for plain tuple/list outputs, or when not from a multi-output.

---

## Op — add `has_saved_gradient` predicate (paralleling `has_saved_activation`)

Currently the code has only `Op.has_saved_outs` (for activations); no analog exists for gradient save state. Fills the gap.

### Current state and rename

| Was | New |
|---|---|
| `Op.has_saved_outs` (current code) | `Op.has_saved_activation` (per locked vocab cluster) |
| (missing) | `Op.has_saved_gradient` (NEW — stored bool) |

### Why both predicates are needed

Activation save and gradient save are independent capture-time choices:
- `save_raw_activations=True` + `save_gradients=False` → activations saved, grads not
- `save_raw_activations=False` + `save_gradients=True` → grads saved, activations not
- `save_raw_activations=True` + `save_gradients=True` → both saved

Users need predicates for EACH save type to filter/iterate cleanly.

### Pairs with the locked saved_* Accessor family

- `Trace.saved_ops` filters Ops on `has_saved_activation`
- `Trace.saved_grad_ops` filters Ops on `has_saved_gradient`
- `Trace.saved_layers` aggregates from `has_saved_activation`
- `Trace.saved_grad_layers` aggregates from `has_saved_gradient`

The predicates are the building blocks for the saved-only accessors. Both must exist.

### Implementation

- `Op.has_saved_activation: bool` — set True when activation tensor saved during capture (replacing existing `has_saved_outs` flow)
- `Op.has_saved_gradient: bool` — set True when gradient tensor saved during backward hook
- Defaults: both False
- Layer-level: derive from constituent Ops (`any(op.has_saved_activation for op in layer.ops)`)

### Cascade scope

- Rename: `Op.has_saved_outs` → `Op.has_saved_activation` (`op_log.py:350,439,708,1568`, `ir/events.py:47`, etc.)
- Add: `Op.has_saved_gradient` (new stored bool field)
- Same on Layer via delegation (`layer_log.py:524,532`)
- Trace count: `Trace.num_saved_ops` (existing); NEW companion `Trace.num_saved_grad_ops` (already locked in the saved_* accessor family)
- FIELD_ORDER, FieldPolicy maps
- Backward-compat: `has_saved_outs` deprecated `@property` aliasing `has_saved_activation`
- Glossary entries for both predicates

---

## TODO (post-rename sprint) — move Trace-level capture flags into `Trace.capture_config` namespace

Resolves the verb-flag-as-attribute ambiguity (`trace.save_gradients`, `trace.detach_saved_activations` read like methods, not flags).

### The fix

Move all capture-time configuration booleans/filters from flat Trace attributes into a `Trace.capture_config` namespace (e.g., a dataclass).

**Before:**
```python
trace = tl.trace(model, x, save_gradients=True, detach_saved_activations=False, ...)
trace.save_gradients            # ambiguous — method or attribute?
trace.detach_saved_activations  # same ambiguity
```

**After:**
```python
trace = tl.trace(model, x, capture_config=tl.CaptureConfig(
    save_gradients=True,
    detach_saved_activations=False,
    ...
))
trace.capture_config.save_gradients          # clearly attribute (under .config namespace)
trace.capture_config.detach_saved_activations
```

### Scope

Trace-level capture flags only. Per-Op state fields stay flat on each Op — `op.has_saved_activation`, `op.has_saved_gradient`, etc. are observations (declarative predicate state), not user intent. No `op.capture_config` namespace needed.

### Fields to migrate (probably)

- `save_raw_activations`
- `save_gradients`
- `save_arg_values`
- `save_arg_templates`
- `detach_saved_activations`
- `backward_ready` (formerly `train_mode`)
- `intervention_ready`
- `emit_nvtx`
- `raise_on_nan`
- `layers_to_save`, `grads_to_save` (filters)
- Any other capture-time options under `options.py` `CaptureOptions`

### Benefits

1. **Verb-flag ambiguity dissolves** — namespace clarifies "this is configuration, not method"
2. **Discoverability** — all capture options grouped (better IDE auto-complete, better docs)
3. **Aligned with architectural endpoint** — single configurable capture substrate; the config object IS the substrate's interface
4. **Trace constructor cleaner** — one `capture_config=` arg instead of many flat kwargs
5. **Backwards-compat path is clean** — flat kwargs accepted with deprecation warning, then migrated

### Why this is a TODO (post-rename sprint)

This is a substantive structural change, not a rename. Doing it during the rename sprint would balloon scope. Better as a focused follow-up that can:
- Design `CaptureConfig` as a proper dataclass (with sub-groups if needed)
- Plan the migration path carefully (deprecation cycle for flat kwargs)
- Update all docs / examples / tutorials
- Coordinate with the architectural endpoint work (capture substrate unification)

### Decision deferred

- Whether `capture_config` becomes its own dataclass or stays as nested attribute access on Trace
- Whether to use sub-groups (e.g., `capture_config.activations`, `capture_config.gradients`)
- Backwards-compat strategy (transitional flat-kwarg acceptance period)

---

## Annotations — bare for primary; qualified for secondary entity

Locks the convention: across all log classes, the primary "annotations" field is bare (`annotations`); secondary-entity annotation fields use a qualifier prefix.

### Per-class

| Class | Primary annotations | Secondary entity's annotations |
|---|---|---|
| Trace | `annotations` (bare; was `trace_annotations`) | `input_annotations` (qualified — about the input) |
| Op | `annotations` (bare, current) | (none) |
| Layer | `annotations` (bare, current) | (none) |
| Module | `annotations` (bare, if added) | (none currently) |
| ModuleCall | `annotations` (bare, if added) | (none currently) |
| Param | `annotations` (bare, if added) | (none currently) |
| Buffer | `annotations` (bare, if added) | (none currently) |
| GradFn | `annotations` (bare, if added) | (none currently) |

### Why asymmetric on Trace

`Trace.annotations` (bare) + `Trace.input_annotations` (qualified) — the asymmetry IS the information:
- Bare "annotations" = the primary subject's annotations (no qualifier needed; one obvious meaning)
- Qualified `input_annotations` = annotations about a SECONDARY entity (the input)

If both were qualified (`trace_annotations` + `input_annotations`), the user would have to scan names to know which is the main one. With asymmetry, it's immediate.

### Rule recap

Bare `annotations` if there's just one annotation field for the class; qualified `<entity>_annotations` only for secondary entities.

---

## Op — `container_path` confirmed lock (coexists with `multi_output_*` cluster)

`Op.container_path` is the **structural traversal path** from the top-level return value down to this op's tensor. Stored as a tuple of typed components (`TupleIndex`, `DictKey`, `NamedField`, `DataclassField`, `HFKey`).

### Examples

| Module return | Tensor location | `container_path` |
|---|---|---|
| `tensor` (single) | the tensor | `()` |
| `(a, b)` tuple | `a` | `(TupleIndex(0),)` |
| `{"hidden": h, "attn": a}` dict | `h` | `(DictKey("hidden"),)` |
| `{"outputs": (a, b)}` nested | `a` | `(DictKey("outputs"), TupleIndex(0))` |
| `BaseModelOutput(last_hidden=h, ...)` dataclass | `h` | `(DataclassField("last_hidden"),)` |

### Coexistence with `multi_output_*` cluster

`container_path` is the source of truth for nested structural paths. The just-locked `multi_output_*` cluster (`multi_output_type`, `multi_output_index`, `multi_output_name`, `is_multi_output`) provides FLAT one-level convenience views derived from `container_path`.

| User question | Use field |
|---|---|
| Is this from multi-output? | `is_multi_output` (predicate) |
| What container type? | `multi_output_type` |
| Quick top-level positional access | `multi_output_index` (int) |
| Quick top-level named access | `multi_output_name` (str) |
| Full nested structural path | `container_path` (tuple of components) |

Coexistence is honest — different abstraction levels for different access patterns.

### Naming kept

`container_path` reads cleanly: "the path through containers to reach this op's tensor." No rename needed.

---

## Trace / Layer / Module — add trainable/frozen tensor-count `@property` symmetry

Fills the asymmetry between parameter VALUE counts (current) and parameter TENSOR counts (incomplete).

### Current state

| Class | Value counts | Tensor counts |
|---|---|---|
| Trace | `num_params`, `num_params_trainable`, `num_params_frozen` ✓ | `num_param_tensors` (total only) — missing trainable/frozen split |
| Layer | `num_params`, `num_params_trainable`, `num_params_frozen` ✓ | `num_param_tensors` (`@property`, total only) — missing split |
| Module | `num_params`, `num_params_trainable`, `num_params_frozen` ✓ | **all three missing** |

### Symmetry lock — add tensor-count split as `@property` derivations

For each class that has value counts, add the parallel tensor counts as `@property` (no storage cost; derived by iterating params with `requires_grad` filter).

| Class | Add (`@property`) |
|---|---|
| Trace | `num_param_tensors_trainable`, `num_param_tensors_frozen` |
| Layer | `num_param_tensors_trainable`, `num_param_tensors_frozen` |
| Module | `num_param_tensors`, `num_param_tensors_trainable`, `num_param_tensors_frozen` (all three) |

### Implementation sketch

```python
# On Trace
@property
def num_param_tensors_trainable(self) -> int:
    return sum(1 for p in self.params if p.requires_grad)

@property
def num_param_tensors_frozen(self) -> int:
    return sum(1 for p in self.params if not p.requires_grad)
```

Same on Layer / Module via their respective `params` iteration scopes.

### Why `@property` (not stored)

- Cheap to compute (one pass through params accessor)
- No storage cost
- Always fresh (no stale-data risk if anything ever changes)
- Pairs naturally with the existing `num_param_tensors` `@property` on Layer

### Full parameter-count surface (post-additions)

```python
# Value counts (existing — stored)
trace.num_params, trace.num_params_trainable, trace.num_params_frozen

# Tensor counts (full symmetry — @property)
trace.num_param_tensors, trace.num_param_tensors_trainable, trace.num_param_tensors_frozen
```

Same six-field surface on Trace, Layer, Module.

### Cascade scope

- Add 2 `@property` on Trace (`num_param_tensors_trainable`, `num_param_tensors_frozen`)
- Add 2 `@property` on Layer (same names)
- Add 3 `@property` on Module (`num_param_tensors` + `_trainable` + `_frozen`)
- Glossary entries — clarify "tensor count" vs "value count" distinction

### Glossary entries

- `num_params`: total count of parameter VALUES (scalars across all parameter tensors)
- `num_params_trainable`: count of trainable parameter VALUES
- `num_params_frozen`: count of frozen parameter VALUES
- `num_param_tensors`: total count of parameter TENSORS (e.g., 50 for ResNet50)
- `num_param_tensors_trainable`: count of trainable parameter TENSORS
- `num_param_tensors_frozen`: count of frozen parameter TENSORS

The two families distinguish: scalars (params) vs containers (param tensors).

---

## Op — add `is_orphan` (`@property`)

Convenience predicate for "this op is in a graph component disconnected from the main flow."

### Definition (strict)

```python
@property
def is_orphan(self) -> bool:
    return not self.has_input_ancestor and not self.is_output_descendent
```

True iff:
- No path back to a model input (`has_input_ancestor = False`)
- AND no path forward to a model output (`is_output_descendent = False`)

### Why strict (both conditions) over lenient (either condition)

- **Strict** captures the "disconnected cluster" mental model — a graph component fully detached from the main input→output flow. Matches "orphan" connotation in graph theory and programming.
- **Lenient** would include dead-end ops (input-reachable but not output-reachable) — different concept (computational dead-end vs. disconnected component). Users wanting dead-end semantics compose with `is_output_descendent` directly.

### Why `is_orphan` (not `is_isolated` or `is_disconnected`)

- `is_isolated` is misleading — implies single op without parents/children; doesn't capture disconnected clusters (where ops have intra-cluster parents but none trace back to model input)
- `is_disconnected` is honest but vague — "disconnected from what?"
- `is_orphan` is the standard graph-theory term; pairs with the existing `Trace.orphans` Accessor (`model_log.py:383`)

### Captures the "disconnected cluster" case correctly

For a cluster of N ops mutually connected but not touching the main graph:
- Each op has parents (within the cluster) → NOT "isolated"
- But none have `has_input_ancestor = True` (no path to MODEL INPUT)
- And none have `is_output_descendent = True` (no path to MODEL OUTPUT)
- ∴ each is `is_orphan = True`

This works because `has_input_ancestor` / `is_output_descendent` are scoped to MODEL boundary ops, not arbitrary parents/children.

### Cascade

- Add `Op.is_orphan` (`@property`, derived)
- Same on Layer via delegation (`@property` of constituent ops)
- Existing `Trace.orphans` Accessor (already at `model_log.py:383`) — verify it filters by this predicate; harmonize if not
- Glossary entry: "True iff this op has no input ancestor and no output descendent — disconnected from the main graph. Includes ops in mutually-connected clusters that don't reach the main flow. Use `Trace.orphans` Accessor for the filtered collection."

---

## Op / Layer — `submodule_*` cluster split: rename two, keep one

The current trio (`is_submodule_input`, `is_submodule_output`, `in_submodule`) uses `submodule_*` inconsistently with the rest of the `module_*` cluster. After analysis, the right answer is to split — rename the two whose semantic applies at any module level, keep the one whose semantic specifically requires non-root.

### Renames (apply to any module — including root)

| Was | New |
|---|---|
| `Op.is_submodule_input` (`@property` at `op_log.py:1017`) | `Op.is_module_input` |
| `Op.is_submodule_output` (`op_log.py:568,861`) | `Op.is_module_output` |

These describe POSITIONAL properties: the op IS at a module's input/output position. Module-input and module-output positions are meaningful at both root and nested levels. The general `module_*` cluster name fits.

### Keep (semantic requires non-root)

| Field | Status | Reason |
|---|---|---|
| `Layer.in_submodule` (`@property` at `layer_log.py:391-392`) | KEEP | `in_module` would always be True for any compute op (every op is in the root model module). The `submodule` qualifier is load-bearing — only "in a NESTED non-root module" is informative. |

### Glossary distinctions

| Predicate | Meaning | Module scope |
|---|---|---|
| `is_input` | synthetic INPUT sentinel (model boundary) | n/a — model level |
| `is_output` | synthetic OUTPUT sentinel (model boundary) | n/a — model level |
| `is_module_input` | op IS at a module's input position (first op inside its forward) | any module (root or nested) |
| `is_module_output` | op IS at a module's output position (final compute op of its forward) | any module (root or nested) |
| `in_submodule` | op was executed inside a NON-ROOT module's forward | submodule only |

### Why the asymmetry is honest

The naming difference (`module_*` vs `submodule_*`) reflects the genuine semantic distinction:
- "module_*" prefix: applies to ANY module (root or nested)
- "submodule_*" prefix: requires NON-ROOT specifically

Not arbitrary — informative. Users learning the cluster see the prefix and infer the scope.

### Cascade scope

- `Op.is_submodule_input` → `Op.is_module_input` (`op_log.py:1017`, FIELD_ORDER `model_log.py:3648`, FieldPolicy `model_log.py:3682`)
- `Op.is_submodule_output` → `Op.is_module_output` (`op_log.py:568,861`, `model_log.py:3649,3683`, `postprocess/finalization.py:501,503`, `postprocess/graph_traversal.py:116`)
- `Layer.in_submodule` — unchanged
- Backward-compat: `is_submodule_input` / `is_submodule_output` as deprecated `@property` aliases for one cycle, then removed
- Glossary entries for all three predicates with the scope distinction made explicit

---

## Op — module-entry cluster: `module_ops_entered` → `module_calls_entered`, `module_entry_argnames` → `module_entry_arg_keys`

Cleans up the module-entry cluster while accepting the TorchLens "Op fields default to output" convention.

### Renames

| Was | New | Reason |
|---|---|---|
| `Op.module_ops_entered` | `Op.module_calls_entered` | Stored values ARE module-CALL labels (e.g. `"encoder.block1:1"`), not arbitrary "ops." Aligns with locked `ModuleCall` class name and `module_calls` accessor. |
| `Op.module_entry_argnames` | `Op.module_entry_arg_keys` | Values are polymorphic (`int` for positional, `str` for kwargs); "argnames" misleads since int positions are also stored. "arg_keys" parallels `dict.keys()` for the int+str polymorphism, same convention as the locked `multi_output_index` polymorphic field. |

### Why not `output_into_*` explicit framing

Considered `output_into_module_calls` + `output_into_module_arg_keys`:
- More explicit about "this op's output enters these module calls"
- But verbose (28 chars for the arg-keys field)
- And the TorchLens convention already handles the ambiguity (see below)

### The TorchLens convention

> **Op fields about tensor flow refer to the op's OUTPUT tensor.** The op IS the producer of its output; what "flows through the graph" from this op is its output. Examples:
> - `op.shape`, `op.dtype`, `op.activation_memory` — properties of the output
> - `op.out` — the output tensor itself
> - `op.module_calls_entered` — modules where the output entered
> - `op.module_entry_arg_keys` — arg positions at those entry events
> - `op.output_of_module_calls` — modules this op IS the output of (final compute of that module)

This convention is documented in the Op glossary section. New users learn it once; subsequent fields are interpreted naturally.

### Pair reads cleanly with the convention

```python
op.module_calls_entered      # ["encoder.block1:1", "decoder.attn:1"]
op.module_entry_arg_keys     # {"encoder.block1:1": [0], "decoder.attn:1": ["query", 1]}
op.output_of_module_calls    # ["mlp:3"]   (modules this op IS the output of)
```

User reads: "my output entered block1 (arg 0) and attn (query + arg 1); I AM the output of mlp:3."

### Cascade scope

- `Op.module_ops_entered` → `Op.module_calls_entered` (`backends/torch/model_prep.py:773,1029`, `backends/torch/sources.py:430`, `backends/torch/ops.py:1396`, `backends/mlx/backend.py:673`, `postprocess/finalization.py:501`, `postprocess/labeling.py:754`, `constants.py:353`)
- `Op.module_entry_argnames` → `Op.module_entry_arg_keys` (same file set; `model_prep.py:777,1028`, etc.)
- FIELD_ORDER updates
- Backward-compat: deprecated `@property` aliases for both old names
- Glossary: add the "Op fields default to output" convention note prominently, then the cluster's per-field entries

---

## Op — `output_versions_per_child` → `out_versions_by_child`

| Was | New |
|---|---|
| `Op.output_versions_per_child` | `Op.out_versions_by_child` |

### Rationale

- `output` → `out` for parity with `Op.out` (the tensor itself)
- `per` → `by` for dict-keying emphasis ("indexed by child")
- `versions` kept because it's **load-bearing** — flags the unusual semantic (multiple snapshots of the SAME conceptual tensor due to in-place mutation between child consumptions). Just `activations_by_child` would mask this weirdness.

### Semantic preserved

Dict mapping `child_label → tensor snapshot at the moment that child consumed the output`. Populated when an in-place op modified the tensor between uses, creating divergent per-child views of the same op's output.

### Cascade

- `Op.output_versions_per_child` → `Op.out_versions_by_child`
- Same on Layer
- `op_log.py:463,736,1444`, `validation/core.py:479,482,483,852,853`, `validation/invariants.py:503,573,574,580`, `capture/trace.py:143,146`
- FIELD_ORDER
- Backward-compat: deprecated `@property` alias for one cycle

---

## TODO (test coverage) — verify class-hierarchy delegation works after the rename sprint

Across the walkthrough we've added/renamed ~50+ fields and introduced new delegation paths. The Layer→Op, Module→ModuleCall, GradFn→GradFnCall delegation must be verified end-to-end after the rename sprint.

### Why this matters

TorchLens uses single-Op delegation pervasively:
- `LayerLog.__getattr__` falls through to `ops[1]` for single-op layers (`data_classes/CLAUDE.md:30`)
- ModuleCall fields delegate to/from Module
- GradFnCall fields delegate to/from GradFn
- Many `@property` derivations depend on this delegation chain

The walkthrough adds NEW fields and renames existing ones. Each touches the delegation surface. Without thorough tests, broken delegation manifests as:
- Renamed field works on Op but raises on Layer (delegation not updated)
- Backward-compat `@property` alias works on Op but not on the inherited Layer access
- Multi-pass per-pass fields raise `ValueError` (not `AttributeError`) per the locked convention — tests must verify this contract holds

### Test surface to cover

For every renamed or newly-added field, write tests covering:

1. **Direct access on the primary class** (e.g., `op.new_field`)
2. **Delegation access through aggregate** (e.g., `layer.new_field` for single-Op layers)
3. **Multi-pass raises correctly** (`ValueError` not `AttributeError` for multi-pass aggregates accessing per-pass fields)
4. **Backward-compat alias works** during deprecation period (`op.old_name` returns same as `op.new_field` with warning)
5. **Save/load round-trip** preserves the renamed field (`.tlspec` write/read invariant)
6. **FIELD_ORDER ordering preserved** in `to_pandas()` and repr output
7. **Predicate consistency** — paired predicates and `@property` derivations match (`op.is_compute_op` agrees with `not (op.is_input or op.is_output or ...)`)

### Cross-class delegation checks specifically

- Layer ↔ Op:
  - Single-op layer: `layer.X == layer.ops[1].X` for all renamed fields
  - Multi-pass layer: `layer.X` raises `ValueError` for per-pass fields
- Module ↔ ModuleCall:
  - Single-call module: `module.X == module.calls[1].X` where applicable
  - Multi-call: same `ValueError` semantic
- GradFn ↔ GradFnCall:
  - Single-call grad_fn: similar delegation contract
- Op ↔ Layer ↔ Module: cross-class refs (`op.module_calls_entered`, `module.ops`, etc.) preserve consistency post-rename

### Implementation hint

Generate the test matrix programmatically:
```python
RENAMED_FIELDS = [
    ("Op", "trace_index", "overall_index"),  # old → new
    ("Op", "compute_index", None),  # removed
    ...
]

@pytest.mark.parametrize("class_name, old_name, new_name", RENAMED_FIELDS)
def test_field_rename(class_name, old_name, new_name):
    # Capture; check new exists; check old issues deprecation warning; check delegation chain
    ...
```

Plus integration tests with real models (small ones in `test_real_world_models.py` per the smoke set) to catch delegation breakage end-to-end.

### Order

Run AFTER the rename sprint completes. Each rename should land with at least 1 paired test (write-tests-first per TDD); this TODO ensures the SYSTEMATIC coverage matrix is built and run.

### Risk if skipped

The renamed sprint is the last big internal API change before launch. Public users may rely on field names from existing notebooks, examples, and external docs. Broken delegation would surface as subtle test failures or user reports post-launch.

---

## CONVENTION — Layer-level `bare` vs `total_*` for aggregate-able fields

Locks the semantic rule for fields that can be either per-pass (single Op) or aggregated (across Layer's recurrent Ops).

### The rule

| Form | Single-pass Layer | Multi-pass Layer |
|---|---|---|
| **Bare** (e.g., `Layer.activation_memory`) | delegates to `ops[1]` (single op's value) | raises `ValueError` (use `layer.ops[N].X` directly) |
| **`total_*` prefix** (e.g., `Layer.total_activation_memory`) | sum across (single) op | **sum across all Ops in the Layer** |

The bare form is per-pass (matches the existing Layer-delegation contract); the `total_*` form is the safe aggregate that handles multi-pass automatically.

### Same `total_` prefix at Trace level

At Trace level, `total_X` means "sum across all Ops in the trace" — consistent semantic, different scope:

| Scope | `total_X` means |
|---|---|
| Trace | sum across all Ops in the trace |
| Layer | sum across all Ops in this Layer (the equivalence class) |
| Op | (n/a — single op; bare values are the per-pass quantities) |

User reads `total_*` and infers "summed across constituent units"; the parent class determines the scope.

### Applies to ALL aggregate-able numeric fields

Not just memory. Convention applies to:

| Field family | Bare on Layer | `total_*` on Layer |
|---|---|---|
| Memory | `activation_memory`, `gradient_memory` | `total_activation_memory`, `total_gradient_memory` |
| Params | `num_params` (per-pass; same across passes) | `total_param_memory` (sum) |
| FLOPs / MACs | `flops_forward`, `flops_backward`, `macs` | `total_flops_forward`, `total_flops_backward`, `total_macs` |
| Timing | `func_duration` | `total_func_duration` (sum across passes) |
| Any other per-pass numeric field | bare | `total_*` |

### Cascade scope

- Audit existing Layer fields: identify per-pass vs aggregate-able
- For each aggregate-able field, ensure BOTH forms exist (bare for per-pass; `total_*` for sum)
- Add missing `total_*` aggregations where users may need them
- Predicate fields don't apply (a predicate is per-pass; "total_is_X" doesn't make sense)

### Glossary documentation

For each Layer aggregate field, document the rule:

> **Layer-level aggregate fields follow the `bare` / `total_*` convention:**
> - **Bare** (e.g., `layer.activation_memory`): per-pass value. Delegates to the single Op for single-pass layers; raises `ValueError` for multi-pass (use `layer.ops[N].activation_memory` to access a specific pass).
> - **`total_*`** (e.g., `layer.total_activation_memory`): sum across ALL Ops in this Layer. Handles multi-pass cleanly.

### Tests (per the just-added test-coverage TODO)

For each pair:
1. Bare form: agrees with `ops[1].X` for single-pass; raises `ValueError` for multi-pass
2. `total_*` form: equals sum of `ops[N].X` across passes for any Layer (single or multi-pass)
3. For single-pass: bare and `total_*` are equal (since N=1)

---

## CONVENTION — cross-class references are always label strings (symmetric across all classes)

Locks the data-model convention: any field that references another log object stores the LABEL (string), not the object. Object resolution is via `trace[label]`.

### Why

1. **Symmetric across all classes** — same rule in every direction (Op↔Layer, ModuleCall↔Layer, Op↔Module, GradFn↔Op, etc.)
2. **Serialization-clean** — labels are strings, save/load is trivial
3. **No accessor-coupling in stored data** — fields don't depend on parent trace resolution at write time
4. **Already locked elsewhere** — the GradFn rename (delta line ~398: `parent_grad_fn_labels` → `parents` returning labels) commits TorchLens to this convention
5. **Users resolve when needed** — `trace[label]` works uniformly; no API churn

### Applies to all cross-class reference fields

| Class | Field | Returns |
|---|---|---|
| Op | `parents`, `children` | list[str] (op labels) |
| Layer | `parents`, `children` | list[str] (layer labels) |
| GradFn | `parents`, `children`, `siblings`, `co_parents` | list[str] (grad_fn labels) |
| ModuleCall | `output_layers`, `input_layers` | list[str] (layer labels) |
| Op | `output_of_modules` | list[str] (module addresses) |
| Op | `output_of_module_calls` | list[str] (module call labels) |
| Op | `module_calls_entered` | list[str] (module call labels) |
| Op | `module_entry_arg_keys` | dict[str, list[int\|str]] (keyed by module call label) |
| Op | `equivalent_ops` | set[str] (op labels) |
| Trace | `equivalent_ops` | dict[str, set[str]] (keyed by equivalence class) |

### Tensor objects are different

Tensor-valued fields are NOT in this convention:
- `ModuleCall.outputs` (list[Tensor]) — actual tensor objects, not labels
- `Op.out`, `Op.grad` (Tensor) — actual tensors

The convention is specifically about REFERENCES TO OTHER LOG OBJECTS. Tensors are data, not references.

### Glossary note

Add prominently in the data-model section:

> **TorchLens convention: cross-class references are always label strings.** Any field on one log class that points to another log class stores the target's LABEL (a string), not the object. To get the object: `trace[label]`. Example:
> ```python
> for label in module_call.output_layers:
>     layer = trace[label]
>     print(layer.activation_memory)
> ```
> This convention keeps the data model serialization-clean and symmetric across all class boundaries.

### No renames required

The convention is already established by the locked GradFn rename. Current field names align. This entry locks the convention explicitly so the rename sprint preserves it and the glossary documents it for users.

### Tests (per the test-coverage TODO)

For each cross-class reference field:
- Verify it returns list[str] (or appropriate string-keyed structure)
- Verify `trace[label]` resolves to the correct object for each label
- Verify save/load preserves labels exactly

---

## Module — `is_train_mode` → `training` (PyTorch idiom match)

| Was | New |
|---|---|
| `Module.is_train_mode` | `Module.training` |

### Why

1. **Direct PyTorch idiom match** — `nn.Module.training` is the canonical PyTorch attribute name for the same semantic
2. **Zero translation for PyTorch users** — they already know `module.training`; TorchLens captures the same attribute
3. **Bare bool, no awkward prefix** — current `is_train_mode` reads as "the module IS a train mode" which is bad grammar
4. **Consistent with PyTorch's full attribute set TorchLens mirrors** — `cls`, `class_name`, `class_qualname`, etc.

### `is_train_mode` rejected

- Grammar: modules aren't modes; they're IN a mode
- Doesn't match PyTorch
- Requires users to learn TorchLens-specific naming for a concept already named in PyTorch

### `in_train_mode` rejected (the other option JMT considered)

- Better grammar than `is_train_mode`
- But still TorchLens-specific naming for a PyTorch concept
- `training` (bare) matches PyTorch — strictly better

### One mild caveat acknowledged

The word "training" is overloaded in English (the activity vs the state). But PyTorch users already navigate this overload via `module.training`; TorchLens just captures the same value. Documentation can clarify: "Reflects `nn.Module.training` at capture time — True iff the module was in training mode (not eval mode)."

### Cascade scope

- `Module.is_train_mode` → `Module.training`
- `module_log.py:511,564,618`
- `constants.py:592`
- `compat/_report.py:809-825` references
- FieldPolicy, FIELD_ORDER
- Backward-compat: `is_train_mode` deprecated `@property` alias for one cycle

### Disambiguation from Trace's `backward_ready`

This is unrelated to `Trace.backward_ready` (the just-locked rename from `train_mode`). Different concepts:
- `Module.training` — module's PyTorch train/eval mode at capture time
- `Trace.backward_ready` — whether the trace was captured with backward machinery prepared

Both can coexist; no naming collision.

---

## Module / Trace — three symmetric source-location triplets (class / init / forward)

Locks symmetric source-location field design across Module-like classes (Module + Trace's model view). Three triplets per class, all named symmetrically.

### The triplets

| Aspect | Field triplet | Captures |
|---|---|---|
| Class definition | `class_source_file`, `class_source_line`, `class_source_location` (`@property`) | source of `class MyModel(nn.Module):` |
| `__init__` method | `init_source_file`, `init_source_line`, `init_source_location` (`@property`) | source of `def __init__(self, ...):` |
| `forward()` method | `forward_source_file`, `forward_source_line`, `forward_source_location` (`@property`) | source of `def forward(self, x):` |

Each `_location` `@property` returns `f"{file}:{line}"` for editor jump.

### Per-class changes

**Module** (currently ambiguous bare `source_file/line`):
- Rename: `Module.source_file/line` → `Module.class_source_file/line` + add `class_source_location` @property
- Add: `Module.init_source_file/line/location` (NEW)
- Add: `Module.forward_source_file/line/location` (NEW)

**Trace** (currently has `model_source_*` for class + `forward_source_*` for forward):
- Rename: `Trace.model_source_file/line/location` → `Trace.class_source_file/line/location` (supersedes delta line 74 lock)
- Keep: `Trace.forward_source_file/line/location` (unchanged)
- Add: `Trace.init_source_file/line/location` (NEW)

### Why drop `model_` prefix on Trace

The Trace's source fields all describe the model (Trace's subject). The `model_` prefix is redundant — scope is implicit. Same field names work on both Trace and Module with the class determining what the source describes:

```python
trace.class_source_file       # the model class's source file
trace.init_source_file        # the model's __init__ source file
trace.forward_source_file     # the model's forward() source file

module.class_source_file      # this submodule's class source file
module.init_source_file       # this submodule's __init__ source file
module.forward_source_file    # this submodule's forward() source file
```

Symmetric naming; scope resolved by which class you're on.

### Glossary entries

- `class_source_file` / `class_source_line` / `class_source_location`: source location of the class definition. Captured via `inspect.getfile(cls)` and `inspect.getsourcelines(cls)[1]`.
- `init_source_file` / `init_source_line` / `init_source_location`: source location of the `__init__` method. Captured via `inspect.getsourcelines(cls.__init__)`. None if `__init__` is the default `nn.Module.__init__`.
- `forward_source_file` / `forward_source_line` / `forward_source_location`: source location of the `forward()` method. Captured via `inspect.getsourcelines(cls.forward)`.

All `_location` companions return `f"{file}:{line}"` (or None if either component is missing).

### Supersedes delta line 74 lock

The earlier `model_source_file/line/location` addition on Trace is renamed to `class_source_file/line/location` per this symmetric design. The `model_` prefix is dropped.

### Cascade scope

- Module: rename `source_file/line` → `class_source_file/line/location` (`module_log.py:486-487,533-534,582-583`)
- Module: add `init_source_*` triplet (NEW)
- Module: add `forward_source_*` triplet (NEW)
- Trace: rename `model_source_*` → `class_source_*` (`model_log.py:173-174,754-755,1021-1022,2366-2367,2795-2797`)
- Trace: keep `forward_source_*` (`model_log.py:173-174` etc.)
- Trace: add `init_source_*` (NEW)
- Capture: extend `_get_class_metadata` (`backends/torch/model_prep.py:403-437`) to populate init + forward source locations via `inspect.getsourcelines(cls.__init__)` and `inspect.getsourcelines(cls.forward)`
- Constants: `constants.py:67-68` FIELD_ORDER updates
- FieldPolicy maps
- Backward-compat: deprecated `@property` aliases for `Module.source_file/line` and `Trace.model_source_file/line` for one cycle

### Edge cases

- `__init__` not explicitly defined (uses `nn.Module.__init__`): init triplet returns None
- C-implemented modules (e.g., some quantized ops): inspect calls may fail; triplets gracefully return None
- Dynamically-generated classes: source locations may be partial or missing; capture handles None robustly

---

## Module — hook info asymmetry confirmed (mirrors PyTorch's 2-forward + 4-backward registries)

Locks the six-field hook info design (already locked in delta line 477-513). Documents the forward/backward asymmetry explicitly so users know it's inherent to PyTorch, not a TorchLens design choice.

### The six fields (per locked design)

```python
Module.forward_pre_hook_info
Module.forward_hook_info
Module.backward_pre_hook_info       # legacy
Module.backward_hook_info           # legacy
Module.full_backward_pre_hook_info  # full
Module.full_backward_hook_info      # full
```

### Why the asymmetry (2 forward + 4 backward)

PyTorch defines 2 forward hook registries and 4 backward hook registries. Backward has both LEGACY (`register_backward_hook`) and FULL (`register_full_backward_hook`) variants because the legacy version had quirks:

1. **Multi-input/multi-output modules** — legacy hook's `grad_input`/`grad_output` tuples could be misaligned
2. **In-place op confusion** — gradients could reflect post-in-place state vs pre-in-place
3. **Sub-module interception** — gradients from nested modules weren't always intercepted as users expected
4. **Module-level vs autograd-level confusion** — legacy hook fires per-autograd-node, can fire multiple times per module backward
5. **Partial gradients** — some module types didn't get correct `grad_input`

`register_full_backward_hook` (added later) fixes all of these.

No equivalent forward issue exists, so no `full_forward_hook` registry in PyTorch.

### Glossary note (add to Module section)

> **Hook registry asymmetry: 2 forward + 4 backward.**
>
> PyTorch defines 2 forward hook registries (`forward_pre_hooks`, `forward_hooks`) and 4 backward hook registries: 2 legacy (`backward_pre_hooks`, `backward_hooks`) plus 2 full (`full_backward_pre_hooks`, `full_backward_hooks`). The "full" variants exist ONLY for backward, added later by PyTorch to fix quirks in the legacy backward hooks (multi-output handling, in-place ops, module-level firing).
>
> TorchLens mirrors all six registries exactly. For new code, prefer `register_full_backward_hook` over the legacy `register_backward_hook` (per PyTorch docs). Both variants stay in the API for backward compatibility.

### Lock confirmation

No changes to the six-field design. Glossary documentation explicitly notes the asymmetry is inherent to PyTorch, not a TorchLens design choice.

---

## Module / GradFn — full source / signature / docstring symmetry

Extends the locked source-triplet design to add signatures and docstrings, plus apply uniformly across Module AND GradFn.

### Symmetric cluster (per class)

| Aspect | Source triplet | Signature | Docstring |
|---|---|---|---|
| Class | `class_source_file/line/location` | (n/a) | `class_docstring` |
| `__init__` | `init_source_file/line/location` | `init_signature` | `init_docstring` |
| `forward()` | `forward_source_file/line/location` | `forward_signature` | `forward_docstring` |
| `backward()` (GradFn only) | `backward_source_file/line/location` | `backward_signature` | `backward_docstring` |

### Per-class scope

**Module** — adds all four families (class, init, forward); NO backward family (Modules don't define backward methods):

```python
# Source triplets (3 — locked in previous delta)
class_source_*, init_source_*, forward_source_*

# Signatures (NEW — 2)
init_signature, forward_signature

# Docstrings (NEW — 3)
class_docstring, init_docstring, forward_docstring
```

**GradFn** — adds four families including backward:

```python
# Source triplets (4 — NEW)
class_source_*, init_source_*, forward_source_*, backward_source_*

# Signatures (existing + NEW)
forward_signature (existing), backward_signature (existing)
init_signature (NEW — rare; often None for autograd Functions)

# Docstrings (existing + NEW)
forward_docstring (existing), backward_docstring (existing)
class_docstring (NEW), init_docstring (NEW — rare)
```

### Trace's `model_*` follows Module pattern

Trace's model view inherits the same fields (via the locked `class_source_*`, `init_source_*`, `forward_source_*` triplets):

```python
trace.class_docstring   # model class docstring
trace.init_docstring    # model __init__ docstring
trace.forward_docstring # model forward() docstring
trace.init_signature, trace.forward_signature
```

### Implementation

Via `inspect` module:
- `inspect.getsourcefile(cls)`, `inspect.getsourcelines(cls)` for class
- `inspect.getsourcelines(cls.__init__)`, `inspect.signature(cls.__init__)`, `cls.__init__.__doc__`
- Same for `forward` / `backward`
- Capture during model_prep / grad_fn_metadata extraction

### Edge cases (handle as None)

- Default `nn.Module.__init__` (user didn't override): init source/signature/docstring all `None`
- staticmethod forward/backward on autograd.Function: signature works; source via the underlying function
- C-implemented modules / functions: inspect may fail; gracefully return `None`
- Default `object.__init__` (autograd Function with no custom init): `None`

### Glossary organization

Each Module-like class gets a "Class introspection" subsection covering source/signature/docstring fields uniformly:

> **Class introspection cluster.** Each Module-like class exposes source location, signature, and docstring info for its `class`, `__init__`, and `forward` methods (and `backward` for GradFn). Useful for debugging, replay, and documentation generation. All fields gracefully return `None` when the underlying method isn't user-defined (e.g., default `nn.Module.__init__`).

### Cascade

- Module: 5 NEW fields (`class_docstring`, `init_docstring`, `forward_docstring`, `init_signature`, `forward_signature`)
- GradFn: 4 source triplets (12 fields) + 4 signature/docstring NEW (`class_docstring`, `init_docstring`, `init_signature`; backward_signature and forward_signature already exist) + 2 existing renamed for cluster consistency if needed
- Trace: inherits the Module pattern via the locked `class_source_*` etc. cluster
- Capture: extend `_get_class_metadata` (`backends/torch/model_prep.py`) for Module and the analogous GradFn introspection in backward setup
- FieldPolicy.KEEP for all (load-survivable)
- FIELD_ORDER updates

### TODO — Op / Layer convenience fields for input info

Currently:
- Op stores `args_template`, `kwargs_template` (structural templates for op inputs — populated when `save_arg_templates=True`)
- Op stores `arg_names` (names of positional args)
- Op stores `func_name`, `func_signature` (info about the function being called)
- Op does NOT store explicit input tensor objects (they're accessed via `parents[i].out`)

**Open question for future review**: should Ops/Layers expose convenience fields for input information, similar to how GradFnCall stores `grad_inputs` AND `grad_outputs` (rather than requiring derivation)?

Candidates:
- `Op.input_shapes` — shapes of input tensors (currently derivable via `[op.trace.ops[p].shape for p in op.parents]`)
- `Op.input_dtypes` — same for dtypes
- `Op.input_memory_total` — total memory of input tensors (sum across parents)
- `Op.inputs` — actual input tensors (would be REDUNDANT with parent outputs)

The current convention (TorchLens Op fields refer to OUTPUT; inputs derived via parents) is consistent and clean. Adding convenience fields:
- Adds API surface
- May or may not earn its keep
- Could be `@property` derivations (zero storage cost) for common cases

**Defer to post-rename sprint**: revisit whether `Op.input_shapes`, `Op.input_dtypes`, etc. as `@property` derivations would be a worthwhile ergonomics addition. Don't add stored fields (avoid the GradFn-style double-storage); evaluate `@property` ergonomics specifically.

Symmetry note: GradFnCall stores both grad_inputs AND grad_outputs because accumulation makes them genuinely different (not derivable). For forward Ops, inputs ARE derivable from parents (no accumulation). So the forward analog would be `@property` convenience, not stored fields.

---

## Super[T] — drop `labels` dict; rename `node_name` → `label`

Simplification based on the locked alignment-by-label semantic.

### Why drop `labels` (the per-member dict)

Per the locked Bundle alignment: `bundle.ops[label]` returns a SuperOp whose members all share the same label by construction. The `labels` dict (`@property` at `_super/_base.py:90`) returns identical values for every member — pure redundancy.

The fallback chain at `_base.py:70` (`label` → `layer_label` → `repr(query)`) is defensive code from earlier design; in the current locked semantic, alignment IS by label, so all members have the same value.

### Why rename `node_name` → `label`

- "Node name" reads as graph-vis-flavored; the canonical identifier IS just a label
- Bare `label` (singular) is consistent with the locked Op/Layer/etc. cluster
- Pairs with `Super[T].members` (dict[trace_label → object]) for object resolution

### Final Super[T] identification surface

```python
super_op.label      # canonical label (one string; same for every member)
super_op.members    # dict[trace_label → Op] for resolving each member's object
```

If misaligned bundles ever become a real concern, add an explicit `align_by` parameter or different alignment strategy; don't carry per-member-labels by default.

### Cascade

- Drop `Super[T].labels` `@property` at `_super/_base.py:89-102`
- Rename `Super[T].node_name` → `Super[T].label` (and `_node_name` → `_label` internally)
- Update all Super* subclasses (SuperOp, SuperLayer, SuperModule, etc.)
- All call sites in `intervention/_super/`, `intervention/bundle.py`, `intervention/_topology/`
- Glossary: single `label` field; per-member object access via `.members[trace_label]`

---

## `tl.trace()` — remove vis cruft (single responsibility: capture only)

`tl.trace()` currently has ~22 vis-related parameters (`vis_opt`, `view`, `depth`, `renderer`, `layout`, `node_style`, `vis_mode`, `vis_call_depth`, `vis_outpath`, `vis_save_only`, `vis_fileformat`, `vis_buffers`, `vis_direction`, `vis_graph_overrides`, `vis_node_mode`, `vis_edge_overrides`, `vis_grad_edge_overrides`, `vis_module_overrides`, `vis_node_placement`, `vis_renderer`, `vis_theme`, `vis_intervention_mode`, `vis_show_cone`, `layer_visualizers`, `save_visualizations`).

These violate single-responsibility — `trace()` should run the forward pass and capture, full stop.

### Clean two-step pattern

```python
# Capture (do one thing: run forward + log)
trace = tl.trace(model, input)

# Render (separately)
trace.draw(mode="default", direction="vertical", ...)
```

### What stays on `tl.trace()` (capture only)

```python
trace = tl.trace(
    model, input_args, input_kwargs=None,
    # Capture config
    save_raw_activations, save_gradients,
    save_arg_values, save_arg_templates,
    backward_ready, intervention_ready,
    detach_saved_activations,
    layers_to_save, grads_to_save,
    # Performance / streaming
    keep_outs_in_memory, save_outs_to, output_device,
    keep_grads_in_memory, save_grads_to,
    # Edge cases
    emit_nvtx, raise_on_nan, random_seed, num_context_lines,
    # NO vis params
)
```

### What moves to `Trace.draw()` (vis only)

All ~22 vis params land on `.draw()`. The method already exists; this becomes the canonical visualization entry point.

`layer_visualizers` passes per-call to `.draw()` (or set on Trace via separate setter if persistent customization needed).
`save_visualizations` becomes part of `tl.save(trace, ...)` or `.draw(save_to=...)`.

### Why this is a clear win

1. **Single responsibility for both functions** — `trace()` captures; `.draw()` renders
2. **Faster capture-only flows** — users who don't need vis don't load Graphviz
3. **Clearer mental model** — capture is data; draw is rendering; they're orthogonal operations
4. **Aligns with the architectural endpoint** — capture substrate + separate viewer
5. **Honest about what happens** — vis is optional postprocessing, not part of capture
6. **trace() signature becomes readable** — ~22 fewer parameters

### Migration (hard break, pre-launch)

```python
# Before
tl.trace(model, x, vis_opt="default")

# After
trace = tl.trace(model, x)
trace.draw(mode="default")
```

No deprecation cycle — pre-launch hard remove. Update all docs, examples, notebooks, README to two-step pattern.

### Cascade scope

- Remove vis params from `tl.trace()` signature (`user_funcs.py:1218-1290`)
- Remove all `vis_*=vis_*` passthrough in trace body
- Move logic to `Trace.draw()` method (verify all functionality lands there)
- `tl.show()` / `tl.show_graph()` / similar one-call convenience wrappers — keep if they exist (they're the "trace + draw" combo for users who want one call); they internally do `trace = tl.trace(...); return trace.draw(...)`
- Update ALL docstrings — remove vis sections from `trace()`, expand `.draw()` docs
- Update examples / notebooks / README to two-step pattern
- Migration guide in docs: "0.X → 2.0 trace() signature changes"
- Tests: `trace()` tests no longer cover viz behavior; viz tests target `.draw()`

### Tutorial / docs framing

> **Capture and visualization are separate steps in TorchLens 2.0.**
>
> `tl.trace(model, x)` captures the forward pass and returns a `Trace` object containing all the data — activations, gradients, structure, metadata. No visualization is rendered.
>
> To visualize, call `trace.draw(...)` separately. This separation lets you:
> - Capture once, render many ways (different modes, themes, focuses)
> - Skip visualization entirely if you only need the data
> - Inspect / save / manipulate the Trace before rendering

---

## Op — add `fx_label` (combined FX-style label as a lookup key)

`Op.fx_qualpath` and `Op.fx_call_index` already exist (`op_log.py:561-562`) but the combined FX-style form is computable-but-not-stored. Users have to construct manually.

### Add

```python
@property
def fx_label(self) -> str | None:
    """torch.fx-style label combining fx_qualpath + fx_call_index."""
    if self.fx_qualpath is None:
        return None
    base = self.fx_qualpath.replace(".", "_")
    return f"{base}_{self.fx_call_index}" if self.fx_call_index else base
```

Register as lookup key in `_add_lookup_keys_for_layer_entry` so `trace[fx_label]` works.

### Use case

Users porting between TorchLens and torch.fx; comparing across naming schemes.

### Cascade

- Add `Op.fx_label` (`@property`) on Op
- Register `fx_label` in `lookup_keys_for_tensor` at `postprocess/labeling.py:735+`
- Glossary entry
- Layer-level analog as delegate (`@property` on Layer)

---

## TODO — variable name introspection (possible future feature)

Possible future addition: introspect Python variable names that bind to tensor objects and store as a field on the corresponding Op.

### The idea

When a user writes:
```python
hidden = encoder(x)
out = decoder(hidden)
```

The tensor object that becomes `Op.out` for the encoder is bound to the local variable `hidden`. If TorchLens could introspect this binding, an Op field like `Op.var_names: list[str]` would store the names this tensor was bound to.

### Why useful

- Debugging: "where is this tensor in the user's code?"
- Visualization: show variable names in graph nodes
- Education: connect captured graph to user's mental model

### Why hard

- Python's introspection of variable bindings is non-trivial (would need to walk frame locals)
- Multiple bindings: a tensor can be bound to many variables across scopes
- Reassignment: `x = encoder(...); x = decoder(x)` — `x` binds to multiple distinct tensors over time
- AST analysis vs runtime introspection trade-offs
- Performance cost (frame inspection is slow)

### Defer

Flag as a possible future feature. Investigate during a post-launch experimentation cycle. Don't add to the rename sprint or current locked design — it's a feature decision, not a renaming concern.

If pursued, likely candidates:
- `Op.var_names: list[str]` — variable names this tensor was bound to during forward
- `Op.first_var_name: str | None` — first / most-recent binding (convenience)

---

## Bundle / class-level constants

| Scope | Old | New | Rationale |
|---|---|---|---|
| Trace (and any other class with these constants) | `DEFAULT_VALUES` | `FIELD_DEFAULTS` | Conveys per-field dict; shared `FIELD_` prefix forms family marker. |
| Trace (and any other class with these constants) | `FORK_POLICY` | `FIELD_FORK_POLICY` | Same. |
| Trace (and any other class with these constants) | `SAVE_POLICY` | `FIELD_SAVE_POLICY` | Same. |

Notes:
- Caveat: external tooling reads these names — verify cascade scope before applying. `git grep` for each old name across torchlens + tests + notebooks + docs + bridge adapters.
- Glossary entry to update: `.project-context/torchlens_glossary.md:47-49`
- Audit notes entry to update: `.project-context/notebook_audit_notes.md:853-860`

## Confirmed locks (no change)

- `root_trace` — kept. Tree vocab pairs cleanly with `parent_trace`; `source_trace` / `origin_trace` mismatch family vocabulary and don't fix the depth-ambiguity. Audit reasoning still holds. (Glossary line 44.)

## Trace counts / memory

| Scope | Old | New | Rationale |
|---|---|---|---|
| Trace | `num_intervening_grad_fns` | `num_grad_fns_without_op` | Mirrors per-instance `has_op` predicate (which itself replaced `is_intervening` in the audit). Zero new vocabulary; symmetric with possible future `num_grad_fns_with_op`. Also resolves "intervening" / "interventions" lexical clash with the new intervention API surface. |

Notes:
- Glossary entry to update: `.project-context/torchlens_glossary.md:56`
- Audit notes entry to update: `.project-context/notebook_audit_notes.md:450`

## Trace memory — new fields (NOT just rename; implementation work)

Three additions to close the activation/gradient/peak parity asymmetries. Use full-word `gradient` form (parallel to `total_activation_memory` / `saved_activation_memory` neighbors in the same Memory cluster).

| Scope | New field | Parallel to | Notes |
|---|---|---|---|
| Trace | `total_gradient_memory` (+ `_str`) | `total_activation_memory` | Bytes for all Op gradients computed during backward (whether saved or not). Computable from Op output shape+dtype since gradient shape == output shape. |
| Trace | `saved_gradient_memory` (+ `_str`) | `saved_activation_memory` | Bytes for Op gradients actually saved by TorchLens. |
| Trace | `total_param_gradient_memory` (+ `_str`) | `param_memory` | Bytes for all parameter gradients populated during backward. JMT confirmed this is its own concern (separate from Op gradients). |
| Trace | `forward_peak_memory` (+ `_str`) | `backward_peak_memory` | Forward-pass peak analog. Audit TODO #15 already tracked this. |

Notes:
- Implementation work, not just naming. Defer to a follow-up sprint after rename pass; only the naming is locked here.
- Existing audit TODO #3 (gradient memory parity) and TODO #15 (forward peak) are both subsumed by these locks.

## Internal source/sink — Trace-level collection parity

Audit renamed per-instance predicates to source/sink vocab (LayerLog cluster 8: `is_internally_initialized` -> `is_internal_source`; OpLog cluster I: 3 ancestor fields) but missed the parallel rename on the Trace-level collection fields.

| Scope | Old | New | Rationale |
|---|---|---|---|
| Trace | `internally_initialized_ops` | `internal_source_ops` | Mirror per-instance `is_internal_source`; same vocab family as `internal_source_parents` / `internal_source_ancestors`. |
| Trace | `internally_terminated_ops` | `internal_sink_ops` | Mirror per-instance `is_internal_sink`. |

Notes:
- Glossary entries to update: `.project-context/torchlens_glossary.md:84-85`
- Third field `internally_terminated_bool_ops` DEFERRED — it intersects `is_internal_sink` AND `is_terminal_bool`, and the bool axis is part of cluster J (Conditional bool details, currently DEFERRED in the audit pending integrated conditional-flow rethink). Resolve when cluster J resolves.

## Trace source-code reference parity

Locked vocabulary at ModuleLog (cluster B) and GradFnLog uses the `source_file` / `source_line` / `source_location` triplet. Trace currently has only `forward_lineno` — incomplete and uses different vocab. Complete the triplet on Trace AND add a class-level triplet (since `model_class` is lost-on-load and a load-survivable handle to "where the model class lives" is useful).

| Scope | Old | New | Rationale |
|---|---|---|---|
| Trace | `forward_lineno` | `forward_source_line` | Match locked `source_*` vocab. |
| Trace | (new) | `forward_source_file` | File where the model's `forward()` method is defined. |
| Trace | (new) | `forward_source_location` (`@property`) | `f"{forward_source_file}:{forward_source_line}"` for editor jump. |
| Trace | (new) | `model_source_file` | File where the model class is defined. Load-survivable. |
| Trace | (new) | `model_source_line` | Line where the model class is defined. |
| Trace | (new) | `model_source_location` (`@property`) | `f"{model_source_file}:{model_source_line}"`. |

Notes:
- Glossary entry to update: `.project-context/torchlens_glossary.md:136`
- Implementation work for the new fields (not just rename); but vocabulary is lockable now.
- Consistent with audit's pending FuncCallLocation alignment (lean A: `source_file` / `source_line` everywhere). Audit notes lines 1695-1710.

## Trace capture-config — args/values vs args/template

The pair `save_function_args` and `capture_full_args` reads as duplicates but serves orthogonal roles: values vs. template.

| Was | New | Rationale |
|---|---|---|
| `save_function_args` | `save_arg_values` | Deep-copies actual tensor VALUES into each Op (heavy memory cost). Used for validation/reproducibility replay. |
| `capture_full_args` | `capture_args_template` | Captures structured TEMPLATE (slot map, shape/dtype placeholders) used by intervention/cache machinery. Lightweight. Transient (DROP on save). |

Notes:
- Glossary entries to update: `.project-context/torchlens_glossary.md:143,162`
- Renames clarify the values-vs-template orthogonality without touching the underlying mechanics.

## OpLog — `num_ops` -> `num_passes` (scope-native, NOT cross-scope)

| Was | New | Where | Pairs with |
|---|---|---|---|
| `num_ops` ("ops in parent Layer") | `num_passes` | OpLog (line 208) | `pass_index` positional ("pass 3 of 5") |
| `num_ops` ("ops aggregated by this Layer") | unchanged | LayerLog (line 369) | `layer.ops` collection — `len(layer.ops) == layer.num_ops` |
| `num_ops` ("total ops in trace") | unchanged | Trace (line 52) | `trace.ops` accessor |
| `num_ops_with_params` | unchanged | Trace (line 62) | |

Initially proposed cross-scope rename to `num_passes` everywhere; revised to scope-native: each scope's count name pairs with its native reference. LayerLog/Trace pair `num_ops` with their respective `ops` collections; OpLog pairs `num_passes` with the positional `pass_index`. "Inconsistency across scopes" is honest — the same number is genuinely viewed differently from each vantage point.

`pass_index` itself stays unchanged — honest as a coordinate field even though "Pass" was dropped from class names.

## OpLog — drop redundant layer back-ref fields

Three fields on OpLog are exact duplicates of Op-scope fields (per `postprocess/labeling.py:84-87` inheritance). Zero uses found in `torchlens/`. Drop:

| Drop | Redundant with |
|---|---|
| `type_index` | `type_index` (same value, inherited from first pass of Layer) |
| `trace_index` (was `trace_index`) | `trace_index` (same value, inherited) |
| `layer_type` | `type` (same value, inherited) |

Stays:
- `layer_label` — distinct from `op.label` (pass-qualified vs. bare-Layer); real cross-class reference.

## OpLog/LayerLog — `detach_saved_tensors` -> `detach_saved_activations`

| Was | New | Where |
|---|---|---|
| `detach_saved_tensors` | `detach_saved_activations` | Trace, OpLog, LayerLog (cascades) |

Aligns with Trace-level activation vocabulary family (`save_raw_activations`, `activation_transform`, `total_activation_memory`, `saved_activation_memory`, `ops_with_saved_activations`).

Footnote — pre-existing typo: `torchlens/constants.py:47` has `"detach_saved_tensorss"` (double-s) at MODELLOG/Trace level; OpLog/LayerLog FIELD_ORDER use single-s. Renaming naturally cleans this up.

## OpLog/LayerLog — gradient_transform parity

`gradient_transform` exists at Trace (line 161) but is missing on OpLog (line 255 has only `activation_transform`) and LayerLog (line 413 same). Add for parity with `activation_transform`.

| Scope | New field | Parallels |
|---|---|---|
| OpLog | `gradient_transform` | `activation_transform` |
| LayerLog | `gradient_transform` | `activation_transform` |

Each documents which callable produced the corresponding `transformed_grad`. Implementation work, not just naming.

## OpLog/LayerLog — `func_call_stack` -> `code_context`

| Was | New | Where |
|---|---|---|
| `func_call_stack` | `code_context` | OpLog (line 271), LayerLog (line 423) |

Pairs with the existing config flag `save_code_context` (line 145) and `num_context_lines` — coherent vocabulary cluster. Honest about what's stored (each `FuncCallLocation` carries source context lines via `linecache`). Advertises the discoverability of the user-code-introspection feature without overcommitting to "this is a code blob" the way `source_code` would have. Field stays singular even though data is `List[FuncCallLocation]` ("context" is collective-noun-friendly).

Cascade note: drops the `func_*` family marker for this field. Other `func_*` fields (`func_qualname`, `func_duration`, etc.) stay.

## OpLog/LayerLog — `is_atomic_module_output` -> `is_atomic_module_op`

| Was | New | Where |
|---|---|---|
| `is_atomic_module_output` | `is_atomic_module_op` | OpLog (line 317), LayerLog (line 466) |

Reframes from tensor/output relation to op-module 1:1 correspondence — the actual key concept (per JMT: an atomic module has ONE op that IS the module's op). Pairs with companion `atomic_module_call` (ModuleCall reference); `atomic_module_*` family preserved.

## OpLog — `feeds_output` -> `is_output_parent`

| Was | New | Where |
|---|---|---|
| `feeds_output` | `is_output_parent` | OpLog (line 341), LayerLog (passthrough) |

Brings into the `is_*` predicate cluster (joins `is_input`, `is_output`, `is_internal_source`, `is_internal_sink`, etc.). Uses the locked `parents`/`children` graph vocabulary — this op IS a parent of the synthetic output node. Self-derives from existing accessors: `op.is_output_parent` ≡ `any(child.is_output for child in op.children)`.

Audit had kept `feeds_output` citing "verb form established," but the cluster is overwhelmingly predicate-form; the verb-object form was the outlier.

## OpLog — `save_tensor_data` -> `save_activation`

| Was | New | Where |
|---|---|---|
| `save_tensor_data(...)` | `save_activation(...)` | OpLog method (glossary line 357) |

Aligns with Trace-level activation vocabulary family (`save_raw_activations`, `activation_transform`, `has_saved_activation`, `ops_with_saved_activations`, `saved_activation_memory`). Singular is honest — one call saves one activation. Resolves the "tensor" outlier vocabulary.

Audit had explicitly deferred this method's final name pending intervention-survey workflow changes; if the survey doesn't restructure the save flow, `save_activation` locks.

## OpLog — add `atomic_module` (@property) for completeness

`atomic_module_call` exists on OpLog (line 327) but the bare `atomic_module` reference is missing. Add for parity with the existing `output_of_modules` / `output_of_module_calls` pairing pattern.

| New field | Type | Returns |
|---|---|---|
| `atomic_module` (`@property`) | `ModuleLog \| None` | The atomic ModuleLog this Op corresponds to; `None` if not an atomic-module op. Derives from `self.atomic_module_call.module` (or via Trace's modules accessor). |

LayerLog gets it via single-Op passthrough.

## ParamLog — `linked_params` -> `co_parent_params`

| Was | New | Where |
|---|---|---|
| `linked_params` | `co_parent_params` | ParamLog (line 606) |

Family-vocab consistency with the locked `co_parents` graph relationship (OpLog/LayerLog). Two params that share a child op (e.g., conv weight + conv bias both feeding the same Conv2d op) are co-parents of that op. The `_params` qualifier scopes the result to param entities only (vs. bare `co_parents` which at param scope would be ambiguous about whether it includes input tensors, constants, etc.).

Side benefit: addresses audit TODO line 2103 ("tighten `linked_params` docstring") — the new name itself documents the relationship.

## ParamLog — add `module` / `modules` properties for parity

ParamLog has `module_address` (line 599) and `all_module_addresses` (line 602) — string addresses — but no direct ModuleLog accessor. Same parity gap fixed for OpLog with `atomic_module`.

| New field | Type | Returns |
|---|---|---|
| `module` (`@property`) | `ModuleLog` | `self.trace.modules[self.module_address]` |
| `modules` (`@property`) | `list[ModuleLog]` | `[self.trace.modules[a] for a in self.all_module_addresses]` |

User benefits: skip the address-lookup hop. `param.module.cls` reads cleaner than `param.trace.modules[param.module_address].cls`.

## BufferLog — `buffer_use_index` -> `buffer_overwrite_index`

The audit's `buffer_use_index` was meant to track buffer interactions broadly, but "use" is ambiguous (could be read or write). JMT clarified the actual concept: buffer reads (one buffer feeding multiple layers) require no special tracking — it's only when contents are REWRITTEN that indexing matters. The index represents WHICH overwrite this entry corresponds to.

`overwrite` is honest about the actor/recipient asymmetry: the buffer doesn't mutate itself (vs. `mutation` which is agentless) — an op overwrites the buffer. Pairs cleanly with the existing `buffer_source` field (the actor doing the overwriting).

| Was | New | Where |
|---|---|---|
| `buffer_use_index` | `buffer_overwrite_index` | BufferLog (line 628) |

Companion vocabulary to slot in alongside (likely additions during implementation):

| New field | Type | Meaning |
|---|---|---|
| `is_overwritten` | bool | True when this buffer is overwritten during forward (vs. static buffers that aren't) |
| `num_overwrites` | int | Total overwrites of this buffer's address during the trace |
| `last_overwrite_source` | Op/Layer ref | The most recent op that overwrote (optional; complements `buffer_source`) |

Edge case noted: KV-cache-style append patterns appear "overwrite" at the storage level (since `buffer.copy_(new_full)` or `buffer = torch.cat(...)` rewrites the underlying storage), so `overwrite` is honest even when the user logically thinks of it as "append."

## Conditional flow — full redesign

The audit deferred the entire conditional-flow surface to an integrated rethink. This walkthrough designed the replacement from first principles. Replaces 18 fragmented fields across Trace + OpLog/LayerLog with 3 stored fields + 7 derived properties + 3 small data classes.

### Three new data classes

```python
@dataclass
class Conditional:
    """One if-chain at one source location."""
    id: str                                          # f"cond_{leading_terminal_bool_op_label}"
                                                      # e.g., "cond_gt_1_4"
    arms: list[ConditionalArm]                       # ordered: leading then, elifs, optional else
    fired_arm_index: int | None                      # which arm fired; None if no arm fired
    fired_arm_kind: Literal["then", "elif", "else"] | None  # denormalized
    source_file: str | None                          # file containing the if-statement
    source_line: int | None                          # line of the `if` keyword

    @property
    def source_location(self) -> str | None:
        """Combined 'file:line' for editor jump; None if either part missing."""

    @property
    def fired_arm(self) -> ConditionalArm | None:
        """Direct access to the fired arm; None if no arm fired."""

    @property
    def has_else(self) -> bool: ...

    @property
    def has_elif(self) -> bool: ...

    @property
    def num_arms(self) -> int: ...

    @property
    def num_elifs(self) -> int: ...


@dataclass
class ConditionalArm:
    """One arm of an if-chain. Each arm has an evaluation side (compute condition)
    and an execution side (run body if won)."""
    kind: Literal["then", "elif", "else"]

    # Evaluation side
    evaluation_op_labels: list[str]                  # ops computing this arm's condition (empty for "else")
    terminal_bool_op_label: str | None               # final scalar bool (None for "else")
    bool_value_at_run: bool | None                   # bool's value when evaluated
    condition_evaluated: bool                        # True iff this arm's condition was reached this run
    evaluation_entry_edge: tuple[str, str] | None    # edge into first eval op
                                                      # labeled "IF"/"elif" in viz; None for "else"

    # Execution side
    execution_op_labels: list[str]                   # ops in this arm's body (empty if didn't fire)
    fired: bool                                      # True iff this arm's body actually ran
    execution_entry_edge: tuple[str, str] | None     # edge into first body op
                                                      # labeled "THEN"/"ELSE" in viz; None if didn't fire


@dataclass
class ConditionalRoleRef:
    """One op's participation in a conditional arm."""
    conditional_id: str                              # the Conditional this op participates in
    arm_index: int                                   # which arm in conditional.arms
    arm_kind: Literal["then", "elif", "else"]        # denormalized arm kind
    role: Literal["evaluation", "body"]              # which side of the arm this op participates in
                                                      # (terminal_bool tagged separately on op via
                                                      #  is_terminal_bool + terminal_bool_for)
```

### Trace-level field surface

| Field | Type | Notes |
|---|---|---|
| `conditionals` | `ConditionalAccessor` | NEW canonical (Accessor: integer ord + `id` string lookup) |
| `is_dynamic_graph` | bool (`@property`) | LOCKED — derives from `has_conditionals` (extends to `has_conditionals OR has_data_dependent_loops` once iteration story lands) |
| `has_conditionals` | bool (`@property`) | NEW — `len(conditionals) > 0` |
| `num_conditionals` | int (`@property`) | NEW — `len(conditionals)` |

### OpLog / LayerLog field surface

```python
op.in_conditionals: list[ConditionalRoleRef]      # stored
op.is_in_conditional: bool                         # @property
op.is_in_conditional_evaluation: bool              # @property — am I computing some arm's condition?
op.is_in_conditional_body: bool                    # @property — am I in some arm's body?
op.conditional_depth: int                          # @property — distinct conditional_id count
op.is_terminal_bool: bool                          # LOCKED
op.is_scalar_bool: bool                            # LOCKED
op.bool_value: bool | None                        # LOCKED
op.terminal_bool_for: tuple[str, int] | None      # NEW — (conditional_id, arm_index) when this op
                                                    # IS the terminal bool of an arm
```

### Fields dropped (canonical replacements above)

**Trace-level (6 fold into `conditionals`):**
- `conditional_records`
- `conditional_arm_entry_edges`
- `conditional_edge_call_indices`
- `conditional_then_entry_edges`
- `conditional_elif_entry_edges`
- `conditional_else_entry_edges`

**Op/Layer-level (12 fold into `in_conditionals` + `terminal_bool_for`):**
- `is_in_conditional_body`
- `conditional_role_stacks`
- `conditional_role_stack_passes`
- `conditional_arm_children`
- `conditional_entry_children`
- `conditional_then_children`
- `conditional_elif_children`
- `conditional_else_children`
- `is_terminal_conditional_bool`
- `conditional_context_kind`
- `conditional_wrapper_kind`
- `terminal_conditional_id`

### Design principles applied

- **Vocabulary unified**: single `conditional_*` prefix throughout; abbreviated `cond_*` and `bool_*_conditional_*` fragmentation eliminated.
- **First-principles ontology**: each conditional has arms; each arm has BOTH an evaluation side (compute condition) and an execution side (run body if won); each op has a role wrt each arm it participates in.
- **Visualization-first**: `evaluation_entry_edge` / `execution_entry_edge` are explicit per arm — these are the edges labeled with arm-kind tags ("IF" / "THEN" / "elif" / "ELSE") in graph diagrams.
- **Stable string ids**: `conditional_id` is `f"cond_{leading_terminal_bool_op_label}"` — survives `.tlspec` save/load, supports cross-trace comparison.
- **Locked predicates preserved**: `is_terminal_bool`, `is_scalar_bool`, `bool_value` stay (tag the bool tensor's nature). One new bool→conditional pointer: `terminal_bool_for: tuple[conditional_id, arm_index] | None`.
- **Loops deferred**: while/for handling is a separate concept (iteration / recurrence). When that lands, the recurrence machinery gets `is_data_dependent: bool` + `continuation_bool_op_label: str` flags, and `is_dynamic_graph` extends to OR them.

### Net field-count

| Surface | Old | New |
|---|---|---|
| Trace conditional fields | 6 stored | 1 stored (Accessor) + 3 properties |
| Op/Layer conditional fields | 12 stored | 2 stored + 4 properties |
| Data classes used | (none — flat fields) | 3 (`Conditional`, `ConditionalArm`, `ConditionalRoleRef`) |

18 stored → 3 stored, with 3 small data classes carrying the structural shape.

## Bundle — `remove_all_but` -> `remove_except`

| Was | New | Where |
|---|---|---|
| `remove_all_but(keep)` | `remove_except(keep)` | Bundle (line 711) |

Shorter (one underscore vs two), avoids the slight redundancy of "all" + "but," and matches set-operation vocabulary (SQL `EXCEPT`, math set difference). Audit-locked `remove_all_but` is fine but `remove_except` reads cleaner in API code.

## GradFnLog — `overall_index` -> `trace_index`

| Was | New | Where |
|---|---|---|
| `overall_index` | `trace_index` | GradFnLog (line 671) |

Propagates the OpLog/LayerLog index family rename (locked earlier). `type_index` stays unchanged. No `capture_index` / `compute_index` analogs at GradFn scope (no orphan-removal pipeline; no I/O sentinel concept). `GradFnCallLog.call_index` stays — semantic-parallel to OpLog's `pass_index` but vocabulary-aligned to its class name (`GradFnCallLog`, not `GradFnPassLog`).

## GradFnLog — full graph-relation parity with OpLog/LayerLog

Currently has only `parent_grad_fn_labels` and `child_grad_fn_labels` (verbose names, label-based). OpLog/LayerLog have full graph-relation cluster with bare names and predicate companions.

Renames + additions:

| Was | New | Notes |
|---|---|---|
| `parent_grad_fn_labels` | `parents` | drop `_grad_fn_labels` qualifier — at GradFn scope "parents" is unambiguous (only one parent relationship). Returns labels (strings), matches OpLog/LayerLog convention. |
| `child_grad_fn_labels` | `children` | drop qualifier; same as above |
| (missing) | `siblings` | add. GradFns sharing a parent grad fn. `@property` derived from existing parent/child data. |
| (missing) | `co_parents` | add. GradFns sharing a child grad fn. `@property` derived. |
| (missing) | `has_parents` | add. Predicate companion. |
| (missing) | `has_children` | add. Predicate companion. |
| (missing) | `has_siblings` | add. Predicate companion. |
| (missing) | `has_co_parents` | add. Predicate companion. |

After locks, GradFnLog graph cluster mirrors OpLog/LayerLog exactly:

```
parents, children, siblings, co_parents
has_parents, has_children, has_siblings, has_co_parents
```

User who learned the OpLog/LayerLog graph navigation API doesn't have to learn a different one for GradFn.

## GradFnLog — `op` -> `@property` + `op_label` backing field

Currently `op` is stored as a direct ref (`LayerLog | None` per type hint, but post-audit should be `OpLog | None`). Refactor to match the `param.module_address` + `param.module` pattern: stable string identifier as stored field + reference via `@property`.

| Field | Type | Storage | Behavior |
|---|---|---|---|
| `op_label` | str \| None | stored (public) | Stable string identifier of the corresponding forward Op (e.g., `"conv2d_1_1:1"`). Persists across save/load. None when no forward Op exists. |
| `op` | OpLog \| None | `@property` | Resolves `self.trace.ops[self.op_label]` if set, else None. Returns the live OpLog. |
| `has_op` | bool | (existing — line 687) | unchanged. Predicate companion: True when `op_label is not None`. |

Resolves the type hint discrepancy (was `LayerLog | None`, becomes `OpLog | None` via the property's return type). Same architecture as locked `param.module` (returns ModuleLog via @property) and `op.atomic_module` (returns ModuleLog via @property).

## Confirmed locks (no change) — additional

- **GradFnLog `forward_signature` / `forward_docstring` / `backward_signature` / `backward_docstring`** — kept. These describe the underlying autograd Function class's forward and backward methods. Once the Function-class context is understood, the names read fine. The `class_*` prefix proposed earlier was withdrawn per JMT.

## ModuleCallLog / ModuleLog / ParamLog / BufferLog — `is_shared` -> `has_multiple_addresses`

| Was | New | Where |
|---|---|---|
| `is_shared` | `has_multiple_addresses` | ModuleCallLog (line 501), ModuleLog (line 516), ParamLog (line 586), BufferLog (line 626) |

`is_shared` is convention-driven (PyTorch users know "shared module" = weight tying / reused submodules) but ambiguous on first read ("shared with whom? in what sense?"). `has_multiple_addresses` is explicit and pairs cleanly with the existing `all_addresses` field — ask the predicate, then access the data. `has_*` form fits the predicate-for-property convention (vs. `is_*` for role/category).

## LayerLog / ModuleLog / GradFnLog — promote sub-record collections to Accessors

Currently inconsistent: trace-level `ops`/`layers`/`modules`/etc. are proper Accessors, but sub-record holders are described as plain "dict-like collections."

| Was | New | Where |
|---|---|---|
| `LayerLog.ops` (dict-like) | `LayerLog.ops` (scoped `OpAccessor`) | LayerLog (line 488) |
| `ModuleLog.calls` (dict-like) | `ModuleLog.calls` (`ModuleCallAccessor`) | ModuleLog (line 540) |
| `GradFnLog.calls` (dict-like) | `GradFnLog.calls` (`GradFnCallAccessor`) | GradFnLog (line 690) |

User benefits: integer + label + substring + slice + standard `keys()`/`values()`/`items()`/`to_pandas()` methods. Pythonic and consistent across scopes. Implementation: scoped instances (filtered to this Layer's ops / Module's calls / GradFn's calls).

## ModuleCallLog / ModuleLog — output activation + gradient passthrough properties

ModuleCallLog has `forward_args`/`forward_kwargs` (inputs) directly accessible but lacks direct accessors for output tensors and gradients — must hop through `output_layers` to resolve Op refs. Asymmetric.

Add `@property`-based passthrough surface that resolves to the underlying Op's tensors. **No data duplication** — Op remains the single source of truth; ModuleCall provides a convenience handle.

Pattern mirrors LayerLog's existing single-Op passthrough (line 386: *"Saved forward output for a single-Op Layer; raises for multi-Op Layers"*). Singular forms raise for multi-output cases; plural forms always work.

**ModuleCallLog new `@property` surface:**

| Property | Singular | Plural |
|---|---|---|
| Output tensor | `out` | `outs` |
| Output shape | `out_shape` | `out_shapes` |
| Output dtype | `out_dtype` | `out_dtypes` |
| Output memory | `out_memory` (+ `_str`) | `out_memories` (+ `_str`) |
| Transformed output (+ companions: shape/dtype/memory) | `transformed_out` (+ companions) | `transformed_outs` (+ companions) |
| Output gradient | `grad` | `grads` |
| Gradient metadata (shape/dtype/memory + `_str`) | `grad_shape` etc. | `grad_shapes` etc. |
| Transformed gradient (+ companions) | `transformed_grad` (+ companions) | `transformed_grads` (+ companions) |

Implementation: each `@property` resolves via `[self.trace.layers[label].<field> for label in self.output_layers]`. Singular returns `outs[0]` if `len == 1` else raises.

**ModuleLog (aggregate)**: same property surface with single-call passthrough. Single-call modules expose all fields directly; multi-call modules raise on the singular form, requiring `module.calls[N].out`.

Deferred: dict/tuple-shape preservation for multi-output forwards (modules returning `{"hidden": h, "attentions": a}` will give a flat list via `outs`, losing dict keys). For v1, plural-list is enough; revisit if container-shape preservation becomes load-bearing.

Implementation work, not just naming. Vocabulary lockable now.

## ModuleLog — exhaustive hook info via `HookInfo` dataclass

Currently ModuleLog has only `has_forward_hooks` / `has_backward_hooks` (2 fields, predicate-only). PyTorch exposes 6 computation-affecting hook registries; current coverage misses pre-hooks, full-backward variants, counts, names, qualnames, source locations.

Add a `HookInfo` dataclass:

```python
@dataclass
class HookInfo:
    count: int                              # len(registry)
    names: list[str]                        # [h.__name__ for h in registry]
    qualnames: list[str]                    # [f"{h.__module__}.{h.__qualname__}" for h in registry]
    source_locations: list[FuncCallLocation]  # where each hook is defined
    # has_any: derives from count > 0; could be a @property or omitted
```

Then ModuleLog gets 6 nested fields (one per hook registry):

| New field | Source |
|---|---|
| `forward_pre_hook_info` | `module._forward_pre_hooks` |
| `forward_hook_info` | `module._forward_hooks` |
| `backward_pre_hook_info` | `module._backward_pre_hooks` (legacy) |
| `backward_hook_info` | `module._backward_hooks` (legacy) |
| `full_backward_pre_hook_info` | `module._full_backward_pre_hooks` |
| `full_backward_hook_info` | `module._full_backward_hooks` |

Reads at user side:
- `module.forward_hook_info.count`
- `module.forward_hook_info.names`
- `module.full_backward_hook_info.qualnames`

Replaces existing `has_forward_hooks` / `has_backward_hooks` (or those become derived `@property` aliases — `module.has_forward_hooks == bool(module.forward_hook_info.count)`).

**Live hook references** intentionally NOT stored persistently (callable refs aren't picklable, prevent GC of closure state, vanish post-`.tlspec`-load anyway). If runtime access is needed later, can be added as a lazy `@property` reaching through a (future) module ref on ModuleLog.

Implementation work, not just naming. Vocabulary is lockable now.

## Op / Layer / GradFn — index field family rename

The current three indices on Op (`creation_index`, `overall_index`, `op_index`) have mutually confusing names. Each plays a distinct, load-bearing role; renaming for symmetric clarity.

| Was | New | Role | Where used |
|---|---|---|---|
| `creation_index` | `capture_index` | Raw capture-time monotonic counter; may have gaps after orphan removal. | Raw labels (`{type}_{type_index}_{capture_index}_raw`); debug/internal. |
| `overall_index` | `trace_index` | 1-based position in trace's linear order (includes I/O sentinels). Dense after postprocessing. | `trace[N]` integer indexing (via zero-based `tensor_index`); final labels (`conv2d_1_5`); `backward.py:129`; visualization rendering. |
| `op_index` | `compute_index` | 1-based count among compute ops only (input/buffer = 0; output = num_ops). | OpLog `__repr__` line "operation 3/5 of layer foo". |

Why this naming family:
- **Symmetric scope-naming**: each name carries its scope (capture / trace / compute). User reads `op.capture_index` / `op.trace_index` / `op.compute_index` and instantly understands the relation.
- **Avoids `op.op_index` stutter** that elevating "overall" to "op" would have caused.
- **`pass_index` stays unchanged** — separately defended; describes "this Op's invocation-index within the parent Layer," and "pass" is honest as a coordinate field even though it was dropped from class names.

Cascade scope:
- OpLog: all three fields
- LayerLog: also has `overall_index` -> `trace_index`
- GradFnLog: has `overall_index` -> `trace_index` (different scope semantics — across all grad-fns; verify before applying)
- Layer label construction in `postprocess/labeling.py:91-93` continues to use the index in (B), now named `trace_index`
- Raw label construction in `capture/output_tensors.py:962` continues to use the index in (A), now named `capture_index`

Notes:
- Glossary entries to update: `.project-context/torchlens_glossary.md:204-207, 211-212, 367-368, 671`
- Cascading internal renames: `op_log.py`, `layer_log.py`, `grad_fn_log.py`, `model_log.py`, `postprocess/labeling.py`, `capture/output_tensors.py`, `intervention/resolver.py`, `bridge/profiler.py`, `visualization/rendering.py`, etc.

## Trace timing cluster — vocabulary cleanup

| Was | New | Rationale |
|---|---|---|
| `function_calls_duration` (+ `_str`) | `func_calls_duration` (+ `_str`) | Align with `func_*` short form used everywhere else (OpLog `func_duration`, `func_name`, `func_signature`, `func_docstring`, `func_call_id`). Trace-level `function_*` was the outlier. Plurality preserved — at Trace scope it IS a sum across many calls. |
| `duration` (+ `_str`) | `total_duration` (+ `_str`) | Match the `total_*` aggregate convention used pervasively (`total_flops`, `total_macs`, `total_params`, `total_activation_memory`, `total_gradient_memory`). Bare `duration` is ambiguous when cluster contains 5+ phase-specific `*_duration` siblings. The audit's first formulation had `total_duration` (line 833) before switching to bare (line 2651) — earlier name is the right one. |

Notes:
- Glossary entries to update: `.project-context/torchlens_glossary.md:170,174,176,177`
- Audit notes entries to update: `.project-context/notebook_audit_notes.md:830-833,2651-2653`

## Trace capture-config — loop detection

| Was | New | Rationale |
|---|---|---|
| `detect_loops` (field) | `recurrence_detection` | (1) Verb-object form `detect_*` reads as a method, not a flag — JMT flagged the method-vs-flag confusion. Noun form fixes it. (2) "Loops" is too narrow per JMT's clarification: the feature finds layers contiguous with same-param layers AND actual graph cycles. "Recurrence" captures both (loops + non-cyclic structural repetition). (3) Resolves the kwarg/field mismatch — `options.py:140` already aliases `detect_loops` to a new kwarg name; align field too. |
| `detect_recurrent_patterns` (kwarg, in `options.py:140` deprecation map) | `recurrence_detection` | Pair the kwarg with the field name for consistency. |

Notes:
- Glossary entry to update: `.project-context/torchlens_glossary.md:152`
- Audit notes entry to update: `.project-context/notebook_audit_notes.md:783`
- **DO NOT rename `equivalent_ops`.** That field correctly stores intrinsic equivalence (same args/params), distinct from the recurrent (positional) concept which is already represented by the LayerLog abstraction (a Layer IS a package of recurrent ops). See memory: `recurrent_vs_equivalent.md`.

## Trace capture state — vocabulary cleanup

| Scope | Old | New | Rationale |
|---|---|---|---|
| Trace | `operation_history` | `ledger` | "Operation" collides with `Op` / `ops` (captured tensor ops). `ledger` is distinctive, append-only-implied, and matches the audit trail framing. Audit had locked `operation_history` unchanged before the `Op` class name was finalized — collision wasn't visible at audit time. **Alternatives noted:** `event_history` (most explicit), `event_ledger` (explicit + ledger), `change_history` (closer to original `_history` shape). Reconsider if `ledger` ever feels too unfamiliar in user-facing docs. |

Notes:
- Glossary entry to update: `.project-context/torchlens_glossary.md:137`
- Audit notes entry to update: `.project-context/notebook_audit_notes.md:871`

## Revisit later (consolidated — feature/design scope, not naming)

- **`backward_peak_memory`** (and `_str` companion). Glossary lines 71-72. JMT flagged for later pass — not yet riffed.

- **`unlogged_ops`** (glossary line 88). Possibly only meaningful when user opts to log only a subset of ops (e.g., only those with saved activations). Open question: do we want that feature at all? If not, the field can disappear. Re-evaluate during a feature-scope review after the rename pass.

- **Args capture: 3-flag interaction.** `save_arg_values`, `capture_args_template`, and `intervention_ready` overlap unclearly: `intervention_ready=True` already triggers template capture, making `capture_args_template` somewhat orphaned (a niche escape valve for "I want template without full intervention"). Audit line 813 explicitly defers "cache semantics." Consider: (a) drop `capture_args_template` entirely and let `intervention_ready` be the only template trigger; (b) collapse to a single enum `arg_capture: Literal["none", "template", "values", "both"]`; (c) keep all three with clearer interactions documented. Resolve during the deferred cache/intervention design review.

- **`module_filter` rename**. Identified as misnamed (not a module filter, not a true filter — it's a per-op save-decision predicate); current proposal `save_predicate`. JMT requested deferring this in a combined fashion with the `fastlog` namespace question (which has its own deferred TODO from the audit). Resolve together.

- **IO surface**: `to_pandas`, CSV / Parquet exports, and related dataframe-conversion methods. Multiple per-class IO methods exist across Trace/LayerLog/OpLog/ModuleLog/ParamLog/BufferLog/GradFnLog with similar shapes. Likely opportunities for vocabulary unification, signature parity, and possibly a shared mixin or protocol. Revisit in integrated fashion AFTER the field-rename pass — easier to spot the patterns once underlying field names are stable.

- **Accessor superclass refactor**. Currently `LayerAccessor`, `OpAccessor`, `ModuleAccessor`, `ParamAccessor`, `BufferAccessor`, `GradFnAccessor`, `TraceAccessor` are independent classes with parallel shapes. Bundle accessors share a private `_BundleLabelAccessor` base. Time to unify: introduce a shared `Accessor` (concrete or generic `Accessor[T]`) base class providing default implementations of `__getitem__` (int + label + substring), `__contains__`, `__iter__`, `__len__`, `keys`, `values`, `items`, `to_pandas`. Subclasses override only type-specific lookup logic. Side benefits: consistent error messages ("did you mean X?"), single place to evolve the API, easy extension for new accessor types (e.g., the just-penciled scoped `OpAccessor` / `ModuleCallAccessor` / `GradFnCallAccessor`). Structural refactor — separate piece of work from the field/method renames; should land AFTER rename pass to avoid double-churning.

- **`bundle.at()` dispatch logic**. Identified during walkthrough as genuinely tricky — the dispatcher auto-detects label format (Op vs Layer vs Module vs Trace name) and routes to the right accessor. Edge cases around ambiguous labels, fuzzy matching, and intervention-spec resolution need a dedicated design pass. Defer until after the rest of Bundle stabilizes; revisit with concrete use cases.

- **Super\* family — universal via generic `Super[T]`, NOT N hand-written classes.** Revisited 2026-05-05. Lean: option C from the design discussion — one `Super[T]` base class handling alignment machinery (wraps `dict[member_name, T]`, member access, generic compare/delta primitives). Per-kind extensions only where they earn it: SuperOp/SuperLayer get tensor-stacking + aggregation, SuperModule/SuperBuffer/SuperGradFn get type-specific comparison hooks, SuperParam/SuperModuleCall/SuperGradFnCall are type aliases or trivial subclasses (exist for symmetry, generic methods only). Universal rule preserved ("every sub-Trace class has a Super counterpart at Bundle level"); maintenance cost bounded. Accessor superclass refactor naturally extends to Super accessors in the same pass.

- **Bundle aggregate metadata.** Considered 2026-05-05. Add fields where the aggregate is *honest* (each member contributes a distinct quantity), skip where it's a misleading sum (members share structure):
  - **Add (memory aggregates, sum across members):** `total_activation_memory` (+ `_str`), `saved_activation_memory` (+ `_str`), `total_gradient_memory` (+ `_str`), `autograd_saved_memory` (+ `_str`). Honest accumulation; useful for "how much RAM does this bundle cost."
  - **Add (structural-agreement predicates):** `is_structurally_consistent` (all members share `graph_shape_hash`), `shared_op_labels` (intersection across members), `divergent_op_labels` (symmetric difference), `shared_layer_labels` / `divergent_layer_labels`. Tell users whether SuperOp/SuperLayer alignment is meaningful for this bundle.
  - **Skip (misleading sums):** `total_num_params`, `total_num_layers`, `total_param_memory` summed across members — useless for same-model bundles, confusing for different-model bundles. Iterate `bundle.traces` for per-member, use structural-agreement predicates for set-level facts.
  - **Maybe later (timing):** `mean_capture_duration`, `median_capture_duration`, `min/max_capture_duration` — niche; defer unless a use case surfaces.
  - **Defer (comparison helpers):** `aligned_pairs`, `compare`, `delta_map`, `norm_delta`, `output_delta`, `show_diff` — Bundle Phase 8 already tracks these.

- **What torchlens owes consumer abstractions (Sweep / Timeline / Tree / etc.) above Bundle.** Considered 2026-05-05. The ontology closes at Bundle — Sweep / Timeline / Tree / etc. are *consumer-level* abstractions outside torchlens's scope. But torchlens's substrate must support those consumers cleanly. Concrete commitments:
  1. **Bundle structural compatibility checks** — `is_structurally_consistent`, `shared_op_labels`, `divergent_op_labels`, etc. (already logged in the Bundle aggregate-metadata note above). Lets consumers verify alignment readiness before projecting Super[X] across cells/timesteps/nodes.
  2. **Bundle-to-Bundle comparison primitives** — `compare`, `delta_map`, `aligned_pairs`, `norm_delta`, `output_delta`, `show_diff` (Bundle Phase 8 deferred helpers). Lets consumers fold their structural axis into pairwise / N-wise Bundle comparisons.
  3. **Cheap iteration over members** — already in place via `bundle.traces` accessor + `len(bundle)`.
  4. **Stable cross-Bundle Trace alignment** — `trace_name` is the convention; consumers building Sweep/Timeline/Tree can match Traces across Bundles by name. Don't break this contract in future renames.

  Net: torchlens provides the substrate (Trace, Bundle, Super[X], the four points above). Consumers build whatever shape they need on top — Sweep grids, Timelines, LineageTrees, ensemble graphs, custom analytics. The boundary is honest and load-bearing for the ecosystem.

- **Ontology closes at Bundle. No "bundles of bundles."** Considered 2026-05-05. Bulletproof structural argument: (a) Super[X] requires multiplicity-with-alignment-keys; bundles lack cross-bundle keys (they're top-level user-managed, like top-level nn.Modules). (b) Every plausible nested-bundle use case decomposes into flatter abstractions: training trajectories → ordered List[Bundle] keyed by epoch; hyperparameter sweeps → Dict[HP, Bundle]; cross-model comparison → either flat merge or pairwise Bundle aggregate comparison; ensembles → List[Bundle] iteration; hierarchies → trees, not nested bundles. (c) MetaBundle adds zero expressive power — anything it could do, List[Bundle] + iteration already does. (d) PyTorch precedent: `nn.ModuleList` exists, `nn.ModuleListList` does not. (e) Hierarchy theory: Trace = alignment substrate; Bundle = curation/multiplicity substrate; Super[X] = projection across Bundle members; whatever sits above Bundle is *consumer-level abstraction* (Sweep, Timeline, Tree) — research-tooling territory, not torchlens's job. The closure is structural (substrate runs out), not arbitrary.

- **Bundle stays Bundle, NOT SuperTrace.** Considered 2026-05-05. Stronger reason than initially logged: SuperTrace would be a *category error*, not just a style choice. The Super[X] pattern structurally requires (a) multiple containers each holding X, and (b) a shared key for aligning X across them. Sub-Trace objects (Op, Layer, etc.) have both: Bundle members provide the containers, labels provide the keys. Trace itself has neither — there's no convention for Bundle-of-Bundles, and traces have no cross-Bundle keys (trace_name is local to one Bundle). Even within one Bundle, traces aren't aligned; they're just collected by name. So Trace is the substrate where alignment keys are *defined*, not an entity that gets *projected*. Universal Super\* rule applies BELOW Trace; ceiling is structural, not an exception. Keep `Bundle`.

- **Bundle aggregate-Trace-method coverage**. Bundle currently exposes some Trace operations as aggregates (e.g., `rerun(model, x, **kwargs)` runs across all member Traces). Need a systematic pass: which Trace methods should be Bundle-aggregated (apply-to-all)? Which should NOT (per-trace only)? Which need different signatures at Bundle scope? Currently inconsistent — some methods are aggregated, others aren't, no clear principle. Resolve when Bundle UX pass lands.

---

## REFERENCE FORM CONVENTION — when between-object references are labels vs objects (LOCKED 2026-05-19)

Replaces the absolute "cross-class references are always label strings" convention with a nuanced four-principle rule that distinguishes portable refs from runtime-only objects, and adds a `_label` + @property resolver pattern for frequently-accessed refs.

### Four principles

**Principle 1: Portability dictates storage form.**
- Field needs to survive `.tlspec` save/load → stored as **label string**
- Field is runtime-only (dies on save/load anyway) → stored as **object reference**

**Principle 2: Tensors, runtime callables, and runtime handles are object-typed by nature.**
These cannot be labels — they ARE the runtime data:
- Tensor values: `Op.out`, `Op.grad`, `Op.transformed_out`, `Op.transformed_grad`
- Runtime callables: `Op.func`
- Autograd handles: `Op.grad_fn`
- Back-pointers: `Op.trace`, `Op.source_trace` (typically weakref to avoid cycles)
- Live model refs: `Trace._source_model_ref`

**Principle 3: Frequently-resolved cross-class refs get a `_label` (storage) + bare-name (@property) resolver.**
Pattern:
- Storage: `<entity>_label: str` — what gets serialized
- Public access: `<entity>: <ClassType>` — @property that does `self.trace[self.<entity>_label]`

Resolver raises if the label cannot be resolved (e.g., Trace cleaned up, label missing). Silent None is bad — it invites bugs downstream.

`@property` lookup cost is negligible (a couple of dict lookups). No need for `functools.cached_property`.

**Principle 4: Uncommon or collection-valued refs stay as label strings (no @property resolver).**
- Sets/lists of labels: `Op.equivalent_ops`, `Op.module_calls_entered`, `Op.parents`, `Op.children` — users iterate via `[trace[lbl] for lbl in ...]` when needed
- Rare-access scalar refs: keep user explicit with `trace[label]`
- Plural resolved-collection properties (e.g., `Op.parent_layers` returning list of Layer objects) are NOT added. Bloats the surface for marginal ergonomics.

### Threshold for "frequently-accessed"

A ref qualifies for the `_label` + @property treatment if it shows up in README examples, tutorial notebooks, or is dotted-accessed >3x in any user example. Otherwise stays as a bare label string.

Going-with-gut on the borderline cases is fine; not worth a quantitative audit.

### Categorization table

| Reference kind | Storage | Public access | Example |
|---|---|---|---|
| Cross-class ref, common access | label string | `_label` storage + bare-name @property resolver | `GradFn.op_label` + `GradFn.op` |
| Cross-class ref, uncommon | label string | label string only | `Op.equivalent_ops` (set of labels) |
| Cross-class collection | label list/set | label list/set | `Op.parents`, `Op.children` |
| Cross-class collection (high-traffic accessor) | Accessor | Accessor returning resolved objects | `Trace.layers`, `Module.calls` |
| Tensor value | object | object | `Op.out` |
| Runtime callable/handle | object | object | `Op.func`, `Op.grad_fn` |
| Back-pointer | object (weakref) | object | `Op.trace` |

### Naming convention for the pair

When a ref gets the `_label` + @property treatment:
- `<entity>_label`: string storage, always portable
- `<entity>`: object resolver (@property); raises on resolution failure

Examples (some existing, some candidates surfaced during sweep — see glossary v2 audit notes):
- `GradFn.op_label` / `GradFn.op` ✓ already in deltas
- `Op.layer_label` / `Op.layer` — CANDIDATE (most common cross-class access)
- `Param.module_address` / `Param.module` ✓ already a @property
- `Buffer.parent_label` / `Buffer.parent` — CANDIDATE
- `Op.parent_module_call_label` / `Op.parent_module_call` — CANDIDATE

### Replacement text for the v2 glossary Conventions section

Current text (line 55-58 of `<vault>/2026-05-18-rename-sprint-glossary-v2/torchlens_glossary.md`):
> Any field on one log class that points to another log class stores the target's LABEL (a string), not the object. To get the object: `trace[label]`.

Proposed replacement:
> **Cross-class references.** Cross-class references that need to survive `.tlspec` save/load are stored as label strings. Frequently-accessed refs additionally expose a `<entity>` @property that resolves the label via `trace[label]` for ergonomic dotted access; the resolver raises if the label cannot be resolved. Runtime-only values (tensors, callables, autograd handles, back-pointers to the owning Trace) are stored as direct object references and do not survive portable save/load.

### Rationale

- **Absolute "always labels" rule was wrong** — tensors, callables, runtime handles, and back-pointers have always been exceptions and need explicit acknowledgement
- **Ergonomic resolver pattern** is already in use (`GradFn.op_label` / `GradFn.op`); formalizing the pattern means it can be applied consistently
- **Portability is the actual invariant**, not "everything must be a label." Labels are the form that survives serialization; runtime objects don't pretend to.
- **Collection refs stay as labels** — bloating the API with parallel resolved-collection properties hurts more than helps; explicit `trace[lbl]` in user code is fine


---

## Trace.equivalent_ops → Trace.op_equivalence_classes (LOCKED 2026-05-21)

Renames the Trace-scope field for the dict-mapping-class-id-to-Op-labels. Op and Layer scope keep `equivalent_ops` (members of the equivalence class) because they have the right structure/semantic there. Only Trace scope had the same-name-different-structure mismatch (Trace returns dict; Op/Layer return list).

| Scope | Field | Returns | Semantic |
|---|---|---|---|
| `Op.equivalence_class` | (unchanged) | class id (str/int) | "what class does this Op belong to?" |
| `Op.equivalent_ops` | (unchanged) | list of Op labels | "other Ops in my class" |
| `Layer.equivalent_ops` | (unchanged) | list of Op labels | "Ops in my Layer's equivalence group" |
| `Trace.equivalent_ops` → `Trace.op_equivalence_classes` | RENAMED | dict[class_id → set of Op labels] | "all equivalence classes in this trace" |

Rationale: prefix-qualification convention at Trace scope for Op-related collections (`Trace.num_ops`, `Trace.op_labels`, `Trace.ops`, `Trace.flops_by_op_type`). `op_equivalence_classes` slots into that pattern. The "classes" plural + Op qualifier disambiguates from the member-list usage at lower scopes.

---

## Universal accessor on Trace (LOCKED 2026-05-21)

`trace[key]` becomes a universal lookup that resolves to any TorchLens object by label, with a deterministic resolution order.

**String key resolution order (try-each, first match wins, `KeyError` if no match):**
1. `trace.ops` — Op label (stylized format with `:N` pass index)
2. `trace.module_calls` — ModuleCall label (`<address>:<callidx>`)
3. `trace.layers` — bare Layer label
4. `trace.modules` — bare PyTorch Module address
5. `trace.params` — Param address
6. `trace.buffers` — Buffer address
7. `trace.grad_fns` — GradFn label
8. Alternate lookup keys (e.g., `op.fx_label` registrations)

**Integer key:** `trace[N]` returns the Op at `ordinal_index == N` (0-based). Unchanged from current behavior. Numeric indexing is Op-specific by convention; explicit accessors required for other classes (`trace.modules[N]`, etc.).

**KeyError message:** enumerates all searched namespaces; fuzzy-suggest neighbors via existing `_lookup_keys.py` machinery extended across all accessors.

### Why this works

Label namespaces are structurally disjoint:
- Op labels: stylized `<type>_<idx>_<idx>:N`
- Module addresses: dotted PyTorch paths
- Param/Buffer addresses: dotted, scoped under module
- GradFn labels: autograd-style names

In practice almost no collision; resolution order handles theoretical edge cases (e.g., a module named to look like an op-type token).

### When to use universal vs explicit

- **Universal `trace[X]`:** casual / REPL / exploratory work. Return type widens to `Layer | Op | Module | ModuleCall | Param | Buffer | GradFn`.
- **Explicit `trace.<class>[X]`:** production code, type-checked code, when return type narrowing matters.

### Type-strict explicit accessors (also locked here)

Existing v3 behavior of `trace.layers["conv2d_1_2:1"]` returning the Op "for convenience" is REMOVED. Explicit accessors are type-strict:
- `trace.layers[X]` → ALWAYS returns Layer (strip `:N` if present)
- `trace.ops[X]` → ALWAYS returns Op (single-pass passthrough for bare Layer labels when Layer has one Op; raises if multi-pass and no `:N`)
- Same rule across all explicit accessors

Universal `trace[X]` covers the cross-type convenience use case.

---

## Filter Accessor consistency harmonization (LOCKED 2026-05-21)

Some Trace-level "filtered subset" fields were documented inconsistently in v3: some as Accessors, some as label/address lists. Harmonized: anything that filters a known class to a subset is an Accessor (consistent with `saved_X` family).

Promoted to Accessor in v3:
- `Trace.input_layers`
- `Trace.output_layers`
- `Trace.buffer_layers`
- `Trace.internal_source_ops`
- `Trace.internal_sink_ops`
- `Trace.layers_with_params`

Stays as label/address list (correct as-is):
- `Trace.layer_labels`, `Trace.op_labels` — by-design label-only lists, used for iteration/order
- `Trace.uncalled_modules` — diagnostic address list (Modules not in this capture; users iterate to check)

Companion `_labels` versions of the promoted Accessors can be added later if explicit label lists are wanted (defer; not added now to keep API surface from bloating).


---

## `Trace.last_run_ctx` → `Trace.last_run` (LOCKED 2026-05-21)

Renames the inspectable summary dict for the most recent replay/rerun/append operation.

**Rationale:**
- `_ctx` is the abbreviation for "context"; TorchLens uses BOTH `context` (full word in `code_context`, `save_code_context`) and `ctx` (abbreviation in `last_run_ctx`, `run_ctx` callback args). Unifying public-API fields to the full word; `ctx` stays only in callback argument names per the autograd `ctx.save_for_backward()` convention.
- "Context" was also a slightly off word for what this field stores — it's a SUMMARY of the last operation (engine, timing, flags, results), not surrounding context. Bare `last_run` is clearer and removes the misleading suffix.

**Naming convention reinforced:**
- Field surfaces: full word `context` (e.g., `code_context`)
- Callback argument names: `ctx` allowed (autograd idiom)

**Pending review (parked):** This field is likely redundant with `state_history[-1]`. See todos for follow-up.


---

## `Trace.start_time` / `end_time` → `capture_start_time` / `capture_end_time` (LOCKED 2026-05-21)

Renames the bookend capture timestamps for consistency with the `capture_duration` family.

| Was | New |
|---|---|
| `Trace.start_time` | `Trace.capture_start_time` |
| `Trace.end_time` | `Trace.capture_end_time` |
| `Trace.start_time_str` | `Trace.capture_start_time_str` |
| `Trace.end_time_str` | `Trace.capture_end_time_str` |

**Rationale:**
- Existing `capture_duration` / `setup_duration` / `forward_duration` / `func_calls_duration` / `cleanup_duration` / `overhead_duration` cluster uses phase prefixes
- Bare `start_time` / `end_time` were inconsistent with the cluster and ambiguous (capture start? forward start? replay start?)
- After rename, all capture-phase time fields carry the `capture_` namespace; sub-phase fields (setup, forward, etc.) use bare phase names
- Recurring operation timings (replay, rerun, backward) live in `state_history` per-event, not at top-level Trace scope

The `capture_` prefix on top-level fields reinforces the clean split: "capture phase = top-level Trace fields with `capture_` prefix; replay/rerun/backward phases = per-event in `state_history` + summary in `last_run`."

---

## Boundary-type Accessor symmetry on Trace (LOCKED 2026-05-21)

Symmetry-fillers added across boundary types. Each boundary type now has BOTH a `_layers` Accessor (grouped by equivalence-class graph position) AND a `_ops` Accessor (flat across all passes), to handle the multi-Op-per-Layer case when boundaries partake in loops.

| Boundary type | Existing | Added |
|---|---|---|
| Input | `input_layers` | `input_ops` |
| Output | `output_layers` | `output_ops` |
| Internal source | `internal_source_ops` | `internal_source_layers` |
| Internal sink | `internal_sink_ops` | `internal_sink_layers` |
| Compute | `compute_layers`, `compute_ops` | (already both) |
| Buffer | `buffer_layers` | `buffer_source_ops`, `buffer_sink_ops` deferred to Buffer Option B post-2.0 |

Plus corresponding count fields:
- `Trace.num_input_ops`, `Trace.num_output_ops` (new)
- `Trace.num_internal_source_layers`, `Trace.num_internal_sink_layers` (new)

**Layer-level Op-predicate delegation:** Layer-scope predicates (`is_input`, `is_output`, `is_internal_source`, `is_internal_sink`, `is_buffer_source`) delegate from the representative Op, same pattern as `Layer.is_compute_layer`.

**Optional convenience unions (NOT added now):** `Trace.boundary_layers` / `Trace.boundary_ops` for "all non-compute X" — defer until user demand surfaces.

---

## `Trace.backward_root_grad_fn_id` → `Trace.backward_root_grad_fn_ids` (plural) (LOCKED 2026-05-21)

Renames to plural list; adds `last_backward_root_grad_fn_id` (`@property`) for convenience.

**Rationale:**
- Singular form was wrong for multi-backward-pass scenarios. `Trace.backward(loss1)` then `Trace.backward(loss2)` produces two distinct root grad-fn ids; the singular field could only hold one.
- Pairs naturally with `Trace.num_backward_passes` (already plural).

**Surface:**
- `Trace.backward_root_grad_fn_ids: list[int]` — one per backward pass, in execution order
- `Trace.last_backward_root_grad_fn_id` (`@property`) — most recent, or `None` if no backward passes

**Companion sanity check:** `has_backward_pass` (singular boolean — "any backward done") and `backward_memory_backend` (backend identifier, singular by nature) remain singular; they're correct as-is.

**Future improvement (deferred):** Promote backward passes to first-class `BackwardPass` records, parallel to `ModuleCall` / `GradFnCall`. Would expose richer per-backward metadata (loss value, duration, status). See todos for follow-up; do not implement now (no user demand established yet).


---

## `Op.grad_fn` swap — TL record primary, runtime handle moves to `grad_fn_handle` (LOCKED 2026-05-21)

Resolves the `_log` exception locked in v3 by swapping the bare-name to TL record (consistent with `op.module`, `op.params`, `op.layer`) and moving the PyTorch runtime handle to a qualified name.

| Was | New |
|---|---|
| `Op.grad_fn` (runtime PyTorch handle) | `Op.grad_fn_handle` |
| `Op.grad_fn_log_label` (storage) | `Op.grad_fn_label` |
| `Op.grad_fn_log` (`@property` TL record) | `Op.grad_fn` (`@property` TL record) |

**Rationale:**
- `op.module` returns TL Module record; `op.params` returns TL Param records; `op.layer` returns TL Layer record. Bare `op.grad_fn` returning TL GradFn record fits the established pattern.
- The `_log` suffix exception (originally accepted as a documented quirk because `op.grad_fn` was taken by the PyTorch handle) is now eliminated.
- Op-level shortcut fields (`grad_fn_class_name`, `grad_fn_cls`, `grad_fn_id`, `grad_fn_class_qualname`) stay — they're ergonomic flat accessors. They become `@property` views derived from `op.grad_fn.class_name` etc. (or stay as denormalized storage; implementation detail).
- Migration cost: users doing `op.grad_fn.next_functions` (low-level autograd) now use `op.grad_fn_handle.next_functions`. Bounded; advanced users.

The same swap also extends through any Layer-level passthrough (`Layer.grad_fn`, `Layer.grad_fn_handle`).

---

## `autograd_saved_memory` → `autograd_memory` propagation across all scopes (LOCKED 2026-05-21)

The earlier Trace-level decision (`total_autograd_saved_memory` → `total_autograd_memory`) propagates to Op and Layer scopes.

| Scope | Was | New |
|---|---|---|
| Trace | `total_autograd_saved_memory` (+ `_str`) | `total_autograd_memory` (+ `_str`) |
| Op | `autograd_saved_memory` (+ `_str`) | `autograd_memory` (+ `_str`) |
| Op | `num_autograd_saved_tensors` | `num_autograd_tensors` |
| Layer | `autograd_saved_memory` (+ `_str`) | `autograd_memory` (+ `_str`) |
| Layer | `num_autograd_saved_tensors` | `num_autograd_tensors` |
| Layer | (new) | `total_autograd_memory` (+ `_str`) — sum across this Layer's Ops |

**Rationale:** Same as the Trace-level decision (locked 2026-05-21 earlier). The "saved" prefix overloaded with TorchLens's own "saved" semantics (TL-saved activations / gradients). Since autograd-saved tensors are the ONLY autograd memory measured (no competing "unsaved autograd memory" category), dropping "saved" loses nothing and removes the overloading. Docstrings carry the precision: "PyTorch's 'saved tensors for backward'."


---

## Param-count parity across classes (LOCKED 2026-05-21)

Filled two gaps in the param-count field family so every class with associated params has the full six-field surface:

- `num_params`, `num_params_trainable`, `num_params_frozen` (scalar counts)
- `num_param_tensors`, `num_param_tensors_trainable`, `num_param_tensors_frozen` (tensor counts)

| Class | Before | After |
|---|---|---|
| Trace | ✓ all 6 | ✓ |
| Layer | ✓ all 6 | ✓ |
| Module | ✓ all 6 | ✓ |
| **Op** | 4 of 6 (missing tensor trainable/frozen) | ✓ all 6 — added `num_param_tensors_trainable` and `num_param_tensors_frozen` (`@property`) |
| **ModuleCall** | 0 of 6 | ✓ all 6 — added as `@property` delegations from `self.module` (params are static during forward; the call's params ARE the Module's) |

Classes without associated params (Buffer, Param, GradFn, GradFnCall) correctly have no param-count fields. Param itself has `num_params` meaning "number of scalars in this single param tensor" (= `numel()`); the trainable/frozen distinction is a single bool (`Param.is_trainable`), so no aggregate counts apply.


---

## Op classification axes — compute vs boundary vs connectivity flags (LOCKED 2026-05-21)

Clarifies a conceptual inaccuracy in v3 where `is_compute_op` was defined as "not (input, output, internal source, internal sink, or buffer source)." Internal source / internal sink / orphan are ORTHOGONAL connectivity descriptors, NOT alternative Op kinds.

### Two orthogonal axes

**Axis 1: Op kind (mutually exclusive)**

| Op kind | When | Predicate |
|---|---|---|
| Compute Op | A torch function was executed (`conv2d`, `relu`, `torch.ones`, `torch.rand`, ANY torch call) | `is_compute_op = True` |
| Input boundary | Model input enters graph | `is_input = True` |
| Output boundary | Model output leaves graph | `is_output = True` |
| Buffer source | Buffer value enters graph | `is_buffer_source = True` |

These four are MUTUALLY EXCLUSIVE.

**Axis 2: Connectivity descriptors (orthogonal flags on compute Ops)**

| Flag | Meaning | Common case |
|---|---|---|
| `is_internal_source` | Compute Op with no external input ancestor | `torch.ones`, `torch.rand`, factory functions |
| `is_internal_sink` | Compute Op whose output isn't used by model output | Disconnected subgraphs |
| `is_orphan` | Compute Op disconnected from both input AND output flow | Truly isolated |

These are NOT alternative Op kinds — they're additional flags on compute Ops. A `torch.ones` Op has BOTH `is_compute_op = True` AND `is_internal_source = True`.

### What changed in v3

Corrected:
- `Op.is_compute_op` definition (now `not (is_input or is_output or is_buffer_source)`; explicitly does NOT exclude internal_source/sink/orphan)
- `Layer.is_compute_layer` definition (same correction)
- `Trace.compute_ops` accessor — now includes internal-source / internal-sink / orphan compute Ops
- `Trace.compute_layers` accessor — same
- `Trace.num_compute_ops` description — clarified inclusion
- `Trace.num_compute_layers` description — clarified inclusion
- Op intro section — replaced "internal source/sink sentinels" framing with "compute Ops with connectivity descriptors"
- Top-level "Each graph event is its own Op" section — rewritten with two-axis table

### Why this matters

A factory function like `torch.ones(shape, device=device)` produces a tensor without taking another graph tensor as input. It's STILL a compute Op (it executes a torch function). Treating it as "not a compute Op" was structurally wrong — the field's purpose is "did a torch function run here?" not "is this node connected to the main flow?"

### Implementation note

If the current code implements `is_compute_op` as the wrong (broader exclusion) form, this is a behavioral correction, not just a doc clarification. Verify on rename-sprint implementation pass.


---

## Dedup: drop `module_calls_entered` (LOCKED 2026-05-21)

Two fields described the same event on both Op and Layer scopes:
- `module_calls_entered`: ModuleCall labels where this Op/Layer's output ENTERED
- `input_to_module_calls`: ModuleCall labels for calls this Op/Layer INPUTS TO

Both stored ModuleCall labels for the same event ("Op's output became an input to module call X"). Redundant.

**Resolution:** drop `module_calls_entered` on both Op and Layer; keep `input_to_module_calls` (which pairs symmetrically with `output_of_module_calls`).

**Companion `module_entry_arg_keys` survives** — it's the detail field keyed by ModuleCall label from `input_to_module_calls`, mapping to arg positions this Op filled at the entry event. "Module entry" is still a well-defined concept (the event of this Op being used as an arg in a module's forward call); the field name accurately describes the per-event arg-position detail.

Cluster after dedup:
- `output_of_module_calls` — relationship (Op is output of these calls)
- `input_to_module_calls` — relationship (Op is input to these calls)
- `module_entry_arg_keys` — detail (which arg positions, keyed by the ModuleCall label from `input_to_module_calls`)


---

## Scoped accessor integer indexing: harmonize to 0-based (LOCKED 2026-05-21)

Resolves an inconsistency where Trace-scope accessors used 0-based ordinal indexing but scoped accessors (`Layer.ops`, `Module.calls`, `GradFn.calls`) used 1-based pass/call-index keying.

**Locked rule:** ALL accessors — Trace-scope AND scoped — use 0-based positional integer indexing, matching Python list/sequence idiom.

| Accessor | Was | New |
|---|---|---|
| `trace.ops[N]` (and other Trace-scope) | 0-based ordinal | 0-based ordinal (unchanged) |
| `Layer.ops[N]` | **1-based pass_index** | **0-based positional** |
| `Module.calls[N]` | 1-based call_index (likely) | 0-based positional |
| `GradFn.calls[N]` | 1-based call_index (likely) | 0-based positional |

**User-facing behavior:**
- `layer.ops[0]` = first pass; `layer.ops[-1]` = last pass; negative indexing works
- `len(layer.ops)` = number of passes
- `for op in layer.ops` = iterate in pass order
- All standard Python list-like operations

**The 1-based pass/call/type indices remain as FIELDS on the records:**
- `op.pass_index` (1-based; part of label format `conv2d_1_5:2`)
- `op.type_index` (1-based; middle of label format)
- `op.step_index` (1-based for compute Ops; 0 for boundaries)
- `module_call.call_index` (1-based; part of label format)
- `grad_fn_call.call_index` (1-based)

The label format encodes 1-based semantic indices (because users / labels naturally read "pass 1" not "pass 0"). The Accessor integer-indexing API uses 0-based positional (because users treating the Accessor like a Python list naturally expect 0-based). The two coexist by accessing different surfaces — labels carry semantic 1-based, Accessors carry positional 0-based.

---

## Label-based accessor lookup accepts short and long Layer-label forms (LOCKED 2026-05-21)

`layer.ops["conv2d_2:1"]` (short form, omits step_index) and `layer.ops["conv2d_2_3:1"]` (long form, includes step_index) resolve to the same Op when both refer to the same Op. The short and long Layer-label forms are equivalent identifiers; both should be accepted by all label-based Accessor lookups.

Applies to:
- `trace.layers[<layer_label>]` — both short and long forms
- `trace.ops[<pass-qualified_label>]` — both `conv2d_2:1` and `conv2d_2_3:1` forms
- `Layer.ops[<pass-qualified_label>]` — same
- Universal `trace[...]` lookup — same

Implementation: lookup matches against both short-form and long-form indices; resolves to the unique Op when unambiguous; raises with disambiguation guidance if ambiguous (rare).


---

## `Module.call_children_addresses` → `Module.call_children` (LOCKED 2026-05-21)

Drops the `_addresses` qualifier on the plural. The qualifier only earns its keep when disambiguating against a bare-name resolver companion; per Principle 4, plurals don't get resolvers, so no disambiguation is needed.

**Rule reinforced:** the `_label` / `_address` qualifier on storage names is for singular refs that pair with bare-name `@property` resolvers (Principle 3). Plurals (Principle 4) take the bare name directly because there's no resolver to displace them.

**Module Hierarchy cluster after fix:**

| Field | Form | Why |
|---|---|---|
| `address_parent` | bare label string | No resolver (not commonly accessed) |
| `address_children` | bare label list | Principle 4 |
| `call_parent_address` | qualified label string | Qualified because bare `call_parent` is taken by the resolver |
| `call_parent` (`@property`) | resolver | Principle 3 — frequently accessed singular |
| `call_children` | bare label list | Principle 4 — no resolver, no qualifier |

All five fields follow the convention.

**Same pattern applies anywhere else** that had qualified-plural-without-resolver. None spotted on a quick audit (most label-plurals like `Op.parents`, `Trace.layer_labels` already use bare names; the `_labels` suffix on `Trace.layer_labels`/`op_labels` is informative because it disambiguates against the `Trace.layers` / `Trace.ops` Accessors).


---

## Principle 4 REVISION: paired plural resolvers (LOCKED 2026-05-21)

JMT flagged a real footgun in the original "blanket no plural resolvers" rule. When a singular ref has a `@property` resolver (Principle 3 applies — commonly accessed) but its paired plural is bare label list (Principle 4 applies), the asymmetry causes return-type confusion:

```python
parent = module.call_parent      # Module object
for child in module.call_children:
    child.address                # AttributeError — child is a STRING
```

The user's mental model of "parent and children are the same kind of thing" breaks at the API surface.

**Revised Principle 4:**
> Uncommon or collection-valued refs stay as label strings (no `@property` resolver). **Exception:** when a plural is the natural pair of a singular resolver in the same conceptual cluster (parent/children, source/sources, owner/owners), the plural ALSO gets a resolver. Return-type consistency across a paired cluster outweighs the general "no plural resolvers" rule.

**Storage naming for paired plural resolvers:**
- Singular: `<entity>_address` (or `_label`) storage + `<entity>` resolver
- Plural: `<entity>s_addresses` (or `_labels`) storage + `<entity>s` resolver

The plural-storage uses the doubled-up suffix (`<entity>s_addresses` for "addresses of multiple entities") to disambiguate against the bare plural resolver name.

### Application across v3

| Cluster | Singular | Plural (before) | Plural (after) |
|---|---|---|---|
| Module call tree | `call_parent_address` + `call_parent` (resolver) | `call_children` (bare label list) — WAS, then revised to `call_children_addresses` (without resolver) — STILL ASYMMETRIC | `call_children_addresses` (storage) + `call_children` (resolver → list of Modules) |
| ModuleCall call tree | `call_parent_label` + `call_parent` (resolver) | `call_children` (bare label list) | `call_children_labels` (storage) + `call_children` (resolver → list of ModuleCalls) |
| Param owners | `module_address` + `module` (resolver) | `all_module_addresses` (bare label list) — PREVIOUSLY REMOVED `Param.modules` per old Principle 4 | `all_module_addresses` (kept as storage) + `modules` (resolver → list of Modules) — REINSTATED |

### Where the revised principle does NOT apply

Paired refs where the singular ALSO doesn't have a resolver — both stay as bare labels:
- `Module.address_parent` / `Module.address_children` (static address tree; both bare label, neither resolver — symmetric without resolvers)
- `Op.parents` / `Op.children` etc. (graph relations — no singular "parent" resolver concept; the singular form isn't separately exposed)

The principle revision only matters when the singular HAS a resolver and the plural would otherwise be the orphan.

### Other paired clusters worth auditing

Should check during implementation:
- `Op.input_to_module_calls` / `Op.output_of_module_calls` — both currently label lists; should both stay or both become resolvers? They're plurals on each side (no singular resolver pair). Stay as label lists per default Principle 4.
- Anywhere else a singular `_address` + resolver pattern coexists with a plural bare-label form — re-audit for the same asymmetry.


---

## ModuleCall Identity rename + add `module` resolver (LOCKED 2026-05-21)

The `ModuleCall.address` field was misnamed — it stored the MODULE's address (cross-class ref), not the ModuleCall's own identity (which is `call_label`). Renamed for clarity; parallels the Param ownership pattern.

**Renames:**
| Was | New | Reason |
|---|---|---|
| `ModuleCall.address` | `ModuleCall.module_address` | The field is a cross-class ref to the called Module's address, not the ModuleCall's own identity |
| `ModuleCall.name` (@property) | `ModuleCall.module_name` (@property) | Last segment of the new `module_address`; parallels `Param.module_name` |
| `ModuleCall.all_addresses` | `ModuleCall.all_module_addresses` | Cross-class ref to all addresses of the called Module |
| `ModuleCall.has_multiple_addresses` | `ModuleCall.module_has_multiple_addresses` (`@property`) | Disambiguate that the predicate refers to the called Module, not the ModuleCall itself |

**Additions (new resolver fields):**
| Field | Form | Purpose |
|---|---|---|
| `ModuleCall.module` | `@property` → Module | Resolver for the called Module record. Closes the gap — ModuleCall now has explicit access to its Module (parallels `Param.module`). |
| `ModuleCall.modules` | `@property` → list of Modules | Paired plural resolver for shared-module case (parallels `Param.modules`). |
| `ModuleCall.module_cls` | `@property` → runtime type | Parallels `Param.module_cls`. |

**Rationale:**
- ModuleCall and Module are tightly coupled (a call is OF a Module); the resolver should be first-class
- Parallels the established Param → Module ownership pattern
- The previous `address` / `name` / `all_addresses` / `has_multiple_addresses` names were ambiguous because they're all about the called Module, not the ModuleCall itself
- The ModuleCall's OWN identity surface is `call_label`, `call_index`, `ordinal_index` — these stay unchanged


---

## Module hook fields: aggregate HookInfo → `list[HookInfo]` (LOCKED 2026-05-21)

Original v3 design stored ONE aggregate `HookInfo` per registry with parallel arrays (`names`, `qualnames`, `source_locations`). Parallel-arrays anti-pattern. Restructure to `list[HookInfo]` where each `HookInfo` is singular per hook.

### Renames on Module

| Was (aggregate HookInfo) | New (list[HookInfo]) |
|---|---|
| `forward_pre_hook_info` | `forward_pre_hooks` |
| `forward_hook_info` | `forward_hooks` |
| `backward_pre_hook_info` | `backward_pre_hooks` |
| `backward_hook_info` | `backward_hooks` |
| `full_backward_pre_hook_info` | `full_backward_pre_hooks` |
| `full_backward_hook_info` | `full_backward_hooks` |

### `HookInfo` dataclass restructure

| Was (aggregate) | New (per-hook singular) |
|---|---|
| `count: int` | REMOVED (use `len(module.forward_hooks)`) |
| `names: list[str]` | `name: str` |
| `qualnames: list[str]` | `qualname: str` |
| `source_locations: list[FuncCallLocation]` | `source_location: FuncCallLocation` |

Each `HookInfo` now describes ONE registered hook. The Module field is a list of these.

### Predicates stay (delegate to len())

- `Module.has_forward_hooks` (`@property`): `len(forward_pre_hooks) + len(forward_hooks) > 0`
- `Module.has_backward_hooks` (`@property`): any of the 4 backward-family lists non-empty

### Rationale

- Parallel-arrays in a single dataclass is an anti-pattern in Python (forced `zip(...)` over aligned arrays)
- Per-hook iteration becomes Pythonic: `for hook in module.forward_hooks: print(hook.name)`
- Each HookInfo can be extended with per-hook fields later (priority, enabled flag, registered_at, etc.) without surface changes
- `len()` replaces the aggregate `count` field naturally
- PyTorch's 6 registry split (2 forward + 4 backward legacy/full variants) is preserved

### Optional companion fields (not adding)

`Module.num_forward_hooks`, `num_backward_hooks` etc. as `@property` from `len(...)`. Skipping for now — users call `len(module.forward_hooks)` directly. Add if summary-table use cases demand.


---

## Argument-handling parity + duration parity on ModuleCall / Module (LOCKED 2026-05-21)

### ModuleCall — argument-handling parity additions

Bring ModuleCall's argument-handling surface up to parity with Op's. The `forward_` prefix stays (semantically meaningful — distinguishes module forward() args from inner torch-op args).

| New field on ModuleCall | Form | Purpose |
|---|---|---|
| `forward_arg_names` | list[str] | Forward arg names (parity with `Op.arg_names`) |
| `num_forward_args_total` | int | Total positional + keyword args (parity with `Op.num_args_total`) |
| `num_forward_pos_args` | int | Positional arg count |
| `num_forward_kwargs` | int | Keyword arg count |
| `has_saved_forward_args` | `@property` bool | Whether forward_args was captured (parity with `Op.has_saved_args`) |

### Op / Layer — summary additions

| New field on Op | Form |
|---|---|
| `args_summary` | str — human-readable summary of positional args |
| `kwargs_summary` | str — human-readable summary of keyword args |

Layer adds passthrough via single-Op delegation (raises for multi-Op Layers).

### Justified asymmetries (NOT filling)

- **Templates** (`args_template` / `kwargs_template`): Op/Layer only. Replay is op-level; module-level "replay" is `tl.trace()` re-execution. ModuleCall doesn't need templates.
- **Non-tensor split** (`non_tensor_pos_args`, `non_tensor_kwargs`): Op/Layer only. Module forward args are usually mostly tensors; the non-tensor split is more useful at op level where there are many scalar/flag args.

### ModuleCall + Module — timing fields (NEW)

ModuleCall and Module had ZERO duration coverage. Add inclusive timing for profiling support.

| New field | Class | Form |
|---|---|---|
| `func_calls_duration` | ModuleCall | `@property` sum of `op.func_duration` for ops in this call (inclusive — includes nested module calls) |
| `func_calls_duration_str` | ModuleCall | `@property` human-readable |
| `func_calls_duration` | Module | `@property` single-call passthrough; raises for multi-call Modules |
| `func_calls_duration_str` | Module | `@property` human-readable |
| `total_func_calls_duration` | Module | `@property` sum across all calls of this Module |
| `total_func_calls_duration_str` | Module | `@property` human-readable |

**Naming rationale:** Op/Layer use singular `func_duration` (one function call per record). ModuleCall/Module/Trace use plural `func_calls_duration` (many function calls per record). Matches the existing Trace-level naming.

### What's NOT being added now (defer pending demand)

- **Exclusive ("self") timing.** `self_func_calls_duration` on ModuleCall/Module — time in THIS module's own ops, excluding nested module calls. Useful for profiling but adds implementation complexity. Add later if needed.
- **Distribution stats.** `mean_`, `median_`, `max_func_calls_duration` on Module. Nice-to-have for profiling outliers; defer until use case surfaces.
- **Overhead attribution.** Per-module torchlens overhead. Trace-level already has `overhead_duration`; per-module breakdown is hard to attribute cleanly. Skip.


---

## ModuleCall / GradFnCall ordinal_index disambiguation (LOCKED 2026-05-21)

The v3 entries for `ModuleCall.ordinal_index` and `GradFnCall.ordinal_index` conflated two different positions:

> "0-based position in `module.calls` (and `trace.module_calls`)"

But `module.calls` (scoped within one Module) and `trace.module_calls` (global across all ModuleCalls in the trace) are different accessors with different positions. Same conflation on GradFnCall.

### Fixed: ordinal_index is TRACE-LEVEL position only

For per-event records (ModuleCall, GradFnCall, Op):
- **`<event>_index`** (1-based) — semantic index within parent scope (e.g., 3rd call of this Module)
- **`ordinal_index`** (0-based) — global position in the trace-level accessor

The scoped 0-based position within the parent is derivable from `<event>_index - 1` (since scoped accessors are now 0-based positional per the earlier harmonization lock).

### Universal pattern across per-event records

| Class | Scoped semantic index | Trace-level ordinal |
|---|---|---|
| Op | `pass_index` (1-based within Layer) | `ordinal_index` (0-based within `trace.ops`) |
| ModuleCall | `call_index` (1-based within Module) | `ordinal_index` (0-based within `trace.module_calls`) |
| GradFnCall | `call_index` (1-based within GradFn) | `ordinal_index` (0-based within `trace.grad_fn_calls`) |

For aggregate records (Layer, Module, Param, Buffer, GradFn), `ordinal_index` is unambiguous — position in `trace.<class>` (no nested scoped accessor inside the same class).

### Round-trip invariants

All three per-event records support:
- `trace.<accessor>[record.ordinal_index] is record` (trace-level round-trip)
- `parent.<accessor>[record.<event>_index - 1] is record` (scoped round-trip, after harmonized 0-based scoped indexing)


---

## `GradFnCall.duration` → `backward_duration` + aggregate at GradFn (LOCKED 2026-05-21)

### Rename

`GradFnCall.duration` was the only bare `duration` field in the API; every other duration carries a prefix (`func_duration`, `capture_duration`, `setup_duration`, etc.). Rename for consistency and direction-explicit semantics.

| Was | New |
|---|---|
| `GradFnCall.duration` | `GradFnCall.backward_duration` |
| (none) | `GradFnCall.backward_duration_str` |

### Why `backward_duration` over alternatives

- `duration` (bare) — locally clear, but inconsistent with rest of API
- `func_duration` — parallels `Op.func_duration` but creates muddled vocab (TorchLens `func_` usually means torch function on the forward side)
- `hook_duration` — too specific (GradFnCall is "hook firing OR backward call event")
- **`backward_duration`** — direction-explicit, slots into TorchLens's forward/backward naming pattern

### Aggregate at GradFn (parallels ModuleCall/Module timing)

| New field on GradFn | Form |
|---|---|
| `backward_duration` | `@property` single-call passthrough (raises for multi-call) |
| `backward_duration_str` | `@property` human-readable |
| `total_backward_duration` | `@property` sum across this GradFn's calls |
| `total_backward_duration_str` | `@property` human-readable |

Mirrors the forward-direction timing surface added to Module/ModuleCall in the previous lock.

### No naming collision with hypothetical Trace.backward_duration

If Trace ever adds `backward_duration` for the whole-backward-pass aggregate, no collision — `total_backward_duration` on GradFn aggregates across ONE GradFn's calls (scoped), not across the trace.


---

## GradFn graph relations follow BACKWARD orientation (LOCKED 2026-05-21)

Flips the orientation of `GradFn.parents` / `GradFn.children` and related graph-relation fields from forward-orientation (v3's earlier convention) to backward-execution orientation.

### Rationale

The autograd graph is its own DAG with its own execution order. PyTorch's `tensor.grad_fn.next_functions` walks backward — `next_functions` are the grad_fns that fire NEXT in backward (downstream in backward orientation).

The forward graph and the backward graph are DIFFERENT graphs even though structurally related. Each should have its own parents/children semantic. The forward graph orientation lives on Op (`op.parents`, `op.children` are forward-upstream/downstream). The backward graph orientation lives on GradFn.

Mixing orientations (using forward-orientation on the backward graph) loses the autograd-graph fidelity and contradicts PyTorch's vocabulary.

### Semantic flip table

| Field | Was (forward orientation) | New (backward orientation) |
|---|---|---|
| `GradFn.parents` | forward-upstream grad_fns | grad_fns firing BEFORE this in backward (autograd-graph upstream) |
| `GradFn.children` | forward-downstream grad_fns | grad_fns firing AFTER this in backward (autograd-graph downstream) |
| `GradFn.siblings` | shares forward-parent | shares backward-parent |
| `GradFn.co_parents` | shares forward-child | shares backward-child |
| `GradFn.has_parents` / `has_children` / `has_siblings` / `has_co_parents` | derived (forward) | derived (backward) |

PyTorch parallel: `grad_fn.next_functions` (PyTorch) = backward-children (TorchLens, in the new convention).

### Forward direction still accessible

Users wanting to walk the FORWARD graph from a GradFn: `grad_fn.op` resolves to the source Op; then `grad_fn.op.parents` / `grad_fn.op.children` are forward-upstream/downstream. The forward direction is one hop away via the `op` resolver.

### Docs requirement

The GradFn section header in the glossary carries an "Orientation convention" callout explaining the flip and the cross-graph access pattern (`grad_fn.op.parents` for forward). Critical to surface upfront so users aren't surprised when `grad_fn.children` doesn't match `op.children`.

### Pedagogical justification (JMT's framing)

A user confused by this learns: (1) the backward graph is its own DAG; (2) PyTorch's `next_functions` walks backward; (3) parents/children semantics are graph-orientation-dependent. All real autograd knowledge worth having. The temporary confusion is self-correcting once they look at the actual graph.

### Implementation note

If existing code populates GradFn.parents/children in forward orientation, this is a behavioral change, not just docs. Verify at rename-sprint implementation pass — the underlying data flow may need to be inverted to match the new convention.


---

## Bundle.add / remove / remove_except — accept both strings/Traces and lists (LOCKED 2026-05-21)

Currently `add` takes single Trace only; `remove` takes single string only; `remove_except` takes list only. Harmonize all three to accept both forms.

| Method | Before | After |
|---|---|---|
| `Bundle.add(log, name=None)` | single Trace, optional name | `add(log_or_logs, names=None)` — Trace or list of Traces; names is str / list[str] / None (must match log count or be None) |
| `Bundle.remove(name)` | single string | `remove(name_or_names)` — str or list[str]; returns Trace or list[Trace] |
| `Bundle.remove_except(keep)` | list only | `remove_except(name_or_names)` — str or list[str] |

Implementation: each method type-dispatches on the argument, looping internally if a list is passed.


---

## Output-naming disambiguation on ModuleCall + Module (LOCKED 2026-05-21)

Output-related fields had grown ambiguous (`outputs` vs `outs`, bare `output_labels` for Op labels alongside `output_layer_labels` for Layer labels). Renaming for unambiguous prefix-buckets.

### Rule

Three prefix-buckets carry granularity in the name:
- **`out*` / `outs`** (no underscore-after) → TENSOR VALUE or METADATA (`outs`, `out_shape`, `out_dtype`, etc.)
- **`output_op_*`** → OP-RECORD granularity (`output_ops`, `output_op_labels`)
- **`output_layer_*`** → LAYER-RECORD granularity (`output_layers`, `output_layer_labels`)
- **`output_<other>`** → other metadata (`output_structure`)

### Renames on ModuleCall

| Was | New |
|---|---|
| `output_labels` | `output_op_labels` |
| `outputs` (`@property`) | `output_ops` (`@property`) |

### Additions on ModuleCall (paired plural resolvers + input-side symmetry)

| New | Form |
|---|---|
| `output_layers` (`@property`) | List of Layer records resolved from `output_layer_labels` |
| `input_op_labels` | Stored list of input boundary Op labels |
| `input_ops` (`@property`) | List of Op records resolved from `input_op_labels` |
| `input_layers` (`@property`) | List of Layer records resolved from `input_layer_labels` |

### Module-scope: aggregate semantics + parallel renames

Module-scope boundary fields are AGGREGATE across calls (union of per-call boundary positions), NOT single-call passthroughs. Exception: `output_structure` IS passthrough (raises for multi-call).

| Module field | Form |
|---|---|
| `layer_labels` | Aggregate union of Layer labels across calls |
| `layers` (`@property`) | Aggregate union of Layer records |
| `input_op_labels`, `input_ops`, `input_layer_labels`, `input_layers` | All aggregate across calls |
| `output_op_labels`, `output_ops`, `output_layer_labels`, `output_layers` | All aggregate across calls |
| `output_structure` | Single-call passthrough; raises for multi-call |

### Why the aggregate vs passthrough asymmetry

- Labels can aggregate (union of label sets makes sense)
- Tensor values can't aggregate (each call has its own values, possibly different shapes/contents) — hence `out` / `outs` are single-call passthroughs at Module level
- `output_structure` is per-call container spec; passthrough because the structure varies by call


---

## Trace-level backward duration aggregates (LOCKED 2026-05-21)

Closes a timing-coverage gap. Per-backward-pass timing existed only per-event in `state_history`; no Trace-level aggregate for "how long did backward take in total?"

### Additions

| Field | Form |
|---|---|
| `Trace.backward_durations` | list[float], one per backward pass in order |
| `Trace.backward_durations_str` | list[str], human-readable per-pass |
| `Trace.last_backward_duration` | `@property`, most recent |
| `Trace.last_backward_duration_str` | `@property`, human-readable |
| `Trace.total_backward_duration` | `@property`, sum across passes |
| `Trace.total_backward_duration_str` | `@property`, human-readable |

Parallels the `backward_root_grad_fn_ids` family (Level 1 lightweight plural). When the deferred `BackwardPass` first-class record lands (post-2.0), these become derivable via `[bp.duration for bp in trace.backward_passes]`.


---

## `id` vs `label` convention across the API (LOCKED 2026-05-21)

Locks the universal naming rule: bare `_id` is REPLACED with either `_object_id` (Python runtime id) or `label` / `_label` (TorchLens-internal stable identifier). Forces explicit disambiguation everywhere.

### Rule

| Suffix | Meaning | Portable? |
|---|---|---|
| `_object_id` (or `_object_ids` plural) | Python `id()` runtime value | NO — within-session only |
| `label` (bare) or `_label` (qualified) | TorchLens-internal stable identifier | YES — survives portable save/load |

Bare `_id` is RESERVED — never use without the `_object_` qualifier (avoids the ambiguity that the two kinds of ids look the same on the API surface).

### Renames applied

**Python `id()` runtime values (→ `_object_id`):**

| Was | New |
|---|---|
| `Op.grad_fn_id` | `Op.grad_fn_object_id` |
| `Layer.grad_fn_id` | `Layer.grad_fn_object_id` (passthrough) |
| `Trace.backward_root_grad_fn_ids` | `Trace.backward_root_grad_fn_object_ids` |
| `Trace.last_backward_root_grad_fn_id` | `Trace.last_backward_root_grad_fn_object_id` |

**TorchLens-internal stable identifiers (→ `label`):**

| Was | New |
|---|---|
| `Conditional.id` | `Conditional.label` |
| `ConditionalRoleRef.conditional_id` | `ConditionalRoleRef.conditional_label` |
| `Op.terminal_bool_for` tuple `(conditional_id, arm_index)` | `(conditional_label, arm_index)` |
| `Layer.terminal_bool_for` tuple `(conditional_id, arm_index)` | `(conditional_label, arm_index)` |

### Conventions section update

Added explicit "id vs label" rule to the v3 glossary Conventions section, immediately before the `_label` user-overridable convention.

### Implementation note

If existing code uses the old field names, this is a behavioral change beyond docs — needs verification at rename-sprint code pass. Should be a search-and-replace across the codebase.


---

## Transformed shape/dtype renames — apply tensor-attribute-short rule uniformly (LOCKED 2026-05-21)

Fixes the inconsistency where transformed-side shape/dtype fields used full prefix (`transformed_activation_shape`) while raw-side used short prefix (`out_shape`). Apply the same rule uniformly:

| Was | New |
|---|---|
| `transformed_activation_shape` (+ `_shapes` plural) | `transformed_out_shape` (+ `_shapes`) |
| `transformed_activation_dtype` (+ `_dtypes` plural) | `transformed_out_dtype` (+ `_dtypes`) |
| `transformed_gradient_shape` (+ `_shapes`) | `transformed_grad_shape` (+ `_shapes`) |
| `transformed_gradient_dtype` (+ `_dtypes`) | `transformed_grad_dtype` (+ `_dtypes`) |

### Unified rule after fix

| Category | Form | Rule |
|---|---|---|
| Tensor itself | short (`out`, `grad`, `transformed_out`, `transformed_grad`) | The raw thing |
| Shape / dtype | short prefix (matches PyTorch tensor attribute idiom) | `out_shape`, `transformed_out_shape`, `grad_shape`, `transformed_grad_shape` |
| Memory (TL-computed) | full prefix (TL-concept marker) | `activation_memory`, `transformed_activation_memory`, `gradient_memory`, `transformed_gradient_memory` |

The rule now applies uniformly to both raw and transformed sides. `_str` companions follow the new names automatically.


---

## `is_module_input` / `is_module_output` — data-flow-direction redefinition (LOCKED 2026-05-21)

Previously: both predicates meant "at a module's boundary position INSIDE the module's forward" (first/last op inside). The "first-op-inside" semantic for `is_module_input` had no useful query angle — derivable from `module_call_stack` comparison if anyone wants it.

Redefined to match the natural data-flow direction:

| Predicate | Was (wrong) | New (right) |
|---|---|---|
| `Op.is_module_input` | first op INSIDE the module's forward | this Op's output FEEDS INTO a module (Op is OUTSIDE; upstream producer of an input arg). Equivalent to `bool(op.input_to_module_calls)`. |
| `Op.is_module_output` | final compute op INSIDE the module's forward | this Op IS the output of a module (Op is INSIDE; its tensor is the module's return value). Equivalent to `bool(op.output_of_module_calls)`. |
| `Layer.is_module_input` | (same wrong meaning) | (same correct meaning, delegated from `Op.is_module_input`) |
| `Layer.is_module_output` | (same wrong meaning) | (same correct meaning, delegated from `Op.is_module_output`) |

### Rationale

Data-flow direction is the natural framing:
- Inputs come FROM outside the module (the upstream producer is the "input feeder")
- Outputs are produced INSIDE the module (the final compute op IS the module's output)

This is ASYMMETRIC across input/output (input is outside; output is inside), but the asymmetry IS the dataflow.

The predicates now align cleanly with the existing companion list fields:
- `is_module_input` ⟺ `bool(input_to_module_calls)`
- `is_module_output` ⟺ `bool(output_of_module_calls)`

### "First op inside" is dropped as a concept

If anyone needs "the first op inside this module's forward" (the boundary marker), it's derivable via `module_call_stack` comparison between adjacent ops — when the stack just got pushed with a new module call, that next op is the "first inside." No dedicated predicate.

### Implementation note

This is a SEMANTIC change to existing predicates, not just docs. Current code populating `is_module_input` for first-op-inside needs to switch to populating for has-non-empty-`input_to_module_calls`. Behavioral change at rename-sprint code pass.

### Companion sanity check

- `Op.is_atomic_module_op` (sole operation output of an atomic module call) — INSIDE the module, IS the output. Aligned with `is_module_output` direction. ✓
- `Op.module_calls_entered` already removed earlier (was dup with `input_to_module_calls`). ✓
- `Layer.in_submodule` (computed inside a non-root module) — distinct from `is_module_input`/`is_module_output`; covers "inside any non-root module's forward." Kept; v3 entry already explains the distinction.


---

## `atomic_module_call_label` naming confirmed + Op/Layer parity (LOCKED 2026-05-21)

### Naming: `_label` is correct (NOT `_address`)

ModuleCall labels include the `:N` call-index suffix (e.g., `decoder.fc2:2`). The pure PyTorch address would be just `decoder.fc2`. Per the locked id-vs-label convention, TorchLens-internal stable identifiers use `_label`:
- `_address` = pure PyTorch dotted path (no `:N`)
- `_label` = TorchLens identifier (may include `:N` or other TL-specific suffixes)

`atomic_module_call_label` correctly carries the `:N`-augmented form. No change.

### Op / Layer atomic_module field parity

Op had the full triplet (`atomic_module_call_label`, `atomic_module_call` @property, `atomic_module` @property) plus `is_atomic_module_op` predicate. Layer had ONLY the predicate. Filled the gap:

| Field | Op | Layer (before) | Layer (after) |
|---|---|---|---|
| `is_atomic_module_op` | ✓ | ✓ | ✓ |
| `atomic_module_call_label` | ✓ | — | ✓ added (single-Op passthrough; raises for multi-Op Layers) |
| `atomic_module_call` (`@property`) | ✓ | — | ✓ added (delegated) |
| `atomic_module` (`@property`) | ✓ | — | ✓ added (delegated) |

Layer fields delegate via the single-Op passthrough rule (raise on multi-Op Layers; users access `layer.ops[N].atomic_module_call` directly when multi-pass).


---

## `is_atomic_module_op` → `is_atomic_module` (LOCKED 2026-05-21)

Drop the `_op` suffix. Applies to both Op and Layer scopes.

| Was | New |
|---|---|
| `Op.is_atomic_module_op` | `Op.is_atomic_module` |
| `Layer.is_atomic_module_op` | `Layer.is_atomic_module` |

Rationale:
- `_op` reads as "this is an op" — fine on Op scope, weird on Layer scope (Layer isn't an Op)
- The predicate's true semantic: "this thing represents an atomic module's compute" — works on both scopes
- Bare `is_atomic_module` is the predicate; bare `atomic_module` is the resolver (@property → Module). Standard predicate-vs-resolver pair pattern (parallel to `has_op` vs `op` on GradFn).

Predicate + resolver cluster after rename:
- `Op.is_atomic_module` (bool predicate)
- `Op.atomic_module_call_label` (storage label)
- `Op.atomic_module_call` (`@property` → ModuleCall)
- `Op.atomic_module` (`@property` → Module)
- Same set on Layer via single-Op passthrough


---

## `Op.in_submodule` documented for parity with Layer (LOCKED 2026-05-21)

`Op.in_submodule` (`@property`) exists in code at `op_log.py:1007` but was missing from v3 entries. Layer side was documented; Op side was not. Filled the parity gap.

| Field | Op | Layer |
|---|---|---|
| `in_submodule` (`@property`) | ✓ added | ✓ already existed |

Semantic on both: True when this record was computed inside a NON-ROOT module's forward. Distinct from `is_module_input` / `is_module_output` (which are about feeds-into / IS-output relationships, see prior lock).

---

## Bare parent identifier accepted by scoped accessors for single-X case (LOCKED 2026-05-21)

Generalization: any scoped accessor (`Layer.ops`, `Module.calls`, `GradFn.calls`) accepts the BARE parent identifier (no `:N`) as a lookup key when there's only one element. Multi-X case raises with disambiguation guidance.

| Accessor | Single-X bare lookup | Multi-X bare lookup |
|---|---|---|
| `Layer.ops` | bare Layer label (`conv2d_2_3`) resolves to the unique Op | raises `AmbiguousOpLookupError` |
| `Module.calls` | bare Module address (`encoder.block`) resolves to the unique ModuleCall | raises |
| `GradFn.calls` | bare GradFn label resolves to the unique GradFnCall | raises |

Pythonic alternatives still work:
- `accessor[0]` — 0-based positional (always safe; harmonized convention)
- `accessor["label:1"]` — `:N`-qualified label (always safe)

The bare-label form is convenience for the common single-X case. Filter expressions like `for op in layer.ops` are unchanged (always iterate by position).

### Rationale

Matches the strict-type accessor rule already locked for Trace-level `trace.ops[bare_layer_label]` (single-pass passthrough). Lifts the same pattern to scoped accessors for consistency.

### Implementation note

Same single-pass-passthrough machinery as the Trace-level rule. Scoped accessors check whether the bare key matches the parent's identifier AND whether there's exactly one element; resolve or raise accordingly.


---

## ModuleCall identity fields revert — bare names, no `module_` prefix (LOCKED 2026-05-21)

Earlier in this session, ModuleCall identity fields were renamed to add `module_` prefix (paralleling Param's `module_address`, etc.). On reconsideration, the prefix is OVER-QUALIFICATION for ModuleCall — Param needs it (Param has its own `address`), but ModuleCall doesn't have competing fields.

ModuleCall's OWN identity is `call_label` / `call_index` (NOT `address`, NOT `name`). So bare `address`, `name`, `class_name`, `has_multiple_addresses` on ModuleCall unambiguously refer to the called Module — no other interpretation makes sense.

### Renames (reverting the earlier qualification)

| Was (over-qualified) | New (bare) |
|---|---|
| `ModuleCall.module_address` | `ModuleCall.address` |
| `ModuleCall.module_name` (`@property`) | `ModuleCall.name` (`@property`) |
| `ModuleCall.module_cls` (`@property`) | `ModuleCall.cls` (`@property`) |
| `ModuleCall.all_module_addresses` | `ModuleCall.all_addresses` |
| `ModuleCall.module_has_multiple_addresses` (`@property`) | `ModuleCall.has_multiple_addresses` (`@property`) |
| (none — was missing) | `ModuleCall.class_name` (`@property`) — added for parity with Module |
| (none — was missing) | `ModuleCall.class_qualname` (`@property`) — added for parity with Module |
| `ModuleCall.module` (`@property` → Module) | unchanged — bare resolver |
| `ModuleCall.modules` (`@property` → list of Modules) | unchanged — paired plural resolver |

### Principle: qualifier only when needed for disambiguation

| Record | Field collision? | Bare or qualified? |
|---|---|---|
| `Param` | Yes — `Param.address` IS the parameter's name | Module fields use `module_*` qualifier |
| `ModuleCall` | No — `ModuleCall.call_label` is the own identity, `address` is free | bare names refer unambiguously to called Module |
| `Buffer` | Yes — `Buffer.address` IS the buffer's name | Module fields would need `module_*` qualifier (separate question) |

The asymmetry between Param and ModuleCall (Param uses `module_*` prefix; ModuleCall uses bare) is justified by the underlying collision structure.

### Companion additions for parity

`class_name` and `class_qualname` were missing on ModuleCall — added so the ModuleCall surface mirrors what's available on Module (class identity fields). Filling parity for the called Module's identity surface.


---

## Drop input/output collection @property resolvers on ModuleCall + Module (LOCKED 2026-05-21)

Reconsideration of the paired-plural-resolver pattern. The pattern is justified for navigation-natural surfaces (`call_parent` / `call_children`, `module` / `modules`) but is OVERKILL for boundary-filter surfaces (input/output ops/layers).

### Dropped from ModuleCall

| Removed |
|---|
| `mc.input_ops` (`@property`) |
| `mc.input_layers` (`@property`) |
| `mc.output_ops` (`@property`) |
| `mc.output_layers` (`@property`) |

### Dropped from Module (aggregate)

| Removed |
|---|
| `module.input_ops` (`@property`) |
| `module.input_layers` (`@property`) |
| `module.output_ops` (`@property`) |
| `module.output_layers` (`@property`) |

Kept (label lists, on both ModuleCall and Module):
- `input_op_labels`, `input_layer_labels`
- `output_op_labels`, `output_layer_labels`

### Refined principle: paired-plural-resolver applies to NAVIGATION surfaces only

| Surface type | Resolver pattern |
|---|---|
| Navigation (parent/children, owner/owners) | KEEP @property paired plural — users walk it to inspect objects |
| Boundary filter (input/output ops/layers, internal source/sink lists, etc.) | DROP @property — label list suffices; resolve via comprehension |
| High-traffic iteration (`Module.layers` — full Layer set) | KEEP @property — common iteration target |

Net: plural @property resolvers earn their keep ONLY for high-traffic navigation/iteration surfaces, not for filter-style label lists.

### Updated Principle 4 (clarification)

The Principle 4 exception (paired plural resolver) applies when:
1. There's a singular @property resolver in the same conceptual cluster (parent/children, module/modules)
2. AND the plural is navigation-natural (users walk it to inspect objects)

Not just "any plural matching a singular resolver." Filter-style boundary collections fail criterion 2.

### What this means for the API

| Field | Type | Resolution |
|---|---|---|
| `mc.module` / `mc.modules` (plural for shared module) | `@property` | KEEP — navigation/identity |
| `mc.address` / `mc.all_addresses` | label / label list | label form only |
| `mc.call_parent` / `mc.call_children` | `@property` | KEEP — navigation |
| `mc.input_op_labels` / `mc.output_op_labels` / etc. | label list | DROP @property resolvers — boundary filter |
| `module.layers` | `@property` | KEEP — high-traffic iteration |
| `module.input_layers` / `module.output_layers` | label list (renamed `_labels`) | DROP @property — boundary filter |


---

## Param.used_by — split into op-level and layer-level (LOCKED 2026-05-21)

Single ambiguous `used_by_layers` field ("Layer or Op labels") replaced with two clear bare label lists.

| Was | New |
|---|---|
| `Param.used_by_layers` (mixed Layer/Op labels) | `Param.used_by_op_labels` (Op labels, per-pass) + `Param.used_by_layer_labels` (Layer labels, equivalence-class) |

Plus count companions:
- `Param.num_used_by_ops` (`@property`)
- `Param.num_used_by_layers` (`@property`)

Both label lists are bare (no @property resolvers — filter-style boundary collections, per the just-locked principle).

---

## GradFn.trace_index → step_index (LOCKED 2026-05-21)

Reverts an earlier intermediate decision. Rename for symmetry with `Op.step_index` and `Layer.step_index`.

| Was | New |
|---|---|
| `GradFn.trace_index` | `GradFn.step_index` |

Semantic: 1-based index across all grad-fns in backward execution order. Same role as `Op.step_index` but in the backward direction. Unlike Op.step_index (0 for boundary ops), GradFn.step_index is always >= 1 (no boundary-equivalent on backward side).

---

## Super[X] member accessor + absent_traces (LOCKED 2026-05-21)

Promote `members` to an Accessor + add the absent-traces complement.

### Changes

| Field | Was | New |
|---|---|---|
| `members` | dict (str → record) | Accessor — supports str OR int (0-based positional) indexing, plus `.keys()` / `.values()` / `.items()` / iteration |
| `traces` | (unchanged) | Set of Trace names represented |
| `absent_traces` | (none) | `@property` — Bundle Trace names where this label does NOT resolve |
| `num_traces` | (none) | `@property` — `len(traces)` |
| `num_absent_traces` | (none) | `@property` — `len(absent_traces)` |
| `is_complete_coverage` | (none) | `@property` — True when no absent traces |
| `coverage` | (unchanged) | fraction; equivalent to `num_traces / total` |

### Rationale

- `members` Accessor harmonizes with the rest of the API (int + str indexing throughout)
- `absent_traces` answers a real query: "which Bundle members are MISSING this label?" Useful for diagnosing alignment gaps.
- `is_complete_coverage` is a one-shot predicate for "is this label present in every member?"
- `num_traces` / `num_absent_traces` give count parity with the existing `num_X` family

### Why complement matters

When users align across Bundle members, they often want to know not just WHO has this label but who DOESN'T. The complement is derivable (`bundle.trace_names - super_layer.traces`) but expensive to construct mentally; making it a direct field surfaces the diagnostic info first-class.


---

## Drop Super[X].outs / .grads — same-word-different-semantics with ModuleCall.outs (LOCKED 2026-05-21)

`outs` had two different semantics in v3 that collide:

| Context | Semantic | Structure |
|---|---|---|
| `ModuleCall.outs` / `Module.outs` | Multi-output container — modules whose `forward()` returns a tuple/list/dict of tensors | LIST of tensors, container-path ordered |
| `SuperOp.outs` / `SuperLayer.outs` (was) | Cross-trace alignment — one label, multiple Bundle members' tensors | DICT of trace_name → tensor |

Same field name, different shape (list vs dict), different axis (container vs member). User reading `obj.outs` had to mentally re-parse the context to know what kind of plural this was.

### Resolution

Drop the cross-trace dict forms on Super[X]:

| Removed |
|---|
| `Super[X].outs` (dict of trace_name → tensor) |
| `Super[X].grads` (dict of trace_name → gradient) |

Per-member tensor access goes through `super_X.members[N].out` (Accessor; int or trace-name). The dict form is one comprehension when needed:
```python
{name: m.out for name, m in super_layer.members.items()}
```

### What stays on Super[X]

Singular convenience forms — distinct from any plural collision:
- `super_X.out` — STACKED tensor when all members compatible; `None` otherwise
- `super_X.grad` — same for gradient
- `super_X.shape`, `super_X.type`, `super_X.module` — single-value diagnostics across members

No naming collision because these are singular cross-member aggregates, not lists.

### Rationale

Same principle as the previously locked "drop input/output collection @property resolvers" — convenience-only resolved collections are removed; the comprehension is one line. The benefit here is doubled because removing the plural ALSO removes the naming collision with ModuleCall.outs (which keeps its container-axis semantic).


---

## Final parity sweep — Op ↔ Layer passthrough completion (LOCKED 2026-05-21)

Final pass through the Op/Layer passthrough pair to fill in fields that Op had but Layer was missing. Single-Op passthrough rule applies uniformly: bare singular fields delegate to the one Op for single-pass Layers, raise for multi-pass.

### Added to Layer

**Identity / FX cluster:**
- `Layer.fx_label` (`@property` passthrough)
- `Layer.fx_qualpath` (passthrough)
- `Layer.fx_call_index` (passthrough)

**Function Identity Passthroughs — post-swap update:**
- Renamed `Layer.grad_fn` (was: runtime handle) to two new fields matching the Op-side post-swap structure:
  - `Layer.grad_fn` (`@property`) → TL GradFn record (passthrough; resolved via `grad_fn_label`)
  - `Layer.grad_fn_handle` → runtime PyTorch autograd Function (passthrough)
- Added `Layer.grad_fn_label` (storage)
- Renamed `Layer.grad_fn_log` to disappear (folded into `Layer.grad_fn` post-swap)

**Multi-output cluster:**
- `Layer.container_spec` — structural description of the return container (passthrough)

**Per-Layer Config and Saved State (additions for parity with Op):**
- `Layer.has_saved_args` (`@property` passthrough)
- `Layer.args_summary` (passthrough)
- `Layer.kwargs_summary` (passthrough)
- `Layer.non_tensor_pos_args` (passthrough)
- `Layer.non_tensor_kwargs` (passthrough)
- `Layer.func_autocast_state` (passthrough)

**Graph Relations (additions for parity with Op — at Layer-label granularity):**
- `Layer.parent_arg_positions` (passthrough)
- `Layer.output_descendants` (list of output boundary Layer labels reachable downstream)
- `Layer.input_ancestors` (list of input boundary Layer labels reaching upstream)
- `Layer.min_distance_from_input`, `Layer.max_distance_from_input`
- `Layer.min_distance_to_output`, `Layer.max_distance_to_output`
- `Layer.root_ancestors`
- `Layer.has_internal_source_ancestor`
- `Layer.internal_source_parents`, `Layer.internal_source_ancestors`
- `Layer.has_output_variations`
- `Layer.interventions`

### Items intentionally NOT added (justified asymmetry)

- `Layer.num_passes` — equivalent to `Layer.num_ops`; not duplicated. Op-side `Op.num_passes` is the Op's view of its parent Layer's count.

### ModuleCall ↔ Module + GradFnCall ↔ GradFn

Both pairs already audited and harmonized via earlier locks this session:
- ModuleCall identity revert (drop `module_` prefix; bare names work in ModuleCall context)
- Module timing fields (`func_calls_duration`, `total_func_calls_duration`) match Layer's `total_func_duration` pattern
- GradFn `step_index` rename (was `trace_index`) matches Op/Layer naming
- GradFn timing aggregates (`backward_duration`, `total_backward_duration`) match Module/ModuleCall pattern
- GradFnCall `backward_duration` matches Op-level `func_duration` semantic (one-call duration)

No additional gaps found that need filling pre-launch.

### Status: ready for implementation sprint

v3 glossary now reflects the complete locked surface. Implementation sprint can proceed using v3 as the spec target. All locks have a corresponding delta entry; behavioral-change items are flagged in their respective delta entries (e.g., `is_module_input` redefinition is semantic, not just docs).


---

## Count field parity completion across boundary types (LOCKED 2026-05-21)

Filled the missing `num_X` fields where the corresponding Accessor existed but no count parallel.

### Added

| Field | Counts |
|---|---|
| `Trace.num_input_layers` (`@property`) | `len(trace.input_layers)` |
| `Trace.num_output_layers` (`@property`) | `len(trace.output_layers)` |
| `Trace.num_buffer_layers` (`@property`) | `len(trace.buffer_layers)` |
| `Trace.num_internal_source_ops` (`@property`) | `len(trace.internal_source_ops)` |
| `Trace.num_internal_sink_ops` (`@property`) | `len(trace.internal_sink_ops)` |
| `Trace.num_orphans` (`@property`) | `len(trace.orphans)` |
| `Trace.num_uncalled_modules` (`@property`) | `len(trace.uncalled_modules)` |

### Coverage matrix after additions

Every Trace-level Accessor now has a paired `num_X` field:

| Boundary type | Layer Accessor | Op Accessor | num_X_layers | num_X_ops |
|---|---|---|---|---|
| Input | ✓ | ✓ | ✓ added | ✓ |
| Output | ✓ | ✓ | ✓ added | ✓ |
| Buffer (boundary) | ✓ | (Option B post-launch) | ✓ added | (Option B post-launch) |
| Internal source | ✓ | ✓ | ✓ | ✓ added |
| Internal sink | ✓ | ✓ | ✓ | ✓ added |
| Compute | ✓ | ✓ | ✓ | ✓ |
| Orphan | (none — Op only) | ✓ | N/A | ✓ added (`num_orphans`) |

Filter family also complete (`num_saved_*` cluster already had full coverage; `num_layers_with_params`, `num_ops_with_params` exist).


---

## Trace-level Accessor entries — explicit bidirectional bare/qualified lookup (LOCKED 2026-05-21)

The strict-type accessor convention was locked earlier; now applied explicitly to each Trace-level Accessor entry in the v3 glossary. Each accessor documents: (a) the return type (always the same; type-strict), (b) which label forms it accepts, (c) the single-X passthrough for bare parent identifiers.

### Universal rule across paired accessors

For any "aggregate vs per-event" pair (Layer/Op, Module/ModuleCall, GradFn/GradFnCall):

- **Aggregate accessor** (`trace.layers`, `trace.modules`, `trace.grad_fns`): ALWAYS returns the aggregate. Accepts bare aggregate identifier OR per-event-qualified label (strips `:N` to find the aggregate parent).
- **Per-event accessor** (`trace.ops`, `trace.module_calls`, `trace.grad_fn_calls`): ALWAYS returns the per-event record. Accepts per-event-qualified label directly. For SINGLE-X aggregates, also accepts the bare aggregate identifier → resolves to the unique per-event record. For multi-X aggregates, the bare aggregate identifier raises with disambiguation guidance.

### Examples

| Accessor call | Returns | Resolution |
|---|---|---|
| `trace.layers["conv2d_1_2"]` | Layer | bare Layer label match |
| `trace.layers["conv2d_1_2:1"]` | Layer | strips `:N` to find parent Layer |
| `trace.ops["conv2d_1_2:1"]` | Op | pass-qualified label direct match |
| `trace.ops["conv2d_1_2"]` | Op (if single-pass) | bare Layer label → unique Op for that single-pass Layer |
| `trace.ops["conv2d_1_2"]` (multi-pass) | raises `AmbiguousOpLookupError` | use `[N]` or `:N`-qualified label |
| `trace.modules["encoder.block"]` | Module | bare Module address |
| `trace.modules["encoder.block:1"]` | Module | strips `:N` to find parent Module |
| `trace.module_calls["encoder.block:1"]` | ModuleCall | direct match |
| `trace.module_calls["encoder.block"]` (single-call) | ModuleCall | bare Module address → unique ModuleCall |
| `trace.module_calls["encoder.block"]` (multi-call) | raises | use `[N]` or `:N`-qualified |
| `trace.grad_fns["AddBackward0"]` | GradFn | direct match |
| `trace.grad_fns["AddBackward0:1"]` | GradFn | strips `:N` |
| `trace.grad_fn_calls["AddBackward0:1"]` | GradFnCall | direct match |
| `trace.grad_fn_calls["AddBackward0"]` (single-call) | GradFnCall | bare label → unique GradFnCall |

### Why this matters

The accessor entries are now self-contained — users can read each entry and know exactly what label forms are accepted without consulting the Conventions section. Avoids "consult the rule book" friction.

### Implementation note

The bidirectional resolution requires the accessor implementation to:
1. Try direct match first (label as given)
2. If no match AND label looks like `:N`-qualified: try stripping `:N` and matching against aggregate
3. If no match AND label looks bare AND would correspond to a single-X aggregate: resolve to that aggregate's unique per-event record
4. Otherwise raise with disambiguation message

This is local to each accessor's `__getitem__` — small implementation cost.


---

## ModuleCall call-context additions + atomic_module storage completeness (LOCKED 2026-05-21)

### ModuleCall — added two complementary "call stack" concepts (parallels Op)

| New field | Form |
|---|---|
| `ModuleCall.code_context` | List of `FuncCallLocation` records — Python call-stack frames at module call invocation. Same semantic as `Op.code_context` but recorded at module-call entry time (vs op-execution time). |
| `ModuleCall.module_call_stack` | List of ModuleCall labels (root-first) for ancestor ModuleCalls active when this ModuleCall was invoked. Excludes self. `len(module_call_stack) == call_depth`. Parallels `Op.module_call_stack`. |

Two different "stack" axes:
- `code_context` — source-code call stack (user/PyTorch code locations)
- `module_call_stack` — TorchLens ModuleCall ancestry chain

Both are useful and live at the same scope.

### Op + Layer — added `atomic_module_address` for storage completeness

Resolves the lone @property-without-storage-companion in the atomic_module cluster.

Was:
- `atomic_module_call_label` (storage)
- `atomic_module_call` (`@property`)
- `atomic_module` (`@property` only — derived via two hops through `atomic_module_call.module`)

After:
- `atomic_module_call_label` (storage)
- `atomic_module_call` (`@property`)
- `atomic_module_address` (storage — NEW; uses `_address` per Module-identifier convention)
- `atomic_module` (`@property` — now resolves directly via `self.trace.modules[self.atomic_module_address]`)

Cluster now has uniform storage+resolver pairs (every @property has its companion storage).

The `_address` suffix (not `_label`) matches the Module-identifier convention — Module is identified by PyTorch dotted address; `atomic_module_call_label` keeps `_label` because ModuleCall identifier includes `:N`.

Layer side gets the same field via single-Op passthrough.

### Slight denormalization accepted

`atomic_module_address` IS derivable from `atomic_module_call_label.split(':')[0]`. But storing it separately:
- Removes the two-hop derivation (faster access for `atomic_module` resolver)
- Completes the storage+resolver pair pattern
- Cluster is internally consistent

Worth the redundancy cost. Per JMT 2026-05-21: "i sorta lean just having the label for completeness. its a weird omission as it stands imo."


---

## `has_output_variations` → `has_out_variations` (LOCKED 2026-05-21)

Cluster-cohesion rename. The pair was rule-consistent (predicate-full + tensor-short) but read with a surface word mismatch (`out_versions_by_child` vs `has_output_variations`). Rename the predicate to match the tensor-side prefix.

| Was | New |
|---|---|
| `Op.has_output_variations` | `Op.has_out_variations` |
| `Layer.has_output_variations` | `Layer.has_out_variations` |

### Rule break accepted

The predicate-full convention (`has_X_Y` where X is the concept) is slightly relaxed for this cluster: when a cluster's dominant data is tensor-side, the predicate borrows the same prefix for cluster cohesion. The rule is now: "predicates use full prefix by default; may use short prefix when in a tensor-anchored cluster."

Applies narrowly here; not a general license to break predicate-full elsewhere.

### After rename — cluster reads cleanly

- `Op.out_versions_by_child` (dict of per-child tensor versions)
- `Op.has_out_variations` (predicate: do different children see different `out` values?)

Both use `out_*` prefix. One named cluster.


---

## Op ↔ Layer atomic_module parity — docstring polish (LOCKED 2026-05-21)

Field-name parity was already complete (Op and Layer both have all 5 atomic_module fields). Polished Layer-side docstrings to make the passthrough semantics + multi-pass behavior explicit.

| Field | Op | Layer (after polish) |
|---|---|---|
| `is_atomic_module` | True when this Op is the sole output of an atomic module call | True when each Op in this Layer is the sole compute of its ModuleCall (delegates; True for both single-pass and multi-pass recurrent atomic layers) |
| `atomic_module_call_label` | Stable ModuleCall label | Single-Op passthrough; raises for multi-pass (each pass has own call label) |
| `atomic_module_call` (`@property`) | Resolves ModuleCall record | Single-Op passthrough; raises for multi-pass |
| `atomic_module_address` | Stable Module address | Single-Op passthrough; raises for multi-pass (conceptually all passes share the same Module, but applying rule uniformly) |
| `atomic_module` (`@property`) | Resolves Module record | Single-Op passthrough; raises for multi-pass |

### Subtle case noted

For a recurrent atomic module (e.g., a single nn.Module called many times in a loop, each call atomic), the multi-pass Layer has:
- `is_atomic_module = True` (every pass is atomic)
- `atomic_module_call_label` raises (each pass has its own call_label with different `:N`)
- `atomic_module_address` could in principle return the shared address (all passes share the Module), but is treated as raise-on-multi-pass for cluster consistency

Users in the recurrent-atomic case access `layer.ops[N].atomic_module_address` for explicit per-pass access. The shared address is available via `layer.ops[0].atomic_module_address` (any pass).

### Why uniform passthrough rule even when address is shared

The single-Op passthrough rule is applied uniformly across the cluster for predictability. A user querying `layer.atomic_module_address` shouldn't have to think "does this field happen to be shared across passes?" — same rule everywhere keeps the mental model simple.

If the recurrent-atomic case becomes a common query pattern, a future addition could be `Layer.shared_atomic_module_address` (raises if multi-pass AND addresses differ) — but defer until use case surfaces.


---

## Drop `ModuleCall.modules` plural resolver (LOCKED 2026-05-21)

Earlier in this session, `ModuleCall.modules` (`@property` → list of Modules) was added as a paired plural resolver matching `ModuleCall.module` (singular). Reconsidered: this was over-application of the paired-plural-resolver pattern.

### Why it's overkill on ModuleCall

A ModuleCall is invoked by **ONE Module instance** — the one whose `forward()` was called. If that nn.Module is registered at multiple addresses (shared-module aliasing), each Module RECORD is at a specific address, but the ModuleCall itself uses ONE specific path.

There's no meaningful "many Modules called" interpretation. The aliases are queryable via `mc.module.all_addresses` (the called Module's shared-address list) or `mc.all_addresses` directly. No need for a plural resolver.

### Removed

| Field | Status |
|---|---|
| `ModuleCall.modules` (`@property` → list of Modules) | REMOVED |

### Why Param.modules stays

`Param.modules` is a different case: an nn.Parameter can be genuinely SHARED across multiple owning Modules (weight tying — e.g., `decoder.embedding.weight = encoder.embedding.weight = ...`). Each Module-record-that-owns-this-parameter is a separate Module record. The plural is real.

### Final atomic Module-relationship surface on ModuleCall

| Field | Form |
|---|---|
| `mc.address` | bare label string — called Module's address |
| `mc.module` (`@property`) | the called Module record (singular; ONE Module) |
| `mc.all_addresses` | list of all addresses the called Module is registered at (shared-module aliases) |
| `mc.has_multiple_addresses` (`@property`) | True if shared-module |

No `mc.modules` plural — semantically meaningless.

### Paired-plural-resolver principle refinement

Refines the previously-locked Principle 4 amendment: paired plural resolvers earn their keep ONLY when:
1. The singular has a resolver (Principle 3 applies)
2. The plural represents a GENUINELY DIFFERENT set of entities (not aliases of the same singular)

For Param: `Param.module` (primary owner) vs `Param.modules` (multiple different Modules sharing the param) — different entities. Plural earns it.

For ModuleCall: `mc.module` (the called Module) vs hypothetical `mc.modules` (aliases of the same Module via shared-address registration) — same entity, different surface. Plural doesn't earn it.


---

## Drop `_labels` suffix on input/output collections (LOCKED 2026-05-21)

Reverts the qualified-suffix names on ModuleCall + Module input/output collections to bare. The `_labels` suffix only earns its keep when paired with a bare-name @property resolver to disambiguate against; since we dropped those resolvers in an earlier lock, the bare names are correct.

### Renames on ModuleCall

| Was | New |
|---|---|
| `mc.op_labels` | `mc.ops` |
| `mc.input_op_labels` | `mc.input_ops` |
| `mc.input_layer_labels` | `mc.input_layers` |
| `mc.output_op_labels` | `mc.output_ops` |
| `mc.output_layer_labels` | `mc.output_layers` |

### Renames on Module (aggregate side)

| Was | New |
|---|---|
| `module.input_op_labels` | `module.input_ops` |
| `module.input_layer_labels` | `module.input_layers` |
| `module.output_op_labels` | `module.output_ops` |
| `module.output_layer_labels` | `module.output_layers` |

### Kept with `_labels` suffix on Module — `layer_labels`

`Module.layer_labels` (bare label list) keeps the `_labels` suffix because it pairs with `Module.layers` (the @property resolver for ALL Layers — high-traffic iteration surface, kept). This is the canonical Principle 4 amendment case where the suffix is load-bearing.

| Field | Type | Why `_labels` suffix |
|---|---|---|
| `Module.layer_labels` | bare label list | Disambiguates against `Module.layers` (`@property`) |
| `Module.layers` | `@property` → list of Layer records | High-traffic iteration; kept as resolver |
| `Module.input_layers` | bare label list | NO `_labels` suffix — there's no `Module.input_layers` resolver to disambiguate against |

### Naming consistency at different scopes

| Scope | `input_layers` field type |
|---|---|
| `Trace.input_layers` | Accessor (returns Layer records on lookup) |
| `Module.input_layers` | bare label list |
| `ModuleCall.input_layers` | bare label list |

Same field name, different return types by scope. Defensible — receiver class makes the type clear in context (`trace.input_layers["X"]` returns a Layer; `mc.input_layers` is just a list of strings).

The convention is: bare name reflects WHAT (input layers); the receiver class determines HOW (Accessor at Trace, bare list at scoped records).


---

## has_trainable_params / has_frozen_params parity across scopes (LOCKED 2026-05-21)

`has_trainable_params` existed only on Module. No `has_frozen_params` anywhere. Filled the parity gap across all scopes that have associated parameters.

### Additions

| Class | Added |
|---|---|
| Op | `has_trainable_params` (`@property`), `has_frozen_params` (`@property`) |
| Layer | `has_trainable_params` (`@property`), `has_frozen_params` (`@property`) |
| Module | `has_frozen_params` (`@property`) — was missing; `has_trainable_params` already existed |
| ModuleCall | `has_trainable_params` (`@property`), `has_frozen_params` (`@property`) — both delegated from `self.module` |
| Trace | `has_trainable_params` (`@property`), `has_frozen_params` (`@property`) — both derived from `num_params_trainable` / `num_params_frozen` |

### Semantic by class

- **Op / Layer:** "this Op/Layer CONSUMES at least one X parameter"
- **Module / ModuleCall:** "this Module OWNS / this call's Module owns at least one X parameter"
- **Trace:** "the model has at least one X parameter"

Same predicate name (`has_X_params`) across scopes; receiver class determines the precise meaning (consume vs own vs has-globally). User adapts to context.

### Pairs with existing uses_params

The cluster on Op/Layer is now three predicates:
- `uses_params` — does this consume ANY parameters? (existing)
- `has_trainable_params` — does this consume at least one TRAINABLE param?
- `has_frozen_params` — does this consume at least one FROZEN param?


---

## ModuleCall + Module — overall forward_duration field (LOCKED 2026-05-21)

Closes a timing gap: wall-clock duration of a Module's `forward()` call (parallel to `Trace.forward_duration` at one-call granularity), complementing the existing `func_calls_duration` (pure torch-compute slice).

### Additions

| New field on ModuleCall | Form |
|---|---|
| `forward_duration` | `@property` — wall-clock seconds for THIS ModuleCall's `forward()` invocation (start to return) |
| `forward_duration_str` | `@property` — human-readable |

| New field on Module | Form |
|---|---|
| `forward_duration` | `@property` — single-call passthrough; raises for multi-call |
| `forward_duration_str` | `@property` |
| `total_forward_duration` | `@property` — sum across all calls |
| `total_forward_duration_str` | `@property` |

### Why both `forward_duration` AND `func_calls_duration`

| Quantity | What it measures |
|---|---|
| `forward_duration` | Wall-clock for the whole `forward()` call (includes Python orchestration, wrapper overhead, nested module calls' overhead) |
| `func_calls_duration` | Pure torch-compute slice (sum of `op.func_duration` for ops in this call, inclusive of nested) |
| `forward_duration - func_calls_duration` | ≈ wrapper overhead + Python orchestration (inclusive) |

User profiling questions:
- "How long did this module take total?" → `forward_duration`
- "How much was pure compute?" → `func_calls_duration`
- "How much was overhead (wrappers + Python)?" → the difference

### Parallel with Trace

| Scope | wall-clock | pure compute |
|---|---|---|
| Trace | `forward_duration` (whole forward pass) | `func_calls_duration` (sum of all op funcs) |
| Module | `forward_duration` (one call) / `total_forward_duration` (aggregate) | `func_calls_duration` / `total_func_calls_duration` |
| ModuleCall | `forward_duration` (this call) | `func_calls_duration` (this call's funcs) |

Same naming pattern across scopes. Trace's bare `forward_duration` is at top level; lower scopes use the same names because the receiver class disambiguates.

### Implementation note

`forward_duration` requires capturing start/end timestamps at module forward entry/exit. Probably via module forward hooks (TorchLens already hooks these for module-call tracking). Small addition to capture instrumentation.


---

## Param usage-tracking cluster cleanup (LOCKED 2026-05-21)

Multiple renames + one removal for clarity and convention consistency.

### Renames

| Was | New |
|---|---|
| `Param.used_by_op_labels` | `Param.used_by_ops` |
| `Param.used_by_layer_labels` | `Param.used_by_layer_labels` → `Param.used_by_layers` |
| `Param.num_used_by_ops` | `Param.num_uses_by_ops` |
| `Param.num_used_by_layers` | `Param.num_uses_by_layers` |

Rationale:
- Drop `_labels` suffix per the locked convention (no resolver companion to disambiguate against — bare names work for label lists)
- Grammar fix: `num_uses_by_ops` reads naturally ("number of uses, by ops"); `num_used_by_ops` reads passively/awkwardly

### Removal

| Removed |
|---|
| `Param.num_uses` |

Reason: redundant with `num_uses_by_ops` (both count Op usages that referenced this parameter). Drop the bare form; the explicit pair `num_uses_by_ops` + `num_uses_by_layers` is unambiguous and symmetric.

### After cleanup

```python
# Bare label lists:
param.used_by_ops              # list of Op labels
param.used_by_layers           # list of Layer labels

# Counts:
param.num_uses_by_ops          # = len(used_by_ops)
param.num_uses_by_layers       # = len(used_by_layers)
```

Both labels and counts are symmetric across the ops/layers axis. No bare `num_uses`.


---

## `Trace.code_context` added (LOCKED 2026-05-21)

Completes the `code_context` parallel across all three scope levels.

| Scope | What `code_context` captures |
|---|---|
| `Op.code_context` | Python call stack at op execution time |
| `ModuleCall.code_context` | Python call stack at module's `forward()` invocation |
| `Trace.code_context` (NEW) | Python call stack at the moment `tl.capture(model, x)` was invoked |

Use cases: provenance ("this Trace came from line X of my training script"), diagnostic (distinguishing similar traces in a notebook), reproducibility (call site is one of the few Trace-level provenance fields).

Cost: one stack snapshot at capture entry. Tiny.


---

## `is_multi_output` → `in_multi_output` (LOCKED 2026-05-21)

Rename for semantic accuracy. The Op isn't itself a multi-output (it's one tensor); it's PART OF a multi-output container.

| Was | New |
|---|---|
| `Op.is_multi_output` (`@property`) | `Op.in_multi_output` (`@property`) |
| `Layer.is_multi_output` | `Layer.in_multi_output` |

### Pattern fit

`is_X` predicates type the Op as being IN CATEGORY X (input boundary, compute op, atomic module). `in_X` predicates mark the Op as being INSIDE / PART OF container X (submodule, conditional).

The Op being part of a multi-output container is membership/containment — fits the `in_X` family.

| Pattern | Examples |
|---|---|
| `is_X` (role/category) | `is_input`, `is_output`, `is_compute_op`, `is_atomic_module`, `is_buffer_source` |
| `in_X` (containment) | `in_submodule`, `is_in_conditional`, **`in_multi_output`** |

Companion fields (`multi_output_type`, `multi_output_index`, `multi_output_name`, `container_path`, `container_spec`) stay — they describe the container, not predicate-test the Op.



---

## Module / ModuleCall internal-memory cluster + `output_`/`internal_` prefixes (LOCKED 2026-05-23)

Adds the three-quantity memory cluster for Module + ModuleCall scope (resolves the deferred decision in the `Module-scope memory aggregate naming` post-2.0 todo).

### Naming asymmetry across scopes

| Scope | Convention | Why |
|---|---|---|
| `Op`, `Layer` | bare `activation_memory` | No "internal" exists at these scopes (Op is atomic; Layer is multi-pass of an atomic op). No ambiguity → no prefix. |
| `ModuleCall`, `Module` | `output_` and `internal_` prefixes always explicit | The two quantities are both real and frequently confused; both pay the disambiguation tax. |

### Prefix choice: `output_` over `out_`

- Matches existing `Module.output_ops`, `Module.output_layers`, `Module.output_structure` field family (already at this scope).
- Symmetric with `internal_` in length and reading flow.
- The `out_` short prefix stays scoped to Op-level tensor-attribute fields (`out_shape`, `out_dtype`, `out_device`) per the locked tensor-attribute-short rule. Module-scope uses fully-spelled `output_`.

### Three orthogonal axes

| Axis | Values |
|---|---|
| Scope-of-aggregation | bare (single-call) / `total_` (cross-call sum) |
| Boundary vs aggregate | `output_` (boundary) / `internal_` (sum of inside ops) |
| Quantity | `activation` / `gradient` / `autograd` / `param` |

Compose for `{total_}_{output|internal}_{quantity}_memory` plus `_str` companions.

### ModuleCall (single-call quadrants only)

```
output_activation_memory          # output of THIS call
output_activation_memory_str
internal_activation_memory        # sum of internal-op activation in THIS call
internal_activation_memory_str

output_gradient_memory
output_gradient_memory_str
internal_gradient_memory
internal_gradient_memory_str

autograd_memory                   # autograd-saved during THIS call (inherently internal — no boundary form)
autograd_memory_str

param_memory                      # own-address params (call-invariant)
param_memory_str
internal_param_memory             # address-recursive sum of param_memory under this Module's subtree (call-invariant)
internal_param_memory_str
```

### Module (cross-call aggregates; single-call versions OMITTED — drill into ModuleCall)

```
total_output_activation_memory
total_output_activation_memory_str
total_internal_activation_memory
total_internal_activation_memory_str

total_output_gradient_memory
total_output_gradient_memory_str
total_internal_gradient_memory
total_internal_gradient_memory_str

total_autograd_memory             # (name already locked at this form)
total_autograd_memory_str

param_memory                      # call-invariant
param_memory_str
internal_param_memory             # call-invariant
internal_param_memory_str
```

### Why no per-call values at Module scope

Calls of the same Module can have varying output shapes (LSTMs with variable sequence length, conditional branches, etc.). A "representative single-call value" at Module scope is ambiguous. Force the user to drill into a specific `ModuleCall` for per-call values. Cross-call sums (`total_*`) are well-defined and well-typed.

### Implementation notes

- All `@property` derived from existing per-Op data — no new capture machinery.
- `internal_param_memory` shares its computation path with the new `recursive_params` accessor (see separate lock entry).
- Names compose with the previously-locked `total_` prefix family without collision.

### Cascade scope

- `data_classes/module_log.py` — add field cluster on `Module` + `ModuleCall`
- `data_classes/constants.py` — FIELD_ORDER additions
- `data_classes/_summary.py` — surface the new memory fields in summary rendering
- v4 glossary — Module + ModuleCall sections


---

## Recursive params accessor on Module (LOCKED 2026-05-23)

Resolves the post-2.0 PyTorch-parity-gap todo.

### Addition

```
module.params                       # existing — directly-owned Params (this Module's address)
module.recursive_params             # NEW — Accessor, address-recursive (this Module + all address-based sub-Modules)
module.num_recursive_params
module.num_recursive_params_trainable
module.num_recursive_params_frozen
module.num_recursive_param_tensors
module.num_recursive_param_tensors_trainable
module.num_recursive_param_tensors_frozen
module.recursive_param_addresses    # label list
```

### Semantic scope

"Recursive" = **address-based** (static `nn.Module` tree), matching PyTorch's `parameters(recurse=True)`. NOT call-tree recursive (which would aggregate Params used by ModuleCalls beneath this in the dynamic call tree).

### Why `recursive_` over alternatives

- `recursive_` matches PyTorch's `recurse` kwarg vocabulary — direct torch user idiom alignment.
- `descendant_` is workable but pulls toward address-tree framing only — confusing if call-tree analog ever materializes.
- `all_params` is ambiguous (all WHICH params?).
- `params_subtree` is verbose.

### Memory tie-in

Memory equivalent is `Module.internal_param_memory` from the cluster lock above (= sum of `param_memory` across `recursive_params`).


---

## Module / ModuleCall call-tree accessor + display (LOCKED 2026-05-23)

Promotes one-hop `call_parent` / `call_children` navigation to a first-class subtree accessor.

### API

```
trace.call_tree                # full call tree from root → leaves; CallTreeNode object
module.call_tree               # subtree rooted at this Module (aggregate across all calls)
module_call.call_tree          # subtree rooted at this specific call (per-invocation view)

trace.show_call_tree()         # ASCII tree printed to stdout (or file via kwarg)
module.show_call_tree()
module_call.show_call_tree()
```

### `CallTreeNode` structure

```python
@dataclass
class CallTreeNode:
    call: ModuleCall                  # the ModuleCall at this node
    children: list[CallTreeNode]      # nested CallTreeNodes
```

### Companion fields (cheap derivations)

```
module.num_descendant_calls          # total ModuleCalls nested under any call of this Module
module.max_descendant_depth          # deepest nesting beneath this Module
module_call.num_descendant_calls     # same, per-call
module_call.max_descendant_depth     # same, per-call
```

### Display kwargs

`max_depth`, `include_atomic`, `show_call_index`, `file` (paralleling `print()`).

### Naming notes

- `call_tree` (not `dynamic_call_tree`) — bare name; receiver class disambiguates from address tree (`Module.address_children` is the static-tree analog).
- `max_descendant_depth` (not `call_depth_from_beneath`) — "descendant" already implies "beneath this node," cleaner reading.
- `show_call_tree()` over `print_call_tree()` — `show_` is the established display-method verb in the codebase.

### Implementation scope

- API + companion fields lock NOW (data shape decided).
- `show_call_tree()` method body can land post-launch; small implementation work (recursive tree-walking + ASCII rendering).


---

## `Trace.num_modules` (LOCKED 2026-05-23)

```
trace.num_modules        # = len(trace.modules); count of registered submodules in source model
```

Bare name. "Modules" is unambiguous at Trace scope (no nesting context). Consistent with `Trace.modules` Accessor. `num_submodules` was considered but adds redundancy with the Trace-level scope.


---

## Op input-side `@property` cluster (LOCKED 2026-05-23)

Adds graph-parent input shortcuts on Op. All `@property` — zero storage cost, no new capture machinery.

### Cluster

```
op.input_ops              # Accessor over op.parents → Op records (not labels)
op.input_activations      # tuple[Tensor | None, ...] in op.parents order
op.input_shapes           # tuple[Shape, ...]
op.input_dtypes           # tuple[torch.dtype, ...]
op.input_memory           # int — sum of input activation_memory across parents
op.num_inputs             # = len(op.parents)
```

### Semantics

**Graph-parents only.** Mirrors `op.parents`. Does NOT include Param/Buffer values — those have their own access paths (`op.params`, `op.buffers`, plus their `.value` attributes). User mental model "input tensors" usually means activation flow, not param weights.

### Relationship to `op.args` / `op.kwargs`

Different views — neither replaces the other:

- **`op.args` / `op.kwargs`** — call signature exactly as the model author wrote it. Use for replay. May include scalars (`stride=1`), nested structures (`torch.cat([x, y], dim=0)` puts tensors in a list at position 0), and per-function arg schema.
- **`op.input_activations`** — graph-edge view; one tensor per Op parent in graph traversal order. Use for data-flow reasoning.

They overlap for simple ops but aren't redundant.

### Documented behaviors / footguns

1. **`None` slots** — when a parent wasn't saved (default save policy saves only a subset), the corresponding `op.input_activations[i]` is `None`. Consistent with `op.activation` semantics.
2. **In-place version resolution** — for ops downstream of in-place modifications, the resolver consults `out_versions_by_child` to return the version THIS op consumed, NOT the last-stored version.
3. **References, not copies** — mutating a returned tensor mutates the saved state, same as `op.activation`.

### Deferred (companion clusters for follow-up)

- `op.param_values` / `op.buffer_values` — analogous shortcuts for param/buffer flows. Defer until demand materializes.


---

## Module / ModuleCall args/kwargs template parity (LOCKED 2026-05-23)

ModuleCall currently has `forward_args_summary` + `forward_kwargs_summary`. Adds the `template` companions for Op parity.

### Addition

```
module_call.forward_args_template       # structural skeleton (shapes-not-values) of positional forward args
module_call.forward_kwargs_template     # structural skeleton of keyword forward args
```

Same semantics as Op's `args_template` / `kwargs_template`. Module aggregates inherit via the per-call structure.

Locks the full quartet on ModuleCall:
- `forward_args_summary` + `forward_args_template`
- `forward_kwargs_summary` + `forward_kwargs_template`


---

## Module / ModuleCall backward duration parity (NAMES LOCKED 2026-05-23; impl in backward-pass sprint)

Names locked now; implementation lands in the backward-pass unified sprint.

```
module_call.backward_duration         # wall-clock backward duration for this call
module_call.backward_duration_str
module.total_backward_duration        # cross-call sum
module.total_backward_duration_str
```

Mirrors the existing `forward_duration` / `total_forward_duration` pattern on ModuleCall/Module.


---

## Drop `CallTreeNode` — `ModuleCall` IS the node (LOCKED 2026-05-23)

The earlier "Module / ModuleCall call-tree accessor + display" lock introduced a `CallTreeNode` dataclass wrapping `ModuleCall` with a `children` list. On review (JMT, 2026-05-23): ModuleCall already has `call_parent` and `call_children`, so the wrapper added indirection for no new capability.

### Final design

- Drop `CallTreeNode` entirely.
- Drop `call_tree` `@property` from Trace, Module, ModuleCall.
- Add `walk_descendants()` method on `ModuleCall` and `Module`.
- Add `walk_calls()` method on `Trace`.
- Add `show_call_tree()` method on `ModuleCall`, `Module`, `Trace`.
- Keep `num_descendant_calls` and `max_descendant_depth` companions (independently useful).

Users iterate the tree via methods on existing records — no wrapper class to learn.


---

## Unit type family — `tl.Quantity` / `tl.Bytes` / `tl.Duration` / `tl.Flops` / `tl.Macs` (LOCKED 2026-05-24)

Replace the proliferation of `*_str` companion fields with a small set of formatted numeric subclasses. Halves the API surface; preserves inline ergonomics; makes aggregation type-preserving. Source proposal: `.project-context/unit_types_proposal.md`.

### The family

| Type | Base | Wraps | Format example |
|---|---|---|---|
| `tl.Bytes` | `int` | memory in bytes | `1234567` -> `"1.2 MB"` |
| `tl.Duration` | `float` | seconds | `0.00345` -> `"3.45 ms"` |
| `tl.Flops` | `int` | floating-point op count | `1230000000` -> `"1.23 GFLOPs"` |
| `tl.Macs` | `int` | multiply-accumulate count | `615000000` -> `"615 MMACs"` |

`tl.Quantity` is the abstract base class encompassing all four. Use `isinstance(x, tl.Quantity)` to test "is this a formattable unit type" (e.g., in serialization registry, summary renderers, etc.). Counts (`num_*` fields) stay as bare `int` — they don't carry formatting.

### Operator semantics

| Operation | Result | Reason |
|---|---|---|
| `Bytes + Bytes` | `Bytes` | aggregation |
| `Bytes + int` (or `int + Bytes`) | `Bytes` | scalar promotion; reflected dispatch handles `sum()` |
| `Bytes - Bytes` | `Bytes` | difference of magnitudes |
| `Bytes * scalar` | `Bytes` | scaling |
| `Bytes * Bytes` | `TypeError` | "bytes squared" is meaningless |
| `Bytes / scalar` | `Bytes` | scaling |
| `Bytes / Bytes` | `float` | ratio, dimensionless |
| `Bytes // Bytes` | `int` | discrete ratio |
| `-Bytes`, `abs(Bytes)` | `Bytes` | sign preserved |
| `min` / `max` / `sorted` | `Bytes` | inherited via `int.__lt__` etc. |
| cross-type (e.g. `Bytes + Duration`) | `TypeError` | units don't mix |

Same template for `Duration` / `Flops` / `Macs`.

### `__format__` support

Each type supports unit selection via the format spec mini-language:
- `f"{x}"` -> default formatted output (`"1.2 MB"`)
- `f"{x:raw}"` -> raw numeric (`"1234567"`)
- `f"{x:MB}"` (Bytes) -> forced units; precision via standard float spec (`f"{x:.2f MB}"`)
- `f"{x:ms}"` (Duration), `f"{x:GFLOPs}"` (Flops), etc.

Exact unit-suffix grammar is finalized at implementation time; the lock is that `__format__` is supported.

### What gets deleted

**Every `*_str` field disappears from the API surface.** Roughly 70-90 fields across Trace / Op / Layer / Module / ModuleCall / GradFn / GradFnCall. The bare numeric field now returns the wrapper type; `print(op.activation_memory)` produces `"1.2 MB"` via `__str__`.

Affected surface (every `*_str` companion):
- Trace: `total_*_memory_str`, `saved_*_memory_str`, `forward_peak_memory_str`, `backward_peak_memory_str`, `total_autograd_memory_str`, `total_param_gradient_memory_str`, `forward_duration_str`, `func_calls_duration_str`, `backward_durations_str`, `last_backward_duration_str`, `total_backward_duration_str`, `total_func_calls_duration_str`.
- Op: `activation_memory_str`, `gradient_memory_str`, `param_memory_str`, `autograd_memory_str`, all `transformed_*_str`, `flops_*_str`, `macs_*_str`, `func_duration_str`.
- Layer: every parallel `_str` field.
- Module / ModuleCall: every 5-23 memory cluster `_str`, plus `forward_duration_str`, `total_forward_duration_str`, `func_calls_duration_str`, `total_func_calls_duration_str`, `backward_duration_str`, `total_backward_duration_str`, etc.
- GradFn / GradFnCall: `backward_duration_str` + aggregates.

### What's preserved

- Inline ergonomics: `print(op.activation_memory)` and `f"{op.activation_memory}"` produce `"1.2 MB"`.
- Arithmetic: `op.activation_memory + 1024` returns `Bytes`. `op.activation_memory * 2` returns `Bytes`. `sum(op.activation_memory for op in ops)` returns `Bytes` thanks to `__radd__`.
- Compatibility: `isinstance(x, int)` still works because `Bytes` IS `int`. Pickle, JSON export, CSV export round-trip as bare numeric (consumer formats columns downstream).
- Raw access when needed: `int(op.activation_memory)` returns the bare integer.

### Serialization

- **`.tlspec`**: writes int/float payload; load path re-wraps based on dataclass type annotation. One central registry in `_io`: `{Bytes, Duration, Flops, Macs} -> wrap callable`.
- **JSON / CSV / Parquet (`to_*`)**: serializes as bare numeric (subclass strips automatically). String-column export option deferred — add when users ask.
- **Pickle**: works without changes.

### Decisions confirmed by JMT 2026-05-24

1. Class name: **`Bytes`** (unit-name parallel with `Duration` / `Flops` / `Macs`).
2. `tl.Quantity` ABC: **yes** — abstract base covering all four formattable unit types.
3. `__format__` support: **yes** — include in initial implementation.
4. Pandas string-column option: **deferred** — default to bare numeric; revisit if users request.

### Cascade scope

- New module `torchlens/units.py` with the ABC + 4 classes + tests (~150 lines).
- Type annotations across all dataclasses: `int`/`float` -> `Bytes`/`Duration`/`Flops`/`Macs` for relevant fields.
- Delete every `*_str` field declaration + construction site.
- Update `_io` registry to re-wrap on load.
- Glossary v7: collapse paired entries; add Conventions section explaining the unit-type family.
- T2 + T3 to confirm no semantic behavior change.



---

## Facets framework — derived semantic views on Op and Module (LOCKED 2026-05-27)

Adds a general-purpose registry of "recipes" that produce derived semantic views on Op and Module records. Headline use case is HuggingFace transformer attention (q/k/v/output access with per-head reshape), but the framework generalizes — anything a user wants to expose as a named view on a record can be registered. Both built-in (shipped by TorchLens) and user-defined recipes use the same mechanism.

### The umbrella term: `facets`

Chosen over `view` (conflicts with `Tensor.view`), `lens` (overloaded by FP), `derived` (generic), `recipe` (the registration thing, not the output), `projection` (conflicts with q/k/v_proj terminology). `facets` reads naturally as "different angles on the same data," works as both the field name (plural collection) and the module name (singular registration verb).

### Field surface on Op and Module

Every `Op` and `Module` record gets a `.facets` field returning a `FacetView` — dict-like AND attribute-access.

```python
op.facets           # FacetView instance
op.facets.q         # the 'q' facet (attribute access)
op.facets['q']      # same — both syntaxes work
op.facets.keys()    # list of available facet names — does NOT trigger computation
op.facets.has('q')  # bool check, no computation
list(op.facets)     # iterates names
len(op.facets)      # how many facets are available
```

Same field, same surface, both on `Op` and `Module`. The recipe that matches determines which scope it applies to.

### `FacetView` semantics

- **Lazy computation**: `view.keys()` returns names cheap; `view.q` invokes the recipe on first access.
- **Per-view caching**: once `view.q` computes, the value is cached on the FacetView instance. `del record.facets` discards the cache; `record.facets` re-creates a fresh FacetView.
- **Not serialized**: facets are derived from the underlying record + the global registry. `.tlspec` save/load drops cached facet values; FacetView is reconstructed on load against the current registry.
- **Provenance**: `view.recipe_source` (str) tells you which recipe produced this view (debugging aid; `None` when no recipe matched).

### Recipe registration

```python
@tl.facets.register(class_name='DistilBertSdpaAttention')
def distilbert_attention(mod):
    n_heads = mod.cls.n_heads
    d_head = mod.cls.dim // n_heads
    B, S = mod.calls[0].input_shapes[0][:2]
    return {
        'q': mod.modules['q_lin'].calls[0].out.view(B, S, n_heads, d_head),
        'k': mod.modules['k_lin'].calls[0].out.view(B, S, n_heads, d_head),
        'v': mod.modules['v_lin'].calls[0].out.view(B, S, n_heads, d_head),
        'attn_out': mod.calls[0].out,
        'n_heads': n_heads,
        'd_head': d_head,
    }
```

Matchers (any combination — recipe fires if ALL specified matchers pass):
- `class_name='Foo'` — short class name; accepts string or tuple of strings
- `class_qualname='full.module.Foo'` — full qualname (disambiguates collisions); string or tuple
- `predicate=callable(record) -> bool` — arbitrary callable on the Op/Module

Recipe function signature: `(record: Op | Module) -> dict[str, Any]`. Return a flat dict of facet name → value. Values can be tensors, scalars, nested structures, callables, anything. The dict is the recipe's contribution to the FacetView; multiple recipes can match the same record (see merge below).

### Multi-recipe merge behavior

When multiple recipes match the same record, their dicts are MERGED into the FacetView. Conflict on the same KEY: last-registered wins, with a `UserWarning` flagging the override. This lets library + user contribute facets to the same class without one nuking the other.

```python
# Library ships:
@tl.facets.register(class_name='DistilBertSdpaAttention')
def _builtin_attention(mod): return {'q': ..., 'k': ..., 'v': ...}

# User adds their own:
@tl.facets.register(class_name='DistilBertSdpaAttention')
def _user_extras(mod): return {'attention_entropy': ..., 'head_similarity': ...}

# Both contribute to the FacetView; both visible in .keys()
```

### Built-in recipes shipped with v1

Initial set covers common HF + torch families. Each recipe includes residual/input access where applicable (residual-stream analysis use case).

- **Attention** (one per HF class flavor; GQA-aware where relevant):
  - `DistilBertSdpaAttention` (separate q_lin/k_lin/v_lin/out_lin)
  - `GPT2Attention` (fused `c_attn` for QKV, separate `c_proj`)
  - `BertSelfAttention` (separate query/key/value)
  - `LlamaAttention` / `LlamaSdpaAttention` (GQA: separate `n_q_heads`, `n_kv_heads`)
  - `MistralAttention` / `MistralSdpaAttention`
  - `T5Attention`
  - `ViTSelfAttention`
- **Normalization**: `LayerNorm`, `RMSNorm`, `LlamaRMSNorm` — facets: `normalized`, `gamma`, `beta` (None for RMS variants), `input`
- **MLP / FFN** (common gated forms): `LlamaMLP`, `MixtralMLP`, `GPT2MLP`, `DistilBertFFN` — facets: `intermediate`, `gated_out` (where applicable), `up_out`, `down_out`, `input`, `output`
- **Embedding**: `nn.Embedding` — facets: `lookup`, `weight`, `indices` (where extractable)

Attention recipes uniformly expose: `q`, `k`, `v`, `attn_out`, `input`, `residual` (when residual stream is identifiable), `n_q_heads`, `n_kv_heads`, `n_heads` (alias for `n_q_heads` for MHA), `d_head`, `head(i)` method returning a sub-view scoped to a single head.

### `AttentionView.head(i)` sub-view

For attention specifically, `.head(i)` returns a sub-view object scoped to one head — `.head(3).q` is `view.q[:, :, 3, :]`, with shape `(B, S, d_head)`. Convenience for the common "show me head 3 of layer 5" pattern.

### Fused-SDPA limitation

For attention modules using PyTorch's fused SDPA (the default in modern HF builds), the attention pattern (post-softmax) is NOT extractable — it lives inside the C++ fused kernel. The `pattern` facet raises an informative error in that case:

```python
view.pattern
# RuntimeError: attention pattern not captured: model uses fused SDPA at <op_label>.
# Re-run with model.config._attn_implementation='eager' to expose the pattern.
```

Not silent None — explicit error tells the user how to fix it.

### Discoverability surface

```python
tl.facets.list()                          # all registered recipes
tl.facets.list(scope='module')            # only Module-scoped
tl.facets.list(scope='op')                # only Op-scoped
tl.facets.list(class_name='*Attention')   # glob match
tl.facets.info('DistilBertSdpaAttention') # which recipes match this class, what facets they provide

record.facets.keys()                       # what's available on this specific record
record.facets.has('q')                     # explicit existence check
record.facets.recipe_source                # which recipe(s) populated this view

# Convenience finders on Trace:
log.attention_blocks()                     # iterate detected attention Modules
log.modules_with_facet('q')                # iterate Modules whose FacetView has a 'q' facet
```

### Serialization rules

- `.tlspec` save/load: facets are derived; cached values dropped on save. FacetView reconstructed lazily on load against the receiving session's registry.
- User-registered recipes must be re-registered in the loading session for their facets to be available after load. Convention: put recipe registrations in a Python module that gets imported by both sides.
- No opt-in to "pickle the registry alongside" for v1; future enhancement if users request.

### Cache lifecycle

Cache lives on the FacetView instance (not on the underlying record). Implication: `record.facets` creates a FacetView per access — but the FacetView is itself cached on the record (lazy-init pattern) so within a session the same FacetView is reused. Explicit cache reset via `record.facets.invalidate()` or by `del record.facets` (re-created on next access, no cache).

### File layout

```
torchlens/semantic/
  __init__.py
  facets.py              # FacetView, register(), list(), info(), registry data
  recipes/
    __init__.py
    attention.py          # all attention recipes
    norm.py               # LayerNorm, RMSNorm variants
    mlp.py                # MLP / FFN families
    embedding.py          # nn.Embedding
```

Top-level exposure: `tl.facets.register`, `tl.facets.list`, `tl.facets.info`, plus the `Op.facets` / `Module.facets` field-level access.

### Cascade scope

- New `torchlens/semantic/` subpackage (~500 lines core + recipes)
- `Op.facets` and `Module.facets` properties on `op_log.py` and `module_log.py`
- `FacetView` class with dict + attribute interface (~100 lines)
- Registry data structures + decorator (~80 lines)
- 8 built-in recipe modules (~300 lines combined)
- Tests: DistilBERT + GPT-2 + BERT + user registration + cache invalidation + multi-recipe merge (~250 lines)
- Glossary v8 entry for the new field family
- `huggingface_tutorial.ipynb` updated with new section demonstrating facets

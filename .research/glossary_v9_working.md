# TorchLens Glossary

> **Reconciliation pass 2026-06-01 (glossary-code-reconcile branch).** A code-vs-glossary
> diff of the PUBLIC API resolved the locked **bare memory passthrough drop** at module scope
> (ModuleCall/Module `out_memory`/`out_memories` removed from code; the bare
> `activation_memory`/`gradient_memory`/`transformed_*_memory` passthrough lines removed from
> the ModuleCall/Module Output Passthroughs sections — the `output_`/`internal_`-prefixed
> cluster covers module-scope memory) and removed the false `tl.log_forward_pass` alias claim
> from the auto-routing section (no such alias exists; the entry point is `tl.trace`).
> **RESOLVED (conformance sprint, commits `bdf4f23`..`3858711` on local main):** the
> implement-in-code path was chosen (per spec-drives-code). The LOCK-BACKED residual set is now
> shipped in code: `is_submodule_input`→`is_module_input`/`is_module_output` (the semantic
> redefinition), `input_to_modules` (distinct from `input_to_module_calls`),
> `Op.args_summary`/`kwargs_summary`, `grad_fn_label` + `grad_fn`-as-resolver, `multi_output_type`,
> the `atomic_module`/`atomic_module_address`/`atomic_module_call_label` resolver cluster,
> `gradient_transform`, `has_saved_gradient`, `Param.is_trainable`, the Buffer overwrite cluster
> (`is_overwritten`/`num_overwrites`/`last_overwrite_source`), Module
> `total_flops`/`total_macs`/`internal_param_memory`, and
> `Module.call_parent_address`/`call_children_addresses`. The **clean-break** decision was honored
> (no deprecation shims — old names removed outright; callers migrated), and the internal record
> module files were de-`_log`'d (`op_log.py`→`op.py`, …, `model_log.py`→`trace.py`). tier-2 green
> (2312 passed, 0 failed). **TorchLens local-main code now FULLY conforms to this glossary** — the
> "code conforms" line below is no longer aspirational.

> **Re-filed to vault 2026-06-02 (v9)** — finalized against the shipped code. Since v8: the **input auto-routing** section, the re-walked 5/21 **parity gaps** (`Op.args_summary`/`kwargs_summary`, the ModuleCall arg-parity cluster), the corrected `Op.grad_fn`/`grad_fn_handle` convention text, and the **5 resolved glossary nits** (phantom `root_call`/`max_call_depth`/`report_values` removed, Layer saved-predicates singular, Op distances `_to_output`). The v7 unit-type family (`tl.Quantity`/`Bytes`/`Duration`/`Flops`/`Macs`, `*_str` deleted, `memory→activation_memory`) is now **implemented in code**. **TorchLens local-main code FULLY conforms to this glossary** — the lock-backed residual set was implemented in the 2026-06 conformance sprint (commits `bdf4f23`..`3858711`); see the reconciliation note above. (Supersedes `2026-05-27-glossary-v8`.)

> **v9 (2026-05-31)** — Adds the **Input auto-routing** section (`tl.autoroute.input` priority registry + the HuggingFace `trace_text` / `trace_image` / `trace_multimodal` bridge tracers that `tl.trace` dispatches to by input type). Closes two verified 5-21 parity gaps re-walked from the lock log: `Op.args_summary` / `Op.kwargs_summary`, and the ModuleCall `forward_arg_names` / `num_forward_args_total` / `num_forward_pos_args` / `num_forward_kwargs` / `has_saved_forward_args` cluster. Removes a stale "Op input convenience fields deferred" bullet (those landed under the 5-23 input-tensor lock). Auto-routing carries no walkthrough lock — it documents the shipped code names (no competing locked target exists in the lock log). **OPEN for JMT:** Op `min/max_distance_to_output` vs Layer `min/max_distance_to_output` naming asymmetry — no lock dictates harmonizing, left as-is pending a call.
>
> **v8 (2026-05-27)** — Adds the **Facets framework** (`tl.facets.*` + `.facets` field on Op and Module). A general-purpose registry of "recipes" that produce derived semantic views on records — q/k/v/output access on attention modules, normalized/gamma/beta on LayerNorms, intermediate/output on MLPs, and anything users register. Ships with built-in recipes for common HF + torch families. See new "Facets framework" subsection in Conventions for full details, plus new bolded entries under Op, Module, and a dedicated `## Facets` section at the bottom. Lock entry: `## Facets framework ... (LOCKED 2026-05-27)` in `.project-context/glossary_walkthrough_deltas.md`.
>
> **v7 (2026-05-24)** — Adds the `tl.Quantity` unit-type family (`tl.Bytes`, `tl.Duration`, `tl.Flops`, `tl.Macs`) and **DELETES every `*_str` companion field** across the API surface (~76 fields removed). Formatted display now lives in `__str__` / `__format__` on the numeric types themselves — `print(op.activation_memory)` produces `"1.2 MB"` directly. See the new "Numeric quantity types" subsection in Conventions for full details. Lock entry: `## Unit type family ... (LOCKED 2026-05-24)` in `.project-context/glossary_walkthrough_deltas.md`.
>
> **v6 (2026-05-24, superseded by v7)** — Built by restoring the clean v3 (2026-05-19) baseline and applying ONLY decisions logged with a `LOCKED` marker in `.project-context/glossary_walkthrough_deltas.md`. Every new or changed entry is **bolded** and annotated with its lock date so it can be traced back to a specific deltas-log section. v4 (2026-05-21) and v5 (2026-05-23) are DEPRECATED — they contained hallucinated phantom entries (e.g., `report_values`, `orphan_logs`/`OrphanAccessor`, separate `conditional_records` after it had been folded into `conditionals`) introduced by auto-extraction from the dataclasses rather than from the lock log. Do not re-introduce those.
>
> **Scope so far (v9):** the v6 5-23 locks, the v7 unit-type cleanup, the v8 facets framework, and a v9 re-walk of the 5-21 rename-sprint locks against the lock log. An adversarial cross-check (Claude + Codex) confirmed the 5-21 locks are substantially already applied in v7/v8; v9 closed the two remaining gaps — `Op.args_summary`/`kwargs_summary` and the ModuleCall arg-parity cluster — corrected the stale `Op.grad_fn`/`grad_fn_handle` convention text, and added the input auto-routing section. The pre-existing v8 inconsistencies the cross-check surfaced (bare memory passthroughs; `root_call`/`max_call_depth`/`report_values` lock-backing; Layer plural saved-predicates; Op distance naming) were RESOLVED 2026-06-02 (JMT-approved): the unlocked/phantom entries were removed, Layer saved-predicates singular-ized, Op distances harmonized to `_to_output`. The TorchLens local-main code now conforms to this glossary.
>
> Public API reference. Reflects the locked naming targets from the rename
> sprint walkthrough deltas (2026-05-03 through 2026-05-19). This is the v3
> revision that adds the locked Reference Form Convention (Principles 1-4
> for label-vs-object refs) and applies the v2→v3 audit findings.
>
> **v3 changes from v2 (2026-05-19 through 2026-05-21):** This v3 revision captures the FULL rename-sprint pass through the glossary, including the final parity sweep across passthrough class pairs (Op/Layer, ModuleCall/Module, GradFnCall/GradFn). All locked decisions logged in `.project-context/glossary_walkthrough_deltas.md`. Ready for implementation-sprint planning.
> - New Conventions subsection: Cross-class references (labels vs objects;
>   `_label` + `@property` resolver pattern)
> - Added `Op.layer` (`@property`) resolver — most common cross-class access
> - Clarified storage form on ambiguous fields: `Op.module`, `Op.module_call_stack`,
>   `Op.atomic_module_call`, `Layer.module`, `Buffer.module`, `Buffer.buffer_source`,
>   `Buffer.last_overwrite_source`, `Module.call_parent`, `Module.call_children`,
>   `Trace.root_module`
> - Split `ModuleCall.outputs` into `output_labels` (storage) + `outputs` (`@property`)
> - Swapped `Op.grad_fn` to return TL GradFn record (matching `op.module` / `op.params` / `op.layer` pattern); runtime PyTorch handle moved to `Op.grad_fn_handle`. The `_log` exception is gone.
> - Removed `Param.modules` plural resolver (Principle 4 violation); kept
>   `Param.module` singular and `Param.all_module_addresses` label list
> - Documented `Trace.parent_trace`, `Trace.root_trace`, all `.trace` back-pointers
>   as runtime-only object refs (Principle 2)
> - Resolved `ModuleCall.layers` field-name confusion (renamed to `ops` for
>   the per-pass Op labels; `input_layers` / `output_layers` for the call's
>   input/output Layer labels — all bare per the locked label-list convention)

## How to read this glossary

This glossary describes the target public API after the rename sprint, not
the current implementation names. Names come from
`.project-context/glossary_walkthrough_deltas.md`.

- An `Op` is one captured tensor-producing operation (or graph-boundary event).
- A `Layer` is the stable label that can aggregate one or more Ops at the same graph position.
- A `ModuleCall` is one invocation of an `nn.Module.forward`.
- A `Module` aggregates all calls of the same module address.
- Labels are TorchLens structured identifiers; addresses are PyTorch dotted paths.
- Example: `conv2d_1_2` is a Layer label, while `conv2d_1_2:1` is an Op label.

Most index fields are 1-based unless an entry explicitly says otherwise
(`ordinal_index` is the deliberate exception — 0-based for Pythonic `trace[N]`
indexing). Private storage names, dropped aliases, and underscore-prefixed
locked names are omitted from the main entries. Deferred items are listed at
the end instead of promoted as final API.

## Conventions

These conventions apply uniformly across all log classes. Learn them once;
each individual field below follows the rules.

### Class naming: drop `Log` suffix uniformly

All sub-Trace classes lose the `Log` suffix. `Trace` already broke the `Log`
pattern; the rest follow for symmetry.

| Class | Was | New |
|---|---|---|
| Op record | `OpLog` | `Op` |
| Layer record | `LayerLog` | `Layer` |
| Module record | `ModuleLog` | `Module` |
| ModuleCall record | `ModuleCallLog` | `ModuleCall` |
| Param record | `ParamLog` | `Param` |
| Buffer record | `BufferLog` | `Buffer` |
| GradFn record | `GradFnLog` | `GradFn` |
| GradFnCall record | `GradFnCallLog` | `GradFnCall` |

`Trace` is unchanged. The Super* family (`SuperOp`, `SuperLayer`,
`SuperModule`, `SuperModuleCall`, `SuperParam`, `SuperBuffer`, `SuperGradFn`,
`SuperGradFnCall`) was always short-form; only their type parameters cascade.

`Module` is intentionally namespaced — `tl.Module` (qualified) avoids
collision with `torch.nn.Module` in documentation and idiomatic usage.

### Cross-class references — labels vs objects

Cross-class references follow four principles:

1. **Portability dictates storage form.** Refs that survive `.tlspec` save/load
   are stored as **label strings**. Refs that are runtime-only (die on save/load
   anyway) are stored as **direct object references**.
2. **Tensors, runtime callables, and runtime handles are object-typed by nature.**
   They cannot be labels — they ARE the runtime data.
3. **Frequently-resolved cross-class refs get `_label` (storage) + bare-name
   (`@property`) resolver.** Storage is the label; the `@property` resolves
   via `trace[label]` and raises if the label cannot be resolved.
4. **Uncommon and collection-valued refs stay as label strings.** No parallel
   resolved-collection properties (they bloat the API for marginal ergonomics).
   **Exception:** when a plural is the natural pair of a singular resolver in
   the same conceptual cluster (parent/children, source/sources, owner/owners),
   the plural ALSO gets a resolver — return-type consistency across a paired
   cluster outweighs the general "no plural resolvers" rule. Storage form for
   the plural is `<entity>s_addresses` or `<entity>s_labels`; the resolver
   takes the bare plural name.

| Pattern | Storage | Public access | Example |
|---|---|---|---|
| Common cross-class ref | `<entity>_label: str` | `<entity>` (`@property`) | `GradFn.op_label` + `GradFn.op` |
| Uncommon cross-class ref | label string | label string only | `Op.equivalent_ops` |
| Cross-class collection | label list / set | label list / set | `Op.parents`, `Op.children` |
| Cross-class collection (high-traffic) | Accessor | Accessor returning objects | `Trace.layers`, `Module.calls` |
| Tensor | object | object | `Op.out`, `Op.grad` |
| Runtime callable / handle | object | object | `Op.func`, `Op.grad_fn_handle` |
| Back-pointer | object (weakref) | object | `Op.trace` |

```python
# Frequent access uses the @property resolver:
for op in trace.ops:
    print(op.layer.activation_memory)  # op.layer resolves op.layer_label

# Uncommon and collection access stays explicit:
for parent_label in op.parents:
    parent = trace[parent_label]
    print(parent.activation_memory)
```

Runtime-only refs (`.trace` back-pointers, `Op.func`, `Op.grad_fn_handle`,
`Trace.parent_trace`, `Trace._source_model_ref`) are NOT serialized and are
unavailable after `.tlspec` load.

### Op fields default to OUTPUT

Op fields about tensor flow refer to the op's OUTPUT tensor. The op IS the
producer of its output; what "flows through the graph" from this op is its
output.

- `op.shape`, `op.dtype`, `op.activation_memory` — properties of the output
- `op.out` — the output tensor itself
- `op.module_calls_entered` — modules where the OUTPUT entered
- `op.module_entry_arg_keys` — arg positions at those entry events
- `op.output_of_module_calls` — modules this op IS the output of

### `total_X` prefix = aggregate sum

The `total_*` prefix uniformly denotes "aggregate sum across constituent
units." Scope is determined by the parent class:

| Scope | `total_X` means |
|---|---|
| Trace | sum across all Ops in the trace |
| Layer | sum across all Ops in this Layer's equivalence class |

Numeric per-pass values on Layer also expose a bare form
(`layer.activation_memory`) that delegates to the single Op for single-pass
Layers and raises `ValueError` for multi-pass Layers (use
`layer.ops[N].activation_memory` instead).

`peak_` is distinct from `total_` — `forward_peak_memory` /
`backward_peak_memory` are maxima, not sums.

### Numeric quantity types (LOCKED 2026-05-24)

Every quantitative field with units (memory in bytes, durations in seconds, FLOPs, MACs) returns a typed wrapper from the `tl.Quantity` family rather than a bare `int` or `float`. The wrapper IS the underlying numeric type (subclass), so arithmetic and comparisons work transparently — the only thing the wrapper adds is formatted display.

This replaces the previous `*_str` companion-field pattern. **There are no `_str` fields in the API.** `print(op.activation_memory)` produces `"1.2 MB"` directly, via the wrapper's `__str__`.

#### The family

| Type | Base | Wraps | Default `__str__` example |
|---|---|---|---|
| **`tl.Bytes`** | `int` | memory in bytes | `1234567` → `"1.2 MB"` |
| **`tl.Duration`** | `float` | seconds | `0.00345` → `"3.45 ms"` |
| **`tl.Flops`** | `int` | floating-point op count | `1230000000` → `"1.23 GFLOPs"` |
| **`tl.Macs`** | `int` | multiply-accumulate count | `615000000` → `"615 MMACs"` |

**`tl.Quantity`** is the abstract base class encompassing all four. Use `isinstance(x, tl.Quantity)` to test "is this a formattable unit type" — useful in serialization registries, summary renderers, and downstream tooling.

Counts (`num_*` fields, `step_index`, `call_index`, etc.) stay as bare `int` — they don't carry units, so they don't need formatting.

#### Inline ergonomics

```python
op.activation_memory           # Bytes(1234567)  -- a tl.Bytes instance
print(op.activation_memory)    # "1.2 MB"        -- __str__ formats
f"size: {op.activation_memory}"  # "size: 1.2 MB"  -- f-strings call __str__
int(op.activation_memory)      # 1234567         -- raw int when needed
op.activation_memory + 1024    # Bytes(1235591)  -- arithmetic preserved
op.activation_memory > 1e6     # True            -- comparisons work (it IS int)
isinstance(op.activation_memory, int)  # True    -- it's an int subclass
```

#### Type-preserving aggregation

`sum()`, arithmetic operators, and unary operations preserve the wrapper type via overloaded `__add__` / `__radd__` / `__sub__` / `__mul__` / etc. The reflected-operator rule for subclasses (Python data model) ensures `sum()` dispatches through `Bytes.__radd__` even when its initial value is `0`:

```python
total = sum(op.activation_memory for op in trace.ops)   # tl.Bytes
print(total)                                             # "12.4 GB"
```

#### Operator semantics

| Operation | Result | Reason |
|---|---|---|
| `Bytes + Bytes` | `Bytes` | aggregation |
| `Bytes + int` (or `int + Bytes`) | `Bytes` | scalar promotion |
| `Bytes - Bytes` | `Bytes` | difference of magnitudes |
| `Bytes * scalar` | `Bytes` | scaling |
| `Bytes * Bytes` | `TypeError` | "bytes squared" is meaningless |
| `Bytes / scalar` | `Bytes` | scaling |
| `Bytes / Bytes` | `float` | ratio, dimensionless |
| `Bytes // Bytes` | `int` | discrete ratio |
| `-Bytes`, `abs(Bytes)` | `Bytes` | sign preserved |
| `min`, `max`, `sorted` | `Bytes` | inherited via `int.__lt__` etc. |
| cross-type (e.g. `Bytes + Duration`) | `TypeError` | units don't mix |

Same template applies to `Duration` / `Flops` / `Macs`.

#### `__format__` support

Each Quantity supports unit selection via Python's format-spec mini-language:

```python
f"{op.activation_memory}"        # "1.2 MB"      -- default formatted
f"{op.activation_memory:raw}"    # "1234567"      -- raw numeric
f"{op.activation_memory:MB}"     # "1.235 MB"     -- forced units
f"{op.activation_memory:.2f MB}" # "1.23 MB"      -- precision + units
f"{op.func_duration:ms}"         # "3.45 ms"      -- Duration in milliseconds
f"{op.flops_forward:GFLOPs}"     # "1.23 GFLOPs"  -- Flops in giga
```

Exact unit-suffix grammar finalized at implementation time.

#### Serialization

- **`.tlspec`**: subclasses write as bare numeric payload; load re-wraps based on the dataclass type annotation. One central type registry in `_io`.
- **JSON / CSV / Parquet via `to_*` exporters**: serialize as bare numeric. Consumers format columns downstream if they want display strings. (String-column export option deferred — add if users request.)
- **Pickle**: works without changes (int/float subclasses pickle as their value).

#### What about user code that previously read `_str`?

```python
# OLD
print(op.activation_memory_str)         # "1.2 MB"

# NEW (one fewer character; identical output)
print(op.activation_memory)             # "1.2 MB"
```

All `_str` fields are gone. Drop the `_str` suffix from any consumer code.

### Facets framework (LOCKED 2026-05-27)

**Every `Op` and `Module` record carries a `.facets` field** — a `FacetView` that exposes named, derived, semantic accessors on the record. Built-in recipes ship for common HF and torch model components (attention blocks, LayerNorm, MLPs, embeddings); users register their own with `@tl.facets.register(...)`.

The framework solves: "give me a named handle on parts of this tensor / module that the raw graph doesn't carry." Q/K/V access with per-head reshape on attention modules. `normalized` and `gamma` on a LayerNorm. `lookup` and `weight` on an embedding. Anything domain-specific a user wants to expose.

**Dict-like AND attribute-access** (both syntaxes work):

```python
attn = log.modules['transformer.layer.0.attention']

attn.facets.q                # tensor, shape (B, S, n_heads, d_head)
attn.facets['q']             # same — dict subscript
attn.facets.keys()           # ['q', 'k', 'v', 'attn_out', 'n_heads', 'd_head', ...]
attn.facets.has('pattern')   # bool — no computation
list(attn.facets)            # iterate facet names
len(attn.facets)             # count
```

**Lazy with per-view caching**: `.keys()` is cheap (just lists names from the matched recipes); `.q` triggers the recipe's compute on first access, then caches. `del record.facets` discards the cache; next access creates a fresh FacetView.

**Recipe registration**:

```python
@tl.facets.register(class_name='MyCustomBlock')
def my_recipe(mod):
    return {
        'foo': mod.layers['inner'].out.view(...),
        'bar': mod.params['weight'].value,
    }
```

Matchers: `class_name`, `class_qualname` (or tuples for multiple), `predicate` (arbitrary callable on the record). All specified matchers must pass.

**Multi-recipe merge**: when multiple recipes match the same record, their dicts merge into one FacetView. Last-registered wins on key conflicts, with a UserWarning. Library + user can both contribute facets to the same class.

**Discoverability**:

```python
tl.facets.list()                          # all registered recipes
tl.facets.list(class_name='*Attention')   # glob match
tl.facets.info('DistilBertSdpaAttention') # what facets this class gets
log.attention_blocks()                    # iterate Modules matched by attention recipes
log.modules_with_facet('q')               # iterate Modules whose FacetView has a 'q' facet
record.facets.recipe_source               # which recipe populated this view
record.facets.invalidate()                # drop cache
```

**Serialization**: facets are derived, not serialized. `.tlspec` save/load drops cached values; FacetView is reconstructed on load against the current registry. User-registered recipes must be re-registered in the loading session for their facets to be available.

**Attention-specific sub-view**: attention recipes uniformly expose a `.head(i)` method returning a sub-view scoped to one head — `view.head(3).q` is `view.q[:, :, 3, :]` with shape `(B, S, d_head)`.

**Fused-SDPA limitation**: when an attention module uses PyTorch's fused SDPA kernel (default in modern HF builds), the attention pattern is NOT extractable — `view.pattern` raises an informative `RuntimeError` telling the user to re-run with `model.config._attn_implementation='eager'` if they need it. Not silent None — explicit error keeps the failure mode discoverable.

### Tensors short, properties qualified

Tensor-attribute style is short; property/concept style is full.

| Category | Form | Examples |
|---|---|---|
| Tensor objects | short | `Op.out` (activation), `Op.grad` (gradient) |
| Capture config flags | full | `save_raw_activations`, `save_gradients` |
| Compound property qualifiers | full | `activation_memory`, `gradient_memory`, `activation_transform` |
| Predicates | full | `has_saved_activation`, `has_saved_gradient` |

### `id` vs `label` — Python runtime vs TorchLens-internal

| Suffix | Meaning | Examples |
|---|---|---|
| `_object_id` | Python `id()` runtime value. Within-session only; NOT portable across processes. | `Trace.model_object_id`, `Op.grad_fn_object_id`, `Trace.backward_root_grad_fn_object_ids` |
| `label` (bare) or `_label` (qualified) | TorchLens-internal stable identifier. Portable; survives `.tlspec` save/load. | `Op.label`, `Conditional.label`, `ConditionalRoleRef.conditional_label`, `Op.layer_label` |

The bare `_id` suffix is reserved — never use it without the `_object_` qualifier. Forces explicit disambiguation; users immediately know whether they're looking at a runtime-only Python id or a portable TorchLens label.

### `_label` is user-overridable; `_name` is auto-derived

| Suffix | Meaning |
|---|---|
| `_label` | User-overridable identifier (e.g., `trace_label`, `model_label`) |
| `_name` | Auto-derived name (e.g., `model_class_name`, `Module.name` = last address segment) |

### Class object vs class name vs class qualname

| Convention | Form | Example |
|---|---|---|
| Class object | `cls` (or `<entity>_cls`) | `Module.cls`, `Param.module_cls`, `Op.grad_fn_cls` |
| Short class name | `class_name` (or `<entity>_class_name`) | `Module.class_name`, `Param.module_class_name` |
| Full qualified class name | `class_qualname` (or `<entity>_class_qualname`) | `Module.class_qualname`, `Param.module_class_qualname` |

`cls` is the Python convention because `class` is a keyword.

### Class-level constants

| Was | New |
|---|---|
| `DEFAULT_VALUES` | `FIELD_DEFAULTS` |
| `FORK_POLICY` | `FIELD_FORK_POLICY` |
| `SAVE_POLICY` | `FIELD_SAVE_POLICY` |

The shared `FIELD_` prefix forms a family marker.

### Buffers are gradient dead-ends

Buffers receive values from upstream ops, but they don't propagate gradients
back through the graph in the same way activations do. Gradient-related
buffer fields exist for symmetry but most static buffers have no gradient
record.

### Each graph event is its own Op (two orthogonal axes)

The graph is a sequence of events; every event gets a node. Op subtypes
participate uniformly in the equivalence/Layer machinery. Two orthogonal
axes describe an Op:

**Axis 1: Op kind (mutually exclusive)** — is this Op a torch function call, or a graph-boundary event?

| Op kind | When | Predicate |
|---|---|---|
| **Compute Op** | A torch function was executed (`conv2d`, `relu`, `torch.ones`, `torch.rand`, factory functions, ALL torch calls) | `is_compute_op = True` |
| **Input boundary** | Marks model input entering the graph | `is_input = True` |
| **Output boundary** | Marks model output leaving the graph | `is_output = True` |
| **Buffer source** | Marks a buffer value entering the graph | `is_buffer_source = True` |

Boundary Ops do NOT execute a torch function — they mark a transition. Compute Ops DO execute a torch function. The three boundary types and "compute" are mutually exclusive.

**Axis 2: Connectivity descriptors (orthogonal flags, apply to compute Ops)** — how is this compute Op connected to the main input→output flow?

| Flag | Meaning |
|---|---|
| `is_internal_source = True` | Compute Op with no external input ancestor (e.g., `torch.ones`, `torch.rand`, factory functions) |
| `is_internal_sink = True` | Compute Op whose output isn't used by any model output |
| `is_orphan = True` | Compute Op disconnected from both input AND output flow (combined internal source + sink) |

These are NOT alternative Op kinds — they're additional flags on compute Ops. A `torch.ones` call is a compute Op that ALSO has `is_internal_source = True`.

Use `Trace.compute_ops` / `Op.is_compute_op` to filter to compute Ops (includes internal-source / internal-sink / orphan compute Ops). Use `Trace.boundary_ops` (if added) or `not op.is_compute_op` for the boundary set.

## Top-level vocabulary

- `tl.trace(model, x, *, backend=None)`: Resolve a registered backend, run capture, and return
  a `Trace`. `backend=None` preserves torch eager default plus MLX module auto-routing; explicit
  backend names fail early on unknown, ambiguous, or mismatched selections. Capture-only —
  visualization is `trace.draw()`.
- `Trace`: Top-level object for one captured model execution, including graph, tensors, modules, params, buffers, gradients, and metadata.
- `Trace.backward(loss)`: Add backward-pass data to an existing Trace by backpropagating an explicit loss.
- `Layer`: A stable graph position such as `conv2d_1_2`; it aggregates the Ops for that position.
- `Op`: One tensor operation invocation such as `conv2d_1_2:1`; includes torch function calls, graph I/O sentinels, internal source/sink sentinels, and buffer source entries. This is where saved `out`, `grad`, args, timing, and function data live.
- `Module`: Aggregate record for one module-like address; `Trace.module_identity_mode` states
  whether records came from torch modules, future pytree modules, or a function root.
- `ModuleCall`: One actual call of a module-like forward/root call.
- `Param`: Record for one parameter leaf; `Trace.param_source` states whether records came from a
  native module, future pytree derivation, or no parameter source.
- `Buffer`: Persistent record for one PyTorch registered-buffer address, with version nodes, initial/final values, and update metadata.
- `GradFn`: Record for one true-backward autograd `grad_fn` node in the backward graph.
- `GradFnCall`: One hook firing or backward call event for a `GradFn`.
- `Bundle`: Coordinated container for comparing multiple Traces.
- `TraceAccessor`: Dict-like Bundle accessor for named Traces.
- `SuperOp`, `SuperLayer`, `SuperModule`, `SuperModuleCall`, `SuperParam`, `SuperBuffer`, `SuperGradFn`, `SuperGradFnCall`: Cross-trace views aligning the same label across Bundle members.
- `SuperOpAccessor`, `SuperLayerAccessor`, etc.: Bundle accessors returning the corresponding Super* objects.

## Trace (per `tl.trace(model, x, *, backend=None)`)

### Identity

- `model_class_name`: Short Python class name for the source model (e.g., `"ResNet"`). Auto-derived from `type(model).__name__`.
- `model_class_qualname`: Fully qualified class name for the source model, useful for verifying reruns.
- `model_cls` (`@property`): Runtime Python type object for the source model; `type(_source_model_ref())` if the model is still alive. Unavailable after portable load if the class cannot be resolved.
- `model_label`: User-overridable display label for the model; defaults to `model_class_name`.
- `model_object_id`: Stored Python `id()` of the model object at capture time. Useful for within-session relationship checks. NOT comparable across processes — different runs yield different ids for the same object.
- `trace_label`: User-set label for this Trace within forks or Bundles; defaults to `model_label`. Used for Bundle alignment.
- `parent_trace`: Runtime-only object reference to the immediate parent Trace when this Trace is forked; `None` for an original Trace. Not portable (Trace has no global label registry; only Bundle resolves Trace names within its membership).
- `root_trace`: Runtime-only object reference to the root Trace of a fork tree; `None` on the root itself. Same portability caveat as `parent_trace`.
- `state`: Enum-like `TraceState` value describing whether the Trace is `PRISTINE`, `SPEC_STALE`, `REPLAY_PROPAGATED`, `RERUN_PROPAGATED`, `LIVE_CAPTURED`, `DIRECT_WRITE_DIRTY`, or `APPENDED`.
- `tlspec_version`: Portable `.tlspec` format version used for save/load compatibility.
- `backend`: `BackendName` identifying the registered capture backend. Survives portable
  save/load. Shipped user-facing specs are `"torch"` and technical-preview `"mlx"`; `"jax"`
  and `"tinygrad"` are reserved preview/future names until their build phases land.
- `module_identity_mode`: `Literal["torch_module", "pytree_module", "function_root"]`
  describing the source of module-like records. Current torch/MLX captures use
  `"torch_module"`; `"function_root"` and `"pytree_module"` are for non-torch functional and
  pytree adapters.
- `param_source`: `Literal["native-module", "pytree-derived", "none"]` describing the source of
  `Trace.params`. Torch/MLX use `"native-module"` when parameters exist; metadata-only fake or
  function-root traces can use `"none"`.
- `FIELD_DEFAULTS`: Class-level defaults applied at initialization and cleanup.
- `FIELD_FORK_POLICY`: Class-level per-field policy for `fork()`.
- `FIELD_SAVE_POLICY`: Class-level per-field policy for portable save/load.

### Counts

- `num_layers`: Number of distinct Layer records, INCLUDING input, output, and buffer boundary Layers. Equals `len(trace.layers)`.
- `num_compute_layers` (`@property`): Number of compute Layers (Layers whose representative Op is compute, i.e., NOT an input / output / buffer-source boundary). Compute Layers may carry connectivity flags like `is_internal_source` or `is_internal_sink`. Equals `len(trace.compute_layers)`.
- `num_ops`: Number of Op records in this Trace, including compute Ops AND boundary Ops (input, output, buffer source). Equals `len(trace.ops)`. NOT equal to `num_compute_ops`.
- `num_compute_ops` (`@property`): Number of compute Ops (Ops that executed a torch function); EXCLUDES boundary Ops (input / output / buffer source). Includes Ops with `is_internal_source` / `is_internal_sink` / `is_orphan` flags since those ARE compute Ops with connectivity descriptors. Equals `len(trace.compute_ops)` and `max(op.step_index for op in trace.compute_ops)`.
- `num_edges` (`@property`): Distinct (parent,child) pairs in the per-pass OP graph, INCLUDING boundary sentinel edges (input->first op, last op->output) and buffer edges. Dedupe pairs (a parent feeding a child via two arg slots counts ONCE). Must equal both sum-over-ops-of-distinct-children and sum-over-ops-of-distinct-parents.
- `num_compute_edges` (`@property`): Edges where BOTH endpoints are compute ops (`is_compute_op` on both) — excludes any edge incident to an input/output/buffer node.
- `num_buffer_edges` (`@property`): Edges with at least one `is_buffer` endpoint.
- `num_layer_edges` (`@property`): Distinct (parent_layer, child_layer) pairs in the aggregate LAYER graph (use Layer.parents/children). The rolled-graph edge count; parallels `num_layers`.
- `num_backward_edges` (`@property`): Distinct (parent,child) pairs in the GradFn backward graph; `None` when no backward/gradients were captured.
- `branching_factor` (`@property`): mean fan-out = children (consumers) per compute Op, `sum(op.num_children for op in compute_ops) / num_compute_ops`; `0.0` when there are no compute Ops. Computed over a single consistent node set (compute Ops) so the ratio is coherent: ~1.0 for a plain chain, >1.0 with reuse (residual streams, shared embeddings, dense connectivity).
- `max_in_degree` (`@property`): Max `num_parents` over `compute_ops` (the meaningful gather hotspot, e.g. DenseNet concat); `0` if no compute ops. Computed over compute_ops, NOT boundary sentinels.
- `max_out_degree` (`@property`): Max `num_children` over `compute_ops` (most-reused tensor / residual stream); `0` if none.
- `num_saved_ops`: Number of Ops whose forward activation was saved.
- `num_saved_layers`: Number of Layers containing >= 1 saved Op.
- `num_saved_module_calls`: Number of ModuleCalls whose output Op has a saved activation.
- `num_saved_modules`: Number of Modules with >= 1 saved ModuleCall.
- `num_saved_grad_ops`: Number of Ops with saved gradient.
- `num_saved_grad_layers`: Number of Layers containing >= 1 saved-grad Op.
- `num_saved_grad_module_calls`: Number of ModuleCalls whose output Op has a saved gradient.
- `num_saved_grad_modules`: Number of Modules with >= 1 saved-grad ModuleCall.
- `num_saved_grad_fn_calls`: Number of GradFnCalls with saved grad payloads.
- `num_saved_grad_fns`: Number of GradFns with at least one saved GradFnCall.
- `num_module_calls`: Total per-invocation ModuleCall records across all Modules.
- **`num_modules`**: **Number of registered submodules in the source model. Equals `len(trace.modules)`. (LOCKED 2026-05-23)**
- `num_grad_fn_calls`: Total per-invocation GradFnCall records across all GradFns.
- `num_grad_fns`: Number of unique autograd grad-fn nodes captured for backward analysis.
- `num_grad_fns_with_op`: Number of grad-fn nodes paired with a forward Op.
- `num_grad_fns_without_op`: Number of grad-fn nodes that have no corresponding forward Op.
- `num_params`: Total scalar parameter count across the model.
- `num_params_trainable`: Total scalar parameter count with `requires_grad=True`.
- `num_params_frozen`: Total scalar parameter count with `requires_grad=False`.
- `num_param_tensors`: Number of `nn.Parameter` tensor objects.
- `num_param_tensors_trainable` (`@property`): Number of trainable parameter tensor objects.
- `num_param_tensors_frozen` (`@property`): Number of frozen parameter tensor objects.
- `num_layers_with_params`: Number of Layers that use parameters.
- `num_ops_with_params`: Number of Op invocations that use parameters.
- `num_backward_passes`: Number of backward passes logged into this Trace.
- `num_conditionals` (`@property`): Number of conditionals captured.
- `num_input_layers` (`@property`): Number of input boundary Layers. Equals `len(trace.input_layers)`.
- `num_input_ops` (`@property`): Number of input boundary Ops (flat across passes). Equals `len(trace.input_ops)`.
- `num_output_layers` (`@property`): Number of output boundary Layers. Equals `len(trace.output_layers)`.
- `num_output_ops` (`@property`): Number of output boundary Ops (flat across passes). Equals `len(trace.output_ops)`.
- `num_buffer_layers` (`@property`): Number of buffer boundary Layers. Equals `len(trace.buffer_layers)`.
- `num_internal_source_layers` (`@property`): Number of Layers representing internal-source positions.
- `num_internal_source_ops` (`@property`): Number of Ops representing internal-source positions (flat across passes). Equals `len(trace.internal_source_ops)`.
- `num_internal_sink_layers` (`@property`): Number of Layers representing internal-sink positions.
- `num_internal_sink_ops` (`@property`): Number of Ops representing internal-sink positions (flat across passes). Equals `len(trace.internal_sink_ops)`.
- `num_orphans` (`@property`): Number of orphan Ops (disconnected from main input→output flow). Equals `len(trace.orphans)`.
- `num_uncalled_modules` (`@property`): Number of Modules registered on the source model but not called in this capture. Equals `len(trace.uncalled_modules)`.
- `has_trainable_params` (`@property`): True when the model has at least one trainable parameter (`num_params_trainable > 0`).
- `has_frozen_params` (`@property`): True when the model has at least one frozen parameter (`num_params_frozen > 0`).

### Memory

- `total_activation_memory`: Bytes for all Op activations, whether saved or not.
- `saved_activation_memory`: Bytes for Op activations actually saved by TorchLens.
- `total_gradient_memory`: Bytes for all Op output gradients computed during backward.
- `saved_gradient_memory`: Bytes for Op output gradients actually saved.
- `total_param_memory`: Bytes used by all model parameter tensors.
- `total_param_gradient_memory`: Bytes for all populated parameter gradients.
- `forward_peak_memory`: Peak observed memory during forward; exact peak semantics depend on backend.
- `backward_peak_memory`: Peak observed memory during backward; exact peak semantics depend on backend.
- `total_autograd_memory`: Bytes PyTorch autograd retained for backward computation (PyTorch's "saved tensors for backward"). Renamed from `total_autograd_saved_memory` to drop the "saved" overloading.

### Layer and Op Collections

- `layer_labels`: Layer labels in execution order, without `:N` pass suffixes.
- `op_labels`: Op labels in execution order, always including `:N` pass suffixes.
- `layers`: Accessor for Layers. ALWAYS returns Layer records. Accepts bare Layer labels (`conv2d_1_2`) directly, AND pass-qualified Op labels (`conv2d_1_2:1`) by stripping the `:N` to find the parent Layer. 0-based positional integer indexing (`trace.layers[0]` → first Layer).
- `ops`: Accessor for Ops. ALWAYS returns Op records. Accepts pass-qualified Op labels (`conv2d_1_2:1`) directly. For **single-pass Layers**, also accepts the bare Layer label (`conv2d_1_2`) → resolves to the unique Op. For multi-pass Layers, the bare Layer label raises `AmbiguousOpLookupError` (use `[N]` or `:N`-qualified). 0-based positional integer indexing (`trace.ops[N]` uses `op.ordinal_index`).

Example: `trace.layers["conv2d_1_2"]` returns the Layer; `trace.ops["conv2d_1_2:1"]` returns its first Op.

- `compute_ops`: Accessor filtered to compute Ops (Ops that executed a torch function). Excludes boundary Ops (`is_input` / `is_output` / `is_buffer_source`). INCLUDES internal-source / internal-sink / orphan compute Ops (e.g., `torch.ones`, `torch.rand`) — those carry connectivity flags but ARE compute Ops.
- `compute_layers`: Accessor filtered to Layers whose representative Op is compute. Same inclusion logic as `compute_ops`.
- `saved_ops`: Accessor for Ops where `has_saved_activation = True`.
- `saved_layers`: Accessor for Layers containing >= 1 saved Op.
- `saved_module_calls`: Accessor for ModuleCalls whose output Op has saved activation.
- `saved_modules`: Accessor for Modules with >= 1 saved ModuleCall.
- `saved_grad_ops`: Accessor for Ops with saved gradient.
- `saved_grad_layers`: Accessor for Layers containing >= 1 saved-grad Op.
- `saved_grad_module_calls`: Accessor for ModuleCalls whose output Op has saved gradient.
- `saved_grad_modules`: Accessor for Modules with >= 1 saved-grad ModuleCall.
- `saved_grad_fn_calls`: Accessor for GradFnCalls with saved grad payloads.
- `saved_grad_fns`: Accessor for GradFns with >= 1 saved GradFnCall.
- `module_calls`: Accessor for all per-invocation ModuleCall records. ALWAYS returns ModuleCall. Accepts ModuleCall labels (`encoder.block:1`) directly. For **single-call Modules**, also accepts the bare Module address (`encoder.block`) → resolves to the unique ModuleCall. For multi-call Modules, bare address raises (use `[N]` or `:N`-qualified). 0-based positional integer indexing.
- `grad_fn_calls`: Accessor for all per-invocation GradFnCall records. ALWAYS returns GradFnCall. Accepts GradFnCall labels (`:N`-qualified) directly. For **single-call GradFns**, also accepts the bare GradFn label → resolves to the unique GradFnCall. For multi-call GradFns, bare label raises. 0-based positional integer indexing.
- `input_layers`: Accessor for Layers representing external model inputs.
- `input_ops`: Accessor for Ops at external model input boundaries (flat across all passes when input positions partake in loops).
- `output_layers`: Accessor for Layers representing model outputs.
- `output_ops`: Accessor for Ops at model output boundaries (flat across all passes).
- `buffer_layers`: Accessor for Layers representing buffers as graph boundary values.
- `internal_source_layers`: Accessor for Layers representing internal-source positions (grouped by equivalence class).
- `internal_source_ops`: Accessor for Ops that begin an internal subgraph without normal external input ancestry (flat across all passes).
- `internal_sink_layers`: Accessor for Layers representing internal-sink positions (grouped by equivalence class).
- `internal_sink_ops`: Accessor for Ops that end an internal subgraph before reaching normal outputs (flat across all passes).
- `orphans`: Accessor for Ops disconnected from the main input→output flow (see `Op.is_orphan`).
- `layers_with_params`: Accessor for Layers that use parameters (harmonized to Accessor per the filter-consistency cleanup).
- `ops_with_params`: Accessor for Ops that use parameters (flat across all passes; parallels `layers_with_params` at Op granularity).
- `op_equivalence_classes`: Dict mapping equivalence-class id to a set of Op labels considered structurally equivalent by loop detection. (Renamed from `equivalent_ops` at Trace scope; the Op/Layer-scope `equivalent_ops` fields remain as member lists.)

### Module, Param, Buffer, and Grad Accessors

- `root_module` (`@property`): Module record for the top-level model, resolved as `self.modules["self"]`. The root module's address is always `"self"`.
- `modules`: Accessor for Module records. ALWAYS returns Module. Accepts bare Module addresses (`encoder.block`) directly, AND ModuleCall labels (`encoder.block:1`) by stripping `:N` to find the parent Module. Also accepts alias forms. 0-based positional integer indexing.
- `uncalled_modules`: Module addresses registered on the source model but not called in this capture.
- `params`: Accessor for Param records. ALWAYS returns Param. Accepts Param addresses (`encoder.block.weight`) by exact match, plus alternate lookup keys. 0-based positional integer indexing.
- `buffers`: Accessor for persistent Buffer records. ALWAYS returns Buffer. Accepts Buffer addresses (`bn.running_mean`) by exact match, short name when unambiguous, and 0-based positional integer indexing.
- `grad_fns`: Accessor for GradFn records. ALWAYS returns GradFn. Accepts GradFn labels directly, AND GradFnCall labels (`:N`-qualified) by stripping `:N` to find the parent GradFn. 0-based positional integer indexing.

### Topology

- `is_branching`: True when any Layer has more than one child in the captured graph.
- `is_recurrent`: True when any Layer has more than one Op.
- `recurrent_layers`: Accessor for aggregate Layers whose `num_passes > 1`.
- `max_layer_op_count`: Maximum number of Ops aggregated by any single Layer.
- `is_dynamic_graph` (`@property`): True when execution path depends on tensor values. Derives from `has_conditionals` (and `has_data_dependent_loops` once iteration story lands).
- `has_conditionals` (`@property`): True when `num_conditionals > 0`.
- `has_backward_pass`: True when backward-pass data has been logged.
- `has_gradients`: True when gradient values are present.

### Backward and GradFn Anchors

- `backward_memory_backend`: Backend used to measure backward memory, such as `cuda`, `cpu`, `mps`, or `unknown`.
- `derived_grads`: Preview-only backend-derived leaf gradients, populated by future JAX
  derived-gradient capture from a second JAX AD execution. Not a true backward graph and must not
  populate `GradFn` / `GradFnCall` accessors.
- `backward_root_grad_fn_object_ids`: List of runtime Python `id()` values, one per backward pass, in execution order. Each entry is the root grad-fn `object_id` from that pass's loss tensor. Empty list if no backward pass has been logged. Renamed from `backward_root_grad_fn_ids` for consistency.
- `last_backward_root_grad_fn_object_id` (`@property`): Convenience accessor for the most recent backward pass's root grad-fn object_id; `None` if no backward pass has been logged.
- `backward_durations`: List of float seconds, one per backward pass, in execution order. Empty list if no backward pass has been logged.
- `last_backward_duration` (`@property`): Most recent backward pass's duration in seconds; `None` if no backward pass has been logged.
- `total_backward_duration` (`@property`): Sum of `backward_durations` across all backward passes.

### Compute

- `total_flops`: Approximate total forward plus backward FLOPs.
- `total_flops_forward`: Approximate forward FLOPs.
- `total_flops_backward`: Approximate backward FLOPs.
- `total_macs`: Approximate total multiply-accumulate count.
- `total_macs_forward`: Approximate forward MACs.
- `total_macs_backward`: Approximate backward MACs.
- `flops_by_op_type`: Dictionary summarizing FLOPs by normalized Op type (keyed by `op.type`).
- `macs_by_op_type`: Dictionary summarizing MACs by normalized Op type.

### Capture State

- `annotations`: User-attached metadata about the whole Trace as a record (`Dict[str, Any]`). Distinct from `input_annotations` (which annotates the specific input data used for capture). Trace is the only class with two annotation surfaces; bare `annotations` is the trace-itself's metadata, prefix-qualified `input_annotations` is the secondary entity's metadata.
- `input_object_id`: Stored Python `id()` of the input tensor or primary input structure at capture time. Within-session only; NOT comparable across processes.
- `input_signature_hash`: Hash of input shapes, dtypes, AND devices for comparability checks.
- `input_annotations`: User-attached metadata about the inputs.
- `random_seed`: RNG seed actually applied for capture, if any.
- `param_hash_quick`: Quick hash of parameter values for relationship checks.
- `param_hash_full`: Full hash of parameter values for stronger verification.
- `graph_shape_hash`: Hash of captured graph topology and shape-relevant structure.
- `state_history`: Append-only record of lifecycle state changes (fork, replay, rerun, direct-write). Each entry carries (timestamp, transition type, payload-as-needed). Pairs with `Trace.state`.
- `last_run`: Inspectable dict summarizing the most recent replay/rerun/append operation (engine name, timing, intervention spec revision, flags, divergence info, hash before/after). Renamed from `last_run_ctx`; the `_ctx` abbreviation is reserved for callback argument names like `run_ctx` per the autograd convention. Possibly redundant with `state_history[-1]`; pending review.
- `is_appended`: True when this Trace currently holds appended chunks. Cleared on non-append `rerun()`; preserved across `replace_state_from`.
- `append_history`: List of per-append provenance dicts recording inputs, hashes, and metadata for each `rerun(append=True)` chunk.
- `code_context`: Python call-stack frames at the moment `tl.capture(model, x)` was invoked, showing where in user code the capture was triggered (list of `FuncCallLocation`). Parallels `Op.code_context` and `ModuleCall.code_context` at Trace scope. Useful for provenance when working with multiple Traces in the same session.

### Source Info (model class introspection)

- `class_source_file`: File where the model class is defined, when available.
- `class_source_line`: Line where the model class is defined.
- `class_source_location` (`@property`): Combined `class_source_file:class_source_line` editor jump string.
- `class_docstring`: Docstring of the model class.
- `init_source_file`: File where the model's `__init__` is defined.
- `init_source_line`: Line where the model's `__init__` is defined.
- `init_source_location` (`@property`): Combined `init_source_file:init_source_line`.
- `init_signature`: Signature of the model's `__init__`, when introspectable.
- `init_docstring`: Docstring of the model's `__init__`.
- `forward_source_file`: File where the model's `forward()` method is defined.
- `forward_source_line`: Source line number for the model `forward()` definition.
- `forward_source_location` (`@property`): Combined `forward_source_file:forward_source_line`.
- `forward_signature`: Signature of the model's `forward`, when introspectable.
- `forward_docstring`: Docstring of the model's `forward`.

All triplets gracefully return `None` when the underlying method isn't
user-defined (e.g., default `nn.Module.__init__`).

### Capture Config

- `save_raw_activations`: Whether raw untransformed activation tensors are saved.
- `save_raw_gradients`: Whether raw untransformed gradient tensors are saved.
- `save_gradients`: Whether TorchLens should save gradients after backward.
- `save_arg_values`: Whether Op function arguments are saved (heavy memory — deep-copies tensor values).
- `save_arg_templates`: Whether structural arg templates (slot map, shape/dtype placeholders) are saved. Lightweight.
- `save_rng_states`: Whether RNG state snapshots are saved.
- `save_code_context`: Whether source-code context around calls is saved.
- `gradients_to_save`: Which gradients are requested for saving.
- `layers_to_save`: Which layers/ops are requested for saving.
- `backward_ready`: Whether the Trace was captured with backward machinery prepared (autograd connectivity retained for later backward).
- `intervention_ready`: Whether the Trace was captured with enough state for interventions.
- `intervention_spec`: Current intervention specification associated with the Trace.
- `capture_mode`: Capture behavior mode.
- `detach_saved_activations`: Whether saved tensors are detached from autograd.
- `recurrence_detection`: Whether TorchLens performs loop/recurrence detection.
- `emit_nvtx`: Whether capture emits NVIDIA NVTX ranges. NVTX = NVIDIA Tools Extension — a CUDA profiling annotation library. When True, TorchLens wraps captured ops with NVTX range markers so they show up labeled in NVIDIA profilers (Nsight Systems, Nsight Compute). Default OFF; enable when profiling on CUDA. Adds small per-op overhead. Field name matches PyTorch's `torch.autograd.profiler.emit_nvtx` API for consistency.
- `mark_layer_depths`: Whether topological input/output depth metadata is marked on Layers.
- `raise_on_nan`: Whether capture raises when NaNs are detected.
- `verbose`: Whether capture prints progress or diagnostic messages.
- `output_device`: Device where saved outputs should be placed.
- `module_filter`: Callable that decides which modules are eligible for module logging. Rename to `save_predicate` deferred.
- `activation_transform`: Callable applied to saved forward outputs before transformed storage.
- `gradient_transform`: Callable applied to saved gradients before transformed storage.

### Timing

- `capture_start_time`: Epoch timestamp when capture started.
- `capture_end_time`: Epoch timestamp when capture ended.
- `setup_duration`: Seconds spent in capture setup.
- `forward_duration`: Seconds spent running the model forward pass.
- `func_calls_duration`: Seconds spent inside the captured op function calls (sum of all `Op.func_duration` values across the trace). Closest proxy to "pure model compute time" — excludes TorchLens wrapper overhead, Python orchestration, and setup/cleanup. Backend-neutral (torch or mlx); the field name avoids backend-specific words like "torch" so MLX captures use the same name.
- `cleanup_duration`: Seconds spent in cleanup.
- `capture_duration`: Total elapsed capture time (sum of setup + forward + cleanup).
- `overhead_duration`: TorchLens overhead duration (capture minus pure forward).

Replay and rerun durations are recorded per-event in `state_history`, not as
Trace-level aggregates.

### Conditional Flow

- `conditionals`: `ConditionalAccessor` for conditional structures by integer ordinal or stable conditional id.
- `has_conditionals`: True when this Trace captured at least one conditional.
- `num_conditionals`: Number of conditionals captured.
- `Conditional`: One if-chain at one source location.
- `Conditional.label`: Stable TorchLens label such as `cond_gt_1_4`, derived from the leading terminal bool Op label. Renamed from `id` per the convention that TorchLens-internal stable identifiers use `label` (id-suffixed names are reserved for Python `id()` runtime values).
- `Conditional.arms`: Ordered list of `ConditionalArm` records: leading `then`, zero or more `elif`, and optional `else`.
- `Conditional.fired_arm_index`: Index of the arm whose body ran, or `None` if no arm fired.
- `Conditional.fired_arm_kind`: Denormalized fired arm kind: `then`, `elif`, `else`, or `None`.
- `Conditional.source_file`: File containing the if-statement, when known.
- `Conditional.source_line`: Line of the `if` keyword, when known.
- `Conditional.source_location` (`@property`): Combined `source_file:source_line`, or `None` if incomplete.
- `Conditional.fired_arm` (`@property`): Direct access to the fired `ConditionalArm`, or `None`.
- `Conditional.has_else` (`@property`): True when an else arm exists.
- `Conditional.has_elif` (`@property`): True when at least one elif arm exists.
- `Conditional.num_arms` (`@property`): Number of arms.
- `Conditional.num_elifs` (`@property`): Number of elif arms.
- `ConditionalArm.kind`: Arm kind: `then`, `elif`, or `else`.
- `ConditionalArm.terminal_bool_op_label`: Final scalar bool Op for the arm, or `None` for `else`.
- `ConditionalArm.bool_value_at_run`: Bool value observed when evaluated, or `None`.
- `ConditionalArm.condition_evaluated`: True when this arm's condition was reached.
- `ConditionalArm.evaluation_entry_edge`: `Tuple[str, str]` of Layer labels `(upstream_parent_label, first_evaluation_layer_label)` for the actual fork edge entering this arm's condition computation. There is a genuine graph fork between the evaluation side (computing the bool) and the execution side (running the body); this field marks the source-side of that fork. Used by visualization for IF/elif label rendering. (NOTE: current implementation has a bug producing a self-loop `(X, X)` instead of the real fork edge — see `known_bugs.md::COND-EVAL-EDGE-SELFLOOP`. Spec describes intended behavior.)
- `ConditionalArm.evaluation_ops` (`@property`): Op labels computing this arm's condition; derived from each Op's `in_conditionals`.
- `ConditionalArm.fired`: True when this arm's body ran.
- `ConditionalArm.execution_entry_edge`: `Tuple[str, str]` or `None` — Layer labels `(upstream_parent_label, first_body_layer_label)` for the edge entering this arm's body. `None` if the arm did not fire (body has no executed Ops). Used by visualization for THEN/ELIF/ELSE label rendering.
- `ConditionalArm.execution_ops` (`@property`): Op labels in this arm's body; derived from each Op's `in_conditionals`.
- `ConditionalRoleRef`: One Op's participation in a conditional arm.
- `ConditionalRoleRef.conditional_label`: Label of the Conditional this Op participates in (cross-class ref to `Conditional.label`). Renamed from `conditional_id`.
- `ConditionalRoleRef.arm_index`: Index into `conditional.arms`.
- `ConditionalRoleRef.arm_kind`: Denormalized arm kind: `then`, `elif`, or `else`.
- `ConditionalRoleRef.role`: `evaluation` or `body`.

### Trace Call-Tree Access (LOCKED 2026-05-23)

- **`walk_calls()`**: **Iterate the full call tree from root → leaves. `ModuleCall` itself is the tree node via `call_parent` and `call_children` (no separate `CallTreeNode` wrapper — see drop entry below). (LOCKED 2026-05-23)**
- **`show_call_tree(max_depth=None, include_atomic=False, show_call_index=False, file=None)`**: **ASCII tree printed to stdout (or `file=` kwarg). (LOCKED 2026-05-23)**

### Trace Methods

- `backward(loss)`: Run backward from an explicit loss and populate grad-related fields.
- `find_layers(query, *, limit=10)`: Find Layer labels matching a query.
- `fork(name=None)`: Duplicate the Trace with a fresh intervention spec and optional new name.
- `rerun(model, x, **kwargs)`: Re-execute the model and update or append Trace state.
- `replay(**kwargs)`: Replay saved activations without re-executing the model.
- `draw(**kwargs)`: Draw the forward graph. All vis params live here (no longer on `tl.trace()`).
  `order_siblings=True` is the default Graphviz/dot unrolled post-pass that verifies and
  applies execution-order placement for true parallel sibling fanouts.
- `draw_backward(**kwargs)`: Draw the backward grad-fn graph.
- `draw_combined(**kwargs)`: Render forward ops and backward grad_fns in a single graph.
- `cleanup()`: Clear circular references and runtime-only heavyweight objects.
- `summary(level, ...)`: Return a textual summary of the Trace.
- `save(path, **kwargs)`: Save the Trace to portable `.tlspec` format.
- `replace_state_from(new_log)`: Atomically replace this Trace's run state from a freshly-built Trace.
- `append_state_from(new_log)`: Merge compatible chunk outs from `new_log` into this Trace.
- `to_pandas()`: Export Trace fields to a pandas DataFrame.
- `torchlens.export.csv(trace, path, **kwargs)`: Write Trace export data to CSV.
- `torchlens.export.parquet(trace, path, **kwargs)`: Write Trace export data to Parquet.
- `torchlens.export.json(trace, path, **kwargs)`: Write Trace export data to JSON.
- **`attention_blocks()`**: **Iterator over Modules detected as attention blocks by the facets registry. Convenience wrapper around `modules_with_facet('q')`. (LOCKED 2026-05-27)**
- **`modules_with_facet(name)`**: **Iterator over Modules whose FacetView contains the named facet. Useful for sweeping across all attention/norm/MLP blocks by facet membership. (LOCKED 2026-05-27)**

Note: `Trace.load(path)` classmethod has been REMOVED. Use the module-level
`tl.load(path)` instead. `Trace.save` instance method remains for natural OOP.

### Trace Intervention Methods

Intervention method names are intentionally not promoted as final here; the
audit deferred them to the integrated `Site` concept review.

## Op (one captured tensor operation)

An `Op` is a graph node representing one captured event. Two orthogonal axes:

- **Op kind (mutually exclusive):** compute Op (`is_compute_op`) — any torch function call, including factory functions like `torch.ones` and `torch.rand` — OR boundary Op (`is_input`, `is_output`, `is_buffer_source`) — a transition marker, no torch function executed.
- **Connectivity descriptors (orthogonal, apply to compute Ops):** `is_internal_source` (no external input ancestor), `is_internal_sink` (output not used by model output), `is_orphan` (disconnected from both flows). These are additional flags on compute Ops, NOT alternative Op kinds.

All Op subtypes participate uniformly in the equivalence/Layer machinery.

### Identity and Labeling

- `label`: Pass-qualified Op label such as `conv2d_1_2:1`.
- `label_short`: Short label omitting the step index where available.
- `type`: Normalized operation type token such as `conv2d`.
- `type_index`: 1-based position among Ops or Layers of the same type. Middle number in labels (`conv2d_1_5` → `1`).
- `step_index`: 1-based step number in the forward computation. Compute ops get sequential values (1, 2, 3, ...); inputs, buffers, and outputs get 0 (they're not computation steps). Appears as the last number in compute-op labels like `conv2d_1_5`.
- `pass_index`: 1-based iteration within parent Layer's equivalence class. After `:` in pass-qualified labels (`conv2d_1_5:2` → `2`).
- `ordinal_index`: 0-based position in `trace.ops` (covers ALL ops uniquely, including boundaries). Use `trace[op.ordinal_index] is op` for round-trip lookup. Pythonic — works with negative indexing too.
- `raw_index`: 1-based raw capture-time counter; may have gaps after orphan removal. Used in `_raw` labels for debug/internal purposes.
- `num_passes`: Number of Op passes in the parent Layer.
- `layer_label`: Parent Layer label without `:N`. Stored form.
- `layer` (`@property`): Parent Layer object resolved via `self.trace.layers[self.layer_label]`. Raises if the label cannot be resolved.
- `fx_label` (`@property`): torch.fx-style label combining `fx_qualpath` + `fx_call_index`. Registered as a lookup key so `trace[fx_label]` works.
- `fx_qualpath`: FX-style qualified path (when applicable).
- `fx_call_index`: FX-style call index (when applicable).
- `trace`: Runtime-only back-pointer to the owning Trace (typically weakref). Not portable; unavailable after `.tlspec` load.

### Function Identity

- `func`: Runtime callable that produced this Op's output.
- `func_name`: Short callable name, preserving in-place suffixes when present.
- `func_qualname`: Fully qualified callable name for verification and display.
- `is_inplace`: True when the underlying PyTorch operation was in-place.
- `arg_names`: Names of callable arguments.
- `num_args_total`: Total number of positional and keyword arguments.
- `num_pos_args`: Number of positional arguments.
- `num_kwargs`: Number of keyword arguments.
- `grad_fn_class_name`: Short class name of the PyTorch autograd node attached to this Op output.
- `grad_fn_class_qualname`: Fully qualified class name of the autograd node.
- `grad_fn_cls` (`@property`): Runtime Python type object for the grad_fn class.
- `grad_fn_object_id`: Stored Python `id()` of the PyTorch autograd handle. Within-session only; NOT comparable across processes. Renamed from `grad_fn_id` for consistency with `model_object_id` / `input_object_id` naming.
- `grad_fn_label`: Stable GradFn label corresponding to this Op, when backward data exists. Stored form (portable).
- `grad_fn` (`@property`): TorchLens GradFn record resolved via `self.trace.grad_fns[self.grad_fn_label]`. Returns `None` when no GradFn was captured for this Op. (Now returns the TorchLens record, consistent with `Op.module` / `Op.params` / `Op.layer`; runtime handle moved to `grad_fn_handle`.)
- `grad_fn_handle`: Runtime PyTorch autograd Function object (what `tensor.grad_fn` returns). Runtime-only; not portable. Use for low-level autograd introspection (`next_functions`, `saved_tensors`, etc.).

### Input Tensors (LOCKED 2026-05-23)

Graph-parent input shortcuts on Op. All `@property` — zero storage cost, no new capture machinery. Mirrors `op.parents`. Does NOT include Param/Buffer values — those have their own access paths (`op.params`, `op.buffers`). For ops downstream of in-place modifications, the resolver consults `out_versions_by_child` to return the version THIS op consumed.

- **`input_ops`** (`@property`): **Accessor over `op.parents` → Op records (not labels). (LOCKED 2026-05-23)**
- **`input_activations`** (`@property`): **`tuple[Tensor | None, ...]` in `op.parents` order. `None` slot when a parent wasn't saved. (LOCKED 2026-05-23)**
- **`input_shapes`** (`@property`): **`tuple[Shape, ...]`. (LOCKED 2026-05-23)**
- **`input_dtypes`** (`@property`): **`tuple[torch.dtype, ...]`. (LOCKED 2026-05-23)**
- **`input_memory`** (`@property`): **`int` — sum of input `activation_memory` across parents. (LOCKED 2026-05-23)**
- **`num_inputs`** (`@property`): **= `len(op.parents)`. (LOCKED 2026-05-23)**

Relationship to `op.args` / `op.kwargs`: call-signature view (with scalars, nested structures, per-function arg schema) vs graph-edge view (one tensor per Op parent in graph traversal order). Different views — neither replaces the other.

### Tensor Properties

- `out`: Saved forward tensor output of this Op when TorchLens saved it.
- `shape`: Shape of `out`.
- `dtype`: Dtype of `out`.
- `dtype_ref`: Backend-neutral dtype reference for the Op output, stored as a `DtypeRef`.
- `activation_memory`: Bytes used by `out`.
- `device_ref`: Backend-neutral device reference for the Op output, stored as a `DeviceRef` when
  available.
- `backend_address`: Backend-native address or path for resolving this Op, when available.
- `resolver_status`: Resolution status for backend-native metadata, currently `"resolved"`,
  `"metadata_only"`, or `"unresolved"`.
- `grad`: Saved gradient of the Op output after backward, when available.
- `grad_shape`: Shape of `grad`.
- `grad_dtype`: Dtype of `grad`.
- `gradient_memory`: Bytes used by `grad`.
- `transformed_out`: Saved transformed version of `out`, after `activation_transform`.
- `transformed_out_shape`: Shape of `transformed_out`.
- `transformed_out_dtype`: Dtype of `transformed_out`.
- `transformed_activation_memory`: Bytes used by `transformed_out`.
- `transformed_grad`: Saved transformed version of `grad`, after `gradient_transform`.
- `transformed_grad_shape`: Shape of `transformed_grad`.
- `transformed_grad_dtype`: Dtype of `transformed_grad`.
- `transformed_gradient_memory`: Bytes used by `transformed_grad`.
- `autograd_memory`: Bytes PyTorch autograd retained for backward computation on this Op (PyTorch's "saved tensors for backward"). Renamed from `autograd_saved_memory` to drop the "saved" overloading.
- `num_autograd_tensors`: Number of tensors PyTorch autograd retained for backward on this Op. Renamed from `num_autograd_saved_tensors`.

### Multi-output Cluster

- `in_multi_output` (`@property`): True when this Op came from a multi-output container.
- `multi_output_type`: The Python class of the container (`tuple`, `list`, `dict`, namedtuple class, dataclass, etc.). `None` if not from a multi-output.
- `multi_output_index`: 0-based positional access for ordered containers. `None` for non-multi-output.
- `multi_output_name`: Semantic name (dict key, namedtuple field, dataclass attribute). `None` for plain tuple/list outputs, or when not from a multi-output.
- `container_path`: Full nested structural path from the top-level return down to this Op's tensor. Tuple of typed components (`TupleIndex`, `DictKey`, `NamedField`, `DataclassField`, `HFKey`). The single source of truth for nested paths; `multi_output_*` provides FLAT one-level convenience views.
- `container_spec`: Structural description of the return container, used to rebuild where this output came from.

### Per-Op Config and Saved State

- `output_device`: Device where saved tensors for this Op were placed.
- `activation_transform`: Transform used for the Op's saved output.
- `gradient_transform`: Transform used for the Op's saved gradient.
- `annotations`: User-attached metadata about this Op.
- `detach_saved_activations`: Whether saved tensors were detached.
- `save_gradients`: Whether gradient saving was requested for this Op.
- `has_saved_activation`: True when this Op's forward output was saved.
- `has_saved_gradient`: True when this Op's gradient was saved.
- `has_saved_args`: True when function arguments were saved.
- `saved_args`: Saved positional argument values.
- `saved_kwargs`: Saved keyword argument values.
- `args_template`: Structured template for saved positional arguments, used by replay/intervention.
- `kwargs_template`: Structured template for saved keyword arguments.
- `non_tensor_pos_args`: Non-tensor positional arguments passed to the function.
- `non_tensor_kwargs`: Non-tensor keyword arguments passed to the function.
- **`args_summary`**: **Human-readable summary of this Op's positional arguments. The Op-level source for `Layer.args_summary` (which delegates via single-Op passthrough; raises for multi-Op Layers). (LOCKED 2026-05-21)**
- **`kwargs_summary`**: **Human-readable summary of this Op's keyword arguments. Op-level source for `Layer.kwargs_summary`. (LOCKED 2026-05-21)**

### Timing, RNG, and Call Context

- `code_context`: Python call-stack frames showing where in user/PyTorch code this Op function ran (list of `FuncCallLocation`).
- `func_duration`: Time spent in the function call for this Op.
- `func_rng_states`: RNG state snapshots associated with this Op.
- `func_autocast_state`: Autocast state active during the function call.
- `lookup_keys`: Alternate lookup keys that can resolve to this Op.

### Compute

- `flops_forward`: Approximate forward FLOPs for this Op.
- `flops_backward`: Approximate backward FLOPs for this Op.
- `flops_total`: Approximate forward plus backward FLOPs for this Op.
- `macs_forward`: Approximate forward MACs.
- `macs_backward`: Approximate backward MACs.
- `macs_total`: Approximate forward plus backward MACs.

### Parameters

- `param_shapes`: Shapes of parameters consumed by this Op.
- `param_names`: Short parameter names consumed by this Op.
- `param_dtypes`: Dtypes of parameters consumed by this Op.
- `num_params`: Number of scalar parameters consumed by this Op.
- `num_params_trainable`: Number of trainable scalar parameters consumed by this Op.
- `num_params_frozen`: Number of frozen scalar parameters consumed by this Op.
- `num_param_tensors`: Number of parameter tensors consumed by this Op.
- `num_param_tensors_trainable` (`@property`): Number of trainable parameter tensor objects consumed by this Op.
- `num_param_tensors_frozen` (`@property`): Number of frozen parameter tensor objects consumed by this Op.
- `param_memory`: Bytes used by consumed parameters.
- `params`: Accessor or list-like view of Param records consumed by this Op.
- `uses_params` (`@property`): True when this Op consumes any parameters (any kind — trainable or frozen).
- `has_trainable_params` (`@property`): True when this Op consumes at least one trainable parameter.
- `has_frozen_params` (`@property`): True when this Op consumes at least one frozen parameter.

### Equivalence and Recurrence

- `equivalence_class`: Loop-detection equivalence class for this Op.
- `equivalent_ops`: Other Op labels in the same equivalence class.

### Role Flags

- `is_compute_op` (`@property`): True when this Op executed a torch function (any function call — `conv2d`, `relu`, `torch.ones`, `torch.rand`, etc.). Equivalent to `not (is_input or is_output or is_buffer_source)`. Internal-source / internal-sink / orphan flags are ORTHOGONAL connectivity descriptors and DO NOT affect `is_compute_op` — a `torch.ones` call has `is_compute_op = True` AND `is_internal_source = True` simultaneously.
- `is_input`: True when this Op represents an external input boundary.
- `is_output`: True when this Op represents an output boundary node.
- `is_final_output`: True when this Op is the final model output marker.
- `is_buffer_source` (`@property`): True when this Op is a buffer-version node (a buffer boundary / overwrite). The public glossary name for the stored `is_buffer` flag (`is_buffer_source == is_buffer`); `compute_ops` and `is_compute_op` exclude these nodes.
- `is_buffer`: Stored flag backing `is_buffer_source` — `True` for plain `Op` records that are buffer-version nodes.
- `buffer_write_kind`: How this buffer version was written — `"reassign"` (`self.buf = ...`), `"inplace"` (`buf.mul_`/`copy_`/`buf[...] =`), or `"fused"` (native running-stat update, e.g. BatchNorm); `None` for the static initial-read node.
- `buffer_value_changed`: For a fused/native buffer version, whether the post-op value differed from the pre-op value (the state transition actually changed the buffer); `None`/unused for non-fused versions.
- `address`: Dotted path for the buffer when buffer-sourced (replaces former `buffer_address`).
- `buffer_source`: Source Layer or Op that wrote this buffer value, or `None` for static buffers.
- `is_internal_source`: True when this Op starts an internally generated graph region.
- `is_internal_sink`: True when this Op terminates an internal graph region.
- `is_orphan` (`@property`): True iff this op has no input ancestor AND no output descendant — disconnected from the main graph. Use `Trace.orphans` Accessor for the filtered collection.
- `is_terminal_bool`: True when this Op is a terminal boolean used by control flow.
- `is_scalar_bool`: True when the Op output is a scalar boolean.
- `bool_value`: Boolean value when `is_scalar_bool` is true.
- `in_conditionals`: List of `ConditionalRoleRef` records describing this Op's conditional roles.
- `is_in_conditional` (`@property`): True when this Op participates in any conditional role.
- `is_in_conditional_evaluation` (`@property`): True when this Op computes a conditional arm condition.
- `is_in_conditional_body` (`@property`): True when this Op is in a conditional arm body.
- `conditional_depth` (`@property`): Number of distinct conditionals this Op participates in.
- `terminal_bool_for`: `(conditional_label, arm_index)` when this Op is the terminal bool of an arm, else `None`.
- `is_atomic_module`: True when this Op is the sole operation output of an atomic module call.
- `is_module_input`: True when this Op's output FEEDS INTO at least one ModuleCall as an input arg. The Op itself is OUTSIDE the module; it's the upstream producer. Equivalent to `bool(input_to_module_calls)`. Direction-of-data-flow framing: inputs come FROM outside the module.
- `is_module_output`: True when this Op IS the output of at least one ModuleCall — its tensor is what the module's `forward()` returned. The Op itself is INSIDE the module's forward (the final compute op). Equivalent to `bool(output_of_module_calls)`. Direction-of-data-flow framing: outputs are produced INSIDE the module.

### Module Containment

- `module` (`@property`): Module containing this Op (innermost from `module_call_stack`); resolves via `self.trace.modules[<address>]`. Raises if no containing module exists.
- `module_call_stack`: List of ModuleCall labels (root-first) active for this Op. Stored as label list per Principle 4.
- `output_of_modules`: List of Module addresses (label strings) for which this Op's output is the module output.
- `input_to_modules`: List of Module addresses (label strings) for which this Op's output is the module input.
- `output_of_module_calls`: List of ModuleCall labels for calls this Op outputs from.
- `input_to_module_calls`: List of ModuleCall labels for calls this Op inputs to. (Subsumes the former `module_calls_entered` field which was redundant — same event, different phrasing.)
- `module_entry_arg_keys`: Dict mapping ModuleCall label (from `input_to_module_calls`) → list of int/str arg keys this Op filled at that entry event (replaces `module_entry_argnames`). Polymorphic: int for positional args, str for kwargs.
- `in_submodule` (`@property`): True when this Op was computed inside a NON-ROOT module's forward (kept distinct from `is_module_input`/`is_module_output` because "in module" would always be true for the root).
- `atomic_module_call_label`: Stable ModuleCall label (`:N`-suffixed) for an atomic module output, when applicable. Stored form.
- `atomic_module_call` (`@property`): ModuleCall record resolved via `self.trace.module_calls[self.atomic_module_call_label]`. Returns `None` when not an atomic module output.
- `atomic_module_address`: Module address (PyTorch dotted path; no `:N`) for an atomic module output, when applicable. Stored form. Paralllels `atomic_module_call_label` storage; one-hop resolution for `atomic_module` resolver. Derivable from `atomic_module_call_label` by stripping `:N` but stored separately for completeness and faster access.
- `atomic_module` (`@property`): Module record resolved via `self.trace.modules[self.atomic_module_address]`. Returns `None` when not applicable.

### Graph Relations

At Layer scope, relation lists use bare Layer labels; at Op scope, relation
lists may use pass-qualified Op labels when per-pass precision matters.

- `parents`: Immediate parent Layer or Op labels feeding this Op.
- `children`: Immediate child Layer or Op labels consuming this Op.
- `siblings`: Ops or Layers sharing graph parents with this Op.
- `co_parents`: Other parents that jointly feed this Op's children.
- `has_parents` (`@property`): True when this Op has parents.
- `has_children` (`@property`): True when this Op has children.
- `num_parents` (`@property`): `len(self.parents)`.
- `num_children` (`@property`): `len(self.children)` — mirror the existing `num_parents`.
- `has_siblings` (`@property`): True when this Op has siblings.
- `has_co_parents` (`@property`): True when this Op has co-parents.
- `has_input_ancestor`: True when an input boundary reaches this Op.
- `has_output_descendant`: True when this Op reaches an output boundary.
- `parent_arg_positions`: Positions in the function call where parent tensors were used.
- `is_output_parent`: True when this tensor was directly returned from `forward`.
- `output_descendants`: List of output boundary Op labels reachable from this Op.
- `min_distance_to_output`: Shortest graph distance from this Op to an output.
- `max_distance_to_output`: Longest graph distance from this Op to an output.
- `input_ancestors`: List of input boundary Op labels that can reach this Op.
- `min_distance_from_input`: Shortest graph distance from an input to this Op.
- `max_distance_from_input`: Longest graph distance from an input to this Op.
- `root_ancestors`: List of root-graph-ancestor Op labels for this Op.
- `has_internal_source_ancestor`: True when this Op descends from an internal source.
- `internal_source_parents`: List of Op labels for immediate parents that are internal sources.
- `internal_source_ancestors`: List of Op labels for internal-source ancestors reachable upstream.

### Output Variations and Interventions

- `has_out_variations`: True when different children observed different versions of this Op output.
- `out_versions_by_child`: Dict mapping child label to tensor snapshot at the moment that child consumed the output. Populated when an in-place op modified the tensor between uses, creating divergent per-child views.
- `interventions`: Intervention records applied to this Op.

### Op Methods

- `save_activation(...)`: Save the activation tensor payload.
- `to_pandas()`: Export this Op as a one-row DataFrame.
- File export: use `torchlens.export.csv(op, path)`, `torchlens.export.parquet(op, path)`, or `torchlens.export.json(op, path)`.

### Facets (LOCKED 2026-05-27)

- **`facets`** (`FacetView`): **Derived semantic views populated by recipes matched against this Op (class / predicate match). Lazy + cached. Dict-like + attribute-access: `op.facets.X` and `op.facets['X']` both work. Empty FacetView when no recipe matches. See the dedicated Facets section at the end of the glossary for the recipe-registration API.**

## Layer (aggregate over one or more Ops)

A `Layer` is an equivalence class of recurrent Ops. Layer-level fields
follow the bare / `total_*` convention for aggregate-able numeric values:
bare is per-pass (raises for multi-pass), `total_*` is sum across all Ops
in the Layer.

### Identity and Labeling

- `label`: Layer label without a pass suffix, such as `conv2d_1_2`.
- `label_short`: Short Layer label without the step index.
- `type`: Normalized operation type token for this Layer.
- `type_index`: 1-based position within Layers of the same type.
- `step_index`: 1-based step in forward computation (0 for boundaries).
- `ordinal_index`: 0-based position in `trace.layers`.
- `raw_index`: 1-based raw capture-time counter.
- `num_ops`: Number of Ops aggregated by this Layer.
- `fx_label` (`@property`): torch.fx-style label for a single-Op Layer (passthrough from `op.fx_label`). Registered as a lookup key so `trace[fx_label]` works.
- `fx_qualpath`: FX-style qualified path for a single-Op Layer (passthrough).
- `fx_call_index`: FX-style call index for a single-Op Layer (passthrough).
- `trace`: Runtime-only back-pointer to the owning Trace (typically weakref). Not portable; unavailable after `.tlspec` load.

### Function Identity Passthroughs

These entries follow the single-Op passthrough rule: they read naturally on
one-Op Layers and raise `ValueError` for multi-Op Layers (use `layer.ops[n]`).

- `func`: Callable for the single Op when this Layer has one Op; raises for multi-Op Layers.
- `func_name`: Short callable name for the Layer's Op.
- `func_qualname`: Fully qualified callable name when available.
- `is_inplace`: True when the underlying Op was in-place.
- `arg_names`: Function argument names.
- `num_args_total`, `num_pos_args`, `num_kwargs`: Argument counts.
- `grad_fn_class_name`: Autograd node class name associated with this Layer's Op.
- `grad_fn_class_qualname`: Fully qualified autograd node class name.
- `grad_fn_cls` (`@property`): Runtime Python type object for the grad_fn class.
- `grad_fn_object_id`: Python `id()` of the autograd handle for a single-Op Layer (passthrough).
- `grad_fn_label`: Stable GradFn label for a single-Op Layer (storage; passthrough from `op.grad_fn_label`).
- `grad_fn` (`@property`): TorchLens GradFn record for a single-Op Layer, resolved via `self.trace.grad_fns[self.grad_fn_label]`. Returns `None` when no GradFn was captured. (Post-swap: bare name = TL record, matching Op convention.)
- `grad_fn_handle`: Runtime PyTorch autograd Function object for a single-Op Layer (passthrough). Runtime-only; not portable.

### Tensor Properties

- `out`: Saved forward output for a single-Op Layer; raises for multi-Op Layers.
- `shape`: Output shape when stable or single-Op.
- `dtype`: Output dtype when stable or single-Op.
- `dtype_ref`: Backend-neutral dtype reference for the Layer output, delegated from its first Op.
- `device_ref`: Backend-neutral device reference for the Layer output, delegated from its first Op
  when available.
- `backend_address`: Backend-native address or path for resolving this Layer, when available.
- `resolver_status`: Resolution status for backend-native metadata, currently `"resolved"`,
  `"metadata_only"`, or `"unresolved"`.
- `activation_memory`: Output memory in bytes (per-pass form).
- `total_activation_memory`: Sum of activation memory across all Ops in this Layer.
- `grad`: Saved gradient for a single-Op Layer.
- `grad_shape`: Gradient shape.
- `grad_dtype`: Gradient dtype.
- `gradient_memory`: Gradient memory in bytes (per-pass form).
- `total_gradient_memory`: Sum of gradient memory across all Ops in this Layer.
- `transformed_out`: Transformed saved output for a single-Op Layer.
- `transformed_out_shape`, `transformed_out_dtype`, `transformed_activation_memory`: Transformed-output companions.
- `transformed_grad`: Transformed gradient for a single-Op Layer.
- `transformed_grad_shape`, `transformed_grad_dtype`, `transformed_gradient_memory`: Transformed-gradient companions.
- `autograd_memory`: Bytes PyTorch autograd retained for backward on this Layer's Op or Ops. Renamed from `autograd_saved_memory`.
- `num_autograd_tensors`: Number of tensors PyTorch autograd retained for backward. Renamed from `num_autograd_saved_tensors`.
- `total_autograd_memory`: Sum of autograd memory across all Ops in this Layer.

### Multi-output Cluster

- `in_multi_output`: True when the output came from a multi-output call.
- `multi_output_type`: Container class.
- `multi_output_index`: 0-based positional access.
- `multi_output_name`: Semantic name.
- `container_path`: Full nested structural path.
- `container_spec`: Structural description of the return container for a single-Op Layer (passthrough from `op.container_spec`).

### Per-Layer Config and Saved State

- `output_device`: Device where saved Layer tensors were placed.
- `activation_transform`: Transform applied to saved outputs.
- `gradient_transform`: Transform applied to saved gradients.
- `annotations`: User-attached metadata for this Layer.
- `detach_saved_activations`: Whether saved tensors were detached for this Layer.
- `save_gradients`: Whether gradient saving was requested for this Layer.
- `has_saved_activation`: True when any Op in this Layer has a saved forward output.
- `has_saved_gradient`: True when any Op in this Layer has a saved gradient.
- `has_saved_args` (`@property`): True when function arguments were saved for a single-Op Layer (passthrough). Raises for multi-Op.
- `saved_args`: Saved positional arguments for a single-Op Layer.
- `saved_kwargs`: Saved keyword arguments for a single-Op Layer.
- `args_template`: Structured argument template for replay/intervention.
- `kwargs_template`: Structured keyword template for replay/intervention.
- `args_summary`: Human-readable summary of positional arguments for a single-Op Layer (passthrough).
- `kwargs_summary`: Human-readable summary of keyword arguments for a single-Op Layer (passthrough).
- `non_tensor_pos_args`: Non-tensor positional arguments for a single-Op Layer (passthrough).
- `non_tensor_kwargs`: Non-tensor keyword arguments for a single-Op Layer (passthrough).
- `code_context`: Python call-stack frames for a single-Op Layer.
- `func_duration`: Function-call duration for a single-Op Layer.
- `total_func_duration`: Sum of function-call duration across all Ops in this Layer.
- `func_rng_states`: RNG state snapshots for a single-Op Layer.
- `func_autocast_state`: Autocast state during the function call for a single-Op Layer (passthrough).
- `lookup_keys`: Alternate lookup keys for this Layer.

### Compute

- `flops_forward`: Approximate forward FLOPs (per-pass).
- `total_flops_forward`: Sum across all Ops in this Layer.
- `flops_backward`, `total_flops_backward`: Backward FLOPs.
- `flops_total`, `total_flops_total`: Total FLOPs.
- `macs_forward`, `total_macs_forward`: Forward MACs.
- `macs_backward`, `total_macs_backward`: Backward MACs.
- `macs_total`, `total_macs_total`: Total MACs.

### Parameters

- `param_shapes`: Shapes of parameters used by this Layer.
- `param_names`: Short names of parameters used by this Layer.
- `param_dtypes`: Dtypes of parameters used by this Layer.
- `num_params`: Number of scalar parameters used by this Layer.
- `num_params_trainable`: Number of trainable scalar parameters used.
- `num_params_frozen`: Number of frozen scalar parameters used.
- `num_param_tensors`: Number of parameter tensors used by this Layer.
- `num_param_tensors_trainable` (`@property`): Number of trainable parameter tensor objects.
- `num_param_tensors_frozen` (`@property`): Number of frozen parameter tensor objects.
- `total_param_memory`: Bytes used by parameters for this Layer.
- `params`: Param accessor for parameters used by this Layer.
- `uses_params` (`@property`): True when this Layer uses any parameters.
- `has_trainable_params` (`@property`): True when this Layer uses at least one trainable parameter (passthrough from its Ops).
- `has_frozen_params` (`@property`): True when this Layer uses at least one frozen parameter.

### Equivalence and Roles

- `equivalence_class`: Loop-detection equivalence class for this Layer.
- `equivalent_ops`: Op labels grouped with this Layer's Ops by equivalence.
- `is_compute_layer` (`@property`): True when the representative Op is compute (`self.ops[0].is_compute_op`). Same semantics as `Op.is_compute_op`: a Layer for `torch.ones` is `is_compute_layer = True` AND `is_internal_source = True`.
- `is_input`: True when this Layer is an input boundary.
- `is_output`: True when this Layer is an output boundary.
- `is_final_output`: True when this Layer marks the final model output.
- `is_buffer_source`: True when this Layer represents a buffer overwrite.
- `address`: Buffer dotted path when this Layer is a buffer boundary (was `buffer_address`).
- `buffer_overwrite_index`: Which overwrite of the buffer this Layer represents.
- `is_internal_source`: True when this Layer starts an internal graph region.
- `is_internal_sink`: True when this Layer ends an internal graph region.
- `is_orphan` (`@property`): Disconnected from the main graph (delegated from Op).
- `is_terminal_bool`: True when this Layer is a terminal boolean for control flow.
- `is_scalar_bool`: True when this Layer output is a scalar boolean.
- `bool_value`: Boolean value when `is_scalar_bool` is true.
- `in_conditionals`: List of `ConditionalRoleRef` records aggregated from this Layer's Ops.
- `is_in_conditional` (`@property`): True when this Layer participates in any conditional role.
- `is_in_conditional_evaluation` (`@property`): True when this Layer computes a conditional arm condition.
- `is_in_conditional_body` (`@property`): True when this Layer is in a conditional arm body.
- `conditional_depth` (`@property`): Number of distinct conditionals this Layer participates in.
- `terminal_bool_for`: `(conditional_label, arm_index)` when this Layer is the terminal bool of an arm.
- `is_atomic_module`: True when this Layer's representative Op is an atomic module — i.e., each Op in this Layer is the sole compute of its ModuleCall. For single-pass Layers, delegates from the one Op. For multi-pass Layers (recurrent atomic module called many times), all passes share this property; the predicate is True.
- `atomic_module_call_label`: Stable ModuleCall label (`:N`-suffixed) for the atomic module output. Single-Op passthrough — for single-pass Layers, returns the one Op's `atomic_module_call_label`; for multi-pass Layers, raises `ValueError` (each pass has its own ModuleCall label; use `layer.ops[N].atomic_module_call_label`).
- `atomic_module_call` (`@property`): ModuleCall record resolved via the label. Single-Op passthrough; same multi-pass behavior as the label field.
- `atomic_module_address`: Module address (PyTorch dotted path; no `:N`) for the atomic module output. Single-Op passthrough — raises for multi-pass Layers. (Conceptually all passes share the same Module address, but passthrough rule applies uniformly across the cluster; use `layer.ops[0].atomic_module_address` for explicit access.)
- `atomic_module` (`@property`): Module record resolved from the address. Single-Op passthrough; raises for multi-pass Layers.
- `is_module_input`: True when this Layer's representative Op's output FEEDS INTO at least one ModuleCall. Layer is OUTSIDE the module. Delegates from `Op.is_module_input` semantics.
- `is_module_output`: True when this Layer's representative Op IS the output of at least one ModuleCall. Layer is INSIDE the module. Delegates from `Op.is_module_output` semantics.

### Module Containment

- `module` (`@property`): Module containing this Layer (innermost from `module_call_stack`); resolves via `self.trace.modules[<address>]`. Raises if no containing module exists.
- `module_call_stack`: List of ModuleCall labels (root-first) active during this Layer's representative Op. Stored as label list per Principle 4.
- `output_of_modules`: List of Module addresses (label strings) for which this Layer is an output.
- `input_to_modules`: List of Module addresses (label strings) for which this Layer is an input.
- `output_of_module_calls`: List of ModuleCall labels this Layer outputs from.
- `input_to_module_calls`: List of ModuleCall labels this Layer inputs to. (Subsumes the former `module_calls_entered` field.)
- `in_submodule`: True when this Layer was computed inside a NON-ROOT module's forward (kept distinct from `is_module_input`/`is_module_output` because "in module" would always be true for the root).
- `module_call_depth`: Depth in the dynamic Module call stack.
- `module_entry_arg_keys`: Dict mapping ModuleCall label (from `input_to_module_calls`) → arg keys this Layer's output filled at that entry event.

### Graph Relations

- `children`: Child Layer labels.
- `parents`: Parent Layer labels.
- `siblings`: Sibling Layer labels that share parents.
- `co_parents`: Other parent Layers that share children.
- `has_children` (`@property`): True when this Layer has children.
- `has_parents` (`@property`): True when this Layer has parents.
- `num_parents` (`@property`): `len(self.parents)`.
- `num_children` (`@property`): `len(self.children)` — mirror the existing `num_parents`.
- `has_siblings` (`@property`): True when this Layer has siblings.
- `has_co_parents` (`@property`): True when this Layer has co-parents.
- `has_input_ancestor`: True when this Layer descends from an input.
- `has_output_descendant`: True when this Layer reaches an output.
- `parent_arg_positions`: Positions in the function call where parent tensors were used (single-Op passthrough).
- `output_descendants`: List of output boundary Layer labels reachable from this Layer.
- `input_ancestors`: List of input boundary Layer labels that can reach this Layer.
- `min_distance_from_input`: Shortest graph distance from an input to this Layer.
- `max_distance_from_input`: Longest graph distance from an input to this Layer.
- `min_distance_to_output`: Shortest graph distance from this Layer to an output.
- `max_distance_to_output`: Longest graph distance from this Layer to an output.
- `root_ancestors`: List of root-graph-ancestor Layer labels.
- `has_internal_source_ancestor`: True when this Layer descends from an internal source.
- `internal_source_parents`: List of Layer labels for immediate parents that are internal sources.
- `internal_source_ancestors`: List of Layer labels for internal-source ancestors reachable upstream.
- `has_out_variations`: True when different children observed different versions of this Layer's output (single-Op passthrough; aggregates if any pass had variations for multi-Op).
- `out_versions_by_child`: Per-child output snapshots when in-place ops created divergent views.
- `is_output_parent`: True when this Layer's tensor was directly returned from forward.
- `interventions`: Intervention records applied to this Layer's Ops (aggregated across passes).

### Pass Management

- `ops`: Scoped `OpAccessor` for this Layer's Ops. **0-based positional integer indexing** (Python list-like — `layer.ops[0]` is the first pass, `layer.ops[-1]` is the last). Also accepts Op labels (both short form `conv2d_2:1` and long form `conv2d_2_3:1` resolve to the same Op when unambiguous). **For single-pass Layers**, the bare Layer label (`conv2d_2_3`, no `:N`) also resolves to the unique Op. For multi-pass Layers, the bare label raises `AmbiguousOpLookupError` (use `[N]` or a `:N`-qualified label). Supports `len(layer.ops)` and iteration in pass order.
- `op_labels`: Op labels belonging to this Layer.

### Layer Methods

- `to_pandas()`: Export this Layer as a one-row DataFrame.
- File export: use `torchlens.export.csv(layer, path)`, `torchlens.export.parquet(layer, path)`, or `torchlens.export.json(layer, path)`.

## Module / ModuleCall

### ModuleCall Identity

- `call_index`: 1-based invocation index for this Module call.
- `call_label`: Pass-qualified ModuleCall label, usually `address:N`.
- `address`: Primary address of the Module being called (label string). Stored form. Bare name because ModuleCall's own identity is `call_label` (not `address`), so there's no collision — `mc.address` unambiguously refers to the called Module.
- `name` (`@property`): Bare local name of the called Module (last segment of `address`). Bare OK — ModuleCalls don't have their own names.
- `cls` (`@property`): Runtime Python type of the called Module.
- `class_name` (`@property`): Short Python class name of the called Module (delegates to `self.module.class_name`).
- `class_qualname` (`@property`): Fully qualified Python class name of the called Module.
- `module` (`@property`): Module record being called, resolved via `self.trace.modules[self.address]`. Parallels `Param.module`. (Bare resolver; a ModuleCall is invoked by ONE Module — there is no plural counterpart, unlike `Param.modules` which exists because a single nn.Parameter can be shared across multiple owning Modules.)
- `all_addresses`: List of all addresses sharing the same Module object (when nn.Module is registered at multiple paths). Stored form. For per-address access to the Module record, use `mc.module.all_addresses` (same list, via the resolver) or iterate `[trace.modules[a] for a in mc.all_addresses]` explicitly.
- `has_multiple_addresses` (`@property`): True when the called Module object appears at multiple addresses (`len(all_addresses) > 1`).
- `ordinal_index`: 0-based position in **`trace.module_calls`** (the trace-level accessor over ALL ModuleCalls across all Modules). Use `trace.module_calls[mc.ordinal_index] is mc` for round-trip lookup. Position within the scoped `module.calls` accessor is derived from `call_index - 1` (since `module.calls` is 0-based positional per the harmonized accessor convention).

### ModuleCall Layers and Args

- `ops`: List of pass-qualified Op labels (`conv2d_1_2:1`) computed during this ModuleCall (ALL ops, not just outputs). Bare label list.
- `num_ops`: Number of Ops associated with this call.
- `num_internal_edges` (`@property`): Edges (p,c) with BOTH endpoints in this scope's op set.
- `num_input_edges` (`@property`): Edges with child IN this scope's op set and parent NOT in it (inward boundary crossings = module fan-in).
- `num_output_edges` (`@property`): Edges with parent IN this scope's op set and child NOT in it (outward = module fan-out).
- `num_edges` (`@property`): Internal + input + output (the scope's total edge footprint).
- `input_ops`: List of pass-qualified Op labels at this call's input boundary positions. Bare label list.
- `input_layers`: List of Layer labels at this call's input boundary positions. Bare label list.
- `output_ops`: List of pass-qualified Op labels for the actual output Ops of this call. Bare label list.
- `output_layers`: List of Layer labels at this call's output boundary positions. Bare label list.
- `output_structure`: `ContainerSpec | None` describing the shape of the call's return container.

Boundary input/output collections are stored as bare label lists (no `_labels` suffix needed — per the convention, the suffix only appears to disambiguate against a bare-name `@property` resolver, and these have no such resolver). Users do `[trace.ops[lbl] for lbl in mc.input_ops]` to resolve when needed.
- `forward_args`: Positional arguments passed to `forward`.
- `forward_kwargs`: Keyword arguments passed to `forward`.
- `forward_args_summary`: Human-readable summary of forward positional arguments.
- `forward_kwargs_summary`: Human-readable summary of forward keyword arguments.
- **`forward_args_template`**: **Structural skeleton (shapes-not-values) of positional forward args. Same semantics as Op's `args_template`. (LOCKED 2026-05-23)**
- **`forward_kwargs_template`**: **Structural skeleton of keyword forward args. Same semantics as Op's `kwargs_template`. (LOCKED 2026-05-23)**
- **`forward_arg_names`**: **Names of the `forward()` arguments (parity with `Op.arg_names`). (LOCKED 2026-05-21)**
- **`num_forward_args_total`**: **Total positional + keyword forward args (parity with `Op.num_args_total`). (LOCKED 2026-05-21)**
- **`num_forward_pos_args`**: **Positional forward-arg count. (LOCKED 2026-05-21)**
- **`num_forward_kwargs`**: **Keyword forward-arg count. (LOCKED 2026-05-21)**
- **`has_saved_forward_args`** (`@property`): **Whether `forward_args` was captured (parity with `Op.has_saved_args`). (LOCKED 2026-05-21)**
- `call_parent_label`: Parent ModuleCall label in the dynamic call tree. Stored form.
- `call_parent` (`@property`): Parent ModuleCall record resolved via `self.trace.module_calls[self.call_parent_label]`. Returns `None` for the root call.
- `call_children_labels`: List of child ModuleCall labels in the dynamic call tree. Stored form (portable).
- `call_children` (`@property`): List of child ModuleCall records resolved via `[self.trace.module_calls[lbl] for lbl in self.call_children_labels]`. Paired plural resolver matches the singular `call_parent` resolver for return-type consistency.
- **`num_descendant_calls`** (`@property`): **Total ModuleCalls nested beneath this call (transitive over `call_children`). (LOCKED 2026-05-23)**
- **`max_descendant_depth`** (`@property`): **Deepest nesting beneath this call. (LOCKED 2026-05-23)**
- **`walk_descendants()`**: **Iterate the subtree rooted at this ModuleCall. `ModuleCall` itself is the tree node via `call_parent` and `call_children`. (LOCKED 2026-05-23)**
- **`show_call_tree(max_depth=None, include_atomic=False, show_call_index=False, file=None)`**: **ASCII tree printed for the subtree rooted at this call. (LOCKED 2026-05-23)**

### ModuleCall Timing

- `forward_duration` (`@property`): Wall-clock seconds from the start of this ModuleCall's `forward()` invocation to its return. Includes Python orchestration inside the module, wrapper overhead per op call, and time spent in nested module calls. Parallels `Trace.forward_duration` at one-call granularity.
- `func_calls_duration` (`@property`): Sum of `op.func_duration` for ALL ops executed during this ModuleCall (inclusive — includes ops in nested module calls). Pure torch-compute slice; matches the Trace-level `func_calls_duration` naming.
- **`backward_duration`**: **Wall-clock backward duration for this call. NAMES LOCKED 2026-05-23; impl lands in the backward-pass unified sprint. Mirrors the existing `forward_duration` pattern.**

The difference `forward_duration - func_calls_duration` ≈ wrapper overhead + Python orchestration inside this call (inclusive of nested calls).

### ModuleCall Memory (LOCKED 2026-05-23)

Three-quantity memory cluster at ModuleCall scope. Two compose axes:

- `output_` (boundary — output of THIS call) vs `internal_` (sum of inside ops in THIS call)
- `activation` / `gradient` / `autograd` / `param` quantity

The `output_` / `internal_` prefixes are always explicit at Module/ModuleCall scope (the two quantities are both real and frequently confused; both pay the disambiguation tax). At Op/Layer scope, bare `activation_memory` remains — no `internal` exists there.

- **`output_activation_memory`** (`@property`): **Output activation memory of THIS call (bytes).**
- **`internal_activation_memory`** (`@property`): **Sum of internal-op activation memory in THIS call.**
- **`output_gradient_memory`** (`@property`): **Output gradient memory of THIS call.**
- **`internal_gradient_memory`** (`@property`): **Sum of internal-op gradient memory in THIS call.**
- **`autograd_memory`** (`@property`): **Autograd-saved memory during THIS call (inherently internal — no boundary form).**
- **`param_memory`** (`@property`): **Own-address param memory of the called Module (call-invariant).**
- **`internal_param_memory`** (`@property`): **Address-recursive sum of `param_memory` under this Module's subtree (call-invariant). Shares its computation path with the `recursive_params` accessor.**

### ModuleCall Call Context

Two "call stack" concepts at the ModuleCall scope — parallel to the same concepts on Op:

- `code_context`: Python call-stack frames at the moment this ModuleCall's `forward()` was invoked, showing where in user/PyTorch code the call originated (list of `FuncCallLocation`). Same semantic as `Op.code_context` but recorded at module-call entry time.
- `module_call_stack`: List of ModuleCall labels (root-first) for ancestor ModuleCalls active when this ModuleCall was invoked. The chain from the outermost call down to this call's IMMEDIATE PARENT (excludes self). Parallels `Op.module_call_stack` (which records the ambient ModuleCall stack at op-execution time). `len(module_call_stack)` equals `call_depth`.

### ModuleCall Parameters

Parameter counts on ModuleCall mirror the parent Module's params (parameters are static during forward; the call doesn't have its own params, it invokes a Module that owns them). All fields are `@property` delegations from `self.module`.

- `params` (`@property`): ParamAccessor for parameters of the invoked Module (delegates to `self.module.params`).
- `num_params` (`@property`): Scalar parameter count of the invoked Module.
- `num_params_trainable` (`@property`): Trainable scalar parameter count.
- `num_params_frozen` (`@property`): Frozen scalar parameter count.
- `num_param_tensors` (`@property`): Number of parameter tensors owned by the invoked Module.
- `num_param_tensors_trainable` (`@property`): Number of trainable parameter tensors.
- `num_param_tensors_frozen` (`@property`): Number of frozen parameter tensors.
- `has_trainable_params` (`@property`): True when the called Module has at least one trainable parameter (delegated from `self.module.has_trainable_params`).
- `has_frozen_params` (`@property`): True when the called Module has at least one frozen parameter (delegated from `self.module.has_frozen_params`).

### ModuleCall Output Passthroughs

These properties resolve through the output Ops. Singular forms
require exactly one output and raise `MultiOutputModuleError` on multi-output
calls; plural forms return lists in container-path order.

Per-output memory is NOT exposed as a bare passthrough here — the
`output_`/`internal_`-prefixed ModuleCall memory cluster
(`output_activation_memory`, `internal_activation_memory`,
`output_gradient_memory`, `internal_gradient_memory`, `autograd_memory`,
`param_memory`) already covers module-scope memory and is unambiguous about
the boundary-vs-internal distinction.

- `out` / `outs`: Saved output tensor or output tensors.
- `out_shape` / `out_shapes`: Output shape or shapes.
- `out_dtype` / `out_dtypes`: Output dtype or dtypes.
- `grad` / `grads`: Saved output gradient or gradients.

### Module Identity

`Module` is the aggregate record for one `nn.Module` address. Disambiguate
from `torch.nn.Module` via `tl.Module` (qualified) — TorchLens docs always
show the qualified form.

- `address`: Primary dotted PyTorch module address.
- `name` (`@property`): Last segment of `address` (e.g., `"layer1"` for `"encoder.block.layer1"`). Fills the Module/Param asymmetry.
- `all_addresses`: All addresses for a shared module object.
- `has_multiple_addresses`: True when a module object has multiple addresses.
- `class_name`: Short Python class name for this Module.
- `cls`: Runtime Python type object for the Module; spelled `cls` because `class` is a Python keyword.
- `class_qualname`: Fully qualified Python class name.
- `ordinal_index`: 0-based position in `trace.modules`.
- `trace`: Runtime-only back-pointer to the owning Trace (typically weakref). Not portable; unavailable after `.tlspec` load.

### Module Source Info

- `class_source_file`: File where the module class is defined, when available.
- `class_source_line`: Line where the module class is defined.
- `class_source_location` (`@property`): Combined `class_source_file:class_source_line`.
- `class_docstring`: Docstring for the module class.
- `init_source_file`: File where `__init__` is defined.
- `init_source_line`: Line where `__init__` is defined.
- `init_source_location` (`@property`): Combined `init_source_file:init_source_line`.
- `init_signature`: Signature of `__init__`, when introspectable.
- `init_docstring`: Docstring of `__init__`.
- `forward_source_file`: File where `forward()` is defined.
- `forward_source_line`: Line where `forward()` is defined.
- `forward_source_location` (`@property`): Combined `forward_source_file:forward_source_line`.
- `forward_signature`: Signature of `forward`, when introspectable.
- `forward_docstring`: Docstring of `forward`.

All triplets gracefully return `None` when the underlying method isn't
user-defined (e.g., default `nn.Module.__init__`).

### Module Hierarchy

The address tree is the static `nn.Module` registration hierarchy; the call
tree is what actually happened as modules called one another during `forward`.

- `address_parent`: Parent module address (label string) in the static `nn.Module` tree.
- `address_children`: List of child module addresses (label strings) in the static tree.
- `address_depth`: Depth in the static address tree.
- `call_parent_address`: Parent Module address (label string) in the dynamic call tree for single-call Modules. Stored form.
- `call_parent` (`@property`): Parent Module record resolved via `self.trace.modules[self.call_parent_address]`. Returns `None` for the root module.
- `call_children_addresses`: List of child Module addresses (label strings) in the dynamic call tree. Stored form (portable).
- `call_children` (`@property`): List of child Module records resolved via `[self.trace.modules[a] for a in self.call_children_addresses]`. Paired plural resolver matches the singular `call_parent` resolver for return-type consistency.
- `call_depth`: Depth in the dynamic call stack.
- `num_calls`: Number of times this Module was called.
- **`num_descendant_calls`** (`@property`): **Total ModuleCalls nested under any call of this Module. (LOCKED 2026-05-23)**
- **`max_descendant_depth`** (`@property`): **Deepest nesting beneath this Module. (LOCKED 2026-05-23)**
- **`walk_descendants()`**: **Iterate the call subtree rooted at this Module (aggregate across all calls). `ModuleCall` itself is the tree node via `call_parent` and `call_children`. (LOCKED 2026-05-23)**
- **`show_call_tree(max_depth=None, include_atomic=False, show_call_index=False, file=None)`**: **ASCII tree printed for the subtree rooted at this Module. (LOCKED 2026-05-23)**
- `calls`: Scoped `ModuleCallAccessor` for this Module's calls. **0-based positional integer indexing** (Python list-like — `module.calls[0]` is the first call). Also accepts ModuleCall labels. **For single-call Modules**, the bare Module address (`encoder.block`, no `:N`) also resolves to the unique ModuleCall. For multi-call Modules, the bare address raises (use `[N]` or a `:N`-qualified label).
- `call_labels`: Labels for this Module's calls.

### Module Parameters and Buffers

Direct (own-address) param fields are bare; address-recursive (this Module + all address-based sub-Modules per PyTorch `parameters(recurse=True)`) carry the `recursive_` prefix.

- `params`: ParamAccessor for parameters owned by this Module.
- `buffers`: Scoped BufferAccessor for persistent `Buffer` entities belonging to this Module — its own address and address-based sub-modules (parallels `params`).
- `num_params`: Scalar parameter count owned by this Module.
- `num_param_tensors` (`@property`): Number of parameter tensors owned.
- `num_params_trainable`: Trainable scalar parameter count.
- `num_params_frozen`: Frozen scalar parameter count.
- `num_param_tensors_trainable` (`@property`): Number of trainable parameter tensors.
- `num_param_tensors_frozen` (`@property`): Number of frozen parameter tensors.
- `total_param_memory`: Bytes used by this Module's parameters.
- `has_trainable_params` (`@property`): True when any owned parameter is trainable.
- `has_frozen_params` (`@property`): True when any owned parameter is frozen (i.e., `requires_grad=False`).
- **`recursive_params`** (`@property`): **Accessor over directly-owned params plus all params of address-based sub-Modules (matches PyTorch `parameters(recurse=True)`). (LOCKED 2026-05-23)**
- **`num_recursive_params`** (`@property`): **Scalar parameter count, address-recursive. (LOCKED 2026-05-23)**
- **`num_recursive_params_trainable`** (`@property`): **Trainable scalar parameter count, address-recursive. (LOCKED 2026-05-23)**
- **`num_recursive_params_frozen`** (`@property`): **Frozen scalar parameter count, address-recursive. (LOCKED 2026-05-23)**
- **`num_recursive_param_tensors`** (`@property`): **Parameter tensor count, address-recursive. (LOCKED 2026-05-23)**
- **`num_recursive_param_tensors_trainable`** (`@property`): **Trainable parameter tensor count, address-recursive. (LOCKED 2026-05-23)**
- **`num_recursive_param_tensors_frozen`** (`@property`): **Frozen parameter tensor count, address-recursive. (LOCKED 2026-05-23)**
- **`recursive_param_addresses`** (`@property`): **Label list of all recursive param addresses. (LOCKED 2026-05-23)**
- `buffer_layers`: Deferred to the integrated buffer rethink.

### Module Memory (LOCKED 2026-05-23)

Cross-call aggregates only at Module scope — calls of the same Module can have varying output shapes (variable-sequence-length LSTMs, conditional branches), so a "representative single-call value" is ambiguous. Drill into a specific `ModuleCall` for per-call values.

The naming asymmetry: at Op/Layer scope, bare `activation_memory` is fine (no internal exists). At ModuleCall/Module scope, `output_`/`internal_` prefixes are always explicit.

- **`total_output_activation_memory`** (`@property`): **Sum of `output_activation_memory` across all calls.**
- **`total_internal_activation_memory`** (`@property`): **Sum of `internal_activation_memory` across all calls.**
- **`total_output_gradient_memory`** (`@property`): **Sum of `output_gradient_memory` across all calls.**
- **`total_internal_gradient_memory`** (`@property`): **Sum of `internal_gradient_memory` across all calls.**
- **`total_autograd_memory`** (`@property`): **Sum of `autograd_memory` across all calls (the existing `total_autograd_memory` name already locked at this form).**
- **`param_memory`** (`@property`): **Own-address param memory (call-invariant).**
- **`internal_param_memory`** (`@property`): **Address-recursive sum of `param_memory` under this Module's subtree (call-invariant). Shares its computation path with `recursive_params`.**

### Module State

- `training`: True when the Module was in PyTorch train mode at capture (matches `nn.Module.training`).
- `forward_pre_hooks`: `list[HookInfo]` for registered forward pre-hooks (one HookInfo per hook).
- `forward_hooks`: `list[HookInfo]` for registered forward hooks.
- `backward_pre_hooks`: `list[HookInfo]` for registered backward pre-hooks (legacy).
- `backward_hooks`: `list[HookInfo]` for registered backward hooks (legacy).
- `full_backward_pre_hooks`: `list[HookInfo]` for registered full backward pre-hooks.
- `full_backward_hooks`: `list[HookInfo]` for registered full backward hooks.
- `has_forward_hooks` (`@property`): True when `len(forward_pre_hooks) + len(forward_hooks) > 0`.
- `has_backward_hooks` (`@property`): True when any of the backward / full_backward / *_pre lists is non-empty.
- `HookInfo.name`: Hook function `__name__` (singular per hook).
- `HookInfo.qualname`: Fully qualified hook name (singular per hook).
- `HookInfo.source_location`: `FuncCallLocation` record for the hook definition (singular per hook).
- `custom_attributes`: User-defined instance attributes beyond standard `nn.Module` state.
- `custom_methods`: User-defined methods beyond inherited `nn.Module` methods.

PyTorch defines 2 forward + 4 backward hook registries (legacy + full
variants for backward). The "full" variants exist only for backward,
added later to fix legacy backward hook quirks. TorchLens mirrors all
six registries exactly.

### Module Compute

- `total_flops_forward`: Sum of forward FLOPs across Layers in this Module.
- `total_flops_backward`: Sum of backward FLOPs across Layers.
- `total_flops`: Sum of forward and backward FLOPs.
- `total_macs_forward`: Approximate forward MACs across this Module.
- `total_macs_backward`: Approximate backward MACs.
- `total_macs`: Approximate total MACs.

### Module Timing

- `forward_duration` (`@property`): Wall-clock seconds of this Module's single `forward()` call (single-call passthrough). Raises for multi-call Modules (use `module.calls[N].forward_duration`).
- `total_forward_duration` (`@property`): Sum of `call.forward_duration` across ALL calls of this Module.
- `func_calls_duration` (`@property`): Sum of `op.func_duration` for ALL ops executed during this Module's single call. Raises for multi-call Modules. Inclusive (includes nested module calls).
- `total_func_calls_duration` (`@property`): Sum of `call.func_calls_duration` across ALL calls of this Module.
- **`total_backward_duration`** (`@property`): **Sum of `call.backward_duration` across ALL calls of this Module. NAMES LOCKED 2026-05-23; impl lands in the backward-pass unified sprint.**

### Module Layer Access

**Module-scope boundary fields are AGGREGATE across calls** (union of per-call boundary positions), not single-call passthroughs. The exception is `output_structure` which IS passthrough.

- `layer_labels`: Aggregate union of Layer labels across this Module's calls (bare label list — keep with `_labels` here because `layers` is taken by the @property below).
- `layers` (`@property`): Aggregate union of Layer records across this Module's calls (resolved). Kept as @property — high-traffic iteration surface.
- `num_layers`: Number of aggregate Layers inside this Module.
- `num_internal_edges` (`@property`): Edges (p,c) with BOTH endpoints in this scope's op set.
- `num_input_edges` (`@property`): Edges with child IN this scope's op set and parent NOT in it (inward boundary crossings = module fan-in).
- `num_output_edges` (`@property`): Edges with parent IN this scope's op set and child NOT in it (outward = module fan-out).
- `num_edges` (`@property`): Internal + input + output (the scope's total edge footprint).
- `input_ops`: Aggregate union of input Op labels across this Module's calls. Bare label list.
- `input_layers`: Aggregate union of input Layer labels across this Module's calls. Bare label list.
- `output_ops`: Aggregate union of output Op labels across this Module's calls. Bare label list.
- `output_layers`: Aggregate union of output Layer labels across this Module's calls. Bare label list.

Boundary input/output collections are bare label lists at Module aggregate scope — no `_labels` suffix needed (no resolver companion to disambiguate from). Same rule as ModuleCall: users resolve via comprehension when records are needed.
- `output_structure`: `ContainerSpec | None` for single-call Modules; reads through to the call's `output_structure`. Raises for multi-call Modules (use `module.calls[N].output_structure`).

### Module Forward Args

- `forward_args`: Positional args for a single-call Module; raises for multi-call Modules.
- `forward_kwargs`: Keyword args for a single-call Module; raises for multi-call Modules.
- `forward_args_summary`: Human-readable args for a single-call Module.
- `forward_kwargs_summary`: Human-readable kwargs for a single-call Module.
- **`forward_args_template`** (`@property`): **Structural skeleton (shapes-not-values) for a single-call Module; raises for multi-call Modules. (LOCKED 2026-05-23)**
- **`forward_kwargs_template`** (`@property`): **Structural kwargs skeleton for a single-call Module; raises for multi-call Modules. (LOCKED 2026-05-23)**

### Module Output Passthroughs

These mirror ModuleCall output passthroughs for single-call Modules;
multi-call Modules raise on singular access and require `module.calls[N]`.
As at ModuleCall scope, per-output memory is NOT exposed as a bare
passthrough here — use the cross-call `total_output_*` / `total_internal_*`
memory cluster (see Module Memory) for module-scope memory.

- `out` / `outs`: Saved output tensor or output tensors.
- `out_shape` / `out_shapes`: Output shape or shapes.
- `grad` / `grads`: Saved output gradient or gradients.

### Module Methods

- `draw(**kwargs)`: Draw this Module's subgraph.
- `to_pandas()`: Export this Module or ModuleCall as tabular data.
- File export: use `torchlens.export.csv(module_or_call, path)`, `torchlens.export.parquet(module_or_call, path)`, or `torchlens.export.json(module_or_call, path)`.

### Module Facets (LOCKED 2026-05-27)

- **`facets`** (`FacetView`): **Derived semantic views populated by recipes matched against this Module (class / predicate match). Lazy + cached. Dict-like + attribute-access: `module.facets.X` and `module.facets['X']` both work. Empty FacetView when no recipe matches. Common built-in facets for attention modules: `q`, `k`, `v`, `attn_out`, `input`, `residual`, `n_heads`, `n_q_heads`, `n_kv_heads`, `d_head`, `head(i)` sub-view method. For normalization modules: `normalized`, `gamma`, `beta`, `input`. For MLP modules: `intermediate`, `up_out`, `gated_out`, `down_out`, `input`, `output`. For Embedding: `lookup`, `weight`, `indices`. See the dedicated Facets section at the end of the glossary for the full registration API and recipe table.**

## Param

### Identity

- `address`: Primary dotted PyTorch parameter address, matching `named_parameters()` style.
- `name`: Last segment of `address`.
- `all_addresses`: All parameter addresses sharing the same `nn.Parameter`.
- `has_multiple_addresses`: True when the same parameter tensor is registered at multiple addresses.
- `ordinal_index`: 0-based position in `trace.params`.
- `trace`: Runtime-only back-pointer to the owning Trace (typically weakref). Not portable; unavailable after `.tlspec` load.

### Tensor Properties

- `shape`: Shape of the parameter tensor.
- `dtype`: Dtype of the parameter tensor.
- `dtype_ref`: Backend-neutral dtype reference for the parameter leaf.
- `device_ref`: Backend-neutral device reference for the parameter leaf when available.
- `backend_address`: Backend-native parameter address or path, usually the torch dotted address for
  native-module parameters.
- `resolver_status`: Resolution status for backend-native metadata, currently `"resolved"`,
  `"metadata_only"`, or `"unresolved"`.
- `num_params`: Number of scalar parameters in this tensor (equivalent to `numel()`).
- `memory`: Bytes used by the parameter tensor.

### Status Flags

- `is_trainable`: True when this parameter has `requires_grad=True`.
- `is_quantized`: True when this parameter uses a quantized dtype or representation.
- `has_optimizer`: True when optimizer metadata is associated with this parameter.

### Module Ownership

- `module_address`: Primary owning Module address (label string). Stored form.
- `module` (`@property`): Primary owning Module record resolved via `self.trace.modules[self.module_address]`. Raises if not resolvable.
- `module_name` (`@property`): Bare local name of the owning module (last segment of `module_address`).
- `module_cls` (`@property`): Runtime Python type object of the owning Module.
- `all_module_addresses`: List of all owning Module addresses (label strings) for shared parameters. Stored form (portable).
- `modules` (`@property`): List of all owning Module records resolved via `[self.trace.modules[a] for a in self.all_module_addresses]`. Paired plural resolver matches the singular `module` resolver for return-type consistency (per revised Principle 4).

### Usage Tracking

- `used_by_ops`: List of pass-qualified Op labels (`conv2d_1_2:1` form) that used this parameter — fine-grained per-pass record. Bare label list. Each Op-usage counts once regardless of how many times the param appears in the op's args (i.e., `mul(w, w)` counts as ONE op-usage).
- `used_by_layers`: List of Layer labels (no `:N`) that used this parameter — equivalence-class aggregated. Bare label list. A recurrent Layer with N passes using this param yields ONE entry here, but N entries in `used_by_ops`.
- `num_uses_by_ops` (`@property`): Count = `len(used_by_ops)`. Number of distinct Op usages.
- `num_uses_by_layers` (`@property`): Count = `len(used_by_layers)`. Number of distinct Layer equivalence-class positions where this parameter was used.

Example: a recurrent `conv2d_1_2` Layer with 3 passes that uses a `weight` Param once per pass yields `num_uses_by_ops = 3` and `num_uses_by_layers = 1`.
- `co_parent_params`: List of Param addresses (label strings) for parameters that co-occur with this one in the same operation (e.g., conv weight + conv bias both feeding a Conv2d op). Family-vocab match with the `co_parents` graph relationship.

Note: Denormalized owner-class info (`module_class_name`, `module_class_qualname`, `module_type`) has been removed. Access via `param.module.class_name`, `param.module.class_qualname`, `param.module.cls` instead.

### Gradient Family

- `has_grad`: True when a gradient was observed for this parameter.
- `grad`: Live `nn.Parameter.grad` tensor when the runtime parameter reference is available.
- `grad_shape`: Shape of the parameter gradient.
- `grad_dtype`: Dtype of the parameter gradient.
- `gradient_memory`: Bytes used by the parameter gradient.

### Param Methods

- `release_param_ref()`: Release the live parameter reference while keeping portable metadata.
- `to_pandas()`: Export this Param as a one-row DataFrame.
- File export: use `torchlens.export.csv(param, path)`, `torchlens.export.parquet(param, path)`, or `torchlens.export.json(param, path)`.

## Buffer

Buffers receive values from upstream ops but don't propagate gradients
back the way activations do. They are gradient dead-ends.

### Buffer Identity

- `address`: Dotted PyTorch buffer address (formerly `buffer_address`). Buffer is an address-level entity; graph versions are plain `Op` records with `is_buffer=True`.
- `name` (`@property`): Last segment of `address`.
- `all_addresses`: All addresses sharing the same buffer object.
- `has_multiple_addresses`: True when the same buffer appears at multiple addresses.
- `ordinal_index`: 0-based position in `trace.buffers`.

### Buffer Dynamics

- `initial_value`: Pre-forward value for this registered-buffer address.
- `final_value`: Final observed value after capture.
- `versions`: Ordered buffer-version `Op` nodes for this address.
- `write_versions`: Ordered write-produced version nodes, excluding static initial reads.
- `buffer_overwrite_index`: 1-based index of the most recent buffer version for its address.
- `buffer_source`: Stable Op label for the most recent Op that overwrote this buffer value, or `None` for static buffers.
- `is_overwritten`: True when this buffer is overwritten during forward (vs. static buffers that aren't).
- `num_overwrites`: Total overwrites of this buffer's address during the trace.
- `value_at(n)`: 1-based value lookup by observed version index.
- `value_after(n)`: 1-based value lookup by overwrite index.

### Inherited Op-Like Fields

These fields are projected from the final graph version node.

- `layer_label`: Layer label associated with the final buffer version.
- `shape`: Shape of the buffer value.
- `dtype`: Dtype of the buffer value.
- `activation_memory`: Bytes used by the buffer value.
- `has_saved_activation`: True when the buffer value was saved.
- `has_saved_gradient`: True when a gradient was saved for the buffer record.
- `module` (`@property`): Module containing the buffer use; resolves via `self.trace.modules[<address>]`. Raises if no containing module exists.
- `grad_shape`: Shape of saved gradient, if any.
- `grad_dtype`: Dtype of saved gradient, if any.
- `gradient_memory`: Bytes used by saved gradient, if any.

### Buffer Methods

- `to_pandas()`: Export this Buffer as tabular data.
- File export: use `torchlens.export.csv(buffer, path)`, `torchlens.export.parquet(buffer, path)`, or `torchlens.export.json(buffer, path)`.

## GradFn / GradFnCall

### GradFnCall

- `call_index`: 1-based hook-firing or backward-call index for this GradFn.
- `call_label`: Stable label for this GradFnCall.
- `backward_duration`: Time spent in this GradFnCall (one hook firing or backward call event) when duration capture is implemented. Renamed from bare `duration` for direction-explicit naming consistent with the rest of the API.
- `grad_inputs`: Gradient inputs observed by the backward hook.
- `grad_outputs`: Gradient outputs observed by the backward hook.
- `is_saved` (`@property`): True when `grad_inputs is not None OR grad_outputs is not None`.
- `ordinal_index`: 0-based position in **`trace.grad_fn_calls`** (the trace-level accessor over ALL GradFnCalls across all GradFns). Use `trace.grad_fn_calls[gc.ordinal_index] is gc` for round-trip lookup. Position within the scoped `grad_fn.calls` accessor is derived from `call_index - 1` (since `grad_fn.calls` is 0-based positional per the harmonized accessor convention).
- `to_pandas()`: Export this GradFnCall as a one-row DataFrame.
- File export: use `torchlens.export.csv(grad_fn_call, path)`, `torchlens.export.parquet(grad_fn_call, path)`, or `torchlens.export.json(grad_fn_call, path)`.

### GradFn Identity

- `class_name`: Short class name of the PyTorch grad-fn object.
- `cls` (`@property`): Runtime Python type object for the grad-fn, when introspectable; may be `None` after load for C++ autograd internals.
- `class_qualname`: Fully qualified class name of the grad-fn object.
- `label`: Stable TorchLens label for this GradFn.
- `trace`: Runtime-only back-pointer to the owning Trace (typically weakref). Not portable; unavailable after `.tlspec` load.

### GradFn Type Info

- `type`: Normalized grad-fn type token.
- `type_index`: 1-based index among grad-fns of the same type.
- `step_index`: 1-based index across all grad-fns in backward execution order. Parallels `Op.step_index` (sequential numbering across the relevant computation domain). Renamed from `trace_index` for symmetry with the Op/Layer naming convention. (Unlike Op.step_index which is 0 for boundary ops, GradFn.step_index is always >= 1 — no boundary-equivalent on the backward side.)
- `ordinal_index`: 0-based position in `trace.grad_fns`.
- `is_custom`: True when this GradFn comes from a custom autograd Function.

### GradFn Source / Signature / Docstring

These fields are best-effort introspection data; built-in PyTorch grad-fns
may have `None` values or only stub-file locations.

- `class_source_file`: File defining the grad-fn class or stub.
- `class_source_line`: Line defining the grad-fn class or stub.
- `class_source_location` (`@property`): Combined `class_source_file:class_source_line`.
- `class_docstring`: Docstring for the grad-fn class.
- `init_source_file`: File defining `__init__`.
- `init_source_line`: Line defining `__init__`.
- `init_source_location` (`@property`): Combined.
- `init_signature`: Signature of `__init__`, when available (often None for autograd Functions).
- `init_docstring`: Docstring of `__init__`.
- `forward_source_file`: File defining the custom Function `forward` method.
- `forward_source_line`: Line defining `forward`.
- `forward_source_location` (`@property`): Combined.
- `forward_signature`: Signature of the custom Function forward method.
- `forward_docstring`: Docstring of the forward method.
- `backward_source_file`: File defining the custom Function `backward` method.
- `backward_source_line`: Line defining `backward`.
- `backward_source_location` (`@property`): Combined.
- `backward_signature`: Signature of the custom Function backward method.
- `backward_docstring`: Docstring of the backward method.

### GradFn Graph Relations

**Orientation convention:** Graph relations on GradFn follow BACKWARD execution order — the autograd graph's own direction. Parents = grad_fns that fire BEFORE this one in backward (closer to the loss); children = grad_fns that fire AFTER this one in backward (closer to the leaves). This matches PyTorch's `tensor.grad_fn.next_functions` semantics (next_functions = backward-children in this convention).

In forward terms: a GradFn's parents (backward orientation) correspond to the forward-DOWNSTREAM of its source Op; its children correspond to forward-UPSTREAM of its source Op. The relations FLIP between forward graph and backward graph because they're different DAGs.

To walk the forward graph from a GradFn, use `grad_fn.op.parents` / `grad_fn.op.children` — the forward direction is accessible via the resolver `grad_fn.op`.

- `parents`: List of GradFn labels that fire BEFORE this GradFn in backward execution (autograd-graph upstream). Equivalent to: "grad_fns whose `next_functions` includes this one."
- `children`: List of GradFn labels that fire AFTER this GradFn in backward execution (autograd-graph downstream). Equivalent to: this GradFn's `next_functions` translated to TorchLens labels.
- `siblings`: GradFns sharing a backward-parent (i.e., they also fire after the same upstream backward node).
- `co_parents`: GradFns sharing a backward-child (i.e., they also feed gradients into the same downstream backward node).
- `has_parents` (`@property`): True when this GradFn has backward parents.
- `has_children` (`@property`): True when this GradFn has backward children.
- `has_siblings` (`@property`): True when this GradFn has siblings.
- `has_co_parents` (`@property`): True when this GradFn has co-parents.
- `op_label`: Stable Op label corresponding to this GradFn, if one exists.
- `op` (`@property`): Op corresponding to `op_label`, resolved via `self.trace.ops[self.op_label]`. Returns `None` when `op_label is None`.
- `has_op`: True when a forward Op was captured for this grad_fn. (Storage flip — was `is_intervening` with inverted polarity. Aligns with `Trace.num_grad_fns_without_op`.)
- `has_saved_call` (`@property`): True when this GradFn has at least one saved GradFnCall.

### GradFn Calls

- `num_calls`: Number of times this GradFn hook fired.
- `calls`: Scoped `GradFnCallAccessor` for this GradFn's calls. **0-based positional integer indexing** (Python list-like — `grad_fn.calls[0]` is the first call). Also accepts GradFnCall labels. **For single-call GradFns**, the bare GradFn label (no `:N`) also resolves to the unique GradFnCall. For multi-call GradFns, the bare label raises (use `[N]` or a `:N`-qualified label).
- `call_labels`: Labels for this GradFn's calls.

### GradFn Timing

- `backward_duration` (`@property`): Single-call passthrough — time spent in this GradFn's single GradFnCall. Raises `ValueError` for multi-call GradFns (use `grad_fn.calls[N].backward_duration`).
- `total_backward_duration` (`@property`): Sum of `backward_duration` across all this GradFn's calls.

### GradFn Methods

- `to_pandas()`: Export this GradFn as a one-row DataFrame.
- File export: use `torchlens.export.csv(grad_fn, path)`, `torchlens.export.parquet(grad_fn, path)`, or `torchlens.export.json(grad_fn, path)`.

## Bundle (cross-trace coordination)

### Bundle Members

- `traces`: TraceAccessor mapping Bundle member names to Trace objects.
- `trace_names`: Names of Traces in the Bundle.
- `baseline_name`: Optional name of the designated baseline Trace.
- `baseline`: Baseline Trace object, or `None` when no baseline is set.
- `capacity`: Maximum number of Traces retained by the Bundle; assign to set it.

### Bundle Graph Properties

- `supergraph`: Cross-trace alignment engine — internal aggregation structure used by Bundle's comparison methods. Returns a `Supergraph` (currently in `torchlens/intervention/_topology/`, internal). Advanced / diagnostic use only; most users go through `bundle.compare_at()`, `bundle.diff_pair()`, the Super[T] family, etc.
- `relationships`: Pairwise relationship dictionary for Trace comparisons.
- `relationship(a, b)`: Return the relationship for one pair of Trace names.

### Bundle State Management

- `add(trace_or_traces, names=None)`: Add a Trace OR list of Traces. `names` accepts a single string, list of strings, or None (auto-derived). Single-add and bulk-add use the same method.
- `remove(name_or_names)`: Remove a Trace by string name, OR multiple Traces by list of strings. Returns single Trace or list of Traces correspondingly.
- `remove_except(name_or_names)`: Remove every Trace EXCEPT the named one(s). Accepts string or list of strings.
- `clear()`: Remove all Traces from the Bundle.
- `copy()`: Return a new Bundle with reference-copies of the same Trace objects (shallow; cheap). Same baseline and capacity settings carry over.
- `fork(name=None)`: Return a new Bundle where each member has been independently forked via `Trace.fork()` (deep; expensive). Each member becomes an independent Trace. `name` is an optional prefix for forked member names.

### Bundle Cross-Member Access

The Super* family is universal — every sub-Trace class has a Super
counterpart at Bundle level (Op, Layer, Module, ModuleCall, Param, Buffer,
GradFn, GradFnCall).

- `ops`: `SuperOpAccessor`; `bundle.ops[label]` returns a SuperOp.
- `layers`: `SuperLayerAccessor`; `bundle.layers[label]` returns a SuperLayer.
- `modules`: `SuperModuleAccessor`; returns SuperModule.
- `module_calls`: `SuperModuleCallAccessor`.
- `params`: `SuperParamAccessor`.
- `buffers`: `SuperBufferAccessor`.
- `grad_fns`: `SuperGradFnAccessor`.
- `grad_fn_calls`: `SuperGradFnCallAccessor`.
- `at`: Universal Super[X] accessor. Supports BOTH `bundle.at["label"]` (square brackets, canonical, matches TorchLens/pandas idiom) and `bundle.at("label")` (round brackets, method-style alias). Resolution order across Super[X] types: SuperOp → SuperModuleCall → SuperLayer → SuperModule → SuperParam → SuperBuffer → SuperGradFn → SuperGradFnCall. Returns the matching Super[X] view; raises `KeyError` with enumeration if no match. Parallels `trace[X]` universal lookup but returns Super[X] versions.
- `compare_at(label)`: Compare all Bundle members at one label.
- `diff_pair(a, b)`: Rank label-level differences between two named Traces.
- `most_changed(baseline, ...)`: Return labels with the largest divergence from a baseline.
- `cluster(...)`: Placeholder clustering entry point; details remain implementation-specific.

### Bundle Metric Operations

- `apply(fn)`: Apply a callable to each Trace and return a name-keyed result dict.
- `save(path, level, overwrite)`: Save all Bundle members in unified TorchLens format.
- `do(*args, **kwargs)`: Apply intervention-style operations across members.
- `fork(name)`: Duplicate the Bundle.
- `attach_hooks(*args, **kwargs)`: Attach hooks across member Traces.
- `replay(**kwargs)`: Replay all member Traces.
- `rerun(model, x, **kwargs)`: Rerun all member Traces.
- `draw(...)`: Draw forward graphs for Bundle member Traces.
- `help()`: Return or print a readiness summary for Bundle operations.

### TraceAccessor

- `TraceAccessor`: Dict-like accessor for Bundle Traces by name or ordinal position.
- `TraceAccessor.__getitem__(key)`: Return a Trace for a name or integer index.
- `TraceAccessor.__contains__(key)`: Test whether a Trace name exists.
- `TraceAccessor.__iter__()`: Iterate over Trace records in insertion order.
- `TraceAccessor.keys()`: Return Trace names.
- `TraceAccessor.values()`: Return Trace objects.
- `TraceAccessor.items()`: Return `(name, trace)` pairs.
- `TraceAccessor.to_pandas()`: Export member Trace summaries to a DataFrame.

### Super* family (cross-member views)

The Super* identification surface is uniform across all 8 sub-Trace classes.

- `label`: Canonical label represented by this Super* object (one string; same for every member by alignment construction).
- `members`: Accessor of Trace name → that Trace's record at this label. Supports both **0-based positional integer indexing** (`super_layer.members[0]` → first member's record) AND **string lookup** (`super_layer.members["baseline"]` → that trace's record). Plus dict-like `.keys()` / `.values()` / `.items()` and iteration (yields trace names in Bundle order).
- `traces`: Set of Trace names represented (= Bundle members WHERE this label resolves to a record). Equivalent to `members.keys()`.
- `absent_traces` (`@property`): Set of Bundle Trace names where this label does NOT resolve to a record (= Bundle members - `traces`). Empty when the Super[X] has complete coverage.
- `num_traces` (`@property`): `len(traces)` — count of members represented.
- `num_absent_traces` (`@property`): `len(absent_traces)` — count of members missing this label.
- `is_complete_coverage` (`@property`): True when `num_absent_traces == 0` (every Bundle member has this label).
- `coverage`: Fraction of Bundle members that contain this label (`= num_traces / (num_traces + num_absent_traces)`).
- `__getitem__(key)`: Convenience direct member access at the Super[X] level. Accepts STRING (Trace name) OR INT (0-based positional). Same as `super_layer.members[key]`. Negative indexing works.
- `__len__()`: Number of Bundle members represented (= `num_traces`).
- `__iter__()`: Iterates Trace names in Bundle order.

Tensor-typed Super* views (SuperOp, SuperLayer) additionally expose:

- `out`: Stacked or single output tensor when all members are compatible (shape, dtype, and structure match). Returns `None` when incompatible across members.
- `grad`: Stacked or single gradient when all members are compatible. Returns `None` when incompatible.
- `shape`: Common output shape when compatible across members. Returns `None` when shapes differ.
- `type`: Normalized type token when consistent across members.
- `module`: Module context when comparable across members.

Per-member tensor access goes through `super_X.members[name].out` / `.grad` / etc. (Accessor supports 0-based int and trace-name string lookups). No collection-of-tensors plural fields like `outs` / `grads` — that naming would collide with ModuleCall.outs (container-axis plural for multi-output `forward()`) which has different semantics (list of tensors, not dict of per-trace tensors). Comprehension is the explicit form:

```python
# Cross-member tensor dict (if needed):
{name: m.out for name, m in super_layer.members.items()}
```

Each Super* class additionally has:

- `diff(metric=...)`: Pairwise or baseline-relative differences.
- `aggregate(fn)`: Apply an aggregation over member tensors.

Note: The per-member `labels` dict has been REMOVED. Bundle alignment is
by label; all members share the same label by construction.

## Cross-class accessor patterns

### Universal accessor: `trace[key]`

`trace[key]` is a universal lookup that resolves to any TorchLens object by label, with a deterministic resolution order. Use it for casual / REPL / exploratory lookups; use the explicit accessors (`trace.layers`, `trace.modules`, etc.) when the return type matters for downstream code.

**String key — resolution order (returns first match, raises `KeyError` if no match):**

1. `trace.ops` — Op label (stylized `<type>_<idx>_<idx>:<pass>` format)
2. `trace.module_calls` — ModuleCall label (`<address>:<callidx>`)
3. `trace.layers` — bare Layer label
4. `trace.modules` — bare Module address (dotted PyTorch path)
5. `trace.params` — Param address (dotted, under module)
6. `trace.buffers` — Buffer address (dotted, under module)
7. `trace.grad_fns` — GradFn label
8. Alternate lookup keys (e.g., `op.fx_label` registrations, custom aliases)

Failed lookup raises with all searched namespaces enumerated and fuzzy suggestions where available.

**Integer key:** `trace[N]` returns the Op at `ordinal_index == N` (0-based; round-trip via `trace[op.ordinal_index] is op`). Numeric indexing is Op-specific by convention; for other classes use the explicit accessor (`trace.modules[N]`, `trace.params[N]`, etc.).

### Examples

- `trace["conv2d_1_2"]` → Layer (matches `trace.layers`)
- `trace["conv2d_1_2:1"]` → Op (matches `trace.ops`; colon disambiguates)
- `trace["encoder.block.0"]` → Module (matches `trace.modules`)
- `trace["encoder.block.0:1"]` → ModuleCall (matches `trace.module_calls`)
- `trace["encoder.block.0.weight"]` → Param (matches `trace.params`)
- `trace["bn.running_mean"]` → Buffer (matches `trace.buffers`)
- `trace[42]` → Op at `ordinal_index == 42`
- `trace.layers["conv2d_1_2"]` → Layer (explicit; same as universal lookup)
- `trace.ops["conv2d_1_2:1"]` → Op (explicit)
- `trace.modules["encoder.block.0"]` → Module (explicit)
- `bundle["baseline"]` → Trace (delegates to `bundle.traces`)
- `bundle.layers["conv2d_1_2"]` → SuperLayer
- `bundle.ops["conv2d_1_2:1"]` → SuperOp
- `bundle.at("conv2d_1_2")` → SuperLayer (no colon → Layer-namespace dispatch)
- `bundle.at("conv2d_1_2:1")` → SuperOp (colon → Op-namespace dispatch)

### Rules

- Single-Op passthrough rule: Layer fields such as `out` and `grad` work directly only when the Layer has one Op.
- Multi-Op rule: use `layer.ops[n].field` or a pass-qualified label when a Layer has multiple Ops.
- `bare` vs `total_*` rule: bare = per-pass (raises for multi-pass); `total_*` = sum across all Ops in the Layer.
- Module aggregate rule: Module fields describe all calls unless explicitly documented as single-call passthrough.
- ModuleCall rule: ModuleCall fields describe one invocation only.
- GradFn aggregate rule: GradFn fields describe one autograd node; GradFnCall fields describe each hook firing.
- **Type-strict explicit accessors:** `trace.layers[...]` ALWAYS returns Layer; `trace.ops[...]` ALWAYS returns Op (with single-pass passthrough for bare Layer labels when unambiguous). The previous "Layer-accessor returns Op for convenience" behavior is removed in favor of consistent return types. Use the universal `trace[...]` for cross-type lookup.
- **Integer indexing convention (UNIFORMLY 0-based across all accessors):** Every Accessor — both Trace-scope (`trace.ops`, `trace.layers`, `trace.modules`, etc.) AND scoped (`layer.ops`, `module.calls`, `grad_fn.calls`) — uses 0-based positional integer indexing, matching Python list/sequence idiom. `layer.ops[0]` is the first pass; `layer.ops[-1]` is the last. Negative indexing works. `len(accessor)` returns the count. Iteration goes in pass/call/ordinal order. The 1-based pass_index / call_index / type_index FIELDS on Op / ModuleCall / etc. are SEPARATE semantic values used in label formats (`conv2d_1_5:2` means type-index 1, step 5, pass 2). Don't confuse the 1-based label-component values with the 0-based positional indexing of Accessors.
- **Label-based accessor lookup accepts BOTH short and long Layer-label forms.** `layer.ops["conv2d_2:1"]` and `layer.ops["conv2d_2_3:1"]` resolve to the same Op when both refer to the same Op (i.e., the short and long forms are equivalent for that Op). The short form omits the `step_index` middle component; both label forms are valid lookup keys.

Rules:

- Single-Op passthrough rule: Layer fields such as `out` and `grad` work directly only when the Layer has one Op.
- Multi-Op rule: use `layer.ops[n].field` or a pass-qualified label when a Layer has multiple Ops.
- `bare` vs `total_*` rule: bare = per-pass (raises for multi-pass); `total_*` = sum across all Ops in the Layer.
- Module aggregate rule: Module fields describe all calls unless explicitly documented as single-call passthrough.
- ModuleCall rule: ModuleCall fields describe one invocation only.
- GradFn aggregate rule: GradFn fields describe one autograd node; GradFnCall fields describe each hook firing.

## Function and method signatures

This section lists the signatures of the main public callables. Parameter
names follow the locked post-rename naming.

### Top-level capture

- `tl.trace(model, input_args, input_kwargs=None, *, backend: BackendName | None = None, layers_to_save="all", transform=None, output_transform=None, save_raw_input="small", save_raw_output="small", keep_orphans=False, output_device="cpu", out_transform=None, grad_transform=None, save_raw_activations=True, save_raw_gradients=True, mark_layer_depths=False, detach_saved_activations=True, save_arg_values=True, save_arg_templates=False, save_gradients=False, gradients_to_save=None, save_code_context=True, save_rng_states=False, save_outs_to=None, keep_outs_in_memory=True, save_grads_to=None, keep_grads_in_memory=True, intervention_ready=False, backward_ready=False, hooks=None, unwrap_when_done=False, verbose=False, source_context_lines=2, compute_input_output_distances=True, recurrence_detection=True, capture=None, save=None, streaming=None, trace_label=None, cache=False, cache_dir=None, module_filter=None, stop_after=None, raise_on_nan=False, ...)`: Resolve the backend, run a forward pass with capture, and return a Trace. `backend=None` preserves torch eager default and MLX module auto-routing. **Visualization params (`vis_*`) have been REMOVED from `tl.trace()`** — use `trace.draw(...)` for visualization.
- `tl.peek(model, x, layer, stop_after=None)`: Return the saved out for one layer. Convenience over `trace()`.
- `tl.extract(model, x, layers)`: Return saved outs for many layers. `layers` is an iterable of lookups or a `{user_label: lookup}` mapping.
- `tl.batched_extract(model, stimuli, layers, batch_size=32, device=None, output_dir=None, postfunc=None, progress=True)`: Extract outs from a batched stimulus set.
- `tl.fastlog.record(model, input_args, input_kwargs=None, *, keep_op=None, keep_module=None, default_op=MISSING, default_module=MISSING, history_size=8, include_source_events=False, max_predicate_failures=32, on_predicate_error="auto", streaming=None, return_output=False, postprocess="none", random_seed=None, out_transform=None, save_raw_activations=True, backward_ready=False)`: Torch-only predicate-driven sparse capture, returns a `Recording`.
- `tl.fastlog.halt(reason="")`: Raise `HaltSignal` to halt the active fastlog recording. Returns `NoReturn`.
- `tl.fastlog.HaltSignal(BaseException)`: Signal class raised by `halt`.
- `tl.fastlog.Recorder.__enter__() -> Recorder`: Begin a fastlog rollout context.
- `Recording.log_backward(loss, *, keep_grad=None, default_grad=None, retain_graph=None, create_graph=False)`: Run backward on the captured forward.

### Save, load, validation, bundle construction

- `tl.save(trace, path, *, level="portable", include_outs=True, include_grads=True, include_saved_args=False, include_rng_states=False, strict=True, overwrite=False)`: Persist a Trace to a `.tlspec` directory.
- `tl.load(path, **kwargs)`: Load a `.tlspec` Trace or Bundle. (Canonical entry point — `Trace.load` classmethod has been REMOVED to avoid the classmethod-on-instance footgun.)
- `tl.validate(model, input_args, input_kwargs=None, *, scope, random_seed=None, verbose=False, validate_metadata=True, loss_fn=None, perturb_saved_grads=False, atol=1e-5, rtol=1e-4, validate_layer_grads=False, layer_grad_atol=None, layer_grad_rtol=None, backend: BackendName | None = None)`: Resolve the backend and run backend-dispatched validation. `scope` is `"forward"`, `"backward"`, `"saved"`, or `"intervention"`; the public contract remains a real bool for supported backend/scope pairs and a typed unsupported error otherwise.
- `tl.aggregate(model, dataloader, metrics, *, target="out", loss_fn=None)`: Stream outs (or grads) through metric accumulators.
- `tl.bundle(*args, **kwargs)`: Construct a Bundle.

### Selectors

- `tl.label(name)`: Selector matching an exact Layer or Op label.
- `tl.func(name, *, output=None)`: Selector matching a function name token.
- `tl.output(target)`: Selector matching by output index or semantic output role.
- `tl.module(address)`: Selector matching a dotted module address.
- `tl.contains(substring)`: Selector matching any label containing the substring.
- `tl.where(predicate, *, name_hint=None)`: Selector wrapping a custom predicate.
- `tl.in_module(address_or_layer, address=None)`: Selector restricted to a module address, or bool check.
- `tl.sites(layer_pattern, ops=None, modes=None)`: Build a structured `SiteCollection`.

### Backward selectors

- `tl.grad_fn(type=None, *, label=None, is_custom=None)`: Match grad_fns by class name, label substring, or custom-autograd flag.
- `tl.grad_fn_label(name)`: Match a grad_fn by its exact stable label.

Note: A backward selector matching grad_fns without a forward Op now uses
the `has_op` predicate (was `is_intervening`). The naming review for the
top-level helper is deferred.

### Intervention helpers (all return `HelperSpec`)

- `tl.zero_ablate(*, force_shape_change=False)`: Replace value with zeros.
- `tl.mean_ablate(source=None, *, over="self", force_shape_change=False)`: Replace value with a mean over `source`.
- `tl.resample_ablate(source=None, *, from_=None, seed=None, force_shape_change=False)`: Replace value with a resample.
- `tl.steer(direction, magnitude=1.0, *, coef=None, feature_axis=None, force_shape_change=False)`: Add a steering vector.
- `tl.scale(factor, *, force_shape_change=False)`: Multiply by a scalar.
- `tl.clamp(*, min=None, max=None, force_shape_change=False)`: Clamp values to a range.
- `tl.noise(std, *, seed=None, force_shape_change=False)`: Add Gaussian noise.
- `tl.project_onto(direction, *, feature_axis=None, force_shape_change=False)`: Project onto a direction.
- `tl.project_off(direction, *, feature_axis=None, force_shape_change=False)`: Project orthogonal to a direction.
- `tl.swap_with(other_label, *, force_shape_change=False)`: Swap activation with another site's saved value.
- `tl.splice_module(module, *, input="out", output="out", force_shape_change=False)`: Splice an `nn.Module` into the forward pass.

### Backward intervention helpers (all return `HelperSpec` with `kind="backward"`)

- `tl.bwd_hook(fn)`: Wrap a backward callback as a HelperSpec.
- `tl.grad_zero(*, force_shape_change=False)`: Replace a backward gradient tensor with zeros.
- `tl.grad_scale(factor, *, force_shape_change=False)`: Multiply a backward gradient tensor by `factor`.
- `tl.grad_clip(max_norm, norm_type=2.0)`: Per-tensor norm clipping over a grad_input tuple.
- `tl.grad_noise(std, *, seed=None)`: Add Gaussian noise to each tensor in a grad_input tuple.
- `tl.grad_clamp(min=None, max=None)`: Elementwise clamp on each tensor in a grad_input tuple.

### Errors raised by intervention helpers and selectors

- `HelperMountError(HookSiteCoverageError)`: Raised when a helper is mounted on an incompatible selector universe.
- `HookSignatureError(ConfigurationError, TypeError)`: Raised when a hook callable does not accept the required signature.
- `HookValueError(InterventionError, ValueError)`: Raised when a hook returns an invalid replacement value.
- `HookSiteCoverageError(SiteResolutionError)`: Raised when hook normalization cannot associate a hook with any site.
- `SelectorCompositionError(SiteResolutionError)`: Raised when forward and backward selectors are composed.
- `UnclassifiedSelectorError(SiteResolutionError)`: Raised when a selector lacks an explicit direction taxonomy bucket.
- `AxisAmbiguityError(ConfigurationError, ValueError)`: Raised when a helper cannot infer a feature axis safely.
- `SpliceModuleDtypeError(CompatibilityError, RuntimeError)`: Raised when `splice_module` returns an unexpected dtype.
- `SpliceModuleDeviceError(CompatibilityError, RuntimeError)`: Raised when `splice_module` returns a tensor on an unexpected device.
- `MultiOutputModuleError(ValidationError, ValueError)`: Raised by singular-output access on a multi-output ModuleCall.
- `AppendMismatchError(ValidationError, ValueError)`: Raised when a chunked append candidate is incompatible with the base log.
- `AppendStreamingNotSupportedError(ValidationError, ValueError)`: Raised when append rerun would mutate active streamed activation blobs.
- `AppendBatchDependenceError(ValidationError, ValueError)`: Raised when append cannot prove helper or grad batch independence.
- `AppendStateValidationWarning(TorchLensInterventionWarning)`: Warning when validators skip fresh checks on stacked appended traces.
- `BatchNormTrainModeWarning(TorchLensInterventionWarning)`: Warning for append reruns through batch-sensitive train-mode modules.

### Standalone intervention verbs

- `tl.do(log, hooks_or_site, value_or_hook=None, *, model=None, x=None, engine=MISSING, confirm_mutation=MISSING, strict=MISSING, intervention=None)`: One-shot intervention call.
- `tl.replay(log, *, strict=MISSING, hooks=MISSING, replay=None)`: Replay saved values.
- `tl.replay_from(log, site, *, strict=MISSING, replay=None)`: Replay starting from a site.
- `tl.rerun(log, model, x=None, *, append=MISSING, strict=MISSING, replay=None, output_transform=None)`: Re-execute the model.

### Observers

- `tl.tap(site, *, direction="forward")`: Create a `TapObserver` for a site.
- `tl.record_span(name, *, direction="both")`: Context manager creating a named observer span.
- `TapObserver.records: list[TapRecord]`: Captured observations.
- `TapObserver.values() -> list[torch.Tensor]`: Detached out snapshots in observation order.
- `TapObserver.record_backward(grad_input, *, grad_output, grad_fn, call_index, run_ctx)`: Backward callback signature. The `grad_fn` parameter is the TL GradFn record (post-swap; was `grad_fn_log`).
- `TapRecord(value, site_label, span_names, timestamp, direction, grad_kind=None, backward_call_index=None)`: One observed tensor value.

### Visualization helpers

All visualization lives on `Trace.draw()` (and friends), not on `tl.trace()`.

- `tl.viz.heatmap(max_size=200)`: Returns a tensor-to-`PIL.Image` callable for `layer_visualizers`.
- `tl.viz.channel_grid(n=16, max_size=300)`: Returns a per-channel grid visualizer.
- `tl.viz.histogram(bins=30, width=240, height=160)`: Returns a histogram visualizer.
- `tl.show_bundle_graph(bundle, ...)`: Draw a Bundle graph.
- `tl.draw_backward(trace, ...)`: Top-level wrapper for `Trace.draw_backward`.
- `tl.draw_combined(trace, ...)`: Top-level wrapper for `Trace.draw_combined`.

### Bridges

- `tl.bridge.hf.trace_text(model, text, *, tokenizer=None, chat_template=False, **kwargs)`: Trace a Hugging Face model with raw text input.

### Trace methods

- `Trace.backward(loss, **backward_kwargs)`: Run backward from a loss and populate grad fields.
- `Trace.find_layers(query, *, limit=10)`: Return Layer labels matching a query.
- `Trace.find_sites(query, *, strict=False, max_fanout=8)`: Return intervention sites. (Deferred for naming review.)
- `Trace.fork(name=None)`: Duplicate the Trace with a fresh intervention spec.
- `Trace.set(site, value, *, strict=False, confirm_mutation=False)`: Set a site value.
- `Trace.attach_hooks(hooks_or_site, hook=None, *extra_hooks, strict=False, prepend=False, confirm_mutation=False)`: Attach hooks to one or many sites.
- `Trace.do(hooks_or_site, value_or_hook=None, *, model=None, x=None, engine=MISSING, confirm_mutation=MISSING, strict=MISSING, intervention=None)`: One-shot intervention application.
- `Trace.replay(*, strict=MISSING, hooks=MISSING, replay=None)`: Replay saved values.
- `Trace.replay_from(site, *, strict=MISSING, replay=None)`: Replay starting from a site.
- `Trace.rerun(model=None, x=None, *, append=MISSING, strict=MISSING, replay=None, transform=USE_STORED, output_transform=USE_STORED)`: Re-execute and update or append Trace state.
- `Trace.draw(**kwargs)`: Draw the forward graph. All vis params live here.
  `order_siblings=True` orders true parallel sibling fanouts in execution order for
  forward unrolled Graphviz/dot renders after local edge-stretch verification.
- `Trace.draw_backward(**kwargs)`: Draw the backward grad-fn graph.
- `Trace.draw_combined(**kwargs)`: Render forward + backward in a single graph.
- `Trace.log_backward(loss, **backward_kwargs)`: Backward capture implementation method.
- `Trace.replace_state_from(new_log)`: Atomically replace this Trace's run state from a freshly-built Trace.
- `Trace.append_state_from(new_log)`: Merge compatible chunk outs from `new_log` into this Trace.
- `Trace.preview_fastlog(predicate=None, keep_op=None, keep_module=None, **kwargs)`: Render a fastlog predicate preview.
- `Trace.resolve_sites(query, *, strict=False, max_fanout=8)`: Resolve intervention sites. (Deferred.)
- `Trace.summary(level="overview", *, fields=None, mode="auto", show_ops=False, preset=None, columns=None, include_ops=None, max_rows=200, print_to=None, count_fma_as_two=False)`: Return a textual summary.
- `Trace.save(path, **kwargs)`: Save the Trace; forwards to `tl.save`.
- `Trace.cleanup()`: Clear circular references and runtime-only heavyweight objects.
- `Trace.to_pandas()`: Build the Trace tabular DataFrame. Use `torchlens.export.csv` / `torchlens.export.parquet` / `torchlens.export.json` for file export.

### Bundle methods

- `Bundle(traces=None, *, baseline=None, capacity=None, ...)`: Construct a Bundle.
- `Bundle.add(trace, name=None)`: Add a Trace.
- `Bundle.remove(name)`: Remove a Trace by name; returns the removed Trace.
- `Bundle.remove_except(keep)`: Remove every Trace except the named Traces.
- `Bundle.clear()`: Remove all Traces.
- `Bundle.fork(name=None)`: Duplicate the Bundle.
- `Bundle.do(*args, **kwargs)`: Apply intervention-style operations across members.
- `Bundle.attach_hooks(*args, **kwargs)`: Attach hooks across member Traces.
- `Bundle.replay(**kwargs)`: Replay all member Traces.
- `Bundle.rerun(model, x=None, **kwargs)`: Rerun all member Traces.
- `Bundle.save(path, *, level="portable", overwrite=False)`: Save all members.
- `Bundle.apply(fn)`: Call `fn(trace)` for each member.
- `Bundle.at(label)`: Resolve a label to the matching Super accessor.
- `Bundle.compare_at(site)`: Stack member tensors at one site.
- `Bundle.diff_pair(a, b=None)`: Out differences between two members or sites.
- `Bundle.most_changed(baseline=None, *, top_k=10, metric="cosine")`: Rank sites by distance.
- `Bundle.cluster(*args, **kwargs)`: Placeholder; raises `NotImplementedError`.
- `Bundle.relationship(a, b)`: Return the recorded relationship between two named Traces.
- `Bundle.help()`: Return or print a per-member readiness summary.

### Notes on signatures

- `MISSING` (and `_USE_STORED_TRANSFORM` for `Trace.rerun`) are sentinels meaning "use the captured default"; users pass concrete values to override.
- `to_pandas()` is uniform across record classes and accessors; file export is centralized under `torchlens.export.*`.

## Validation reports

- `LayerGradReport`: Dataclass returned by `validate_backward_pass(..., validate_layer_grads=True)` summarising per-module-output gradient parity. Fields: `mode` (`Literal["module_output"]`), `overall_passed: bool`, `coverage: dict[str, str]` (module-call label -> bucket), `covered_count`, `mismatched_count`, `skipped_no_first_leaf_count`, `skipped_module_less_count`, `skipped_no_grad_count`, `skipped_identity_output_count`, `skipped_root_module_count`, `unexpected_count`, `candidate_grad_count`, `atol`, `rtol`, `mismatched_labels: tuple[str, ...]`, `max_abs_diffs: dict[str, float]`, `max_rel_diffs: dict[str, float]`. `__bool__` returns `overall_passed`.
- Coverage buckets:
  - `covered`: candidate grad matched stock grad within tolerance.
  - `mismatched`: candidate grad differed from stock or shapes disagreed.
  - `skipped_no_first_leaf`: call has no resolvable first output leaf layer.
  - `skipped_module_less`: candidate layer has a grad but no module containment.
  - `skipped_no_grad`: candidate or stock side missing a grad for this module-call.
  - `skipped_identity_output`: stock module output is identity-equivalent to its input; skipped to avoid double-counting.
  - `skipped_root_module`: the top-level model (address `"self"`); skipped because it has no enclosing module.
- `MIN_MODULE_OUTPUT_COVERAGE: float = 0.80`: Minimum covered ratio required for `overall_passed`.

## Backend integration

- `BackendName`: Literal/string backend identifier accepted by `backend=` and persisted as
  `Trace.backend`; shipped user-facing names are `"torch"` and technical-preview `"mlx"`.
- `BackendSpec`: Public registry object owning backend detection, capture dispatch, validation
  dispatch, replay hooks, serialization policy, capabilities, and canonical errors.
- `register_backend_spec(spec, *, replace=False)`: Register a process-local `BackendSpec`.
- `unregister_backend_spec(name)`: Remove a process-local backend spec by name or alias.
- `get_backend_spec(name)`: Return a registered `BackendSpec` or raise a typed unknown-backend
  error.
- `registered_backend_specs()`: Return the unique registered backend specs.
- `resolve_backend_spec(backend, model, input_args, input_kwargs)`: Resolve the public backend
  selection; explicit names win, `backend=None` falls back to detector priority, and ambiguity or
  mismatch raises before capture.
- `BackendUnsupportedError`: Raised when a registered backend lacks the requested capability.
- `BackendMismatchError`: Raised when an explicit backend cannot handle the supplied model/input.
- `BackendAmbiguityError`: Raised when `backend=None` has multiple equal-priority matches.
- `BackendPayloadUnsupportedError`: Raised when audit-only or metadata-only backend payloads are
  requested after load.
- `BackendRuntimeCompatibilityError`: Raised when serialized backend runtime metadata is
  incompatible with the current backend runtime.
- MLX wrapped surface: Conv2d, normalization layers, Embedding, Dropout, MultiHeadAttention,
  reductions, shape ops, and activations. Unsupported MLX capture surfaces now raise canonical
  backend capability errors.

## Portable save scrub

- `_io.scrub.scrub_for_save(trace, *, include_outs=True, include_grads=True, include_saved_args=False, include_rng_states=False) -> tuple[dict, list[BlobSpec], list[dict]]`: Scrub a Trace into portable metadata, tensor blob specs, and an unsupported-tensor audit list.
- `backend_runtime`: Manifest schema-v2 runtime fingerprint for the backend that wrote a bundle.
- `payload_policy`: Manifest schema-v2 payload materialization policy, currently `full`,
  `audit_only`, or `metadata_only`.
- TLSPEC version axes: the on-disk family detector (`kind` + `tlspec_version`), the manifest
  schema version, and the pickled object-state `_io.TLSPEC_VERSION` are independent.
- Manifest schema v2: backend-aware manifest shape adding `backend`, `backend_runtime`, nullable
  torch-specific fields, and `payload_policy`; non-torch preview payloads are audit-only or
  metadata-only until a backend codec can materialize them.

## Removed (relative to the v1 glossary)

These items have been removed entirely as part of the rename sprint:

- `OpLog.compute_index`, `Op.compute_index`: merged into `step_index`.
- `OpLog.trace_index`, `Op.trace_index`: renamed to `step_index` (intermediate name during walkthrough).
- `Op.overall_index`: renamed to `step_index` (intermediate name).
- `Op.creation_index`, `Op.capture_index`: renamed to `raw_index`.
- `Op.buffer_address`: renamed to `address` (buffer-sourced ops).
- `Op.output_versions_per_child`: renamed to `out_versions_by_child`.
- `Op.is_part_of_iterable_output`: renamed to `in_multi_output` (now `@property`).
- `Op.multi_output_role`: renamed to `multi_output_name`.
- `Op.is_submodule_input`, `Op.is_submodule_output`: renamed to `is_module_input`, `is_module_output`.
- `Op.module_ops_entered`: renamed to `module_calls_entered`.
- `Op.module_entry_argnames`: renamed to `module_entry_arg_keys`.
- `Op.has_saved_outs`, `Layer.has_saved_outs`: renamed to `has_saved_activation`.
- `Op.memory`: renamed to `activation_memory`.
- `Op.grad_memory`: renamed to `gradient_memory`.
- `Op.transformed_out_memory`, etc.: renamed to `transformed_activation_memory`.
- `Op.grad_fn_name`: renamed to `grad_fn_class_name`.
- `Op.feeds_output`: renamed to `is_output_parent`.
- `Op.save_tensor_data(...)`: renamed to `save_activation(...)`.
- `Layer.buffer_address`: renamed to `address`.
- `Layer.unsupported_op`, `Trace.unsupported_ops`: REMOVED as vestigial (always empty in current code).
- `Trace.run_state`: renamed to `state`. `RunState` enum renamed to `TraceState`.
- `Trace.train_mode`: renamed to `backward_ready`.
- `Module.is_train_mode`: renamed to `training` (PyTorch idiom match).
- `Trace.ledger`, `Trace.operation_history`: renamed to `state_history`.
- `Trace.io_format_version`: renamed to `tlspec_version`.
- `Trace.input_id`, `Trace.model_id`: renamed to `input_object_id`, `model_object_id`.
- `Trace.input_shape_hash`: renamed to `input_signature_hash` (covers shape + dtype + device).
- `Trace.trace_annotations`: renamed to `annotations`.
- `Trace.flops_by_type`, `Trace.macs_by_type`: renamed to `flops_by_op_type`, `macs_by_op_type`.
- `Trace.total_out_memory`: renamed to `total_activation_memory`.
- `Trace.saved_out_memory`: renamed to `saved_activation_memory`.
- `Trace.param_memory`: renamed to `total_param_memory`.
- `Trace.autograd_saved_memory`: renamed to `total_autograd_saved_memory`.
- `Trace.duration` / `Trace.total_duration`: renamed to `capture_duration`.
- `Trace.function_calls_duration`: renamed to `func_calls_duration`.
- `Trace.detect_loops`: renamed to `recurrence_detection`.
- `Trace.capture_full_args`: renamed to `save_arg_templates`.
- `Trace.save_function_args`: renamed to `save_arg_values`.
- `Trace.differentiable`: renamed to `backward_ready`.
- `Trace.internally_initialized_ops`, `Trace.internally_terminated_ops`: renamed to `internal_source_ops`, `internal_sink_ops`.
- `Trace.model_source_*`: renamed to `class_source_*` (matches Module pattern; `model_` prefix is redundant).
- `Trace.forward_lineno`: renamed to `forward_source_line`.
- `Trace.model_name` (overloaded): split into `model_class_name` + `model_label`.
- `Trace.model_class`: renamed to `model_class_qualname`.
- `Trace.name`: renamed to `trace_label`.
- `Trace.ops_with_saved_outs`: renamed to `ops_with_saved_activations`, then replaced by the `saved_ops` Accessor.
- `Trace.ops_with_saved_grads`: replaced by the `saved_grad_ops` Accessor.
- `Trace.num_intervening_grad_fns`: renamed to `num_grad_fns_without_op`.
- `Trace.unlogged_ops` / `Trace.dropped_ops`: REMOVED with the `keep_unsaved_layers=False` feature removal.
- `Trace.internally_terminated_bool_ops`: deferred (intersects with conditional bool axis).
- `Trace.load(path)` classmethod: REMOVED (use `tl.load(path)`).
- `tl.trace()` vis params: ALL REMOVED (use `trace.draw()` instead). ~22 params moved.
- `keep_unsaved_layers=False` mode: REMOVED entirely (use `saved_*` Accessors instead).
- `Trace.conditional_records`, `conditional_arm_entry_edges`, `conditional_*_entry_edges` (6 fields): folded into `conditionals` Accessor.
- `Op.is_in_conditional_body` (stored), `conditional_role_stacks`, `conditional_*_children` (8 fields): folded into `in_conditionals` and `terminal_bool_for`.
- `Layer.unsupported_op`: REMOVED.
- `Buffer.buffer_address`: renamed to `address`.
- `Buffer.buffer_parent`: REMOVED (use `parents`).
- `Buffer.buffer_use_index`: renamed to `buffer_overwrite_index`.
- `Param.linked_params`: renamed to `co_parent_params`.
- `Param.module_class_name`, `Param.module_class_qualname`, `Param.module_type`: REMOVED. Use `param.module.X` instead.
- `GradFn.name`: renamed to `class_name`.
- `GradFn.is_intervening`: storage flip to `has_op` (positive form; storage default inverts).
- `GradFn.grad_fn_label`: renamed to `label` (bare; matches GradFn-as-primary-subject pattern).
- `GradFn.module_path`: flagged for review during rename sprint.
- `GradFn.parent_grad_fn_labels`, `GradFn.child_grad_fn_labels`: renamed to `parents`, `children`.
- `GradFn.overall_index`: renamed to `trace_index`.
- `GradFn.op` (was direct ref): refactored to `op_label` (stored) + `op` (`@property`).
- `Module.is_shared`, `ModuleCall.is_shared`, `Param.is_shared`, `Buffer.is_shared`: renamed to `has_multiple_addresses`.
- `Module.source_file`, `Module.source_line`: renamed to `class_source_file`, `class_source_line` (matches symmetric triplet design).
- `Super[T].node_name`: renamed to `label`.
- `Super[T].labels` dict: REMOVED (alignment is by label; per-member dict was redundant).
- `Trace.detach_saved_tensors` (with typo `detach_saved_tensorss`): renamed to `detach_saved_activations`.
- `OpLog/LayerLog.func_call_stack`: renamed to `code_context`.
- `OpLog.is_atomic_module_output`: renamed to `is_atomic_module`.
- `Bundle.remove_all_but`: renamed to `remove_except`.

## Facets framework (LOCKED 2026-05-27)

**Derived semantic views on Op and Module records, populated by recipes matched against the record's class or via predicates. Both built-in (shipped by TorchLens) and user-defined recipes use the same registry. Every `Op` and `Module` carries a `.facets` field.**

### The `FacetView` object

**`record.facets` returns a `FacetView`. Lazy + cached. Supports dict subscript and attribute access uniformly.**

| Surface | Type | Behavior |
|---|---|---|
| **`view.X`** | (varies per facet) | **Attribute access; triggers recipe on first access; cached.** |
| **`view['X']`** | (varies per facet) | **Dict-style subscript; same lazy behavior.** |
| **`view.keys()`** | **`list[str]`** | **Available facet names; no recipe invocation.** |
| **`view.has(name)`** | **`bool`** | **Existence check; no recipe invocation.** |
| **`list(view)`** | **`list[str]`** | **Iterate names.** |
| **`len(view)`** | **`int`** | **Number of available facets.** |
| **`view.recipe_source`** | **`str | tuple[str, ...] | None`** | **Which recipe(s) populated this view; None when no match.** |
| **`view.invalidate()`** | **`None`** | **Drop cached values; next access recomputes.** |

### Recipe registration: `tl.facets.register`

**Decorator that registers a recipe function against one or more matchers. Recipe receives the record and returns a flat `dict[str, Any]` of facet name → value.**

```python
@tl.facets.register(class_name='DistilBertSdpaAttention')
def distilbert_attention(mod):
    n_heads = mod.cls.n_heads
    d_head = mod.cls.dim // n_heads
    B, S = mod.calls[0].input_shapes[0][:2]
    return {
        'q':  mod.modules['q_lin'].calls[0].out.view(B, S, n_heads, d_head),
        'k':  mod.modules['k_lin'].calls[0].out.view(B, S, n_heads, d_head),
        'v':  mod.modules['v_lin'].calls[0].out.view(B, S, n_heads, d_head),
        'attn_out': mod.calls[0].out,
        'n_heads': n_heads,
        'd_head':  d_head,
    }
```

**Matchers (any combination; ALL specified must pass):**

| Matcher | Type | Example |
|---|---|---|
| **`class_name`** | **`str | tuple[str, ...]`** | **`'GPT2Attention'` or `('LayerNorm', 'RMSNorm')`** |
| **`class_qualname`** | **`str | tuple[str, ...]`** | **`'transformers.models.llama.modeling_llama.LlamaAttention'`** |
| **`predicate`** | **`callable(record) -> bool`** | **`lambda r: isinstance(r, tl.Op) and r.layer_label.startswith('softmax')`** |

**Multi-recipe merge: when multiple recipes match the same record, their dicts merge into the FacetView. Last-registered wins on key conflicts, emitting a `UserWarning` to flag the override. Library + user recipes can both contribute facets to the same class without one nuking the other.**

### Discoverability functions

| Function | Returns | Purpose |
|---|---|---|
| **`tl.facets.list()`** | **`list[FacetRecipe]`** | **All registered recipes.** |
| **`tl.facets.list(scope='module' | 'op')`** | **`list[FacetRecipe]`** | **Filter by record-type scope.** |
| **`tl.facets.list(class_name='*Attention')`** | **`list[FacetRecipe]`** | **Glob match on class name.** |
| **`tl.facets.info(class_name)`** | **`dict`** | **Which recipes match this class and what facets they provide.** |

### Trace-level finders (LOCKED 2026-05-27)

| Method | Returns | Purpose |
|---|---|---|
| **`trace.attention_blocks()`** | **iterator over Modules** | **Modules matched by any built-in attention recipe (shortcut for `modules_with_facet('q')`).** |
| **`trace.modules_with_facet(name)`** | **iterator over Modules** | **Modules whose FacetView contains the named facet.** |

### Attention-specific sub-view: `.head(i)`

**Attention recipes uniformly expose a `.head(i)` method on the FacetView (when the view contains per-head facets) returning a scoped sub-view for one head. `view.head(3).q` is equivalent to `view.q[:, :, 3, :]` with shape `(B, S, d_head)`.**

### Built-in recipes (initial set shipped in v1)

| Recipe target | Facets provided |
|---|---|
| **`DistilBertSdpaAttention`** | **`q`, `k`, `v`, `attn_out`, `input`, `n_heads`, `d_head`, `head(i)`** |
| **`GPT2Attention`** | **`q`, `k`, `v`, `attn_out`, `input`, `n_heads`, `d_head`, `head(i)` (handles fused `c_attn`)** |
| **`BertSelfAttention`** | **`q`, `k`, `v`, `input`, `n_heads`, `d_head`, `head(i)`** |
| **`LlamaAttention`, `LlamaSdpaAttention`** | **`q`, `k`, `v`, `attn_out`, `input`, `n_q_heads`, `n_kv_heads`, `d_head`, `head(i)` (GQA-aware)** |
| **`MistralAttention`, `MistralSdpaAttention`** | **same as Llama (GQA-aware)** |
| **`T5Attention`** | **`q`, `k`, `v`, `attn_out`, `input`, `n_heads`, `d_head`, `head(i)`** |
| **`ViTSelfAttention`** | **`q`, `k`, `v`, `attn_out`, `input`, `n_heads`, `d_head`, `head(i)`** |
| **`LayerNorm`, `RMSNorm`, `LlamaRMSNorm`** | **`normalized`, `gamma`, `beta` (None for RMS variants), `input`** |
| **`LlamaMLP`, `MixtralMLP`, `GPT2MLP`, `DistilBertFFN`** | **`intermediate`, `up_out`, `gated_out` (where applicable), `down_out`, `input`, `output`** |
| **`nn.Embedding`** | **`lookup`, `weight`, `indices` (where extractable)** |

### Fused-SDPA limitation

**When an attention module uses PyTorch's fused SDPA kernel (default in modern HF builds), the post-softmax attention pattern is NOT extractable — it lives inside the C++ fused kernel. The `pattern` facet raises an informative `RuntimeError` directing the user to re-run with `model.config._attn_implementation='eager'` to expose it. Not silent None — explicit error keeps the failure mode discoverable.**

### Serialization

**Facets are derived, not stored. `.tlspec` save drops cached facet values; FacetView reconstructs lazily on load against the receiving session's registry. User-registered recipes must be re-registered in the loading session for their facets to be available. Recommended convention: keep recipe registrations in a Python module that is imported on both sides.**

### Facets — P1 spec-model update (LOCKED 2026-06-05; supersedes the recipe-authoring API above)

The facets framework was rebuilt on a structured spec model (facets sprint P1; dual-lab reviewed). The full
narrative guide is `docs/facets.md`. Canonical new public names:

- **`FacetSpec`**: the data model behind a facet — `(home op reference, structural transform chain)`. Recipes now
  return `FacetSpec`s (via op-anchored helpers like `child_output_spec(module, name, recipe)` + transform builders),
  NOT computed tensors. Read = `transform(home.out)`; grad = `transform(home.grad)`; intervene (P2) = write-back.
- **Transform builders** on a `FacetSpec`: `__getitem__` (slice/index), `.heads(n_heads, d_head)`, `.split(sections, dim)`,
  `.reshape(*shape)`, `.transpose`, `.select`. Each declares capability flags.
- **`FacetCapabilityFlags`** {read, grad, write, portable, reconstructed} and the **capability class** of a spec
  (`bijective_view` | `selection` | `aliasing_selection` | `computed`). Chain capability = INTERSECTION of its
  primitives' flags (`computed` -> read-only; `aliasing_selection`, e.g. GQA K/V -> read+grad, no write; `write`
  needs all primitives bijective/selection). System FAILS CLOSED: only op-anchored structural specs claim grad/write.
- **`Facet`**: the resolved facet object (`.value`, `.grad`, `.spec`). **`facet.grad`** = the home op's gradient
  projected through the transform chain — available ONLY when the home op has a saved gradient (grad capture on:
  `tl.trace(..., backward_ready=True, gradients_to_save=...)` + `log_backward(loss)`); otherwise RETURNS a
  **`MissingGradient`** sentinel (does NOT raise; only tensor-USE raises) carrying the exact recapture instruction.
- **`MissingGradient`**: typed sentinel (parallel to `MissingFacet`). `get`/`in`/iteration uniform across both.
- **Registration scoping (additions):** `tl.facets.register` is additive (non-matching recipes inert; user
  overrides built-ins for the same class) with **specificity ordering** (qualname > class_name > predicate; user >
  built-in), NOT registration order. `tl.trace(..., recipes=[...])` (additive) and `with tl.facets.using(r): ...`
  (contextvar) affect capture-time snapshot construction. Each `Trace` SNAPSHOTS the active recipe set at capture
  (`Trace.facet_registry_snapshot`, `FieldPolicy.DROP` — not serialized) so facets are reproducible/immune to later
  registry mutation. `tl.facets.reset()` restores built-ins; `tl.facets.snapshot()` / `FacetRegistrySnapshot`.
- **Structural output facets:** every output is a facet via the one `.facets` door — named outputs (dict keys /
  NamedTuple / dataclass / `torch.return_types` structseq) by NAME; positional by `out{i}`; single output stays
  `.out`; nested by dotted index (`out1.0`). Canonical TYPED item access (path keys); attribute access is
  best-effort sugar for valid identifiers not colliding with FacetView methods (`keys`/`get`/`head`/`recipe_source`).
  `module.outs` is a thin alias. Capability inventory recorded as data (op_structural / parameter / module_input /
  module_output / computed_read_only / missing); parameter facets (norm gamma/beta, embedding weight) are read-only.

### Facets — P2 intervention update (LOCKED 2026-06-05)

Facet-level intervention is selector sugar over the existing TorchLens hook machinery, not a parallel hook system.
Canonical public names and semantics:

- **`tl.facet(name)`**: selector for intervention on a named facet. Resolves against captured `FacetView`s to the
  facet's HOME op plus transform chain, then stores a normal home-op hook target.
- **`tl.head(index, name=None)`**: selector for one attention head. `tl.head(3, "q")` is equivalent to
  `tl.facet("q").head(3)`; `name=None` targets the default attention projection facets `q`, `k`, and `v`.
- **`scatter_update(home_out, edited_slice, mode)`**: write-back ABI on facet transform specs. It writes the edited
  facet slice back into the FULL home-op output and returns that full edited tensor. `bijective_view` writes through
  exact view transforms; `selection` scatters into selected regions such as split Q/K/V thirds or selected heads.
- **Write capability**: a facet can be edited only when its home is an op raw output and every primitive in the
  chain is scatter-back capable (`bijective_view` or `selection`). Non-op homes, `computed` facets, and non-raw
  `value_version` facets fail closed.
- **Alias policy**: `aliasing_selection` (for example GQA repeated K/V query-head aliases) is read+grad only by
  default. Writes require a future explicit alias policy; P2 refuses and names the alias group.
- **Conflict policy**: same-home facet writes compose in attach order when their boolean write masks do not overlap
  (for example GPT-2 fused `c_attn` edits to different q/k/v regions). Overlapping same-home writes raise before
  rerun.
- **Validation semantics**: an edited home op is an intervention boundary. Validation still checks saved outputs,
  downstream propagation, parent-edge logging, graph shape, and metadata; it does not require the intentionally
  replaced op output to equal the original callable replay.

- NOT in P2 (later phases): attention pattern RECONSTRUCTION (P3); paired-grad_fn input-side gradients (deferred —
  capture discards the input index); per-module-eager reconstruction/fallback.

### Facets — P3 reconstruction and coverage update (LOCKED 2026-06-05)

Canonical new public names and semantics:

- **`reconstruction_ready`**: `tl.trace(..., reconstruction_ready=True)` convenience flag that enables the saved
  argument/RNG prerequisites for read-only reconstructed facets. It is equivalent to opting into the lower-level
  capture needed by fused SDPA reconstruction (`save_arg_values=True`, `save_rng_states=True`).
- **Reconstructed attention facets**: **`scores`**, **`pattern`**, and **`z`** on fused SDPA attention modules are
  read-only computed facets. They use the fused SDPA op's actual saved Q/K/V inputs (post-RoPE when RoPE is present),
  not q_proj/k_proj recipe facets. Validation target is the fused SDPA op output: `z` compares directly; `scores` and
  `pattern` recompute `z = pattern @ V` and compare that target. Missing prerequisites return **`MissingFacet`** with
  a named prerequisite.
- **`result`**: per-head output-projection contribution facet. For an attention output projection `W_O`, each
  `result[..., head, :]` is that head's contribution to the projected residual stream. Summing heads and adding
  projection bias validates against the captured output-projection tensor.
- **`resid_pre` / `resid_mid` / `resid_post`**: transformer-block residual stream facets. Anchors are block input op,
  post-attention residual-add op when identifiable, and block output op. Op-anchored residual facets support
  `facet.grad` under the normal saved-gradient contract.
- **Module-path fallback**: modules with no semantic recipe still expose structural facets through
  `log.modules["path"].facets["out"]` and named output/container facets.
- **`tl.facets.enable_transformerlens_aliases(enabled=True)`**: opt-in process setting that exposes TL-style aliases
  such as `hook_pattern`, `hook_z`, `hook_result`, `hook_resid_pre`, `hook_resid_mid`, and `hook_resid_post` when the
  native TorchLens facet exists. **`tl.facets.transformer_lens_aliases_enabled()`** reports the current setting.
- **Recipe entry-point group**: installed packages may expose recipe registration modules through the
  **`torchlens.recipes`** setuptools entry-point group. Entry-point failures warn and skip; TorchLens does not scan
  or auto-execute local directories.

### Facets — P4 patching and attribution update (LOCKED 2026-06-05)

Canonical new public names and semantics:

- **`tl.facets.patching.activation_patch_residual_stream(model, clean_input, corrupted_input, metric, ...)`**:
  clean-vs-corrupted causal mediation helper for residual stream facets. Patches clean `resid_pre` by default into
  corrupted reruns and returns metric values shaped **`[layer, pos]`**; with `patch_positions=False`, patches each
  full residual tensor and returns **`[layer]`**.
- **`tl.facets.patching.activation_patch_attention_output(...)`**: patches each clean attention-output facet
  (`attn_out` by default) into the corrupted run and returns **`[layer]`**.
- **`tl.facets.patching.activation_patch_attention_heads(...)`**: patches each clean per-head attention-output
  facet (`result` by default) into the corrupted run and returns **`[layer, head]`**.
- **`tl.facets.patching.activation_patch_mlp_output(...)`**: patches each clean MLP output facet (`output` by
  default on modules exposing the built-in MLP facet family) into the corrupted run and returns **`[layer]`**.
- **`tl.facets.patching.attribution_patch_attention_heads(...)`**: fast linear approximation to attention-head
  activation patching. Computes `grad(metric wrt corrupted component) * (clean component - corrupted component)`,
  summed over component dimensions, and returns **`[layer, head]`**. Requires saved facet gradients; missing capture
  raises with the `MissingGradient` recapture instruction.
- **`tl.facet(name).in_module(address)`**: module-scoped facet selector used by patching helpers to target one layer
  while still routing through the standard `fork` / `attach_hooks` / `rerun` facet scatter path.

## Input auto-routing (added v9 2026-05-31 — documents shipped API; no walkthrough lock yet)

`tl.trace(model, x)` auto-routes certain
`(model, input)` combinations to specialized HuggingFace bridge tracers based on
the input type and the model's resolvability. The mechanism is a priority-ordered
registry of detector callables. Auto-routing fires ONLY when the user did not
supply an explicit `transform=` (passing `transform` means the user has taken
manual control of preprocessing, so routing is skipped). The first detector that
returns a non-`None` result wins; if none match, `trace` falls through to ordinary
eager capture on the raw input (the default for plain tensors).

### The registry: `tl.autoroute.input`

`tl.autoroute` is importable but intentionally NOT in the top-level `__all__`
(public but unadvertised). The input registry is `tl.autoroute.input`.

| Surface | Signature | Behavior |
|---|---|---|
| `register` | `register(*, name, priority=100)` -> decorator | Register a detector. `name` must be unique (else `ValueError`); lower `priority` runs earlier. Returns the function unchanged. |
| `unregister` | `unregister(name)` | Remove a detector by name (`KeyError` if absent). |
| `iter_by_priority` | `iter_by_priority()` -> iterator | Yield detector funcs in dispatch order — what `trace` iterates. |
| `list` | `list(name_glob=None)` -> `list[Detector]` | List detectors (optional `fnmatch` filter), in dispatch order. |
| `info` | `info(name)` -> `dict` | Diagnostic metadata for one detector. |
| `snapshot` | `snapshot()` -> context manager | Snapshot/restore registry state (temporary registration, e.g. in tests). |

`Detector` is a frozen dataclass (`name`, `priority`, `registration_order`, `func`).
A detector has signature `(model, payload, **kwargs) -> Trace | None`: return `None`
to decline (the next detector tries) or a `Trace` to claim the dispatch.
`tl.autoroute.output` is reserved — any attribute access raises `NotImplementedError`.

### Built-in detectors

Registered by TorchLens, in dispatch order (lower priority first — text, then
multimodal, then image):

| Detector | priority | Routes to | Fires when |
|---|---|---|---|
| `hf_text` | 10 | `tl.bridge.hf.trace_text` | input is a `str` / list of `str` / chat-message list, AND the model exposes a resolvable `name_or_path` (tokenizer-resolvable) |
| `hf_multimodal` | 20 | `tl.bridge.hf.trace_multimodal` | input is a modality-keyed dict (`text`/`image`/`images`/`audio`/`videos`), AND `AutoProcessor` resolves |
| `hf_image` | 30 | `tl.bridge.hf.trace_image` | input is a PIL image / list of PIL images, AND the model has NO already-attached image processor |

A chat-message-list input (a non-empty list whose first element is a dict with
`role`/`content`) sets `chat_template=True`, applying the tokenizer's chat template
before tokenizing.

### The HuggingFace bridge tracers

| Function | Signature | Behavior |
|---|---|---|
| `tl.bridge.hf.trace_text` | `trace_text(model, text, *, tokenizer=None, chat_template=False, **kwargs)` | Resolves an `AutoTokenizer` (unless given), tokenizes, traces; records tokenizer provenance on `trace.input_preprocessor`. |
| `tl.bridge.hf.trace_image` | `trace_image(model, image, **kwargs)` | Traces an image model from PIL input via a four-tier preprocessing cascade: HF image processor -> torchvision `weights.transforms()` -> timm `default_cfg` -> ImageNet default (resize 256 / center-crop 224 / ImageNet-normalize, with a `UserWarning`). |
| `tl.bridge.hf.trace_multimodal` | `trace_multimodal(model, input_dict, **kwargs)` | Resolves `AutoProcessor.from_pretrained(name_or_path)`, applies it, traces. Raises `ValueError` if no `name_or_path` resolves. |

All three set `trace.input_preprocessor` to a `ResolvedPreprocessing` provenance
record (`source` in `hf_auto_tokenizer`, `hf_auto_processor`,
`hf_auto_image_processor`, `torchvision_weights`, timm, imagenet-default).

### Registering a custom detector

```python
import torchlens as tl

@tl.autoroute.input.register(name="my_router", priority=15)
def my_router(model, payload, **kwargs):
    if not _matches(payload):
        return None                       # decline -> next detector tries
    return tl.trace(model, _preprocess(payload), transform=..., **kwargs)  # claim
```

Default `priority=100` places custom detectors after the built-ins (10/20/30).
Inspect/manage with `tl.autoroute.input.list()`, `.info(name)`, `.unregister(name)`,
and `.snapshot()` (a context manager for temporary registration).

## Deferred items

- `_edge_uses`: deferred until EdgeUseRecord's public role is decided.
- Rolled visualization properties (`children_per_op`, `parents_per_op`, `child_ops_per_layer`, `parent_ops_per_layer`, `edges_vary_across_ops`, `atomic_module_ops`, `parent_arg_positions`): deferred to a report/visualization namespace move.
- `buffer_num_passes`: deferred at Trace scope; likely follows Buffer vocabulary as `buffer_use_counts`.
- `recording_kept`: deferred because `recording` conflicts with fastlog `Recording` vocabulary.
- `capture_cache_hit`, `capture_cache_key`, `capture_cache_path`: deferred to the save/cache design review.
- `manual_tensor_connections`: deferred until manual graph-link workflows are reviewed.
- `observer_spans`: deferred until span recording surface is reviewed.
- `relationship_evidence`: deferred until rerun relationship reporting is reviewed.
- `streaming_pass_logs` / `num_streamed_ops` / `num_streamed_passes`: REMOVED pre-launch — redundant with Bundle (`tl.bundle(*[tl.trace(model, x) for x in inputs])` covers the same use case). Activation-streaming during capture (bundle_path streaming, out_callback) is a separate concept and is kept; documented for review separately.
- `find_sites` and `resolve_sites`: deferred to the integrated intervention `Site` concept survey.
- `set`, `attach_hooks`, `do`, `clear_hooks`, `remove`, `detach_hooks`, `save_intervention`, and `intervention_spec`: deferred to intervention API review.
- `preview_fastlog`: moved to `tl.fastlog.preview`.
- `peek -> pluck`: top-level `tl.peek` rename target. Deferred.
- `tl.rerun(append=...)` rename target: deferred.
- `replay` / `replay_from` verb rename: deferred.
- Bundle dynamic helpers `aligned_pairs`, `compare`, `delta_map`, `norm_delta`, `output_delta`, and `show_diff`: deferred to Bundle helper redesign.
- `save_activations` and other workflow-level save kwargs: deferred pending decision on workflow vocabulary.
- Fast-path naming such as `fastlog`: deferred.
- `tl.bridge.hf` extras footprint: the auto-routed tracers (`trace_text`, `trace_image`, `trace_multimodal`) are documented in the "Input auto-routing" section; deeper bridge internals remain under `tl.bridge.*` namespaces.
- `gradient_postfunc` vs `grad_transform`: silent alias landed in alpha.3. Rename pass should pick one canonical name.
- `Trace.capture_config` namespace migration: move capture flags into `Trace.capture_config` dataclass. Deferred to post-rename sprint.
- Variable name introspection (`Op.var_names`): deferred (possible future feature).
- Accessor superclass refactor (`Accessor[T]` base): deferred to post-rename structural pass.
- Bundle aggregate metadata fields (`total_activation_memory`, `is_structurally_consistent`, etc.): deferred.
- `module_filter` rename to `save_predicate`: deferred (paired with `fastlog` namespace decision).
- `Trace.backward_peak_memory`: deferred (feature scope).
- `bundle.at()` dispatch logic edge cases: deferred to dedicated design pass.

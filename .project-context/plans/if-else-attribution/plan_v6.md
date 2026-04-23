# Sprint Plan: Full `if` / `elif` / `else` Branch Attribution for TorchLens

**Date:** 2026-04-22
**Branch:** `sprint/2026-04`
**Version:** v6 (post-adversarial-review-5)
**History:** `plan_v{1,2,3}.md`, `plan_v5.md`, `review_round_{1,2,3,4,5}.md`.

## Goals

1. Correctly attribute every captured op to its enclosing Python `if` / `elif` / `else` branch in eager `forward()` methods.
2. Robust false-positive filtering (`assert`, `bool()` cast, comprehension filters, `while`, `ifexp`, `match` guards, `unknown`).
3. Explicit per-pass branch representation on `LayerPassLog`; lossless aggregate representation on `LayerLog` and `ModelLog` (including rolled-edge pass divergence and multi-arm-entry edges).
4. Render THEN / ELIF / ELSE edge labels in Graphviz (primary) + dagua bridge; ELK documented as unsupported.
5. Comprehensive tests for positive paths, every documented limitation, multi-pass divergence, multi-arm entry, and lifecycle integration.
6. Strict backward compatibility — no existing field is renamed or removed; old fields derived from new primary structures.

## Out-of-Scope for v1 (explicit limitations)

**Non-branch bool consumers — classified, not attributed:**
- Ternary (`x if cond else y`) → `bool_context_kind="ifexp"`
- `while cond:` → `bool_context_kind="while"`
- Comprehension filters → `bool_context_kind="comprehension_filter"`
- `match case ... if guard:` → `bool_context_kind="match_guard"`
- `assert tensor_cond` → `bool_context_kind="assert"`
- `bool(tensor)` cast NOT nested inside `If.test` → `bool_context_kind="bool_cast"`

`bool(tensor_cond)` nested *inside* an `If.test`/elif test IS a branch (see D3).

**Source-unavailable — graceful degradation (`bool_context_kind="unknown"`, no attribution):**
- Jupyter / REPL cells, `exec`/`eval`, `.pyc`-only, native `.so` modules
- `torch.compile`, `torch.jit.script`, `torch.jit.trace` wrapped models
- `nn.DataParallel` / `DistributedDataParallel`
- Monkey-patched `forward` where source resolves to the patch

**Documented false negatives — torchlens cannot see:**
- Pure Python predicates: `if self.training:`, `if python_bool:`
- `if tensor.item() > 0:` — scalarization bypasses `Tensor.__bool__`
- Shape/metadata predicates: `if x.shape[0] > 0:`
- Functional conditionals: `torch.where`, `torch.cond`, masked blending

**Deferred to v2 release:**
- Column-offset disambiguation for multi-op lines
- `ast.IfExp` full attribution
- `while` loop body attribution
- ELK conditional edge rendering

## Decisions (architect-level)

| # | Decision |
|---|----------|
| D1 | AST + per-op frame inspection. NOT `sys.settrace`, NOT bytecode. |
| D2 | Flatten `if/elif/.../else` into ONE conditional record with branch arms `{then, elif_1, elif_2, ..., else}` (Python AST has no `Elif` node). |
| D3 | Pre-classify every terminal scalar bool by enclosing AST context. `bool(x)` nested inside `If.test`/elif test normalizes to `if_test`/`elif_test` + `bool_wrapper_kind="bool_cast"`. Standalone `bool(x)` → `bool_cast`. |
| D4 | Attribution uses full `func_call_stack` (every frame), not just the user-facing frame. |
| D5 | `LayerPassLog.conditional_branch_stack` is the per-pass source of truth. `LayerLog` aggregates via unique signatures + pass map. |
| D6 | Keep every existing field; old fields populated from new primary structures as derived views. |
| D7 | AST logic lives in new module `torchlens/postprocess/ast_branches.py`. |
| D8 | File cache keyed by `(filename, stat.st_mtime_ns)`. Process-local. |
| D9 | AST classification runs whenever `(file, line)` is on `func_call_stack` (which is always, after D18 ungates capture). `save_source_context` only gates rich source-text capture. |
| **D10 (REVISED)** | Two-stage identifier scheme: `ConditionalKey = (file, func_firstlineno, if_stmt_lineno, if_col_offset)` is the AST-internal structural key used in Phases 5a/5b/5e. Phase 5c is the **sole** step that materializes `ModelLog.conditional_events` and assigns dense per-`ModelLog` `conditional_id: int`. Phase 5c rewrites all `bool_conditional_key` → `bool_conditional_id` and all stack entries `(ConditionalKey, kind)` → `(cond_id, kind)` in one pass. No other code outside `ast_branches.py` sees `ConditionalKey`. |
| D11 | Graphviz gets ELIF/ELSE labels with explicit precedence (see Visualization). Dagua bridge extended. ELK gets a comment. |
| D12 | Old post-validation ("clear IF markings if no `ast.If`") is REMOVED; pre-classification makes it unnecessary. |
| D13 | Branch-entry edges detected by diffing `conditional_branch_stack` across EVERY forward edge, not only edges from branch-start nodes. Legitimate: a single edge can gain MULTIPLE stack entries simultaneously (e.g., `self.bias → x` inside nested `if A: if B:` enters both outer and inner THEN). All gained entries are recorded. |
| **D14 (REVISED v2)** | `FuncCallLocation` augmented with `code_firstlineno: int` (always) + `code_qualname: Optional[str]` (Python 3.11+; None on 3.9/3.10). Scope resolution in `attribute_op`: (1) exact match on `(filename, code_firstlineno, code_qualname)` if qualname non-None; (2) exact match on `(filename, code_firstlineno, func_name)` IFF exactly one scope matches; (3) if step 2 has zero or multiple candidates, **FAIL CLOSED** — return empty branch stack for that frame (no "smallest containing scope" heuristic). This prevents silent wrong attribution on same-line nested defs/lambdas, especially under 3.9/3.10 where `code_qualname` is absent. |
| D15 | `ModelLog.conditional_edge_passes` maps `(parent_no_pass, child_no_pass, cond_id, branch_kind) → sorted List[int]` of pass numbers. Rolled-mode renderers consult this for mixed-arm labels. |
| **D16 (NEW)** | **Cond-id-aware primary structures.** Primary arm-edge representation is `ModelLog.conditional_arm_edges: Dict[Tuple[int, str], List[Tuple[str, str]]]` (`(cond_id, branch_kind) → [(parent, child)]`). Primary per-node child list is `LayerPassLog.cond_branch_children_by_cond: Dict[int, Dict[str, List[str]]]` (`cond_id → branch_kind → [child_labels]`). Existing `conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges`, `cond_branch_then_children`, `cond_branch_elif_children`, `cond_branch_else_children` become DERIVED VIEWS computed from the primary structures. This lets a single edge correctly record entry into multiple arms simultaneously (D13). |
| **D17 (REVISED v3)** | **Explicit capture-path ungating + disabled-source state identical to the existing "no source available" contract on `FuncCallLocation`.** Phase 1 unconditionally populates identity fields (`file`, `line_number`, `func_name`, `code_firstlineno`, `code_qualname`, `source_loading_enabled`) on every captured op at `output_tensors.py:311`, `source_tensors.py:169`, `graph_traversal.py:67`. `source_loading_enabled` mirrors `save_source_context` at capture time. **When `source_loading_enabled=False`:** construction sets `_source_loaded=True` immediately and initializes the same observable state that `_load_source()` produces when source can't be loaded (see `func_call_location.py:149-153,165-169`): `code_context=None`, `source_context="None"` (literal 4-char string — current API), `code_context_labeled=""`, `call_line=""`, `num_context_lines=0`, `func_signature=None`, `func_docstring=None`, `_frame_func_obj=None`. Accessors and dunder methods inherit current no-source behavior verbatim — `len==0`, `__getitem__` raises `IndexError`, `__repr__` ends with "code: source unavailable" — with zero disk access. Clearing `_frame_func_obj` at construction eliminates pickle fragility with nested/local function objects (previously only released inside `_load_source()`). This reuses the already-existing no-source contract for complete API/test backward compatibility. D17 is distinct from D9 (AST classification needs identity fields only). |

## Data Model

### `FuncCallLocation` (augmented — D14, D17)

```python
@dataclass
class FuncCallLocation:
    file: str
    line_number: int
    func_name: str
    code_firstlineno: int                 # NEW — frame.f_code.co_firstlineno, always populated
    code_qualname: Optional[str]          # NEW — frame.f_code.co_qualname (3.11+); None pre-3.11
    source_loading_enabled: bool          # NEW — mirrors save_source_context at capture time
    # When source_loading_enabled=False, construction sets _source_loaded=True
    # immediately with the no-source state already produced by _load_source() when
    # source can't be loaded (see func_call_location.py:149-153, 165-169):
    #   code_context=None
    #   source_context="None"       # literal 4-char string (current API)
    #   code_context_labeled=""
    #   call_line=""
    #   num_context_lines=0
    #   func_signature=None
    #   func_docstring=None
    #   _frame_func_obj=None        # cleared at construction -> safe for pickle
    # Existing accessor semantics carry over: len(loc)==0,
    # loc[i] raises IndexError, repr(loc) ends with "code: source unavailable".
    # No disk access ever; _load_source is effectively never entered.
```

### `LayerPassLog` — new fields

```python
# Bool classification (meaningful when is_terminal_bool_layer=True and is_scalar_bool=True)
bool_is_branch: bool                                 # iff bool_context_kind ∈ {"if_test","elif_test"}
bool_context_kind: Optional[str]                     # 9 kinds + "unknown"
bool_wrapper_kind: Optional[str]                     # e.g. "bool_cast" when wrapped; else None
bool_conditional_id: Optional[int]                   # dense; None for non-branch kinds

# Branch attribution (any op)
conditional_branch_stack: List[Tuple[int, str]]      # [(cond_id, branch_kind), ...] outer→inner
conditional_branch_depth: int                         # = len(stack)

# PRIMARY per-node arm-child structure (D16)
cond_branch_children_by_cond: Dict[int, Dict[str, List[str]]]
    # cond_id → branch_kind → [child labels]
```

**Preserved (now DERIVED from `cond_branch_children_by_cond`):**
- `cond_branch_start_children: List[str]` — still the set-of-parents marked by 5d IF-backward flood (UNCHANGED semantics).
- `cond_branch_then_children: List[str]` — `sorted(set(chain_from_iterable(m["then"] for m in cond_branch_children_by_cond.values() if "then" in m)))`.
- `cond_branch_elif_children: Dict[int, List[str]]` — aggregate by elif-index across cond_ids.
- `cond_branch_else_children: List[str]` — analogous.

Also preserved: `is_terminal_bool_layer`, `is_scalar_bool`, `scalar_bool_value`, `in_cond_branch` (derived `len(conditional_branch_stack) > 0`).

### `LayerLog` — multi-pass aggregation

```python
# Aggregate
conditional_branch_stacks: List[List[Tuple[int, str]]]     # unique signatures, first-seen order
conditional_branch_stack_passes: Dict[
    Tuple[Tuple[int, str], ...], List[int]
]                                                           # signature → sorted pass numbers

# Per-node cond-id-aware aggregate children (primary — pass-stripped labels)
cond_branch_children_by_cond: Dict[int, Dict[str, List[str]]]
```

**Derived views (pass-stripped unions):** `cond_branch_then_children`, `cond_branch_elif_children`, `cond_branch_else_children`, `cond_branch_start_children`.

**Merge rules** (update `_build_layer_logs`):
- `in_cond_branch`: OR across passes.
- `conditional_branch_stacks`: unique signatures in first-seen order.
- `conditional_branch_stack_passes`: accumulate sorted pass numbers per signature.
- `cond_branch_children_by_cond`: union across passes after stripping `:N` suffix; keep cond_id granularity.

### `ModelLog` — new fields

```python
conditional_events: List[ConditionalEvent]            # dense cond_id indexing (D10)

# PRIMARY edge structure (D16)
conditional_arm_edges: Dict[Tuple[int, str], List[Tuple[str, str]]]
    # (cond_id, branch_kind) → [(parent, child), ...]

# Rolled-mode pass-divergence (D15)
conditional_edge_passes: Dict[
    Tuple[str, str, int, str], List[int]
]                                                     # (parent_no_pass, child_no_pass, cond_id, branch_kind) → sorted pass numbers
```

**Preserved (DERIVED from `conditional_arm_edges`):**
- `conditional_branch_edges: List[Tuple[str, str]]` — IF-edges from 5d (unchanged).
- `conditional_then_edges: List[Tuple[str, str]]` — `[(p,c) for (cid, bk), edges in conditional_arm_edges.items() if bk=="then" for (p,c) in edges]`.
- `conditional_elif_edges: List[Tuple[int, int, str, str]]` — `[(cid, int(bk.split("_")[1]), p, c) for (cid, bk), edges if bk.startswith("elif_") for (p,c) in edges]`.
- `conditional_else_edges: List[Tuple[int, str, str]]` — `[(cid, p, c) for (cid, bk), edges if bk=="else" for (p,c) in edges]`.

`ConditionalEvent`:
```python
@dataclass
class ConditionalEvent:
    id: int                                                           # dense per-ModelLog
    source_file: str
    function_qualname: str
    function_span: Tuple[int, int]
    if_stmt_span: Tuple[int, int]
    test_span: Tuple[int, int]
    nesting_depth: int
    branch_ranges: Dict[str, Tuple[int, int]]                         # "then", "elif_1", ..., "else"
    branch_test_spans: Dict[str, Tuple[int, int]]                     # "then", "elif_1", ...
    parent_conditional_id: Optional[int]
    parent_branch_kind: Optional[str]
    bool_layers: List[str]                                            # labels of bools driving this event
```

### `constants.py` FIELD_ORDER updates (one commit)

- `LAYER_PASS_LOG_FIELD_ORDER`: add 7 new entries (`bool_is_branch`, `bool_context_kind`, `bool_wrapper_kind`, `bool_conditional_id`, `conditional_branch_stack`, `conditional_branch_depth`, `cond_branch_children_by_cond`). Derived fields (`cond_branch_then_children`, `cond_branch_elif_children`, `cond_branch_else_children`) stay in FIELD_ORDER — they're materialized for serialization.
- `MODEL_LOG_FIELD_ORDER`: add 3 new primary entries (`conditional_events`, `conditional_arm_edges`, `conditional_edge_passes`). Derived fields (`conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges`) stay.
- `LAYER_LOG_FIELD_ORDER`: add 4 aggregate entries (`conditional_branch_stacks`, `conditional_branch_stack_passes`, `cond_branch_children_by_cond`, plus the aggregate derived children).

## Algorithm

### New module: `torchlens/postprocess/ast_branches.py`

Public API:
```python
def get_file_index(filename: str) -> Optional["FileIndex"]: ...
def classify_bool(filename: str, line: int) -> BoolClassification: ...
    # Returns: (kind, wrapper_kind, conditional_key_or_None, branch_test_kind_or_None)
def attribute_op(func_call_stack: List[FuncCallLocation]
                 ) -> List[Tuple[ConditionalKey, str]]: ...
    # Returns ConditionalKeys, not dense ints. Dense IDs are assigned in Phase 5c.
def invalidate_cache(filename: Optional[str] = None) -> None: ...

ConditionalKey = Tuple[str, int, int, int]   # (file, func_firstlineno, if_stmt_lineno, if_col_offset)
```

### Postprocess Step 5 restructure (six-phase — revised per D10)

**5a. Build file indexes.**
For every terminal scalar bool layer, collect filenames from `func_call_stack`. Eagerly build `FileIndex` per file (cache).

**5b. Classify bools (structural keys).**
For each `is_terminal_bool_layer` scalar bool, **iterating in `ModelLog.layer_list` order for determinism**:
- Walk frames; call `classify_bool(frame.file, frame.line_number)`; take the most-informative result (innermost non-`unknown` kind).
- Record on the bool's `LayerPassLog` temporarily:
  - `bool_context_kind`, `bool_wrapper_kind`, `bool_is_branch`, and a TEMPORARY attribute `_bool_conditional_key` (not in FIELD_ORDER — transient).
- Collect unique `ConditionalKey`s in **first-seen order** (insertion-ordered dict, not a set).

**5c. Materialize events; assign dense IDs; translate keys → ints. (Sole id-assignment step.)**
- Build `ModelLog.conditional_events` list: for each unique `ConditionalKey` in the **first-seen order from 5b**, create a `ConditionalEvent` with a dense `id` (0, 1, 2, ...). This guarantees stable `cond_id` assignment across runs on the same model/input.
- Walk every bool layer with `_bool_conditional_key`: set `bool_conditional_id = events_by_key[_bool_conditional_key].id`; register label in `ConditionalEvent.bool_layers`; delete `_bool_conditional_key`.
- Also translate `parent_conditional_id` fields inside `ConditionalEvent`s that reference other keys.
- From this point forward, no `ConditionalKey` escapes `ast_branches.py` — only dense `int` IDs are exposed.

**5d. IF backward flood** (PR #127 behavior, now pre-filtered).
Starting from bools with `bool_is_branch=True` only, backward-flood through `parent_layers`; mark output-ancestor parents with `cond_branch_start_children`; emit `conditional_branch_edges`. Unchanged by D13.

**5e. Forward branch attribution + arm edges.**
For every op:
- `stack_keys = attribute_op(func_call_stack)` → list of `(ConditionalKey, kind)`.
- Translate keys to dense IDs: for each `(key, kind)`, if `key ∈ events_by_key` emit `(events_by_key[key].id, kind)`; otherwise **DROP** the entry. A `ConditionalKey` may be returned by `attribute_op` that was never materialized in 5c — this happens when an op is lexically inside an `if` whose predicate was not driven by a captured tensor bool (the documented false-negative scope: `if self.training:`, `if x.shape[0] > 0:`, `if tensor.item() > 0:`). Dropping unmaterialized keys keeps the attribution consistent with the event set.
- Assemble `conditional_branch_stack: List[Tuple[int, str]]` from the surviving entries.
- Set `conditional_branch_depth`, update `in_cond_branch`.

For every forward edge `(parent, child)` (ALL edges — D13):
- Compute set difference `gained = child.stack - parent.stack` (prefix-order, so elements not in parent's stack in order).
- Each gained `(cond_id, branch_kind)` triggers:
  - `ModelLog.conditional_arm_edges[(cond_id, branch_kind)].append((parent.label, child.label))`.
  - `parent.cond_branch_children_by_cond[cond_id][branch_kind].append(child.label)`.
- If `parent.stack` has entries not in `child.stack` (branch exit / reconvergence): no edge is recorded; valid per Invariant 6.

**5f. Derived-view materialization + `conditional_edge_passes`.**
- Compute derived fields (`conditional_then_edges`, `cond_branch_then_children`, etc.) from primary cond-id-aware structures.
- For rolled-mode awareness, record each arm-edge's pass-stripped form: `conditional_edge_passes[(parent_no_pass, child_no_pass, cond_id, branch_kind)].append(pass_num)`; sort lists at the end.

### Classification algorithm (`classify_bool`)

```python
def classify_bool(filename: str, line: int) -> BoolClassification:
    idx = get_file_index(filename)
    if idx is None:
        return BoolClassification(kind="unknown", wrapper_kind=None,
                                   conditional_key=None, branch_test_kind=None)

    # Walk bool consumers whose span contains `line`, innermost outward.
    # Each consumer has kind ∈ {if_test, elif_test, assert, bool_cast,
    #                            comprehension_filter, while, ifexp, match_guard}.
    consumers = [c for c in idx.bool_consumers if c.span.contains(line)]
    consumers.sort(key=lambda c: c.depth, reverse=True)   # innermost first

    wrapper = None
    for consumer in consumers:
        if consumer.kind == "bool_cast" and wrapper is None:
            wrapper = "bool_cast"
            continue   # keep walking outward; bool_cast doesn't decide alone
        if consumer.kind in {"if_test", "elif_test"}:
            return BoolClassification(
                kind=consumer.kind,
                wrapper_kind=wrapper,
                conditional_key=consumer.conditional_key,
                branch_test_kind=consumer.branch_test_kind,
            )
        # any other consumer wins outright (assert/while/ifexp/comp/match)
        return BoolClassification(kind=consumer.kind, wrapper_kind=wrapper,
                                   conditional_key=None, branch_test_kind=None)

    # Reached here: maybe a standalone bool_cast with no outer branch
    if wrapper == "bool_cast":
        return BoolClassification(kind="bool_cast", wrapper_kind=None,
                                   conditional_key=None, branch_test_kind=None)
    return BoolClassification(kind="unknown", wrapper_kind=None,
                               conditional_key=None, branch_test_kind=None)
```

### Attribution algorithm (`attribute_op`)

Per frame (shallow → deep):
1. Scope resolution per D14:
   - `(filename, code_firstlineno, code_qualname)` exact match if `code_qualname` is non-None; otherwise
   - `(filename, code_firstlineno, func_name)` match IFF exactly one scope in the file index matches; otherwise
   - **FAIL CLOSED**: skip this frame's contribution entirely (no branch-interval query, no heuristic, no smallest-scope guess). Emit a debug counter for diagnostics.
2. If scope was resolved: query branch-interval tree at `frame.line_number` → list of `(ConditionalKey, branch_kind, nesting_depth)` sorted by nesting depth asc.
3. Append to the aggregate stack (deduplicate adjacent).

## Visualization

### Label precedence (Graphviz + dagua bridge)

1. **Arm-entry label** (highest) — edge appears in `conditional_arm_edges[(cond_id, "then")]` / `elif_N` / `else`:
   - Single-arm: `THEN` / `ELIF N` / `ELSE`.
   - Multi-arm on same edge (D13 case): composite like `THEN(cond_outer) · THEN(cond_inner)` — concise form to be agreed; default: `N arms`.
   - Rolled mode with pass divergence (D15): `THEN(1,3) / ELSE(2)` or `mixed`.
2. **IF label** — edge is in `conditional_branch_edges`.
3. **Arg labels** — move from `edge_dict["label"]` to `edge_dict["headlabel"]` / `xlabel` so they don't compete.

### Graphviz (`torchlens/visualization/rendering.py`)

- Refactor existing `edge_dict["label"]` writes (lines ~985-996 and ~1049-1090) into a single `_compute_edge_label(parent, child, model_log)` function implementing the precedence rules.
- Move argument-name labels off `label` → `headlabel` / `xlabel`.

### Dagua bridge (`torchlens/visualization/dagua_bridge.py`)

- Extend `_classify_forward_edge` to return new edge kinds: `"elif_N"`, `"else"`, `"multi_arm"`, `"mixed_pass"`.

### ELK (`torchlens/visualization/elk_layout.py`)

- Single comment: ELK does not render conditional edge labels.

## Capture / Postprocess Lifecycle (full integration — addresses F4, F9 partials)

Every new label-bearing field must be wired end-to-end. Phase 3 delivers all of this BEFORE Phase 4 (Step 5) can populate data.

### Capture-time defaults (Phase 1)

Files: `torchlens/capture/source_tensors.py`, `torchlens/capture/output_tensors.py`, `torchlens/postprocess/graph_traversal.py`.
- Init every new `LayerPassLog` field with default (empty list, `{}`, `False`, `None`).
- Remove gating of `func_call_stack` population on `save_source_context` (D17). Populate core fields (`file`, `line_number`, `func_name`, `code_firstlineno`, `code_qualname`) unconditionally. Lazy source-text properties remain gated.

### Raw→final rename (Phase 3, `torchlens/postprocess/labeling.py`)

Extend `_rename_raw_labels_in_place` to rewrite labels in:
- `conditional_arm_edges` (each `(parent, child)` tuple).
- `conditional_edge_passes` (keys — positions 0, 1).
- `conditional_events[*].bool_layers`.
- `cond_branch_children_by_cond` per `LayerPassLog` (every nested list member).
- Existing fields (`conditional_branch_edges`, `conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges`, `cond_branch_start_children`, `cond_branch_then_children`, `cond_branch_elif_children`, `cond_branch_else_children`).

### Cleanup / `keep_unsaved_layers=False` (Phase 3, `torchlens/data_classes/cleanup.py`)

Extend filter logic to scrub every new surface when a label is removed:
- `conditional_arm_edges`: drop tuples where parent OR child is removed; drop empty keys.
- `conditional_edge_passes`: drop tuple keys where parent or child is removed.
- `conditional_events[*].bool_layers`: remove labels.
- `cond_branch_children_by_cond`: drop removed children; drop empty branch_kind keys; drop empty cond_id keys.
- `LayerLog.conditional_branch_stack_passes`: unchanged (signatures are cond_id tuples; labels aren't in keys).
- Also scrub derived views (`cond_branch_then_children`, etc.); they'll be recomputed anyway, but the pruning must be consistent.

### Export (`torchlens/data_classes/interface.py::to_pandas`)

Add columns for: `conditional_branch_depth`, `bool_is_branch`, `bool_context_kind`, `bool_wrapper_kind`, `bool_conditional_id`, plus a compact string form of `conditional_branch_stack`.

## Validation / Invariants (`torchlens/validation/invariants.py`)

1. **`cond_branch_children_by_cond` ↔ `conditional_arm_edges` bidirectional consistency.**
   For each `(cond_id, branch_kind) → edges` in `conditional_arm_edges`, every `(parent, child)`: `child ∈ parent.cond_branch_children_by_cond[cond_id][branch_kind]` and vice versa.
2. **Derived views consistency.**
   `cond_branch_then_children`, `cond_branch_elif_children`, `cond_branch_else_children` per-node and `conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges` on `ModelLog` equal the projections of primary structures.
3. **Children labels exist in `ModelLog`.**
4. **Bool classification invariants.**
   `bool_is_branch=True` ⟺ `bool_context_kind ∈ {"if_test", "elif_test"}`.
   `bool_is_branch=True` ⟹ `bool_conditional_id is not None`.
   `bool_context_kind is not None` ⟹ layer has `is_terminal_bool_layer=True`.
   `bool_wrapper_kind is not None` ⟹ `bool_context_kind ∈ {"if_test", "elif_test", "bool_cast"}`.
5. **Conditional event references are valid.**
   Every `cond_id` in any stack entry, `bool_conditional_id`, `conditional_arm_edges` key, or `cond_branch_children_by_cond` key has matching `ConditionalEvent.id`.
6. **Stack monotonicity on every parent→child edge (REVISED).**
   Exactly one of: child.stack extends parent.stack (entry); parent.stack extends child.stack (exit/reconvergence); stacks equal. Non-prefix reorderings are invalid.
7. **Elif contiguity on `ConditionalEvent` (per-event, not per-node).**
   `branch_ranges` and `branch_test_spans` keys for elif arms are contiguous from `elif_1`. Per-node `cond_branch_children_by_cond[cid]` may be sparse — that's legal.
8. **`bool_layers` back-references (NEW).**
   For every `ConditionalEvent.bool_layers`: each label exists, and each referenced layer's `bool_conditional_id` equals the event's `id`.
9. **`LayerLog` aggregate consistency (NEW).**
   `LayerLog.conditional_branch_stacks` equals exactly the set of distinct `LayerPassLog.conditional_branch_stack` values from that layer's passes.
   `LayerLog.conditional_branch_stack_passes[sig]` equals the sorted list of pass numbers for which that signature appeared.
10. **`conditional_edge_passes` exactness (NEW).**
    For every `(parent_no_pass, child_no_pass, cond_id, branch_kind) → [pass_nums]`:
    - `pass_nums` is sorted ascending and has no duplicates.
    - For each `pass_num` there exists an unrolled edge `(parent:pass_num, child:pass_num)` in `conditional_arm_edges[(cond_id, branch_kind)]`.
    - Every unrolled edge in `conditional_arm_edges` is represented somewhere in `conditional_edge_passes`.
11. **No orphan `_bool_conditional_key` (NEW, dev-mode).**
    After 5c, no `LayerPassLog` has a `_bool_conditional_key` attribute. Enforced in dev mode to catch incomplete 5c migrations.
12. **`LayerLog.cond_branch_children_by_cond` aggregate exactness (NEW).**
    For every `LayerLog`, `cond_branch_children_by_cond` equals the pass-stripped union of `cond_branch_children_by_cond` across its constituent `LayerPassLog` passes. No cond_id or branch_kind present in any pass is missing from the aggregate; no extraneous entries appear.
13. **Legacy IF-view consistency (NEW).**
    `conditional_branch_edges` ↔ `cond_branch_start_children` bidirectionally consistent: every `(parent, bool_label)` tuple in `conditional_branch_edges` has `bool_label ∈ parent.cond_branch_start_children`; and every label in any node's `cond_branch_start_children` corresponds to a tuple with that node as parent in `conditional_branch_edges`.

## Tests

### New file `tests/test_conditional_branches.py` + models in `tests/example_models.py` (or dedicated new file if size warrants)

**Baseline / positive:**
- `SimpleIfElseModel` / `ElifLadderModel` / `NestedIfThenIfModel` / `NestedInElseModel` / `MultilinePredicateModel` — verify THEN/ELIF/ELSE edges, `conditional_events` entries, branch_ranges correctness.

**Branch-entry via non-predicate ancestors (D13):**
- `BranchUsesOnlyParameterModel` — `if c: y = self.bias + 1`. Assert `self.bias → y` appears in `conditional_arm_edges[(cid, "then")]`.
- `BranchUsesOnlyConstantModel` — analogous with a constant.
- **`MultiArmEntryNestedModel` (NEW — Finding 2):** `if A: if B: y = self.bias + 1`. Assert the single `self.bias → y` edge appears in BOTH `conditional_arm_edges[(outer_id, "then")]` AND `conditional_arm_edges[(inner_id, "then")]`, and `self.bias.cond_branch_children_by_cond` has both `outer_id` and `inner_id` mapping to `[y]`. Verify derived `cond_branch_then_children` also lists `y` (uniquely, not twice).

**Wrapped bool (F2 resolution):**
- `IfBoolCastModel` — `if bool(x.sum() > 0):`. Assert `bool_is_branch=True`, `bool_context_kind="if_test"`, `bool_wrapper_kind="bool_cast"`.

**Multi-pass / rolled (D15):**
- `LoopedIfAlternatingModel` — `for i: if i%2`. Per-pass stacks differ; aggregate has 2 signatures.
- `AlternatingRecurrentIfModel` — recurrent, THEN pass 1 / ELSE pass 2.
- **`RolledMixedArmModel` (strengthened — F7 resolution):** Assert `ModelLog.conditional_edge_passes` has entries with non-trivial pass lists AND each `(parent_no_pass, child_no_pass, cond_id, branch_kind) → pass_nums` is exactly verifiable — sorted ascending, no duplicates, every `pass_num` corresponds to a real unrolled edge in `conditional_arm_edges[(cond_id, branch_kind)]` on that pass. Assert rolled renderer output (graphviz+dagua) contains a composite label. **D13×D15 interaction:** add a companion scenario where the mixed-pass arm edge is ALSO a multi-arm-entry edge (nested conditional taking different inner arms across passes); assert both dimensions survive end-to-end. **Post-cleanup check:** after `keep_unsaved_layers=False`, assert no pruned labels appear in `conditional_edge_passes` keys.

**Reconvergence (F5):**
- `ReconvergingBranchesModel` — `if c: x=a else: x=b; y=f(x)`. Invariant 6 passes. Assert `y.conditional_branch_stack == []` and both parents' stacks were non-empty.

**Scope resolution (D14):**
- `NestedHelperSameNameModel` — two local `helper()` functions, same name, different branches. Assert attribution uses `code_firstlineno`; verify `bool_conditional_id` differs between the two helpers' if-statements.
- **`NestedQualnameModel` (NEW — replaces LambdaBranchModel):** `class Outer: def forward(self): def helper(self): if cond: op() return helper(self)` — nested function named `helper` with a distinct `code_qualname` from a method also named `helper` elsewhere. Assert attribution distinguishes them via `code_qualname` (Python 3.11+).
- **`SameLineNestedDefModel` (NEW — D14 fail-closed):** Two local defs with the same `func_name` and same `code_firstlineno` (degenerate case, forceable via `exec` or metaprogramming). Assert D14 step 3 triggers: returns empty branch stack for that frame rather than silently picking one; no `conditional_branch_edges` or arm edges are emitted from that site. Also run this under Python 3.9/3.10 CI (where `code_qualname` is None) to verify fail-closed behavior across supported versions.
- `DecoratedForwardModel` — decorator on forward. Graceful degradation OR correct attribution.

**False positives — non-branch bools:**
- `AssertTensorCondModel`, `BoolCastOnlyModel`, `TernaryIfExpModel`, `ComprehensionIfModel`, `WhileLoopModel`, `MatchGuardModel`.

**Compound / negation / walrus:**
- `NotIfModel`, `AndOrIfModel`, `WalrusIfModel`.

**Documented false negatives (no attribution):**
- `PythonBoolModel`, `ItemScalarizationModel`, `TorchWhereModel`, `ShapePredicateModel`.

**save_source_context gating (F1 resolution — STRENGTHENED):**
- **`SaveSourceContextOffModel` (strengthened):** `save_source_context=False`. Assert `func_call_stack` is NON-EMPTY on every captured op; each `FuncCallLocation` has non-None `file`, `line_number`, `code_firstlineno`, and `source_loading_enabled=False`. Assert every accessor returns the existing no-source state: `loc.code_context is None`, `loc.source_context == "None"` (literal 4-char string per current API at `func_call_location.py:150`), `loc.code_context_labeled == ""`, `loc.call_line == ""`, `loc.num_context_lines == 0`, `loc.func_signature is None`, `loc.func_docstring is None`. Assert `len(loc) == 0`, `loc[0]` raises `IndexError`, and `repr(loc)` ends with `"code: source unavailable"`. Assert `loc._frame_func_obj is None` (cleared at construction per D17). Assert no disk access occurred (e.g., by patching `linecache.getlines` with a sentinel; or by monitoring `_source_loaded=True` straight after construction). Branch attribution populated correctly. **Pickle round-trip:** pickle/unpickle a `ModelLog` captured with `save_source_context=False`; unpickled log round-trips cleanly (exercises the `_frame_func_obj` retention fix).
- **`SaveSourceContextOffAssertModel`:** `save_source_context=False` + `assert tensor_cond`. No false-positive IF edge.

**Field-lifecycle integration (F4 resolution — STRENGTHENED):**
- **`KeepUnsavedLayersFalseModel` (strengthened):** `keep_unsaved_layers=False` with conditionals. Assert NO removed label appears anywhere in:
  - `conditional_arm_edges` (tuples).
  - `conditional_edge_passes` (tuple keys).
  - `conditional_events[*].bool_layers`.
  - `cond_branch_children_by_cond` (any nested list) — check on BOTH `LayerPassLog` AND `LayerLog` aggregates.
  - `conditional_branch_stack_passes` values on `LayerLog`.
  - All derived views (`cond_branch_then_children`, `cond_branch_elif_children`, `cond_branch_else_children`, `cond_branch_start_children`, `conditional_then_edges`, `conditional_elif_edges`, `conditional_else_edges`).
- `ToPandasConditionalModel` — DataFrame has all conditional columns populated.

**Visualization label precedence (NEW — F8 confirmation):**
- **`BranchEntryWithArgLabelModel` (NEW):** branch-entry edge that ALSO has an argument label. Assert graphviz renderer output shows branch label in `edge_dict["label"]` and arg label in `edge_dict["headlabel"]` (or `xlabel`) — they do not collide. Assert dagua bridge classifies the edge (`_classify_forward_edge` returns `"then"`, not `"default"`) AND the emitted dagua edge metadata carries BOTH the branch kind and the arg label in distinct fields (specification: branch kind in `edge.type`, arg label in a separate `arg_label` attribute that the dagua bridge adds alongside `edge.label`).

**Smoke:** `@pytest.mark.smoke` on `SimpleIfElseModel`, `ElifLadderModel`.

## Backward Compatibility

- Every existing field retained with identical external semantics.
- Old fields materialized as derived views from primary cond-id-aware structures. Serialization includes both primary and derived (FIELD_ORDER includes both).
- `in_cond_branch` remains a direct field; recomputed from `conditional_branch_stack`.
- `FIELD_ORDER` updates in single commit.
- Pickle compat: `__setstate__` default-fill for 1 prior minor version. If primary structures are missing on load, reconstruct from legacy derived fields where possible (best-effort for N-1 compat; newer loaders prefer primary).

## Implementation Phases

Each phase gate: tier-1 smoke passes. Tier-2 at phases 4, 6, 9. Tier-3 before PR.

1. **Foundation: fields + capture ungating.**
   - Add all new fields to `LayerPassLog`, `LayerLog`, `ModelLog`, `FuncCallLocation` + `FIELD_ORDER` updates + defaults.
   - Ungate `func_call_stack` population in `output_tensors.py`, `source_tensors.py`, `graph_traversal.py` (D17).
   - Populate `code_firstlineno`, `code_qualname` in stack capture (`utils/introspection.py`).
   - Gate: existing test suite passes unchanged. New **conditional-branch** fields remain default (empty list/dict, `False`, `None`). New `FuncCallLocation` core identity fields (`code_firstlineno`, `code_qualname`, `source_loading_enabled`) ARE populated per D17; verify via test that `func_call_stack` is non-empty even when `save_source_context=False`, and that lazy source-text properties still return empty/None in that case.
2. **AST module.**
   - Write `ast_branches.py` (FileIndex, ConditionalRecord, BoolConsumer index, branch interval trees, classify_bool, attribute_op, cache management).
   - Unit tests on synthetic source snippets covering all 9 bool-context kinds, flattened elif chains, nested ifs, ternary, match guards, etc.
   - Gate: `ast_branches` tests pass in isolation.
3. **Lifecycle wiring (upstream of Step 5 integration).**
   - Extend `labeling.py::_rename_raw_labels_in_place` for every new field.
   - Extend `cleanup.py` for every new field.
   - Extend `interface.py::to_pandas` with new columns.
   - Integration test: synthetic model with new fields pre-populated manually; `keep_unsaved_layers=False` + `to_pandas()` round-trip.
   - Gate: lifecycle tests pass.
4. **Postprocess Step 5 integration (6 phases 5a-5f).**
   - Restructure `control_flow.py`. 5c is the sole key→id translation.
   - Wire to `ast_branches.py`.
   - Remove old post-validation (D12).
   - Gate: new integration test with `SimpleIfElseModel` passes end-to-end.
5. **Multi-pass merge + `conditional_edge_passes`.**
   - Update `_build_layer_logs` for aggregate fields.
   - Populate `ModelLog.conditional_edge_passes`.
   - Gate: `AlternatingRecurrentIfModel`, `RolledMixedArmModel` pass.
6. **Invariants.**
   - Add 11 checks to `validation/invariants.py`.
   - Wire into `validate_forward_pass`.
   - Gate: all tests still pass; invariants catch deliberately-corrupted metadata in unit tests.
7. **Graphviz rendering.**
   - Refactor to `_compute_edge_label` precedence function.
   - Move arg labels to `headlabel`/`xlabel`.
   - Implement multi-arm and rolled-mixed labels.
   - Gate: visual-diff regression check on existing aesthetic tests.
8. **Dagua bridge.**
   - Extend `_classify_forward_edge` with new edge kinds.
   - Gate: existing dagua tests still pass; new conditional coverage.
9. **Integration tests.**
   - All ~30 conditional tests with assertions.
   - Gate: all pass; coverage of every listed scenario.
10. **Documentation.**
    - Update `AGENTS.md` conditional section.
    - Update `architecture.md` Step 5 section.
    - Update `torchlens/user_funcs.py` docstrings: `save_source_context=False` no longer zeros `func_call_stack` — only source-text properties are gated (D17). Any README/API doc references to this behavior get the same update.
    - User-facing limitations page.
11. **PR.**
    - Migration notes with before/after field table.
    - Compat matrix.

## Deliverables

- 1 new module (~400 lines): `torchlens/postprocess/ast_branches.py`.
- Modified modules (~11):
  - `torchlens/data_classes/{layer_pass_log,layer_log,model_log,func_call_location,cleanup,interface}.py`
  - `torchlens/constants.py`
  - `torchlens/postprocess/{control_flow,labeling,graph_traversal}.py`
  - `torchlens/capture/{source_tensors,output_tensors}.py`
  - `torchlens/utils/introspection.py`
  - `torchlens/validation/invariants.py`
  - `torchlens/visualization/{rendering,dagua_bridge,elk_layout}.py`
- Tests: `tests/test_conditional_branches.py` (~550 lines), ~32 new models.
- Docs: AGENTS.md, architecture.md, user limitations.

## Risks

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | `FIELD_ORDER` mismatch | All field additions in one commit. |
| R2 | Multi-pass regressions on recurrent tests | `AlternatingRecurrentIfModel` / `RolledMixedArmModel` in Phase 5 test gate. |
| R3 | Source resolution fragile on decorated/wrapped forwards | Use `frame.f_code.co_firstlineno`/`co_qualname` (D14); don't rely on `inspect.getsourcefile(module_class)`. |
| R4 | `linecache` stale source after edit | Cache by `(filename, mtime)`; invalidate on mismatch; fail-soft to `unknown`. |
| R5 | Pre-classification misses a syntactic form → silent `unknown` | Unit tests for every kind; dev-mode assertion: line inside `If.test` span but classifier returns `unknown` → log warning. |
| R6 | Interval tree perf on huge `forward` | Parse once; O(log k + d) query; benchmark `HugeNestedIfModel` in Phase 2. |
| R7 | Old pickled ModelLogs fail | `__setstate__` default-fill; document window. |
| R8 | Arg-label relocation to `headlabel` breaks existing aesthetic tests | Visual-diff gate in Phase 7; gate behind a flag if regressions appear. |
| R9 | D16 primary structure inflates memory on large models | Benchmark; cond-id keys compact; derived views computed on-demand only. |
| R10 | D17 ungating breaks something depending on empty `func_call_stack` | Phase 1 tests check `func_call_stack` non-empty; audit any code reading it. |
| R11 | Multi-arm entry edges produce confusing visuals | Dedicated test (`MultiArmEntryNestedModel`) + label precedence rule + user docs. |

## Open Questions for Architect / User

1. ELK: document as unsupported (default). Confirm?
2. Multi-arm edge label format: `THEN(cond_outer) · THEN(cond_inner)` vs `2 arms` vs a renderer-side decision. Default: readable form with cond name/IDs.
3. Pickle compat window: 1 prior minor version (default).
4. v2 scope confirmation: ternary + while + column offsets + ELK parity.
5. Test model count: ~32. OK?
6. `code_qualname` fallback on Python 3.9/3.10 (no `co_qualname`): accept identity-fallback to `code_firstlineno` + `func_name`?

---

## CHANGELOG v2 → v3 (addresses `review_round_2.md`)

| Finding | Disposition |
|---------|-------------|
| R2-F1 Conditional ID contract (BLOCKER) | D10 revised + Step 5c is sole key→id translator + `ConditionalKey` explicit type. |
| R2-F2 Multi-entry edges cond-id-blind (HIGH) | D16 new: `conditional_arm_edges` + `cond_branch_children_by_cond` primary; legacy fields derived. |
| R2-F3 `code_qualname` not used in matching (MEDIUM) | D14 revised: exact match on `(filename, code_firstlineno, code_qualname)` first; fallbacks explicit. |
| R2-F4 Cleanup incomplete for D15 (MEDIUM) | Lifecycle/Cleanup section enumerates every new surface; `KeepUnsavedLayersFalseModel` strengthened. |
| R2 F1 partial (save_source_context capture path) | D17 new: Phase 1 explicitly ungates `output_tensors.py:311`, `source_tensors.py:169`, `graph_traversal.py:67`. `SaveSourceContextOffModel` strengthened to assert non-empty `func_call_stack`. |
| R2 F9 partial | Covered by D14 revision and test swap (NestedQualnameModel replacing LambdaBranchModel). |
| R2 missing invariants (bool_layers back-ref, aggregate stacks, edge_passes exactness) | Invariants 8, 9, 10, 11 added. |
| R2 phase ordering | Phases reordered: lifecycle wiring is Phase 3, BEFORE Step 5 integration (Phase 4). Phase 1 gate no longer references Phase 5 deliverables. |
| R2 test gaps | Added `MultiArmEntryNestedModel`, `NestedQualnameModel`, `BranchEntryWithArgLabelModel`; strengthened `SaveSourceContextOffModel`, `KeepUnsavedLayersFalseModel`, `RolledMixedArmModel`. |

## CHANGELOG v3 → v4 (addresses `review_round_3.md`)

| Finding | Disposition |
|---------|-------------|
| R3-F1 5e can attribute via unmaterialized keys (BLOCKER) | Step 5e spec now explicitly DROPS `ConditionalKey`s not in `events_by_key`; rationale and test link to documented false-negative scope. |
| R3-F2 D17 promises ungated stack but `FuncCallLocation` lazy props eagerly load (HIGH) | D17 revised; `FuncCallLocation` gains `source_loading_enabled: bool`; all lazy props + `__repr__`/`__len__`/`__getitem__` must fail closed when disabled. Phase 1 gate wording updated. |
| R3-F3 D14 fallback can still silently pick wrong scope on 3.9/3.10 (MEDIUM) | D14 revised v2: step 2 requires unique match; step 3 FAILS CLOSED (no smallest-containing-scope heuristic). New `SameLineNestedDefModel` test exercises the fail-closed path. |
| R3-F4 Invariants incomplete (MEDIUM) | Added invariants 12 (`LayerLog.cond_branch_children_by_cond` aggregate exactness) and 13 (legacy IF-view consistency). |
| R3-F5 Phase 1 gate imprecise (MEDIUM) | Gate wording split: conditional-branch fields stay default; `FuncCallLocation` identity fields populated; verify via explicit sub-test. |
| R3-F6 Nondeterministic `cond_id` assignment (MEDIUM) | 5b/5c spec updated: iterate `ModelLog.layer_list` order, collect keys in first-seen order via insertion-ordered dict (not set). |
| R3-F7 Docs miss user_funcs.py (LOW) | Phase 10 adds `user_funcs.py` docstring updates + README/API doc corrections. |
| R3 test gaps (MultiArm/NestedQualname/BranchEntry/SaveSrcOff/KeepUnsaved/RolledMixed) | All six test rows strengthened with concrete assertion lists; `SameLineNestedDefModel` added. |

## CHANGELOG v4 → v5 (addresses `review_round_4.md`)

| Finding | Disposition |
|---------|-------------|
| R4-F1 `source_loading_enabled` incomplete for full `FuncCallLocation` surface (HIGH) | D17 revised v2 + data model expanded to enumerate all 7 lazy accessors (`source_context`, `code_context`, `code_context_labeled`, `call_line`, `num_context_lines`, `func_signature`, `func_docstring`). When disabled, all lazy backing fields are initialized empty at construction (no disk access); `_frame_func_obj` is cleared at construction to prevent pickle fragility with nested/local functions. |
| R4-F2 `attribute_op` subsection contradicts D14 v2 (MEDIUM) | Algorithm subsection updated: no more "smallest containing scope" heuristic; step 3 fails closed (skips frame) matching D14 v2. |
| R4-F3 `source_context == []` type contradicts current `str` API (LOW) | `SaveSourceContextOffModel` assertions aligned to `str` types: `source_context == ""` etc. Pickle round-trip also added. |
| R4 test gap: disabled-source accessor coverage | Strengthened `SaveSourceContextOffModel` now asserts every lazy accessor + `_frame_func_obj is None` + pickle round-trip. |

## CHANGELOG v5 → v6 (addresses `review_round_5.md`)

| Finding | Disposition |
|---------|-------------|
| R5-F1 disabled-source contract contradicted current `code_context` API (MEDIUM) | D17 revised v3 + data-model comment + `SaveSourceContextOffModel` all re-aligned to the current "no source available" contract: `code_context=None`, `source_context="None"` (literal string), `code_context_labeled=""`, `call_line=""`, `num_context_lines=0`, `func_signature=None`, `func_docstring=None`, `_frame_func_obj=None`; `len==0`; `__getitem__` raises `IndexError`; `__repr__` ends with "code: source unavailable". No disk access when disabled. |

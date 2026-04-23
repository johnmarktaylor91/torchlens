# Sprint Plan: Full `if` / `elif` / `else` Branch Attribution for TorchLens

**Date:** 2026-04-22
**Branch:** `sprint/2026-04`
**Version:** v2 (post-adversarial-review-1)
**Change log:** See `CHANGELOG_v1_to_v2` at the bottom for findings addressed.

## Goals

1. Correctly attribute every captured op to its enclosing Python `if` / `elif` / `else` branch in eager `forward()` methods.
2. Robust false-positive filtering — distinguish actual branching bool events from incidental bool conversions (`assert`, `bool()`, comprehension filters, `while`, `ifexp`, `match` guards).
3. Represent branch structure explicitly in torchlens data classes (`LayerPassLog`, `LayerLog`, `ModelLog`).
4. Render THEN / ELIF / ELSE edge labels in the visualization (Graphviz primary; dagua bridge; ELK documented as unsupported).
5. Comprehensive test coverage for the positive path AND every documented limitation.
6. Strict backward compatibility — no existing field is renamed or removed.

## Out-of-Scope for v1 (explicitly documented limitations)

**Non-branch bool consumers — classified but not attributed:**
- Ternary (`x if cond else y`) → `bool_context_kind="ifexp"`
- `while cond:` → `bool_context_kind="while"`
- Comprehension filters → `bool_context_kind="comprehension_filter"`
- `match case ... if guard:` → `bool_context_kind="match_guard"`
- `assert tensor_cond` → `bool_context_kind="assert"`
- `bool(tensor)` cast NOT nested inside `If.test` → `bool_context_kind="bool_cast"`

**Note:** `bool(tensor_cond)` nested *inside* an `If.test`/`elif` test IS a branch. See D3 below.

**Source-unavailable situations — graceful degradation (no attribution, `bool_context_kind="unknown"`):**
- Jupyter / REPL cells (`<ipython-input-...>`)
- `exec` / `eval`-injected forwards (`<string>`)
- Byte-compiled only (`.pyc` without `.py`), native `.so` modules
- `torch.compile` / `torch.jit.script` / `torch.jit.trace` wrapped models
- `nn.DataParallel` / `DistributedDataParallel` (torchlens is already single-threaded)
- Monkey-patched `forward` where source resolves to the patch function, not the class body

**Branches torchlens fundamentally cannot see — documented false negatives:**
- Pure Python predicates: `if self.training:`, `if python_bool:`
- `if tensor.item() > 0:` — scalarization bypasses `Tensor.__bool__`
- Shape/metadata predicates: `if x.shape[0] > 0:`
- Functional conditionals: `torch.where`, `torch.cond`, masked blending

**Deferred to v2 (documented, not silently missed):**
- Column-offset disambiguation for multi-op lines
- Ternary (`ast.IfExp`) full attribution — data model extension
- `while` loop body attribution — different semantics
- ELK conditional edge rendering (legacy; dagua is replacing)

## Decisions (architect-level)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | AST + per-op frame inspection. NOT `sys.settrace`, NOT bytecode. | Convergent recommendation from all 3 research agents. Matches TorchScript + coverage.py prior art. Minimal runtime cost. |
| D2 | Flatten `if/elif/.../else` into ONE conditional record with branch arms `{then, elif_1, elif_2, ..., else}` | Python AST has no `Elif` node; elif is nested `If` in `orelse`. Flattening prevents line-based confusion. |
| **D3 (revised)** | **Pre-classify bools by enclosing AST context. For `bool(x)` nested inside an `If.test` or elif test: classify as `if_test`/`elif_test` with a secondary `bool_wrapper_kind="bool_cast"` marker. Standalone `bool(x)` (not in an If-test) classifies as `bool_cast`.** | Fixes adversarial Finding 2: `if bool(tensor_cond):` is a real branch and must not be suppressed. Wrapper context preserved separately. |
| D4 | Attribution uses **full `func_call_stack`**, not just the user-facing frame. | Catches helper functions: `def helper(): if cond: op()` called from forward. |
| D5 | `LayerPassLog.conditional_branch_stack` is the per-pass source of truth. `LayerLog` aggregate stores a **set of unique signatures** plus a pass-to-signature map. | Same rolled layer can be in THEN on pass 1, ELSE on pass 2. First-pass-wins is incorrect. |
| D6 | Keep every existing field (`cond_branch_start_children`, `cond_branch_then_children`, `conditional_branch_edges`, `conditional_then_edges`, `in_cond_branch`). Populate from new fields. | Strict backward compat. |
| D7 | AST indexing/caching lives in a NEW module `torchlens/postprocess/ast_branches.py`. Step 5 imports it. | Isolates AST logic from graph-flood logic; testable. |
| D8 | Cache key: `(filename, stat.st_mtime_ns)`. Invalidate on mtime change. Cache lifetime = process-local. | Hot-reload safety. |
| **D9 (revised)** | **AST classification runs whenever `(file, line)` is available on `func_call_stack` (which is always, even with `save_source_context=False`). `save_source_context` only gates rich source-text capture for user-facing display.** | Fixes adversarial Finding 1: D3+D12 require classification to always run; gating on `save_source_context` re-introduces false positives. (file, line) is already captured unconditionally via `FuncCallLocation`. |
| D10 | Conditional IDs: dense per-`ModelLog` integers. Structural key `(file, func_start, if_lineno, col_offset)` is an AST-index-internal detail; translated to dense ID during postprocess. | Simple external model. |
| D11 | Graphviz gets ELIF/ELSE labels with explicit precedence rules (see Visualization). Dagua bridge gets extended edge classification. ELK gets a comment documenting it as unsupported. | Dagua is active; ELK is legacy. |
| D12 | Old post-validation ("clear IF markings if no `ast.If`") is REMOVED. Pre-classification makes it unnecessary. | Cleaner algorithm. |
| **D13 (NEW)** | **Branch-entry edges are detected by diffing `conditional_branch_stack` across EVERY forward edge `(parent, child)`, NOT only edges out of branch-start nodes.** | Fixes adversarial Finding 3: branch-local ops may depend on parameters/buffers/constants (e.g. `y = self.bias + 1` inside `if`). The `self.bias→y` edge must be classified correctly even though `self.bias` is not in the predicate's dataflow. |
| **D14 (NEW)** | **Augment `FuncCallLocation` with `code_firstlineno: int` (captured via `frame.f_code.co_firstlineno`) and `code_qualname: str` (best-effort via `frame.f_code.co_qualname` on Python 3.11+, fallback to `co_name`). These identify a scope by its code-object identity, not just line number.** | Fixes adversarial Finding 9: nested helpers with the same name, lambdas, and decorated wrappers cannot be resolved by line alone. |
| **D15 (NEW)** | **Add `ModelLog.conditional_edge_passes` map: `(parent_no_pass_label, child_no_pass_label, cond_id, branch_kind) → List[int]` of pass numbers that took that arm on that rolled edge. Rolled-mode renderers consult this for mixed-arm edges and produce composite labels like `THEN(1,3) / ELSE(2)`.** | Fixes adversarial Finding 7: unioned child lists hide pass-level arm divergence in rolled mode. |

## Data Model Changes

### `FuncCallLocation` — augmented (D14)

```python
@dataclass
class FuncCallLocation:
    file: str
    line_number: int
    func_name: str
    code_firstlineno: int                 # NEW: frame.f_code.co_firstlineno
    code_qualname: Optional[str]          # NEW: frame.f_code.co_qualname on 3.11+, else None
    # existing lazy properties: source_context, func_signature, etc.
```

**Population:** `utils/introspection.py` stack-capture path. Both new fields are always captured; do NOT gate on `save_source_context` (per D9).

### `LayerPassLog` — new fields

```python
# Bool classification (meaningful when is_terminal_bool_layer=True and is_scalar_bool=True)
bool_is_branch: bool                      # True iff bool_context_kind ∈ {"if_test", "elif_test"}
bool_context_kind: Optional[str]          # one of: "if_test", "elif_test", "assert", "bool_cast",
                                          #          "comprehension_filter", "while", "ifexp",
                                          #          "match_guard", "unknown"
bool_wrapper_kind: Optional[str]          # NEW (D3 revision): e.g. "bool_cast" when classified as
                                          # if_test/elif_test but wrapped in a bool() call.
                                          # None if no wrapper.
bool_conditional_id: Optional[int]        # id into ModelLog.conditional_events (if_test/elif_test only)

# Branch attribution (meaningful for any op)
conditional_branch_stack: List[Tuple[int, str]]   # [(cond_id, branch_kind), ...], outer→inner
conditional_branch_depth: int                      # = len(conditional_branch_stack)

# Per-branch child label lists
cond_branch_else_children: List[str]
cond_branch_elif_children: Dict[int, List[str]]    # elif_index (1-based) → child labels
```

**Preserved (populated from new fields):**
- `is_terminal_bool_layer`, `is_scalar_bool`, `scalar_bool_value`, `in_cond_branch`
- `cond_branch_start_children` — still "edges where a branch is entered from this node"
- `cond_branch_then_children` — still "children in THEN body only"

### `LayerLog` — multi-pass aggregation

```python
# New aggregate fields
conditional_branch_stacks: List[List[Tuple[int, str]]]   # unique signatures in first-seen order
conditional_branch_stack_passes: Dict[
    Tuple[Tuple[int, str], ...], List[int]
]                                                         # signature → list of pass numbers

cond_branch_else_children: List[str]                     # union across passes (pass-stripped)
cond_branch_elif_children: Dict[int, List[str]]          # union across passes (pass-stripped)
```

**Merge rules** (`_build_layer_logs` update):
- `in_cond_branch`: OR across passes
- `conditional_branch_stacks`: unique signatures in first-seen order
- `conditional_branch_stack_passes`: accumulate
- `cond_branch_*_children`: UNION across passes after stripping `:N` pass suffix

### `ModelLog` — new fields

```python
conditional_events: List[ConditionalEvent]       # normalized if-chain records
conditional_elif_edges: List[Tuple[int, int, str, str]]  # (cond_id, elif_idx, parent, child)
conditional_else_edges: List[Tuple[int, str, str]]       # (cond_id, parent, child)

# NEW (D15): pass-level rolled-edge arm map
conditional_edge_passes: Dict[
    Tuple[str, str, int, str], List[int]
]                                                 # (parent_no_pass, child_no_pass, cond_id, branch_kind) → [pass_nums]
```

`ConditionalEvent` (dataclass or typed dict):
```python
@dataclass
class ConditionalEvent:
    id: int
    source_file: str
    function_qualname: str
    function_span: Tuple[int, int]
    if_stmt_span: Tuple[int, int]
    test_span: Tuple[int, int]
    nesting_depth: int
    branch_ranges: Dict[str, Tuple[int, int]]        # "then", "elif_1", ..., "else" → line range
    branch_test_spans: Dict[str, Tuple[int, int]]    # "then", "elif_1", ... → test span
    parent_conditional_id: Optional[int]
    parent_branch_kind: Optional[str]
    bool_layers: List[str]
```

**Preserved (populated from new fields):**
- `conditional_branch_edges` — still `(parent, bool_layer_label)` pairs
- `conditional_then_edges` — still `(parent, child)` for THEN edges only

### `constants.py` FIELD_ORDER updates

- `LAYER_PASS_LOG_FIELD_ORDER` — add 8 new entries (including `bool_wrapper_kind`)
- `MODEL_LOG_FIELD_ORDER` — add 4 new entries (including `conditional_edge_passes`)
- `LAYER_LOG_FIELD_ORDER` — add 5 new aggregate entries

All field additions done in ONE commit to prevent `FIELD_ORDER_SET` mismatch crashes.

## Algorithm

### New module: `torchlens/postprocess/ast_branches.py`

Public API:
```python
def get_file_index(filename: str) -> Optional["FileIndex"]: ...
def classify_bool(filename: str, line: int) -> BoolClassification: ...
def attribute_op(func_call_stack: List[FuncCallLocation]) -> List[Tuple[int, str]]: ...
def invalidate_cache(filename: Optional[str] = None) -> None: ...
```

**`FileIndex` contents per file:**
- Parsed `ast.Module`
- Function-scope list keyed by `(code_firstlineno, qualname)` (NOT by line span alone — D14)
- Normalized `ConditionalRecord` list (flattened if/elif/else chains)
- Bool-consumer index: every AST node that evaluates to bool (`If.test`, `While.test`, `IfExp.test`, `Assert.test`, comprehension `if`, `match_case.guard`, direct `bool(...)` calls) — each tagged with kind
- Per-function branch interval tree (centered interval tree keyed by line → list of `(cond_id, branch_kind, nesting_depth)`)

**`classify_bool(filename, line)` algorithm (D3 revision):**
1. Get `FileIndex`.
2. Walk bool consumers whose source span contains `line`, from **innermost outward**.
3. Determine the classification:
   - If the innermost non-`bool_cast` consumer is an `If.test` or flattened `elif` test → `kind = if_test` or `elif_test`, `wrapper_kind = "bool_cast"` if `bool_cast` is on the inner stack, else `None`.
   - If the innermost consumer IS `bool_cast` AND there's no enclosing `If.test`/`elif_test` → `kind = "bool_cast"`, `wrapper_kind = None`.
   - Otherwise use the innermost kind directly: `assert`, `comprehension_filter`, `while`, `ifexp`, `match_guard`.
   - If no consumer contains `line` → `kind = "unknown"`.
4. Return `BoolClassification(kind, wrapper_kind, maybe_cond_id, maybe_branch_test_kind)`.

**`attribute_op(func_call_stack)` algorithm:**
1. For each frame (shallow → deep):
   - Resolve scope by `(filename, code_firstlineno, func_name)` — exact match preferred; fallback to smallest containing scope by line (D14).
   - Query branch interval tree at `frame.line_number`.
2. Concatenate per-frame stacks, deduplicating adjacent repeats.
3. Return the merged stack.

### Postprocess Step 5 restructure (revised: D13)

**5a. Build file indexes**
For every internally-terminated scalar bool, collect source filenames across its `func_call_stack`. Eagerly build `FileIndex` for each.

**5b. Classify bools (D3 revision)**
For each `is_terminal_bool_layer` scalar bool:
- Call `classify_bool(file, line)` on each frame; take the innermost matching classification.
- Set `bool_context_kind`, `bool_wrapper_kind`, `bool_is_branch`, `bool_conditional_id`.
- If `bool_is_branch=True`, register in `ModelLog.conditional_events[cond_id].bool_layers`.

**5c. Emit conditional_events**
For each unique `bool_conditional_id`, materialize a `ConditionalEvent` from the `FileIndex` (translate structural key → dense `ModelLog`-local id).

**5d. IF edge backward flood** (PR #127 behavior, now source-filtered)
Starting from bools with `bool_is_branch=True` only, backward-flood through `parent_layers`; mark output-ancestors with `cond_branch_start_children` and emit `conditional_branch_edges`.

**5e. Branch attribution forward (D13 revision)**
For every op, compute `conditional_branch_stack` via `attribute_op(func_call_stack)`. Set `conditional_branch_stack`, `conditional_branch_depth`.

**Edge-level branch detection — inspect EVERY forward edge `(parent, child)`:**
- Compute `parent_stack`, `child_stack`.
- If `child_stack` has `(cond_id, kind)` entries that `parent_stack` doesn't → those are **branch-entry** edges.
  - For each gained `(cond_id, "then")` → add child to `parent.cond_branch_then_children`, emit `conditional_then_edges`.
  - For each gained `(cond_id, "elif_N")` → add child to `parent.cond_branch_elif_children[N]`, emit `conditional_elif_edges`.
  - For each gained `(cond_id, "else")` → add child to `parent.cond_branch_else_children`, emit `conditional_else_edges`.
- If `parent_stack` has entries that `child_stack` doesn't → **branch-exit / reconvergence**. No edge list is emitted (these edges are normal, non-labeled).
- Populate `conditional_edge_passes` (D15) keyed by pass-stripped labels.

**Note:** `cond_branch_start_children` (from 5d) and `cond_branch_then_children`/elif/else (from 5e) are related but distinct: 5d marks the parents of the predicate bool (IF edges in backward flood), 5e marks branch-entry edges (forward, stack-diff based). A node can participate in both if it sits at the divergence.

### Multi-pass `LayerLog` merge

Update `_build_layer_logs` per the rules in the Data Model section. Also populate `ModelLog.conditional_edge_passes` during this step.

## Visualization

### Label precedence rules (addresses adversarial Finding 8)

Edge-label composition, in priority order:

1. **Branch-entry label** (highest) — if the edge appears in any of:
   - `conditional_then_edges` → `THEN`
   - `conditional_elif_edges` (with elif_idx = N) → `ELIF N`
   - `conditional_else_edges` → `ELSE`
   - Multiple branch kinds across passes (rolled mode only, via `conditional_edge_passes`) → composite label `THEN(1,3) / ELSE(2)` or `mixed`.
   An edge with a branch-entry label does NOT simultaneously show `IF`.
2. **IF label** (second) — edge is in `conditional_branch_edges` (predicate-chain edge).
3. **Arg-label overlay** (compatible with above) — argument names (e.g., `arg0`) move from `label` to `headlabel` / `xlabel` so they never compete with branch labels.

In rolled mode, if `conditional_edge_passes` shows the edge took different arms across passes, use a composite label (per above). If a single arm across all passes, use the plain label.

### Graphviz (`torchlens/visualization/rendering.py`)

Extend the existing IF/THEN edge-label block (currently `rendering.py:985-996`):
- Move argument labels from `edge_dict["label"]` (currently lines 1049-1090) to `edge_dict["headlabel"]` or `edge_dict["xlabel"]`. This separates semantic concerns.
- Refactor the IF/THEN decision into a precedence function:
  ```python
  def _compute_branch_label(parent, child, edge_passes) -> Optional[str]: ...
  ```
- For rolled mode: consult `conditional_edge_passes` before emitting a plain label.

### Dagua bridge (`torchlens/visualization/dagua_bridge.py`)

Extend `_classify_forward_edge` with the same precedence. Returns `"then"`, `"elif_1"`, ..., `"else"`, or `"mixed"` for conflicting arms across passes. Dagua's rendering backend handles styling.

### ELK (`torchlens/visualization/elk_layout.py`)

Add a comment near the top:
```python
# ELK renderer does NOT render conditional branch edge labels (IF/THEN/ELIF/ELSE).
# For branch-aware visualization use vis_renderer="graphviz" (default) or
# the dagua renderer.
```

No functional change. Legacy.

## Capture / Postprocess Integration (addresses adversarial Finding 4)

Every new label-bearing field must be wired through the full field lifecycle:

1. **Capture-time defaults** (`torchlens/capture/source_tensors.py`, `torchlens/capture/output_tensors.py`):
   Add init defaults for ALL new `LayerPassLog` fields (empty list, False, None).

2. **Raw→final rename** (`torchlens/postprocess/labeling.py`):
   Extend `_rename_raw_labels_in_place` to also rewrite labels in:
   - `cond_branch_else_children` (each list member)
   - `cond_branch_elif_children` (each value's list members)
   - `conditional_elif_edges` (positions 2, 3)
   - `conditional_else_edges` (positions 1, 2)
   - `conditional_edge_passes` (dict keys — positions 0, 1 of the tuple)
   - `conditional_events[*].bool_layers`

3. **Cleanup / keep_unsaved_layers=False** (`torchlens/data_classes/cleanup.py`):
   Extend the conditional-edge filtering logic to prune the new edge lists when referenced labels are removed.

4. **Export** (`torchlens/data_classes/interface.py::to_pandas`):
   Add columns for `conditional_branch_depth`, `bool_is_branch`, `bool_context_kind`, `bool_wrapper_kind`, `bool_conditional_id`, and a compact string rendering of `conditional_branch_stack`.

**Phase 1 gate**: every new field must be verified wired through rename AND cleanup before Phase 1 is declared green. Add an integration test with `keep_unsaved_layers=False` to catch missed wiring.

## Validation / Invariants (`torchlens/validation/invariants.py`)

Add consistency checks (run in `validate_forward_pass`):

1. **cond_branch_X_children ↔ conditional_X_edges bidirectional consistency**
   For each `(parent, child)` in `conditional_then_edges`, `child ∈ parent.cond_branch_then_children`. Same for elif and else.
2. **Children labels exist in ModelLog**
   Every label in `cond_branch_*_children` corresponds to an actual `LayerPassLog`.
3. **Bool classification invariants**
   `bool_is_branch=True` ⟺ `bool_context_kind ∈ {"if_test", "elif_test"}`.
   `bool_is_branch=True` ⟹ `bool_conditional_id is not None`.
   `bool_context_kind is not None` ⟹ layer has `is_terminal_bool_layer=True`.
   `bool_wrapper_kind is not None` ⟹ `bool_context_kind in {"if_test", "elif_test", "bool_cast"}`.
4. **Stack-depth self-consistency**
   `conditional_branch_depth == len(conditional_branch_stack)`.
   `in_cond_branch == (conditional_branch_depth > 0)`.
5. **Conditional event references are valid**
   Every `cond_id` in any stack entry, `bool_conditional_id`, or edge list has a matching `ModelLog.conditional_events[i].id`.
6. **Branch stack monotonicity (REVISED — Finding 5)**
   For every parent→child edge, ONE of the two stacks is a prefix of the other:
   - child_stack extends parent_stack → branch-entry
   - parent_stack extends child_stack → branch-exit / reconvergence
   - stacks equal → inside same branch level
   Rejecting both directions (as v1 did) is wrong because reconvergence is legal.
7. **Elif index contiguity (REVISED — Finding 6 — moved to ConditionalEvent)**
   For each `ConditionalEvent`, `branch_ranges` keys for elif arms are contiguous from `elif_1`. `cond_branch_elif_children` per-node map MAY have sparse keys (legitimate — a parent may only have children in `elif_2`).
8. **Rolled-edge pass consistency (NEW)**
   Every entry in `conditional_edge_passes` must reference a real rolled edge in the unrolled graph.

## Test Matrix

### New file: `tests/test_conditional_branches.py`

Test models live in `tests/example_models.py` (or a new `example_conditional_models.py`).

All tests run `log_forward_pass(..., save_source_context=True)` unless otherwise noted. Classification is expected to work even with `save_source_context=False`; that's verified explicitly by two tests.

**Baseline / positive:**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `SimpleIfElseModel` | `if/else` | THEN + ELSE edges, `conditional_events` has 1 entry |
| `ElifLadderModel` | `if/elif/elif/else` | ELIF 1, ELIF 2, ELSE; `branch_ranges` has all 4 arms |
| `NestedIfThenIfModel` | `if A: if B:` | 2 events, stack depth 2 inside inner body |
| `NestedInElseModel` | `if A: ...else: if B:` | 2 events, inner's `parent_branch_kind="else"` |
| `MultilinePredicateModel` | `if (a \n and b \n):` | test span 3 lines, resolves to one conditional |

**Branch-entry via non-predicate ancestors (Finding 3):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `BranchUsesOnlyParameterModel` | `if c: y = self.bias + 1` | `self.bias → y` edge labeled THEN; `y ∈ parent.cond_branch_then_children` |
| `BranchUsesOnlyConstantModel` | `if c: y = x * 2.0` (2.0 is constant) | similar — branch-entry edge labeled |

**Wrapped bool (Finding 2):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `IfBoolCastModel` | `if bool(x.sum() > 0):` | `bool_is_branch=True`, `bool_context_kind="if_test"`, `bool_wrapper_kind="bool_cast"`, THEN edges emitted |

**Multi-pass / recurrent (Finding 7):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `LoopedIfAlternatingModel` | `for i: if i%2: ...` | per-pass stacks differ; aggregate has 2 signatures |
| `AlternatingRecurrentIfModel` | recurrent, THEN pass 1 / ELSE pass 2 | `conditional_branch_stacks` has both; `conditional_branch_stack_passes` maps correctly; `conditional_edge_passes` shows pass divergence on rolled edges |
| `RolledMixedArmModel` | same rolled edge takes THEN one pass and ELSE another | rolled-mode renderer emits composite label |

**Reconvergence (Finding 5):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `ReconvergingBranchesModel` | `if c: x=a else: x=b; y=f(x)` | invariant 6 passes (prefix-drop allowed); `y` has empty stack but parents `a`, `b` had non-empty stacks |

**Scope resolution (Finding 9):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `NestedHelperSameNameModel` | two local `helper()` with same name in different branches | attribution uses `code_firstlineno`, not line-only |
| `LambdaBranchModel` | `lambda x: op(x) if cond else op2(x)` in forward | lambda scope resolved distinctly |
| `DecoratedForwardModel` | `@decorator` on forward | scope resolves to real forward, not decorator wrapper; best-effort |

**False positives — non-branch bools:**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `AssertTensorCondModel` | `assert tensor_cond` | `bool_context_kind="assert"`, `bool_is_branch=False`, no IF/THEN edges |
| `BoolCastOnlyModel` | `flag = bool(tensor)` (no enclosing if) | `bool_context_kind="bool_cast"`, `bool_is_branch=False` |
| `TernaryIfExpModel` | `y = a if cond else b` | `bool_context_kind="ifexp"`, no edges |
| `ComprehensionIfModel` | `[f(x) for x in xs if pred(x)]` | `bool_context_kind="comprehension_filter"`, no edges |
| `WhileLoopModel` | `while tensor_cond:` | `bool_context_kind="while"`, no edges |
| `MatchGuardModel` | `match v: case x if tensor_cond: ...` | `bool_context_kind="match_guard"`, no edges |

**Compound / negation / walrus:**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `NotIfModel` | `if not cond:` | `bool_context_kind="if_test"`, THEN attribution correct |
| `AndOrIfModel` | `if a and b:` | multiple bools share same `cond_id`, both `bool_is_branch=True` |
| `WalrusIfModel` | `if (x := expr) > 0:` | normal attribution |

**Documented false negatives (no attribution expected):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `PythonBoolModel` | `if self.flag:` (no tensor) | no bool events captured, no edges |
| `ItemScalarizationModel` | `if tensor.item() > 0:` | no bool events captured (documented limitation) |
| `TorchWhereModel` | `x = torch.where(cond, a, b)` | no conditional_events |
| `ShapePredicateModel` | `if x.shape[0] > 0:` | no conditional_events |

**save_source_context gating (Finding 1):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `SaveSourceContextOffModel` | `save_source_context=False` | AST classification STILL runs (because only file+line needed); branch attribution populated; source_context field empty |
| `SaveSourceContextOffAssertModel` | `save_source_context=False`, `assert tensor_cond` | classification works; no false-positive IF edge emitted |

**Field-lifecycle integration (Finding 4):**
| Test | Purpose | Assertions |
|------|---------|-----------|
| `KeepUnsavedLayersFalseModel` | `keep_unsaved_layers=False` with conditionals | new edge lists properly pruned; no dangling labels |
| `ToPandasConditionalModel` | `to_pandas()` with conditionals | DataFrame has conditional columns populated |

**Smoke:**
Add `@pytest.mark.smoke` to `SimpleIfElseModel` and `ElifLadderModel`.

## Backward Compatibility

- Every existing field retained with identical semantics. Verify by running the existing test suite post-implementation.
- Old fields populated from new. `in_cond_branch` remains a real field (for serialization compat) but is recomputed from `conditional_branch_stack`.
- `FIELD_ORDER` tuples updated in same commit as class definitions.
- Pickle compat: `__setstate__` default-fill on each affected class. Missing new fields get empty defaults. Document as "load-with-defaults for 1 prior minor version."

## Implementation Phases

1. **Foundation** — new fields on all classes + `constants.py` updates + `__init__` defaults + capture-time defaults (`source_tensors.py`, `output_tensors.py`) + `FuncCallLocation` augmentation with `code_firstlineno`/`code_qualname`. Update `utils/introspection.py` stack-capture. Run tier-1 smoke; verify no crashes. New fields stay empty.
2. **AST module** — `ast_branches.py` with file indexing, classification, attribution. Unit tests on synthetic source snippets (no model needed). Cover every edge case in the classifier.
3. **Postprocess integration** — restructure Step 5 into 5a-5e per spec. Wire classification and edge-level attribution. Remove post-validation (D12).
4. **Multi-pass merge + rolled edges** — `_build_layer_logs` updates + `conditional_edge_passes` population.
5. **Rename / cleanup / export wiring** — `labeling.py`, `cleanup.py`, `interface.py::to_pandas`. Integration test with `keep_unsaved_layers=False`.
6. **Invariants** — add 8 new consistency checks.
7. **Graphviz rendering** — label precedence function; move arg labels to headlabel; ELIF/ELSE support; rolled-mode composite labels.
8. **Dagua bridge** — extend edge classification.
9. **Integration tests** — all ~30 models with assertions.
10. **Documentation** — update `AGENTS.md` conditional section, `architecture.md`, user-facing limitations page.
11. **PR** — with migration notes.

Each phase: tier-1 smoke before moving on. Tier-2 at phases 3, 5, 7, 9. Tier-3 before PR.

## Deliverables

- 1 new module: `torchlens/postprocess/ast_branches.py` (~400 lines)
- Modified modules (~10):
  - `torchlens/data_classes/layer_pass_log.py`
  - `torchlens/data_classes/layer_log.py`
  - `torchlens/data_classes/model_log.py`
  - `torchlens/data_classes/func_call_location.py` (augmentation)
  - `torchlens/data_classes/cleanup.py` (new field filtering)
  - `torchlens/data_classes/interface.py` (`to_pandas` updates)
  - `torchlens/constants.py` (FIELD_ORDER)
  - `torchlens/postprocess/control_flow.py` (Step 5 restructure)
  - `torchlens/postprocess/labeling.py` (rename new fields)
  - `torchlens/capture/source_tensors.py` (init defaults)
  - `torchlens/capture/output_tensors.py` (init defaults)
  - `torchlens/utils/introspection.py` (augmented stack capture)
  - `torchlens/validation/invariants.py` (new checks)
  - `torchlens/visualization/rendering.py` (label precedence + new labels)
  - `torchlens/visualization/dagua_bridge.py` (edge classification)
  - `torchlens/visualization/elk_layout.py` (comment only)
- Tests: `tests/test_conditional_branches.py` (~500 lines), `tests/example_models.py` additions (~30 new small models)
- Docs: AGENTS.md, architecture.md, user-facing limitations page
- PR with migration notes

## Risks (tracked)

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | `FIELD_ORDER` mismatch on partial rollout | All field additions in ONE commit; full test suite before push |
| R2 | Multi-pass merge regressions on recurrent tests | `AlternatingRecurrentIfModel` in early phases; tier-2 suite |
| R3 | Source-line resolution fragile for decorated/wrapped forwards | Use `frame.f_code.co_filename` + `co_firstlineno` (D14); don't rely on `inspect.getsourcefile(module_class)` |
| R4 | `linecache` serves stale source after edit | Cache by `(filename, mtime)`; on mtime change invalidate; fail-soft to `unknown` on mismatch |
| R5 | Pre-classification misses a syntactic form → silent `unknown` for real `if_test` | Dev-mode assertion: if line is inside `If.test` span but classifier returns `unknown`, log warning. Unit tests for each enumerated kind. |
| R6 | Interval tree perf regresses on huge `forward` | One-time file parse; O(log k + d) query; benchmark `HugeNestedIfModel` |
| R7 | Old pickled ModelLogs fail to load | `__setstate__` default-fill; document window |
| R8 | Arg-label relocation to headlabel breaks existing visual output | Gate behind a flag, OR verify with visual diff on existing aesthetic tests |
| R9 | Edge-level attribution (D13) produces too many edge entries on dense branch bodies | Benchmark; batch emit |

## Open Questions for Architect / User

1. **ELK support**: Default = document limitation. Confirm?
2. **conditional_id**: Default = dense per-`ModelLog` int. Confirm vs structural key?
3. **Pickle compat window**: 1 minor version (default). OK?
4. **v2 scope confirmation**: ternary + while + column offsets + ELK parity → v2?
5. **Test model count**: ~30 models. OK?

---

## CHANGELOG_v1_to_v2

Addresses adversarial review round 1 (`review_round_1.md`):

| Finding | Severity | Fix in v2 |
|---------|----------|-----------|
| F1 D3/D9/D12 contradiction | blocker | D9 revised: AST classification always runs when `(file, line)` available; `save_source_context` only gates rich text. |
| F2 `if bool(tensor_cond):` misclassified | high | D3 revised + new `bool_wrapper_kind` field; wrapped bools normalize to `if_test` with wrapper marker. |
| F3 Step 5e too narrow | blocker | New D13: branch-entry detection on ALL forward edges, not just branch-start children. |
| F4 Rename/cleanup/export not wired | high | New "Capture/Postprocess Integration" section; Phase 1 gate; modified-modules list expanded. |
| F5 Invariant 6 rejects reconvergence | high | Invariant 6 revised: allow either direction of prefix. |
| F6 Invariant 7 on wrong structure | medium | Invariant 7 moved from per-node map to `ConditionalEvent`. |
| F7 Rolled-mode loses pass arms | high | New D15: `ModelLog.conditional_edge_passes`; visualization precedence adds composite labels. |
| F8 Viz precedence unspecified | medium | New "Label precedence rules" section; arg labels move to headlabel; refactor existing label-overwrite code. |
| F9 FuncCallLocation too weak | high | New D14: augment with `code_firstlineno` + `code_qualname`; scope resolution by identity first, line second. |

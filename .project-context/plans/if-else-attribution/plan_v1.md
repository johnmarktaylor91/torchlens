# Sprint Plan: Full `if` / `elif` / `else` Branch Attribution for TorchLens

**Date:** 2026-04-22
**Branch:** `sprint/2026-04`
**Status:** draft v1 (pre-adversarial)

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
- `bool(tensor)` cast → `bool_context_kind="bool_cast"`

**Source-unavailable situations — graceful degradation (no attribution):**
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
- Column-offset disambiguation for multi-op lines (`x = f(g(h(a)))` with multiple ops in one branch)
- Ternary (`ast.IfExp`) full attribution — would extend data model; v1 classifies only
- `while` loop body attribution — different semantics from one-shot branching
- ELK conditional edge rendering (ELK is legacy; dagua is replacing)

## Decisions (architect-level)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | AST + per-op frame inspection. NOT `sys.settrace`, NOT bytecode. | Convergent recommendation from all 3 codex agents. Matches TorchScript + coverage.py prior art. Minimal runtime cost. |
| D2 | Flatten `if/elif/.../else` into ONE conditional record with branch arms `{then, elif_1, elif_2, ..., else}` | Python AST has no `Elif` node; elif is nested `If` in `orelse`. Flattening prevents line-based confusion between top-level `if` and inner `elif`. |
| D3 | Pre-classify all terminal scalar bools by AST context BEFORE Step 5 flood. Only `if_test`/`elif_test` participate in branch marking. | Replaces current "mark first, clear later" pattern. Cleaner, fewer false edges, reusable classification. |
| D4 | Attribution uses **full `func_call_stack`**, not just the user-facing frame. | Catches helper functions: `def helper(): if cond: op()` called from forward. |
| D5 | `LayerPassLog.conditional_branch_stack` is the per-pass source of truth. `LayerLog` aggregate stores a **set of unique signatures** plus a pass-to-signature map. | Same rolled layer can be in THEN on pass 1, ELSE on pass 2. First-pass-wins aggregation is incorrect. |
| D6 | Keep every existing field (`cond_branch_start_children`, `cond_branch_then_children`, `conditional_branch_edges`, `conditional_then_edges`, `in_cond_branch`). Populate from new fields. | Strict backward compat — downstream code consuming these stays working. `in_cond_branch` becomes a derived property (`len(conditional_branch_stack) > 0`). |
| D7 | AST indexing/caching lives in a NEW module `torchlens/postprocess/ast_branches.py`. Step 5 (`_mark_conditional_branches`) imports it. | Isolates file-parsing / AST-node logic from graph-flood logic. Testable in isolation. |
| D8 | Cache key: `(filename, stat.st_mtime_ns)`. Invalidate on mtime change; always reparse on mismatch. Cache lifetime = process-local. | Matches design recommendation. Hot-reload safety. |
| D9 | AST attribution activation: gated on `save_source_context=True` (same precondition as current PR #127 behavior). | `(file, line)` is already captured on every op via `func_call_stack` even when `save_source_context=False`, but we conservatively keep the gate to avoid surprising new work in the default no-source-context path. Revisit in v2 if user asks. |
| D10 | Conditional IDs: dense per-`ModelLog` integers. A per-file structural key lives INSIDE the AST index; translates to `ModelLog`-local ID during postprocess. | Simpler data model for users. Structural key is an implementation detail. |
| D11 | Graphviz gets full ELIF/ELSE edge labels. Dagua bridge gets extended edge classification. ELK gets a one-line comment stating it does not render conditional labels. | Dagua is the active visualization path; ELK is legacy; the user's time is better spent on the current and future renderers. |
| D12 | Old post-validation step (clear IF markings if no `ast.If` found) is REMOVED. Pre-classification makes it unnecessary. | Cleaner algorithm, no double-checking. |

## Data Model Changes

### `LayerPassLog` — new fields

```python
# Bool classification (meaningful when is_terminal_bool_layer=True and is_scalar_bool=True)
bool_is_branch: bool                      # True iff bool_context_kind ∈ {"if_test", "elif_test"}
bool_context_kind: Optional[str]          # one of: "if_test", "elif_test", "assert", "bool_cast",
                                          #          "comprehension_filter", "while", "ifexp",
                                          #          "match_guard", "unknown"
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
- `in_cond_branch`: OR across passes (if ANY pass has non-empty stack)
- `conditional_branch_stacks`: unique signatures in first-seen order
- `conditional_branch_stack_passes`: accumulate
- `cond_branch_*_children`: UNION across passes after stripping `:N` pass suffix

### `ModelLog` — new fields

```python
conditional_events: List[ConditionalEvent]       # list of normalized if-chain records
conditional_elif_edges: List[Tuple[int, int, str, str]]  # (cond_id, elif_idx, parent, child)
conditional_else_edges: List[Tuple[int, str, str]]       # (cond_id, parent, child)
```

where `ConditionalEvent` is a dataclass (or typed dict):
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
    bool_layers: List[str]                            # labels of atomic bools that drove this event
```

**Preserved (populated from new fields):**
- `conditional_branch_edges` — still `(parent, bool_layer_label)` pairs
- `conditional_then_edges` — still `(parent, child)` for THEN edges only

### `constants.py` FIELD_ORDER updates

- `LAYER_PASS_LOG_FIELD_ORDER` — add 7 new entries in the "Conditional info" block
- `MODEL_LOG_FIELD_ORDER` — add 3 new entries
- `LAYER_LOG_FIELD_ORDER` — add 5 new aggregate entries (conditional fields now properly tracked, not "hidden")

All field additions must be done in one PR to prevent `FIELD_ORDER_SET` mismatch crashes.

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
- Function-scope list (`FunctionDef`, `AsyncFunctionDef`, `Lambda` with usable spans)
- Normalized `ConditionalRecord` list (flattened if/elif/else chains)
- Bool-consumer index: every `ast.If.test`, `ast.While.test`, `ast.IfExp.test`, `ast.Assert.test`, comprehension `if` filters, `match_case.guard`, direct `bool(...)` calls — each tagged with kind
- Per-function branch interval tree (centered interval tree keyed by line → list of `(cond_id, branch_kind, nesting_depth)`)

**`classify_bool(filename, line)` algorithm:**
1. Get `FileIndex` (parse + cache if new).
2. Find innermost AST bool consumer whose source span contains `line`.
3. Return `(kind, maybe_cond_id, maybe_branch_test_kind)` where `kind` is one of the 9 context kinds above.

**`attribute_op(func_call_stack)` algorithm:**
1. For each frame (shallow → deep): look up scope, query branch interval tree at `frame.line_number`.
2. Concatenate per-frame stacks, deduplicating adjacent repeats.
3. Return the merged stack.

### Postprocess Step 5 restructure

Rename the current monolithic `_mark_conditional_branches` into 5 sub-phases:

**5a. Build file indexes**
For every internally-terminated scalar bool, collect source filenames across its `func_call_stack`. Eagerly build `FileIndex` for each (amortizes cache cost).

**5b. Classify bools**
For each `is_terminal_bool_layer` scalar bool:
- Call `classify_bool(file, line)` on each frame; take the innermost non-`unknown` kind.
- Set `bool_context_kind`, `bool_is_branch`, `bool_conditional_id`.
- If kind is `if_test` or `elif_test`, register this bool in `ModelLog.conditional_events[cond_id].bool_layers`.

**5c. Emit conditional_events**
For each unique `bool_conditional_id`, materialize a `ConditionalEvent` from the `FileIndex` (translate structural key → dense `ModelLog`-local id).

**5d. IF edge backward flood** (unchanged from PR #127, but now source-selected)
Starting from bools with `bool_is_branch=True` only, backward-flood through `parent_layers`; mark output-ancestors with `cond_branch_start_children` and emit `conditional_branch_edges`.

**5e. Branch attribution forward**
For every op, call `attribute_op(func_call_stack)`. Set `conditional_branch_stack`, `conditional_branch_depth`.

For each branch-start node's children, diff parent vs child stack to determine which branch was entered:
- If child gained `(cond_id, "then")`: add to `cond_branch_then_children`, emit `conditional_then_edges`.
- If child gained `(cond_id, "elif_N")`: add to `cond_branch_elif_children[N]`, emit `conditional_elif_edges`.
- If child gained `(cond_id, "else")`: add to `cond_branch_else_children`, emit `conditional_else_edges`.

### Multi-pass `LayerLog` merge

Update `_build_layer_logs` to compute aggregate conditional fields per the rules in the Data Model section.

## Visualization

### Graphviz (`torchlens/visualization/rendering.py`)

Extend the existing IF/THEN edge-label block at lines 985-996:
- Match child against `cond_branch_elif_children[n]` → label `ELIF n`
- Match child against `cond_branch_else_children` → label `ELSE`
- Respect rolled mode (strip `:N` suffix)

Same font size/bold/underline as existing IF/THEN labels.

### Dagua bridge (`torchlens/visualization/dagua_bridge.py`)

Extend `_classify_forward_edge`:
- Return `"elif_1"`, `"elif_2"`, ... for elif child matches
- Return `"else"` for else child matches

Dagua's rendering backend handles the styling; no changes needed there.

### ELK (`torchlens/visualization/elk_layout.py`)

Add a single comment near the top documenting:
```python
# ELK renderer does NOT render conditional branch edge labels (IF/THEN/ELIF/ELSE).
# For branch-aware visualization use vis_renderer="graphviz" (default) or
# the dagua renderer.
```

No functional change. ELK is legacy (dagua replacement in progress).

## Validation / Invariants (`torchlens/validation/invariants.py`)

Add consistency checks (existing invariant pattern, run in `validate_forward_pass`):

1. **cond_branch_X_children ↔ conditional_X_edges bidirectional consistency**
   For each `(parent, child)` in `conditional_then_edges`, `child ∈ parent.cond_branch_then_children`.
   Same for elif and else.
2. **Children labels exist in ModelLog**
   Every label in `cond_branch_*_children` corresponds to an actual `LayerPassLog`.
3. **Bool classification invariants**
   `bool_is_branch=True` ⟺ `bool_context_kind ∈ {"if_test", "elif_test"}`.
   `bool_is_branch=True` ⟹ `bool_conditional_id is not None`.
   `bool_context_kind is not None` ⟹ layer has `is_terminal_bool_layer=True`.
4. **Stack-depth self-consistency**
   `conditional_branch_depth == len(conditional_branch_stack)`.
   `in_cond_branch == (conditional_branch_depth > 0)`.
5. **Conditional event references are valid**
   Every `cond_id` in any `conditional_branch_stack` entry, any `bool_conditional_id`, or any edge list has a matching `ModelLog.conditional_events[i].id`.
6. **Branch stack monotonicity**
   For a parent→child edge, the child's stack contains the parent's stack as a prefix (or enters a new branch, which is the edge-classification case).
7. **Elif index contiguity**
   For a given `cond_id`, `cond_branch_elif_children` keys are contiguous from 1.

## Test Matrix

### New file: `tests/test_conditional_branches.py`

Test models added to `tests/example_models.py` (or a new `example_conditional_models.py` if that file grows too large).

Each test runs `log_forward_pass(..., save_source_context=True)` and asserts on specific fields.

| Test Model | Purpose | Assert |
|------------|---------|--------|
| `SimpleIfElseModel` | baseline `if/else` | THEN + ELSE edges present, `conditional_events` has 1 entry |
| `ElifLadderModel` | `if/elif/elif/else` | ELIF 1, ELIF 2, ELSE edges; `branch_ranges` has all 4 arms |
| `NestedIfThenIfModel` | `if A: if B: ...` | 2 conditional_events, stack depth 2 inside inner body |
| `NestedInElseModel` | `if A: ... else: if B: ...` | 2 events, inner nested under `else` branch kind |
| `MultilinePredicateModel` | `if (a \n and b \n):` | test span spans 3 lines, resolves to one conditional |
| `LoopedIfAlternatingModel` | `for i: if i%2: ...` | per-pass `conditional_branch_stack` differs; aggregate has 2 signatures |
| `AlternatingRecurrentIfModel` | recurrent, THEN on pass 1, ELSE on pass 2 | `LayerLog.conditional_branch_stacks` has both; `conditional_branch_stack_passes` maps correctly |
| `EarlyReturnIfModel` | `if cond: return x` | THEN child marked; ops after if aren't attributed (there are none on this path) |
| `HelperBranchModel` | helper fn called from forward contains the `if` | attribution uses helper's frame, not forward's |
| `NotIfModel` | `if not cond:` | bool classified as `if_test`, THEN attribution works |
| `AndOrIfModel` | `if a and b:` | multiple bools share same `cond_id`, both `bool_is_branch=True` |
| `WalrusIfModel` | `if (x := expr) > 0:` | normal attribution |
| `AssertTensorCondModel` | `assert tensor_cond` | `bool_context_kind="assert"`, `bool_is_branch=False`, no IF/THEN edges emitted |
| `BoolCastOnlyModel` | `flag = bool(tensor)` | `bool_context_kind="bool_cast"`, no edges |
| `TernaryIfExpModel` | `y = a if cond else b` | `bool_context_kind="ifexp"`, no edges |
| `ComprehensionIfModel` | `[f(x) for x in xs if pred(x)]` | `bool_context_kind="comprehension_filter"`, no edges |
| `WhileLoopModel` | `while tensor_cond:` | `bool_context_kind="while"`, no edges |
| `PythonBoolModel` | `if self.flag:` (no tensor) | no bool events captured, no edges |
| `ItemScalarizationModel` | `if tensor.item() > 0:` | no bool events captured, no edges (documented false negative) |
| `TorchWhereModel` | `x = torch.where(cond, a, b)` | no conditional_events |
| `ShapePredicateModel` | `if x.shape[0] > 0:` | no conditional_events (documented false negative) |
| `ReconvergingBranchesModel` | `if cond: x=a else: x=b; y=f(x)` | `y` outside branches, x-children correctly attributed to THEN vs ELSE |
| `DecoratedForwardModel` | `@decorator` on forward | best-effort; either graceful degradation or correct attribution |
| `SaveSourceContextOffModel` | `save_source_context=False` | no branch attribution runs; `conditional_branch_stack=[]` on all ops; existing `in_cond_branch` flood still works for IF edges |

### Smoke coverage

Add 1-2 tests with `@pytest.mark.smoke` — a simple `if/else` model + an `if/elif/else` ladder. Both assert on new fields.

## Backward Compatibility

- Every existing field retained with identical semantics.
- Old fields populated from new. `in_cond_branch` is now derived (`@property`) but can stay as a direct field for serialization compat; updated per pass via attribution.
- `FIELD_ORDER` tuples updated in the same PR as class definitions.
- Pickle compat: a loader-side default fill (`__setstate__` or `__post_init__` that fills missing fields with empty defaults) lets older pickles load into new classes. Document as "load-with-defaults for N-1 minor version only."

## Implementation Phases

1. **Foundation** — new fields + `constants.py` updates + default `__init__` values. Run smoke tests. No behavior change yet (new fields stay empty).
2. **AST module** — `ast_branches.py` + unit tests (Test the indexer, classifier, attributor in isolation on synthetic source snippets).
3. **Postprocess integration** — restructure Step 5 into 5a-5e. Wire in classification and attribution.
4. **Multi-pass merge** — `_build_layer_logs` updates.
5. **Invariants** — add 7 new consistency checks.
6. **Graphviz rendering** — extend edge labels.
7. **Dagua bridge** — extend edge classification.
8. **Integration tests** — all 24 models with assertions.
9. **Documentation** — update `AGENTS.md` conditional section, architecture.md, a user-facing "conditional handling limitations" page.
10. **PR** — with explicit migration notes for any downstream consumers.

Each phase: pass tier-1 smoke tests before moving to the next. Tier-2 test suite at phases 3, 5, and 8. Tier-3 before PR.

## Open Questions for Architect (me) / User

(Listed here for adversarial-review input and optional user input. I'll make defaults but surface for override.)

1. **ELK support**: Default = document limitation. Confirm?
2. **conditional_id**: Default = dense per-`ModelLog` int. Confirm vs structural key?
3. **Save-source-context gate**: Default = keep gate (PR #127 convention). Should AST attribution activate unconditionally instead?
4. **Pickle version compat**: Default = "load-with-defaults for 1 minor version." Acceptable, or stricter?
5. **v2 scope**: confirm ternary + while + column offsets + ELK go to v2?
6. **Test model count**: 24 is aggressive. Acceptable, or should we trim?

## Deliverables

- 1 new module: `torchlens/postprocess/ast_branches.py` (~300 lines)
- Modified modules (~5): `layer_pass_log.py`, `layer_log.py`, `model_log.py`, `constants.py`, `control_flow.py`, `_build_layer_logs`, `validation/invariants.py`, `visualization/rendering.py`, `visualization/dagua_bridge.py`
- Test additions: `tests/test_conditional_branches.py` (~400 lines), `tests/example_models.py` additions (~24 new small models)
- Docs: `AGENTS.md` conditional section update, `architecture.md` Step 5 update, user-facing limitations page in docs/
- PR with migration notes and explicit "Before/After" table of conditional-related fields

## Risks (tracked)

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | `FIELD_ORDER` mismatch causes hard crashes on any partial rollout | Do all field additions in ONE commit; run full test suite before pushing |
| R2 | Multi-pass merge semantics cause subtle regressions on recurrent tests | Add `AlternatingRecurrentIfModel` test early; run tier-2 suite |
| R3 | Source-line resolution for decorated/wrapped forwards produces wrong `FileIndex` lookups | Require frame source via `frame.f_code.co_filename` + `f_lineno`, not `inspect.getsourcefile(module_class)` |
| R4 | `linecache` serves stale source after file edit, AST parses new; mismatch produces confident-wrong attribution | Cache by `(filename, mtime)`; on mtime change invalidate; on `linecache` miss fail-soft to `bool_context_kind="unknown"` |
| R5 | Pre-classification misses a syntactic form and silently emits `unknown` for a real `if_test` | Explicit unit tests for every enumerated bool-consumer kind; strict `raise` (in dev mode) when classifier returns `unknown` but line is syntactically inside an `If.test` |
| R6 | Interval tree perf regresses on 1000-line `forward` | Parse once per file; interval tree query is `O(log k + d)`; benchmark on `HugeNestedIfModel` |
| R7 | Old pickled ModelLogs fail to load | `__setstate__` default-fill; document load-with-defaults window |

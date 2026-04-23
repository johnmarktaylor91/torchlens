# AST Design for Full `if` / `elif` / `else` Attribution

## Context

TorchLens already has the raw ingredients for Python-level branch attribution:

- scalar bool tensor ops are detectable (`is_scalar_bool`, `is_terminal_bool_layer`)
- every relevant op can be associated with user source locations via `FuncCallLocation`
- Step 5 already marks conditional subgraphs and has partial AST-based THEN detection
- visualization already understands graph-level conditional edge metadata

Current coverage is intentionally narrow:

- `torchlens/postprocess/control_flow.py` backtracks from terminal bool tensors to mark `in_cond_branch`
- AST use is limited to detecting whether a child edge lands in the root `if` body
- only THEN is modeled explicitly (`cond_branch_then_children`, `conditional_then_edges`)
- false positives are cleaned up after the fact by clearing IF markings when no `ast.If` is found
- `LayerLog` currently inherits branch metadata from the first pass only, which is insufficient for rolled graphs

This document proposes a full attribution layer for `if` / `elif` / `else`, designed to sit on top of the existing Step 5 flow without changing the rest of TorchLens's graph semantics.

## 1. Core Algorithm For AST Branch Discovery

### Goal

Given `(source_file, bool_line_number)`, locate the enclosing `ast.If` whose `test` expression contains `bool_line_number`, flatten any `elif` chain, and emit a normalized conditional record with explicit THEN / ELIF / ELSE branch ranges.

### Key Design Choice

Treat one Python `if` / `elif` / `else` chain as one logical conditional event.

Example:

```python
if a:
    ...
elif b:
    ...
elif c:
    ...
else:
    ...
```

This becomes one `conditional_id` with branch kinds:

- `then`
- `elif_1`
- `elif_2`
- `else`

The nested `ast.If` nodes used by CPython to represent `elif` are flattened into one record. A true nested `if` inside a branch body remains a separate conditional event.

### Source Index Built Per File

For each parsed file, build:

1. A parent map for AST nodes.
2. A function-scope index:
   - `FunctionDef`
   - `AsyncFunctionDef`
   - `Lambda` when it has usable line spans
3. A normalized conditional table for every root `if` chain in each function scope.
4. A bool-consumer index for all AST contexts that may consume a bool result:
   - `If.test`
   - `Assert.test`
   - `While.test`
   - `IfExp.test`
   - comprehension `if` filters
   - `match case ... if guard`
   - `bool(...)` casts

### Conditional Record

Recommended normalized record:

```python
ConditionalRecord = {
    "id": int,
    "source_file": str,
    "function_span": (int, int),
    "if_stmt_span": (int, int),
    "test_span": (int, int),
    "nesting_depth": int,
    "branch_ranges": {
        "then": (int, int),
        "elif_1": (int, int),
        "elif_2": (int, int),
        "else": (int, int),
    },
    "branch_test_spans": {
        "then": (int, int),
        "elif_1": (int, int),
        "elif_2": (int, int),
    },
    "parent_conditional_id": Optional[int],
    "parent_branch_kind": Optional[str],
}
```

Notes:

- `id` can be assigned densely per `ModelLog`, but the cache should internally key records by structural identity: `(source_file, function_start, if_lineno, if_col_offset)`.
- `branch_ranges` are inclusive line intervals covering the executed body for that branch.
- `else` is present only for a terminal non-`If` `orelse`.

### Branch Range Extraction

For a root `ast.If`:

1. `then` range = from `body[0].lineno` to `end_lineno(body[-1])`
2. If `orelse` starts with a single `ast.If`, flatten it as `elif_n`
3. For each flattened `elif_n`:
   - test span = nested `If.test`
   - branch range = nested `If.body`
4. If the terminal `orelse` is not a single `ast.If`, record one `else` range

### Nested-If Containment Stack

Every conditional record stores its parent relation if it sits inside another branch body:

```python
(conditional_id, branch_kind, line_range)
```

That tuple is the unit carried in the nested branch stack.

### Matching A Bool Line To A Conditional

Given `(source_file, bool_line_number)`:

1. Find the smallest function scope containing `bool_line_number`.
2. Within that scope, find the deepest normalized conditional whose root `test_span` or any `elif_n` test span contains `bool_line_number`.
3. Return that `ConditionalRecord` and the matching branch-test kind:
   - root test => `then`
   - first flattened `elif` test => `elif_1`
   - second flattened `elif` test => `elif_2`
   - etc.

This lets multiple atomic bool ops inside `if a and b:` or `if not cond:` resolve to the same conditional record.

### Pseudocode

```text
function build_file_index(filename):
    source = read(filename)
    tree = ast.parse(source)
    annotate_parents(tree)

    scopes = collect_function_scopes(tree)
    conditionals = []

    for scope in scopes:
        for if_node in direct_and_nested_if_nodes(scope):
            if is_elif_synthetic_child(if_node):
                continue
            record = flatten_if_chain(if_node, scope)
            conditionals.append(record)

    assign_parent_conditionals(conditionals)
    bool_consumers = collect_bool_consumers(tree, conditionals)

    return FileIndex(scopes, conditionals, bool_consumers)


function flatten_if_chain(if_node, scope):
    record = new ConditionalRecord(...)
    record.test_span = span(if_node.test)
    record.branch_ranges["then"] = body_span(if_node.body)

    cursor = if_node
    elif_index = 0
    while len(cursor.orelse) == 1 and isinstance(cursor.orelse[0], ast.If):
        cursor = cursor.orelse[0]
        elif_index += 1
        kind = "elif_" + str(elif_index)
        record.branch_test_spans[kind] = span(cursor.test)
        record.branch_ranges[kind] = body_span(cursor.body)

    if cursor.orelse is not empty:
        record.branch_ranges["else"] = body_span(cursor.orelse)

    return record


function resolve_bool_line(filename, bool_line_number):
    index = get_cached_file_index(filename)
    scope = smallest_scope_containing(index.scopes, bool_line_number)
    if scope is None:
        return None

    matches = []
    for record in scope.conditionals:
        for kind, test_span in record.branch_test_spans.items():
            if test_span.start <= bool_line_number <= test_span.end:
                matches.append((record.depth, record, kind))

    if matches is empty:
        return None

    return deepest_match(matches).record, deepest_match(matches).kind
```

## 2. Op Attribution Algorithm

### Goal

For each captured op source line `L`, compute:

```python
[(conditional_id, branch_kind), ...]
```

ordered from outermost to innermost active branch.

### Important Distinction

Attribution is based on branch body intervals, not the line span of the entire `if` statement.

That means:

- an op on a line inside `if` / `elif` / `else` body is attributed
- an op on a line after the conditional, in the same function and same forward pass, is not attributed

This is the core rule that distinguishes "inside the branch" from "after the branch."

### Recommended Attribution Unit

Use the full `func_call_stack`, not just one frame.

Reason:

- the current stack is ordered shallow-to-deep
- an op inside a helper called from a branch should still inherit the caller's active branch
- relying on only the first frame is brittle and under-attributes nested user calls

### Per-Frame Attribution

For each `FuncCallLocation(file, line_number, func_name)` in an op's call stack:

1. Load the cached file index.
2. Find the smallest function scope containing `line_number`.
3. Query that function's branch interval tree with `line_number`.
4. Return the matching stack of `(conditional_id, branch_kind)` for that frame.

### Cross-Frame Aggregation

Concatenate per-frame stacks shallow-to-deep, deduplicating adjacent repeats.

Example:

- caller frame line is inside `if outer_cond:`
- callee frame line is inside `elif inner_cond:`

Result:

```python
[(outer_id, "then"), (inner_id, "elif_1")]
```

### Direct Child Edge Attribution

For visualization edge lists, direct branch-entry children of a branch-start node are identified by comparing stacks:

1. Attribute the parent op and each child op.
2. The first `(conditional_id, branch_kind)` present on the child but not on the parent is the branch entered by that edge.
3. Emit:
   - `conditional_then_edges`
   - `conditional_elif_edges`
   - `conditional_else_edges`

This avoids hard-coding THEN-only logic in Step 5.

### Proposed `LayerPassLog` Data Structure

Per pass, store the executed branch stack directly:

```python
conditional_branch_stack: List[Tuple[int, str]]
```

Semantics:

- empty list => op is not inside an attributed branch body
- ordered outermost-to-innermost
- values use normalized branch kinds: `then`, `elif_1`, `elif_2`, `else`

Recommended convenience fields:

```python
conditional_branch_depth: int
in_cond_branch: bool  # keep for compatibility; derived from stack length > 0
```

### Pseudocode

```text
function attribute_op(op):
    aggregate_stack = []

    for frame in op.func_call_stack:  # shallow -> deep
        index = get_cached_file_index(frame.file)
        if index is None:
            continue

        scope = smallest_scope_containing(index.scopes, frame.line_number)
        if scope is None:
            continue

        frame_stack = query_branch_interval_tree(scope.branch_tree, frame.line_number)
        aggregate_stack = merge_outer_to_inner(aggregate_stack, frame_stack)

    return aggregate_stack


function query_branch_interval_tree(branch_tree, line_number):
    intervals = branch_tree.query(line_number)  # all branch-body intervals containing line
    sort intervals by nesting depth ascending
    return [(interval.conditional_id, interval.branch_kind) for interval in intervals]
```

## 3. Caching Strategy

### Per-File Parse Cache

Cache key:

```python
(filename, stat(filename).st_mtime_ns)
```

Cached value:

- parsed AST module
- function-scope index
- normalized conditional records
- bool-consumer index
- per-function branch interval trees

This matches the task requirement and is sufficient for interactive model debugging where source files may change between runs.

### Per-Function Branch Interval Tree

For each function scope, pre-build an interval tree over branch body ranges.

Each interval stores:

- `conditional_id`
- `branch_kind`
- `line_range`
- `nesting_depth`

Query complexity is `O(log k + d)` where:

- `k` = number of branch intervals in the function
- `d` = nesting depth of matches returned

`d` is expected to be small.

### Implementation Shape

No third-party dependency is required. A small centered interval tree or equivalent balanced interval index is enough.

The important contract is:

- build once per function
- query by source line
- return all containing branch intervals in nesting order

### Invalidation Policy

1. On any lookup, recompute the cache key from current file mtime.
2. If the mtime changed, discard the whole file entry.
3. Rebuild AST, conditional records, bool-consumer index, and all function trees.
4. No partial invalidation inside a file.

This is conservative and low-risk.

### Lifetime

Keep the cache process-local and reuse it across `log_forward_pass()` calls. Clear only on process exit or explicit manual purge.

## 4. Edge Cases And Handling

### Ternary (`ast.IfExp`)

Handling for v1:

- detect it
- classify the bool conversion as `ifexp`
- do not attribute it to branch stacks

Reason:

- it is real Python control flow, but it does not map cleanly onto the current graph-edge visualization model
- supporting it now would broaden scope beyond the sprint's explicit `if` / `elif` / `else` target

### Comprehension With `if` Filter

Handling for v1:

- detect `comprehension.ifs`
- classify as `comprehension_filter`
- do not mark conditional branches

### `assert cond`

Handling:

- detect `ast.Assert.test`
- classify as `assert`
- do not mark conditional branches

This is a bool consumer, not a branch body selector.

### `while cond`

Handling for v1:

- detect `ast.While.test`
- classify as `while`
- do not attribute branch stacks

Reason:

- loop-body attribution needs different semantics from one-shot branch attribution
- the user explicitly called out `ast.While` as a v1 scope decision

### `if (x := expr):` Walrus

Handling:

- support it
- the walrus expression lives inside `If.test`
- the enclosing `ast.If` still resolves normally by line-span matching

### `if not cond:` And `if a and b:`

Handling:

- support them
- any atomic bool layer whose source line falls inside the root test span or an `elif` test span maps to the owning conditional record
- all such bool layers share the same `conditional_id`

For `and` / `or`, short-circuiting means not all sub-bools will fire on every pass. That is acceptable; the conditional record is still the same.

### Match-Case Guards

Handling for v1:

- detect `match_case.guard`
- classify as `match_guard`
- do not attribute branch stacks

Documented skip, not a silent miss.

### Multi-Line If-Test

Handling:

- use `node.test.lineno` through `node.test.end_lineno`
- do not rely on `node.lineno` alone

This is required for correct matching of wrapped conditions such as:

```python
if (
    torch.mean(x) > 0
    and torch.sum(y) > 0
):
    ...
```

### Nested Conditionals

Handling:

- fully supported
- nested `if` statements inside any branch body become child conditional records
- branch interval query returns all containing intervals, producing the stack naturally

### Early `return` / `raise` Inside A Branch Body

Handling:

- no special-case change to attribution
- ops before the `return` / `raise` remain attributed to the branch body
- ops after the enclosing `if` do not exist on that execution path, so there is nothing extra to mark

## 5. False Positive Filtering

### Problem

Current Step 5 starts from every internally terminated scalar bool and only later clears some false positives if no `ast.If` is found. That is too broad for full branch attribution.

### Proposed Rule

Only scalar bool events classified as `If.test` or flattened `elif` test should participate in branch marking.

Everything else is excluded before graph-level branch metadata is emitted.

### Classification Pipeline

For each internally terminated scalar bool layer:

1. Resolve its best user frame.
2. Look up all bool-consuming AST contexts whose span contains `bool_line_number`.
3. Choose the most specific matching context.
4. Set:
   - `bool_is_branch = True` only for `if_test` or `elif_test`
   - `bool_is_branch = False` otherwise

### Non-Branch Bool Contexts To Detect Explicitly

- no enclosing AST bool consumer
- `assert`
- `bool(...)` cast
- comprehension filter
- `while`
- `ifexp`
- `match_guard`

### Recommended Field On Atomic Bool Event

Until a dedicated bool-event class exists, store classification on the bool-producing `LayerPassLog`:

```python
bool_is_branch: bool
bool_context_kind: Optional[str]
bool_conditional_id: Optional[int]
```

Recommended `bool_context_kind` values:

- `if_test`
- `elif_test`
- `assert`
- `bool_cast`
- `comprehension_filter`
- `while`
- `ifexp`
- `match_guard`
- `unknown`

### Step-5 Impact

The Step 5 flood should start only from bool layers with `bool_is_branch=True`.

That is better than the current "mark first, clear later" pattern because it:

- reduces false graph labels
- reduces unnecessary AST post-validation work
- gives a reusable explanation for every excluded bool event

## 6. False Negatives (Document, Do Not Fix)

### `if python_bool:`

Invisible to TorchLens.

Reason:

- no tensor bool conversion occurs
- no tensor operation terminates in a Python truthiness check

### `if tensor.item() > 0:`

Also effectively invisible to the proposed mechanism.

Reason:

- `tensor.item()` converts to a Python scalar first
- `> 0` yields a Python `bool`
- `Tensor.__bool__` does not run on the branch decision

So this should be documented as a false negative, not treated as a bug in the AST layer.

### `torch.compile` / `torch.jit`

Degrade gracefully.

Expected failure modes:

- synthetic or rewritten frames
- source files unavailable
- line numbers pointing into generated code
- user frames absent from `func_call_stack`

Handling:

- if branch attribution cannot be resolved from real user source, leave conditional metadata empty
- keep the rest of TorchLens logging intact
- do not raise from postprocess

## 7. Data Model Proposal

### `LayerPassLog` Additions

Concrete additions:

```python
conditional_branch_stack: List[Tuple[int, str]]
conditional_branch_depth: int
bool_is_branch: bool
bool_context_kind: Optional[str]
bool_conditional_id: Optional[int]
cond_branch_else_children: List[str]
cond_branch_elif_children: Dict[int, List[str]]
```

Notes:

- `in_cond_branch` stays for compatibility and becomes derivable from `conditional_branch_stack`
- `cond_branch_then_children` stays
- `cond_branch_elif_children` maps `elif_index -> child layer labels`
- `cond_branch_else_children` mirrors the current THEN child convention

### `ModelLog` Additions

Concrete additions:

```python
conditional_events: List[Dict[str, Any]]
conditional_elif_edges: List[Tuple[int, int, str, str]]
conditional_else_edges: List[Tuple[int, str, str]]
```

Recommended `conditional_events` entry shape:

```python
{
    "id": int,
    "kind": "if_chain",
    "source_file": str,
    "function_span": (int, int),
    "if_stmt_span": (int, int),
    "test_span": (int, int),
    "nesting_depth": int,
    "branch_ranges": Dict[str, Tuple[int, int]],
    "branch_test_spans": Dict[str, Tuple[int, int]],
    "parent_conditional_id": Optional[int],
    "parent_branch_kind": Optional[str],
    "bool_layers": List[str],
}
```

Edge tuple semantics:

- `conditional_then_edges` can remain as-is for compatibility or be widened to include `conditional_id`
- `conditional_elif_edges`: `(conditional_id, elif_index, parent_label, child_label)`
- `conditional_else_edges`: `(conditional_id, parent_label, child_label)`

### Multi-Pass Merge Semantics On `LayerLog`

This is the important rolled-graph design point.

Current behavior copies conditional metadata from the first pass only. That is not acceptable once full branch stacks exist.

Recommended aggregate semantics:

```python
conditional_branch_stacks: List[List[Tuple[int, str]]]
conditional_branch_stack_passes: Dict[Tuple[Tuple[int, str], ...], List[int]]
in_cond_branch: bool  # OR across passes
cond_branch_then_children: List[str]  # union across passes, no-pass labels
cond_branch_else_children: List[str]  # union across passes, no-pass labels
cond_branch_elif_children: Dict[int, List[str]]  # union across passes, no-pass labels
```

Rules:

1. `in_cond_branch` = OR across passes
2. `conditional_branch_stacks` = unique stack signatures in first-seen order
3. `conditional_branch_stack_passes` = which passes used each signature
4. branch-entry child lists are unions across passes after stripping pass suffixes

Why this matters:

- the same rolled layer may execute in THEN on one pass and ELSE on another
- preserving only first-pass metadata would silently mislabel the rolled graph

## 8. Algorithmic Complexity Analysis

Let:

- `F` = number of source files touched by a run
- `N_f` = AST size of file `f`
- `C_f` = number of normalized conditionals in file `f`
- `K_s` = number of branch intervals in function scope `s`
- `M` = number of captured ops
- `S` = average user-visible call-stack depth per op
- `B` = number of internally terminated scalar bool layers

### Preprocessing

Per file:

- parse AST: `O(N_f)`
- normalize all conditionals: `O(C_f)`
- build per-function interval trees: `O(sum(K_s log K_s))`

Across all files:

- `O(sum_f N_f + sum_f C_f + sum_s K_s log K_s)`

### Bool Classification

Per scalar bool event:

- file cache lookup: `O(1)`
- scope lookup: `O(log number_of_scopes_in_file)` if scopes are indexed
- conditional / bool-consumer lookup: `O(log C_f + d)` with interval indexes

Total:

- `O(B log C)` in the indexed case

### Op Attribution

Per op:

- one branch query per call-stack frame
- `O(S * (log K + d))`

Total:

- `O(M * S * (log K + d))`

In practice:

- `S` is small
- `d` is small
- AST work only happens when source-context-driven branch attribution is needed

### Memory

Per cached file:

- AST tree: `O(N_f)`
- normalized conditionals and indexes: `O(C_f + sum K_s)`

This is acceptable because branch attribution runs only when atomic bools are present.

## 9. Open Questions Requiring Architect Decision

1. Should full branch attribution require `save_source_context=True`, or should TorchLens always capture minimal `(file, line)` call-location data even when full source snippets are disabled?
2. Should `conditional_id` be a dense per-`ModelLog` integer or a structural key derived from `(file, function_start, if_lineno, col_offset)`?
3. Should `conditional_then_edges` be widened to include `conditional_id`, or should compatibility with the current two-tuple shape be preserved and only new edge lists carry IDs?
4. In rolled graphs, should visualization show unioned branch labels, pass-number-qualified branch labels, or both when one `LayerLog` appears in different branches across passes?
5. Is line-level attribution precise enough, or should TorchLens capture column offsets for bool ops and regular ops to disambiguate multiple bool consumers on the same line?
6. Should `ast.IfExp` and `ast.While` stay explicitly out of v1 scope, or should they be promoted into the same conditional-event model now to avoid a second data-model expansion later?

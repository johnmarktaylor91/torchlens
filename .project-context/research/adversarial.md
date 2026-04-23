# AST Branch Attribution Adversarial Report

## Scope

This is a red-team inventory for an AST-based `if` / `elif` / `else` attribution scheme layered onto TorchLens forward-pass introspection.

Codebase-grounded baseline observed during survey:

- Current Step 5 starts from `internally_terminated_bool_layers`, not from a dedicated "`Tensor.__bool__` happened here" event.
- Current THEN detection is source-driven, requires `save_source_context=True`, parses the file with `ast.parse()`, and matches `ast.If` by line number proximity.
- `FuncCallLocation` source context is lazily loaded through `linecache`, so source text can drift after capture.
- Aggregate `LayerLog` merge rules only merge 3 fields; `cond_branch_then_children` is first-pass-only today.
- Validation currently has no invariant enforcing `cond_branch_then_children <-> conditional_then_edges` consistency.
- Visualization already has conditional-edge limitations: ELK ignores IF/THEN labeling and Graphviz can dedupe away labels.

Assumption used in this report: the proposed design is "attribute subsequent captured ops to THEN / ELIF / ELSE branches using Python AST + source line numbers", and should work on ordinary eager `forward()` execution unless explicitly documented as unsupported.

## 1. Source-Code Availability Failures

### 1.1 Notebook and REPL cells have unstable or synthetic filenames
- Severity: blocker
- Description: `inspect` and frame metadata often report `<ipython-input-...>` or `<stdin>`. The text may exist only in notebook machinery, not as a stable `.py` path. AST lookup by filesystem path fails or resolves stale cell contents after re-execution.
- Mitigation: support notebook cell source providers explicitly, or document notebooks / REPL as unsupported for AST branch attribution.
- Test sketch: `NotebookCellIfModel` defined and redefined across Jupyter cell executions; verify branch attribution after re-run.

### 1.2 `exec()` / `eval()` / dynamically generated functions have no durable source file
- Severity: blocker
- Description: `forward` or helper predicates built from strings often carry filenames like `<string>`. There may be no recoverable source, or the source text used at runtime is different from anything reachable later.
- Mitigation: document limitation; optionally allow caller-supplied source text or content hash.
- Test sketch: `ExecForwardModel` where `forward` is attached via `exec`.

### 1.3 Lambdas and nested closures collapse attribution context
- Severity: high
- Description: lambdas, local helper functions, and nested closures often share line numbers with surrounding code or point at enclosing blocks. The AST range may identify the outer function body while the actual branch lives in an inner scope.
- Mitigation: require branch attribution only for real `FunctionDef` / `AsyncFunctionDef` bodies with stable qualnames, or add scope-aware AST resolution.
- Test sketch: `ClosurePredicateModel` with nested helper returning a tensor bool used by outer `if`.

### 1.4 Decorated `forward()` methods shift apparent source ownership
- Severity: high
- Description: decorators can replace `forward` with a wrapper whose file and line numbers belong to decorator code, not model code. Class `source_line` from `inspect.getsourcelines(module_class)` can also point at decorators rather than the class body.
- Mitigation: unwrap only when safe, record both wrapped and unwrapped metadata, or document decorated `forward` as best-effort only.
- Test sketch: `DecoratedForwardModel` with a logging decorator that wraps `forward`.

### 1.5 Monkey-patched `forward` methods break class-level source assumptions
- Severity: high
- Description: instance-level or class-level monkey patching means the executed `forward` may not match the class source file, line, or AST. A design anchored on module class metadata will silently read the wrong file.
- Mitigation: resolve source from the executing callable object, not from module class metadata alone.
- Test sketch: `MonkeyPatchedForwardModel` where `model.forward = patched_forward.__get__(model, type(model))`.

### 1.6 `.pyc`-only deployments and native extensions expose no parseable Python source
- Severity: blocker
- Description: packaged or frozen environments may retain bytecode only, or implement `forward` in C++ / pybind / `.so`. AST branch attribution cannot recover the source structure.
- Mitigation: document limitation and fail closed.
- Test sketch: synthetic test skipped in normal CI but covered in docs / explicit unsupported-case tests.

### 1.7 `torch.compile`, TorchDynamo, tracing, and scripting can erase Python-level branch structure
- Severity: blocker
- Description: `torch.compile`, `torch.jit.trace`, and `torch.jit.script` can inline, specialize, trace once, or replace Python control flow with graph IR. The runtime path may no longer correspond to the original Python AST, or may execute generated wrapper code instead.
- Mitigation: detect compiled / scripted / traced models and disable AST branch attribution, or clearly scope support to eager Python execution only.
- Test sketch: `CompiledIfModel`, `ScriptedIfModel`, `TracedIfModel`.

### 1.8 Encoding, BOM, and newline normalization can skew line mapping
- Severity: medium
- Description: AST parsing and line-number arithmetic can go wrong when source contains UTF-8 BOM, CRLF normalization changes, or encoding declarations. A captured frame line may still be correct while re-read file offsets drift after normalization.
- Mitigation: open files with the same decoding strategy Python used to compile them, preserve newline handling, and test BOM / CRLF explicitly.
- Test sketch: `CRLFEncodedIfModel` stored with BOM and CRLF.

## 2. AST vs Runtime Misalignment

### 2.1 Multi-line predicates break naive "nearest line" `ast.If` matching
- Severity: blocker
- Description: predicates can span many lines across parentheses, helper calls, and boolean operators. Matching the bool-producing op to the nearest `ast.If` on `lineno` can select the wrong statement or a parent / sibling conditional.
- Mitigation: resolve the exact AST node whose test span contains the runtime event and verify structural ancestry.
- Test sketch: `MultilinePredicateModel` with a 6-line `if (...)` test.

### 2.2 `elif` is represented as nested `If` inside `orelse`
- Severity: blocker
- Description: Python AST does not have a distinct `Elif` node. A line-based scheme can confuse the top-level `if`, inner `elif`, and final `else`, especially when several tests share nearby lines.
- Mitigation: normalize `if` / `elif` ladders into explicit branch arms before attribution.
- Test sketch: `ElifLadderModel` with tensor conditions in `if`, `elif`, and `else`.

### 2.3 Hot reload changes file contents after call-stack capture
- Severity: high
- Description: the frame line number is captured during execution, but AST is parsed later from disk and source context is loaded lazily through `linecache`. If the file changes in between, the recorded line now points at different code.
- Mitigation: capture source snapshot or file content hash at logging time; reject mismatched hashes.
- Test sketch: `HotReloadIfModel` where the file is rewritten between execution and postprocess.

### 2.4 `linecache` invalidation is not automatic enough for live-edit workflows
- Severity: high
- Description: even if disk contents change, `linecache` may continue serving old lines until explicitly invalidated. AST may parse new text while `FuncCallLocation` shows old text, producing self-contradictory attribution evidence.
- Mitigation: call `linecache.checkcache(path)` or cache explicit source snapshots.
- Test sketch: modify source after first access to `code_context` and compare with AST parse.

### 2.5 AST spans are ranges; runtime evidence is point samples
- Severity: high
- Description: a branch body can cover a large line interval containing nested `if`, `with`, `try`, helper calls, and comprehensions. "Child op line falls in body range" is necessary but not sufficient to prove direct branch membership.
- Mitigation: use AST parent chains and statement ownership, not body start/end ranges alone.
- Test sketch: `NestedBodySameRangeModel` with nested `if` inside outer `if`.

### 2.6 Python 3.9 -> 3.13 AST metadata differences alter edge cases
- Severity: medium
- Description: `end_lineno` support, pattern-matching nodes, parser details, and exact spans differ across versions. A design tuned on one Python minor version can silently misattribute on another.
- Mitigation: version-gated parsing logic and a cross-version regression matrix.
- Test sketch: same conditional corpus run under 3.9, 3.10, 3.11, 3.12, 3.13.

### 2.7 Semicolon-packed statements and one-line branches collapse multiple events onto one line
- Severity: medium
- Description: `if cond: a = f(x); b = g(x)` makes two branch-local ops share one physical line. Line-only attribution cannot distinguish them from adjacent same-line statements.
- Mitigation: document limitation or move to column offsets / bytecode events if exactness is required.
- Test sketch: `OneLineBranchModel`.

## 3. Attribution Ambiguity

### 3.1 Multiple ops on one line cannot be uniquely assigned
- Severity: blocker
- Description: chained calls, tuple unpacking, helper calls, or semicolon-separated ops produce several captured layers at the same source line. A line-number scheme can only say "some branch-local work happened here".
- Mitigation: document limitation or enrich attribution with column offsets / AST expression mapping.
- Test sketch: `MultiOpSameLineModel`.

### 3.2 List / set / dict comprehensions and generator expressions create hidden scopes
- Severity: high
- Description: comprehension filters (`[f(x) for x in xs if pred(x)]`) are AST control-flow nodes but not ordinary statement blocks. Runtime ops often execute in an implicit nested scope with confusing line ownership.
- Mitigation: either explicitly support comprehension scopes or document them as unsupported.
- Test sketch: `ComprehensionIfModel`.

### 3.3 Ternary expressions (`ast.IfExp`) put both arms on the same line
- Severity: high
- Description: `y = a(x) if cond else b(x)` has THEN and ELSE on one expression line. Simple line-range checks cannot tell which captured op belongs to which arm, especially if both calls share the same source line.
- Mitigation: support `ast.IfExp` explicitly or document limitation.
- Test sketch: `TernaryIfExpModel`.

### 3.4 `torch.where` and masked tensor ops look like conditionals but are not Python branching
- Severity: medium
- Description: users may expect them to appear as THEN / ELSE branches, but both arms can execute eagerly and the condition is tensorized, not a Python `if`. Mislabeling these as branch edges would be wrong.
- Mitigation: document as non-Python control flow; never infer branch edges from tensorized selection ops.
- Test sketch: `TorchWhereModel`.

### 3.5 The same `if` can execute multiple times inside one forward
- Severity: blocker
- Description: when an `if` lives in a loop, branch-local ops from different iterations share source lines but correspond to different runtime decisions. A no-pass or no-iteration attribution will merge contradictory outcomes.
- Mitigation: store per-execution-instance branch records keyed by pass / iteration.
- Test sketch: `LoopedIfModel` iterating over timesteps with alternating branch outcomes.

### 3.6 Cross-pass aggregation can collapse contradictory branch histories
- Severity: blocker
- Description: aggregate `LayerLog` fields represent only first-pass values for most fields today. If pass 1 took THEN and pass 2 took ELSE, a LayerLog-level branch view will be stale or outright false.
- Mitigation: keep branch attribution on `LayerPassLog` only, or add merge semantics for branch metadata.
- Test sketch: `RecurrentElifModel` with different branch outcome on pass 2.

### 3.7 ELSE attribution is fundamentally harder than THEN attribution
- Severity: high
- Description: THEN can be approximated by body membership after finding an `If`. ELSE requires distinguishing `elif`, explicit `else`, and "no branch-local op executed". Missing ELSE edges can be misread as "condition false and no else", which is not the same thing.
- Mitigation: explicitly model branch arms, including empty arms and `elif` desugaring.
- Test sketch: `ExplicitElseModel` and `NoElseModel`.

## 4. Dynamic Dispatch and Hooks

### 4.1 Forward pre/post hooks can emit ops outside the lexical branch body
- Severity: high
- Description: hooks run because a branch-local module was invoked, but the hook code lives elsewhere. Captured ops from the hook may be semantically branch-caused yet source-located outside the branch body.
- Mitigation: document lexical-only attribution, or separately tag hook-originated ops.
- Test sketch: `ForwardHookIfModel`.

### 4.2 Backward hooks and autograd callbacks muddy branch provenance
- Severity: medium
- Description: if branch metadata is later surfaced on backward or gradient-related artifacts, hook-created ops will not correspond cleanly to forward lexical structure.
- Mitigation: scope branch attribution to forward graph only; document that backward hooks are out of scope.
- Test sketch: `BackwardHookIfModel`.

### 4.3 `nn.Module.apply` and helper traversals can execute conditionals outside `forward`
- Severity: medium
- Description: code may call utility methods that traverse submodules or mutate state before / during forward. A branch event observed in helper code may not map to the module's `forward` source.
- Mitigation: restrict attribution to frames inside the active `forward` stack root.
- Test sketch: `ApplyInsideForwardModel`.

### 4.4 `custom autograd.Function.forward` hides user logic behind a different callable boundary
- Severity: high
- Description: branching can happen inside `autograd.Function.forward`, not the enclosing module `forward`. If attribution assumes all relevant source is in module `forward`, it misses or misattributes those ops.
- Mitigation: support arbitrary user frames, not just module-forward frames.
- Test sketch: `AutogradFunctionIfModel`.

### 4.5 `DataParallel` / `DistributedDataParallel` replicate execution context
- Severity: blocker
- Description: TorchLens is already documented as effectively single-threaded. Replica threads / processes can produce duplicated or interleaved branch events, differing branch outcomes per device, and non-deterministic frame ordering.
- Mitigation: fail closed for DP / DDP branch attribution, or explicitly implement replica-aware logging.
- Test sketch: `DataParallelIfModel`.

## 5. Control-Flow Subtleties

### 5.1 Early `return` inside a branch truncates the graph
- Severity: high
- Description: a THEN arm may end the forward pass before any "post-branch" reconvergence exists. Graph-based heuristics that expect a merge point can misidentify branch starts or over-assign downstream ops.
- Mitigation: handle branch arms with no reconvergence as first-class cases.
- Test sketch: `EarlyReturnIfModel`.

### 5.2 `raise` inside a branch prevents postprocess assumptions from holding
- Severity: high
- Description: the forward pass aborts before normal cleanup, so branch metadata may be partial or internally inconsistent. Any design assuming a completed graph is brittle here.
- Mitigation: document unsupported for exception-raising forwards, or add partial-pass semantics.
- Test sketch: `RaiseInIfModel`.

### 5.3 `try/except/else/finally` nested in branches creates overlapping lexical regions
- Severity: high
- Description: an op may be inside an outer THEN arm and also inside an `except` or `finally` block. Line-range-only ownership cannot express which control-flow decision actually caused it.
- Mitigation: treat `try` constructs as separate control-flow nodes or document limitation.
- Test sketch: `TryExceptInsideIfModel`.

### 5.4 `with` blocks inside branches can trigger hidden enter/exit work
- Severity: medium
- Description: context-manager `__enter__` / `__exit__` code executes because of the branch, but source lines belong to the context manager implementation, not the body.
- Mitigation: document lexical-only attribution.
- Test sketch: `WithInsideIfModel`.

### 5.5 `continue` / `break` inside loop-local `if` statements change future control flow
- Severity: high
- Description: the branch does not only own local ops; it changes whether later loop-body ops run. A naive scheme that attributes only direct body lines misses these control effects.
- Mitigation: either document "lexical body only" semantics or model control-transfer consequences explicitly.
- Test sketch: `BreakContinueIfModel`.

### 5.6 Re-entrant or nested conditionals can share the same predicate-producing line
- Severity: high
- Description: helper functions may compute a bool once, then consume it in nested or repeated `if` statements. A single terminal bool event can map to multiple branch consumers.
- Mitigation: tie attribution to the exact bool-consumption site, not just the bool-producing tensor.
- Test sketch: `ReentrantBoolUseModel`.

### 5.7 Pattern matching guards (`match` / `case ... if`) are control flow too
- Severity: high
- Description: Python 3.10+ introduces guarded `case` arms whose semantics are branch-like but not expressed as `ast.If`. A design focused on `ast.If` / `IfExp` will miss them.
- Mitigation: either support `ast.Match` guards or document unsupported syntax.
- Test sketch: `MatchGuardModel`.

## 6. False Positives: Bool Used Non-Branchingly

### 6.1 `assert tensor_cond` is not a THEN / ELSE branch
- Severity: blocker
- Description: it consumes a truthy value but does not create ordinary branch arms. Treating it as IF/THEN would invent edges that never existed.
- Mitigation: whitelist only actual branching constructs; document asserts separately.
- Test sketch: `AssertTensorCondModel`.

### 6.2 `bool(tensor)` cast for logging / storage is not branch control
- Severity: blocker
- Description: users may cast to Python bool to stash a flag, emit debug text, or pass into a helper. The bool conversion happened, but no model ops should be attributed to THEN / ELSE.
- Mitigation: require evidence of a control-flow consumer, not bool conversion alone.
- Test sketch: `BoolCastOnlyModel`.

### 6.3 `any()` / `all()` over tensor-derived Python bools is reduction, not branch ownership
- Severity: high
- Description: these calls may consume scalarized tensor predicates while remaining outside any branching statement.
- Mitigation: classify them as scalar reductions, not branch sites.
- Test sketch: `AnyAllReductionModel`.

### 6.4 Short-circuit expressions in assignments and returns are not statement branches
- Severity: high
- Description: `x = tensor_bool and helper()` or `return a if cond else b` can use truthiness without creating the same graph semantics as a statement-level `if`.
- Mitigation: support them explicitly or document them as unsupported; do not silently fold them into ordinary IF/ELSE semantics.
- Test sketch: `ShortCircuitAssignModel`.

### 6.5 Test scaffolding and debug guards inside helper code can contaminate model attribution
- Severity: medium
- Description: branching used only for logging, asserts, or test instrumentation inside helpers can appear in captured call stacks and be mistaken for model-semantic control flow.
- Mitigation: optionally filter frames by module / source root, or document that attribution is purely lexical.
- Test sketch: `DebugGuardHelperModel`.

### 6.6 Walrus expressions inside comprehensions are especially easy to misclassify
- Severity: medium
- Description: `[(y := f(x)) for x in xs if bool(pred(x))]` combines hidden scopes, implicit control flow, and bool conversion. The branchy-looking bool use may have no meaningful THEN / ELSE edges in the forward graph.
- Mitigation: document unsupported syntax surface.
- Test sketch: `WalrusComprehensionModel`.

## 7. False Negatives: Branches Without Atomic Bool Capture

### 7.1 Pure Python predicates create real branches with no tensor bool event
- Severity: blocker
- Description: `if training:`, `if self.flag:`, and shape / mode guards produce genuine branch-local ops but never touch `Tensor.__bool__`.
- Mitigation: document that the feature covers tensor-driven Python control flow only, not all branches.
- Test sketch: `PythonBoolIfModel`.

### 7.2 `tensor.item()` / `int(tensor)` / `float(tensor)` scalarize before branching
- Severity: blocker
- Description: `if tensor.item() > 0:` creates a real branch but bypasses `Tensor.__bool__`. The captured tensor may terminate internally, but the actual control decision happened after scalarization.
- Mitigation: detect scalarization APIs explicitly, or document them as unsupported.
- Test sketch: `ItemScalarizationIfModel`.

### 7.3 Metadata-derived predicates (`len(tensor)`, `tensor.shape[0]`, `tensor.device`) bypass bool wrappers
- Severity: high
- Description: these are semantically data-dependent branches in user code, but no tensor-bool op exists to anchor attribution.
- Mitigation: document limitation; do not imply full branch coverage.
- Test sketch: `ShapePredicateModel`.

### 7.4 Functional conditionals (`torch.where`, masked blending, `torch.cond`) have no Python branch edge
- Severity: medium
- Description: users may interpret these as branches, but they are dataflow constructs or alternate control-flow APIs. A Python-AST scheme will miss them entirely.
- Mitigation: document them as out of scope.
- Test sketch: `TorchCondModel`.

### 7.5 Predicates cached before the branch site can hide provenance
- Severity: high
- Description: `flag = (x.sum() > 0).item(); ...; if flag:` separates tensor creation from bool consumption. The branch exists, but the nearest tensor op line may be far away or in another helper.
- Mitigation: track scalarization / bool-consumption events, not just tensor creation sites.
- Test sketch: `CachedFlagModel`.

## 8. Data-Model and Multi-Pass Failure Modes

### 8.1 Same source `if` can take different arms on different passes
- Severity: blocker
- Description: recurrent or looped layers can execute the same source line many times with different outcomes. A single aggregate THEN / ELSE annotation on the logical layer is under-specified and can be wrong.
- Mitigation: keep branch history per `LayerPassLog` and expose aggregate summaries explicitly as sets / counts, not scalars.
- Test sketch: `AlternatingRecurrentIfModel`.

### 8.2 `LayerLog` first-pass-wins semantics make branch metadata stale by construction
- Severity: blocker
- Description: today only 3 fields merge across passes. Any new aggregate branch field added to `LayerLog` without explicit merge rules will silently report pass-1 truth for a multi-pass layer.
- Mitigation: either never aggregate branch-arm membership onto `LayerLog`, or define merge semantics up front.
- Test sketch: `FirstPassThenSecondPassElseModel`.

### 8.3 `recurrent_group` and `equivalent_operations` can hide branch divergence
- Severity: high
- Description: loop detection groups structurally equivalent ops across passes. If the same op appears in different branch arms across iterations, equivalence grouping can imply sameness where control provenance differs.
- Mitigation: keep branch attribution orthogonal to equivalence grouping.
- Test sketch: `EquivalentOpDifferentBranchModel`.

### 8.4 Cross-run reuse of a model object can change branch behavior while keeping old structural assumptions
- Severity: high
- Description: a second logged pass on new inputs may take different branches. Any cached or reused branch intervals from a prior run can be wrong even if source text is unchanged.
- Mitigation: scope branch-attribution caches to a single `ModelLog`.
- Test sketch: same model instance logged twice with opposite branch outcomes.

### 8.5 Selective saving / cleanup can leave partial branch metadata
- Severity: medium
- Description: if future APIs allow dropping unsaved layers or bool nodes before visualization / export, branch edges can dangle or become uninterpretable.
- Mitigation: either require all layers for branch attribution or aggressively validate and prune metadata during cleanup.
- Test sketch: `SelectiveSaveIfModel`.

## 9. Performance at Scale

### 9.1 Large `forward()` files make repeated AST search expensive
- Severity: medium
- Description: a 1000-line `forward` with many nested conditionals can turn naive `ast.walk` per bool event into noticeable postprocess overhead.
- Mitigation: parse each file once and pre-index control-flow nodes by span.
- Test sketch: `HugeNestedIfModel`.

### 9.2 100k-op graphs with many scalar bools amplify backtracking and interval membership checks
- Severity: high
- Description: every bool-to-branch-start traversal plus every op-to-branch-interval lookup compounds memory and CPU. Worst case is a highly recurrent model with frequent branch decisions.
- Mitigation: bound metadata, store compact branch IDs, and benchmark before shipping.
- Test sketch: synthetic `DeepLoopBranchModel` generating many passes and bool nodes.

### 9.3 Source-context capture already adds memory; branch indexing adds more
- Severity: medium
- Description: `save_source_context=True` stores call stacks on every op. Adding per-op branch-arm sets or interval trees can push large-model memory up sharply.
- Mitigation: make branch attribution opt-in and measure memory overhead.
- Test sketch: compare `save_source_context=False` vs `True` vs branch mode on a large model.

### 9.4 Degenerate interval trees can become almost as large as the graph
- Severity: medium
- Description: if every branch arm is small, nested, and repeated, an interval-tree or arm-map representation can approach O(number of ops + number of branch instances) memory.
- Mitigation: deduplicate by source span and branch instance only where semantics allow.
- Test sketch: `ManyTinyNestedIfsModel`.

## 10. Visualization Risks

### 10.1 Deep nesting will create unreadable IF / THEN / ELSE overlays
- Severity: medium
- Description: multiple nested arms across modules can produce dense colored edge bundles and labels that obscure the core dataflow graph.
- Mitigation: add opt-in layers, collapse modes, and hover / detail views; document that deep conditional graphs may need filtered rendering.
- Test sketch: `DeepNestedVisualizationModel`.

### 10.2 Ops outside any `if` need clear semantics relative to branch-local ops
- Severity: medium
- Description: users will ask whether unlabeled edges mean "outside control flow", "unknown", or "unsupported syntax". Ambiguity here becomes a correctness problem in interpretation.
- Mitigation: distinguish explicit "outside any attributed branch" from "source unavailable / unsupported".
- Test sketch: mixed branch / non-branch forward with unsupported constructs.

### 10.3 Edge routing for IF / THEN / ELSE can be ambiguous at reconvergence points
- Severity: high
- Description: once branches merge, a single downstream op may have parents from multiple arms. Edge labels alone may not reveal whether the op executed unconditionally after merge or belongs to one arm.
- Mitigation: render arm nodes / branch blocks, not just edge labels.
- Test sketch: `ReconvergingBranchesModel`.

### 10.4 Legend complexity scales poorly once ELSE, ELIF, unknown, and unsupported states appear
- Severity: medium
- Description: the more states the system admits, the easier it is for users to misread the visualization. Visual semantics can become more confusing than helpful.
- Mitigation: keep a minimal, explicit legend and surface unsupported cases as warnings.
- Test sketch: combined `if` / `elif` / `else` / unsupported construct demo model.

### 10.5 Existing renderer limitations will already undercut new branch semantics
- Severity: high
- Description: ELK currently drops IF/THEN logic entirely, and Graphviz can dedupe away conditional labels. Adding richer branch attribution without fixing renderer semantics will create inconsistent outputs across backends.
- Mitigation: either keep conditional rendering Graphviz-only and document it, or close renderer parity gaps first.
- Test sketch: same model rendered with Graphviz and ELK; compare branch labels.

## 11. Backward-Compatibility Traps

### 11.1 Existing `cond_branch_then_children` users may assume THEN-only semantics forever
- Severity: high
- Description: repurposing current fields to mean generic branch attribution, or adding ELSE into the same containers, will break downstream code that interprets them as current THEN-only edges.
- Mitigation: add new fields for new semantics; preserve old fields and document them as legacy THEN-only.
- Test sketch: compatibility test reading older THEN-only outputs.

### 11.2 `FIELD_ORDER` mismatches will raise hard initialization errors
- Severity: blocker
- Description: `LayerPassLog` enforces exact field-set equality with `LAYER_PASS_LOG_FIELD_ORDER`. Adding branch metadata without updating constants, renaming, cleanup, and constructors yields immediate crashes or missing-attribute breakage.
- Mitigation: update class definitions and all canonical field-order registries together.
- Test sketch: serialization / construction round-trip with new fields present.

### 11.3 Pickled / serialized `ModelLog`s from older versions will miss new branch fields
- Severity: blocker
- Description: unpickling older artifacts into newer classes can fail, or succeed with partially missing branch metadata that downstream code assumes exists.
- Mitigation: version gates, defaults on load, or explicit non-compatibility documentation.
- Test sketch: load old fixture pickle into new code and inspect branch fields.

### 11.4 `to_pandas`, exporters, cleanup, and rename steps can silently drop new metadata
- Severity: high
- Description: current project notes already flag missing `cond_branch_then_children` coverage in `to_pandas`. Any new fields will need matching updates across exporters, cleanup filters, and label-renaming steps.
- Mitigation: audit every branch-metadata consumer before release.
- Test sketch: pandas export and cleanup round-trip preserve branch columns / edges.

### 11.5 Validation currently will not catch internal branch-edge inconsistency
- Severity: high
- Description: there is no invariant checking `cond_branch_then_children <-> conditional_then_edges` consistency today. New fields increase the chance of silently inconsistent logs.
- Mitigation: add invariants for every new branch container before shipping.
- Test sketch: deliberately corrupt metadata in a unit test and assert validation failure.

## 12. Regression Coverage Proposal

Recommended concrete test models for long-term coverage:

- `NotebookCellIfModel`: defined in notebook-style synthetic source; verifies unsupported or supported-notebook behavior is explicit.
- `ExecForwardModel`: `forward` injected via `exec`; verifies fail-closed behavior.
- `DecoratedForwardModel`: wrapped `forward`; verifies wrapper vs user source handling.
- `MultilinePredicateModel`: multi-line `if` test; verifies exact `If` selection.
- `ElifLadderModel`: `if` / `elif` / `else`; verifies arm separation.
- `TernaryIfExpModel`: ternary expression; verifies explicit unsupported or correct handling.
- `ComprehensionIfModel`: comprehension filter; verifies unsupported or scope-aware behavior.
- `LoopedIfModel`: same `if` executed repeatedly in one forward with alternating outcomes.
- `AlternatingRecurrentIfModel`: same logical layer across passes with different branch outcomes; verifies `LayerPassLog` vs `LayerLog` semantics.
- `ForwardHookIfModel`: hook-created ops around a branch-local module call.
- `AutogradFunctionIfModel`: conditional inside `autograd.Function.forward`.
- `DataParallelIfModel`: verifies explicit disablement or replica-aware behavior.
- `EarlyReturnIfModel`: branch ends forward before merge.
- `TryExceptInsideIfModel`: nested exception flow inside an `if`.
- `MatchGuardModel`: Python 3.10+ pattern-match guard.
- `AssertTensorCondModel`: bool consumption without branch arms; verifies no false IF edge.
- `BoolCastOnlyModel`: `flag = bool(tensor_cond)` with no branch; verifies no attribution.
- `ItemScalarizationIfModel`: `if tensor.item() > 0:`; verifies documented false-negative or dedicated support.
- `ShapePredicateModel`: branch on `x.shape[0]`; verifies non-tensor control flow is out of scope.
- `TorchWhereModel`: tensorized conditional op; verifies not mislabeled as Python branching.
- `ReconvergingBranchesModel`: explicit merge after THEN and ELSE; verifies visualization semantics.
- `HugeNestedIfModel`: performance benchmark with deep nesting and many source intervals.
- `CRLFEncodedIfModel`: BOM + CRLF source file; verifies stable parsing and line mapping.
- `HotReloadIfModel`: file changed after execution; verifies snapshot/hash mismatch handling.

## Bottom Line

The dominant failure mode is not "AST parsing is hard"; it is "lexical source ranges are an incomplete proxy for runtime control-flow ownership". The scheme is viable only if it is explicitly scoped, aggressively validated, and honest about unsupported syntax and execution modes. Without that discipline, it will produce confident-looking wrong answers in exactly the cases users most need help understanding.

# super-family-sprint -- Sprint Plan

Goal: ship the full `Super[T]` family at Bundle level, unify the
`Accessor[T]` pattern Trace-side and Bundle-side, fold `multi_trace/`
into `intervention/`, add structural-agreement predicates, and ship
the `bundle.at(label)` auto-dispatcher.

Branch: `naming-sprint-impl` (one-working-branch policy).

## Locked architectural decisions

(Pulled from `.project-context/todos.md:586-609`, ratified 2026-05-09.)

1. **Super[T] is one generic base class with per-kind extensions**, NOT
   N hand-written classes. Universal alignment machinery lives in the
   base; tensor stacking lives in a `_TensorBearing` mixin (or stays
   inline in SuperOp); per-kind compare hooks earn their existence.
2. **Bundle stays Bundle, NOT SuperTrace.** Trace defines alignment
   keys; Super[X] projects sub-Trace objects across Bundle members;
   Trace itself doesn't get projected.
3. **Ontology closes at Bundle.** No bundles-of-bundles; consumer
   abstractions (Sweep, Timeline, LineageTree) live above us.
4. **Universal rule applies BELOW Trace.** Every sub-Trace class gets
   a Super counterpart at Bundle level. No piecemeal exceptions.
5. **Naming locked: `Super[T]`** (not `Aligned`, `Stacked`, `Column`,
   `Matched`, `View`, `Cohort`). Pairs with existing `Supergraph` /
   `SupergraphNode` for prefix consistency. Compound-noun ugliness
   (`SuperModuleCall`, `SuperGradFnCall`) accepted; will revisit
   underlying nouns in naming-sprint v3.
6. **Sparse alignment is OK.** SuperGradFn includes only members that
   have backward; surfaces low `coverage`; no error.
7. **`bundle.aligned_pairs()` keeps its name.** No collision with
   `Super[T]` since we're not using `Aligned`.

## Phase 0 -- Sprint scaffolding (DONE; this PR)

- `.research/super-family-sprint_STATE.md` -- supervised-sequential
  protocol, wake-up case routing, fallback chain, shutdown procedure.
- `.research/super-family-sprint_PLAN.md` -- this file.
- Each phase gets `super-family-sprint_phase{N}_PROMPT.md` written
  immediately before dispatch.

## Phase 1 -- Trace-side `Accessor[T]` base

**Scope**: extract a generic `Accessor[T]` from the existing accessor
classes. Pure refactor; no public API change.

**Touched files**:

- `torchlens/data_classes/layer_log.py` -- `LayerAccessor`
- `torchlens/data_classes/op_log.py` -- (OpAccessor lives elsewhere; verify)
- `torchlens/data_classes/module_log.py` -- `ModuleAccessor`
- `torchlens/data_classes/param_log.py` -- `ParamAccessor`
- `torchlens/data_classes/buffer_log.py` -- `BufferAccessor`
- `torchlens/data_classes/grad_fn_log.py` -- `GradFnAccessor`
- New: `torchlens/data_classes/_accessor_base.py` (or
  `torchlens/accessors/_base.py` if cleaner)

**Universal API**: `Accessor[T]` provides default implementations of:

- `__getitem__(key: int | str) -> T` -- ordinal int, exact label, fuzzy substring
- `__contains__(key: object) -> bool`
- `__iter__() -> Iterator[T]`
- `__len__() -> int`
- `keys() -> list[str]`, `values() -> list[T]`, `items() -> list[tuple[str, T]]`
- `to_pandas()` -> DataFrame (where T has a sensible row representation)
- "Did you mean...?" suggestion machinery (consolidate from current
  per-class implementations)

**Subclass overrides**: only type-specific lookup logic (e.g.,
pass-qualified label disambiguation for `OpAccessor`).

**Acceptance**:

- All existing accessor tests pass unchanged.
- `pytest tests/ -m smoke -x --tb=short` green.
- No public API change; no deprecation warnings.

## Phase 2 -- Bundle-side `Super[T]` base

**Scope**: extract `Super[T]` base from `SuperOp`. Universal alignment
machinery moves up; tensor stacking stays in `SuperOp` (or factored
into a `_TensorBearing` mixin -- TBD by codex).

**Touched files**:

- `torchlens/multi_trace/super_op.py`
- New: `torchlens/multi_trace/_super_base.py`

**Universal API on `Super[T]`**:

- `members: dict[str, T]`
- `traces: set[str]`
- `coverage: float`
- `node_name: str`
- `labels: dict[str, str]`
- `__repr__`
- Sparse-alignment handling (some members may lack the alignment key).

**Tensor-bearing API** (stays on `SuperOp`, will be inherited by future
`SuperBuffer`/`SuperParam`/`SuperGradFn` -- decide mixin vs inheritance):

- `out`, `outs`, `grad`, `grads`
- `_stacked`, `aggregate`, `diff_pair`, `_diff_matrix`, `_diff_row`
- `op_type`, `module_path`, `shape`

**Acceptance**:

- `SuperLayer` becomes a small subclass (~5-10 lines).
- All existing SuperOp / SuperLayer tests pass unchanged.
- `bundle.ops["..."]` and `bundle.layers["..."]` behave identically to before.
- Smoke green.

## Phase 3 -- Bundle-side `SuperAccessor[T, S]` base

**Scope**: generalize `_BundleLabelAccessor` into parameterized
`SuperAccessor[T, S]` where `T` is the underlying Trace-side type
(e.g., `LayerLog`) and `S` is the wrapping Super class
(e.g., `SuperLayer`).

**Touched files**:

- `torchlens/multi_trace/super_op.py` (existing accessors)
- New: `torchlens/multi_trace/_super_accessor_base.py`

**Universal API on `SuperAccessor[T, S]`**:

- `__getitem__(label: str) -> S` -- lookup by alignment key, build Super[T]
- `__contains__(label: object) -> bool`
- Sparse-alignment handling (some members may lack the key).
- Coverage reporting on the resulting Super[T].

**Subclass overrides**: only `_resolve_in_member(trace, label) -> T`
(how to look up the inner type in each Trace).

**Acceptance**:

- `SuperOpAccessor` and `SuperLayerAccessor` reduce to ~5 lines each.
- All existing tests pass.
- Smoke green.

## Phase 4 -- multi_trace -> intervention consolidation

**Scope**: fold `torchlens/multi_trace/` into `torchlens/intervention/_super/`
(Super* classes + accessors) and `torchlens/intervention/_topology/`
(Supergraph, SupergraphNode, TopologyDiff, build_supergraph,
compare_topology). Drop `torchlens/multi_trace/` entirely.

**Touched files**:

- All of `torchlens/multi_trace/`
- All import sites: `torchlens/intervention/bundle.py`, tests, etc.
- `torchlens/CLAUDE.md` (update subpackage list)

**Acceptance**:

- `torchlens/multi_trace/` no longer exists.
- All imports updated; no broken references.
- `grep -rn "torchlens.multi_trace\|torchlens/multi_trace" .` returns
  zero hits in `torchlens/` and `tests/`.
- Smoke green.
- Internal-only modules (no public API change), so no deprecation shim
  needed -- the multi_trace module was always documented as
  internal-only.

## Phase 5 -- Add missing Super kinds + Bundle accessors

**Scope**: ship the universal-rule completion. New Super classes + new
Bundle accessor surface.

**New Super classes**:

- `SuperModule` -- aligns ModuleLog across members. Type-specific
  compare hooks: parameter signature equality, child module count.
- `SuperBuffer` -- aligns BufferLog. Tensor-bearing (buffer values).
- `SuperParam` -- aligns ParamLog. Tensor-bearing (weight + grad).
  Type-specific: weight-distribution diff.
- `SuperGradFn` -- aligns GradFnLog. Sparse (some members may lack
  backward). Tensor-bearing for grads. Type-specific: backward-graph
  structural diff.
- `SuperModuleCall` -- aligns ModuleCallLog. Per-call alignment.
  Trivial subclass; mostly inherits.
- `SuperGradFnCall` -- aligns GradFnCallLog. Trivial subclass.

**New Bundle accessors** (mirror Trace's accessor names exactly):

- `bundle.modules` -> `SuperModuleAccessor`
- `bundle.buffers` -> `SuperBufferAccessor`
- `bundle.params` -> `SuperParamAccessor`
- `bundle.grad_fns` -> `SuperGradFnAccessor`
- `bundle.module_calls` -> `SuperModuleCallAccessor`
- `bundle.grad_fn_calls` -> `SuperGradFnCallAccessor`

**Acceptance**:

- `bundle.modules["..."]`, `bundle.buffers["..."]`, etc. all callable
  on a fixture trace.
- Each returns a Super[T] with `coverage`, `members`, `__repr__`.
- Tier-1 smoke + tier-2 (`-m "not slow"`) green.
- No regression on `bundle.ops` / `bundle.layers`.

## Phase 6 -- Bundle structural-agreement predicates

**Scope**: ship structural-consistency predicates so users can verify
Bundle alignment readiness before projecting Super[X] across members.

**New methods on `Bundle`**:

- `is_structurally_consistent: bool` -- True iff all members share
  `graph_shape_hash`.
- `shared_op_labels: list[str]` -- intersection across members.
- `divergent_op_labels: list[str]` -- symmetric difference.
- `shared_layer_labels: list[str]`
- `divergent_layer_labels: list[str]`
- (For symmetry) shared/divergent for module / buffer / param / grad_fn
  / module_call / grad_fn_call labels.

**Acceptance**:

- `bundle.is_structurally_consistent` returns True for two clones of
  the same trace.
- Returns False for two traces with different topology (e.g., one with
  conditional THEN fired, one with ELSE).
- Tier-1 smoke + tier-2 green.

## Phase 7 -- bundle.at(label) auto-detect dispatcher

**Scope**: a single entry point that resolves a label string to the
right Super accessor by inspecting the label format.

**Heuristics**:

- Dotted-path with module-class suffix (`encoder.layer.0.attn`) -> `bundle.modules`
- Pass-qualified label (`conv2d_1_1`) -> `bundle.ops`
- Bare op-type label (`conv2d_1`) -> `bundle.layers`
- Parameter path (`encoder.layer.0.weight`) -> `bundle.params`
- (Etc. -- enumerate in implementation.)

Ambiguity raises a clear error: "Label X matches both Y and Z; use
bundle.<accessor>[X] to disambiguate."

**Acceptance**:

- `bundle.at("conv2d_1_1")` returns the right SuperOp.
- `bundle.at("conv2d_1")` returns the right SuperLayer.
- Ambiguous labels raise.
- Tier-1 smoke + tier-2 green.

## Phase 8 -- Hygiene + docs

**Scope**: tier-2 confirmation, docstring sweep, `MEMORY.md` updates.

- Update `torchlens/CLAUDE.md` (drop `multi_trace/` from subpackage
  list, document `intervention/_super/` and `intervention/_topology/`).
- Update `torchlens/intervention/CLAUDE.md` if it exists.
- Update `~/.claude/projects/-home-jtaylor-projects-torchlens/memory/MEMORY.md`
  to mention the Super* family is shipped + universal rule landed.
- Write `.research/super-family-sprint_SUMMARY.md` with phases shipped,
  test counts before/after, files touched, commits.
- Final tier-2 run -> all pass.

## Out of scope (deliberately deferred)

- Naming-sprint v3 (`peek -> pluck`, selector poetry, `OpLog -> Op`,
  `ModuleCallLog -> ModulePass`). Filed as todos.md entry.
- Comparison-verb consolidation (`compare`, `delta_map`,
  `aligned_pairs` -> single `compare()`). Naming sprint.
- Bundle Phase 8 comparison helpers (`norm_delta`, `output_delta`,
  `show_diff`, `delta_map` extensions). Separate work.
- IO surface unification (to_pandas across classes). Post-naming sprint.

## Notes

- One commit per phase, conventional-commit prefix (`refactor(super):`,
  `feat(super):`, etc.).
- No major version bumps without explicit JMT approval.
- Pre-commit hook reformats; expect to re-stage.
- Test tiers per phase (see STATE.md "Per-phase verification").

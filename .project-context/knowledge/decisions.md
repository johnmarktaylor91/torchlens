# TorchLens Architectural Decisions

## 2024 — Toggle Architecture (Permanent Decoration)
Context: Originally TorchLens re-wrapped/un-wrapped torch functions on every `log_forward_pass` call.
Decision: Wrap all ~2000 torch functions once at `import torchlens` time, gate with single `_logging_enabled` bool.
Rationale: Eliminates per-call decoration overhead (~200ms), makes wrappers stateless. Single bool check when disabled is negligible.
Alternatives considered: Context-manager-based decoration (too slow), monkey-patching per call (fragile).

## 2024 — Global State in _state.py
Context: Decorated wrappers need access to session state (active ModelLog, toggle, etc.).
Decision: Single `_state.py` module holds all mutable state. No imports from other torchlens modules.
Rationale: Prevents circular imports. Wrappers only need to import `_state`, not heavy torchlens modules.
Alternatives considered: Thread-local state (too complex), class-based state (no benefit over module globals).

## 2025 — LayerLog Class Hierarchy (PR #92)
Context: TensorLog was both per-pass and aggregate. RolledTensorLog was a separate class for rolled views.
Decision: Split into LayerPassLog (per-pass) and LayerLog (aggregate). Eliminate RolledTensorLog.
Rationale: Clean separation of concerns. LayerLog delegates to single-pass LayerPassLog via `__getattr__`.
Alternatives considered: Keep RolledTensorLog (too much duplication).

## 2025 — BufferLog Stays as Subclass
Context: BufferLog has `name`/`module_address` fields that don't apply to generic LayerLog.
Decision: BufferLog(LayerPassLog) keeps buffer-specific properties. Single-pass LayerLogs access them via delegation.
Rationale: LayerLog is too generic for buffer metadata. Delegation handles the single-pass common case.

## 2025 — Backward-Only Conditional Flood (Bug #88, PR #127)
Context: Bidirectional flood from terminal booleans falsely marked non-conditional children.
Decision: `_mark_conditional_branches` floods backward-only (parent_layers). AST-based THEN detection when `save_source_context=True`.
Rationale: Forward flood follows data flow, not control flow. Backward-only correctly marks ancestors of the branch decision.

## 2026 — ELK Stress Bypass for >100k Nodes (PR #132)
Context: ELK stress allocates two n^2 × 8-byte distance matrices. 100k nodes = 160GB.
Decision: >100k nodes bypass ELK entirely → Python topological layout (Kahn's algorithm, O(n+m)).
Rationale: No size guard possible in elkjs. The old >150k stress switch was fundamentally broken.

## 2026 — Dagua Integration (Opt-In)
Context: Graphviz rendering has limitations for large graphs and interactive exploration.
Decision: Add dagua as optional renderer (`vis_renderer="dagua"`). Graphviz remains default.
Rationale: Dagua provides GPU-accelerated layout and richer interaction model, but visual semantics still under iteration. Keep stable default.

## 2026 — Global Undecorate Override (PR latest)
Context: Advanced users need clean PyTorch environment for benchmarking, profiling, or debugging decorator interactions.
Decision: Expose `undecorate_all_globally()` / `redecorate_all_globally()` as explicit user API.
Rationale: Permanent decoration is the right default, but escape hatch needed for power users.

## 2026-04 — Conditional Attribution Sprint Reference
Context: Phase 1-8 implementation for full `if` / `elif` / `else` attribution landed across the sprint branch.
Decision: Treat `.project-context/plans/if-else-attribution/plan.md` (v7) as the canonical design record, with implementation spread across commits `ced5a06`, `21eabe8`, `f4c59c1`, `71f976c`, `c04fe8b`, `7eb3baa`, `4236c24`, `11e408c`, and `f97f19d`.
Rationale: Future sessions should start from the plan plus the shipped implementation commits, not from partial memory notes or intermediate review docs.

## 2026-04 — D1: AST + Frame Inspection
Context: TorchLens needed eager-Python branch attribution without global tracing hooks.
Decision: Use AST indexes plus per-op `func_call_stack` frame inspection, not `sys.settrace` or bytecode instrumentation.
Rationale: Reuses existing capture data, keeps overhead bounded, and preserves Python-level structure.

## 2026-04 — D2: Flatten If/Elif Chains
Context: Python AST represents `elif` as nested `If` nodes in `orelse`.
Decision: Flatten each `if` / `elif` / `else` chain into one conditional record with arms `then`, `elif_N`, and `else`.
Rationale: Gives one stable event id per chain and avoids nested-elif bookkeeping bugs.

## 2026-04 — D3: Pre-Classify Terminal Bools
Context: Scalar bool producers appear in both branch-driving and non-branch contexts.
Decision: Pre-classify every terminal scalar bool by enclosing AST context, treating `if_test`, `elif_test`, and `ifexp` as branch-participating kinds.
Rationale: Separates true branch predicates from `assert`, standalone `bool(x)`, `while`, and other classified-only consumers.

## 2026-04 — D4: Use Full FuncCallStack
Context: The user-facing frame alone can miss the scope that actually contains the branch syntax.
Decision: Attribute from the full `func_call_stack`, not a single selected frame.
Rationale: Nested helpers, wrappers, and local functions need all captured frames to resolve scope correctly.

## 2026-04 — D5: LayerPassLog Is Source of Truth
Context: Branch membership can differ by pass in recurrent models.
Decision: Store per-pass branch state on `LayerPassLog.conditional_branch_stack`; aggregate later on `LayerLog`.
Rationale: Prevents rolled views from discarding pass-specific attribution.

## 2026-04 — D6: Preserve Existing Fields
Context: Existing consumers already read legacy conditional fields.
Decision: Keep old public fields and populate them as derived views from the new primary structures.
Rationale: Maintains backward compatibility while allowing a richer internal representation.

## 2026-04 — D7: Isolate AST Logic
Context: Conditional parsing and attribution logic is separate from graph mutation logic.
Decision: Put AST indexing, bool classification, and op attribution in `torchlens/postprocess/ast_branches.py`.
Rationale: Keeps `control_flow.py` focused on Step 5 orchestration and graph writes.

## 2026-04 — D8: Cache AST By File + Mtime
Context: Parsing source repeatedly is expensive and stale source must not survive edits.
Decision: Cache file indexes by `(filename, stat.st_mtime_ns)`.
Rationale: Avoids repeated parsing while failing soft when source changes on disk.

## 2026-04 — D9: Ungated Identity Capture
Context: Conditional attribution only needs call-site identity, not full source text.
Decision: Run AST classification whenever `(file, line)` is available on `func_call_stack`; `save_source_context` gates rich source text only.
Rationale: Branch attribution remains available in low-overhead mode.

## 2026-04 — D10: Sole Key-to-ID Translation In Step 5c
Context: Structural AST keys and public dense ids serve different purposes.
Decision: Use `ConditionalKey` internally in 5a/5b/5e, then materialize dense `cond_id`s exactly once in Step 5c.
Rationale: Centralizing translation keeps ids deterministic and prevents mixed key/id states.

## 2026-04 — D11: Graphviz-First Labeling
Context: The sprint needed user-visible branch labels without expanding renderer scope indefinitely.
Decision: Ship explicit Graphviz `THEN` / `ELIF` / `ELSE` labels first; defer dagua/ELK follow-through.
Rationale: Delivers the primary visualization outcome while respecting the scoped renderer deferrals.

## 2026-04 — D12: Remove Old Post-Validation Clearing
Context: Older IF detection cleared conditional markings after the fact if AST confirmation failed.
Decision: Remove that post-validation cleanup path.
Rationale: Pre-classification makes the old defensive clearing redundant and potentially destructive.

## 2026-04 — D13: Diff Branch Stacks On Every Forward Edge
Context: An edge can enter a branch even when its parent is not itself the predicate.
Decision: Detect arm-entry edges by diffing parent/child `conditional_branch_stack` across every forward edge.
Rationale: Captures legitimate multi-arm entry such as parameter/buffer dependencies entering nested branches.

## 2026-04 — D14: Strict Scope Resolution With Fail-Closed Fallback
Context: Same-line nested defs and older Python versions can make scope matching ambiguous.
Decision: Resolve scopes by exact `(filename, code_firstlineno, code_qualname)` first, then unique `(filename, code_firstlineno, func_name)` fallback, else fail closed.
Rationale: Avoids silently attributing ops to the wrong conditional.

## 2026-04 — D15: Record Rolled Edge Pass Divergence
Context: A rolled edge can belong to different branch arms on different passes.
Decision: Add `ModelLog.conditional_edge_passes` keyed by `(parent_no_pass, child_no_pass, cond_id, branch_kind)`.
Rationale: Renderers and downstream tooling need exact pass lists to label mixed-arm rolled edges correctly.

## 2026-04 — D16: Make Cond-Id-Aware Structures Primary
Context: Legacy THEN/ELIF/ELSE edge lists cannot represent multi-arm entry cleanly.
Decision: Promote `conditional_arm_edges` and `cond_branch_children_by_cond` to the primary representation.
Rationale: Cond-id-aware structures preserve simultaneous entry into multiple arms while keeping legacy views derivable.

## 2026-04 — D17: Disabled-Source Mode Reuses No-Source Contract
Context: `save_source_context=False` previously dropped call-stack data and retained `_frame_func_obj` too long.
Decision: Always capture identity fields, mirror `save_source_context` into `source_loading_enabled`, and initialize the existing no-source placeholder state immediately when disabled.
Rationale: Branch attribution works without rich source text, zero disk access is preserved, and clearing `_frame_func_obj` at construction avoids pickle fragility.

## 2026-04 — D18: Promote Ternary Attribution With Col Offsets
Context: Ternary arms share a source line, so line numbers alone are ambiguous.
Decision: Treat `ast.IfExp` as a first-class conditional and capture `col_offset` for `(line, col)` branch-range matching.
Rationale: Enables full ternary attribution on Python 3.11+ while failing closed on older versions when ambiguity remains.

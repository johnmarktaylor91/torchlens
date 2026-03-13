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

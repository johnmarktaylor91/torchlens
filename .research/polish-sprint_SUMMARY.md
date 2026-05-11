# polish-sprint -- SUMMARY

Five small launch-prep wins shipped after the super-family-sprint
landed. Each phase was bounded; the largest was `keep_orphans` at
~20 minutes of codex time.

## What shipped

- Phase 1 `cb141cf` -- `fix(viz): handle headless trace drawing`.
  `Trace.draw()` no longer crashes via `xdg-open` on headless
  machines; auto-falls-back to save-only with a printed path.
  `Trace.render_graph()` restored as a deprecation alias.
- Phase 2 `da060bf` -- `fix(viz): use sequential bundle diff palette`.
  Replaced the diverging blue→white→red palette with a sequential
  white→red palette. Snapshots regenerated. The L2 delta is unsigned
  and non-negative; white = zero is now the correct semantic.
- Phase 3 `72cec3c` -- `feat(op): add FX qualpath metadata`.
  `OpLog.fx_qualpath` (dotted module path + op type) and
  `OpLog.fx_call_index` (re-call disambiguator) populated as
  derived metadata fields. Not a lookup key (yet); pure metadata
  for cross-tool reference. Slight cosmetic redundancy on trivial
  modules (`relu.relu`); refine in follow-up.
- Phase 4 `07f15e1` -- `feat(capture): keep orphan ops with accessor`.
  `tl.trace(model, x, keep_orphans=True)` retains true island ops
  (internally-generated subgraphs with no input or output ancestry).
  `trace.orphans` promoted from a label list to a proper
  `Accessor[OpLog]`. Default `keep_orphans=False` preserves current
  purge behavior.
- Phase 5 `de2fe34` -- `feat(intervention): add layer and op shortcuts`.
  `op.do(transform)`, `op.set(transform)`, `op.attach_hooks(...)`
  and the analogous `layer.*` shortcuts. Equivalent to
  `trace.do(label, transform)` etc. Detached logs raise a clear
  error pointing at the canonical Trace-level form.

## Test counts

- Smoke: 170/170 passed (every phase).
- Bundle: 78/78 passed (every phase).
- Tier-2 (excluding pre-existing nnsight + rendering colon
  failures): see iteration log in STATE.md.

## Residual / known issues

- `tests/test_migrations.py::test_migration_example_runs[from_nnsight.md-...]`
  fails on main due to nnsight env -- not introduced by this sprint.
- `tests/test_param_log.py::TestVisualizationParams::*` fails on
  working tree due to unrelated rendering.py edits -- not introduced
  by this sprint.
- `OpLog.fx_qualpath` reads slightly redundant on trivial single-op
  modules (e.g., `relu.relu`, `encoder.linear`). The format is
  technically accurate (op-type appended as final segment per spec)
  but visually awkward when the module's role IS just that op.
  Refine in follow-up.

## Next-sprint candidates

The polish-sprint cleared the highest-priority launch-blockers.
Items most often discussed in the conversation that are NOT yet
shipped:

- Generic `transform=` primitive on `tl.trace` + raw-input
  rendering (input + output transforms, batched-input rendering,
  layer visualizers) -- afternoon-scale per the design riff.
- HuggingFace bridge `trace_text(model, "prompt")` -- 5-line
  wrapper once `transform=` ships.
- Naming-sprint v3: `peek -> pluck`, `replay -> {play|cascade|propagate}`,
  `OpLog/LayerLog/etc. -> drop -Log suffix` (per principled rule),
  selectors-as-poetry, comparison-verb consolidation. Coordinated
  rename pass; do NOT bundle with feature work.
- Module-attribution simplification: replace input-tensor-derived
  attribution with a thread-local module call-stack snapshot. Vast
  simplification (collapses three mechanisms into one). Defer until
  launch stable.

## Files changed

`git diff main..HEAD --stat | tail -1`:
all polish-sprint phases combined ~700 lines net.

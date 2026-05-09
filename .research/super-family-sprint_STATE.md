---
run: super-family-sprint
created: 2026-05-09T12:22:36-04:00
state: PHASE_1_RUNNING
current_round: 1
---

# super-family-sprint -- Sprint State

Canonical "where are we" record. Every wake-up reads this FIRST.

This sprint is **supervised sequential**, not fully-autonomous: each phase
dispatches to codex; on `CODEX_DONE` Claude reviews the diff, runs
phase-appropriate tests, commits, then dispatches the next phase.

## Stop criteria (observable, quantitative)

All of:

1. All 8 phases below shipped (each = own commit on `naming-sprint-impl`).
2. Tier-1 smoke: `pytest tests/ -m smoke -x --tb=short` -> all pass.
3. Tier-2: `pytest tests/ -m "not slow" -x --tb=short` -> all pass.
4. Bundle accessor surface complete and callable on a fixture trace:
   - `bundle.modules["..."]` -> `SuperModule`
   - `bundle.buffers["..."]` -> `SuperBuffer`
   - `bundle.params["..."]` -> `SuperParam`
   - `bundle.grad_fns["..."]` -> `SuperGradFn` (sparse OK)
   - `bundle.module_calls["..."]` -> `SuperModuleCall`
   - `bundle.grad_fn_calls["..."]` -> `SuperGradFnCall`
5. `bundle.is_structurally_consistent` (bool) callable.
6. `bundle.at("conv2d_1_1")` auto-routes to right Super accessor.
7. `multi_trace/` removed; contents folded into
   `intervention/_super/` and `intervention/_topology/`.

max_rounds: 8 (one per phase)

## Phase plan (canonical reference; do not edit during sprint)

| # | Title | Risk | Effort |
|---|---|---|---|
| 1 | Trace-side `Accessor[T]` base | low (refactor) | half day |
| 2 | Bundle-side `Super[T]` base | low (refactor) | half day |
| 3 | `SuperAccessor[T, S]` base | low (refactor) | half day |
| 4 | `multi_trace/` -> `intervention/_super/` + `_topology/` | low (mechanical move) | half day |
| 5 | Add SuperModule/Buffer/Param/GradFn/ModuleCall/GradFnCall + accessors | medium | 1 day |
| 6 | Bundle structural-agreement predicates | low | half day |
| 7 | `bundle.at(label)` auto-detect dispatcher | medium | half-to-full day |
| 8 | Tests + docs + MEMORY.md updates | low | half day |

Full prompts live alongside as `super-family-sprint_phase{N}_PROMPT.md`.

## Wake-up case routing (supervised sequential)

| Observable signal | State | Claude's action |
|---|---|---|
| codex pid alive | RUNNING | ack briefly, yield turn |
| `CODEX_DONE` + commit on branch | ROUND_DONE | review diff, run tier-appropriate tests |
| tests PASS, stop criterion not met | NEXT_PHASE | compose Phase N+1 prompt, dispatch |
| tests PASS, stop criterion met | SHUTDOWN | run shutdown procedure below |
| tests FAIL | RECOVER | open focused fixup spec via `codex-bg.sh --resume`, ONE retry max |
| fixup retry FAIL | ESCALATE | stop, ask JMT |
| `CODEX_FAILED` (quota) | QUOTA_BLOCKED | fall back per chain below |
| `CODEX_FAILED` (other) | INSPECT | read log tail, decide |
| same un-closeable issue 3 rounds in a row | RESIDUAL | accept residual, log, continue or stop |

## Fallback chain (resource limits)

1. Primary: codex via `~/.claude/scripts/codex-bg.sh`
2. Quota blocked: pivot to `Agent(subagent_type="general-purpose", model="opus")`
   with spec adapted (drop XML scaffolding, keep contracts + file:line refs).
3. Both blocked: write `state: BLOCKED` here, ping JMT, schedule wakeup, stop.

NEVER silently stall. NEVER export `OPENAI_API_KEY`.

## Shutdown procedure

When stop criterion triggers:

1. Write `/home/jtaylor/projects/torchlens/.research/super-family-sprint_SUMMARY.md`
   with: phases shipped, key design decisions, residual issues, test
   counts before/after, files touched, commits.
2. Update MEMORY.md to point at the SUMMARY (and remove or update
   `architecture.md`, `data_structures.md` references that this sprint
   superseded).
3. Mark this file: `state: DONE`, append "shutdown" row to log below.
4. Notify JMT via terminal output (NOT iMessage; he's awake).

## Per-phase verification (run after each `CODEX_DONE`)

Mandatory after every phase:

```
git status --short
git log --oneline -3
git diff HEAD~1 --stat
pytest tests/ -m smoke -x --tb=short
```

Plus phase-specific checks:

- Phase 1 (Accessor refactor): `python -c "import torchlens; ..."` smoke
  on every accessor type (LayerAccessor, OpAccessor, ModuleAccessor,
  ParamAccessor, BufferAccessor, GradFnAccessor) returning sensible
  results.
- Phase 2 (Super[T] base): same for `SuperOp` / `SuperLayer` round-trip.
- Phase 3 (SuperAccessor): instantiate `bundle.ops["..."]` /
  `bundle.layers["..."]` and verify `coverage`, `members`.
- Phase 4 (multi_trace move): every old import path either re-exports
  or raises a clear deprecation. Run `grep -r "from torchlens.multi_trace"
  tests/` to confirm test imports updated.
- Phase 5 (new kinds): instantiate each new `bundle.<accessor>["..."]`,
  verify shape + coverage + repr.
- Phase 6 (predicates): `bundle.is_structurally_consistent` returns
  `True` for two clones of the same trace; `False` for two
  structurally-different traces.
- Phase 7 (at-dispatcher): `bundle.at("conv2d_1_1")` resolves; ambiguity
  surfaces a clear error.
- Phase 8: full tier-2 (`pytest tests/ -m "not slow" -x --tb=short`)
  green; docstrings/CLAUDE.md updated.

## Iteration log (append per round)

| Round | Start | End | Commit | Result | Notes |
|---|---|---|---|---|---|
| 1 | 2026-05-09T12:30 | -- | -- | RUNNING | Phase 1: Accessor[T] base extraction; codex pid=3883775; log=/tmp/super-sprint_phase1.log |

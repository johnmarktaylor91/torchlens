---
run: polish-sprint
created: 2026-05-09T15:10:21-04:00
state: DONE
current_round: 5
---

# polish-sprint -- Sprint State

Canonical "where are we" record. Queued behind super-family-sprint
Phase 8. Kicks off when super-family-sprint hits `state: DONE`.

This sprint is **supervised sequential**: each phase dispatches to
codex; on `CODEX_DONE` Claude reviews the diff, runs phase-
appropriate tests, commits, then dispatches the next phase.

## Stop criteria (observable, quantitative)

All of:

1. All 5 phases below shipped (each = own commit on
   `naming-sprint-impl`).
2. Tier-1 smoke: `pytest tests/ -m smoke -x --tb=short` -> all pass.
3. Tier-2 minus pre-existing failures:
   `pytest tests/ -m "not slow" --ignore=tests/test_migrations.py
    --deselect tests/test_param_log.py::TestVisualizationParams
    -x --tb=short -q` -> all pass.
4. `tl.trace(model, x).draw(...)` works on headless (no `xdg-open`
   crash).
5. `bundle.show_diff(...)` renders white→red sequential palette.
6. `op.fx_qualpath` and `op.fx_call_index` populate.
7. `tl.trace(model, x, keep_orphans=True)` populates
   `trace.orphans` accessor.
8. `op.do(transform)` and `layer.do(transform)` work as ergonomic
   shortcuts.

max_rounds: 5 (one per phase)

## Phase plan (canonical reference; do not edit during sprint)

| # | Title | Risk | Effort |
|---|---|---|---|
| 1 | Trace.draw() headless fix + render_graph alias | low | ~30 min |
| 2 | Bundle diff color scale: blue-white-red -> white-red | low | ~20 min + regen |
| 3 | OpLog.fx_qualpath + fx_call_index metadata | low | ~1-2 hours |
| 4 | keep_orphans=True flag + trace.orphans accessor | medium | ~half day |
| 5 | Layer/op-level intervention verb wrappers | medium | ~half day |

Full prompts live alongside as `polish-sprint_phase{N}_PROMPT.md`,
written immediately before each dispatch.

## Wake-up case routing (supervised sequential)

| Observable signal | State | Claude's action |
|---|---|---|
| super-family-sprint state: DONE | READY_TO_START | dispatch Phase 1 |
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
   with spec adapted (drop XML scaffolding, keep contracts +
   file:line refs).
3. Both blocked: write `state: BLOCKED` here, ping JMT, stop.

NEVER silently stall. NEVER export `OPENAI_API_KEY`.

## Shutdown procedure

When stop criterion triggers:

1. Write `/home/jtaylor/projects/torchlens/.research/polish-sprint_SUMMARY.md`
   with phases shipped, key decisions, residual issues, test counts,
   files touched, commits.
2. Update MEMORY.md if any architectural shift is worth flagging
   (likely not — this is polish work).
3. Mark this file: `state: DONE`, append "shutdown" row to log.
4. Notify JMT via terminal output.

## Per-phase verification (run after each `CODEX_DONE`)

Mandatory after every phase:

```
git status --short
git log --oneline -3
git diff HEAD~1 --stat
pytest tests/ -m smoke -x --tb=short
```

Plus phase-specific checks:

- Phase 1 (headless draw): run `trace.draw(...)` in a `DISPLAY=`
  unset shell; confirm no exception, file written.
- Phase 2 (color scale): visually inspect a regenerated bundle diff
  SVG; confirm white→red gradient with no blue.
- Phase 3 (fx_qualpath): `op.fx_qualpath` populated for a sample
  model; format matches `dotted.module.path`.
- Phase 4 (keep_orphans): construct a model with an orphan op
  (e.g., `_ = torch.arange(10).mean()` in forward); confirm
  `trace.orphans` is non-empty when `keep_orphans=True`, empty
  when `keep_orphans=False`.
- Phase 5 (op.do shortcut): `op.do(tl.zero_ablate())` modifies
  the trace identically to `trace.do(op.layer_label, tl.zero_ablate())`.

## Iteration log (append per round)

| Round | Start | End | Commit | Result | Notes |
|---|---|---|---|---|---|
| 1 | 2026-05-09T15:35 | 2026-05-09T15:36 | cb141cf | DONE (smoke 170/170, bundle 78/78) | Phase 1: headless draw + render_graph alias with deprecation warning. |
| 2 | 2026-05-09T15:38 | 2026-05-09T15:42 | da060bf | DONE (smoke 170/170, bundle 78/78) | Phase 2: white->red sequential palette. Snapshots regenerated; no blue. |
| 3 | 2026-05-09T15:44 | 2026-05-09T15:57 | 72cec3c | DONE (smoke 170/170, bundle 78/78) | Phase 3: fx_qualpath + fx_call_index. Slight cosmetic redundancy on trivial modules (relu.relu, encoder.linear); refine in follow-up. |
| 4 | 2026-05-09T16:00 | 2026-05-09T16:19 | 07f15e1 | DONE (smoke 170/170, bundle 78/78) | Phase 4: keep_orphans=True flag + trace.orphans accessor. Verified: 3 orphans retained on synthetic case, default still purges. |
| 5 | 2026-05-09T16:25 | 2026-05-09T16:39 | de2fe34 | DONE (smoke 170/170, bundle 78/78) | Phase 5 (final): op.do / layer.do / op.set / layer.set / op.attach_hooks / layer.attach_hooks shortcuts. Verified equivalent to Trace-level forms. |
| shutdown | 2026-05-09T16:42 | 2026-05-09T16:42 | -- | DONE | SUMMARY written; tier-2 final confirmation in flight; sprint state DONE. |

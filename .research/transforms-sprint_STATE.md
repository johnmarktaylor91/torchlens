---
run: transforms-sprint
created: 2026-05-09T16:51:45-04:00
state: DONE
current_round: 5
---

# transforms-sprint -- Sprint State

Canonical "where are we" record. Supervised sequential — each phase
dispatches to codex; on `CODEX_DONE` Claude verifies, commits, and
dispatches the next phase.

## Stop criteria (observable, quantitative)

All of:

1. All 5 phases below shipped (each = own commit on `naming-sprint-impl`).
2. Tier-1 smoke green.
3. Tier-2 minus pre-existing failures green.
4. `tl.trace(model, x, transform=fn)` works.
5. `Trace.raw_input` returns the original; rendered SVG includes
   inline raw-input on the input node.
6. `tl.trace(model, x, output_transform=fn)` works.
7. `Trace.raw_output` returns the human-readable form; rendered
   SVG output node includes the labels/decoded text.
8. At least 3 layer visualizers ship in `tl.viz.*`.
9. Batched-input renders as a montage / table per the sampling
   policy.
10. `tl.bridge.hf.trace_text(model, "prompt")` is callable.

max_rounds: 5

## Phase plan

| # | Title | Effort |
|---|---|---|
| 1 | `transform=` primitive + raw-input storage + basic input rendering | ~1 afternoon |
| 2 | `output_transform=` + output rendering | ~1 afternoon |
| 3 | Layer visualizers (`tl.viz.heatmap`, `channel_grid`, `histogram`) | ~1 afternoon |
| 4 | Batched-input rendering (montage + table) | ~1 hour |
| 5 | HF bridge `trace_text(model, prompt)` | ~1 hour |

Full prompts live alongside as `transforms-sprint_phase{N}_PROMPT.md`.

## Wake-up case routing

| Observable signal | State | Claude's action |
|---|---|---|
| codex pid alive | RUNNING | ack briefly, yield turn |
| `CODEX_DONE` + commit on branch | ROUND_DONE | review diff, run tier-appropriate tests |
| tests PASS, stop criterion not met | NEXT_PHASE | compose Phase N+1 prompt, dispatch |
| tests PASS, stop criterion met | SHUTDOWN | run shutdown procedure below |
| tests FAIL | RECOVER | open focused fixup spec, ONE retry max |
| fixup retry FAIL | ESCALATE | stop, ask JMT |
| `CODEX_FAILED` (quota) | QUOTA_BLOCKED | fall back per chain below |
| `CODEX_FAILED` (other) | INSPECT | read log tail, decide |

## Fallback chain

1. Primary: codex via `~/.claude/scripts/codex-bg.sh`.
2. Quota blocked: pivot to `Agent(subagent_type="general-purpose", model="opus")`.
3. Both blocked: write `state: BLOCKED`, ping JMT, stop.

NEVER silently stall. NEVER export `OPENAI_API_KEY`.

## Shutdown procedure

When stop criterion triggers:

1. Write `.research/transforms-sprint_SUMMARY.md`.
2. Mark this file: `state: DONE`.
3. Notify JMT via terminal output.

## Iteration log

| Round | Start | End | Commit | Result | Notes |
|---|---|---|---|---|---|
| 1 | 2026-05-09T16:55 | 2026-05-09T17:27 | c466ffe + 7fa8a8f | DONE (smoke 170/170) | Phase 1: transform= primitive + raw_input + input rendering. Codex shipped the primitive; manual fixup `7fa8a8f` restored raw_input across rerun atomic swap (codex's commit nulled raw_input on rerun). |
| 2 | 2026-05-09T17:32 | 2026-05-09T18:53 | 0bf0004 | DONE (smoke 170/170, bundle 78/78) | Phase 2: output_transform= + raw_output + output-node rendering. Slow due to dual-pytest contention (~80 min) but verified clean. |
| 3 | 2026-05-09T18:55 | 2026-05-09T19:15 | 81d3457 | DONE (smoke 170/170, bundle 78/78) | Phase 3: tl.viz.heatmap / channel_grid / histogram + layer_visualizers kwarg + per-op visualizer_path + SVG image embed. |
| 4 | 2026-05-09T19:18 | 2026-05-09T19:36 | ceffa08 | DONE (smoke 170/170, bundle 78/78) | Phase 4: batched raw-input rendering (text table with +N more, image montage, override knob). |
| 5 | 2026-05-09T19:39 | 2026-05-09T19:58 | aee7dd6 | DONE (smoke 170/170, bundle 78/78, live GPT-2 OK) | Phase 5 (final): tl.bridge.hf.trace_text. Live HF round-trip verified. |
| shutdown | 2026-05-09T20:01 | 2026-05-09T20:01 | -- | DONE | SUMMARY written; sprint state DONE. |

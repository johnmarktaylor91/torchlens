---
run: rename-sprint-2.0
created: 2026-05-21T17:35:08-04:00
state: INIT
current_round: 0
branch: rename-sprint-2.0
---

# rename-sprint-2.0 -- Autonomous Loop State

This file is the canonical "where are we" record for this run.
Every wake-up event (watcher fire, user ping, schedule trigger) must
read this file FIRST and act on the case routing below given the
current observable state. Do not act on intuition; act on the case.

## Stop criteria (observable, quantitative)

All of the following must hold for the run to be considered DONE:
1. Every locked decision in `.project-context/glossary_walkthrough_deltas.md` (51 entries dated 2026-05-21) has been applied to the codebase.
2. Smaller API changes locked alongside renames are applied (0-based accessor indexing, short+long Layer-label form acceptance, universal accessor behavior consistency).
3. `pytest tests/ -m smoke -x --tb=short` PASSES.
4. `pytest tests/ -m "not slow" -x --tb=short` PASSES.
5. `ruff check .` PASSES.
6. `mypy torchlens/` does not regress (compare to baseline at branch start).
7. Branch `rename-sprint-2.0` is merged to local `main` (fast-forward, no push).
8. Fresh definitive glossary written to `<vault>/brain/projects/torchlens/reports/2026-05-21-glossary-v4-definitive/torchlens_glossary.md` reflecting current code state.
9. iMessage sent to JMT confirming completion with one-line result.
10. `pytest tests/ -m "not slow" -x --tb=short` PASSES at very end (user explicit ask: "all tests except super long ones").

max_rounds: 14

## Source-of-truth documents (every round reads these FIRST)

| File | Role |
|---|---|
| `<vault>/2026-05-19-rename-sprint-glossary-v3/torchlens_glossary.md` | The locked rename surface. 1943 lines. Every field/class/method name is canonical here. |
| `.project-context/glossary_walkthrough_deltas.md` | 51 lock entries explaining WHY each decision was made. Read when ambiguity arises. |
| `MEMORY.md` (CC memory dir) + repo `CLAUDE.md` | Project conventions, anti-patterns, test layout. |

Vault path resolves to `/home/jtaylor/Documents/Second Brain/`. Concrete absolute path for the glossary:
`/home/jtaylor/Documents/Second Brain/brain/projects/torchlens/reports/2026-05-19-rename-sprint-glossary-v3/torchlens_glossary.md`

## Phase plan (one phase per round)

Each phase is one Codex round. Phase N must commit before phase N+1 dispatches.

| Phase | Subject | Scope summary |
|---|---|---|
| P1 | Top-level capture API + Trace fields | `forward` vs `rerun` decision honored, `code_context`, `num_modules`, `ops_with_params`, `model_object_id`, `equivalent_ops` removal, `num_streamed_passes` removal, `total_autograd_memory`, `op_equivalence_classes` (renamed from trace-level `equivalent_ops`), Trace `state/state_history/last_run` left alone but documented, capture_config left alone |
| P2 | Op + Layer renames | `grad_fn` (was `grad_fn_log`), `grad_fn_object_id`, `in_multi_output` (was `is_multi_output`), `has_out_variations` (was `has_output_variations`), `in_atomic_module` + `atomic_module_address` + `atomic_module_label`, `is_atomic_module` (drop `_op` suffix), `out_device` (was `output_device`), `autograd_memory` (was `gradient_memory`), `num_param_tensors`, `is_compute_op` description correction, removal of `module_calls_entered` redundancy with `input_to_module_calls`, Op vs Layer parity polish |
| P3 | Module + ModuleCall renames | ModuleCall `class_name`/`class_qualname` parity, `call_parent` + `call_children` (both `_addresses` and resolver), `inputs`/`outputs` as bare label lists (drop `_labels` suffix), `has_frozen_params`, `forward_duration`, `code_context`, `module_call_stack`, hook lists as `list[HookInfo]`, `forward_args_summary` + `forward_args_template` + `forward_kwargs_summary` + `forward_kwargs_template` parity confirmation, removal of `ModuleCall.modules` |
| P4 | Param + Buffer + GradFn + GradFnCall renames | Param `used_by_ops` + `used_by_layers` (drop `_labels`), `object_id` standardization, GradFn type_index single source of truth, GradFn forward orientation honoring autograd direction, `duration` -> contextualize on GradFn, recurrent-loop semantics deferred to backward sprint (no change here) |
| P5 | Bundle + Super[T] + ConditionalRecord renames | Super[T] `outs` decision (lock per glossary), ConditionalArm `evaluation_ops` + `execution_ops` (bare lists, drop `_labels`), ConditionalAccessor accepts integer + label, `conditional_arm_entry_edges` honoring two-distinct-op invariant (file bug if not, do NOT silently fix here) |
| P6 | Accessor API rewrite | Universal: 0-based positional integer indexing for ALL scoped accessors. Short+long Layer label form acceptance (`conv2d_2:1` resolves when unambiguous). `trace.layers[label]` always returns Layer (strict type). Universal lookup behavior audit |
| P7 | Test suite + tier-2 green | Update `tests/` for every rename; smoke + tier-2 must pass; ruff clean; mypy non-regress |
| P8 | Definitive glossary + merge | Regenerate `<vault>/.../2026-05-21-glossary-v4-definitive/torchlens_glossary.md`. Merge branch to local main. Send iMessage. |

## Wake-up case routing

When a wake-up event fires, use this table BEFORE doing anything else:

| Observable signal | State | Action |
|---|---|---|
| codex pid alive | RUNNING | ack briefly, yield turn (do not poll); next event will fire |
| codex exited cleanly + HEAD advanced + tier-1 smoke green | ROUND_DONE | bump `current_round`, dispatch next phase per phase plan |
| codex exited cleanly + HEAD advanced + tier-1 smoke FAIL | TIER1_BROKE | open targeted fixup spec via `codex --resume` (one retry max), then re-verify |
| codex exited cleanly + HEAD NOT advanced (no new commit) | NO_PROGRESS | inspect dispatch log, write fixup spec restating scope + completeness contract, resume |
| codex hit quota (CODEX_FAILED with usage_limit reason) | QUOTA_BLOCKED | fall back per "Fallback chain" below |
| same un-closeable issue 3 rounds in a row | RESIDUAL | accept as residual, log in iteration log, continue or shut down |
| post-round verification PASS, stop criterion met | SHUTDOWN | run shutdown procedure below |

## Fallback chain (resource limits)

1. Primary: codex via `~/.claude/scripts/codex-bg.sh` + `codex-watch.sh`.
2. Quota blocked: pivot to Agent(subagent_type="general-purpose", model="opus") with the spec adapted (drop XML scaffolding, keep contracts + file:line refs). All phases here are <500 LOC effective edits at the spec level, plus the broad rename touch — opus subagent can handle each phase.
3. Both blocked: write `state: BLOCKED` here, send iMessage to JMT "rename-sprint-2.0 blocked, will resume <reset-time>", schedule a wakeup, stop.

NEVER silently stall. NEVER export OPENAI_API_KEY to "work around" quota.

## Anti-flail rules

- 3 failed phase attempts on same issue → declare residual, log it, move on.
- Test suite degraded by >5% relative to baseline → STOP phase, fix before continuing.
- Codex output diff >5000 lines without commit → STOP, ask for plan adjustment.
- Phase 7 (test fixes) is the only phase allowed to mass-edit tests; earlier phases must leave tests broken and let P7 fix.

## Shutdown procedure (mechanical -- user is asleep)

When stop criterion triggers, run these in order:

1. Verify smoke + tier-2 green one final time.
2. Compare full pytest results to branch-start baseline; document any net-new failures.
3. Write `/home/jtaylor/projects/torchlens/.research/rename-sprint-2.0_SUMMARY.md` covering: phases run, commits per phase, LOC changed, tests passing, residual issues if any.
4. Regenerate fresh definitive glossary at `/home/jtaylor/Documents/Second Brain/brain/projects/torchlens/reports/2026-05-21-glossary-v4-definitive/torchlens_glossary.md`. Treat it as new/fresh — no need to enumerate change history, just be canonical.
5. `git checkout main && git merge --ff-only rename-sprint-2.0`. If non-ff, STOP and ask (do not force).
6. Mark this file: `state: DONE`, append last round = "shutdown" to log.
7. Send final iMessage via `~/.claude/scripts/send-to-jmt.sh`: "rename-sprint-2.0 DONE. <one-line result>. <N> phases. <commit-count> commits. <vault-glossary-path>."

DO NOT push. DO NOT open a PR. JMT explicitly said "merge it to local main" and "dont push it yet".

## Iteration log (append per round)

| Round | Start | End | Commit | Score / Result | Notes |
|---|---|---|---|---|---|
| 1 | 2026-05-21T17:35:08-04:00 | 2026-05-21T17:50:41-04:00 | 31e2d95 | OK | 123 files; 1193+/1202- |

| 2 | 2026-05-21T17:50:41-04:00 | 2026-05-21T18:08:08-04:00 | f32a68e | OK | 90 files; +464/-370 |

| 3 | 2026-05-21T18:08:08-04:00 | 2026-05-21T18:25:40-04:00 | pending | OK | trace capture-only API, pass vocabulary, total memory prefixes; smoke + ruff green |

| 4 | 2026-05-21T18:25:40-04:00 | 2026-05-21T18:48:05-04:00 | pending | OK | Trace field renames/additions/removals; exact old-name grep clean; import/dataclass probes, ruff, and smoke green |
| 5 | 2026-05-21T18:48:05-04:00 | 2026-05-21T19:01:06-04:00 | ebda8d5 | OK | Op + Layer field renames and grad_fn split; ruff and smoke green; `num_ops` left as Trace/Layer canonical per glossary |

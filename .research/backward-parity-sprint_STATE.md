---
run: backward-parity-sprint
created: 2026-05-11T20:16:52-04:00
state: DONE
current_round: 15
plan_ready_at: 2026-05-12T03:55-04:00
completed_at: 2026-05-12T08:07:19-04:00
critique_rounds_completed: 14
final_plan_lines: 12109
total_findings_closed: 80+
residuals_doc: .research/backward-parity-sprint_RESIDUALS.md
summary_doc: .research/backward-parity-sprint_SUMMARY.md
next_action: JMT review and manual merge
---

# backward-parity-sprint -- Autonomous Loop State

This file is the canonical "where are we" record for this run.
Every wake-up event (watcher fire, user ping, schedule trigger) must
read this file FIRST and act on the case routing below given the
current observable state. Do not act on intuition; act on the case.

## Stop criteria (observable, quantitative)

Both labs return READY on adversarial critique; plan covers α.3 + a-e + F1-F5 strong; implementation phases sequenced; then mega-sprint dispatches autonomously

max_rounds: 10

## Wake-up case routing

When a wake-up event fires, use this table BEFORE doing anything else:

| Observable signal | State | Action |
|---|---|---|
| codex pid alive | RUNNING | ack the user, yield turn (do not poll); next event will fire |
| codex exited cleanly + worktree advanced (HEAD changed) | ROUND_DONE | run post-round verification (tests / lint / project-specific) |
| post-round verification PASS, stop criterion not met | NEXT_ROUND | dispatch round N+1 with refreshed prompt |
| post-round verification PASS, stop criterion met | SHUTDOWN | run shutdown procedure below |
| post-round verification FAIL | RECOVER | FILL_ME_IN -- e.g. open targeted fixup spec to codex --resume |
| codex hit quota (CODEX_FAILED with usage_limit reason) | QUOTA_BLOCKED | fall back per "Fallback chain" below |
| same un-closeable issue 3 rounds in a row | RESIDUAL | accept as residual, log it, continue or shut down |

## Sprint phases

### Phase 1: Research push (parallel, redundant labs)
- 5 subtopics x 2 labs = 10 parallel dispatches:
  - `alpha3` -- α.3 hot-path capture finish
  - `viz` -- combined forward+backward viz + module clustering
  - `intervention` -- intervening capture + selector DSL + intervention parity
  - `validation` -- validation audit + observer parity + bundle backward viz
  - `capture` -- gradient_postfunc + fastlog gradient + gradient stats
- Codex: `~/.claude/scripts/codex-bg.sh` x 5 with `.research/backward-parity_<sub>_PROMPT.md`
- Claude: Agent(subagent_type="general-purpose", model="sonnet") x 5 with same prompts
- Each writes to `.research/backward-parity_<sub>_{CODEX,CLAUDE}.md`

### Phase 2: Synthesis brief
- I (the orchestrator) merge all 10 research outputs into
  `.research/backward-parity-sprint_SYNTHESIS_BRIEF.md`.
- Resolve disagreements, capture user constraints, list open
  autonomous decisions.

### Phase 3: Plan v1
- Write `.research/backward-parity-sprint_PLAN.md` v1.
- Include: final architecture, implementation phases, parity gates,
  test strategy, file:line refs to current code.

### Phase 4: Adversarial critique loop (redundant labs per round)
- Per round: Claude Opus + Codex high-effort run critique in parallel,
  write to `_PLAN_CRITIQUE_v<N>_{CLAUDE,CODEX}.md`.
- Integrate findings into Plan v<N+1>.
- Continue until both labs return READY (no substantive findings).

### Phase 5: When plan is READY
- iMessage JMT: "Plan READY. Dispatching impl. Will iMessage per phase."
- Begin autonomous impl phases.

### Phase 6: Autonomous implementation
- Per phase: dispatch codex via codex-bg.sh + Monitor watcher.
- iMessage JMT after each phase commits.
- On CODEX_DONE, verify, dispatch next phase.

## Implementation phase plan (refined during planning)

Sequenced impl phases (to be finalized in Plan v1):
- P1: α.3 capture finish (clears the deck)
- P2: Combined forward+backward viz (a)
- P3: Module clustering for backward viz (b)
- P4: Backward intervention selector DSL + intervening capture (c + e + F2)
- P5: gradient_postfunc + fastlog gradient (F1 + F5)
- P6: Backward validation audit + observers + bundle viz (d + F3 + F4)
- P7: Gradient stats / aggregate (F6)
- (stretch) P8: Backward Super* audit (F7)

Edit this table with run-specific cases. The defaults are minimums.

## Fallback chain (resource limits)

1. Primary: codex via /codex:rescue --background OR codex-bg.sh
2. Quota blocked: pivot to Agent(subagent_type="general-purpose", model="opus")
   with the spec adapted (drop XML scaffolding, keep contracts + file:line refs).
3. Both blocked: write `state: BLOCKED` here, send iMessage to JMT
   "blocked, will resume <reset-time>", schedule a wakeup, stop.

NEVER silently stall. NEVER export OPENAI_API_KEY to "work around" quota.

## Shutdown procedure (mechanical -- user is asleep)

When stop criterion triggers, run these in order:

1. FILL_ME_IN -- write final SUMMARY.md (path: `/home/jtaylor/projects/torchlens/.research/backward-parity-sprint_SUMMARY.md`)
2. FILL_ME_IN -- run any visualization / artifact generation
3. FILL_ME_IN -- send images / artifacts via `~/.claude/scripts/send-to-jmt.sh`
4. Mark this file: `state: DONE`, append round N+1 = "shutdown" to log.
5. Send a final iMessage: "Run backward-parity-sprint done. <one-line result>. <count> artifacts shipped."

No "figure out what to do at the end" -- spell it out concretely BEFORE
round 1 dispatches.

## Iteration log (append per round)

| Round | Start | End | Commit | Score / Result | Notes |
|---|---|---|---|---|---|
| Impl P1-P7 | 2026-05-12T03:55-04:00 | 2026-05-12T08:07:19-04:00 | 31d20a8 | DONE | Six implementation phases plus final wrap-up completed: alpha.3 LiveOpRecord materialization, backward validation hardening, combined forward/backward viz, backward selector/helper parity, fastlog gradient capture, gradient stats, backward observers, bundle viz regression coverage, and final T2 sweep. Summary: `.research/backward-parity-sprint_SUMMARY.md`. |

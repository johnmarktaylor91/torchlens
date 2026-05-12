---
run: post-backward-megasprint
created: 2026-05-12T08:15:17-04:00
state: DONE
current_round: 0
resumed_at: 2026-05-12T09:00-04:00
completed_at: 2026-05-12T18:11-04:00
---

# post-backward-megasprint -- Autonomous Loop State

This file is the canonical "where are we" record for this run.
Every wake-up event (watcher fire, user ping, schedule trigger) must
read this file FIRST and act on the case routing below given the
current observable state. Do not act on intuition; act on the case.

## Stop criteria (observable, quantitative)

Both labs READY on adversarial critique; plan covers all 5 subtopic items; impl phases sequenced; per-phase commits + iMessage; final T2 green

max_rounds: 14

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

1. FILL_ME_IN -- write final SUMMARY.md (path: `/home/jtaylor/projects/torchlens/.research/post-backward-megasprint_SUMMARY.md`)
2. FILL_ME_IN -- run any visualization / artifact generation
3. FILL_ME_IN -- send images / artifacts via `~/.claude/scripts/send-to-jmt.sh`
4. Mark this file: `state: DONE`, append round N+1 = "shutdown" to log.
5. Send a final iMessage: "Run post-backward-megasprint done. <one-line result>. <count> artifacts shipped."

No "figure out what to do at the end" -- spell it out concretely BEFORE
round 1 dispatches.

## Iteration log (append per round)

| Round | Start | End | Commit | Score / Result | Notes |
|---|---|---|---|---|---|
| shutdown | 2026-05-12T17:24-04:00 | 2026-05-12T18:11-04:00 | pending-finalize-docs | DONE | Full non-slow T2 green; scoped ruff, mypy, and smoke green. |

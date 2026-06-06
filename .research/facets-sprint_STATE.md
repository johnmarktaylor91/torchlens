---
run: facets-sprint
created: 2026-06-05T14:30:00-04:00
state: RUNNING
current_stage: STAGE_4_P3_BUILD
result: ""
---

# facets-sprint — Autonomous Loop State

READ FIRST on every wake-up. Triage before acting: `git log -5 --oneline`, `pgrep -f "codex exec"` /
check registered runs, `git branch`, read findings/log files, then act per case routing.

## Mission
Build the facets system per `.research/facets-sprint-plan.md` (design locked in `.research/facets-proposal.md`),
AUTONOMOUS to the end unless a freezing blocker. JMT directive: "plz cook. write a plan, iterate with
adversarial claude and codex till they satisfied, dispatch and implement, review changes and merge to local
main. text me as you finish each stage. if u hit a freezing blocker let me know. add tests, documentation,
glossary entries. make/update a notebook to demonstrate all features."

## Constraints (LOCKED)
- Stay 2.x.x (minor bump per phase, NO major). No backward-compat (facets unused). Validation tripwire SACRED.
- Per-phase: branch -> validated -> independent review -> merge to LOCAL main. NO stacked PRs. No AI attribution.
- Each phase ships tests + docs (`docs/facets.md`) + glossary entries IN THE SAME CHANGE. Notebook at the end.
- iMessage JMT at the end of EACH STAGE (`~/.claude/scripts/send-to-jmt.sh`). On a freezing blocker: iMessage + stop.

## Stages
- STAGE 1 (PLAN REVIEW): dual-lab adversarial review of the plan (Claude Opus agent + Codex), iterate till BOTH
  satisfied it's buildable/bulletproof. Reconcile findings -> update plan/proposal. iMessage when locked.
- STAGE 2 (P1 build, DE-RISKED v2): P1a trace-owned registry snapshot + specificity + expose container names
  via `.facets` (typed item access) + verify/lock LSTM regression; P1b FacetSpec ABI + transform capability
  classes (bijective/selection/aliasing/computed) + op-anchored READ+GRAD + `MissingGradient` contract; P1c
  recipe migration + capability inventory (fail-closed). NO reconstruction/intervention/paired-grad_fn.
  Codex P1 build pid 1723350, log /tmp/facets_p1_build_run.log, watcher bym4sj0l1, branch feat/facets-p1.
- STAGE 3 (P2 build): facet INTERVENTION — scatter-back write-back + conflict/GQA-alias policy + tl.head/
  tl.facet selectors + per-module eager-for-edit. Build -> review -> merge. iMessage.
- STAGE 4 (P3 build): attention RECONSTRUCTION (reconstruction-ready capture, anchor-to-SDPA-input, per-facet
  validation target) + recipe expansion + residual/head-output + module-path fallback + TL aliases. iMessage.
- STAGE 5 (P4 build): patching/attribution helpers. iMessage.
- STAGE 6 (NOTEBOOK + wrap): demonstrate-all-features notebook + final docs sweep. iMessage final.

## Dispatch tracking (update each round)
- STAGE 1: Codex plan-critic pid=? log=/tmp/facets_plan_review_codex.log -> findings .research/facets_plan_review_codex.md
           Claude plan-critic = background Agent -> findings .research/facets_plan_review_claude.md

## Wake-up case routing
| Observable | State | Action |
|---|---|---|
| both plan-critics done | STAGE 1 | reconcile BLOCKING/MAJOR; if real holes -> update plan, optional 2nd round (max ~2); else lock plan -> STAGE 2 dispatch P1 |
| codex pid alive | RUNNING | yield; don't re-dispatch |
| phase build CODEX_DONE | REVIEW | independent verification (run gates myself, hand-check behavior, tripwire intact); fix gaps (1 codex --resume max, else inline) |
| phase green + reviewed | MERGE | merge to local main, sweep branch, iMessage, advance stage |
| build FAILED/quota | RECOVER | log tail; quota -> Agent(opus) fallback; else 1 codex --resume; never weaken validation |
| same blocker 3 rounds | BLOCKER | iMessage JMT with specifics, set state BLOCKED, stop |

## Fallback chain
codex-bg.sh -> quota -> Agent(general-purpose, opus) with the phase spec -> both blocked: BLOCKED, iMessage, stop.
NEVER export OPENAI_API_KEY. NEVER weaken validation to pass.

## Stop criteria (observable)
P1-P4 merged to local main; all stress models `validate_forward_pass` green; facet read/grad/intervene work on
real models; ruff+mypy+smoke+not-slow green each phase; docs/facets.md + glossary + tests shipped; demo notebook runs.

## Shutdown
1. Confirm all phases merged + branch swept + tree clean. 2. Notebook runs. 3. Write `.research/facets-sprint_SUMMARY.md`.
4. iMessage JMT final. 5. Mark state DONE.

## Iteration log
| Round | Stage | Result | Notes |
|---|---|---|---|
| 1 | STAGE 1 dispatched | plan critics launched | Codex + Claude Opus adversarial on the locked plan/proposal. |
| 2 | STAGE 1 round-1 done | BOTH found blocking, CONVERGENT | grad/args not default; LSTM+container already done; recipes-run-not-record; registry-snapshot; RoPE anchor; GQA/shared-home write-back; drop paired-grad_fn. Findings: facets_plan_review_{claude,codex}.md. |
| 3 | STAGE 1 plan REVISED v2 | sprint-plan.md rewritten | 11 reconciled decisions; capability classes; FacetSpec ABI-first; de-risked P1 (no reconstruction/intervention/paired-grad_fn); reconstruction->P3. Round-2 confirm dispatched: Codex pid 1716359 + Claude agent ab56c8c6 -> facets_confirm_{codex,claude}.md. |
| 4 | STAGE 1 DONE (both SATISFIED) | plan LOCKED | Round-2: both labs SATISFIED, 0 blockers; folded 3 polish notes (MissingGradient returns-not-raises; capability inventory as data; multi-pass home-op selection). iMessage sent. STAGE 2 dispatched: P1 build codex pid 1723350. |
| 5 | STAGE 2 P1 DONE | merged to local main | Commits 77b63c2/7df3213/79581fc (ff) + glossary c9ef746. Independently verified: read shape, facet.grad default->MissingGradient + captured->correct slice, GQA write=False, registry snapshot, structural naming, capability inventory as data. Gates: facets 28, ruff+mypy clean, smoke 223, not-slow 2358 (codex). Tripwire untouched. docs/facets.md + glossary done. Branch swept. iMessage sent. |
| 6 | STAGE 3 P2 DONE | merged to local main | Commit 9221e30 (ff). Independently verified: head ablation CHANGES output; build tests (9) assert edit-changes-output+validates, GPT2 c_attn compose+conflict, GQA/computed/in-place writes REFUSED, whole-model ablation reruns+validates. TRIPWIRE: P2's validation carve-out (skip func-replay of `intervention_replaced` ops) is NARROW + correct — provably inert on plain capture (0 flagged ops), downstream still validated, aligns w/ the 2026-06-02 narrow-exemption discipline. Gates: facets interven 9, smoke 223, not-slow 2364 (codex). docs+glossary done. Branch swept. iMessage sent. |
| 7 | STAGE 4 P3 DONE | merged to local main | Commit 445e20a (ff). Independently verified P3 tests (9, exact matrix): reconstruction anchors post-RoPE SDPA inputs + validates; NON-VACUOUS (corrupt recon -> FAIL); GQA+causal mask; MissingFacet names prereq; residual resid_pre/mid/post read+grad; module-path fallback; TL aliases opt-in; entry-point fail-safe. Tripwire untouched (no validation files). `tl.trace(reconstruction_ready=True)` + `enable_transformerlens_aliases()`. Gates: semantic 43, smoke 223, not-slow 2373. docs+glossary done. Branch swept. iMessage sent. **RESIDUAL: auto-scoped-eager-on-edit NOT wired — fused-internal pattern intervention fail-closes + names the eager prereq (capability exists via manual eager capture + P2; auto-trigger deferred). Documented; fast-follow candidate.** |
| 8 | STAGE 5 P4 dispatched | RUNNING | feat/facets-p4: patching/attribution helpers (activation-patching residual/attn-out/per-head clean-vs-corrupted -> [layer,head]; attribution-patching grad*(clean-corrupted) via facet grads). Then STAGE 6: notebook + wrap. |

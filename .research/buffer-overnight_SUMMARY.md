# Buffer Overnight — SUMMARY (for JMT, morning of 2026-06-05)

## TL;DR

**Your promise was never actually broken.** The scary "recurrent buffer fails validation"
was a one-line bug in the *validation harness*, not in capture or replay. It's **FIXED and
merged to local main** (`39a5029`). The buffer **write-version data model** turned out to be
a genuinely hard, separate capture-pipeline project — I researched it with 4 agents, ran it
through 2 rounds of dual-lab adversarial review, and the gauntlet produced a **much better,
lower-risk design** than we'd sketched. I did NOT rush that build overnight (it's a
materially-changed data model on the wrapper hot path, and it needs two decisions from you) —
it's handed over as a validated plan ready for a careful session.

## What SHIPPED (merged to local main, not pushed)

**`39a5029` fix(validation): snapshot ground-truth outputs before state restore.**
- Root cause (4 agents unanimous, file:line): `validate_forward_pass` saved the ground-truth
  output tensor *by reference*, then called `model.load_state_dict` (which writes buffers
  in-place). When a model returns a buffer it reassigned (`return self.h`), the saved ground
  truth WAS that buffer object and got clobbered to its initial value on restore — so the
  validator compared the *correct* captured output against a corrupted `[0,0,0]`. A
  false-negative; capture + full replay were always correct.
- Fix: `.detach().clone()` each ground-truth output before the restore (mirrors the existing
  input deep-copy). Recurrent-state buffer now validates True. **174 validation/buffer tests +
  223 smoke + mypy all green; zero regression.** Regression test added.
- This is the piece you actually cared about: **perfect-replay validation is now honest.**

## The big finding: the promise does NOT depend on the version data model

Replay already survives buffer mutation (BatchNorm, in-place, reassignment all validate True)
via `out_versions_by_child` (`ops.py:1542`), which records post-mutation parent values used in
replay. So the buffer write-version *data model* (versions, `num_overwrites`, history) is a
**richer-introspection enrichment, not a correctness/replay requirement.** Replay was never the
gap.

## The write-capture data model: researched, adversarially hardened, DEFERRED

The model DOESN'T currently expose writes as versions (`num_overwrites=0`, only the initial
buffer node). Making it real is hard. Two independent adversarial reviews (Codex + Claude, 6 +
5 blocking, all measured) demolished the obvious approaches and converged on a better one:

- **`_version`-diff detection is a systematic false-negative on fused BatchNorm** — the native
  kernel writes running-stat storage directly without bumping the autograd version (measured
  both labs, torch 2.8). Can't be the write signal.
- View/slice/`.data` aliases of a buffer don't carry buffer tags; a global `__setattr__`
  monkeypatch leaks across untraced modules; minting new identity version-nodes double-models
  the writer op's `out` and can silently corrupt replay.

**The gauntlet's recommended design (validated, lower-risk) — see `.research/buffer-sprint/`:**
1. Build the version chain as a **VIEW over `out_versions_by_child`** (value-diff via
   `tensor_nanequal` catches the BatchNorm write that `_version` misses), under
   `save_arg_values=True` as a stated precondition.
2. **The existing mutating op IS the version producer** — no new identity nodes.
3. **Module-EXIT buffer re-scan** for reassignment — no global monkeypatch.
4. Keep "node only if read" for the graph; unread-write history lives on the entity.
5. Sequence the capture work before locking the dual-label scheme.

**Two decisions for you before that build:** (a) is `save_arg_values=True` an acceptable
precondition for full buffer history (memory cost), or should buffer-value-diff be captured
independently? (b) the dual-label / two-loop-split question must be RE-MEASURED once writes are
real (the spec-v1 table was built on a false premise). Both are noted in the plan.

## Other gaps found (documented, not fixed overnight — load-bearing/narrow)

- **Static buffer as a model's SOLE output** (`return self.b`, nothing else) → no output layer
  captured → `MetadataInvariantError`. Real bug, but the fix is in load-bearing postprocess
  (output-creation/orphan logic). Filed as a todo. Repro: `/tmp/orphan_diag2.py`.
- **Non-registered mutable state** (a plain `self.state=[tensor]` list mutated across the run)
  isn't in `state_dict`, so validation can't reset it → false-negative. Narrow/non-idiomatic;
  real models use `register_buffer` (which validates correctly). Filed as a todo.

## Where everything lives
- Shipped fix: commit `39a5029` on local main.
- Research (4 agents) + plans + 2 adversarial-review rounds:
  `.research/buffer-sprint/{RESEARCH_*,SPEC.md,SCOPE_NOTE.md,PLAN_PHASE2.md,PLAN2_REVIEW_*}.md`
- Run state: `.research/buffer-overnight_STATE.md`. Todos: `.project-context/todos.md`.

## Recommended next steps (your call)
1. Build the gauntlet's validated design (out_versions_by_child view) — exciting, lower-risk,
   well-scoped. Needs decisions (a) and (b) above first.
2. Fix the static-sole-output orphan bug (small, but careful postprocess change).
3. The entity API (truthful `num_overwrites`/`versions`, alias discovery) layers on (1).

Net: the thing that scared you is fixed and proven; the rest is a clean, validated runway —
nothing rushed onto the hot path while you slept. Revert anything freely; it's all on local
main, unpushed.

# buffer-build — SUMMARY (2026-06-05)

**Result: SHIPPED to local main. Buffer write-capture + version data model complete.**
Not pushed (64 commits ahead of origin/main); no version bump; branch swept (main is sole branch).

## What shipped
Codex built `feat/buffer-datamodel` per `PLAN_v5_BUILD.md` (option-2, capture-at-the-moment),
fast-forward merged to local main. Then an independent review round (this session) verified the
tripwire, proved gradient safety, and root-caused + fixed a real crash codex's tests missed.

Commits (local main):
- `91e1645` capture registered buffer write events (scoped class `__setattr__` + storage snapshot + fused value-snapshot)
- `18b16c5` synthesize buffer version nodes (producer -> version -> readers; `is_buffer` Op flag)
- `5c292f4` persistent `Buffer` entity (Module/Param-sibling; `Buffer(Op)` retired)
- `6e3a291` buffer datamodel integration tests (16)
- `430357a` **(review-found fix)** scrub per-op equivalence lists on removal — fixes RNN-cell loop-detection crash; + gradient-flow tests; + glossary/docs lockstep

## Review verification (independent, this session — did NOT trust codex's word)
1. **Validation tripwire INTACT + strengthened.** The two validation changes are both stricter:
   silent `parent.out` fallthrough now RAISES for a dangling buffer-version parent (source-equality
   guard); a `Buffer` entity with no version nodes now RAISES. Proved NON-VACUOUS: poisoning a
   captured buffer value makes the source-equality guard return False -> raises. No weakening.
2. **Gradient flow VERIFIED.** Recurrent reassignment (non-detached hidden state) + BatchNorm-train:
   grads through a traced model MATCH an untraced run; `backward_ready` path OK. The capture hooks
   are observational — `scoped_setattr` passes the original grad-carrying tensor to the real setattr
   and only records a `.detach().clone()` copy; never replaces the live buffer. 2 regression tests added.
3. **REAL BUG found + fixed.** RNN cell (`self.h = f(cell(x), self.h)` in a loop) CRASHED loop
   detection: `'buffer_1_raw' is not a known raw label`. Root cause = a GENERAL latent bug:
   `_remove_log_entry_references` / `_batch_remove_log_entries` scrub the global
   `op_equivalence_classes` but never the stored per-op `equivalent_ops`/`recurrent_ops` lists
   (FieldPolicy.KEEP), so removing the merged initial buffer node left a dead label in the output
   node's `equivalent_ops`. Fixed in both removal paths (`_scrub_per_op_equivalence_lists`). Codex's
   test model had no inner submodule so loop detection never engaged — the gap. Regression test added.
4. **Glossary + docs lockstep.** Added `buffer_write_kind`, `buffer_value_changed`, `is_buffer`
   (stored flag behind the `is_buffer_source` property), `Module.buffers`. Verified all
   `docs/buffers.md` examples run (num_overwrites==2, value_at/value_after, BN entities).

## Gates (all green)
- ruff: clean. mypy: clean (220 files).
- buffer suite: 19 passed (incl. RNN-cell + 2 gradient tests).
- smoke: 223 passed.
- not-slow: 2340 passed, 28 skipped, 2 xfailed, 0 failures (15m18s) — exercises the core removal path broadly.

## Documented residual edges (by design, NOT bugs)
- `self.buf.data = tensor` reassignment unsupported (reconciliation diagnostic RAISES on unjournaled change).
- Dead intermediate fused write (never read, overwritten before any read) not displayable — computationally inert.
- Non-registered Python-attr state out of scope (non-idiomatic).

## Open follow-on
Layer-vs-op parity: op-side buffer accessors (`buffer_source_ops`/`buffer_sink_ops`) — see todos.md
"Unify buffer layer-vs-op treatment".

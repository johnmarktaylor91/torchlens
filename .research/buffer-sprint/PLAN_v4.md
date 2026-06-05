# Buffer PLAN v4 — Option 2: capture buffer writes AT THE MOMENT

Replaces PLAN_v3 (postprocess synthesis, which both confirm-critics showed can't represent
fused/recurrent writes as validatable graph nodes). v4 captures every buffer WRITE when it
happens, so the version values + producers are real captured data, not postprocess guesses.
Goal: every buffer write becomes a clean, validatable graph version node + a persistent
`Buffer` entity. Replay promise already fixed (P1, 39a5029). Validation = the gate; tripwire
SACRED (no self-feeding, no exemptions).

## The two capture hooks (the whole mechanism)

### 1. Reassignment: scoped `setattr` interceptor
Install a `__setattr__` interceptor on the PREPARED module instances during active capture
(gated by the capture toggle + membership in the prepared set; uninstalled in the capture
`finally`). NOT a global class monkeypatch (that leaked — prior B5). When a registered-buffer
name is rebound to a new tensor, record a write event `(address, producer=new_tensor's op
label, value=new tensor)`.
- KEY: `setattr` fires on ASSIGNMENT, independent of `forward` decoration — so it catches
  `self.h = tanh(...)` at ANY level, **including the top-level recurrent loop** that the
  module-exit approach (v3) could not see (root forward undecorated). This is the fix for
  confirm-round BLOCKING-A.
- Must also cover `self._buffers[name] = t` (direct dict write) — intercept via the same
  prepared-instance mechanism or a buffer-dict proxy. Plain non-registered attrs/lists are
  OUT of scope (non-idiomatic; documented).

### 2. In-place / fused: post-op buffer value-check in the op wrapper
In the existing op wrapper, for any op whose tensor args (including storage-aliasing
views/slices/`.data`) reach a buffer-tagged storage: detect a write by VALUE change.
- Cheap filter: autograd `_version` bump = definitely written (fast positive). But `_version`
  is NOT sufficient (BatchNorm doesn't bump) — so for a KNOWN-FUSED-MUTATOR list
  (`batch_norm`, `instance_norm`, ...) ALWAYS value-snapshot the buffer args pre/post and
  compare. If changed -> record write event `(address, producer=this op, value=post snapshot)`.
- Fused multi-buffer (BatchNorm writes 3): check all buffer args, record in deterministic
  order. This is the fix for the `_version`-false-negative (prior BLOCKING-1) — value
  comparison catches what `_version` misses.
- Cost: bounded to ops that touch a buffer-tagged storage (few). Measure on resnet50 +
  a transformer; `_version` fast-path keeps non-fused in-place cheap.
- Alias coverage (prior B2/D): resolve buffer identity by STORAGE, so a write through a
  view/slice/`.data` of a buffer is attributed to the buffer. Needs a storage->address index.

## Version nodes (from the recorded write events)
- Each write event -> a buffer-version node (plain `Op` + `is_buffer`): parent = the producing
  op (the `tanh`/`mul_`/fused op), `func=identity`, `out` = the recorded value, children =
  the readers of that version. Re-link readers to read vN. Chain v1(initial) -> v2 -> ... per
  address. (Option (a), as chosen.)

## Replay validation — the careful part (fixes confirm BLOCKING-B, NO tripwire weakening)
- **Explicit/reassignment version nodes:** value == the producing op's output -> identity
  replay validates against that op's output (exactly today's captured-buffer passthrough). Real.
- **Fused version nodes:** the value is NOT the fused op's output. BUT replay RE-RUNS
  `batch_norm`, which re-performs the mutation -> the buffer's live state after the fused op
  in replay IS the new value. So validate the fused version node's captured value against the
  buffer's ACTUAL live state after the fused op executes during replay (a REAL check that
  replay reproduces the write) — NOT a self-feed, NOT an exemption. The replay engine must
  read the live buffer post-op for these nodes.
- Convert the silent fallthrough at `validation/core.py:~1000` (`else: parent_values =
  parent.out`) to a RAISE for buffer-version parents, so a mis-link fails LOUDLY (fixes the
  silent-fallthrough trap, prior G).

## Entity (projection over version nodes + write events)
`Buffer`: address/name, module ownership (Param-parallel), shape/dtype/memory, `initial_value`,
`final_value`, `versions` (the nodes, ordered), `num_overwrites` (= write-event count),
`num_versions`, `num_uses`, `used_by_layers`, `value_at(op)`/`value_after(op)`. Accessors:
`trace.buffers`, `trace["addr:N"]`, `buffer.versions`, `Module.buffers`. `compute_ops` excludes
`is_buffer`. Retire `Buffer(Op)` -> name moves to the entity; nodes are plain `Op`+`is_buffer`.

## Decisions (lock)
- **Event-history, not value-history:** each WRITE event is a version (a no-op `b.mul_(1)`
  that the wrapper sees as a write still counts). `num_overwrites` = write-event count. Dedup
  NOT by value. (Confirm-round flagged this needed deciding; event-history is the honest
  "what happened" model and matches "capture everything".)
- `num_batches_tracked` (integer buffer, `+= 1`): it IS mutated by a real tensor op, so it's
  capturable as a normal buffer; ensure `buffer_layers`/accessor/entity agree (no entity with
  `final_value` but empty `versions`).
- Dual-label / two-loop: re-measure once version nodes exist, then lock (sequenced into build).

## Phasing (each phase: commit + every stress model still `validate_forward_pass` True)
- P4a: storage->address index + the two hooks (setattr + op-wrapper value-check) recording
  write events. Verify events fire for all patterns; replay still True.
- P4b: version-node synthesis from events + reader re-link + the fused replay-validation path
  + the fallthrough->raise. Validate all patterns.
- P4c: entity API + accessors + targeted fixes (copy_ source edge, alias discovery,
  num_batches_tracked); retire `Buffer(Op)`; docs-lockstep.
- P4d: exhaustive stress tests + ruff+mypy+smoke+tier2.

## Stress models (gauntlet + tests MUST cover)
recurrent top-level reassignment; BatchNorm1d/2d train (fused 3-buffer); in-place
`mul_`/`add_`/`copy_`; multi-overwrite in one call; same buffer in two loops; view/slice/`.data`
write of a buffer; shared-alias buffers; `num_batches_tracked`; in-place dual-role (read+write
in one call); static read-only buffer.

## What the gauntlet must break
1. `setattr` scoping: re-entrancy, exception safety, no cross-trace leak, catches `_buffers`
   direct writes; does it actually fire for the top-level recurrent loop (the v3 killer)?
2. The fused replay-validation: is "captured value vs live re-run buffer state" actually a
   real, non-vacuous check the replay engine can do? Construct a case where it self-feeds or
   misses.
3. Hot-path cost of the op-wrapper value-check (measure).
4. Storage-aliasing coverage (view/slice/`.data` writes).
5. Loop detection with the new version nodes (recurrent grouping / cycles).
6. Event-history correctness (no-op writes, multi-overwrite ordering, fused multi-buffer order).

# Buffer Write-Capture + Version Data Model — PLAN (Phase 2)

Synthesizes the 4-agent research (RESEARCH_{claudeA,claudeB,codexA,codexB}.md). Phase 1
(validator GT-aliasing fix) already shipped on `feat/buffer-capture` (commit 39a5029) —
perfect-replay validation is now honest. THIS phase builds the buffer WRITE/VERSION data
model. To be adversarially reviewed before building.

## Established truth (4-agent unanimous, measured)

- The recurrent `validate_forward_pass` "failure" was a validator aliasing bug (FIXED, P1).
  Capture + full replay are CORRECT for all tested buffer-mutation patterns.
- Replay survives buffer mutation via `out_versions_by_child` (`ops.py:1542-1566`) — a
  child-specific parent-value record used in replay (`validation/core.py:987-1001`). This is
  why BatchNorm/in-place/reassignment all validate.
- BUT the buffer DATA MODEL is absent: writes flow as ordinary ops; `num_overwrites=0`;
  reassignment -> `buffer_layers==[]`; in-place -> no `b:v2` node; only the INITIAL buffer
  node exists. The persistent `Buffer` entity + version chain does NOT fall out.
- Version nodes are only minted at module re-entry (`_tag_untagged_buffers`,
  `model_prep.py:724`), never on the actual write.

## Goal

Capture buffer WRITES as explicit version nodes (read vN -> op -> write v(N+1)), feeding
BOTH a persistent `Buffer` entity and the graph, WITHOUT breaking replay/validation, with
bounded hot-path cost. This makes "all info about a buffer" + versions real.

## Converged approach (Claude B Rank-1 "B+D" == Codex B Rank-1 hybrid)

### Core: post-op buffer-write DETECTOR (catches fused + in-place)
After a wrapped op executes, for any argument tensor that is buffer-tagged (`_tl.address`
present — already scanned at `wrappers.py:441`), check whether the buffer was WRITTEN by the
op: compare autograd `_version` (fast signal) and/or value vs the pre-op snapshot. If
changed -> a write occurred. Mint a buffer **write-version node** (`v(N+1)`), with the
mutating op as its producer (parent), reuse the Step-6 identity-passthrough machinery, and
route subsequent reads of that buffer through the new version node (read v(N+1)).
- Catches: fused BatchNorm running-stat updates, in-place `mul_`/`add_`/`copy_`.
- Cost: bounded to ops that actually touch a buffer-tagged tensor. `_version` is an
  optimization signal, NOT sole source of truth (Codex B caveat) — confirm with value when
  needed. Measure hot-path delta.
- Fused multi-buffer ordering: BatchNorm writes 3 buffers in one op -> assign deterministic
  version-event order (Codex B).

### Reassignment: narrow `__setattr__`/`register_buffer` hook (active-capture-gated)
`self.h = <new tensor>` rebinds the buffer to a NEW object the wrapper never sees. Intercept
`nn.Module.__setattr__` (and `register_buffer`) DURING active capture only, exception-safe,
no user-semantic change outside capture: when a registered-buffer name is reassigned, tag
the new tensor as a buffer write-version of that address.

### On top: persistent `Buffer` entity + version chain
With writes captured, the version chain (v1 initial -> v2 -> ...) is real. Build the
persistent `Buffer` entity (Module/Param-sibling), version nodes = plain `Op`s (`is_buffer`),
`.buffer` backref. Accessors: `trace.buffers` (entities), `buffer.versions`, `trace["addr:N"]`,
`Module.buffers`. Reference-form per glossary.

### Targeted fixes (from research)
- `copy_` source edge missing (`ops.py` parent extraction) -> the copied-from tensor must be
  a parent of the `copy_` op (Codex A/B).
- Shared-alias discovery: `named_buffers(remove_duplicate=False)` / direct `_buffers` scan,
  grouped by storage identity (Codex B #7) -> `all_addresses` truthful.
- `num_batches_tracked` consistency: `buffer_layers`/`buffer_num_calls`/`buffers`/accessible
  ops must agree (Codex B #8); decide if integer/scalar buffers are entities.
- Static buffer as SOLE output orphan-pruned -> `output_layers==[]` ->
  `MetadataInvariantError` (Claude A) — fix the prune to keep an output-feeding buffer.

## Data-model questions to RE-DECIDE now that writes are captured

These were decided in SPEC v1 under the FALSE premise that writes weren't captured. Revisit:
1. **"Node only if read" vs "full history":** with the write-detector minting a node ON WRITE
   (not on read), an unread final write CAN be a node -> "full history" is now achievable.
   RE-DECIDE the node-creation rule. (Lean: mint on write AND on initial read; full history.)
2. **Dual-label / two-Layer split:** the earlier "buffers collapse to one Layer, dual-label
   fictional" finding was measured BEFORE write-version nodes existed. With real write-version
   nodes per loop position, RE-MEASURE whether the two-loop case splits into Layers and
   whether op-label-pass vs address-global-version actually diverge. Decide the label scheme
   on the NEW reality, not the old.
3. **Node class:** plain `Op` + `is_buffer` (locked) — confirm still right with write-versions.

## Validation = the gate (throughout; tripwire SACRED)
- Every buffer-mutation stress model passes `validate_forward_pass` (now honest) AND backward
  where applicable AND metadata invariants. NEVER weaken a check to pass — root-cause.
- The write-detector ADDS version nodes; replay MUST still pass (the new nodes are identity
  passthroughs feeding the same downstream math). If replay breaks, the detector is wrong.
- Bisect any tier-2 regression against `feat/buffer-capture@39a5029` (post-P1, pre-P2).

## Risks (adversarial review must probe)
- Hot-path cost of post-op buffer-write detection (measure on real models; cap if needed).
- `__setattr__` hook correctness: gating, exception safety, re-entrancy, non-buffer attrs.
- Fused multi-buffer write ordering; `_version` as signal vs ground truth.
- Backward-ready: do NOT detach user tensors just to snapshot versions.
- Loop detection + the new write-version nodes (spurious cycles? correct grouping?).
- Interaction with `out_versions_by_child` (don't double-handle mutation).
- Migration: retire `Buffer(Op)`; docs-lockstep (glossary, CLAUDE/AGENTS, examples, notebooks).

## Phasing (each phase: commit + validate every stress model still replays True)
- P2a: write-detector (post-op `_version`-diff) + `__setattr__` hook -> version nodes + chain.
- P2b: persistent `Buffer` entity + accessors.
- P2c: targeted fixes (copy_ edge, aliases, num_batches_tracked, orphan-prune).
- P2d: stress tests (recurrent-state, BatchNorm, in-place mul_/add_/copy_, multi-overwrite,
  multi-loop, shared-alias, num_batches_tracked, static-sole-output) ALL pass
  validate_forward_pass + invariants; docs-lockstep; ruff+mypy+smoke+tier2 green.

## Honest scope note
P2a (write-capture) is the proposal's flagged "hard part" — a capture-PIPELINE change on the
wrapper hot path. Higher risk than P1. The validation gate + branch isolation + "can revert"
(JMT) bound the downside. If P2a proves to broadly break replay and can't be made clean, the
fallback is: ship P1 (done) + the truthful-accessor/entity read-side (low-risk subset) and
defer write-version capture with a documented honest limitation. Decide by evidence, not hope.

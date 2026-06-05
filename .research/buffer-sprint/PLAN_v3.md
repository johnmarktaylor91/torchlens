# Buffer Sprint — PLAN v3 (revised; the design to confirm + build)

Supersedes `PLAN_PHASE2.md` (v2). v2's core mechanism (hot-path `_version`-diff write
detector + global `nn.Module.__setattr__` patch) was DEMOLISHED by dual-lab adversarial
review (`PLAN2_REVIEW_{codex,claude}.md`, 6 + 5 blocking, all measured: BatchNorm doesn't
bump `_version`; alias/view/`.data` evasion; monkeypatch leak; new-node double-model;
`save_arg_values` off by default; orphan conflict). v3 keeps the GOAL, swaps the MECHANISM
to postprocess synthesis + reuse of already-captured data, per the reviews' converged
recommendation and the 2026-06-05 JMT design session.

This sprint is the buffer DATA MODEL (introspection: a `Buffer` entity + version chain).
**The replay PROMISE is already fixed and merged (P1, `39a5029`)** and does NOT depend on
this. Validation stays the gate; the tripwire is SACRED.

## Settled decisions (JMT, 2026-06-05)

1. **Option (a): dedicated identity buffer-version nodes.** Each version is a node:
   `producer -> buffer-version node (func=identity, holds the value) -> readers`. This
   EXTENDS the existing pattern — a captured buffer write is ALREADY a `func=identity`
   passthrough (`_fix_buffer_layers`). The value living on both the producer and the
   identity node is benign for replay (identity = passthrough, no recompute).
2. **Buffer node = plain `Op` + `is_buffer` flag** (no subclass, no `BufferVersion` class —
   settled earlier; consistent with all other op-roles; role-gated repr hides buffer fields
   on non-buffer ops).
3. **Buffer ENTITY** = persistent Module/Param-sibling, a PROJECTION (built in postprocess)
   over the version nodes + address/module metadata. The name `Buffer` moves from the old
   `Buffer(Op)` subclass to this entity.
4. **Value capture = option (i) + always-snapshot-final:**
   - DEFAULT (`layers_to_save="all"` is already the default -> op OUTPUTS saved):
     explicit-write and reassignment version values are captured (they ARE op outputs).
   - ALWAYS (new, trivial, off-hot-path): snapshot registered buffers at end-of-pass ->
     `final_value` + the final version node's value.
   - OPT-IN (`save_arg_values=True`): the ONLY remaining case — a FUSED buffer write whose
     new value is RE-READ mid-forward (rare; its value lives in `out_versions_by_child`,
     populated only under `save_arg_values`).
5. **Mechanism = POSTPROCESS synthesis, NOT a hot-path detector, NOT a global
   `__setattr__` patch.** Reuse: (a) producing ops as version producers (explicit +
   reassignment); (b) `out_versions_by_child` for fused-read-later values; (c) end-of-pass
   snapshot for fused-final. Reassignment caught via a bounded **module-EXIT re-scan**
   (compare registered-buffer object/value at module exit), not a global patch.
6. Honest residual (documented, accepted): an INTERMEDIATE fused write that is never read
   AND overwritten before any read -> value lost (vanishingly rare).

## Value-source table (how each version's value is obtained)

| Write kind | Value source | Default-captured? |
|---|---|---|
| explicit in-place (`mul_`/`add_`/`copy_`) | the op's own output | yes |
| reassignment (`self.h = tanh(...)`) | the producing op's output | yes |
| fused, final (BatchNorm's last update) | end-of-pass buffer snapshot | yes (new snapshot) |
| fused, re-read mid-forward | `out_versions_by_child` | only w/ `save_arg_values` |
| fused, intermediate, unread, overwritten | (none) | LOST (rare, documented) |

## Build mechanics

### Version nodes (postprocess, per address, execution order)
- v1 = the initial buffer node (already exists when the buffer is read).
- Per write event, synthesize an identity version node: parent = producer (the explicit op,
  or attribution to the fused op, or none for v1); `func=identity`; `out` = the version
  value (from the table); children = the readers of that version.
- Re-link each reader to its correct version node (read vN -> version-node-N).
- Chain v1 -> ... -> vK threaded through the graph.

### Capture changes (the ONLY hot-path-adjacent change is trivial)
- ADD: end-of-pass snapshot of `model.named_buffers()` -> per-address final value. Runs ONCE
  at trace end, off the per-op hot path. That's it for capture.
- Everything else is postprocess (synthesis) + the entity projection. NO per-op detector.

### Reassignment without a global patch
- Module-EXIT re-scan: at each module-call exit, compare current registered-buffer
  object/value to entry; a changed binding = a reassignment write -> attribute the version to
  the producing op already in the graph (e.g. the `tanh`). Bounded per module-call, not a
  global `__setattr__` monkeypatch. (Direct `self._buffers[...]=` is covered by the same
  exit re-scan; plain non-registered list/attr "state" is OUT of scope — non-idiomatic,
  documented.)

### Entity (postprocess projection)
- `Buffer`: address/name, module ownership (Param-parallel), shape/dtype/memory,
  `initial_value`, `final_value`, `versions` (the nodes, ordered), `num_overwrites`,
  `num_versions`, `num_uses`, `used_by_layers`, `value_at(op)`/`value_after(op)`.
- Accessors: `trace.buffers` (entities), `trace["addr:N"]` (version node), `buffer.versions`,
  `Module.buffers`. `compute_ops` excludes `is_buffer`.

### Labels (sequenced; re-measure during build)
- Dual label per node: op-label (pass) + `address:version` (global). The two-loop-split /
  dual-label question was measured on the FALSE premise that writes weren't captured ->
  RE-MEASURE once version nodes exist, then lock the scheme. Not a pre-build blocker.

### Targeted fixes (low-risk, in scope)
- `copy_` source edge (copied-from tensor must be a parent).
- Shared-alias discovery (`named_buffers(remove_duplicate=False)` / `_buffers` by storage id).
- `num_batches_tracked` consistency (bookkeeping vs accessor agreement).

### NOT in scope (separate todos)
- Static-buffer-as-sole-output orphan -> MetadataInvariantError (load-bearing postprocess).
- Non-registered mutable-state validation reset.

## THE replay-safety trap (must be exhaustively tested)
Buffer-version nodes are identity passthroughs, so replay passes values through. BUT the
replay parent-value lookup SILENTLY falls through to `parent.out` on a missing/mis-keyed
link (`validation/core.py` ~1000) -> a mis-linked reader validates against the WRONG value
with NO error. So: (1) exhaustive `validate_forward_pass` tests for every pattern; (2) a
structural invariant that each buffer-version node's value equals its producer/snapshot
value; (3) confirm the new identity nodes don't double-count in metadata/FLOPs/distances
(`compute_ops` already excludes `is_buffer`).

## Phasing (each phase: commit + validate every stress model still True)
- P3a: end-of-pass snapshot + entity scaffold + version nodes for explicit/reassignment
  writes (reuse producing ops) + module-exit reassignment re-scan.
- P3b: fused-write version nodes (synthesize from `out_versions_by_child` + snapshot);
  re-measure two-loop -> lock dual-label.
- P3c: entity API + accessors + targeted fixes (copy_, aliases, num_batches_tracked);
  retire `Buffer(Op)`; docs-lockstep (glossary, CLAUDE/AGENTS, examples, notebooks).
- P3d: exhaustive stress tests (recurrent-state, BatchNorm train, in-place mul_/add_/copy_,
  multi-overwrite, multi-loop, shared-alias, num_batches_tracked, fused-multi-buffer,
  in-place dual-role) ALL pass `validate_forward_pass` + invariants; ruff+mypy+smoke+tier2.

## Stress models the confirm round + tests must cover
BatchNorm1d/2d train (fused multi-buffer); static read-only buffer; buffer read N times in a
loop; recurrent read-modify-write reassignment; in-place `mul_`/`add_`/`copy_`; in-place
dual-role (reads+writes in one call); multi-overwrite in one forward; same buffer in two
loops; shared-alias buffers; `num_batches_tracked`.

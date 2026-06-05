# Buffer Sprint — Round-1 reconciliation + SCOPE DECISION (2026-06-04)

Spec v1 (`SPEC.md`, commit 52c0575) went through dual-lab adversarial review (Codex +
Claude). Findings: `ROUND1_codex_findings.md`, `ROUND1_claude_findings.md` (this dir).
Both: **NOT BULLETPROOF** (Codex 5 blocking, Claude 4). They CONVERGED.

## The convergent finding (empirically proven by both)

**TorchLens does not currently capture buffer WRITES / OVERWRITES.** The spec's load-bearing
premise — "the sprint is largely ADDITIVE; detection exists" — is FALSE for every case that
motivates versioning:

- **BatchNorm train**: the running-stat update is inside the fused `batch_norm` kernel ->
  TorchLens captures only the INITIAL read node (pre-update value), `num_overwrites=0`.
- **In-place (`mul_`/`add_`/`copy_`)**: captured as a plain compute op with `is_buffer=False`
  and NO buffer linkage; no v2 buffer node is created; `copy_` even drops its source edge.
- **Reassignment mid-forward (`self.b = y+1`)**: invisible — `_tag_untagged_buffers` only
  tags at module-entry, so a buffer assigned then immediately read isn't even a buffer node.
- **Two-loop overwrite does NOT split into two Layers**: buffer `equivalence_class =
  "buffer_<address>"` ignores loop position -> all versions collapse into ONE Layer. So the
  **dual-label is fictional** (op-label-pass == address-version always; the `ModuleCall`
  parallel does not hold), and the worked two-loop table cannot be produced.
- **"node only if read" vs "full history" are incompatible**: a final write that's never read
  produces no node, yet changes `final_value`. Can't have both.

## What IS real and worth shipping (both critics agree)

The high-level factoring is RIGHT, and the CURRENT `Buffer(Op)` is demonstrably bad:
- `Buffer(Op)` mixes graph-position identity with PyTorch-state (address) identity.
- `trace.buffers` ALREADY LIES for overwritten buffers: `num_overwrites` returns 0,
  the accessor is collapsed-per-address so it sees only 1 of N buffer ops.
- Shared buffer aliases aren't discovered (`named_buffers` default dedups) -> `all_addresses`
  would lie.
- The READ-side model is clean (static buffer read N times = one node, N children — confirmed).
- Entity-as-noun + `.buffer` backref is a genuine improvement.

## The scope fork (JMT's call — materially changes what we co-designed)

**Option A — descope to the honest, additive refactor (RECOMMENDED).**
Ship: persistent `Buffer` entity (replace `Buffer(Op)`) + `is_buffer` flag + `.buffer`
backref + read-side model + FIX the lying accessors (`num_overwrites`/`versions` built from
the ACTUAL buffer ops, alias discovery via `named_buffers(remove_duplicate=False)`) +
`Module.buffers`. State PLAINLY that, under current capture, most buffers have ONE version
(overwrites in fused/in-place/reassignment are not yet captured). DROP the dual-label (it
collapses to one label), `value_after`/in-place versioning, the two-loop table.
Cost: days. Honest, shippable, fixes real bugs (the lying accessors), sets up versioning
correctly for later. No fiction.

**Option B — expand to also build WRITE CAPTURE (the proposal's flagged "hard part").**
Add: intercept `nn.Module.__setattr__`/`register_buffer` for reassignment; post-call buffer
snapshot/diff for fused ops (BatchNorm); detect in-place ops mutating a buffer-tagged
receiver and synthesize the write-version + rewire reads; fix `copy_` source edge; give
buffers loop-position context so two loops split. THEN versioning + dual-label + `value_after`
become real. Cost: WEEKS, risky — it's a capture-PIPELINE change, not a data-model change,
and touches the wrapper hot path + loop detection + validation.

**Recommendation: A now, B as a separately-scoped follow-on sprint.** B is a different beast
(capture pipeline) and shouldn't be silently bolted onto a data-model refactor. A delivers a
real, honest improvement immediately and makes B clean to layer on later. Both critics
independently proposed this same split.

## Status

Build HELD pending JMT's scope call (the design changed fundamentally from what was
co-designed; not a silent pick). Spec v1 stands as the "full vision" target; SPEC v2 (Option
A) to be written once scope is confirmed. The write-capture gap is filed as its own todo —
it's a real, important TorchLens limitation regardless of this sprint.

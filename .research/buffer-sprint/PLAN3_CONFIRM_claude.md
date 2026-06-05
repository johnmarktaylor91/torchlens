# PLAN v3 (buffer postprocess-synthesis) — adversarial CONFIRM round (Claude/Anthropic)

Env: torch 2.8.0+cu128, torchlens 2.18.0, py311. All evidence MEASURED via live probes in
`/tmp/bufprobe/`. No tracked source modified. Design review.

Scope of confirm: did v3 close the 11 prior blockers (5 Claude + 6 Codex), and are there NEW
holes? Verdict at end.

---

## BLOCKING-A: The module-EXIT re-scan — v3's sole reassignment-capture mechanism — DOES NOT FIRE for the recurrent repro, and where it does fire it sees only the FINAL value (misses every intermediate version)

**SEVERITY: blocking.** This is the load-bearing replacement for the demolished `__setattr__`
patch (prior Claude B5 / Codex B3). v3 §"Reassignment without a global patch" stakes the entire
reassignment story on "at each module-call exit, compare current registered-buffer object/value
to entry." Two measured failures collapse it.

### A1: the top-level model forward is NOT decorated -> exit re-scan never fires for the brief's exact repro

The brief's named case: `for _ in range(4): self.h = torch.tanh(self.h + x)` with NO submodule.

```
/tmp/bufprobe/p2_exit_scan.py  (spy on _record_module_exit_metadata):
  === exit events for Recurrent (top-level loop) ===
  num exit events: 0
```
```
/tmp/bufprobe/p3_topwrap.py    (spy on _record_module_entry_metadata):
  A: top-level recurrent: entry events = []          <- ZERO
  B: submodule recurrent : entry events = [('inner', 'Inner')]
```
ROOT CAUSE (code, verified): `_prepare_model_once` (`model_prep.py:205-208,237-246`) EXPLICITLY
skips wrapping the root module's forward ("Root module is handled separately by trace"), and the
root's "separate handling" in `capture/trace.py` runs `_record_module_entry/exit_metadata` ONLY
in `capture_mode == "predicate"` (fastlog projection path, trace.py:577-677). Plain exhaustive
`tl.trace` takes the bare `else: outputs = model(...)` branch (trace.py:679) with NO root
entry/exit and NO `_tag_untagged_buffers` call. So for a model whose buffer writes live in its
OWN `forward`, neither `_record_module_exit_metadata` nor any buffer re-scan ever runs. The
recurrent state buffer — the single most important buffer-version case in the whole sprint
(RNN/SSM hidden state) — produces **ZERO** entry/exit events, so v3's exit re-scan (which v3
places "at each module-call exit") has nothing to hook. Current trace (p1): `buffer_layers ==
['buffer_1']` (initial zeros only); the 4 writes flow as ordinary `add`/`tanh` op outputs in a
recurrent group; `validate==True` ONLY because replay rides the op-output chain, not because the
versions were captured. (v3 COULD add a NEW root-exit buffer re-scan in trace.py's plain-capture
branch — but it does not specify one, and even if added, A2 below still defeats it.)

### A2: even WITH a submodule boundary, ONE exit re-scan sees only the FINAL value -> intermediate versions lost

Wrap the same loop in a submodule (case B): exactly ONE entry + ONE exit. The 4 reassignments
all happen BETWEEN that single entry and exit. A re-scan at exit compares entry-`h` vs exit-`h`
and sees ONE delta -> it can attribute at most ONE version, not 4. v3's value-source table claims
reassignment versions are "captured (the producing op's output)" with `num_versions` reflecting
the overwrite count — but the exit re-scan cannot even COUNT 4 overwrites, let alone attribute
each to its `tanh`. Multi-overwrite-per-call (an explicit v3 stress model) is unreachable by a
boundary-granularity scan: the boundary is coarser than the write events.

**Why this is the same hole, relocated, not closed.** Prior review B5/B3 said single-entry
reassignment was a silent capture gap and the only catch was a dangerous global patch. v3 swaps
the global patch for an exit re-scan — but the exit re-scan (a) never fires when the writes are
in the top-level forward, and (b) collapses N intra-call overwrites to 1 boundary diff. The gap
is NOT closed; it is renamed and made invisible for the headline recurrent case.

**Proposed fix.** A boundary scan cannot capture per-write reassignment versions. Either: (i)
decorate the top-level model forward too AND insert a hook at EACH wrapped-op boundary that
re-binds a registered buffer (i.e. detect reassignment when the new object is produced, in the op
wrapper, not at module exit) — but that reintroduces hot-path work the plan tried to avoid; or
(ii) honestly scope reassignment-version capture to "value attributed to the producing op when
that op's output object is the new buffer binding, discovered by a post-pass walk that matches
each registered-buffer object-identity at end-of-pass back to the op that produced it" — and
accept that intra-loop intermediate reassignments that are overwritten are LOST (already in v3's
residual #6, but v3 currently claims the recurrent case is CAPTURED, which is false). The plan
must STOP claiming reassignment versions are default-captured for the top-level / multi-overwrite
case; right now the value-source table row "reassignment -> producing op's output -> yes" is
empirically false for the brief's own repro.

---

## BLOCKING-B: Fused-write version-node synthesis breaks v3's OWN structural invariant — the producer's output is NOT the buffer value, so the synth node either FAILS validation or must self-feed (vacuous, tripwire disarmed)

**SEVERITY: blocking.** v3 §"Build mechanics" + value-source table: fused-final writes get a
synth identity version node with "attribution to the fused op" as parent and value from the
end-of-pass snapshot / `out_versions_by_child`. v3 §"THE replay-safety trap" item (2) proposes a
structural invariant: "each buffer-version node's value equals its producer/snapshot value."
MEASURED: for BatchNorm these two sources DISAGREE, so the invariant is unsatisfiable.

### Evidence (p9): the fused op does not output the buffer value
```
BatchNorm1d.train() trace:
  buffer_1 (running_mean) out_shape=(4,)   <- PRE-write value, read-side parent only
  buffer_2 (running_var)  out_shape=(4,)
  batchnorm_1_1 func=batch_norm parents=[input_1, buffer_1, buffer_2] out_shape=(8,4)  <- ACTIVATION
running_mean changed: True  running_var changed: True   (the writes happened)
```
`batch_norm.out` is the normalized activation `[8,4]`. The new `running_mean`/`running_var`
values `[4]` appear NOWHERE as an op output — they live only in `named_buffers()` post-pass.

### Why the synth node cannot validate honestly (contrast with the real reassignment node)
The reassignment case works precisely because the producer's output IS the buffer value:
```
p12: real promoted reassignment: buffer_1 buffer_source=tanh_1_2; identity(tanh_1_2.out)==buffer_1.out -> validate True
```
For fused, v3 must set the synth node's parent = `batch_norm`. Then identity-replay computes
`identity(batch_norm.out [8,4])` and compares to the synth node's value `running_mean [4]`:
shape+value mismatch -> validation FAILS (correctly, it's a tripwire). To make it pass, v3 must
EITHER (a) feed the synth node its OWN snapshot value as saved_args (self-feeding identity:
`identity(rm_v2)==rm_v2` trivially true) -> the check is **self-referential and vacuous**, so a
WRONG snapshot value would still pass — the exact tripwire-disarming v3 swears off in
"Validation Integrity"; OR (b) add a validation EXEMPTION for fused buffer-version nodes ->
also a tripwire weakening; OR (c) make `batch_norm` itself a multi-output op that emits the new
stats — a real capture change, NOT postprocess synthesis, contradicting v3's "the only hot-path
change is the end-of-pass snapshot."

**This is BLOCKING-1 of the prior reviews (the `_version` BatchNorm false-negative) returning in
a new costume.** v2 couldn't DETECT the BN write; v3 can detect it (snapshot) but cannot REPRESENT
it as a validatable graph node, because the value has no producer edge. The headline case is still
unsolved at the data-model level.

**Proposed fix.** Be explicit that fused-write version nodes are NOT forward-replay-validatable
the way ordinary ops are (their value has no functional producer in the graph). Mark them with a
NARROW, named exemption ("buffer-version node sourced from end-of-pass snapshot / out_versions,
value provably outside forward-replay's contract") AND add a SEPARATE invariant that the synth
node's value byte-equals the snapshot source it was built from (so the snapshot path itself is
checked, even though forward-replay can't be). Do NOT self-feed identity and call it validated;
that is a vacuous green. And state in the glossary that fused buffer versions are snapshot-backed,
not replay-backed.

---

## BLOCKING-C: Fused-multi-buffer ordering + the running_var-depends-on-running_mean question is unaddressed; and num_batches_tracked still has no node

**SEVERITY: blocking (correctness of the headline BN case).** BatchNorm writes THREE buffers per
call (running_mean, running_var, num_batches_tracked). v3 says "synthesize from
out_versions_by_child + snapshot" but:
- MEASURED (p9): only running_mean/running_var ever appear as buffer nodes; num_batches_tracked
  is NEVER a buffer node (it is not passed to the kernel; updated in Python). It IS in
  named_buffers(), so the snapshot picks it up -> an ENTITY with a final_value but ZERO version
  nodes and is_static ambiguous. Prior MINOR-9 flagged this; v3's "num_batches_tracked
  consistency (bookkeeping vs accessor agreement)" targeted fix is listed but the entity/version
  inconsistency (entity says overwritten, versions==[]) is NOT resolved by a bookkeeping tweak —
  it needs an explicit, narrow invariant carve-out which v3 does not specify.
- ORDERING: snapshot gives only the FINAL values of all 3 buffers; there is no per-write ordering
  among running_mean vs running_var within the single fused call. v3's "multi-buffer ordering"
  is named as a concern but the snapshot mechanism provides NO ordering signal — all three get
  one version pinned to the same fused op with identical execution timestamp. If any consumer
  reads running_var between the two stat writes (it cannot in BN, but the data model claims
  generality), the order is unrecoverable. For BN specifically this is benign, but v3 must say so
  rather than imply ordered multi-buffer versions exist.

**Proposed fix.** Scope the fused multi-buffer version model to "one version per buffer per fused
call, unordered within the call, value from snapshot," document it, and add the
num_batches_tracked entity shape as an explicit allowed invariant case (entity present, versions
empty, final_value set, is_static computed from value-equality of initial vs final, NOT from
version count).

---

## MAJOR-D: View/slice/`.data` alias writes to REGISTERED buffers are still unattributed; v3's targeted "shared-alias discovery" fix addresses a DIFFERENT problem (Codex B2 not closed)

**SEVERITY: major.** Prior Codex B2 (alias/view/slice/detach/`.data` writes) was a blocking issue.
v3's only alias-related item is "Shared-alias discovery (`named_buffers(remove_duplicate=False)` /
`_buffers` by storage id)" — that is about TWO names sharing ONE buffer, NOT about a WRITE through
a view/slice/`.data` alias. MEASURED (p16):
```
view.add_   : buffer_layers=['buffer_1'] writer_op_addr=None validate=True  (writer on unaddressed alias)
slice.add_  : buffer_layers=['buffer_1'] writer_op_addr=None validate=True
.data.add_  : buffer_layers=['buffer_1'] writer_op_addr=None validate=True
```
The mutating op (`add_`) operates on an alias with NO `_tl.address`, so v3's "reuse the producing
op" cannot attribute it to the buffer — the producing op isn't recognized as a buffer writer. The
module-EXIT value re-scan CAN notice the value changed (object identity unchanged for in-place
alias writes; value differs), but only IF a submodule boundary exists, and only as ONE coarse
boundary delta — it cannot point at the `add_` op as producer (the buffer object's binding never
changed, so "compare object/value at exit" yields a value-changed-but-same-object signal that the
re-scan must then somehow attribute to an op it has no handle on). `.data` writes don't bump
`_version` AND don't change the object, so a value-only re-scan is the only hope, again
boundary-coarse. v3 does not state these are in or out of scope.

**Proposed fix.** Explicitly classify view/slice/`.data`-alias writes: either (a) propagate buffer
identity through view-like ops at capture (storage-id alias tracking, Codex's proposed
BufferRegistry) so the writer op is recognized, or (b) DOCUMENT them as an unsupported write class
with a test proving the limitation (final_value still correct via snapshot; per-write version NOT
captured). Silent partial coverage is the worst option.

---

## MAJOR-E: Loop detection collapses per-pass version nodes — recurrent buffer versions cannot carry distinct per-iteration values as graph nodes

**SEVERITY: major.** v3 wants a version node per write, chained through the graph. For a recurrent
buffer the writes are inside a loop that loop-detection collapses by `equivalence_class`. MEASURED
(p13, in-place `add_` x4):
```
add_1_1 parents=['buffer_1','input_1','add_1_1'] children=['add_1_1','clone_1_2']  (self-loop, 4 passes add_1_1:1..4)
buffer_1 children=['add_1_1:1']   (read only by pass 1; passes 2-4 ride the recurrent edge)
```
The 4 in-place writes are ONE recurrent op with 4 passes, NOT 4 distinct nodes. If v3 inserts a
synth version node per write, loop detection (groups by `equivalence_class`, p-loop_detection.py
BFS+iso-refine) will either (a) collapse all 4 synth nodes into one recurrent version node ->
per-pass version VALUES are lost at the node level (only the LayerLog pass-values survive, matching
the project's "Layer IS a package of recurrent ops" model), or (b) the inserted identity nodes
perturb isomorphism and BREAK the recurrent grouping that currently works. v3's "RE-MEASURE
two-loop -> lock dual-label" defers this, but the deeper issue — version nodes are
pass-collapsed by design — means `buffer.versions` as a list of DISTINCT-valued nodes is
inconsistent with how recurrence is modeled everywhere else. Prior MAJOR-7 (circular
re-measurement) is acknowledged-but-deferred, not closed; this is its concrete failure mode.

**Proposed fix.** Decide NOW (not "re-measure later") that recurrent buffer versions are
represented as PASSES of a single recurrent version node (LayerLog pass-values carry the
per-iteration values), NOT as N distinct graph nodes. Then `buffer.versions` for a recurrent
buffer returns the pass-values of that node, and `num_versions` = pass count. This is consistent
with the EQUIVALENT-vs-RECURRENT model in project memory. Do not promise N distinct version nodes
in a loop.

---

## MINOR-F: (a) double-representation is benign for compute metrics but inflates graph distance/depth

**SEVERITY: minor.** MEASURED (p15): `compute_ops` correctly excludes `is_buffer` nodes (only the
3 `add` ops; buffers absent), so FLOPs/compute-distance are NOT double-counted — the brief's
double-count worry is empirically unfounded for compute metrics. HOWEVER, inserting an identity
version node between a producer and reader ADDS one hop to `distance_from_input`/depth for every
downstream node (the reader is now 1 further from input). For in-place writes (prior Claude B3:
the writer op ALREADY is the version producer), inserting a redundant identity node is pure
distance inflation with no information gain. v3 §1 claims "value living on both producer and
identity node is benign for replay" — true for replay, but it is NOT free for graph-distance
semantics.

**Proposed fix.** Per prior Claude B3 (which v3 does NOT explicitly adopt): do NOT mint identity
nodes for IN-PLACE writes — the mutating op already IS the version producer; flag it
(`is_buffer_writer`, `buffer_version=N`) instead. Reserve synth identity nodes for reassignment
(genuinely no node today) and fused (snapshot-backed). This removes the distance inflation and the
double-representation for the most common write kind.

---

## CONFIRMED-CLOSED (credit where due — v3 genuinely fixed these)

- **Prior Claude B1 / Codex B1 (`_version` BatchNorm false-negative): CLOSED as a detector.** v3
  abandons `_version` entirely; the end-of-pass snapshot (p9: running_mean/var changes detected via
  named_buffers post-pass) reliably gets the final fused values. (But REPRESENTING them as nodes is
  BLOCKING-B above — detection != representation.)
- **Prior Claude B5 / Codex B9 (global `__setattr__` monkeypatch leakage): CLOSED.** v3 removes the
  global patch; no cross-module leak risk. (But its replacement, the exit re-scan, is BLOCKING-A.)
- **Prior Codex B6 / Claude MAJOR-6 (orphan-prune kills full-history nodes): CLOSED for the entity.**
  MEASURED (p18): an unread final write `self.b.add_(1.0)` is fully orphan-pruned (buffer_layers==[],
  writer op gone) — so v3 is RIGHT to put final-write history in the snapshot-backed ENTITY, not a
  graph node. v3 §"NOT in scope: static-buffer-as-sole-output orphan" and the entity `final_value`
  correctly route around orphan pruning. Snapshot recovers the value even when the writer is pruned.
- **Prior Codex B4 / Claude MAJOR-8 (hot-path second-scan cost): CLOSED.** v3's only capture change
  is one end-of-pass snapshot (off the per-op hot path); no second per-op arg scan. Sound.
- **Prior Claude B2 (save_arg_values off by default): PARTIALLY CLOSED, honestly scoped.** v3
  value-source table explicitly marks the fused-re-read case as "only w/ save_arg_values" and makes
  default capture rely on op-outputs + snapshot. This is the honest scoping prior B2 demanded.
- **Save/load + snapshot (p14): round-trips fine; snapshot timing within tl.trace is correct** (reads
  named_buffers at trace end before any re-run). The `final_value` field is a benign addition.

---

## MAJOR-G: The silent-replay-fallthrough trap is REAL but narrowed by an existing invariant; v3's proposed structural invariant is NECESSARY and must be specified precisely (not "value equals producer/snapshot")

**SEVERITY: major** (it is the brief's headline fire; downgraded from blocking ONLY because an
existing invariant catches the most likely mistake, but the residual silent path is real and v3's
proposed guard is under-specified).

The fallthrough at `core.py:1000-1001` IS real and silent: when neither `target_op_label` nor
`layer_label` is a key in `parent.out_versions_by_child`, replay uses `parent.out` with NO error.
MEASURED in isolation (p7): feeding a parent whose `.out` is WRONG `[99,99,99]` returns `[99,99,99]`
to the reader's replay with no complaint.

WHAT SAVES v3 from the worst case (measured, p19): the topology invariant `out_versions_by_child
keys subset of children` (invariants.py:572-580) FIRES if a reader is relinked off the buffer but
its stale ovc key is left behind:
```
after incomplete relink (mul_1_3 removed from buffer_1.children, ovc key kept):
  invariants: FAIL (GOOD): [graph_topology] out_versions_by_child has keys not in children: {'mul_1_3'}
```
So the "forgot to migrate the key" mistake is NOT silent. GOOD.

THE RESIDUAL SILENT PATH (still real): v3 relinks reader -> synth node, correctly drops the stale
buffer-node key, but does NOT install a correct ovc entry on the synth node (or the synth node's
`.out` is itself wrong). Then replay of the reader silently uses `synth.out`. If `synth.out` is the
CORRECT version value, validation is correct. If `synth.out` is WRONG (mis-sourced snapshot, wrong
fused attribution per BLOCKING-B, off-by-one version pick), the reader recomputes from the wrong
value and:
- if the reader's func genuinely depends on the parent, the recompute differs from the reader's
  saved out -> validation FAILS (caught, good); BUT
- for an IDENTITY synth node validated on its OWN (self-fed) value (BLOCKING-B case (a)), the check
  is vacuous and a wrong value passes silently.

So the trap bites specifically at the intersection with BLOCKING-B: a fused buffer-version node
that self-feeds identity is BOTH unvalidatable-by-replay AND its wrong value silently propagates to
any reader of that version. v3's proposed mitigation (2) "a structural invariant that each
buffer-version node's value equals its producer/snapshot value" is the RIGHT instinct but is
UNDER-SPECIFIED and partly impossible: for fused nodes there IS no producer whose out equals the
value (BLOCKING-B). The invariant must be: "synth buffer-version node value byte-equals the
SPECIFIC source it was constructed from (the snapshot slot / the out_versions_by_child entry),
checked against an INDEPENDENT re-read of that source" — not "equals its producer," which is
unsatisfiable for fused.

**Proposed fix.** (1) Make the parent-value fallthrough at core.py:1000-1001 RAISE (or warn->fail)
when the parent is a buffer-version node and the expected ovc key is absent — convert the silent
fallthrough into a tripwire for buffer-version parents specifically. (2) Specify the structural
invariant as source-equality (snapshot slot / ovc entry re-read), not producer-equality. (3) Never
self-feed identity for fused nodes and call it validated (see BLOCKING-B). (4) Add the exhaustive
per-pattern `validate_forward_pass` tests v3 already lists, but assert they fail when a version
value is deliberately corrupted (a NEGATIVE test that the tripwire bites).

---

## Verdict summary

v3 is a GENUINE improvement over v2 and closes the mechanism-level blockers that demolished v2:
the `_version` detector is gone (snapshot is reliable for detection), the global `__setattr__`
monkeypatch is gone (no cross-module leak), the hot-path second-scan is gone (one end-of-pass
snapshot), and the orphan-prune-vs-full-history tension is correctly resolved by routing final
history to the snapshot-backed ENTITY rather than graph nodes. Credit there is real.

But v3 has NOT closed the two hardest substantive gaps; it relocated them:

1. **Reassignment capture (BLOCKING-A):** the module-EXIT re-scan, v3's sole reassignment
   mechanism, never fires for the brief's exact recurrent repro (top-level forward is undecorated
   in plain capture, MEASURED zero events) and, where a submodule boundary exists, sees only the
   FINAL value (one boundary diff cannot recover N intra-call overwrites). The value-source table
   row "reassignment -> producing op output -> captured: yes" is empirically false for the
   headline case.

2. **Fused-write representation (BLOCKING-B + C):** the snapshot reliably DETECTS the BN write,
   but the value has no producing op whose output equals it (`batch_norm` emits the [8,4]
   activation, not the [4] stat). So a synth version node either fails forward-replay validation
   or must self-feed identity (vacuous green = tripwire disarmed, forbidden by Validation
   Integrity). v3's proposed "value equals its producer" invariant is unsatisfiable for fused.
   num_batches_tracked + multi-buffer ordering remain unspecified.

The silent-replay-fallthrough trap (MAJOR-G) is real but the existing topology invariant catches
the most likely mistake; the residual silent path bites exactly at the BLOCKING-B intersection and
needs the fallthrough converted to a raise for buffer-version parents.

These are design-level, not wording: they require deciding (a) how reassignment versions are
captured without a hot-path detector AND without a boundary scan that misses top-level/intra-loop
writes, and (b) how fused buffer-version nodes are validated honestly. Both are answerable, but
they are NOT answered in v3 as written, and the plan currently CLAIMS they are captured/validated.

VERDICT: NOT BULLETPROOF -- 3 blocking issues (A: module-exit re-scan never fires for top-level/recurrent reassignment and is boundary-coarse for multi-overwrite; B: fused-write synth node has no producer-equal value -> fails replay or self-feeds vacuously, disarming the tripwire; C: fused multi-buffer ordering + num_batches_tracked entity shape unspecified). Plus 4 major (D alias/view/.data writes unattributed; E loop-detection collapses per-pass version nodes; F identity nodes inflate distance for in-place writes that need no new node; G silent fallthrough must be converted to a raise for buffer-version parents).

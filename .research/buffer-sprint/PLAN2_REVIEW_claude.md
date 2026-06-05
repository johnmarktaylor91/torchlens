# Adversarial review — Buffer Phase-2 write-capture PLAN (Claude, Anthropic)

Env: torch 2.8.0+cu128, py311. Repo @ feat/buffer-capture-era. Scratch in /tmp; no tracked source modified.
All evidence below is MEASURED (probe scripts run live), not asserted.

---

## BLOCKING-1: `_version`-diff detector has a SYSTEMATIC false-negative on BatchNorm running stats — the plan's headline case

**SEVERITY: blocking**

The plan's core mechanism (PLAN_PHASE2 §"Core: post-op buffer-write DETECTOR", lines 29-41)
claims autograd `_version` is the "fast signal" that "catches fused BatchNorm running-stat
updates". MEASURED: it does NOT.

Construction + evidence (torch 2.8.0):
```
BatchNorm1d.train(); x=randn(64,4); bn(x)
running_mean: value CHANGED (delta max 0.329) but _version 1 -> 1  (NO bump)
running_var : value CHANGED                      but _version 1 -> 1  (NO bump)
num_batches_tracked: _version 1 -> 2  (bumps, because it's updated via python .add_(1))
Across 3 more calls: rm._version stays 1, rv._version stays 1 forever.
BatchNorm2d: identical (rm value changed, _version 1 -> 1).
F.batch_norm(x, rm, rv, ...): rm value changed, rm._version 0 -> 0.
Storage data_ptr BEFORE==AFTER -> it IS an in-place write into the same storage.
```
Root cause: the native fused kernel (`torch.batch_norm` -> `native_batch_norm`) writes the
running-stat storage directly and does NOT go through the autograd `VariableVersion` bump that
Python-level in-place ops (`mul_`/`add_`/`copy_`/`zero_`/`__setitem__`/`out=`) use. Contrast:
InstanceNorm1d's running-mean update DOES bump `_version` (1->2) — different code path. So the
signal is **kernel-specific and torch-version-specific**: silently correct for some buffers,
silently wrong for the single most common and most important one (BN running stats).

Why this is blocking, not minor:
- BatchNorm is THE motivating example throughout SPEC.md and PLAN_PHASE2 (lines 36, 88, 98,
  SPEC §validation). A detector that misses it misses the point.
- Per the project's LOCKED "validation is a tripwire" principle: a write-detector that produces
  ZERO version nodes for a buffer that WAS written during plain capture is exactly a silent
  capture gap. It will not throw — `num_overwrites` will read 0 for BN running_mean, the version
  chain will be `[v1]` only, and `is_static` will be True for a buffer that is demonstrably
  overwritten. That is the spec's own failure mode (PLAN line 16) re-introduced, now masked by a
  detector that "looks like" it works on the toy in-place tests.

The plan already hedges ("`_version` is an optimization signal, NOT sole source of truth ...
confirm with value when needed", line 38). But it gives NO rule for *when* to confirm with value.
The only correct rule, given BN, is "confirm with value ALWAYS for fused/native ops" — which
means the `_version` fast path buys nothing for the case that matters and you pay the full
value-snapshot+compare on every buffer-touching op regardless. See BLOCKING-2.

Proposed fix: ABANDON `_version` as the primary write signal. It cannot be the gate. Two honest
options: (a) value-snapshot + `tensor_nanequal` compare on buffer-tagged args unconditionally
(accept the cost, see BLOCKING-2 for the real number); OR (b) detect writes structurally per-op
by enumerating the known fused-mutator kernels (`batch_norm`, `instance_norm`, etc.) plus the
in-place op-name family TorchLens already tracks (`_`-suffixed, `__i*`, `copy_`, `zero_`,
`__setitem__`, `out=` kwarg). Option (b) is what TorchLens already half-does via
`out_versions_by_child` (see BLOCKING-4). `_version` should be at most a *negative* fast-skip
(if it bumped, definitely written) — never a *positive* "no bump => no write" conclusion, which
is the false inference the plan currently relies on.

---

## BLOCKING-2: `save_arg_values` is False by DEFAULT — the value-snapshot the detector needs does not exist in plain capture, and the plan's "reuse existing snapshot" assumption is false for the default path

**SEVERITY: blocking**

The plan's only honest write-detection rule (given BLOCKING-1) is value-snapshot + compare. The
plan implies this is cheap because args "are already scanned at wrappers.py:441". MEASURED: the
PRE-op arg snapshot (`arg_copies = safe_copy(arg)`, wrappers.py:452) is GATED behind
`trace.save_arg_values`, which is **False by default**:
```
tl.trace(BatchNorm1d, x).save_arg_values -> False
default in-place-buffer trace: out_versions_by_child is {} for EVERY op; has_out_variations False everywhere
```
So in the default capture path there is NO pre-op snapshot to diff against. To detect writes by
value the detector must take a FRESH `clone()` of every buffer-tagged arg before every op — a NEW
cost that does not exist today. Validation (`validate_forward_pass`) forces `save_arg_values=True`
+ `layers_to_save='all'`, so the snapshot exists THERE — which is exactly why the 4-agent
"replay survives via out_versions_by_child" finding looks clean: it was measured under validation
settings, not default capture. The plan's "Established truth" (PLAN lines 11-15) silently inherits
validation-mode behavior and presents it as the default-capture data model.

Consequence: the write-detector either (a) only works when `save_arg_values=True` (so the buffer
data model is absent in the default `tl.trace`, contradicting the goal "all info about a buffer +
versions real"), or (b) takes fresh clones unconditionally on the hot path (new cost on EVERY
buffer-touching op even when the user didn't ask to save args).

Proposed fix: state explicitly which capture mode the buffer data model is guaranteed under. If
the answer is "only with save_arg_values/validation", say so in the spec and the glossary — do NOT
claim version chains are real in default capture. If default capture must have them, budget the
unconditional clone cost and justify it (see BLOCKING-1 fix option b: structural per-op-name
detection avoids the value clone entirely for the known mutator set).

---

## BLOCKING-3: the in-place version chain ALREADY EXISTS via the mutating op's output edge — minting new identity-passthrough version nodes DOUBLE-MODELS it and risks replay double-handling

**SEVERITY: blocking** (design-correctness; the plan's central new artifact is partly redundant
and the redundancy is where replay breaks)

Construction: `h=ones(4); a = x*h (read v1); h.add_(10) (write); b = x-h (read v2)`. Traced with
`save_arg_values=True`:
```
buffer_1   out=[1,1,1,1]   children=[mul_1_1(read v1), add_1_2(the in-place writer)]
add_1_2 (add_)  out=[11,11,11,11]   children=[sub_1_3(read v2)]   <-- THIS IS v2
sub_1_3 (sub)  reads from add_1_2, NOT from buffer_1
```
The in-place op `add_` ALREADY produces v2 as its `out`, and the downstream read `sub` ALREADY
reads from it. The version chain `v1 (buffer_1) -> writer (add_) -> v2 read` is structurally
present TODAY, with correct values, and replay passes. There is NO separate identity node and none
is needed for the in-place case: the COMPUTE op is the version producer (matches SPEC.md line 126-129's
own description: "the *mutator op* carries the duality on its edges").

So the plan's "mint a buffer write-version node (v(N+1)) ... reuse the Step-6 identity-passthrough
machinery, route subsequent reads through the new version node" (PLAN lines 33-35) would INSERT an
extra identity node between `add_1_2` and `sub_1_3` that isn't there now. Risks:
1. **Replay double-handling.** `out_versions_by_child` (ops.py:1542-1566) records pre-mutation
   parent values keyed by CHILD label. Re-routing `sub`'s parent from `add_1_2` to a new
   `buffer:v2` identity node changes the child/parent label topology that `_validate_parents`
   (core.py:987-1001) keys on. The existing `out_versions_by_child[target_op_label]` /
   `[layer_label]` lookups silently fall through to `parent_layer.out` (core.py:1000-1001) when the
   key is absent — so a mis-keyed re-route produces the WRONG parent value with NO error. That is a
   silent replay corruption, the exact tripwire-defeating failure mode the project forbids.
2. **Perturbation.** `_perturb_layer_outs` (core.py:1006-1008) perturbs a parent and asserts the
   output changes. An identity passthrough inserted in the path is a trivially-invertible node; if
   the perturbation is applied at the identity node vs the real writer, sensitivity bookkeeping
   shifts. Must be re-proven per stress model, not assumed.

The reassignment case (BLOCKING-5) is the ONLY case that genuinely lacks a node today and needs new
machinery. The plan conflates "in-place needs a new node" (false — writer op is the node) with
"reassignment needs a new node" (true). Treating them uniformly with one identity-minting path is
the source of the double-handling risk.

Proposed fix: do NOT mint identity nodes for in-place writes — flag the existing mutating op as the
version producer (`is_in_place`, `buffer_version=N+1` on its out edge), exactly as SPEC.md:126-129
already describes. Reserve new-node creation strictly for reassignment (new object, no writer op in
the graph). Re-key `out_versions_by_child` only if a node is genuinely inserted, and add a replay
invariant that ERRORS (not silently falls through to `parent.out`) when a buffer-version parent
lookup misses its expected key.

---

## BLOCKING-4: `out_versions_by_child` is the existing mutation model — the plan must not run a parallel write-model that can disagree with it

**SEVERITY: blocking**

The plan lists "Interaction with out_versions_by_child (don't double-handle mutation)" as a mere
risk bullet (PLAN line 91). It is the central architectural collision, not a footnote. TorchLens
ALREADY has a complete, validated in-place-mutation model:
- ops.py:1555-1566: for each parent, if the parent's value at THIS child's call time differs from
  its saved `out`, record the pre-mutation value in `parent["out_versions_by_child"][child_label]`
  and set `has_out_variations`.
- core.py:987-1001: replay swaps in that per-child pre-mutation value.

This is a per-CHILD, value-diff model of "what version did this reader see". The plan's write-detector
is a per-WRITER, version-node model. Two independent models of the same physical event (a buffer
mutation) will drift: e.g. the value-diff model fires when `tensor_nanequal` is False; the
version-node model fires when `_version` bumps (BLOCKING-1: these DISAGREE for BatchNorm — value
differs, version doesn't). A reader could be assigned version N+1 by one model and see the v_N value
via the other. There is no single source of truth.

Proposed fix: pick ONE model. The existing `out_versions_by_child` value-diff is already the de
facto write detector and is validation-proven. Build the `Buffer` version chain AS A VIEW over
it (writer = the op whose `has_out_variations` covers this buffer parent; version = count of
distinct values seen), rather than introducing a second, `_version`-based detector that can
contradict it. This also dissolves BLOCKING-1 (no reliance on `_version`) and BLOCKING-2 (reuses
the snapshot that validation already takes) — but ONLY works when `save_arg_values=True`, which must
then be stated as the data-model's precondition.

---

## BLOCKING-5: reassignment within a SINGLE module entry is invisible today AND `_version` is meaningless for it; the `__setattr__` hook is the only catch but it is a global, leaky, 101-line-semantics monkeypatch

**SEVERITY: blocking** (both a correctness gap and a high-risk mechanism)

Two measured facts:
1. Reassignment makes a NEW object (`self.h = self.h*0.9 + x`): id changes, new tensor has
   `_version 0` and NO `_tl` meta. `_version`-diff is structurally inapplicable (plan concedes
   this). MEASURED: id before != id after; new object _version 0.
2. The existing reassignment-capture path (`_tag_untagged_buffers` ->
   `promote_label_to_buffer_source_and_clear_label`, model_prep.py:724-744) only runs at module
   RE-ENTRY. For a reassignment inside a single forward with no re-entry, it NEVER runs:
   ```
   ReassignOnce.forward: self.h = x.mean(0)*2; return x + self.h
   trace -> NO buffer node for h at all; the mul op feeds output directly.
   num_overwrites would be 0; is_static would be True for a buffer that WAS overwritten.
   ```
   This is a real, currently-silent capture gap — and it's the one the `__setattr__` hook targets.

The `__setattr__` hook is therefore load-bearing, but the mechanism is dangerous:
- **Global blast radius.** `nn.Module.__setattr__` is a class method; monkeypatching it patches
  EVERY module in the process. MEASURED: with a patched `__setattr__` and an "active capture" flag
  on, an UNRELATED `nn.BatchNorm1d().running_mean = ...` (never traced) fires the hook. There is no
  cheap way inside `__setattr__` to know whether `self` belongs to the traced model
  (`_state._active_trace` gates the window, but not the identity of `self`). Cross-trace / cross-model
  leakage is the default unless every fired event checks model membership.
- **101 lines of semantics to preserve.** `nn.Module.__setattr__` has intricate Parameter / Module /
  Buffer / plain-attr branches (MEASURED: 101 lines; buffer branch calls `register_buffer`, handles
  `torch.nn.Buffer`, persistent flags, signature introspection). The hook must call through to all of
  it exactly, under exceptions, for ALL attribute sets — including the overwhelmingly common
  non-buffer attr set (MEASURED ~2.5us/set baseline, on the path for EVERY `self.x = ...` in EVERY
  module's forward and __init__ during the capture window).
- **Re-entrancy / exception safety.** If the hook itself touches torch (to snapshot/tag), and that
  re-enters wrapped funcs or `__setattr__`, infinite recursion or partial state. A user forward that
  reassigns a buffer then raises must leave global `__setattr__` restored — a try/finally around the
  whole capture, not per-op.

Proposed fix: do NOT monkeypatch `nn.Module.__setattr__` globally. Two safer routes: (a) keep the
existing `_tag_untagged_buffers` mechanism but ALSO run a buffer re-scan at module EXIT (not just
entry) — catches single-entry reassignment without any `__setattr__` patch, reusing a path that is
already exception-scoped to the wrapped module forward. (b) If `__setattr__` interception is truly
required, gate every fired event on `self in <weakset of prepared modules for the active trace>`
(membership check, ~ns) and wrap install/uninstall in the existing capture try/finally; never leave
it patched outside `active_logging`. Either way: a stress test where a SECOND, untraced module
mutates a buffer during the capture window must show ZERO effect on the trace.

---

## MAJOR-6: "full history / mint on write even if unread" (re-decision #1) collides with orphan pruning — the spec-v1 "node only if read" rule is correct for a structural reason the plan dismisses

**SEVERITY: major**

PLAN re-decision #1 (lines 68-71) proposes minting a node ON WRITE so "an unread final write CAN
be a node -> full history is now achievable", overturning SPEC's "node only if read" (SPEC line 38).
MEASURED structural obstacle: postprocess Step 3 `_remove_orphan_nodes` (graph_traversal.py:223-262)
floods bidirectionally from inputs AND outputs and DELETES any node reachable from neither. An
unread final-write version node has no read child and does not feed an output -> it is unreachable
-> it is pruned. So "full history" nodes would be silently removed unless the plan ALSO carves
buffer-version nodes out of orphan pruning — which then weakens the orphan-prune invariant (a
de-facto tripwire) and risks keeping genuinely-dead nodes. The spec-v1 "node only if read" rule
exists precisely because the graph is a graph of data that FLOWS; an unread write is a sink with no
consumer. The plan's "re-decide now that writes are captured" framing treats this as a free choice;
it is constrained by orphan pruning and should not be reopened without addressing that.

Proposed fix: keep "node only if read" for the GRAPH; record unread final writes (if wanted) in the
PERSISTENT `Buffer` ENTITY's history (`final_value`, `num_overwrites`) which is not subject to graph
pruning — entity history and graph nodes are different surfaces. This gives "full history" in the
entity without fighting orphan pruning. Do NOT mint graph nodes for unread writes.

---

## MAJOR-7: the dual-label two-loop re-measurement (re-decision #2) is circular — the version nodes it must be measured on do not exist yet, and reassignment-in-loop produces ZERO of them today

**SEVERITY: major**

PLAN re-decision #2 (lines 72-75) says "RE-MEASURE whether the two-loop case splits into Layers ...
on the NEW reality, not the old." MEASURED current reality for the SPEC's own worked example
(buffer overwritten per pass in two loops):
```
TwoLoops.forward: loop A: self.s = self.s + xs[t]; loop B: self.s = self.s * xs[t]
trace -> exactly ONE buffer node (buffer_1, the initial value); the 5 subsequent writes are
ordinary add_/mul_ op outputs. ZERO version nodes. SPEC's dual-label table assumes SIX.
```
The re-measurement cannot be performed until P2a actually mints version nodes, so the data-model
decision (op-label-per-pass vs address-global-version, two-Layer split) is being made on a
hypothetical. This is the same trap that produced the spec-v1 problems: deciding label semantics
against an assumed capture shape rather than a measured one. Recurrent reassignment ALSO produces no
version nodes today (MEASURED: recurrent reassign -> 1 buffer node, writes flow as op outputs); only
in-place recurrence even has a writer op to attach a version to.

Proposed fix: SEQUENCE the work. Land P2a (real write/version nodes, in-place + reassignment) and
its stress traces FIRST; then MEASURE the actual two-loop node shape; only then decide the label
scheme. Do not lock the dual-label semantics in the plan ahead of the capture that determines them.
Mark re-decisions #1/#2 as "decide post-P2a from measured traces", not "decided now".

---

## MAJOR-8: hot-path cost — the post-op write-check SCAN runs on every arg of every op; "bounded to buffer-touching ops" is false for the scan

**SEVERITY: major**

MEASURED on resnet18 (default `tl.trace`): tracing already costs 5.1x a bare forward (297ms ->
1523ms). Buffer-touching ops: 20. The value-snapshot+compare added cost for those 20 ops is small
(~0.09% of forward) — so the per-WRITE cost is not the problem. The problem is the per-OP SCAN that
the plan needs to DECIDE which ops touched a buffer:
- `get_buffer_address(untagged_tensor)` MEASURED at ~182 ns/arg; it must run on every arg of every
  op to find buffer-tagged args. The PRE-op registration scan (wrappers.py:438-443) already pays
  this once; a POST-op write-check is a SECOND full scan of args -> roughly doubles the per-arg
  buffer-scan cost on the hot path, paid even by the ~80% of ops that touch no buffer.
- Worst ratio is many-op/few-buffer models (transformers: thousands of matmul/softmax ops, a
  handful of buffers) where nearly all the post-op scan work finds nothing.

The plan's claim "Cost: bounded to ops that actually touch a buffer-tagged tensor" (PLAN line 37)
is only true for the WRITE-RECORDING work, not for the DETECTION scan that gates it.

Proposed fix: do not add a second post-op arg scan. Reuse the pre-op scan's result: the hot path
already computes `arg_tensorlike` and checks `get_buffer_address` at 438-443 — capture the set of
buffer-tagged args (and their pre-op `_version`/value) from THAT pass, and check them post-op
without re-scanning. And fold the write-check into the EXISTING `_tag_tensor_and_track_variations`
parent loop (ops.py:1555) that already iterates parents and compares values — that loop is where
mutation is already detected; piggyback, don't add a parallel scan.

---

## MINOR-9: `num_batches_tracked` is an int scalar buffer that never enters the kernel — it will be an entity with no version node, breaking the consistency the plan wants

**SEVERITY: minor** (but a guaranteed invariant headache the plan flags but underestimates)

MEASURED: `num_batches_tracked` is a 0-dim int64 buffer; it is updated in Python (`Module.forward`
`+= 1`, which DOES bump `_version`) but it is NOT passed to the `batch_norm` kernel — so it never
appears as a buffer-tagged ARG to any wrapped op, and TorchLens captures NO buffer node for it
(MEASURED: trace shows `buffer_1=running_mean`, `buffer_2=running_var`, no nbt). Yet
`named_buffers()` lists it. So `trace.buffers` (entities from named_buffers) would include an entity
with zero version nodes and zero reads, while `buffer_layers`/accessible ops would not — the exact
`buffer_layers`/`buffer_num_calls`/`buffers` disagreement the plan lists (PLAN line 60) but treats
as a "decide if scalar buffers are entities" aside. It is forced: nbt is read by NO traced op, so by
"node only if read" it gets no node, but it IS overwritten, so `is_static=False` with `versions=[]`
— an internally inconsistent entity.

Proposed fix: decide explicitly that buffers never read by any traced op (nbt, and any python-only
buffer) are entities with `is_static=False`, `num_overwrites>0`, but `versions=[]` and `reads=[]`,
and make the metadata invariant TOLERATE this specific shape (python-mutated-but-graph-invisible)
WITHOUT a blanket exemption — narrow to "buffer present in named_buffers, mutated via _version bump,
never an arg to a wrapped op". Document it in the entity API as a known shape.

---

## MINOR-10: fallback honesty — "ship read-side + document limitation" is clean ONLY if the entity is built over the EXISTING out_versions_by_child model, not the new detector

**SEVERITY: minor**

The fallback (PLAN lines 102-107) is "ship P1 + truthful-accessor/entity read-side, defer
write-version capture." This is clean IF the read-side entity is built over data that exists without
the write-detector. But several entity fields are write-dependent: `num_overwrites`, `versions`,
`is_static`/`is_overwritten`, `final_value`, `value_after`, `layers`. If P2a is dropped, these
fields either lie (e.g. `is_static=True` for a BatchNorm running_mean that was overwritten — see
BLOCKING-1) or must be omitted. The fallback "leaves the entity half-built" unless it explicitly
ships a REDUCED entity (identity/ownership/shape/dtype/initial_value/current_value + reads, and a
DOCUMENTED `versions` that reflects only what `out_versions_by_child` already sees under
`save_arg_values`). As written, the fallback's entity surface is the FULL SPEC surface, which is not
honestly deliverable without writes.

Proposed fix: define the fallback entity's EXACT field subset now (which fields are guaranteed
without P2a), and make the deferred-write limitation a typed/raising behavior (e.g. `num_overwrites`
raises `NotImplemented`/returns None with a documented reason) rather than a silently-wrong value.

---

## Confirmation that the recommended single-source fix works (measured)

```
F.batch_norm(x, rm, rv, ...) with rm=zeros:
  tensor_nanequal(snapshot, rm)  -> False  => value-diff CATCHES the write   (correct)
  rm._version unchanged (0)      -> _version MISSES the write                (the trap)
```
The value-diff model TorchLens already uses (`out_versions_by_child` via `tensor_nanequal`) is the
correct, single source of truth. `_version` must be demoted to (at most) a negative fast-skip.

## Summary of blocking issues (root causes, deduplicated)

1. **`_version` is unreliable as the write signal** (B1): native fused kernels (BatchNorm) write
   buffers without bumping `_version`; the plan's headline case is a silent false-negative.
2. **The needed value snapshot is off by default** (B2): `save_arg_values=False` default; the
   "replay survives via out_versions_by_child" truth was measured under validation settings, not
   default capture — the buffer data model is not real in default `tl.trace` as claimed.
3. **In-place version chain already exists via the writer op's out edge** (B3): minting identity
   nodes double-models it and risks silent replay corruption (the missing-key fall-through to
   `parent.out` in core.py:1000-1001 never errors).
4. **Two competing mutation models** (B4): the new `_version` write-detector and the existing
   `out_versions_by_child` value-diff disagree (B1) — no single source of truth.
5. **Global `nn.Module.__setattr__` monkeypatch** (B5): leaks across untraced modules (measured),
   wraps 101 lines of torch semantics, hot for every non-buffer attr set; and single-entry
   reassignment is a real currently-silent capture gap that this risky mechanism is the only catch
   for.

The plan's GOAL (first-class Buffer entity + version chain) is sound and largely achievable by
building the version chain as a VIEW over the existing, validation-proven `out_versions_by_child`
value-diff model under `save_arg_values=True`, plus a module-EXIT buffer re-scan for reassignment —
NOT by a `_version`-diff detector and NOT by a global `__setattr__` patch. As written, the plan's
core mechanism (B1) is empirically broken on its headline case, runs a parallel model that
contradicts the existing one (B3/B4), depends on a default-off snapshot it presents as always-on
(B2), and reaches for a process-global monkeypatch (B5). These must be resolved before build.

VERDICT: NOT BULLETPROOF -- 5 blocking issues (B1 _version false-negative on BatchNorm; B2 save_arg_values off by default; B3 in-place identity-node double-model + silent replay fall-through; B4 dual mutation models disagree; B5 global __setattr__ monkeypatch leakage)

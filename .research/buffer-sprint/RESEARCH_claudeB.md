# Buffer WRITE/UPDATE capture — root-cause + robust design (Claude research agent B)

**TL;DR (headline finding, contradicts the brief's framing):** The "recurrent stateful
buffer" `validate_forward_pass -> False` is **NOT a capture or replay bug**. Capture is
correct; the full BFS forward-replay + perturbation + metadata invariants all PASS once one
line is fixed. The failure is a **validation-harness aliasing bug**: the ground-truth output
tensor is a *live reference to the buffer*, and `model.load_state_dict(...)` clobbers that
tensor **in-place** before the Phase-0 comparison reads it. Fix = clone GT outputs before the
state restore (`torchlens/user_funcs.py:2680-2696`). Verified: recurrent case -> True, no
regressions.

The *data-model* gap the round-1 review found (overwrites not captured as version-nodes) is
**real and separate** from replay — but it is a DATA-MODEL/UX gap, not a correctness/promise
hole. Replay is already universal.

---

## 1. Root cause (measured, file:line)

### 1.1 The failure is in Phase 0 (ground-truth check), before any replay

`Trace.validate_forward_pass` runs three phases (`validation/core.py:1-15`,
`validate_saved_outs` at `core.py:159`):
- **Phase 0** (`core.py:191-204`): logged output must match a fresh forward pass
  (`_ground_truth_output_matches_saved`, rtol 1e-6/atol 1e-8 — `core.py:71-72`).
- **Phase 1** BFS forward replay (`core.py:206-226`).
- **Phase 2** perturbation + metadata invariants (`core.py:238-244`).

The recurrent error string `"The 0th output layer, output_1, does not match the ground truth
output tensor."` is emitted at **`core.py:202`** — i.e. **Phase 0 fails; replay never runs.**

### 1.2 The captured graph is CORRECT

Inspecting the trace of `for _ in range(4): self.h = tanh(lin(x) + self.h)` (`/tmp/inspect_recurrent.py`):

```
buffer_1     buf=True  addr=h  src=None  out0=[0.0,0.0,0.0]            <- initial version only
add_1_2 p1   par=[linear_1_1:1, buffer_1]   out0=[0.752,...]          <- reads initial h
tanh_1_3 p1  out0=[0.637,...]                                          <- new h (iter 1)
add_1_2 p2   par=[linear_1_1:2, tanh_1_3:1] out0=[1.389,...]          <- reads PREV tanh (chained!)
...
tanh_1_3 p4  out0=[0.933,-0.913,0.756]
output_1     par=[tanh_1_3:4] out0=[0.933,-0.913,0.756]
ground truth (reset buf):  [0.933,-0.913,0.756]   <- MATCHES capture exactly
```

The read->write chain across iterations is threaded correctly through the ordinary `add`/`tanh`
ops. The saved output equals a correct fresh forward pass. **Capture is right.**

### 1.3 The actual bug: GT output aliases a buffer clobbered by the restore

The harness (`user_funcs.py:validate_forward_pass`, line 2613):
- `2676`: `state_dict = _clone_state_dict_with_metadata(model)`
- `2680-2686`: **GT run first** — `ground_truth_output_tensors` is built from
  `model(*input_args_copy)`. For this model `forward` does `return self.h`, so the GT tensor
  **IS the registered buffer object** (no clone taken).
- `2696`: `model.load_state_dict(state_dict)` — `load_state_dict` does `buf.copy_(saved)`
  **in-place**, writing zeros into the exact tensor the GT list still references.
- `2701`: capture run (correct, saves a proper copy).
- `2715`: `trace.validate_forward_pass([... clobbered GT ...])` -> GT now reads `[0,0,0]`.

Direct proof (`/tmp/test_alias.py`):
```
out is m.h: True
GT out before restore: [0.9328, -0.913, 0.7558]
GT out AFTER restore : [0.0, 0.0, 0.0]   <-- clobbered in-place by load_state_dict
```
Spy on the real harness comparison (`/tmp/test_real_harness.py`):
```
[spy] saved: [0.9328,-0.913,0.7558]  gt: [0.0,0.0,0.0]  match: False
```
**GT is wrong, capture is right.** The check is *working as designed* (it's a tripwire); the
input it was handed was corrupted by the harness's own restore step.

### 1.4 Why the other three cases pass (precise aliasing condition)

Failure requires: **model output is a `return self.<buf>` whose buffer was REASSIGNED to a new
object** (`self.h = ...`). Then the GT-list reference is to a tensor that the slot restore
zeroes. Cases that pass:
- BatchNorm / in-place `mul_` / reassignment-loop **return a non-buffer tensor** (`x + self.b`,
  `y`) -> GT doesn't alias a buffer -> survives the restore.
- **In-place return** (`self.h.copy_(...); return self.h`) passes too (`/tmp/test_precise.py`):
  the buffer object is stable; Phase-0 comparison runs *before* the final restore (`:2719`),
  and at that point GT-aliased slot still holds the (deterministic) capture-final value ==
  saved. So in-place-return matches by aliasing-to-same-final-state; reassignment-return does
  not (its GT object got zeroed at `:2696`).

This is why ONLY "recurrent reassignment that returns the buffer" fails. A one-character change
to that model (`return self.h + 0.0`) makes the **current** harness pass
(`/tmp/landscape.py`, row "recurrent but returns h+0 -> True").

---

## 2. Landscape table (measured: `/tmp/landscape.py`, `/tmp/test_precise.py`)

`HARNESS(now)` = current `validate_forward_pass`. `FIXED` = same but GT cloned before restore.
"data-model versions" = does `buffer.num_overwrites`/version-nodes expose the writes
(`/tmp/datamodel_check*.py`).

| Pattern | capture graph correct | HARNESS(now) | FIXED | data-model versions |
|---|---|---|---|---|
| BatchNorm train, fused `batch_norm` (returns non-buffer) | yes | True | True | no (init node only) |
| in-place `mul_`, returns `x+b` | yes | True | True | no |
| reassignment `self.b=y+1` loop, returns `y` | yes | True | True | no (0 buffer nodes) |
| **recurrent read-modify-write, `return self.h` (reassign)** | **yes** | **False** | **True** | no |
| recurrent, `return self.h + 0.0` (non-aliased out) | yes | True | True | no |
| recurrent in-place `self.h.copy_()`, `return self.h` | yes | True | True | no |
| in-place `mul_`/`add_`, `return self.b` (aliased) | yes | True | True | no |
| multi-overwrite per forward, returns copy | yes | True | True | no |
| integer buffer `+=` (num_batches_tracked-like) | yes | True | True | no |
| submodule re-entered per loop iter (re-tag at entry) | yes | True | True | partial: N `buffer_1` nodes, but `num_overwrites=0`, `src=None` |

**Key reads:**
1. `capture graph correct` is **yes for every row** — forward replay is already universal.
2. The single `False` flips to `True` under the GT-clone fix; nothing else changes.
3. `data-model versions` is **NO for in-forward mutation** in every row. Overwrites inside a
   single forward (loop reassignment, in-place `mul_`) are captured as ordinary ops, NOT as
   buffer version-nodes. Buffer version-nodes are only minted on **module re-entry**
   (`_tag_untagged_buffers` runs per submodule forward, `model_prep.py:724,855`), and even
   then `buffer_source`/`num_overwrites` are not populated. This is the round-1 review's "write
   capture gap" — it is a **data-model** gap, orthogonal to replay.

---

## 3. THE replay fix (highest-confidence, ~1 line)

In `torchlens/user_funcs.py:validate_forward_pass`, clone GT outputs **before** the
state restore so the in-place `load_state_dict` cannot clobber them. After building
`ground_truth_output_tensors` (`user_funcs.py:2690-2695`) and BEFORE `model.load_state_dict`
(`:2696`):

```python
from .utils.tensor_utils import safe_copy   # already imported in module
ground_truth_output_tensors = [
    safe_copy(g, detach_tensor=True) for g in ground_truth_output_tensors
]
model.load_state_dict(state_dict)
```

(Equivalently `g.detach().clone()`.) GT tensors are used ONLY for the Phase-0 `torch.allclose`
comparison (`core.py:200`), so detaching/cloning is safe and has no other effect.

**Verified** (`/tmp/test_real_fix.py`, `/tmp/test_full_replay.py`):
- recurrent-return-buffer: FULL pipeline (Phase0 + BFS replay + perturbation + invariants) -> **True**
- BatchNorm / in-place / reassignment / all extras -> **True** (unchanged)
- `tests/test_validation.py` + `tests/test_buffer_visibility.py` smoke -> 8 passed.

This is a validation-harness correctness fix and does NOT touch the capture hot path, loop
detection, or any invariant (the tripwire stays armed — we're feeding it a correct GT instead
of a corrupted one, never weakening a check). It belongs in the same family as the existing
`input_args_copy` deep-copy (`:2663-2666`), which already guards inputs against in-place
mutation; outputs simply weren't given the same guard.

---

## 4. The DATA-MODEL question (the brief's real depth) — robust write-capture design

Replay is solved by §3. The remaining (separate) goal: make the persistent `Buffer` entity +
version chain fall out cleanly, so in-forward overwrites become first-class version-nodes
(`buffer.versions[N]`, `num_overwrites`, `value_at/after`), per `.research/buffer-sprint/SPEC.md`.
Today (measured) **only the initial version is a node**; in-forward writes are invisible to the
entity model. Below: 4 candidate mechanisms, ranked.

### Background — current write-detection (what exists)
- A buffer becomes a source node only when read while untagged (`wrappers.py:438-443`,
  `model_prep.py:816-820`): `get_buffer_address(t) is not None and get_tensor_label(t) is None`.
- Re-tagging happens at **module entry only** (`_tag_untagged_buffers`, `model_prep.py:724`,
  called from `_record_module_entry_metadata` `:855`). So a buffer reassigned mid-`forward`
  is not re-detected until the NEXT module call. Inside one forward's loop, no new version-node.
- `buffer_source` is populated by `promote_label_to_buffer_source_and_clear_label`
  (`_tl.py:237-248`) — moves a prior intermediate label into `_tl.buffer_source`. Only fires at
  re-entry, and only if the new buffer value already carried a label.
- Step 6 `_fix_buffer_layers` (`control_flow.py:635-702`) wires `buffer_source` as parent, sets
  `func=identity`, dedups by `modules+buffer_source+address` (`:672`), assigns `buffer_pass`.
  The one-node-per-version passthrough machinery EXISTS; it's just never fed multiple versions
  for in-forward mutation.

### Approach A — intercept `nn.Module.__setattr__` / `register_buffer` during tracing
Wrap setattr so that `self.h = <tensor>` on a buffer-addressed slot emits a version event:
tag the new tensor with the address + a monotonically increasing version, and record a
read-vN -> write-v(N+1) link using the OLD slot tensor's label as `buffer_source`.
- **Fixes recurrent (reassign) data model:** YES — every reassignment is observed at the
  source, giving a clean version per write with correct parent (the producing op's tensor).
- **Replay:** unaffected/already-correct; the new version-nodes are `func=identity`
  passthroughs in the main graph (Step-6 machinery already handles them).
- **Hot-path cost:** `__setattr__` is per-attribute-write, NOT per-op — fires only on
  `self.x = ...`, which is rare vs the torch-func hot path. Gate on
  `_state._logging_enabled and isinstance(value, Tensor) and name in self._buffers`. Negligible
  for normal models; bounded by reassignment count.
- **In-place gap:** does NOT catch `mul_`/`copy_`/fused `batch_norm` (no setattr). Needs B as a
  complement -> two mechanisms.
- **Risk:** monkeypatching `nn.Module.__setattr__` globally is invasive; must be installed
  only during active capture and removed after (mirrors `active_logging`), and must not perturb
  user models that override `__setattr__`. Medium-high blast radius.
- **Loop detection:** version-nodes group by `equivalence_class=buffer_<addr>`
  (`sources.py:258`) exactly like the initial node -> recurrence -> buffer-Layer. Acyclic per
  version time-order. Clean.

### Approach B — post-op buffer snapshot/diff for ops that mutate buffer-tagged args
After each wrapped op, if any *buffer-tagged* tensor arg was written (in-place or fused),
detect the mutation and mint a write version. Detection options: (b1) compare a cheap
fingerprint (version counter `t._version`, or storage data_ptr + a hash) before/after; (b2) use
`torch.Tensor._version` (autograd version counter) — bumps on in-place ops, free and exact.
- **Fixes:** in-place `mul_`/`add_`/`copy_` AND fused `batch_norm` running-stat writes (the
  cases A misses). Combined with the existing read-detection, gives read-vN/write-v(N+1).
- **Replay:** already-correct; version-nodes are passthroughs.
- **Hot-path cost:** runs on EVERY op (the torch-func hot path). Mitigations: only inspect args
  that are buffer-tagged (`get_buffer_address(t) is not None`) — already computed at
  `wrappers.py:441`; reuse that scan. Read `t._version` (a C-level int, ~free). So cost is one
  int read per buffer-tagged arg, only when a model HAS buffers in the op args. Low for
  non-buffer-heavy models, modest for BN-heavy nets (one int per running stat per BN call).
- **Risk:** `_version` is a private autograd field but stable across torch 2.x; aliasing/views
  can share a version counter (false positives on views of buffers) — must key on the buffer
  *storage*/address, not arbitrary views. Reassignment (A's case) does NOT bump `_version` (new
  object) -> B alone misses pure reassignment; needs A as complement.
- **Loop detection / invariants:** same clean grouping as A.

### Approach C — write journal keyed by tensor identity (id/data_ptr)
Maintain, during capture, a dict `{address: [version events]}` and a reverse map
`{id(tensor) or data_ptr: (address, version)}`. On any op output or setattr, if the tensor's
identity maps to a buffer address (or replaces a buffer slot), append a version event. At
postprocess, materialize version-nodes + entity from the journal.
- **Fixes:** all patterns (it's a superset bookkeeping layer over A+B's signals).
- **Replay:** unaffected.
- **Hot-path cost:** dict lookups per op output + per setattr. `id()`/`data_ptr()` keys are
  fragile: `id()` is reused after GC; `data_ptr()` is reused after free/realloc; views share
  data_ptr. Must reconcile against the existing `_tl` tensor-meta (which already carries
  address/label). Higher constant cost than B's `_version` int read, and identity-key hazards.
- **Risk:** highest — identity reuse + view aliasing demand careful invalidation; this is
  effectively re-implementing the tensor-meta tagging that `_tl.py` already provides, so it
  partially duplicates existing infrastructure. Medium-high.
- **Loop detection / invariants:** journal must emit nodes in execution order with stable
  equivalence classes; otherwise risks the spurious-cycle hazard SPEC §"Loop detection" flags.

### Approach D — explicit read-vN -> write-v(N+1) version chaining feeding BOTH replay and model
Make the version chain the single source of truth: every buffer read records "I read version
N", every write records "I produced version N+1", and these events both (i) form the
parent/child edges the replay already consumes and (ii) populate `Buffer.versions`. This is the
SPEC's intent stated as a mechanism rather than a detection trick — it is **the consumer** of
whatever detector (A and/or B) provides the raw read/write signals.
- **Fixes:** all patterns, IF fed by A (reassign) + B (in-place/fused). Cleanest data model:
  `versions`, `num_overwrites`, `value_at/after`, `reads`, `writes`, dual-labels all fall out
  of one ordered event log per address.
- **Replay:** already-correct; D doesn't change replay, it makes the entity model *consistent
  with* the graph the replay already walks.
- **Hot-path cost:** the cost is in the DETECTORS (A/B), not in D itself (D is postprocess
  materialization, Step 6-adjacent). So D's marginal hot-path cost ~0.
- **Risk:** low *as a design*, but D is not self-sufficient — it presupposes A and/or B for
  detection. Its risk is inherited from whichever detector(s) feed it.
- **Loop detection / invariants:** version-nodes already group correctly (measured: the initial
  node groups into the recurrent Layer); extending to N versions per address keeps the same
  `equivalence_class` keying -> recurrence -> buffer-Layer, acyclic by time-order. The SPEC's
  spurious-cycle risk must be regression-tested but the existing single-version path shows the
  grouping is sound.

### Ranking (for the DATA-MODEL goal; replay is already fixed by §3)

1. **B + D (post-op `_version`-diff detector feeding an explicit version chain)** — RECOMMENDED
   for the data model. B catches in-place + fused (the hard, common cases: BatchNorm, `mul_`)
   at near-zero hot-path cost by reading the autograd version counter on already-scanned
   buffer-tagged args (`wrappers.py:441`). D turns those + existing read events into a clean
   ordered version log that materializes the SPEC entity. Reassignment (`self.h=...`) is then
   the *only* residual; add a narrow A-style `__setattr__` hook (B's blind spot) gated to active
   capture, OR accept that reassignment versions surface via the existing re-entry path and the
   produced-op edge. Lowest cost-to-coverage ratio; reuses Step-6 passthrough machinery.
2. **A (setattr intercept) + D** — clean for reassignment, but misses in-place/fused and needs B
   anyway; the global `__setattr__` patch is more invasive than B's local `_version` read.
3. **A + B + D (full)** — total coverage but two detectors + a chain; more surface area, more
   regression risk; justified only if BOTH reassignment AND in-place must be first-class
   version-nodes universally.
4. **C (identity journal)** — most general on paper, worst risk/cost: id/data_ptr fragility,
   view aliasing, duplicates existing `_tl` tagging. Avoid unless A/B prove insufficient.

---

## 5. Reconciling SPEC.md with reality

- SPEC §"What's already built" claims Step 6 "connects each buffer to its writer via
  `buffer_source` ... creates a node only when read ... dedups". **Verified true for the
  detection that fires**, but it FIRES ONLY at module re-entry. In-forward mutation
  (loop reassignment, in-place `mul_`) mints **no** version-node and leaves
  `num_overwrites=0`/`buffer_source=None` (measured, §2). The SPEC's "one node per version" is
  **aspirational** for in-forward writes; the sprint must ADD a write-DETECTOR (Approach B/A),
  not just an entity wrapper. This is the round-1 "write capture gap" — confirmed real.
- SPEC §"In-place capture" risk #1 ("mul_/add_ producing read-vN/write-v(N+1) edges cleanly"):
  currently **not produced at all** for a buffer mutated purely in-place within one forward
  (no re-entry). Approach B (autograd `_version` diff) is the concrete mechanism that closes it.
- SPEC dual-label / `versions[N]` / `value_at`/`value_after`: all depend on the version log
  (Approach D). Sound to build once a detector feeds it.
- **Crucial correction to the brief's premise:** "recurrent read-modify-write replays False" is
  a HARNESS bug, not a capture gap. Do NOT design replay around it. The data-model sprint and
  the §3 replay fix are independent; ship §3 immediately regardless of the sprint.

---

## RANKED APPROACHES + RECOMMENDATION

**The recurrent-replay bug:**
1. **(RECOMMENDED, ship now) Clone GT output tensors before `load_state_dict`** in
   `user_funcs.py:validate_forward_pass` (~line 2695, before `:2696`). ~1 line, verified to fix
   the recurrent case and the whole `validate_forward_pass(model,x)` family with zero
   regressions and no weakening of any check (it CORRECTS the GT fed to the tripwire). Mirrors
   the existing input deep-copy at `:2663-2666`.
2. (defensive complement, optional) Also `safe_copy` outputs inside the capture/GT extraction so
   any future in-place-restore path is immune — lower priority; #1 is sufficient.

**The data-model write-capture (separate sprint, the SPEC's depth):**
1. **(RECOMMENDED) Approach B + D**: post-op autograd-`_version`-diff detector on
   already-scanned buffer-tagged args (`wrappers.py:441`) feeding an explicit
   read-vN -> write-v(N+1) version chain (Step-6-adjacent). Near-zero hot-path cost, catches the
   hard common cases (BatchNorm fused, in-place `mul_`), reuses existing passthrough machinery,
   clean loop grouping. Add a narrow `__setattr__` hook (Approach A) ONLY for pure-reassignment
   versions, gated to active capture.
2. A + D (reassignment-only, misses in-place).
3. A + B + D (full coverage, more surface area).
4. C identity journal (avoid — fragile).

**Single highest-confidence action:** clone the ground-truth output tensors before the
state-dict restore. It robustly fixes the recurrent-replay failure for ALL mutation patterns
that return a buffer, is independently verifiable, and leaves capture/loop-detection/invariants
untouched.

# Buffer PLAN v4 (capture-at-the-moment) — adversarial review (Claude/Anthropic)

Env: torch 2.8.0+cu128, torchlens 2.18.0, py311. All evidence MEASURED via live probes in
`/tmp/bufv4/`. No tracked source modified. Design review.

## TL;DR
v4's two core mechanisms are MORE sound than v3's, and the headline-killer crux (fused
replay-validation, prior BLOCKING-B) is **genuinely closed and non-vacuous** — empirically
verified the replay engine re-runs `batch_norm`, mutates the seeded clone to the correct post
value, and corrupting the captured value FAILS the check (p8, p20). BLOCKING-A (top-level
recurrent reassignment) is also closed IF the hook is a `__setattr__`-on-the-class/subclass,
not "on the instance." But v4's PROSE is wrong about the mechanism ("interceptor on the
prepared module INSTANCES" does not fire — dunders are type-looked-up), and several locked
decisions are internally contradicted by measured behavior. 3 blocking, 4 major.

---

## BLOCKING-1: "scoped `__setattr__` interceptor ON THE INSTANCE" is not buildable as written; a per-instance `__setattr__` attribute NEVER fires. The hook MUST patch the type (subclass / `__class__`-swap), which v4 neither specifies nor accounts for.

**SEVERITY: blocking** (the entire reassignment mechanism, and thus the BLOCKING-A fix, rests
on a Python misconception in the plan).

v4 §1: "Install a `__setattr__` interceptor on the PREPARED module INSTANCES." Python dunders
are looked up on the TYPE, never the instance. Setting `m.__setattr__ = fn` as an instance
attribute is dead code: `self.h = ...` ignores it.

MEASURED (p1_setattr.py):
```
A) per-instance __setattr__ attr fired: 0   (expect 0 -- dunder is type-looked-up)
B) __class__-swap subclass __setattr__ fired for self.h=...: 4   (expect 4)
```
The ONLY ways to intercept `self.h = t` per-instance are (i) swap `m.__class__` to a
synthesized subclass overriding `__setattr__` (works, per-instance, no sibling leak — p4
(2)=0), or (ii) globally patch the class `__setattr__` (the leak v4 explicitly rejected as
"prior B5"). v4 picks neither; it describes the non-working instance-attribute reading.

**This is closeable** — `__class__`-swap is the right mechanism and I verified it is
per-instance, exception-safe, and nestable (p4): exception path restores the class (finally),
no leak to sibling instances of the same class, nested submodule installs both fire. But v4
must be rewritten to specify `__class__`-swap (or document a global-patch-with-membership-gate
and re-defend against the B5 leak), AND handle the edge cases below, none of which v4 mentions:

- **`type(self).__name__` changes to `Scoped_<Cls>` during the forward** (p5). Any user
  `forward` that branches on `type(self).__name__` or `self.__class__ is SomeClass` behaves
  differently under capture — a violation of TorchLens's "run a NORMAL forward pass" contract.
  (`isinstance` survives via subclass; the name/identity does not.)
- **A model with its OWN `__setattr__`** (common in HF / custom modules) gets it SHADOWED by
  the synthesized subclass (p4 (4)): our `Sub.__setattr__` runs instead of the user's, and if
  we call `__mro__[1].__setattr__` we SKIP the user's override entirely — behavior change
  mid-forward. The subclass must delegate to the *actual* original `__setattr__` of the real
  class, not blindly to `nn.Module.__setattr__`.

**Fix.** Specify `__class__`-swap explicitly; build `Sub.__setattr__` to call the ORIGINAL
class's `__setattr__` (preserving user overrides); document the `type(self).__name__` change as
a known capture-time side effect (or use a metaclass trick that keeps `__name__`); uninstall in
`finally`. Verify on a model with a custom `__setattr__`.

---

## BLOCKING-2: The two hooks DOUBLE-COUNT every reassignment, directly corrupting the LOCKED `num_overwrites = write-event count`; and the locked "no dedup by value" forbids the obvious fix.

**SEVERITY: blocking** (breaks a locked decision + the headline Buffer entity field).

v4 §1 says it must cover BOTH `self.h = t` (via the `__setattr__` hook) AND
`self._buffers[name] = t` (via "a buffer-dict proxy"). But `nn.Module.__setattr__`, when
reassigning a registered buffer, INTERNALLY calls `self.register_buffer(name, value)` which
does `self._buffers[name] = value` (verified in torch source, p3). So with BOTH hooks
installed, a single `self.h = t` fires the `__setattr__` hook AND the dict-proxy.

MEASURED (p19_doublecount_final.py):
```
user reassignments: 4
setattr-hook events: 4
dict-proxy events  : 4
TOTAL write events captured: 8   -> num_overwrites would be 8, NOT 4 (2x inflation)
```
v4 LOCKED: "Event-history... `num_overwrites` = write-event count. Dedup NOT by value." The
double-fire makes `num_overwrites` exactly 2x the truth for every idiomatic reassignment, and
the locked "no dedup by value" rule blocks the simplest dedup. A provenance-based dedup
(ignore the dict write when it originates from inside `__setattr__`) is possible but unspecified
and fragile (must distinguish user `self._buffers[name]=t` from `__setattr__`'s internal one —
e.g. a re-entrancy guard flag set across the `__setattr__` body).

**Fix.** Pick ONE hook as authoritative for reassignment. Recommended: keep ONLY the
`__setattr__` hook for the `self.h=` form, and detect direct `self._buffers[name]=t` ONLY when
it does NOT pass through `__setattr__` — implement via a re-entrancy guard (set a thread-local
"in __setattr__" flag around the original call; the dict-proxy records only when the flag is
clear). Add a stress test asserting `num_overwrites==4` for the 4-reassignment loop.

---

## BLOCKING-3: Fused replay-validation is REAL and non-vacuous (BLOCKING-B genuinely closed) — BUT it requires `save_arg_values=True`, which is NOT the default, so fused buffer-version nodes are UNVALIDATABLE in plain `tl.trace`; and for MULTI-PASS fused buffers only pass-1 seeds correctly.

**SEVERITY: blocking** (the mechanism works, but its applicability is narrower than v4 claims;
left unspecified this re-introduces a silent gap).

### 3a — the crux WORKS (credit, prior BLOCKING-B closed)
MEASURED end-to-end against the real engine (p8, p20):
```
replay seeds running_mean from buffer_1 (pre): [0,0,0,0]
after core._execute_func_with_restored_state(batch_norm): seeded clone mutated to
   [0.0194, -0.0342, 0.0237, -0.0063]
live m.running_mean (true post):  [0.0194, -0.0342, 0.0237, -0.0063]   -> MATCH
break test (corrupt captured value +99): check FAILS  -> NON-VACUOUS
```
The replay engine ALREADY mutates the seeded running_mean clone to the correct post value when
it re-runs `batch_norm` (training=True seeds from the PRE parents, which torchlens correctly
cloned pre-call: p9 `saved_arg == graph-pre? True`). So validating the captured fused-version
value against `args[3]` AFTER `_execute_func_with_restored_state` is a REAL, non-self-feeding
check. This is the central v4 win and it holds. It is NOT an exemption and NOT a self-feed.

### 3b — but it needs `save_arg_values=True`
MEASURED (p7): default `tl.trace` saves NO args for `batch_norm` (`saved_args==[]`). The fused
op's running_mean/var args only exist when `save_arg_values=True` (p7b). `validate_forward_pass`
forces that internally, so VALIDATION works — but a plain `tl.trace(model, x)` capture has no
way to reproduce/validate the fused version value (the value lives only in the end-of-pass
snapshot, no replay path). v4 must state: fused buffer-version nodes are **snapshot-backed in
plain capture, replay-validated only under `save_arg_values`/`validate_forward_pass`**. Silent
on this = the prior B2 "off by default" gap, reopened.

### 3c — multi-pass fused buffers: only pass-1 seeds correctly
MEASURED (p10/p11): recurrent `for _ in range(3): x = self.bn(x)` produces 3 `batch_norm`
passes that read DIFFERENT running_mean values (pass2 reads pass1's write):
```
pass 1 running_mean arg = [0,0,0,0]
pass 2 running_mean arg = [0.0194,-0.0342, 0.0237,-0.0063]   (= pass1's write)
pass 3 running_mean arg = [0.0174,-0.0308, 0.0214,-0.0056]
```
But the GRAPH has ONE buffer node (`buffer_1`, the initial value); every pass lists `buffer_1`
as parent. The intermediate running-stat versions (after pass1, after pass2) are NOT represented
as buffer-version nodes — only initial + final-snapshot. v4's "re-run batch_norm, read live
state" seeds from the buffer node (initial), which reproduces ONLY pass-1's write. Pass-2/3
fused versions have no correct seed in the graph and cannot be replay-validated this way.

**Fix.** (1) Document fused versions as snapshot-backed in plain capture, replay-validated only
with saved args. (2) For multi-pass fused buffers, EITHER capture the per-pass pre-value as the
seed (the `saved_args[3]` is already per-pass — use it as the version node's seed, not the
single buffer node) OR scope the per-pass fused version model to "one fused version per buffer
per call, seeded from that call's saved running-stat arg," and require `save_arg_values` for the
recurrent-fused case. State it; do not imply N validatable distinct fused versions in plain
capture.

---

## MAJOR-4: `num_batches_tracked` produces an entity with `final_value` set but `versions==[]` — the exact inconsistency v4 names, with the carve-out left UNSPECIFIED; and v4's premise "it IS mutated by a real tensor op, so capturable as a normal buffer" is empirically FALSE for `nn.BatchNorm`.

**SEVERITY: major.** MEASURED (p16/p17): BN's `self.num_batches_tracked.add_(1)` bumps
`_version` and IS a real in-place op — but its `add_` op does NOT appear in the trace at all
(`add-family ops traced: []`), because its only "reader" is
`exponential_average_factor = 1.0/float(self.num_batches_tracked)`, where `float()` severs the
tensor graph → the `add_` is orphan-pruned. So:
- v4's claim "`num_batches_tracked` ... IS mutated by a real tensor op, so it's capturable as a
  normal buffer" is false: the op is pruned, no version node survives.
- The end-of-pass snapshot still records `final_value=tensor(1)`. Result: `Buffer` entity with
  `final_value` set, `num_overwrites`/`versions` empty — the precise shape v4 says to avoid
  ("no entity with final_value but empty versions"), but v4 gives NO carve-out rule.

**Fix.** Add an explicit, narrow invariant case: an entity MAY have `final_value != initial_value`
with `versions==[]` when the only writer was orphan-pruned (Python-scalar-severed reader);
`is_static` must be computed from value-equality of initial vs final, NOT from version count.
Document `num_batches_tracked` as the canonical instance.

---

## MAJOR-5: Loop detection collapses per-write version nodes for recurrent reassignment — `buffer.versions` cannot be a list of N distinct-valued nodes; v4 DEFERS this ("re-measure once version nodes exist") instead of deciding it.

**SEVERITY: major** (prior MAJOR-E, not closed). MEASURED (p18): the top-level recurrent
reassignment ALREADY loop-groups — `tanh`/`add` are ONE node with 4 passes; `buffer_1` (initial
h) is read only by pass-1. If v4 mints an identity version node per write, the 4 nodes are
per-pass isomorphic → loop detection collapses them into ONE recurrent version node (per-pass
values survive only as LayerLog pass-values), OR the inserted identities perturb isomorphism and
break the recurrent grouping that currently validates True. Either way, `buffer.versions` as "N
distinct nodes with distinct values" is inconsistent with the EQUIVALENT-vs-RECURRENT model the
whole codebase uses. v4 §"Decisions" defers this ("Dual-label / two-loop: re-measure once
version nodes exist, then lock") — but the collapse is structural and decidable NOW.

**Fix.** Decide now: recurrent buffer versions are PASSES of one recurrent version node;
`buffer.versions` returns that node's pass-values; `num_versions` = pass count. Consistent with
"a Layer IS a package of recurrent ops." Do not promise N distinct graph nodes in a loop.

---

## MAJOR-6: Storage→address index keyed on `data_ptr` MISATTRIBUTES after storage reuse; and `.data`/view writes by arbitrary ops bypass BOTH the `_version` fast-path AND the fused-mutator value-check.

**SEVERITY: major** (prior Codex B2 / Claude MAJOR-D, partially addressed but with new holes).

MEASURED (p15): `data_ptr` is REUSED after a tensor's storage is freed
(`data_ptr reused after free: True`). A storage→address index keyed on `data_ptr` (v4 §2 "Alias
coverage: resolve buffer identity by STORAGE") can attribute a write to a FREED-then-reallocated
buffer's stale address — a correctness hazard, not just a miss. The index must be invalidated
when a buffer tensor is freed/reallocated (weakref-keyed, or re-validate the tensor identity at
write time), which v4 does not specify.

MEASURED (p15): `.data.add_` does NOT bump `_version`. v4's two detectors are: (a) `_version`
fast-path, (b) value-snapshot for a KNOWN-FUSED-MUTATOR list. A `.data`-alias write by an
ARBITRARY op (`tmp = self.buf.data; tmp.add_(...)` or library code) is neither on the fused list
nor `_version`-bumping → slips BOTH detectors silently. (View/slice in-place DO bump `_version`
on the base — p15 path differs — so those are catchable, but `.data` specifically is not.)
Current torchlens does chain `view`→`add_` from the buffer node for the simple view case (p14),
but does NOT relink the post-write reader to a new version and validate==True only because the
mutated tensor flows live, not via the graph edge.

**Fix.** (1) Weakref-key (or tensor-identity-revalidate) the storage→address index; never trust
a bare `data_ptr`. (2) Classify `.data`-alias writes explicitly: either add `.data`/`detach()`
mutating ops to the value-check set, or DOCUMENT `.data`-write per-version capture as
unsupported with a test proving the limitation (final_value still correct via snapshot). Silent
partial coverage is the worst outcome.

---

## MAJOR-7: Convert silent replay fallthrough to a RAISE — correct instinct, must be specified as SOURCE-equality not PRODUCER-equality, and the raise must not fire for the legitimate fused case.

**SEVERITY: major** (prior MAJOR-G, carried forward; v4 adopts the raise — good — but
under-specifies). `core.py:1000-1001` `else: parent_values = parent.out` is a real silent
fallthrough. v4 §"Replay validation" rightly converts it to a RAISE "for buffer-version
parents." Two cautions from the code (validation/core.py:994-1001):
- The raise must trigger only when a buffer-version parent's expected `out_versions_by_child`
  key is ABSENT (a mis-link), NOT for the fused case where the parent IS the pre-buffer node
  and the seeding is correct-by-design. Over-broad raise will false-positive on every valid BN
  trace.
- The structural invariant must be SOURCE-equality (synth node value byte-equals the snapshot
  slot / saved-arg it was built from, re-read independently), not "value equals its producer" —
  for fused there is no producer whose `.out` equals the value (the [4] stat vs the [8,4]
  activation). v4's prose ("value == the producing op's output" for reassignment) is right for
  reassignment but must NOT be applied to fused nodes.

**Fix.** Specify the raise condition precisely (buffer-version parent + missing ovc key only);
specify the invariant as independent source-equality; add a NEGATIVE test that a deliberately
corrupted version value makes `validate_forward_pass` return False (the tripwire bites).

---

## MINOR-8: Event-history counts idiomatic no-op reassignments (`self.h = self.h.detach()`) as writes, inflating `num_overwrites` for RNN/SSM grad-detach loops.

**SEVERITY: minor.** MEASURED (p21): a forward doing `self.h = self.h.detach()` then
`self.h = tanh(...)` records 2 write events. Under v4's locked event-history ("even a no-op
`b.mul_(1)` counts"), this is internally consistent and arguably correct, but `detach()`
reassignment is extremely common in recurrent state handling and will roughly double
`num_overwrites` there. Defensible; worth an explicit note in the glossary so users aren't
surprised.

---

## CONFIRMED-CLOSED (credit)
- **Prior BLOCKING-B (fused node not validatable):** CLOSED at the validation level. The replay
  engine re-runs `batch_norm` and mutates the seeded clone to the correct post value; comparing
  the captured value to it is real and non-vacuous (p8 corruption fails, p20 live-mutation
  confirmed). This is v4's genuine breakthrough over v3. (Scope caveats in BLOCKING-3.)
- **Prior BLOCKING-A (top-level recurrent reassignment invisible):** CLOSED *mechanism-wise* —
  a class/subclass `__setattr__` hook fires for the top-level `for _ in range(4): self.h=tanh()`
  loop (p1 B = 4 events), which the v3 module-exit scan could not see. (Buildability caveat in
  BLOCKING-1: must be `__class__`-swap, not instance attr.)
- **Hot-path cost (prior MAJOR-8):** STAYS closed. MEASURED (p13b): all 53 resnet50 BN
  snapshots = ~518us/pass vs ~0.79ms for ONE BN kernel. Value-snapshot of `(C,)` stat vectors
  is negligible. (p13's 81.8% was a Python-loop/sync artifact, corrected in p13b.)
- **Global `__setattr__` leak (prior B5):** the `__class__`-swap is per-instance — no sibling
  leak (p4 (2)=0), exception-safe restore (p4 (1)), nestable (p4 (3)). Good, IF BLOCKING-1's
  prose is fixed to actually use it.
- **In-place `add_` (prior MINOR-F):** confirmed the mutating op IS the producer and its output
  IS the new value (p12) — so an identity version node is redundant for in-place writes; flag
  the writer op instead (`is_buffer_writer`/`buffer_version=N`). v4 should adopt this to avoid
  distance inflation, but it's not blocking.

---

## Verdict summary
v4 is a real advance: the fused replay-validation crux (the thing v2/v3 could not do) is
empirically REAL and non-vacuous — measured against the actual replay engine, corrupting the
captured value fails the check. BLOCKING-A's mechanism (catch top-level recurrent reassignment)
also works. Credit is earned and the two prior headline blockers ARE closeable with v4's
approach. But v4 as WRITTEN ships three blocking defects: (1) the reassignment hook is described
as an instance-level `__setattr__` that cannot fire — it must be a `__class__`-swap/subclass,
with user-`__setattr__`/`type().__name__` side effects unhandled; (2) the two hooks DOUBLE-COUNT
every reassignment (4 writes -> 8 events), corrupting the LOCKED `num_overwrites` field with no
permitted dedup; (3) fused replay-validation needs `save_arg_values=True` (absent in plain
`tl.trace`) and seeds correctly only for pass-1 of multi-pass fused buffers — leaving fused
versions unvalidatable in plain capture and multi-pass intermediate versions unseeded, unless
scoped explicitly. Plus 4 major (num_batches_tracked entity inconsistency unspecified;
loop-detection collapse of recurrent version nodes deferred not decided; data_ptr-keyed storage
index misattributes after reuse + `.data` writes bypass both detectors; silent-fallthrough raise
must be source-equality and not fire for valid fused).

These are design-level (decide the hook mechanism + dedup + fused-scope), not wording — but they
are all closeable and v4's core bet is sound.

VERDICT: NOT BULLETPROOF -- 3 blocking issues (1: reassignment hook is an unbuildable instance-level __setattr__; must be __class__-swap, with user-override/type-name side effects unhandled; 2: dual hooks double-count every reassignment, corrupting the locked num_overwrites with no permitted dedup; 3: fused replay-validation requires save_arg_values [absent in plain tl.trace] and seeds only pass-1 of multi-pass fused buffers -> fused versions unvalidatable in plain capture / multi-pass intermediates unseeded unless scoped). Plus 4 major (4 num_batches_tracked entity{final_value, versions==[]} carve-out unspecified + "real op" premise false; 5 loop-detection collapses recurrent version nodes, deferred not decided; 6 data_ptr-keyed storage index misattributes after reuse and .data writes bypass both detectors; 7 silent-fallthrough raise must be source-equality and must not fire for valid fused).

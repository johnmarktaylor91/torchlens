# Buffer WRITE/UPDATE capture + replay — root cause & landscape (Claude A)

Repo: /home/jtaylor/projects/torchlens. Env py311. All results MEASURED.

## TL;DR (read this first)

The "recurrent stateful buffer fails `validate_forward_pass`" symptom is **NOT a
capture bug, NOT a loop-detection bug, NOT a replay-engine bug, and NOT a buffer
data-model bug.** The forward graph for `for _ in range(N): self.h = tanh(lin(x)+self.h)`
is captured **correctly and acyclically**, and the saved output equals ground truth.

The failure is a **ground-truth aliasing bug in the validation HARNESS**
(`torchlens/user_funcs.py::validate_forward_pass`). The model returns a *live buffer
tensor object* (`return self.h`); the harness later calls `model.load_state_dict(state_dict)`
to undo the buffer mutation, which **overwrites that same tensor in place**, corrupting
the already-extracted ground-truth output BEFORE it is compared. Saved (correct) output
!= mutated GT -> `False`.

The discriminator across all tested cases is exactly: **"does the model's output tensor
alias a model buffer?"** — nothing about recurrence.

## Evidence chain (measured)

### 1. Capture is correct
`/tmp/diag1.py`, `/tmp/diag2.py`. Trace of the recurrent StateCell (N=4):
- Final `tanh_1_3` out = `[0.9328, -0.9130, 0.7558]` == ground-truth iter-3 == model output. CORRECT.
- Graph topology is acyclic and threads state through the tanh outputs:
  - `buffer_1` (initial zeros) -> `add_1_2:1`
  - `tanh_1_3:1` -> `add_1_2:2`  (iter 1 reads PREVIOUS tanh, not a buffer node)
  - `tanh_1_3:2` -> `add_1_2:3`, etc.
- Loop detection groups the 4 iterations into recurrent passes (`linear_1_1`, `add_1_2`,
  `tanh_1_3` each appear with passes :1..:4). No spurious cycle.
- Only ONE buffer node exists: `buffer_1` = the initial value. `num_overwrites=0`,
  `is_overwritten=False`, `buffer_source=None`. The reassigned `self.h` versions are NOT
  captured as buffer version-nodes; the recurrent state flows as ordinary data deps
  (tanh:N -> add:N+1). (This matters for the DATA MODEL, see SPEC reconciliation — but it
  does not break replay.)

### 2. The output comparison receives a corrupted ground truth
`/tmp/diag7.py` instrumented `validation/core.py::_ground_truth_output_matches_saved`:
```
[OUTPUT MATCH] saved=[0.9328, -0.9130, 0.7558]  gt=[0.0, 0.0, 0.0] -> False
```
GT is ZEROS — the buffer's *restored* initial value, not the model's actual output.
Failure raised at `validation/core.py:200-202`.

### 3. The aliasing mechanism (the actual bug)
`/tmp/diag8.py`:
```
id(gt) == id(m.h)?                  True      # model returns the live buffer
gt before load_state_dict:          [0.9328, -0.9130, 0.7558]
gt AFTER  load_state_dict:          [0.0, 0.0, 0.0]   # MUTATED IN PLACE
```
`validate_forward_pass` (user_funcs.py) flow:
1. `state_dict = _clone_state_dict_with_metadata(model)`  (h=zeros snapshot)
2. `ground_truth_output_all = get_vars_of_type_from_obj(model(...), ...)`  — stores a
   REFERENCE to the returned tensor (== `model.h`). **No clone/detach.**
3. `model.load_state_dict(state_dict)` — `load_state_dict` copies into buffers IN PLACE,
   zeroing the tensor `ground_truth_output_tensors[0]` still points at.
4. trace runs (correctly) -> saved output `[0.9328,...]`.
5. compare saved vs (now-zeroed) GT -> mismatch -> `False`.

### 4. The fix is confirmed
`/tmp/diag9.py` (manual fixed flow) and `/tmp/diag13.py` (monkeypatched real
`validate_forward_pass`): cloning/detaching the GT output tensors BEFORE
`load_state_dict` makes the recurrent case pass AND regresses nothing:
```
BatchNorm-train            -> True
in-place mul_              -> True
reassignment-loop          -> True
recurrent-state-buffer     -> True   (was False)
nonrecurrent-return-buffer -> True
```

### 5. The discriminator is alias, not recurrence
`/tmp/diag10.py`:
```
case                       out_aliases_buffer   GT_mutated_by_restore
BatchNorm-train            False                False   PASS
in-place mul_              False                False   PASS  (returns x + s.b)
reassignment-loop          False                False   PASS  (returns y, not s.b)
recurrent-state-buffer     True                 True    FAIL  <-- only this
recurrent-return-copy      False                False   (s.h*1.0; PASS, /tmp/diag11.py)
```
`/tmp/diag11.py`: a recurrent buffer that returns `s.h * 1.0` (a copy) PASSES. A
NON-recurrent model that returns the buffer directly hits the same alias path. Proves the
failure is `return <buffer>`, independent of recurrence.

## Exact fix site

`torchlens/user_funcs.py`, `validate_forward_pass`:
- L2680: `ground_truth_output_all = get_vars_of_type_from_obj(model(...), ...)`
- L2691-2695: builds `ground_truth_output_tensors` by appending `entry[0]` **(raw ref, no clone)**
- L2696: `model.load_state_dict(state_dict)` -> mutates buffers IN PLACE -> corrupts any
  GT tensor that aliases a buffer.

FIX (1 line, L2694): `ground_truth_output_tensors.append(entry[0].detach().clone())`
Verified there is currently NO clone/deepcopy of GT outputs anywhere in this function
(grep clean). The inputs ARE deep-copied (`safe_copy_args`); the OUTPUTS are not.

## Landscape table (MEASURED; baseline = current code)

Repro models in /tmp/diag*.py. `random_seed` fixed. "capture" = forward graph correct &
output matches true model output; "replay (current)" = `validate_forward_pass` today;
"replay (after GT-clone fix)" = with the L2694 clone; "data model" = does the Buffer
entity/version chain expose the mutation.

| pattern | example | capture | replay (current) | replay (after fix) | data model exposes versions? |
|---|---|---|---|---|---|
| fused-kernel update | BatchNorm train (`batch_norm`) | OK | True | True | partial: 1 buffer node, no per-step version chain |
| in-place op, NOT returned | `b.mul_(0.9); return x+b` | OK | True | True | no write-version node (mutator carries it implicitly) |
| reassignment in loop, NOT returned | `for: b=y+1; y=b*2; return y` | OK | True | True | only initial buffer node; rewrites flow as data deps |
| recurrent read-modify-write, NOT returned | `for: h=tanh(lin(x)+h); return h*1.0` | OK | True | True | only initial buffer node; state threads via tanh outputs |
| **recurrent read-modify-write, RETURNED** | `for: h=tanh(lin(x)+h); return h` | OK | **FALSE** | **True** | only initial buffer node |
| reassignment, RETURNED | `b=b+1; return b` | OK | **FALSE** | **True** | only initial buffer node |
| in-place mul_, RETURNED | `b.mul_(0.9); return b` | OK | True (lucky*) | True | initial buffer node + `mul_` op |
| static buffer, RETURNED as sole output | `return b` (no mutation) | **BROKEN** | RAISES `MetadataInvariantError: No output layers found` | (still raises) | buffer node dropped as orphan |
| static buffer, read & consumed | `return x + b` | OK | True | True | 1 node, N children |

\* `mul_`-RETURNED passes only because its returned GT tensor is NOT mutated by the restore
(measured GT=0.9, /tmp/diag17.py) — an autograd-view subtlety, not robustness. The reassign-
RETURNED twin FAILS identically to the recurrent case (/tmp/diag16.py), proving the cause is
the alias+restore, not recurrence and not the op kind.

### Two DISTINCT bugs surfaced
1. **(PRIMARY, the brief's target) GT-alias-mutation in the validation harness.** Any model
   that RETURNS a live buffer that the trace's `load_state_dict` then restores -> false
   negative. Fix = clone GT before restore (L2694). HIGH confidence, surgical.
2. **(SECONDARY, separate) static-buffer-as-sole-output is dropped.** `return self.b` with no
   other consumer -> the buffer node is orphan-pruned (Step 3 `_remove_orphan_nodes`) before
   an output node is built over it, so `output_layers == []` and the invariant raises. This is
   a genuine CAPTURE gap (a returned buffer should become an output), independent of #1. Out of
   scope for the recurrent-replay fix but should be tracked.

## SPEC.md / data-model reconciliation

The SPEC's "one node per version" model is ASPIRATIONAL and NOT yet how reassignment/recurrent
buffers are captured. Measured reality (/tmp/diag2.py):
- A reassigned/recurrent buffer produces only ONE buffer node = the INITIAL value
  (`num_overwrites=0`, `is_overwritten=False`, `buffer_source=None`). The rewritten versions
  are NOT captured as `is_buffer` version-nodes; they flow as ordinary tensor data deps.
- `_fix_buffer_layers` (control_flow.py:635) only rewires buffers that were *logged as buffer
  source tensors AND already carry a `buffer_source`* (the in-place/fused write path). It does
  NOT detect "a Python attribute `self.h` was rebound to a fresh tensor" — that rebinding is
  invisible to torch-function wrapping (no torch op fires on `self.h = ...`).
- Hence the round-1 adversarial critique ("overwrites not captured as nodes", "dual-label
  fictional") is CORRECT for the reassignment/recurrent path, and SIMULTANEOUSLY replay is
  fine — because replay never needed the version-nodes; the data dependency was already
  threaded correctly through the compute ops. The two findings are consistent.
- SPEC revision needed: to make the version chain "fall out naturally" for reassignment/
  recurrent buffers, capture must intercept the *write* (rebind), which torch-function
  wrapping cannot see. That requires `nn.Module.__setattr__`/buffer-write interception
  (proposal R3 below) — a capture feature, NOT required for replay correctness.

## Ranked proposals

### P1 (RECOMMENDED — fixes the brief's failure, highest confidence): clone GT outputs before restore
- WHAT: `torchlens/user_funcs.py:2694` -> append `entry[0].detach().clone()`. Optionally also
  snapshot GT before any TorchLens activity for defense-in-depth.
- FIXES recurrent case: yes (measured True, /tmp/diag9.py & /tmp/diag13.py). Also fixes the
  reassign-RETURNED twin. Zero regressions across all 9 patterns.
- COST: negligible (one clone of the output tensors). NOT in the wrapper hot path — pure
  validation harness. RISK: ~none. Interaction w/ loop detection / invariants: none (capture
  untouched). Data model: untouched.
- CONFIDENCE: VERY HIGH. This is the correct, minimal fix and it is the genuine root cause.

### P2 (defense-in-depth, complements P1): make `load_state_dict` restore non-destructive of GT
- Restore via `model.load_state_dict(state_dict, assign=True)` (rebinds params/buffers to new
  tensors instead of `copy_` into existing), OR snapshot+restore around a model COPY for GT.
  Either prevents in-place corruption of any retained references generally. Lower priority than
  P1 (P1 already fully fixes the observed bug); keep as belt-and-suspenders.
- CONFIDENCE: MED-HIGH. `assign=True` semantics need a compat check across torch versions.

### P3 (data-model completeness, NOT needed for replay): intercept buffer WRITES at capture
- To realize SPEC's per-version Buffer entity for reassignment/recurrent buffers, intercept
  `nn.Module.__setattr__` (and `register_buffer`) during the prepared forward to emit a
  write-version event when a registered-buffer attribute is rebound, and snapshot the
  post-op buffer for in-place/fused mutators (diff against pre-op). Feed read-vN -> write-
  v(N+1) into the version chain.
- FIXES recurrent REPLAY? No — replay already passes after P1. This is purely to populate the
  `Buffer` entity + version chain + dual-label (the SPEC's actual deliverable).
- COST: `__setattr__` interception is per-attribute-set during forward (cheap, gated to
  registered buffers); the post-op snapshot/diff is per buffer-tagged-arg op. RISK: medium
  (must not double-count, must interact correctly with loop detection so version-nodes group
  into buffer-Layers without spurious cycles — SPEC risk #2). Bundle with the buffer sprint,
  NOT with the replay fix.
- CONFIDENCE: MED. Correct long-term direction for the data model; orthogonal to replay.

### P4 (separate tracked bug): returned-buffer-as-output capture
- Ensure a model output that IS a (static or mutated) buffer gets an output node built over the
  buffer node before orphan pruning, so `output_layers` is non-empty. Fixes the
  `static-RETURN-buffer` `No output layers found` raise. Independent of P1.
- CONFIDENCE: MED (needs a look at Step 1 `_add_output_layers` vs Step 3 orphan removal).

## ROOT CAUSE (one paragraph)

The recurrent stateful buffer `for _ in range(N): self.h = tanh(lin(x)+self.h); return self.h`
fails `validate_forward_pass` NOT because of any capture, loop-detection, dedup, or replay-
threading defect — the forward graph is captured correctly and acyclically (state threads
through the `tanh` outputs) and the saved output equals the true model output. It fails because
the validation HARNESS in `torchlens/user_funcs.py::validate_forward_pass` extracts the ground-
truth output as a *live reference* to the returned tensor, which aliases the model buffer
`self.h` (`id(gt)==id(model.h)`, measured /tmp/diag8.py); the harness then calls
`model.load_state_dict(state_dict)` (L2696) to undo the buffer mutation, and `load_state_dict`
writes into buffers IN PLACE, zeroing the very tensor the ground-truth list still points at. By
comparison time GT has been corrupted from `[0.9328,...]` to `[0.0,0.0,0.0]` (measured
/tmp/diag7.py) while the (correct) saved output is `[0.9328,...]`, so they mismatch. The
discriminator across every tested pattern is exactly "does the model output alias a buffer
that the restore mutates" — not recurrence: a recurrent buffer returned as `self.h*1.0`
(a copy) PASSES, and a non-recurrent `self.b = self.b+1; return self.b` FAILS identically.

## RECOMMENDED FIX (single, highest confidence)

`torchlens/user_funcs.py:2694` — clone/detach each ground-truth output tensor at extraction,
BEFORE `model.load_state_dict(state_dict)`:
`ground_truth_output_tensors.append(entry[0].detach().clone())`.
Measured to flip the recurrent case (and the reassign-returned twin) from False to True with
zero regressions across all 9 buffer-mutation patterns and the existing buffer test suite.
This is purely a validation-harness correctness fix; capture, loop detection, the data model,
and the wrapper hot path are untouched. The SPEC's per-version Buffer data model (P3) is a
separate, additive capture feature and is NOT required to make replay/validation pass.

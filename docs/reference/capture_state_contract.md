# Capture state contract (entry-surface restoration)

Normative spec for what a `tl.trace` call may leave changed on the user's
model and process, what it restores, and what it discloses. This is the
A06 deliverable of M(oracles) item 8 ("instance-forward cleanup +
entry-surface state restoration"): every cell is implemented and pinned by
tests. The success-path default state relation (FORK-A) is ruled
PURE_OBSERVER (U40.1, 2026-10-11).

## Vocabulary

- **PURE**: after the call returns (or raises), the model's observable state
  -- instance `__dict__` entries, buffers, parameter flags, training modes,
  pickle-ability -- is what it was before the call.
- **FORWARD_EQUIVALENT**: state advances exactly as one direct `model(x)`
  would have advanced it (norm running statistics update, RNG draws consume),
  and nothing else changes.
- **Instrumentation**: TorchLens-owned additions -- per-module instance
  `forward` wrappers (non-root), `._tl` module metadata, `tl_*` attributes,
  session tensor labels.

## The settled cells (implemented; tests named)

| Cell | Contract | Where |
|---|---|---|
| FAILED `tl.trace` (any phase: entry refusal after preparation, forward error, TL-side capture failure, postprocess failure) | PURE of instrumentation: the failure path releases the preparation (same semantics as `tl.release_model`), so the model pickles and the next capture re-prepares from scratch with full module attribution. Under the default the observer bracket below also rolls back the state the partial forward mutated (buffers, RNG, modes, plain attributes), before the release, so the release's normalization of held torch functions stands; with `carry_state=True` it is NOT rolled back. Materialized lazy modules are never rolled back. `exc.partial_log` stays recoverable after the release. | `torchlens/user_funcs.py::_release_preparation_after_failed_capture`; pinned by `tests/test_capopts_truth_failed_capture_release.py`, `tests/test_capture_pure_observer.py` |
| Per-session parameter state | `requires_grad` forced True for graph construction is restored to the user's exact values at session cleanup, every capture, success or failure. | `model_prep._cleanup_model_session` / `_restore_session_param_state` |
| Session tensor labels | Stripped from model/buffer/input tensors at session cleanup. | `model_prep._undecorate_model_tensors` |
| Successful `tl.trace` state (default) | PURE_OBSERVER: buffers (persistent and non-persistent), every module's `training` flag and the global random states (torch, live CUDA devices, Python, NumPy) are snapshotted before the forward and restored after it. A buffer is written back only where its bytes changed; a write that left its version counter unchanged (train-mode BatchNorm running statistics) is undone the same way, so `backward()` through the capture's graph keeps working, and trace payloads that alias a restored buffer keep their forward-time values. Every plain instance attribute is restored too (AGT-2360): a rebound or removed attribute gets its original object back, a tensor or builtin `list`/`dict`/`set` changed in place is restored in place (identity kept), and attributes the forward added are removed. What the forward wrote is recorded on the trace, `trace.modules[address].attribute_writes` (`AttributeWrite` rows, session-time only), so a value the forward stashes on a module is read there. Builtin containers (`list`, `tuple`, `dict`, `set`, `frozenset` and their subclasses) are snapshotted at any size, and a tensor anywhere inside one is cloned whatever its size. Custom objects, including custom objects held inside a container, are tracked by identity: rebinding is undone, a change inside them is neither undone nor recorded. A container nested more than 256 levels deep is tracked by identity below level 256 (the snapshot is recursive, at most two Python frames per level, inside the default 1000-frame recursion limit), and every default capture names such an attribute in its `capture_state_restore_incomplete` warning. The stale-reference rebind journal never reads post-forward module state. An item that cannot be restored is named in one `capture_state_restore_incomplete` warning. | `torchlens/_runnable_state_live.py::capture_observer_bracket`; pinned by `tests/test_capture_pure_observer.py` |
| Train-mode running statistics | Restored on the default path. Under `CaptureOptions(carry_state=True)` (FORWARD_EQUIVALENT) they advance exactly as one direct `model(x)` would, disclosed once per process with warning code `batchnorm_train_stats_mutated`. | `torchlens/user_funcs.py::_warn_once_train_mode_running_stats`; pinned by `tests/test_capopts_truth_batchnorm_warn.py` |
| Successful `tl.trace` instrumentation | Persistent by design until `tl.release_model(model)` (re-capture speed; the release door restores pickling and is the documented workflow). Dissolve-on-teardown is List-B feature work, not part of this contract. | `model_prep.release_model` |
| Observational surfaces (summary, render/read/validate/export) | PURE under either FORK-A branch (3/3 panel agreement). The summary eval/no_grad default with full state restoration is A07's implementation (summary A4); extraction's no_grad/eval wrap with exact restore is A11's. Both consume the same inventory below. | A07 / A11 |

## FORK-A (ruled: PURE_OBSERVER, U40.1)

- **PURE_OBSERVER (the default)**: trace snapshots and restores
  buffers/RNG/modes/plain attributes and records the forward's attribute
  writes on the trace (`Module.attribute_writes`); the explicit opt-in
  `CaptureOptions(carry_state=True)` carries FORWARD_EQUIVALENT for
  train-through-trace users (the same spelling as `trace.run(inputs=...,
  carry_state=True)`): buffers, modes and plain attributes advance as one
  direct `model(x)` would, and the same attribute writes are recorded. Random state never advances under either: every
  capture draws from its own seed and restores the global engines (R57). A
  capture-cache hit runs no forward, so nothing advances under either.

`trace.run()` keeps its own bracket: a default run restores the declared
state (parameters and buffers) it changed, but not the random state or plain
attributes, so repeated training steps through `run()` keep drawing fresh
dropout masks.

Neither branch makes "your model can no longer be saved" an honest outcome
-- the unpicklable-model residue is a bug under both (the FAILED-call release
above fixes the failure half; the success half is fixed by dissolve-on-teardown:
submodule `forward` wrappers are session-scoped, so a successful capture leaves
the model picklable, and `release_model` remains only for torch-function
attributes held across a later wrap-state change).

### Restoration inventory (what a PURE implementation must snapshot)

1. Persistent buffers (norm running stats, counters) -- `state_dict()` clone
   with `_metadata` (the `_clone_model_state_dict` helper already does this).
2. Non-persistent buffers in use (the runnable-save family already
   enumerates them).
3. Training modes, per module (`module.training`).
4. Global + per-device RNG states (captured today under `save_rng`).
5. Plain instance attributes mutated by `forward` (bounded scan; the
   implementation-fingerprint attribute walk is the existing precedent).
6. Parameter `requires_grad` (already restored per session).

Cost model: 1-4 are cheap snapshots (buffers are cloned, parameters are not);
5 is the plain-attribute scan (value snapshots for scalars, tensors of any
size and builtin containers of any size up to 256 levels deep, identity for
everything else); its cost grows with the number of items the attributes hold. The bracket wraps the capture driver in `_trace_torch_model`; under
`carry_state=True` only the attribute snapshot runs (for the record) and
nothing is restored.

## Disclosure rules

- A `carry_state=True` capture that mutates running statistics discloses
  once per process (`batchnorm_train_stats_mutated`); the default
  PURE_OBSERVER capture restores them and never warns.
- A failed capture's warning (`CaptureAttemptFailedWarning`) names exactly
  what was restored (instrumentation, torch environment) and what was not
  (state the partial forward mutated).
- A secondary failure while releasing after a failed capture warns coded
  (`failed_capture_release_incomplete`) and never masks the capture
  exception.

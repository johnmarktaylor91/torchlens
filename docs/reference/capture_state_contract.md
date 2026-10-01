# Capture state contract (entry-surface restoration)

Normative spec for what a `tl.trace` call may leave changed on the user's
model and process, what it restores, and what it discloses. This is the
lane-A06 deliverable of M(oracles) item 8 ("instance-forward cleanup +
entry-surface state restoration"): the settled cells are implemented and
pinned by tests; the one genuinely open cell (the success-path default state
relation, FORK-A) is specified here for both branches so the ruling lands as
a default flip plus tests, not a design.

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
| FAILED `tl.trace` (any phase: entry refusal after preparation, forward error, TL-side capture failure, postprocess failure) | PURE of instrumentation: the failure path releases the preparation (same semantics as `tl.release_model`), so the model pickles and the next capture re-prepares from scratch with full module attribution. State the partial forward already mutated (running stats, RNG) is NOT rolled back; the failure warning discloses this boundary. `exc.partial_log` stays recoverable after the release. | `torchlens/user_funcs.py::_release_preparation_after_failed_capture`; pinned by `tests/test_capopts_truth_failed_capture_release.py` |
| Per-session parameter state | `requires_grad` forced True for graph construction is restored to the user's exact values at session cleanup, every capture, success or failure. | `model_prep._cleanup_model_session` / `_restore_session_param_state` |
| Session tensor labels | Stripped from model/buffer/input tensors at session cleanup. | `model_prep._undecorate_model_tensors` |
| Train-mode running statistics | NOT restored on the default path (the forward really runs; `torch.no_grad` does not stop running-stat updates). Disclosed once per process with warning code `batchnorm_train_stats_mutated`. | `torchlens/user_funcs.py::_warn_once_train_mode_running_stats`; pinned by `tests/test_capopts_truth_batchnorm_warn.py` |
| Successful `tl.trace` instrumentation | Persistent by design until `tl.release_model(model)` (re-capture speed; the release door restores pickling and is the documented workflow). Dissolve-on-teardown is List-B feature work, not part of this contract. | `model_prep.release_model` |
| Observational surfaces (summary, render/read/validate/export) | PURE under either FORK-A branch (3/3 panel agreement). The summary eval/no_grad default with full state restoration is lane A07's implementation (summary A4); extraction's no_grad/eval wrap with exact restore is lane A11's. Both consume the same inventory below. | A07 / A11 lanes |

## FORK-A (open; JMT ruling pending): the success-path default

Both branches are fully designed in the oracles memo (section 11). The
harness, registry, and every other cell above are identical under both;
only the default state relation of a SUCCESSFUL `tl.trace` on a stateful
(train-mode) model differs:

- **PURE_OBSERVER (panel lean, 2-1)**: trace snapshots and restores
  buffers/RNG/modes/attributes by default; an explicit opt-in kwarg carries
  FORWARD_EQUIVALENT for train-through-trace users.
- **FORWARD_EQUIVALENT**: trace is an instrumented execution; state advances
  as one direct `model(x)` would, disclosed in the returned artifact; an
  explicit observational mode is PURE by restoration.

Neither branch makes "your model can no longer be saved" an honest outcome
-- the unpicklable-model residue is a bug under both (the FAILED-call release
above fixes the failure half; the success half is `release_model` today and
List-B dissolve-on-teardown later).

### Restoration inventory (what a PURE implementation must snapshot)

1. Persistent buffers (norm running stats, counters) -- `state_dict()` clone
   with `_metadata` (the `_clone_model_state_dict` helper already does this).
2. Non-persistent buffers in use (the runnable-save family already
   enumerates them).
3. Training modes, per module (`module.training`).
4. Global + per-device RNG states (captured today under `save_rng_states`).
5. Plain instance attributes mutated by `forward` (bounded scan; the
   implementation-fingerprint attribute walk is the existing precedent).
6. Parameter `requires_grad` (already restored per session).

Cost model: 1-4 are cheap snapshots (one `state_dict` clone bounded by model
size); 5 is the only open-ended item and both branches bound it to the
disclosed plain-attribute scan. The snapshot belongs at
`_prepare_model_session`; the restore at `_cleanup_model_session`, keyed on
the ruled default (or the explicit kwarg overriding it).

## Disclosure rules

- A capture that mutates running statistics discloses once per process
  (`batchnorm_train_stats_mutated`). This stays true under
  FORWARD_EQUIVALENT and becomes unreachable-by-default under PURE_OBSERVER
  (the warn site then guards the opt-in path).
- A failed capture's warning (`CaptureAttemptFailedWarning`) names exactly
  what was restored (instrumentation, torch environment) and what was not
  (state the partial forward mutated).
- A secondary failure while releasing after a failed capture warns coded
  (`failed_capture_release_incomplete`) and never masks the capture
  exception.

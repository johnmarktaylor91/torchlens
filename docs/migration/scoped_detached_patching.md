# Detached-reference handling: rescue re-run + mechanical belt

TorchLens installs persistent wrappers around torch callables on the first torch capture. A Python
binding created before that installation—such as `from torch import relu`—can still point to the raw
callable after the torch namespace itself is wrapped.

**The historical sys.modules crawler (and its `patch_policy` rollout) is deleted.** TorchLens no
longer rewrites module attributes broadly, reads module sources, or mutates user model instances to
repair stale bindings. Coverage is now:

1. **Rescue re-run.** A completed capture that carries an escape signal — the tensor-provenance
   warning, an `escape_detector` diagnostic, or an output-attribution failure — is re-run ONCE with
   a `TorchFunctionMode` net armed. The net redirects any stale pre-wrap reference to its exact
   wrapper, so recovered ops are logged with full wrapper fidelity. The primary capture is never
   mode-armed (an armed mode flips torch's fused fast paths, e.g. eval `MultiheadAttention`
   3 ops -> 27 ops), so ordinary captures are byte-identical to earlier releases. The rescue also
   covers holder classes the crawler never reached: closure cells, staticmethods, module-level
   partials, plain object attributes, pre-bound tensor methods, torch-free-source modules, and
   C-held references. The most reliable pattern is still to wrap early: call
   `torchlens.backends.torch.wrappers.wrap_torch()` before creating aliases, closures, partials,
   or object-held torch callables.
   Then those bindings capture the wrappers directly and no rescue is needed.
   The MIRROR direction is a declared residual for BARE references: a plain attribute read taken
   WHILE wrappers are installed (`held = F.relu`) hands the user the wrapper object, and
   `torchlens.backends.torch.wrappers.unwrap_torch()` does not repair user-held wrapper references
   — TorchLens never crawls or mutates user objects during capture. The held reference stays
   callable (it delegates to the original) but is identity-poisoned after unwrap: `held is F.relu`
   is `False` and pickling it (or any object holding it) fails. For references held on a MODEL,
   `tl.release_model(model)` is the shipped repair: it normalizes held torch-function attributes
   (one level of exact builtin containers, namedtuples included, dict keys included) to the
   currently-live values, and registers the model so every later wrap-state flip re-normalizes it
   — released models stay serializable in every epoch. Bare references held outside a model still
   require re-reading the attribute after unwrap, or a fresh process.
2. **Mechanical belt.** A small, per-build DERIVED set of wrapped functions is invisible to every
   `TorchFunctionMode` (zero protocol callbacks, measured at wrap time): on current builds
   `torch.from_numpy`, `torch.from_dlpack`, `torch.frombuffer`, `torch.Tensor.as_subclass`, and `torch.Tensor._make_subclass`
   (the authoritative set is the live derivation in `belt_report().members`, pinned per build in
   `tests/test_mechanical_belt.py`). A stale reference to one
   of these produces no signal a rescue could trigger on, so module-level attribute references to
   them keep targeted patching, with a conditional reversal ledger restored at
   `torchlens.backends.torch.wrappers.unwrap_torch()`.

## Honesty semantics

- A rescued trace is disclosed: `capture_verified=False`,
  `capture_verification_reason == "mode_rescue_rerun"`, and a session-time `trace.rescue_rerun`
  record naming the trigger and the recovered ops. Rescue captures are never byte-attestable (the
  forward ran twice and mode presence can de-fuse fast paths).
- An escape the net cannot recover — a worker-thread stale reference (modes are thread-local; op
  logging is owner-thread-scoped by design) or a stale reference inside a third-party
  `handle_torch_function` composite body (the protocol pops the mode before the body runs) — is
  DISCLOSED, never silent: `capture_verified=False` with reason `"escape_rescue_unrecovered"`
  unless a more specific verdict (dispatch witness, shadow detector, Dynamo boundary) is already
  present, which stays authoritative.
- An armed completeness witness that VERIFIES the capture outranks the heuristic provenance signal
  (for example an `autograd.grad` boundary is a known no-provenance source); no rescue runs.
- Ineligible captures skip only the RE-RUN, never the disclosure. Every channel the re-run would
  invoke a second time refuses (fail closed): streaming saves, `streaming.out_callback` callbacks, disk grad
  storage, `halt=` predicates, `intervene=` predicates, pre-attached `hooks=`, and the
  `activation_transform`, `grad_transform`, and `output_transform` callables. A live escape
  signal on such a capture still settles `capture_verified=False` with reason
  `"escape_rescue_unrecovered"` (`skipped_reason == "rescue_ineligible"`, `forward_runs == 1`)
  plus a `UserWarning` — never clean-capture fields.
- A rescue re-run executes the user's forward a SECOND time. When the primary forward ACTUALLY
  WROTE module buffer state (train-mode BatchNorm running stats and `num_batches_tracked`,
  in-forward buffer counters), the re-run is refused — RNG is restored between runs, module state
  is not — and the escape stands disclosed (`skipped_reason == "buffer_writes_double_forward"`,
  `forward_runs == 1`). The refusal keys on a VALUE-CHANGING write, not on journal presence:
  fused norm mutators are journaled unconditionally, so eval-mode BN/IN/GN captures carry
  `buffer_value_changed == False` records and stay rescuable; an unknown change status refuses
  fail-closed. The refusal covers EVERY trigger, including an output-attribution failure — there
  the re-run is refused, the failure propagates with its `exc.partial_log`, and the skip is
  disclosed on the partial's trace (`rescue_rerun`). On rescued (eval-mode) captures, a custom
  in-forward PYTHON-attribute counter (not a registered buffer) still mutates twice: a declared
  residual of the double forward, visible via `trace.rescue_rerun["forward_runs"] == 2`.
- The capture-attempt-failed advisory ("Partial diagnostics ride the exception") is emitted on a
  dedicated `RuntimeWarning` subclass and DEFERRED while a rescue re-run may still swallow the
  failure: a successful rescue drops it (the rescue is disclosed on the returned trace), and
  every re-raising path flushes it, so it never points at an exception the user does not receive.
- A rescue that would count as recovery must produce a strict SUPERSET of the primary's op
  multiset. Mode presence can de-fuse fused fast paths; any op LOSS keeps the mode-free primary
  authoritative, with both deltas disclosed (`recovered_ops` / `lost_ops`).

## Removed surface

- `tl.wrap_torch(patch_policy=..., patch_modules=...)` — removed (formerly deprecated no-ops).
- `patch_detached_references(...)` / `clear_patch_detached_references_cache()` — removed
  (formerly no-op shims for the deleted crawler; `PatchReport` removed with them).
- Trace fields `detached_patch_policy` / `detached_patch_epoch` — removed.
- The `"scoped_dispatch_witness_not_enabled"` verification reason — removed with the policy
  machinery.

## Shadow detector semantics

`escape_detector="shadow"` observes raw callable execution using exact object/code identity. It
reports a `TorchLensCaptureGapWarning` with the callable, registered export sites, source callsite,
storage hint, short stack, and remediation. Reports also appear in `trace.escape_diagnostics`, and
a diagnostic now also TRIGGERS the rescue re-run (the report from the primary run is preserved in
`trace.rescue_rerun["primary_escape_diagnostics"]`).

Every wrapper-to-original edge uses a one-shot immediate-caller token. Tokens are identity-compatible
with transient Tensor-bound builtins by requiring a Tensor receiver and an exact method name from the
wrapped-method inventory. They never exempt the dynamic duration of a wrapper. Thus a raw descriptor
called by a user callback inside a composite wrapper remains reportable. TorchLens-internal
exceptions, if ever required, must match an exact parent/child/callsite row; the audited table has a
hard budget of 16 entries. Its current rows are exhausted by the two branches of the eight
`torch._jit_internal.boolean_dispatch` pooling functions; each row is restricted to the exact
parent wrapper, child callable, `_jit_internal.py` callsite, and `fn` caller.

The one-shot token also cannot represent a Python inventory original that recursively calls itself
through a pre-wrap reference. For example, a synthetic registered original shaped like
`def original(x): return original(x - 1) if x else x` consumes its wrapper token on the first call,
then the recursive call can self-convict. This class is documented rather than exempted because its
callsite is arbitrary original/user code; a general recursion exemption would become an ancestry
blanket and hide real escaped calls.

Shadow is default-off. When it is enabled, the resulting trace remains unverified even with no
reports (`"shadow_diagnostic_mode"`); pair it with `completeness_witness=True` for a positive
verdict.

## Machine-readable qualification

Live torch traces expose these diagnostic fields:

| Field | Meaning |
| --- | --- |
| `escape_detector_mode` | `"off"` or diagnostic `"shadow"` |
| `escape_diagnostics` | Structured raw-call reports accumulated across forward passes |
| `rescue_rerun` | Session-time rescue disclosure (trigger, recovery, primary diagnostics) |
| `capture_verified`, `capture_verification_reason` | Completeness status |
| `capture_owner_thread_id`, `capture_owner_thread_qualified` | Supported proof-domain owner |
| `capture_thread_count_start`, `capture_thread_count_end` | Cheap Python thread-count tripwire samples |
| `capture_thread_activity_detected` | Whether the count changed across a guarded forward |
| `capture_guard_passes` | Per-active-logging pass index, mode, and owner thread |
| `escape_detector_event_count`, `escape_detector_callback_ns` | Detector event and callback-cost counters |
| `escape_detector_backward_coverage` | `"not_armed"` for deferred backward in this rollout |

Public `tl.record(...)` uses the same guarded forward hot path and mirrors these fields onto the
returned `Recording`.

The verification VERDICT now survives a `.tlspec` round-trip in the negative direction only: a
capture the producer refused to bless loads with `capture_verified=False` and its string
`capture_verification_reason` intact, so an escape-disclosed artifact is never
byte-indistinguishable from a clean one. A positive claim never persists — a loaded trace reports
`None`/unknown rather than a falsely preserved (or forged) `True`, and load degrades any
non-`False` persisted value the same way. `rescue_rerun` (and the heavyweight
`escape_diagnostics`) remain live session-time diagnostic state and still do not survive the
round-trip.

## Honest boundaries

| Channel | Now |
| --- | --- |
| Closure, dict/list, unrelated class/default, ordinary Python partial | Rescued on signal; shadow reports when Python exposes the call |
| Saved Tensor method descriptor or Tensor-bound builtin | Rescued on signal; shadow reports via descriptor compatibility |
| Protocol-invisible constructors (`from_numpy`, `from_dlpack`, `frombuffer`, `as_subclass`, `_make_subclass`) | Mechanical belt (module-attr patching; membership derived per build) |
| C `functools.partial` around a C builtin | Known profile blind spot; shadow mode stays machine-readably unverified |
| De-moded `handle_torch_function` composite interiors | Beyond any mode; disclosed `escape_rescue_unrecovered` |
| `DataLoader(num_workers=0)` callback executed inside forward | Owner-thread domain; shadow reports visible escapes |
| Worker process preprocessing before model invocation | Outside the armed model-forward domain |
| Model tensor work delegated to another thread/process | Unsupported; owner-thread qualification applies; TENSOR-crossing escapes are disclosed. A PRE-EXISTING thread running a stale op whose result crosses back only as a python scalar (a float, never a tensor) is a declared SILENT residual — no mode, belt, or tripwire observes it |
| OWNER-thread stale op whose result crosses to host as a scalar only (`stale_norm(x).item() > t`: no intermediate tensor op consumes it) | Declared SILENT residual on default captures. The scalar-protocol read of the untracked intermediate emits no record, and unlabeled receivers cannot be flagged without false-positives on parameter/attribute scalar reads. The armed shadow detector reports the stale call itself; default captures never claim `capture_verified=True`, so no verdict is inflated |
| Deferred `trace.log_backward(...)` / `Recording.log_backward(...)` | Explicitly `not_armed` in this rollout |
| `torch.func` / functorch transform internals | Existing transform boundary warning/marker remains authoritative |
| `stacklevel`-attributed torch warnings raised inside wrapped Python functionals (e.g. `F.softmax` implicit-dim) | Declared residual while wrappers are installed: the wrapper adds one Python frame, so the warning is attributed to torch internals instead of the user call site, and Python's default-filter dedup (keyed on the attributed location) collapses DISTINCT user call sites into one warning per process. The frame is inherent to Python-level wrapping; pinned by `test_wrapped_functional_warning_attribution_residual_shape` |
| User/extension `__torch_function__` handler tables keyed by a namespace read taken WHILE wrappers are installed (`HANDLED = {torch.mean: ...}` after the first capture — the official "Extending torch" `@implements` pattern in a library imported mid-session) | Declared residual: the C protocol delivers the ORIGINAL as `func`, so a wrapper-keyed table silently misses during the wrapped epoch and after unwrap. TorchLens shims torch-INTERNAL tables only; it cannot rewrite arbitrary user registries. Remedies: build handler tables before the first capture, key by `torch.overrides.resolve_name(func)`, or normalize keys through `unwrap_torch()` |
| User-side callable-keyed registries generally (`REGISTRY = {F.relu: "relu"}` built before the first capture, membership-tested after it) | Declared residual while wrappers are installed: the namespace read now returns the wrapper, so identity/membership answers flip. Key by name (`resolve_name`) or call `unwrap_torch()` before consulting the registry |
| `inspect.signature` on wrapped C builtins (`torch.cos`) | Declared residual while wrappers are installed: pre-wrap it RAISES `ValueError` (no signature for a C builtin); post-wrap it returns the permissive `(*args, **kwargs)` fabrication, so a caller using the raise to detect "C builtin, cannot bind" gets a silent wrong answer. `__wrapped__` must stay deleted for JIT compatibility, so the fabrication is inherent |

The thread tripwire compares `threading.active_count()` at forward entry and exit. It catches a live
count delta cheaply, but a worker that starts and joins entirely inside the forward can evade that
sample. This is why the guarantee remains explicitly owner-thread-qualified.

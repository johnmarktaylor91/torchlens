# Scoped detached-reference patching rollout

TorchLens installs persistent wrappers around torch callables on the first torch capture. A Python
binding created before that installation—such as `from torch import relu`—can still point to the raw
callable after the torch namespace itself is wrapped. Detached-reference patching repairs reachable
bindings before forward execution.

## What ships in this rollout

The release default is unchanged: omitting `patch_policy` uses the legacy broad crawl. Scoped is real
and opt-in:

```python
import torchlens as tl

tl.wrap_torch(
    patch_policy="scoped",
    patch_modules=("my_project.model_helpers",),
    escape_detector="shadow",
)
trace = tl.trace(model, inputs)
```

Wrapper configuration is process-global, not per-trace. `patch_modules` is additive within a wrapper
epoch. Call `tl.unwrap_torch()` and then `tl.wrap_torch(...)` for a clean policy epoch.

The policies are:

| Policy | Direct module identities | Class/default deep scan | Source reads |
| --- | --- | --- | --- |
| `legacy` (release default) | Broad incremental scan | Existing source-filtered behavior | Possible |
| `scoped` (opt-in) | Exact new identities plus bounded hot set | Exact-positive, model provenance, prior-positive, and allowlisted modules | None |
| `full` | Broad scan | Every eligible module | No source prefilter |

The compatibility spelling `patch_detached_references(full=True)` and the string policy `"default"`
are deprecated. `"default"` resolves to the release default; it is not a scoped alias.

## Scoped is deliberately less permissive

Scoped drops the legacy source-mentions-`torch` deep scan for unrelated, non-provenance modules.
Consequently, a helper module whose only stale reference is hidden in a class attribute or function
default may previously have traced under the legacy default but remain unpatched under scoped. Add
that module/package through `patch_modules`, rebind after wrapping, use a live `torch.relu` lookup,
or choose `patch_policy="full"`.

This difference is intentional and must not be hidden by silently falling back to legacy. The
post-build certification gate is a shadow-mode soak over the full test-suite model zoo plus a
menagerie validation run. The gate requires zero unaudited convictions and checks for mass
regression before any enforcement or default flip.

## Shadow detector semantics

`escape_detector="shadow"` observes raw callable execution using exact object/code identity. It
reports a `TorchLensCaptureGapWarning` with the callable, registered export sites, source callsite,
storage hint, short stack, and remediation. Reports also appear in `trace.escape_diagnostics`.
Shadow mode does not substitute a wrapper, rerun the model, or raise the future completeness error.

Every wrapper-to-original edge uses a one-shot immediate-caller token. Tokens are identity-compatible
with transient Tensor-bound builtins by requiring a Tensor receiver and an exact method name from the
wrapped-method inventory. They never exempt the dynamic duration of a wrapper. Thus a raw descriptor
called by a user callback inside a composite wrapper remains reportable. TorchLens-internal
exceptions, if ever required, must match an exact parent/child/callsite row; the audited table has a
hard budget of 16 entries and ships empty.

Shadow is default-off. When it is enabled, the resulting trace remains unverified even with no
reports. Scoped traces are also unverified without shadow because the dispatcher completeness
witness is not part of this build.

## Machine-readable qualification

Live torch traces expose these diagnostic fields:

| Field | Meaning |
| --- | --- |
| `detached_patch_policy`, `detached_patch_epoch` | Effective process policy and wrapper epoch |
| `escape_detector_mode` | `"off"` or diagnostic `"shadow"` |
| `escape_diagnostics` | Structured raw-call reports accumulated across forward passes |
| `capture_verified`, `capture_verification_reason` | Completeness status; scoped is `False` pre-witness |
| `capture_owner_thread_id`, `capture_owner_thread_qualified` | Supported proof-domain owner |
| `capture_thread_count_start`, `capture_thread_count_end` | Cheap Python thread-count tripwire samples |
| `capture_thread_activity_detected` | Whether the count changed across a guarded forward |
| `capture_guard_passes` | Per-active-logging pass index, mode, and owner thread |
| `escape_detector_event_count`, `escape_detector_callback_ns` | Detector event and callback-cost counters |
| `escape_detector_backward_coverage` | `"not_armed"` for deferred backward in this rollout |

Public `tl.record(...)` uses the same guarded forward hot path and mirrors these fields onto the
returned `Recording`.

## Honest boundaries

| Channel | This rollout |
| --- | --- |
| Closure, dict/list, unrelated class/default, ordinary Python partial | Shadow reports when Python exposes the call |
| Saved Tensor method descriptor or Tensor-bound builtin | Shadow reports via descriptor compatibility |
| C `functools.partial` around a C builtin | Known profile blind spot; trace stays machine-readably unverified pre-witness |
| `DataLoader(num_workers=0)` callback executed inside forward | Owner-thread domain; shadow reports visible escapes |
| Worker process preprocessing before model invocation | Outside the armed model-forward domain |
| Model tensor work delegated to another thread/process | Unsupported; owner-thread qualification applies |
| Deferred `trace.log_backward(...)` / `Recording.log_backward(...)` | Explicitly `not_armed` in this rollout |
| `torch.func` / functorch transform internals | Existing transform boundary warning/marker remains authoritative |

The thread tripwire compares `threading.active_count()` at forward entry and exit. It catches a live
count delta cheaply, but a worker that starts and joins entirely inside the forward can evade that
sample. This is why the guarantee remains explicitly owner-thread-qualified.

## Rollout gates

No warning-only result is advertised as verified. Before scoped enforcement or a default flip, the
separate dispatcher witness must land with per-wrapper expected decomposition or bottom-level
barcode correlation—never duration-window correlation. Detector plus witness must then complete the
zero-unaudited-conviction test-zoo/menagerie soak, the adversarial holder matrix, version coverage,
and the honest legacy-no-guard versus scoped-with-guard performance gate.

"""Rescue re-run: recover escaped ops with a TorchFunctionMode net.

Stage-2 safety net (tri-lab verdict, 2026-08-12). The primary capture is
ALWAYS mode-free: an armed ``TorchFunctionMode`` flips torch's fused fast
paths (eval MultiheadAttention 3 ops -> 27 ops, output no longer
byte-identical), so arming during a normal capture would change what
TorchLens records. Instead, when a completed capture carries an ESCAPE
SIGNAL — an escape-detector diagnostic, the unattributed-tensor-args
provenance flag, or a typed output-attribution failure — the capture is
re-run once with the net armed. The net redirects any stale pre-wrap torch
function reference to its exact wrapper (``_orig_to_decorated``), so
recovered ops have full wrapper fidelity: torch pops the mode inside the
handler, so the redirect needs no dedup token (verified experimentally).

Honesty contract:

- A rescued capture is disclosed: ``capture_verified=False``,
  ``capture_verification_reason="mode_rescue_rerun"``, and a session-time
  ``rescue_rerun`` record (never portable — the forward ran twice and mode
  presence may de-fuse fast paths).
- A rescue that recovers nothing returns the PRIMARY (mode-free) trace,
  marked ``capture_verified=False`` with reason
  ``"escape_rescue_unrecovered"`` — the residual classes (worker-thread
  stale refs, de-moded composite interiors) are declared and disclosed,
  never silent.
- A non-re-runnable capture (streaming, halt predicates) skips the rescue
  and reports the escape as before.
"""

from __future__ import annotations

import threading
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from torch.overrides import TorchFunctionMode

from ... import _state
from ..._errors import OutputAttributionError, TorchLensCaptureGapWarning
from ...errors._base import TorchLensWarning
from ...utils.display import user_stacklevel
from ...utils.rng import log_current_rng_states, set_rng_from_saved_states

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "CaptureAttemptFailedWarning",
    "RescueTorchFunctionMode",
    "capture_with_rescue",
]


class CaptureAttemptFailedWarning(RuntimeWarning, TorchLensWarning):
    """Warning category for the one capture-attempt-failed advisory.

    A dedicated category (still a ``RuntimeWarning``, so user filters keep
    matching; also a ``TorchLensWarning`` so the documented root category
    covers it -- R64-1) lets the rescue driver DEFER the advisory while a rescue re-run
    is still possible: the warning tells the user diagnostics ride the
    exception (``exc.partial_log``), which is only truthful when that
    exception actually propagates. A successful rescue swallows the failure,
    so the deferred advisory is dropped; every path that re-raises flushes it
    first.
    """


_thread_local = threading.local()
"""Per-thread rescue state: the mode's ``busy`` token and the re-run guard.

Both are thread-local for the same reason. ``busy`` guards reentrancy of a
handler that fires on EVERY thread's torch calls. ``rescue_active`` guards
against a rescue re-run triggering a nested rescue, which is a property of one
call stack -- and as a process global it also let an unrelated thread's capture
inherit the suppression and silently lose its own safety net, while a capture
that is mid-rescue is doing model prep outside ``active_logging`` where the
admission refusal does not apply.
"""


def _rescue_is_active() -> bool:
    """Return whether this thread is already inside a rescue re-run.

    Returns
    -------
    bool
        ``True`` while this thread runs a rescue re-run.
    """

    return bool(getattr(_thread_local, "rescue_active", False))


class RescueTorchFunctionMode(TorchFunctionMode):
    """Redirect stale original-torch-function calls to their wrappers.

    Armed ONLY during a rescue re-run. When a call reaches the mode with a
    function TorchLens wrapped (a stale pre-wrap reference — the wrapped
    namespaces call wrappers directly), it is redirected to the wrapper so
    the op is logged exactly like a normal capture. Everything else passes
    through untouched. The handler gates on ``_state._logging_enabled`` so
    TorchLens-internal tensor work (postprocess, ``pause_logging`` regions)
    never redirects.
    """

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if getattr(_thread_local, "busy", False) or not _state._logging_enabled:
            return func(*args, **kwargs)
        decorated = _state._orig_to_decorated.get(id(func))
        if decorated is None:
            return func(*args, **kwargs)
        _thread_local.busy = True
        try:
            return decorated(*args, **kwargs)
        finally:
            _thread_local.busy = False


@contextmanager
def _record_emitted_warnings(seen: set[tuple[type, str]]) -> Iterator[None]:
    """Record every warning shown during the block, still forwarding it.

    One user ``tl.trace()`` call may run the forward twice (primary + rescue
    re-run); per-session advisory warnings (functorch boundary, provenance)
    must reach the user ONCE per trace call, not once per forward.
    """

    forward = warnings.showwarning

    def recorder(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        """Record ``(category, message)`` in ``seen``, then forward to the real handler."""

        seen.add((category, str(message)))
        forward(message, category, *args, **kwargs)

    warnings.showwarning = recorder
    try:
        yield
    finally:
        # Identity-checked restore (the profile-slot standard): a user or
        # callback that installed its own showwarning during the window must
        # not be silently reverted.
        if warnings.showwarning is recorder:
            warnings.showwarning = forward


@contextmanager
def _suppress_repeated_warnings(seen: set[tuple[type, str]]) -> Iterator[None]:
    """Drop warnings already emitted by the primary run; forward novel ones."""

    forward = warnings.showwarning

    def dedup(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        """Forward only warnings whose ``(category, message)`` is not already in ``seen``."""

        if (category, str(message)) in seen:
            return
        forward(message, category, *args, **kwargs)

    warnings.showwarning = dedup
    try:
        yield
    finally:
        if warnings.showwarning is dedup:
            warnings.showwarning = forward


@contextmanager
def _defer_capture_failed_warnings(
    deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]],
) -> Iterator[None]:
    """Hold back capture-attempt-failed advisories; forward everything else.

    The advisory points the user at ``exc.partial_log`` — truthful only when
    the exception propagates. While a rescue re-run may still swallow the
    failure, the advisory is parked in ``deferred``; the driver flushes it on
    every re-raising path and drops it when the rescue succeeds.
    """

    forward = warnings.showwarning

    def hold(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        """Park capture-failed advisories in ``deferred``; forward the rest."""

        if isinstance(category, type) and issubclass(category, CaptureAttemptFailedWarning):
            deferred.append((message, category, args, kwargs))
            return
        forward(message, category, *args, **kwargs)

    warnings.showwarning = hold
    try:
        yield
    finally:
        if warnings.showwarning is hold:
            warnings.showwarning = forward


def _flush_deferred_warnings(
    deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]],
) -> None:
    """Re-emit parked advisories through the current warning handler."""

    for message, category, args, kwargs in deferred:
        warnings.showwarning(message, category, *args, **kwargs)
    deferred.clear()


def _escape_signal(trace: Trace) -> str | None:
    """Return the escape-signal kind carried by a finished trace, if any.

    An authoritative POSITIVE verdict outranks the heuristic provenance
    flag: when the armed dispatch witness accounted for every dispatch and
    verified the capture (e.g. an ``autograd.grad`` boundary is a known
    no-provenance source), the flag is a false alarm and no rescue runs.
    """

    if getattr(trace, "capture_verified", None) is True:
        return None
    if getattr(trace, "escape_diagnostics", None):
        return "escape_detector_diagnostic"
    if getattr(trace, "_had_unattributed_tensor_args", False):
        return "unattributed_tensor_args"
    return None


def _op_name_counts(trace: Trace) -> Counter[str]:
    """Multiset of canonicalized op func names for recovery comparison.

    Mode presence respells tensor dunders through the override protocol
    (``__add__`` -> ``add``, the pinned stage-0 delta), so spellings are
    canonicalized by stripping underscores before diffing — otherwise every
    operator-using model would read as a false "recovery".

    Bookkeeping SOURCE nodes (``func_name == "none"``: minted
    ``internalsource`` adoptions, other functionless placeholders) are
    excluded: they represent the ABSENCE of a captured function. A perfect
    rescue replaces the primary's minted orphan-source with the real op, so
    counting them made the rescued trace read one ``none`` short —
    ``lost_ops=('none',)`` — and the two-sided oracle refused the recovery
    (the R16-1 false negative).
    """

    return Counter(
        name.strip("_")
        for op in getattr(trace, "ops", ())
        if (name := getattr(op, "func_name", None)) and name != "none"
    )


def _disclosure(
    *,
    trigger: str,
    recovered: bool,
    recovered_ops: tuple[str, ...] = (),
    lost_ops: tuple[str, ...] = (),
    primary_escape_diagnostics: tuple[Any, ...] = (),
    primary_error: str | None = None,
    rescue_error: str | None = None,
    residual_signal: str | None = None,
    skipped_reason: str | None = None,
    forward_runs: int = 2,
) -> dict[str, Any]:
    """Build the session-time ``rescue_rerun`` disclosure record."""

    return {
        "trigger": trigger,
        "recovered": recovered,
        "recovered_ops": recovered_ops,
        "lost_ops": lost_ops,
        "primary_escape_diagnostics": primary_escape_diagnostics,
        "primary_error": primary_error,
        "rescue_error": rescue_error,
        "residual_signal": residual_signal,
        "skipped_reason": skipped_reason,
        "forward_runs": forward_runs,
    }


def _buffer_write_labels(trace: Trace) -> tuple[str, ...]:
    """Labels of primary-forward ops that ACTUALLY wrote module buffer state.

    A rescue re-run executes the user's forward a SECOND time. When the
    primary forward wrote buffers (train-mode BatchNorm running stats and
    ``num_batches_tracked``, any in-forward buffer counter), the re-run
    double-applies those writes: RNG is restored between runs, module state is
    not restorable. Captures whose primary shows buffer writes therefore
    refuse the re-run. A custom in-forward PYTHON-attribute counter (not a
    registered buffer) still mutates twice on rescued captures -- the
    documented residual (see docs/migration/scoped_detached_patching.md).

    The refusal keys on an ACTUAL write, not on journal presence: fused norm
    mutators (``batch_norm``, ``instance_norm``, ``native_group_norm``) are
    journaled unconditionally, so every EVAL-mode BN/IN/GN capture carries
    ``buffer_write_kind`` records whose ``buffer_value_changed`` is ``False``
    (bytes provably unchanged; re-running is state-neutral). Only a record
    whose value changed -- or whose change status is unknown (fail closed) --
    refuses the re-run.
    """

    labels: list[str] = []
    for op in getattr(trace, "ops", ()) or ():
        if getattr(op, "buffer_write_kind", None) is None:
            continue
        if getattr(op, "buffer_value_changed", None) is False:
            continue
        label = getattr(op, "label_raw", None) or getattr(op, "layer_label", None)
        labels.append(str(label or getattr(op, "func_name", "?")))
    return tuple(labels)


def _partial_buffer_write_labels(exc: BaseException) -> tuple[str, ...] | None:
    """Buffer addresses ACTUALLY written by a FAILED primary forward.

    The failed-capture cleanup stamps the value-changing journal record on
    the exception (``_torchlens_actual_buffer_writes``) while the journal is
    still live — postprocess never ran, so materialized op fields do not
    exist, and session cleanup clears ``capture_events`` before the driver
    sees the partial. Mirrors :func:`_buffer_write_labels`: only
    value-changing (or unknown-change, fail closed) events count. The live
    partial journal is read as a fallback for paths that skipped the stamp.

    Returns
    -------
    tuple[str, ...] | None
        Addresses of actual writes (empty tuple = provably none), or ``None``
        when the write record is unreachable/unarmed — the caller must treat
        ``None`` as unprovable and refuse the re-run.
    """

    stamped = getattr(exc, "_torchlens_actual_buffer_writes", None)
    if stamped is not None:
        return tuple(str(address) for address in stamped)
    partial = getattr(exc, "partial_log", None)
    trace = getattr(partial, "trace", None)
    if trace is None:
        return None
    if getattr(trace, "capture_mode", None) != "exhaustive":
        # The buffer-write tracker only arms exhaustive sessions; an empty
        # journal on any other mode proves nothing.
        return None
    events = getattr(getattr(trace, "capture_events", None), "buffer_write_events", None)
    if events is None:
        return None
    return tuple(
        str(getattr(event, "address", None) or "?")
        for event in events
        if getattr(event, "value_changed", None) is not False
    )


def _snapshot_declared_state(model: Any) -> dict[str, Any] | None:
    """Byte snapshot of the model's declared state (parameters + ALL buffers).

    Taken ONLY on the escape path, immediately before a rescue re-run: the
    journal-based refusal above it is structurally blind to PARAMETER writes
    (the write index covers registered buffers only) and to writes performed
    BY the escaped op itself (an escaped call leaves no journal record), so
    journal emptiness is not proof of state-neutrality. The transient
    O(model-state) copy is confined to the rare escape path — the plain
    capture path pays nothing (the W6 arming contract).

    Returns
    -------
    dict[str, Any] | None
        ``slot key -> detached clone`` for every named parameter and buffer,
        or ``None`` when the model is unavailable.
    """

    if model is None or not isinstance(model, torch.nn.Module):
        return None
    snapshot: dict[str, Any] = {}
    # detach-ok: read-only pre-rescue state snapshot under no_grad; restored
    # verbatim on divergence, never a training-path payload.
    with _state.pause_logging(), torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter is not None:
                snapshot[f"param:{name}"] = parameter.detach().clone()
        for name, buffer in model.named_buffers():
            if buffer is not None:
                snapshot[f"buffer:{name}"] = buffer.detach().clone()
    return snapshot


def _state_bytes_equal(current: torch.Tensor, baseline: torch.Tensor) -> bool:
    """NaN-safe bitwise equality over two same-shape/dtype state tensors.

    This audit detects WRITES, so the compare must be byte-exact (a tolerant
    compare would hide a genuine small state write) and NaN-safe (IEEE
    ``torch.equal`` returns False for NaN==NaN, so a model legitimately
    holding a NaN parameter/buffer was falsely accused of a double-applied
    write and lost the rescue path). Element-extent uint8 reinterpret over
    resolved contiguous copies; exotic layouts fall back to the historical
    ``torch.equal`` verdict.
    """

    # detach-ok: read-only byte comparison of state snapshots; never a
    # training-path payload, nothing retains the detached views.
    try:
        current_bytes = (
            current.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
        ).view(torch.uint8)
        baseline_bytes = (
            baseline.detach().resolve_conj().resolve_neg().contiguous().reshape(-1)
        ).view(torch.uint8)
        return bool(torch.equal(current_bytes, baseline_bytes))
    except (RuntimeError, TypeError, NotImplementedError):
        return bool(torch.equal(current, baseline))


def _restore_changed_state(model: Any, snapshot: dict[str, Any]) -> tuple[str, ...]:
    """Restore snapshot values into every changed state slot; name the changes.

    Called after a rescue re-run: any slot whose bytes differ from the
    post-primary snapshot proves the rescue forward WROTE declared state — and
    therefore (same forward, same restored RNG) the primary wrote it too, so
    the model is doubly mutated. Copying the snapshot back leaves the model
    exactly as ONE forward left it. A slot that vanished or changed
    shape/dtype cannot be restored in place and is reported as-is.

    Returns
    -------
    tuple[str, ...]
        The slot keys whose values changed during the rescue re-run.
    """

    changed: list[str] = []
    with _state.pause_logging(), torch.no_grad():
        live: dict[str, Any] = {}
        for name, tensor in model.named_parameters():
            live[f"param:{name}"] = tensor
        for name, tensor in model.named_buffers():
            live[f"buffer:{name}"] = tensor
        for key, baseline in snapshot.items():
            current = live.get(key)
            if current is None:
                changed.append(key)
                continue
            try:
                # W1-FAB (weightsfree memo D6, defect L5): on a storage-less
                # substrate the byte comparison is not an observation — the
                # historical except-means-changed reading claimed the forward
                # wrote parameters it never touched, discarded every valid
                # meta rescue, and kept the degraded primary
                # (substrate-asymmetric). Weights-free the verdict is
                # UNKNOWN: no state-change claim, no restore copy.
                if current.is_meta or baseline.is_meta:
                    continue
                # NaN-aware oracle (r8 R16): ``torch.equal`` answers False for
                # bitwise-identical NaNs, so a state slot legitimately holding
                # NaN (a running stat poisoned upstream, a sentinel buffer)
                # read as "changed by the rescue" every run -- a false
                # double-mutation report plus a pointless restore copy.
                if (
                    current.shape == baseline.shape
                    and current.dtype == baseline.dtype
                    and _state_bytes_equal(current, baseline)
                ):
                    continue
                changed.append(key)
                if current.shape == baseline.shape and current.dtype == baseline.dtype:
                    current.data.copy_(baseline)
            except (RuntimeError, TypeError, NotImplementedError):
                changed.append(key)
        changed.extend(sorted(set(live) - set(snapshot)))
    return tuple(dict.fromkeys(changed))


# Verification reasons carrying MORE-SPECIFIC diagnostic content than the
# generic rescue stamps: a recovered rescue must not demote any of these to
# ``mode_rescue_rerun`` (b6-fable/b3-fable: only dynamo was protected, so an
# armed detector/witness verdict on the rescued run -- or a teardown failure
# -- was clobbered into the generic reason).
_SPECIFIC_VERIFICATION_REASONS = frozenset(
    {
        "dynamo_region_not_logged",
        "callable_escape_shadow_report",
        "dispatch_witness_unaccounted_ops",
        "input_boundary_unverifiable",
        "transform_call_route_unverified",
        "owner_thread_tripwire_changed",
        "escape_detector_teardown_failed",
    }
)


def _warn_rescue_success(trigger: str) -> None:
    """Disclose a SUCCESSFUL rescue re-run at the user's call site.

    Every rescue outcome that REFUSES to re-run warns; the success path was
    the only silent one (R67), yet it is the outcome that actually ran the
    user's forward twice -- doubling wall-clock and double-applying any
    undeclared Python side effects (prints, counters, appended lists, HTTP
    calls) that the declared-state journals cannot see.

    Parameters
    ----------
    trigger:
        Escape-signal trigger that motivated the re-run.
    """

    warnings.warn(
        "TorchLens re-ran the forward pass once to recover ops hidden by a "
        f"stale pre-wrap torch reference (trigger: {trigger}). The returned "
        "trace is the rescue capture (capture_verified=False, reason "
        "'mode_rescue_rerun', details on trace.rescue_rerun) and the model's "
        "forward executed TWICE -- undeclared Python side effects inside "
        "forward() double-applied. Remedy: fix the stale torch reference "
        "(bind/import torch functions after TorchLens has wrapped torch) to "
        "avoid the second forward.",
        TorchLensCaptureGapWarning,
        stacklevel=user_stacklevel(),
    )


def _mark(trace: Trace, reason: str, info: dict[str, Any]) -> None:
    """Stamp the rescue disclosure onto a trace (session-time facts).

    A rescue stamp must never SILENCE a more specific verdict: when the trace
    already carries a specific verification reason (dispatch witness, shadow
    detector, dynamo boundary, thread tripwire, teardown failure), that
    reason stays authoritative and the rescue attempt is disclosed only
    through ``rescue_rerun``. The ``escape_rescue_unrecovered`` reason is
    reserved for the formerly-silent class where the primary made no claim at
    all; ``mode_rescue_rerun`` replaces only a generic or absent reason.
    """

    trace.capture_verified = False
    existing_reason = getattr(trace, "capture_verification_reason", None)
    if getattr(trace, "_raw_dynamo_region_detected", False) or (
        existing_reason in _SPECIFIC_VERIFICATION_REASONS
    ):
        # R16-3 generalized: a specific verdict (dynamo top precedence, an
        # armed detector/witness report, the thread tripwire) outranks both
        # generic rescue stamps. The rescue attempt stays disclosed through
        # ``rescue_rerun`` below.
        pass
    elif reason == "mode_rescue_rerun" or not existing_reason:
        trace.capture_verification_reason = reason
    trace.rescue_rerun = info


def capture_with_rescue(
    run_capture: Callable[[], Trace],
    *,
    eligible: bool = True,
    model: Any = None,
) -> Trace:
    """Run one capture; on an escape signal, re-run once with the net armed.

    Parameters
    ----------
    run_capture:
        Zero-argument callable performing exactly one full capture with
        identical configuration each call.
    eligible:
        Whether a rescue re-run is permitted. Streaming saves, sinks, and
        halt-predicate partials are not re-runnable; they report the escape
        and skip the rescue.
    model:
        The live model being captured. When provided, the escape path takes a
        post-primary byte snapshot of the declared state and, after the
        rescue re-run, restores and refuses on ANY state write — covering the
        PARAMETER writes and escaped-op writes the journal-based refusal is
        structurally blind to. ``None`` (direct internal callers only) keeps
        the journal-only guard.

    Returns
    -------
    Trace
        The rescued trace when the re-run recovered ops (disclosed with
        reason ``"mode_rescue_rerun"``); otherwise the primary trace, marked
        ``"escape_rescue_unrecovered"`` when a signal fired, unchanged when
        no signal fired.
    """

    if _rescue_is_active():
        return run_capture()
    if not eligible:
        # R16-1: ineligibility skips the RE-RUN, never the DISCLOSURE. This
        # path formerly returned the primary with clean-capture fields
        # (verified=None / reason=None) even when an escape signal fired --
        # bit-indistinguishable from a genuinely clean capture across all
        # nine ineligible channels. Settle the escape on the trace exactly
        # like a refused re-run does.
        ineligible_trace = run_capture()
        signal = _escape_signal(ineligible_trace)
        if signal is not None:
            warnings.warn(
                "TorchLens detected an escape signal but skipped the rescue "
                "re-run: this capture uses a channel the re-run would invoke "
                "a second time (streaming/sink storage, disk grad storage, "
                "halt or intervention predicates, hooks, or a user transform "
                "callable). The escape stands unrecovered; fix the stale "
                "torch reference (or re-capture without the non-re-runnable "
                "channel) to recover the escaped ops.",
                # Capture-fidelity ceiling notice: routed through the typed
                # honesty category like the sibling escape-detection
                # disclosures, never bare UserWarning -- users filtering or
                # promoting torchlens advisories via TorchLensWarning must
                # see it (R66 / grind-r5 b8 fixwave-4 drift instance).
                TorchLensCaptureGapWarning,
                stacklevel=user_stacklevel(),
            )
            _mark(
                ineligible_trace,
                "escape_rescue_unrecovered",
                _disclosure(
                    trigger=signal,
                    recovered=False,
                    primary_escape_diagnostics=tuple(
                        getattr(ineligible_trace, "escape_diagnostics", ()) or ()
                    ),
                    skipped_reason="rescue_ineligible",
                    forward_runs=1,
                ),
            )
        return ineligible_trace

    rng_snapshot = log_current_rng_states()
    primary: Trace | None = None
    primary_error: OutputAttributionError | None = None
    emitted_warnings: set[tuple[type, str]] = set()
    primary_deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]] = []
    try:
        with (
            _record_emitted_warnings(emitted_warnings),
            _defer_capture_failed_warnings(primary_deferred),
        ):
            primary = run_capture()
    except OutputAttributionError as exc:
        primary_error = exc
    except BaseException:
        # No rescue for this failure class: the parked advisory is truthful
        # (the exception propagates with its diagnostics), so re-emit it.
        _flush_deferred_warnings(primary_deferred)
        raise

    if primary_error is not None:
        trigger = "output_attribution_failed"
        # R16-2 applies to EVERY trigger: an attribution-failed rescue also
        # runs the forward a second time, so a primary that WROTE buffer
        # state (train-mode BatchNorm counters and running stats) refuses the
        # re-run here too — the failed capture's journal is the write record.
        # An unreadable record is unprovable and refuses fail-closed.
        partial_writes = _partial_buffer_write_labels(primary_error)
        if partial_writes is None or partial_writes:
            _flush_deferred_warnings(primary_deferred)
            shown = (
                ", ".join(partial_writes[:3]) if partial_writes else "buffer-write state unprovable"
            )
            warnings.warn(
                "TorchLens skipped the rescue re-run after the output-attribution "
                f"failure: the forward wrote module buffer state ({shown}), and "
                "re-running it would double-apply those writes. Call model.eval() "
                "(or fix the stale torch reference) and re-capture.",
                UserWarning,
                stacklevel=3,
            )
            partial_trace = getattr(getattr(primary_error, "partial_log", None), "trace", None)
            if partial_trace is not None:
                try:
                    partial_trace.rescue_rerun = _disclosure(
                        trigger=trigger,
                        recovered=False,
                        primary_error=str(primary_error),
                        skipped_reason="buffer_writes_double_forward",
                        forward_runs=1,
                    )
                except Exception:  # noqa: BLE001 — disclosure is best-effort
                    pass
            raise primary_error
    else:
        assert primary is not None
        signal = _escape_signal(primary)
        if signal is None:
            return primary
        trigger = signal
        # R16-2: a rescue re-run executes the user's forward a SECOND time.
        # When the primary forward WROTE buffer state (train-mode BatchNorm
        # counters and running stats, in-forward buffer counters), the re-run
        # double-applies those writes (RNG is restored, module state is not),
        # so the re-run is refused and the escape stands disclosed.
        buffer_writes = _buffer_write_labels(primary)
        if buffer_writes:
            shown = ", ".join(buffer_writes[:3])
            warnings.warn(
                "TorchLens detected an escape signal but skipped the rescue "
                f"re-run: the forward wrote module buffer state ({shown}), and "
                "re-running it would double-apply those writes. The escape "
                "stands unrecovered; call model.eval() (or fix the stale torch "
                "reference) and re-capture.",
                UserWarning,
                stacklevel=3,
            )
            _mark(
                primary,
                "escape_rescue_unrecovered",
                _disclosure(
                    trigger=trigger,
                    recovered=False,
                    primary_escape_diagnostics=tuple(
                        getattr(primary, "escape_diagnostics", ()) or ()
                    ),
                    skipped_reason="buffer_writes_double_forward",
                    forward_runs=1,
                ),
            )
            return primary

    # R02-1/R02-2: the journal-based refusals above prove nothing about
    # PARAMETER writes (the write index covers registered buffers only) or
    # about writes performed BY the escaped op (which leaves no journal
    # record). Snapshot the declared state now — post-primary, escape path
    # only — so the re-run below can be byte-audited and undone.
    state_snapshot = _snapshot_declared_state(model)

    rescue_deferred: list[tuple[Any, Any, tuple[Any, ...], dict[str, Any]]] = []
    rescue_exc: Exception | None = None
    changed_state: tuple[str, ...] = ()
    try:
        # Armed INSIDE the try (the house set-inside-try standard): a
        # KeyboardInterrupt between an outside arm and the try's first line
        # leaked rescue_active=True for the thread's lifetime, silently
        # disabling every later rescue on it.
        _thread_local.rescue_active = True
        set_rng_from_saved_states(rng_snapshot)
        # The rescue run's own capture-failed advisory is deferred too: its
        # exception never propagates (the primary's error or trace does), so
        # an advisory pointing at ITS exc.partial_log would always be untrue.
        with (
            _suppress_repeated_warnings(emitted_warnings),
            _defer_capture_failed_warnings(rescue_deferred),
            RescueTorchFunctionMode(),
        ):
            rescued = run_capture()
    except Exception as exc:
        rescue_exc = exc
    finally:
        _thread_local.rescue_active = False
        # R63: the snapshot restore runs on EVERY exit — success, rescue
        # failure, and interrupt (KeyboardInterrupt propagates through this
        # finally). A failed or interrupted re-run may have already written
        # declared state before dying; skipping the restore left those
        # writes double-applied on the user's model.
        if state_snapshot is not None:
            changed_state = _restore_changed_state(model, state_snapshot)

    if rescue_exc is not None:
        if primary_error is not None:
            # The primary's failure propagates with its diagnostics attached,
            # so its parked advisory is truthful again — re-emit it.
            _flush_deferred_warnings(primary_deferred)
            raise primary_error from None
        assert primary is not None
        _mark(
            primary,
            "escape_rescue_unrecovered",
            _disclosure(
                trigger=trigger,
                recovered=False,
                primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
                rescue_error=f"{type(rescue_exc).__name__}: {rescue_exc}",
            ),
        )
        return primary

    if state_snapshot is not None:
        if changed_state:
            # The rescue forward WROTE declared state, so the primary wrote it
            # too and the writes were double-applied. The snapshot restore
            # above already undid the second application (the model now holds
            # exactly one forward's worth of mutation); the rescue trace is a
            # second, differently-parameterized forward and is discarded.
            shown = ", ".join(changed_state[:3])
            warnings.warn(
                "TorchLens discarded the rescue re-run: the forward wrote model "
                f"state ({shown}) the buffer-write journal could not see, and "
                "running it twice double-applied those writes. The duplicate "
                "application was restored from a snapshot, so the model now "
                "holds exactly one forward's mutation. The escape stands "
                "unrecovered; fix the stale torch reference (or make the "
                "forward state-neutral) and re-capture.",
                UserWarning,
                stacklevel=3,
            )
            if primary_error is not None:
                _flush_deferred_warnings(primary_deferred)
                partial_trace = getattr(getattr(primary_error, "partial_log", None), "trace", None)
                if partial_trace is not None:
                    try:
                        partial_trace.rescue_rerun = _disclosure(
                            trigger=trigger,
                            recovered=False,
                            primary_error=str(primary_error),
                            skipped_reason="state_writes_double_forward_undone",
                            forward_runs=2,
                        )
                    except Exception:  # noqa: BLE001 — disclosure is best-effort
                        pass
                raise primary_error
            assert primary is not None
            _mark(
                primary,
                "escape_rescue_unrecovered",
                _disclosure(
                    trigger=trigger,
                    recovered=False,
                    primary_escape_diagnostics=tuple(
                        getattr(primary, "escape_diagnostics", ()) or ()
                    ),
                    skipped_reason="state_writes_double_forward_undone",
                    forward_runs=2,
                ),
            )
            return primary

    if primary_error is not None:
        # The primary could not even attribute its output; a completed rescue
        # capture is the recovery by definition.
        _warn_rescue_success(trigger)
        _mark(
            rescued,
            "mode_rescue_rerun",
            _disclosure(
                trigger=trigger,
                recovered=True,
                primary_error=str(primary_error),
                residual_signal=_escape_signal(rescued),
            ),
        )
        return rescued

    assert primary is not None
    # R16-1: the recovery oracle is TWO-SIDED. ``Counter.__sub__`` alone drops
    # losses, so mode-induced de-fusion (eval MHA: 3 fused ops -> 27 small
    # ops) read as pure gains and a benign false alarm silently swapped the
    # user's canonical fused trace for a structurally different
    # mode-perturbed one marked recovered. A rescue counts as recovery ONLY
    # when the rescued op multiset is a strict SUPERSET of the primary's;
    # any loss means mode perturbation, and the mode-free primary stays
    # authoritative with both deltas disclosed.
    primary_counts = _op_name_counts(primary)
    rescued_counts = _op_name_counts(rescued)
    recovered_counts = rescued_counts - primary_counts
    lost_counts = primary_counts - rescued_counts
    if recovered_counts and not lost_counts:
        _warn_rescue_success(trigger)
        _mark(
            rescued,
            "mode_rescue_rerun",
            _disclosure(
                trigger=trigger,
                recovered=True,
                recovered_ops=tuple(sorted(recovered_counts.elements())),
                primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
                residual_signal=_escape_signal(rescued),
            ),
        )
        return rescued

    # Nothing new, or a structurally different (mode-perturbed) graph: keep
    # the mode-free primary and disclose that the escape stands unrecovered,
    # including both deltas and whether the rescue still carried the signal.
    _mark(
        primary,
        "escape_rescue_unrecovered",
        _disclosure(
            trigger=trigger,
            recovered=False,
            recovered_ops=tuple(sorted(recovered_counts.elements())),
            lost_ops=tuple(sorted(lost_counts.elements())),
            primary_escape_diagnostics=tuple(getattr(primary, "escape_diagnostics", ()) or ()),
            residual_signal=_escape_signal(rescued),
        ),
    )
    return primary

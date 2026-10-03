"""Witness finalization and capture scope."""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ..._errors import TorchLensCaptureGapWarning
from ...utils._torch_symbols import torch_attr
from .escape_detection import (
    ExpectedOriginalToken,
)

if TYPE_CHECKING:
    from .completeness_witness import (
        _ACTIVE_WITNESS_STATE,
        _HOST_ESCAPE_MUTABLE_WRITEBACK,
        CompletenessWitnessMode,
        _CompletenessDispatchMode,
        _DispatchCallsite,
        _event_is_capture_accounted,
        _finalize_runnable_ledger,
        _is_expected_opaque_dispatch,
        _observe_invisible_host_escapes,
        _register_authorized_caller_namespace,
        _whole_storage_uint8,
        _WitnessState,
    )

__all__ = (
    "_STORAGE_RAW_POINTER_TARGETS",
    "_MODULE_ESCAPE_TARGETS",
    "_check_writeback_watch",
    "_effective_mode",
    "_barcode_text",
    "_finalize_census",
    "_reports_include_non_input_boundary",
    "_finalize_input_semantics_without_census",
    "capture_completeness_witness",
    "_collect_authorized_internal_caller_modules",
)


def _STORAGE_RAW_POINTER_TARGETS() -> tuple[Any, ...]:
    """Return storage classes whose ``data_ptr`` accessor leaks the raw pointer (r16-C1).

    ``tensor.untyped_storage()`` yields a ``torch.UntypedStorage`` and ``tensor.storage()`` a
    ``torch.TypedStorage``; ``data_ptr()`` on either hands out the same raw pointer the r15 Tensor
    patch fails closed on. Both are Python-visible classes whose ``data_ptr`` method is patchable.
    """

    targets: list[Any] = []
    for name in ("UntypedStorage", "TypedStorage"):
        cls = torch_attr(name)  # r47 secD_1: no lazy ``torch.__getattr__``
        if cls is not None and hasattr(cls, "data_ptr"):
            targets.append(cls)
    return tuple(targets)


def _MODULE_ESCAPE_TARGETS() -> tuple[tuple[Any, str], ...]:
    """Return ``(module, attribute)`` pairs for module-level zero-copy export C bindings."""

    targets: list[tuple[Any, str]] = []
    dlpack_mod = getattr(torch.utils, "dlpack", None)
    if dlpack_mod is not None:
        targets.append((dlpack_mod, "to_dlpack"))
    # The PUBLIC ``torch.to_dlpack`` alias is the SAME object re-exported as a separate
    # module attribute: patching ``torch.utils.dlpack.to_dlpack`` repoints only that
    # site, so a forward calling ``torch.to_dlpack(x)`` escaped the witness entirely
    # (the exact unswept-alias sibling of the decorated ``from_dlpack`` pair).
    if hasattr(torch, "to_dlpack"):
        targets.append((torch, "to_dlpack"))
    c_mod = getattr(torch, "_C", None)
    if c_mod is not None and hasattr(c_mod, "_to_dlpack"):
        targets.append((c_mod, "_to_dlpack"))
    return tuple(targets)


def _check_writeback_watch(state: _WitnessState) -> None:
    """Detect a host write-back through any watched mutable zero-copy alias at forward end.

    The honest rule is keyed on the WHOLE aliased storage's BYTES, never on the version counter
    (r14-H1) and never on only the view's element extent (r15-H2). A watched source whose whole
    storage is UNCHANGED since the mutable-alias exposure was only read: it stays VERIFIED (a pure
    read-only ``.numpy().sum()`` / storage-pointer identity check is not over-triggered). A watched
    source whose storage bytes CHANGED anywhere -- INCLUDING outside the view's own window (a
    storage ``__setitem__`` / ``np.as_strided`` write) -- is UNVERIFIABLE, in BOTH sub-cases:

    * version UNCHANGED -> no tracked op touched it, so the byte diff can only be an opaque host
      write-back through the alias (no aten dispatch, no version bump) -> host write-back;
    * version BUMPED -> a tracked in-place op ALSO touched the source since the exposure, so the
      raw byte comparison is AMBIGUOUS -- the diff could be the tracked op OR an additional host
      write layered on top, and cannot prove the ABSENCE of a host write -> conservatively opaque.

    Gating detection on ``version unchanged`` (the pre-r14 behaviour) let a tracked in-place op
    that bumps the version AFTER the ``.numpy()`` / ``.data`` snapshot skip the byte compare, so a
    host write-back on the same storage went undetected and the run falsely VERIFIED. Comparing
    bytes alone closes that gate while keeping read-only exposures honestly VERIFIED.
    """

    if not state.writeback_watch:
        return
    try:
        with _state.pause_logging():
            for source, _version, before in state.writeback_watch:
                try:
                    if not torch.equal(
                        _whole_storage_uint8(source), before
                    ):  # byte-exact uint8 view
                        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                        break
                except (RuntimeError, TypeError, NotImplementedError):
                    _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                    break
    finally:
        state.writeback_watch.clear()


def _effective_mode() -> CompletenessWitnessMode:
    """Return the validated process-level witness mode.

    Returns
    -------
    CompletenessWitnessMode
        Current dispatcher witness mode.
    """

    mode = _state._completeness_witness_mode
    if mode not in {"off", "shadow"}:
        raise RuntimeError(f"Invalid TorchLens completeness witness mode {mode!r}.")
    return cast(CompletenessWitnessMode, mode)


def _barcode_text(value: object | None) -> str | None:
    """Return a stable diagnostic rendering of a wrapper barcode.

    Parameters
    ----------
    value:
        Random wrapper barcode or ``None``.

    Returns
    -------
    str | None
        String barcode suitable for a machine-readable report.
    """

    return None if value is None else str(value)


def _finalize_census(state: _WitnessState) -> None:
    """Cross-check dispatch ownership and attach structured Trace diagnostics.

    Parameters
    ----------
    state:
        Completed per-forward census.
    """

    trace = state.trace
    owner_events: dict[int, tuple[ExpectedOriginalToken, list[str]]] = {}
    # Owners whose aten dispatch fired inside a genuine raw replacement hook. The
    # census excuses ONLY these orphaned owners (replacement construction) -- an
    # orphaned owner outside a replacement hook stays a real silent-drop mismatch.
    owner_in_replacement_hook: dict[int, bool] = {}
    diagnostics: list[dict[str, Any]] = []
    expected_opaque_count = 0
    accounted_count = 0
    for event_index, event in enumerate(state.events, start=1):
        owner = event.owner
        if owner is not None:
            owner_entry = owner_events.setdefault(id(owner), (owner, []))
            owner_entry[1].append(event.operator)
            if event.in_replacement_hook:
                owner_in_replacement_hook[id(owner)] = True
        if owner is not None and _is_expected_opaque_dispatch(event.operator, owner):
            expected_opaque_count += 1
            continue
        if _event_is_capture_accounted(event):
            accounted_count += 1
            continue
        reason = "unowned_dispatch" if owner is None else "owner_not_captured"
        callsite = event.callsite
        if callsite is None and owner is not None and owner.capture_callsite is not None:
            callsite = _DispatchCallsite(*owner.capture_callsite)
        diagnostics.append(
            {
                "violation_id": len(diagnostics) + 1,
                "event_index": event_index,
                "operator": event.operator,
                "reason": reason,
                "owner_wrapper": owner.wrapper_name if owner is not None else None,
                "owner_func_name": owner.func_name if owner is not None else None,
                "owner_func_call_id": owner.func_call_id if owner is not None else None,
                "owner_barcode": _barcode_text(owner.call_barcode) if owner is not None else None,
                "file": callsite.file if callsite is not None else None,
                "line": callsite.line if callsite is not None else None,
                "function": callsite.function if callsite is not None else None,
                "owner_thread_id": state.owner_thread_id,
                "guard_pass_index": state.guard_pass_index,
                "capture_mode": getattr(trace, "capture_mode", None),
                "scope": "active_logging",
                "enforced": False,
                "in_replacement_hook": event.in_replacement_hook,
                "mutates": event.mutates,
                # A ``.data``-property accessor view (``aten.detach``/``aten.alias``) on a
                # registered buffer -- the intrinsic, legitimately-uncaptured dispatch of the
                # ``self.b.data.copy_(x)`` buffer-write idiom. The completeness backstop credits
                # these apples-to-apples against the dispatch census (see
                # ``validation.core.completeness_backstop_counts``); a genuine untraced op is
                # never flagged here (unowned + non-mutating + pure-view + buffer only).
                "state_view_accessor": event.state_view_accessor,
            }
        )
    decompositions = trace.__dict__.setdefault("completeness_decompositions", [])
    for owner, operators in owner_events.values():
        owner_scope = (
            "expected_opaque"
            if operators
            and all(_is_expected_opaque_dispatch(operator, owner) for operator in operators)
            else owner.census_scope
        )
        decompositions.append(
            {
                "guard_pass_index": state.guard_pass_index,
                "owner_wrapper": owner.wrapper_name,
                "owner_func_name": owner.func_name,
                "owner_func_call_id": owner.func_call_id,
                "owner_barcode": _barcode_text(owner.call_barcode),
                "capture_accounted": owner.capture_accounted,
                "capture_accounted_boundary_labels": tuple(
                    raw_label for _, raw_label in owner.capture_accounted_outputs.values()
                ),
                "scope": owner_scope,
                "aten_ops": tuple(operators),
                "in_replacement_hook": owner_in_replacement_hook.get(id(owner), False),
            }
        )
    reports = trace.__dict__.setdefault("completeness_diagnostics", [])
    reports.extend(diagnostics)
    trace.completeness_witness_event_count = int(
        getattr(trace, "completeness_witness_event_count", 0)
    ) + len(state.events)
    trace.completeness_witness_accounted_count = (
        int(getattr(trace, "completeness_witness_accounted_count", 0)) + accounted_count
    )
    trace.completeness_witness_expected_opaque_count = (
        int(getattr(trace, "completeness_witness_expected_opaque_count", 0)) + expected_opaque_count
    )
    trace.completeness_witness_unaccounted_count = int(
        getattr(trace, "completeness_witness_unaccounted_count", 0)
    ) + len(diagnostics)
    trace.completeness_witness_callback_ns = (
        int(getattr(trace, "completeness_witness_callback_ns", 0)) + state.callback_ns
    )
    trace.completeness_witness_verified = not reports
    if reports:
        trace.capture_verified = False
        trace.capture_verification_reason = (
            "dispatch_witness_unaccounted_ops"
            if diagnostics or _reports_include_non_input_boundary(reports)
            else "input_boundary_unverifiable"
        )
        if diagnostics:
            first = diagnostics[0]
            warnings.warn(
                "TorchLens completeness witness observed "
                f"{len(diagnostics)} unaccounted aten dispatch event(s); first: "
                f"{first['operator']} ({first['reason']}). The Trace is marked "
                "capture_verified=False; inspect trace.completeness_diagnostics.",
                TorchLensCaptureGapWarning,
                stacklevel=3,
            )
        return
    if getattr(trace, "_raw_dynamo_region_detected", False):
        # More specific than either reason below: the transform-escape flag is shared with
        # the functorch boundary, and a shadow-mode escape report is a downstream symptom
        # of the same bypassed compiled region.
        trace.capture_verified = False
        trace.capture_verification_reason = "dynamo_region_not_logged"
    elif getattr(trace, "escape_detector_verified", None) is False:
        trace.capture_verified = False
        trace.capture_verification_reason = "callable_escape_shadow_report"
    elif getattr(trace, "_raw_transform_escape_detected", False):
        trace.capture_verified = False
        trace.capture_verification_reason = "transform_call_route_unverified"
    else:
        trace.capture_verified = True
        detector_verified = getattr(trace, "escape_detector_verified", None)
        trace.capture_verification_reason = (
            "dispatch_witness_and_detector_verified"
            if detector_verified is True
            else "dispatch_witness_verified"
        )


def _reports_include_non_input_boundary(reports: Any) -> bool:
    """Return whether accumulated reports include a non-input-boundary gap.

    Parameters
    ----------
    reports:
        Sequence of previously accumulated completeness diagnostics.

    Returns
    -------
    bool
        ``True`` when any report is not an ``input_boundary``-scoped entry.
    """

    return any(
        not isinstance(report, Mapping) or report.get("scope") != "input_boundary"
        for report in reports
    )


def _finalize_input_semantics_without_census(trace: Any) -> None:
    """Apply input-boundary fail-closed state when the dispatch census is off.

    Parameters
    ----------
    trace:
        Trace or Recording runtime trace carrying private input-gap diagnostics.

    Returns
    -------
    None
        Ceilings ``capture_verified`` without claiming that the disabled dispatch
        witness itself ran.
    """

    reports = getattr(trace, "completeness_diagnostics", ())
    if not any(
        isinstance(report, Mapping) and report.get("scope") == "input_boundary"
        for report in reports
    ):
        return
    trace.capture_verified = False
    trace.capture_verification_reason = "input_boundary_unverifiable"


def _init_witness_fields(trace: Any, mode: str) -> None:
    """Initialize the witness disclosure fields and counters on the trace."""

    trace.completeness_witness_mode = mode
    if not hasattr(trace, "completeness_witness_verified"):
        trace.completeness_witness_verified = None
    trace.__dict__.setdefault("completeness_diagnostics", [])
    trace.__dict__.setdefault("completeness_decompositions", [])
    for counter_field in (
        "completeness_witness_event_count",
        "completeness_witness_accounted_count",
        "completeness_witness_expected_opaque_count",
        "completeness_witness_unaccounted_count",
        "completeness_witness_callback_ns",
    ):
        trace.__dict__.setdefault(counter_field, 0)


def _build_witness_state(
    trace: Any,
    mode: str,
    *,
    record_escapes: bool,
    record_aten: bool,
    plane_p: bool,
    event_stream: Any,
) -> _WitnessState:
    """Construct the per-forward witness state for the dispatch mode."""

    guard_passes = getattr(trace, "capture_guard_passes", [])
    guard_pass_index = len(guard_passes) if guard_passes else 1
    return _WitnessState(
        trace,
        threading.get_ident(),
        guard_pass_index,
        census=(mode == "shadow"),
        record_escapes=record_escapes,
        ledger=record_escapes,
        record_aten=record_aten,
        plane_p=plane_p,
        aten_events=event_stream,
        capture_phase="forward",
    )


def _plane_p_requested() -> bool:
    """Whether the distributed opt-in is armed (plane-P observes this capture).

    Merge-ranks C2: plane-P physical dispatch observation rides the shared
    TorchLens dispatch mode for ARMED captures only, so unarmed dense capture
    keeps its exact pre-C2 dispatch-mode-free path (zero interference). An
    armed process with no initialized process group (armed before creating
    one, or after ``destroy_process_group``) is dormant: no collective can run,
    so the capture takes the same unarmed path.
    """

    try:
        from torchlens.distributed._lifecycle import capture_armed_state

        return capture_armed_state() is not None
    except Exception:
        return False


def _finalize_plane_p(state: _WitnessState) -> None:
    """Publish the plane-P dispatch journal on the trace (session-only).

    Stamps ``trace._distributed_plane_p`` (FieldPolicy.DROP, private-named,
    never persisted) with the per-dispatch records the capture-fidelity census
    criteria 2-4 consume. A no-op for states without plane-P.
    """

    if not state.plane_p:
        return
    state.trace._distributed_plane_p = {
        "schema": "plane_p_dispatch_v0",
        "records": tuple(state.plane_p_events),
    }


@contextmanager
def capture_completeness_witness(trace: Any) -> Iterator[None]:
    """Optionally run an aten census around one active-logging forward.

    Parameters
    ----------
    trace:
        Trace or Recording runtime trace receiving diagnostics.

    Yields
    ------
    None
        The backend enters active logging inside this context.
    """

    mode = _effective_mode()
    _init_witness_fields(trace, mode)
    # A runnable-eligible (``intervention_ready``) capture always records
    # tensor->host escape sources so the sparse descriptor can witness the escape
    # by its producing op, keyed on the ESCAPE EVENT. This is a passive observer:
    # it records raw op labels only and never alters a captured op, so goldens are
    # unchanged. The default (non-runnable) capture path installs nothing.
    from ._aten_capture import _aten_recording_requested

    record_escapes = bool(getattr(trace, "intervention_ready", False))
    record_aten = _aten_recording_requested()
    plane_p = _plane_p_requested()
    event_stream = getattr(trace, "capture_events", None)
    if record_aten and event_stream is not None:
        event_stream.aten_recording_enabled = True
    if mode == "off" and not record_escapes and not record_aten and not plane_p:
        try:
            yield
        finally:
            _finalize_input_semantics_without_census(trace)
        return
    state = _build_witness_state(
        trace,
        mode,
        record_escapes=record_escapes,
        record_aten=record_aten,
        plane_p=plane_p,
        event_stream=event_stream,
    )
    mode_context = _CompletenessDispatchMode(state)
    # plane-P owner attribution needs the wrapper ownership tokens minted even
    # when the aten recorder itself is off.
    with _state.aten_recording(record_aten or plane_p):
        # A runnable capture additionally observes census-INVISIBLE ``.tolist()`` /
        # ``.numpy()`` / ``__array__`` escapes via a scoped method patch so every escape
        # mechanism feeds one uniform source-witness pass. The patch is a pure observer,
        # restored unconditionally, and is skipped entirely for the non-runnable census path.
        if record_escapes:
            # r35 I2: arm wrapper ownership tokens so raised / host-returning dispatch
            # events can be attributed to their exact wrapper owner (the ledger's
            # owner-accounted discharge rule) even with both shadow modes off.
            prior_ledger_armed = _state._runnable_ledger_armed
            _state._runnable_ledger_armed = True
            # r43: publish the witness state so wrappers.py string-hook interception can
            # classify owner vs non-owner (the ONE place a non-owner thread must not flip
            # the global ``pause_logging`` toggle). Cleared FIRST on exit.
            global _ACTIVE_WITNESS_STATE
            prior_active_state = _ACTIVE_WITNESS_STATE
            _ACTIVE_WITNESS_STATE = state
            try:
                with _observe_invisible_host_escapes(state), mode_context:
                    try:
                        yield
                    finally:
                        if mode == "shadow":
                            _finalize_census(state)
                        else:
                            _finalize_input_semantics_without_census(trace)
                        _finalize_runnable_ledger(state)
                        _finalize_plane_p(state)
            finally:
                _ACTIVE_WITNESS_STATE = prior_active_state
                _state._runnable_ledger_armed = prior_ledger_armed
            return
        with mode_context:
            try:
                yield
            finally:
                if mode == "shadow":
                    _finalize_census(state)
                else:
                    _finalize_input_semantics_without_census(trace)
                _finalize_plane_p(state)


def _collect_authorized_internal_caller_modules() -> None:
    """Register the witness-family and buffer-write modules' own code objects.

    The witness was one module until the r3 split; every ``_completeness_*``
    sibling (plus the ``completeness_witness`` facade) holds internal callers
    that must stay on the authorized roster, or the detector self-trips on the
    witness's own no-observe storage reads (the r3 cluster regression).
    """

    import importlib
    import pkgutil

    from . import buffer_writes as _buffer_writes_module

    _register_authorized_caller_namespace(globals(), __file__)
    buffer_writes_file = _buffer_writes_module.__file__
    assert buffer_writes_file is not None, "a real source module always has a file"
    _register_authorized_caller_namespace(vars(_buffer_writes_module), buffer_writes_file)
    package = importlib.import_module(__package__)
    package_path = package.__path__
    for module_info in pkgutil.iter_modules(package_path):
        name = module_info.name
        if not (name.startswith("_completeness_") or name == "completeness_witness"):
            continue
        module = importlib.import_module(f"{__package__}.{name}")
        module_file = module.__file__
        assert module_file is not None, "a real source module always has a file"
        _register_authorized_caller_namespace(vars(module), module_file)

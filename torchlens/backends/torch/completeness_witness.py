"""Opt-in aten-dispatch completeness witness for torch capture.

The witness correlates dispatcher events to the exact wrapper edge token and
leaf barcode used by callable escape detection and ordinary TorchLens capture.
It deliberately does not use clocks or inferred time windows.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ... import _state
from ..._errors import TorchLensCaptureGapWarning
from .escape_detection import ExpectedOriginalToken, _active_token

CompletenessWitnessMode = Literal["off", "shadow"]
"""Supported dispatcher-witness rollout modes."""

MAX_AUDITED_COMPLETENESS_BOUNDARIES = 8
"""Hard budget preventing expected-opaque wrapper scopes from growing unchecked."""


@dataclass(frozen=True)
class AuditedCompletenessBoundary:
    """One exact wrapper/operator boundary that is intentionally not captured."""

    wrapper_name: str
    operator: str | None
    reason: str


AUDITED_COMPLETENESS_BOUNDARIES: tuple[AuditedCompletenessBoundary, ...] = (
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:numpy:not_logged",
        operator=None,
        reason="existing metadata/export conversion boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__array__:not_logged",
        operator=None,
        reason="existing NumPy protocol conversion boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:size:not_logged",
        operator=None,
        reason="existing tensor shape metadata boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:dim:not_logged",
        operator=None,
        reason="existing tensor rank metadata boundary; TorchLens never records it as an op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:item:logged",
        operator="aten._local_scalar_dense.default",
        reason="item extracts a Python scalar, and TorchLens intentionally records no scalar-output op",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__bool__:logged",
        operator="aten._local_scalar_dense.default",
        reason="tensor truth testing extracts a Python bool, which is intentionally not an op output",
    ),
)
"""Reviewable exact expected-opaque rows; additions require a regression test and reason."""

if len(AUDITED_COMPLETENESS_BOUNDARIES) > MAX_AUDITED_COMPLETENESS_BOUNDARIES:
    raise RuntimeError("TorchLens completeness boundary budget exceeded.")

_EXPECTED_OPAQUE_WRAPPERS = frozenset(
    row.wrapper_name for row in AUDITED_COMPLETENESS_BOUNDARIES if row.operator is None
)


def _is_expected_opaque_dispatch(operator: str, owner: ExpectedOriginalToken) -> bool:
    """Return whether one owned dispatch exactly matches an audited boundary.

    Parameters
    ----------
    operator:
        Stable dispatcher operator name.
    owner:
        Exact wrapper token active for the dispatch.

    Returns
    -------
    bool
        ``True`` for an exact wrapper/operator row or a wrapper-wide metadata boundary.
    """

    return any(
        row.wrapper_name == owner.wrapper_name
        and (row.operator is None or row.operator == operator)
        for row in AUDITED_COMPLETENESS_BOUNDARIES
    )


def completeness_scope_for_wrapper(
    wrapper_name: str,
) -> Literal["owned", "expected_opaque"]:
    """Return the exact audited census scope for a wrapper edge.

    Parameters
    ----------
    wrapper_name:
        Stable wrapper edge name.

    Returns
    -------
    Literal["owned", "expected_opaque"]
        Audited scope; unknown wrappers always remain owned and fail closed.
    """

    return "expected_opaque" if wrapper_name in _EXPECTED_OPAQUE_WRAPPERS else "owned"


@dataclass(frozen=True)
class _DispatchCallsite:
    """Stable user-side source location captured at dispatch time."""

    file: str
    line: int
    function: str


@dataclass
class _DispatchEvent:
    """One aten dispatcher event and its exact live wrapper owner, if any."""

    operator: str
    owner: ExpectedOriginalToken | None
    callsite: _DispatchCallsite | None


@dataclass
class _WitnessState:
    """Per-forward dispatch census state."""

    trace: Any
    owner_thread_id: int
    guard_pass_index: int
    events: list[_DispatchEvent] = field(default_factory=list)
    callback_ns: int = 0


def _operator_name(func: Any) -> str:
    """Return a stable dispatcher operator and overload name.

    Parameters
    ----------
    func:
        Dispatcher callable received by ``__torch_dispatch__``.

    Returns
    -------
    str
        Best-effort qualified operator name such as ``aten.relu.default``.
    """

    try:
        return str(func)
    except Exception:
        return type(func).__name__


def _is_aten_operator(func: Any) -> bool:
    """Return whether a dispatcher callable belongs to the aten namespace.

    Parameters
    ----------
    func:
        Dispatcher callable received by ``__torch_dispatch__``.

    Returns
    -------
    bool
        ``True`` only for the aten census domain.
    """

    namespace = getattr(func, "namespace", None)
    if isinstance(namespace, str):
        return namespace == "aten"
    return _operator_name(func).startswith("aten.")


def _dispatch_callsite() -> _DispatchCallsite:
    """Return the first non-framework frame above the dispatch callback.

    Returns
    -------
    _DispatchCallsite
        Best-effort source location for an unowned dispatcher event.
    """

    frame: Any = sys._getframe(2)
    torch_root = Path(torch.__file__).resolve().parent
    torchlens_root = Path(__file__).resolve().parents[2]
    fallback = frame
    while frame is not None:
        filename = frame.f_code.co_filename
        try:
            resolved = Path(filename).resolve()
            framework_frame = resolved.is_relative_to(torch_root) or resolved.is_relative_to(
                torchlens_root
            )
        except (OSError, RuntimeError, ValueError):
            framework_frame = False
        if not framework_frame:
            return _DispatchCallsite(filename, frame.f_lineno, frame.f_code.co_name)
        fallback = frame
        frame = frame.f_back
    return _DispatchCallsite(
        fallback.f_code.co_filename,
        fallback.f_lineno,
        fallback.f_code.co_name,
    )


def record_uncaptured_owner_callsite(token: ExpectedOriginalToken | None) -> None:
    """Attach a user callsite only when an owned interval emitted no op.

    Parameters
    ----------
    token:
        Completed exact wrapper token, if diagnostics were armed.

    Returns
    -------
    None
        The token receives a stable file, line, and function tuple in place.
    """

    if token is None:
        return
    callsite = _dispatch_callsite()
    token.capture_callsite = (callsite.file, callsite.line, callsite.function)


class _CompletenessDispatchMode(TorchDispatchMode):
    """Census aten calls while TorchLens active logging is enabled."""

    def __init__(self, state: _WitnessState) -> None:
        """Store the per-forward witness state.

        Parameters
        ----------
        state:
            Mutable event census for this forward pass.
        """

        super().__init__()
        self.state = state

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Record an in-scope aten event, then redispatch it unchanged.

        Parameters
        ----------
        func:
            Dispatcher operator overload.
        types:
            Participating tensor subclass types.
        args:
            Positional dispatcher arguments.
        kwargs:
            Keyword dispatcher arguments.

        Returns
        -------
        Any
            The unmodified operator result.
        """

        del types
        started = time.perf_counter_ns()
        try:
            in_scope = (
                threading.get_ident() == self.state.owner_thread_id
                and _state._logging_enabled
                and _state._active_trace is self.state.trace
                and _is_aten_operator(func)
            )
            if in_scope:
                owner = _active_token()
                callsite = (
                    _dispatch_callsite() if owner is None or owner.func_call_id is None else None
                )
                self.state.events.append(_DispatchEvent(_operator_name(func), owner, callsite))
        finally:
            self.state.callback_ns += time.perf_counter_ns() - started
        return func(*args, **(kwargs or {}))


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
    diagnostics: list[dict[str, Any]] = []
    expected_opaque_count = 0
    accounted_count = 0
    for event_index, event in enumerate(state.events, start=1):
        owner = event.owner
        if owner is not None:
            owner_entry = owner_events.setdefault(id(owner), (owner, []))
            owner_entry[1].append(event.operator)
        if owner is not None and _is_expected_opaque_dispatch(event.operator, owner):
            expected_opaque_count += 1
            continue
        if owner is not None and owner.capture_accounted is True:
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
                "scope": owner_scope,
                "aten_ops": tuple(operators),
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
        trace.capture_verification_reason = "dispatch_witness_unaccounted_ops"
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
    if getattr(trace, "escape_detector_verified", None) is False:
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
    trace.completeness_witness_mode = mode
    if not hasattr(trace, "completeness_witness_verified"):
        trace.completeness_witness_verified = None
    trace.__dict__.setdefault("completeness_diagnostics", [])
    trace.__dict__.setdefault("completeness_decompositions", [])
    if mode == "off":
        yield
        return
    guard_passes = getattr(trace, "capture_guard_passes", [])
    guard_pass_index = len(guard_passes) if guard_passes else 1
    state = _WitnessState(trace, threading.get_ident(), guard_pass_index)
    mode_context = _CompletenessDispatchMode(state)
    with mode_context:
        try:
            yield
        finally:
            _finalize_census(state)

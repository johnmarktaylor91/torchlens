"""Opt-in shadow detection for stale detached torch callable invocations.

The detector is diagnostic only in this rollout. It reports exact callable/code
identity escapes and marks the Trace unverified; it does not enforce capture
completeness without the separate dispatcher witness.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
import traceback
import types
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch

from ... import _state
from ..._errors import TorchLensCaptureGapWarning

EscapeDetectorMode = Literal["off", "shadow"]
"""Supported callable escape-detector rollout modes."""

MAX_AUDITED_EXEMPTIONS = 16
"""Hard budget preventing audited exceptions from becoming a stack blanket."""


@dataclass(frozen=True)
class AuditedEscapeExemption:
    """One exact parent/child/callsite false-positive exemption."""

    parent_wrapper: str
    child_qualname: str
    caller_filename_suffix: str
    caller_name: str
    reason: str


AUDITED_ESCAPE_EXEMPTIONS: tuple[AuditedEscapeExemption, ...] = (
    AuditedEscapeExemption(
        parent_wrapper="torch_func:adaptive_max_pool1d:logged",
        child_qualname="torch.nn.functional._adaptive_max_pool1d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:adaptive_max_pool1d:logged",
        child_qualname="torch.nn.functional.adaptive_max_pool1d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:adaptive_max_pool2d:logged",
        child_qualname="torch.nn.functional._adaptive_max_pool2d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:adaptive_max_pool2d:logged",
        child_qualname="torch.nn.functional.adaptive_max_pool2d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:adaptive_max_pool3d:logged",
        child_qualname="torch.nn.functional._adaptive_max_pool3d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:adaptive_max_pool3d:logged",
        child_qualname="torch.nn.functional.adaptive_max_pool3d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:fractional_max_pool2d:logged",
        child_qualname="torch.nn.functional._fractional_max_pool2d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:fractional_max_pool2d:logged",
        child_qualname="torch.nn.functional.fractional_max_pool2d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:fractional_max_pool3d:logged",
        child_qualname="torch.nn.functional._fractional_max_pool3d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:fractional_max_pool3d:logged",
        child_qualname="torch.nn.functional.fractional_max_pool3d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:max_pool1d:logged",
        child_qualname="torch.nn.functional._max_pool1d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:max_pool1d:logged",
        child_qualname="torch.nn.functional.max_pool1d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:max_pool2d:logged",
        child_qualname="torch.nn.functional._max_pool2d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:max_pool2d:logged",
        child_qualname="torch.nn.functional.max_pool2d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:max_pool3d:logged",
        child_qualname="torch.nn.functional._max_pool3d",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch false branch uses its pre-wrap closure-held pool callable",
    ),
    AuditedEscapeExemption(
        parent_wrapper="torch_func:max_pool3d:logged",
        child_qualname="torch.nn.functional.max_pool3d_with_indices",
        caller_filename_suffix="torch/_jit_internal.py",
        caller_name="fn",
        reason="boolean_dispatch true branch uses its pre-wrap closure-held pool callable",
    ),
)
"""Reviewable exact exemptions; additions require a regression test and reason."""

if len(AUDITED_ESCAPE_EXEMPTIONS) > MAX_AUDITED_EXEMPTIONS:
    raise RuntimeError("TorchLens callable escape exemption budget exceeded.")


@dataclass(frozen=True)
class DetectorTables:
    """Immutable exact-identity detector inventory for one wrapper epoch."""

    epoch: int
    raw_by_id: Mapping[int, Callable[..., Any]]
    python_code_to_raw_ids: Mapping[types.CodeType, tuple[int, ...]]
    c_raw_ids: frozenset[int]
    tensor_method_name_to_raw_ids: Mapping[str, tuple[int, ...]]
    export_sites: Mapping[int, tuple[str, ...]]
    excluded_raw_ids: frozenset[int]


@dataclass
class ExpectedOriginalToken:
    """One-shot authorization for one wrapper-to-original call edge."""

    raw_id: int
    original: Callable[..., Any]
    wrapper_name: str
    wrapper_frame_id: int
    owner_thread_id: int
    pending: bool = True
    func_name: str | None = None
    func_call_id: int | None = None
    call_barcode: object | None = None
    census_scope: Literal["owned", "expected_opaque"] = "owned"
    capture_accounted: bool | None = None
    capture_callsite: tuple[str, int, str] | None = None


@dataclass
class _GuardState:
    """Per-forward detector state installed only in the capture owner thread."""

    trace: Any
    tables: DetectorTables
    owner_thread_id: int
    mode: EscapeDetectorMode
    guard_pass_index: int
    prior_profile: Callable[[types.FrameType, str, Any], Any] | None = None
    monitoring_tool_id: int | None = None
    monitoring_codes: tuple[types.CodeType, ...] = ()
    seen_violations: set[tuple[int, str, int]] = field(default_factory=set)
    event_count: int = 0
    callback_ns: int = 0


_THREAD_STATE = threading.local()
_TABLES: DetectorTables | None = None


def reset_detector_tables() -> None:
    """Drop cached raw identities at a wrapper epoch boundary."""

    global _TABLES
    _TABLES = None


def _safe_qualname(value: Any) -> str:
    """Return a stable callable diagnostic name without foreign-code failures."""

    try:
        module = getattr(value, "__module__", None)
        name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    except Exception:
        return type(value).__name__
    if not isinstance(name, str):
        return type(value).__name__
    return f"{module}.{name}" if isinstance(module, str) else name


def _build_export_sites() -> dict[int, tuple[str, ...]]:
    """Build exact raw-id to currently registered torch export-site names."""

    from ...constants import get_orig_torch_funcs
    from ...utils.introspection import nested_getattr

    sites: dict[int, list[str]] = {}
    for namespace_name, func_name in get_orig_torch_funcs():
        namespace_key = namespace_name.removeprefix("torch.")
        try:
            namespace = torch if namespace_name == "torch" else nested_getattr(torch, namespace_key)
            current = getattr(namespace, func_name)
        except (AttributeError, TypeError):
            continue
        original = _state._decorated_to_orig.get(id(current))
        if original is None:
            continue
        sites.setdefault(id(original), []).append(f"{namespace_name}.{func_name}")
    return {raw_id: tuple(sorted(set(names))) for raw_id, names in sites.items()}


def build_detector_tables() -> DetectorTables:
    """Return immutable exact detector tables for the current wrapper epoch."""

    global _TABLES
    if _TABLES is not None and _TABLES.epoch == _state._detached_patch_epoch:
        return _TABLES
    raw_by_id: dict[int, Callable[..., Any]] = {}
    python_codes: dict[types.CodeType, list[int]] = {}
    c_raw_ids: set[int] = set()
    tensor_methods: dict[str, list[int]] = {}
    excluded: set[int] = set()
    for raw_id, wrapper in _state._orig_to_decorated.items():
        original = _state._decorated_to_orig.get(id(wrapper))
        if not callable(original):
            continue
        raw_by_id[raw_id] = original
        code = getattr(original, "__code__", None)
        if isinstance(code, types.CodeType):
            python_codes.setdefault(code, []).append(raw_id)
        else:
            c_raw_ids.add(raw_id)
        owner = getattr(original, "__objclass__", None)
        name = getattr(original, "__name__", None)
        try:
            is_tensor_descriptor = isinstance(owner, type) and issubclass(torch.Tensor, owner)
        except TypeError:
            is_tensor_descriptor = False
        if is_tensor_descriptor and isinstance(name, str):
            tensor_methods.setdefault(name, []).append(raw_id)
        if bool(getattr(wrapper, "__tl_detector_excluded__", False)):
            excluded.add(raw_id)
    tables = DetectorTables(
        epoch=_state._detached_patch_epoch,
        raw_by_id=types.MappingProxyType(raw_by_id),
        python_code_to_raw_ids=types.MappingProxyType(
            {code: tuple(raw_ids) for code, raw_ids in python_codes.items()}
        ),
        c_raw_ids=frozenset(c_raw_ids),
        tensor_method_name_to_raw_ids=types.MappingProxyType(
            {name: tuple(raw_ids) for name, raw_ids in tensor_methods.items()}
        ),
        export_sites=types.MappingProxyType(_build_export_sites()),
        excluded_raw_ids=frozenset(excluded),
    )
    _TABLES = tables
    return tables


def _token_stack() -> list[ExpectedOriginalToken]:
    """Return the owner-thread expected-edge token stack."""

    stack = getattr(_THREAD_STATE, "tokens", None)
    if not isinstance(stack, list):
        stack = []
        _THREAD_STATE.tokens = stack
    return cast(list[ExpectedOriginalToken], stack)


class _ExpectedOriginalContext(AbstractContextManager[ExpectedOriginalToken]):
    """Concrete context that pushes and removes one prebuilt edge token."""

    def __init__(self, token: ExpectedOriginalToken) -> None:
        """Store the token whose caller frame was captured at construction."""

        self.token = token

    def __enter__(self) -> ExpectedOriginalToken:
        """Push and return the expected-edge token."""

        _token_stack().append(self.token)
        return self.token

    def __exit__(self, exc_type: Any, exc: Any, traceback_obj: Any) -> None:
        """Remove the token without suppressing an exception."""

        del exc_type, exc, traceback_obj
        stack = _token_stack()
        if stack and stack[-1] is self.token:
            stack.pop()
        elif self.token in stack:
            stack.remove(self.token)


def expected_original_call(
    original: Callable[..., Any],
    wrapper_name: str,
    *,
    func_name: str | None = None,
    func_call_id: int | None = None,
    call_barcode: object | None = None,
    census_scope: Literal["owned", "expected_opaque"] = "owned",
) -> AbstractContextManager[ExpectedOriginalToken]:
    """Authorize exactly one immediate wrapper-to-original profiler event.

    Parameters
    ----------
    original:
        Saved original callable invoked by the wrapper.
    wrapper_name:
        Stable wrapper/call-edge label used in diagnostics and exemptions.
    func_name:
        TorchLens function name for dispatch ownership diagnostics.
    func_call_id:
        Monotone captured-call identity shared with the resulting ``Op``.
    call_barcode:
        Leaf-detection barcode written to the active Trace before the call.
    census_scope:
        Whether dispatcher work belongs to a prospective captured op or an
        exact documented opaque boundary.

    Returns
    -------
    AbstractContextManager[ExpectedOriginalToken]
        Context carrying a token bound to the calling wrapper frame.
    """

    caller = inspect.currentframe()
    caller = caller.f_back if caller is not None else None
    return _ExpectedOriginalContext(
        ExpectedOriginalToken(
            raw_id=id(original),
            original=original,
            wrapper_name=wrapper_name,
            wrapper_frame_id=id(caller) if caller is not None else 0,
            owner_thread_id=threading.get_ident(),
            func_name=func_name,
            func_call_id=func_call_id,
            call_barcode=call_barcode,
            census_scope=census_scope,
        )
    )


def mark_expected_original_accounted(
    token: ExpectedOriginalToken | None,
    *,
    captured: bool,
) -> None:
    """Record whether a token's dispatch interval became a captured leaf.

    Parameters
    ----------
    token:
        Token returned by :func:`expected_original_call`, if diagnostics were armed.
    captured:
        Whether the wrapper passed the shared barcode leaf test and emitted tensor ops.
    """

    if token is not None:
        token.capture_accounted = captured


def _active_token() -> ExpectedOriginalToken | None:
    """Return the innermost live token owned by the current thread."""

    stack = _token_stack()
    return stack[-1] if stack else None


def _bound_tensor_method_candidates(target: Any, tables: DetectorTables) -> tuple[int, ...]:
    """Resolve transient bound Tensor builtins by owner type and method name.

    This is the descriptor identity-compatibility rule: exact object identity is
    unavailable for transient bound builtins, so compatibility requires
    ``target.__self__`` to be a Tensor and its exact method name to appear in the
    wrapped Tensor-method inventory. The same rule convicts such calls outside a
    matching one-shot token.
    """

    try:
        bound_self = target.__self__
        name = target.__name__
    except Exception:
        return ()
    if not isinstance(bound_self, torch.Tensor) or not isinstance(name, str):
        return ()
    return tables.tensor_method_name_to_raw_ids.get(name, ())


def _candidate_raw_ids(
    frame: types.FrameType,
    event: str,
    arg: Any,
    tables: DetectorTables,
) -> tuple[int, ...]:
    """Return exact/code/descriptor-compatible raw candidates for one event."""

    if event == "call":
        return tables.python_code_to_raw_ids.get(frame.f_code, ())
    if event != "c_call":
        return ()
    raw_id = id(arg)
    if raw_id in tables.c_raw_ids:
        return (raw_id,)
    return _bound_tensor_method_candidates(arg, tables)


def _event_immediate_caller_id(frame: types.FrameType, event: str) -> int:
    """Return the frame identity that directly issued the observed call."""

    if event == "call":
        return id(frame.f_back) if frame.f_back is not None else 0
    return id(frame)


def _consume_expected_token(
    frame: types.FrameType,
    event: str,
    candidate_ids: tuple[int, ...],
) -> ExpectedOriginalToken | None:
    """Consume a token only for its exact immediate compatible call edge."""

    token = _active_token()
    if token is None or not token.pending:
        return None
    if token.owner_thread_id != threading.get_ident():
        return None
    if token.raw_id not in candidate_ids:
        return None
    if _event_immediate_caller_id(frame, event) != token.wrapper_frame_id:
        return None
    token.pending = False
    return token


def _diagnostic_callsite(frame: types.FrameType, event: str) -> types.FrameType:
    """Return the user-side caller frame for one profile event."""

    if event == "call" and frame.f_back is not None:
        return frame.f_back
    return frame


def _is_audited_exemption(
    token: ExpectedOriginalToken | None,
    raw_id: int,
    callsite: types.FrameType,
    tables: DetectorTables,
) -> bool:
    """Return whether an event matches one exact budgeted exemption row."""

    if token is None:
        return False
    original = tables.raw_by_id.get(raw_id)
    child_name = _safe_qualname(original)
    for exemption in AUDITED_ESCAPE_EXEMPTIONS:
        if (
            exemption.parent_wrapper == token.wrapper_name
            and exemption.child_qualname == child_name
            and callsite.f_code.co_filename.endswith(exemption.caller_filename_suffix)
            and callsite.f_code.co_name == exemption.caller_name
        ):
            return True
    return False


def _storage_hint(callsite: types.FrameType) -> str:
    """Infer a conservative storage-channel hint from the callsite frame."""

    if callsite.f_code.co_freevars:
        return "closure_or_partial"
    return "unknown_or_container"


def _report_escape(
    guard: _GuardState,
    raw_ids: tuple[int, ...],
    callsite: types.FrameType,
) -> None:
    """Attach and warn once for a shadow-mode callable escape candidate."""

    report_ids = tuple(raw_id for raw_id in raw_ids if raw_id not in guard.tables.excluded_raw_ids)
    if not report_ids:
        return
    primary = report_ids[0]
    key = (primary, callsite.f_code.co_filename, callsite.f_lineno)
    if key in guard.seen_violations:
        return
    guard.seen_violations.add(key)
    original = guard.tables.raw_by_id.get(primary)
    names = tuple(_safe_qualname(guard.tables.raw_by_id.get(raw_id)) for raw_id in report_ids)
    export_sites = tuple(
        site for raw_id in report_ids for site in guard.tables.export_sites.get(raw_id, ())
    )
    detail = {
        "violation_id": len(guard.seen_violations),
        "policy": _state._detached_patch_policy,
        "detector_backend": "sys.monitoring"
        if guard.monitoring_tool_id is not None
        else "setprofile",
        "callable_candidates": names,
        "export_sites": tuple(sorted(set(export_sites))),
        "file": callsite.f_code.co_filename,
        "line": callsite.f_lineno,
        "function": callsite.f_code.co_name,
        "storage_hint": _storage_hint(callsite),
        "owner_thread_id": guard.owner_thread_id,
        "guard_pass_index": guard.guard_pass_index,
        "capture_mode": getattr(guard.trace, "capture_mode", None),
        "exhaustive_pass": bool(getattr(guard.trace, "_in_exhaustive_pass", False)),
        "stack": tuple(traceback.format_stack(callsite, limit=5)),
        "witness_corroborated": False,
        "enforced": False,
    }
    reports = guard.trace.__dict__.setdefault("escape_diagnostics", [])
    reports.append(detail)
    guard.trace.escape_detector_verified = False
    guard.trace.capture_verified = False
    guard.trace.capture_verification_reason = "callable_escape_shadow_report"
    callable_name = _safe_qualname(original)
    warnings.warn(
        "TorchLens shadow detector observed a raw torch callable outside its registered "
        f"wrapper edge: {callable_name} at {detail['file']}:{detail['line']} in "
        f"{detail['function']} (storage hint: {detail['storage_hint']}). The Trace is marked "
        "capture_verified=False. Rebind after tl.wrap_torch(), use a live torch namespace "
        "lookup, add the owning module via patch_modules, or compare with patch_policy='full'.",
        TorchLensCaptureGapWarning,
        stacklevel=2,
    )


def _handle_event(guard: _GuardState, frame: types.FrameType, event: str, arg: Any) -> None:
    """Classify one profile/monitoring event against exact detector tables."""

    if threading.get_ident() != guard.owner_thread_id:
        return
    if not _state._logging_enabled or _state._active_trace is not guard.trace:
        return
    candidate_ids = _candidate_raw_ids(frame, event, arg, guard.tables)
    if not candidate_ids:
        return
    guard.event_count += 1
    consumed = _consume_expected_token(frame, event, candidate_ids)
    if consumed is not None:
        return
    callsite = _diagnostic_callsite(frame, event)
    token = _active_token()
    if any(
        _is_audited_exemption(token, raw_id, callsite, guard.tables) for raw_id in candidate_ids
    ):
        return
    _report_escape(guard, candidate_ids, callsite)


def _profile_callback(frame: types.FrameType, event: str, arg: Any) -> None:
    """Dispatch CPython profile events and deliberately chain an existing hook."""

    guard = getattr(_THREAD_STATE, "guard", None)
    if not isinstance(guard, _GuardState):
        return
    started = time.perf_counter_ns()
    try:
        if event in {"call", "c_call"}:
            _handle_event(guard, frame, event, arg)
        if guard.prior_profile is not None:
            guard.prior_profile(frame, event, arg)
    finally:
        guard.callback_ns += time.perf_counter_ns() - started


def _install_setprofile(guard: _GuardState) -> None:
    """Install a compositional owner-thread profile callback."""

    prior = sys.getprofile()
    if prior is _profile_callback:
        raise RuntimeError("TorchLens escape detector cannot nest its setprofile adapter.")
    guard.prior_profile = prior
    sys.setprofile(_profile_callback)


def _uninstall_setprofile(guard: _GuardState) -> None:
    """Restore the exact profile hook that preceded detector installation."""

    sys.setprofile(guard.prior_profile)


def _find_monitoring_tool_id(monitoring: Any) -> int:
    """Reserve one free sys.monitoring tool id or fail loudly."""

    candidates = range(5, -1, -1)
    for tool_id in candidates:
        try:
            if monitoring.get_tool(tool_id) is None:
                monitoring.use_tool_id(tool_id, "torchlens.escape_detection")
                return tool_id
        except (AttributeError, ValueError):
            continue
    raise RuntimeError("No free sys.monitoring tool id is available for TorchLens diagnostics.")


def _install_monitoring(guard: _GuardState) -> None:
    """Install local Python-start and diagnostic global-CALL monitoring."""

    monitoring = cast(Any, getattr(sys, "monitoring"))
    tool_id = _find_monitoring_tool_id(monitoring)
    guard.monitoring_tool_id = tool_id
    codes = tuple(guard.tables.python_code_to_raw_ids)
    guard.monitoring_codes = codes

    def on_python_start(code: types.CodeType, instruction_offset: int) -> None:
        """Classify one locally monitored raw Python-function entry."""

        del instruction_offset
        frame = sys._getframe(1)
        _handle_event(guard, frame, "call", None)

    def on_call(
        code: types.CodeType,
        instruction_offset: int,
        callable_obj: Any,
        arg0: Any,
    ) -> None:
        """Classify one diagnostic CALL event for exact C/descriptor targets."""

        del code, instruction_offset, arg0
        frame = sys._getframe(1)
        _handle_event(guard, frame, "c_call", callable_obj)

    monitoring.register_callback(tool_id, monitoring.events.PY_START, on_python_start)
    monitoring.register_callback(tool_id, monitoring.events.CALL, on_call)
    for code in codes:
        monitoring.set_local_events(tool_id, code, monitoring.events.PY_START)
    monitoring.set_events(tool_id, monitoring.events.CALL)


def _uninstall_monitoring(guard: _GuardState) -> None:
    """Restore sys.monitoring state by releasing TorchLens's private tool id."""

    monitoring = cast(Any, getattr(sys, "monitoring"))
    tool_id = guard.monitoring_tool_id
    if tool_id is None:
        return
    monitoring.set_events(tool_id, 0)
    for code in guard.monitoring_codes:
        monitoring.set_local_events(tool_id, code, 0)
    monitoring.register_callback(tool_id, monitoring.events.PY_START, None)
    monitoring.register_callback(tool_id, monitoring.events.CALL, None)
    monitoring.free_tool_id(tool_id)
    guard.monitoring_tool_id = None


def _install_detector(guard: _GuardState) -> None:
    """Install the best exact callable detector available on this Python."""

    monitoring = getattr(sys, "monitoring", None)
    if monitoring is not None:
        _install_monitoring(guard)
    else:
        _install_setprofile(guard)


def _uninstall_detector(guard: _GuardState) -> None:
    """Uninstall the active detector adapter with exception-safe restoration."""

    if guard.monitoring_tool_id is not None:
        _uninstall_monitoring(guard)
    else:
        _uninstall_setprofile(guard)


def _effective_mode() -> EscapeDetectorMode:
    """Return the validated process-level escape detector mode."""

    mode = _state._escape_detector_mode
    if mode not in {"off", "shadow"}:
        raise RuntimeError(f"Invalid TorchLens escape detector mode {mode!r}.")
    return cast(EscapeDetectorMode, mode)


def _copy_recording_diagnostics(trace: Any) -> None:
    """Mirror guard diagnostics onto a fastlog Recording when one exists."""

    recording = getattr(trace, "_fastlog_recording", None)
    if recording is None:
        return
    for field_name in (
        "detached_patch_policy",
        "detached_patch_epoch",
        "escape_detector_mode",
        "escape_detector_verified",
        "completeness_witness_mode",
        "completeness_witness_verified",
        "capture_owner_thread_id",
        "capture_owner_thread_qualified",
        "capture_thread_count_start",
        "capture_thread_count_end",
        "capture_thread_activity_detected",
        "capture_verified",
        "capture_verification_reason",
        "escape_detector_backward_coverage",
        "escape_diagnostics",
        "escape_detector_event_count",
        "escape_detector_callback_ns",
        "completeness_diagnostics",
        "completeness_decompositions",
        "completeness_witness_event_count",
        "completeness_witness_accounted_count",
        "completeness_witness_expected_opaque_count",
        "completeness_witness_unaccounted_count",
        "completeness_witness_callback_ns",
        "capture_guard_passes",
    ):
        if hasattr(trace, field_name):
            object.__setattr__(recording, field_name, getattr(trace, field_name))


@contextmanager
def capture_escape_guard(trace: Any) -> Iterator[None]:
    """Record owner-thread qualification and optionally arm shadow detection.

    Parameters
    ----------
    trace:
        Trace or Recording runtime trace receiving diagnostics.

    Yields
    ------
    None
        The backend enters its existing active-logging context inside the guard.
    """

    mode = _effective_mode()
    owner_thread_id = threading.get_ident()
    thread_count_start = threading.active_count()
    tables = build_detector_tables()
    guard_passes = trace.__dict__.setdefault("capture_guard_passes", [])
    guard_pass_index = len(guard_passes) + 1
    guard_passes.append(
        {
            "guard_pass_index": guard_pass_index,
            "capture_mode": getattr(trace, "capture_mode", None),
            "owner_thread_id": owner_thread_id,
        }
    )
    guard = _GuardState(trace, tables, owner_thread_id, mode, guard_pass_index)
    trace.detached_patch_policy = _state._detached_patch_policy
    trace.detached_patch_epoch = _state._detached_patch_epoch
    trace.escape_detector_mode = mode
    if not hasattr(trace, "escape_detector_verified"):
        trace.escape_detector_verified = True if mode == "shadow" else None
    trace.capture_owner_thread_id = owner_thread_id
    trace.capture_owner_thread_qualified = True
    trace.capture_thread_count_start = thread_count_start
    trace.capture_thread_activity_detected = False
    trace.escape_detector_backward_coverage = "not_armed"
    trace.__dict__.setdefault("escape_diagnostics", [])
    if _state._detached_patch_policy == "scoped":
        trace.capture_verified = False
        trace.capture_verification_reason = "scoped_dispatch_witness_not_enabled"
    else:
        trace.capture_verified = False if mode == "shadow" else None
        trace.capture_verification_reason = "shadow_diagnostic_mode" if mode == "shadow" else None
    _THREAD_STATE.guard = guard
    try:
        if mode == "shadow":
            _install_detector(guard)
        yield
    finally:
        if mode == "shadow":
            _uninstall_detector(guard)
        _THREAD_STATE.guard = None
        _THREAD_STATE.tokens = []
        thread_count_end = threading.active_count()
        trace.capture_thread_count_end = thread_count_end
        trace.capture_thread_activity_detected = thread_count_end != thread_count_start
        trace.escape_detector_event_count = (
            int(getattr(trace, "escape_detector_event_count", 0)) + guard.event_count
        )
        trace.escape_detector_callback_ns = (
            int(getattr(trace, "escape_detector_callback_ns", 0)) + guard.callback_ns
        )
        if trace.capture_thread_activity_detected:
            if mode == "shadow":
                trace.escape_detector_verified = False
            if getattr(trace, "completeness_witness_verified", None) is True:
                trace.completeness_witness_verified = False
            trace.capture_verified = False
            trace.capture_verification_reason = "owner_thread_tripwire_changed"
            warnings.warn(
                "TorchLens observed a thread-count change during capture. The supported proof "
                "domain is the capture-owner thread; move model tensor work into that thread "
                "or treat this Trace as unverified.",
                TorchLensCaptureGapWarning,
                stacklevel=2,
            )
        _copy_recording_diagnostics(trace)

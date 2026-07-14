"""Opt-in aten-dispatch completeness witness for torch capture.

The witness correlates dispatcher events to the exact wrapper edge token and
leaf barcode used by callable escape detection and ordinary TorchLens capture.
It deliberately does not use clocks or inferred time windows.

Observational contract (cooperative-model assumption). The witness observes
operations through the torch dispatcher, exactly like every dispatch-based
tracer. It therefore assumes a cooperative model: one that does not deliberately
hide operations from the dispatcher. An uncaptured op that the witness CAN see
is reported (an uncaptured *mutating* dispatch fails completeness). But a model
that intercepts an op inside a tensor-subclass ``__torch_dispatch__`` and runs a
hidden mutation under ``torch._C._DisableTorchDispatch()`` suppresses dispatcher
re-entry, so the nested op is genuinely invisible to the witness and cannot be
detected. This is an adversarial construction, not a capture bug -- a normal
model validating its own forward pass cannot trigger it. See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import sys
import threading
import time
import warnings
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ... import _state
from ..._errors import TorchLensCaptureGapWarning
from ._tl import get_tensor_label
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
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__float__:logged",
        operator="aten._local_scalar_dense.default",
        reason="float(tensor) extracts a Python float, which is intentionally not an op output",
    ),
    AuditedCompletenessBoundary(
        wrapper_name="torch_func:__int__:logged",
        operator="aten._local_scalar_dense.default",
        reason="int(tensor) extracts a Python int, which is intentionally not an op output",
    ),
)
"""Reviewable exact expected-opaque rows; additions require a regression test and reason."""

if len(AUDITED_COMPLETENESS_BOUNDARIES) > MAX_AUDITED_COMPLETENESS_BOUNDARIES:
    raise RuntimeError("TorchLens completeness boundary budget exceeded.")

_EXPECTED_OPAQUE_WRAPPERS = frozenset(
    row.wrapper_name for row in AUDITED_COMPLETENESS_BOUNDARIES if row.operator is None
)

_REPLACEMENT_HOOK_FILE = Path(__file__).resolve().parent / "model_prep.py"
"""File owning the ``wrapped_hook`` frame that brackets raw replacement hooks."""

_REPLACEMENT_HOOK_FUNC = "wrapped_hook"
"""Torchlens-owned frame name that wraps a raw ``register_forward_hook`` call."""


def _in_replacement_hook_frame() -> bool:
    """Return whether a genuine raw replacement hook is executing above the dispatch.

    A genuine output-replacement ``register_forward_hook`` runs the user hook inside
    TorchLens's own ``wrapped_hook`` frame (``model_prep._instrumented_forward_hook``).
    Every aten dispatch emitted while that torchlens-owned frame is live is genuine
    replacement construction -- either a raw-aten call (unowned) or a python-wrapped
    call whose op is orphaned out of the final trace because its only consumer is the
    untraceable replacement tensor. This exact, per-event signal lets the completeness
    census excuse ONLY the untraceable dispatch attributable to a real replacement
    while STILL failing on any unrelated silent drop, which fires OUTSIDE a
    replacement hook.

    Returns
    -------
    bool
        ``True`` only when a torchlens replacement-hook frame is live on the stack.
    """

    frame: Any = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if code.co_name == _REPLACEMENT_HOOK_FUNC:
            try:
                if Path(code.co_filename).resolve() == _REPLACEMENT_HOOK_FILE:
                    return True
            except (OSError, RuntimeError, ValueError):
                pass
        frame = frame.f_back
    return False


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
    in_replacement_hook: bool = False
    mutates: bool = False


@dataclass
class _WitnessState:
    """Per-forward dispatch census state."""

    trace: Any
    owner_thread_id: int
    guard_pass_index: int
    events: list[_DispatchEvent] = field(default_factory=list)
    callback_ns: int = 0
    census: bool = True
    record_escapes: bool = False


HOST_ESCAPE_OPERATORS = frozenset({"aten._local_scalar_dense.default"})
"""Aten operators that hand a captured tensor's value to the Python host.

``aten._local_scalar_dense.default`` is the single dispatcher footprint of every
tensor->Python scalar escape TorchLens can see: ``.item()``, ``int()``,
``float()``, ``__index__``, and ``bool()`` all lower to it. Recording the SOURCE
tensor of this escape lets the runnable descriptor witness the escape by its
producing op (keyed on the ESCAPE EVENT), never by correlating a baked literal by
value. Multi-element ``.tolist()`` / ``.numpy()`` conversions do NOT emit this op
(they are dispatcher-invisible) and are handled by the descriptor's complementary
value-equality net.
"""

_HOST_ESCAPE_SOURCE_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = weakref.WeakKeyDictionary()
"""Per-trace raw producing-op labels of tensor->host escape sources.

Kept off the Trace ``__dict__`` (and therefore out of portable-state scrub) in a
weak-keyed side table so the runnable descriptor can read escape sources without
registering a new serialized Trace field. Entries are dropped automatically when a
Trace is garbage collected.
"""


def host_escape_source_labels(trace: Any) -> frozenset[str]:
    """Return the recorded raw escape-source op labels for one trace."""

    labels = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


_PRUNED_RNG_CONTROL_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = weakref.WeakKeyDictionary()
"""Per-trace raw labels of pruned torch-RNG ops that DROVE control flow.

A torch-RNG op (``torch.rand``/``randn``/... -- any ATen ``nondeterministic_seeded``
overload) whose result steered pure-Python control flow (``if torch.rand(()) > 0.5``)
is INPUT-DISCONNECTED: the ``rand -> gt`` predicate chain reaches neither an input nor
an output, so orphan removal drops it entirely and the runnable descriptor never sees
it. The recorded taken branch is then nondeterministic (a fresh seeded forward may take
the other arm) yet unwitnessed. Kept in a weak-keyed side table (like the escape-source
labels above, and out of the Trace field schema) so the runnable producer can downgrade
witness completeness to keep such a model honestly UNVERIFIABLE + NOT_APPLICABLE instead
of falsely VERIFIED + ATTESTED. A genuinely-dead RNG draw (result influences nothing) is
NOT recorded here, so a deterministic model stays VERIFIED.
"""


def pruned_rng_control_source_labels(trace: Any) -> frozenset[str]:
    """Return raw labels of pruned torch-RNG ops that steered control flow."""

    labels = _PRUNED_RNG_CONTROL_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def record_pruned_rng_control_source(trace: Any, label: str) -> None:
    """Record one pruned torch-RNG op whose result drove a control decision."""

    labels = _PRUNED_RNG_CONTROL_LABELS.get(trace)
    if labels is None:
        labels = set()
        _PRUNED_RNG_CONTROL_LABELS[trace] = labels
    labels.add(label)


def _record_host_escape_source(trace: Any, func: Any, args: tuple[Any, ...]) -> None:
    """Record the raw producing-op label of one tensor->host escape source.

    When a captured tensor's value escapes to the Python host through
    ``aten._local_scalar_dense`` (``.item()`` / ``int()`` / ``float()`` /
    ``__index__`` / ``bool()``), the escaped value can be baked into a downstream
    op literal (verbatim OR after arbitrary Python arithmetic) or steer pure-Python
    control flow -- neither of which the sparse DAG can recompute. The source
    tensor is the op whose output was read; its raw capture label is recorded so
    the runnable descriptor can witness that source slot and, at run time, refuse a
    false VERIFIED when the slot recomputes different bytes for a changed input.
    """

    if _operator_name(func) not in HOST_ESCAPE_OPERATORS or not args:
        return
    source = args[0]
    if not isinstance(source, torch.Tensor):
        return
    label = get_tensor_label(source)
    if isinstance(label, str):
        sources = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
        if sources is None:
            sources = set()
            _HOST_ESCAPE_SOURCE_LABELS[trace] = sources
        sources.add(label)


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


def _is_mutating_operator(func: Any) -> bool:
    """Return whether a dispatcher operator writes to any of its arguments.

    Mutation is read from the operator's own ``FunctionSchema`` (torch ground
    truth), never a name-string heuristic: ``schema.is_mutable`` covers every
    in-place operator (trailing-underscore names such as ``mul_``/``copy_``) as
    well as ``out=`` overloads whose name does NOT end in an underscore. Per-arg
    ``alias_info.is_write`` is used as a robust fallback when the schema flag is
    unavailable. Pure reads such as ``aten.equal`` / ``aten.allclose`` return
    ``False``, which is exactly why benign ``owner_not_captured`` control-flow
    comparisons are never mistaken for value-affecting drops.

    Parameters
    ----------
    func:
        Dispatcher callable received by ``__torch_dispatch__``.

    Returns
    -------
    bool
        ``True`` when the operator mutates (writes) at least one argument.
    """

    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    is_mutable = getattr(schema, "is_mutable", None)
    if isinstance(is_mutable, bool):
        return is_mutable
    arguments = getattr(schema, "arguments", ()) or ()
    for argument in arguments:
        alias_info = getattr(argument, "alias_info", None)
        if alias_info is not None and getattr(alias_info, "is_write", False):
            return True
    return False


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
                if self.state.record_escapes:
                    _record_host_escape_source(self.state.trace, func, args)
                if self.state.census:
                    owner = _active_token()
                    callsite = (
                        _dispatch_callsite()
                        if owner is None or owner.func_call_id is None
                        else None
                    )
                    in_replacement_hook = _in_replacement_hook_frame()
                    mutates = _is_mutating_operator(func)
                    self.state.events.append(
                        _DispatchEvent(
                            _operator_name(func),
                            owner,
                            callsite,
                            in_replacement_hook,
                            mutates,
                        )
                    )
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
                "in_replacement_hook": event.in_replacement_hook,
                "mutates": event.mutates,
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
    for counter_field in (
        "completeness_witness_event_count",
        "completeness_witness_accounted_count",
        "completeness_witness_expected_opaque_count",
        "completeness_witness_unaccounted_count",
        "completeness_witness_callback_ns",
    ):
        trace.__dict__.setdefault(counter_field, 0)
    # A runnable-eligible (``intervention_ready``) capture always records
    # tensor->host escape sources so the sparse descriptor can witness the escape
    # by its producing op, keyed on the ESCAPE EVENT. This is a passive observer:
    # it records raw op labels only and never alters a captured op, so goldens are
    # unchanged. The default (non-runnable) capture path installs nothing.
    record_escapes = bool(getattr(trace, "intervention_ready", False))
    if mode == "off" and not record_escapes:
        yield
        return
    guard_passes = getattr(trace, "capture_guard_passes", [])
    guard_pass_index = len(guard_passes) if guard_passes else 1
    state = _WitnessState(
        trace,
        threading.get_ident(),
        guard_pass_index,
        census=(mode == "shadow"),
        record_escapes=record_escapes,
    )
    mode_context = _CompletenessDispatchMode(state)
    with mode_context:
        try:
            yield
        finally:
            if mode == "shadow":
                _finalize_census(state)

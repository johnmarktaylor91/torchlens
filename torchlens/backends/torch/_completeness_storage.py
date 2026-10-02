"""Storage observation and scalar-escape helpers."""

from __future__ import annotations

import functools
import inspect
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ...errors import ScalarEscapeWarning
from ._tl import (
    DescriptorCompatProperty,
    get_tensor_label,
)

if TYPE_CHECKING:
    from .completeness_witness import (
        _HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS,
        _HOST_ESCAPE_BOOL_SOURCE_LABELS,
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED,
        _HOST_ESCAPE_MUTABLE_WRITEBACK,
        _HOST_ESCAPE_RAW_POINTER,
        _HOST_ESCAPE_STATE_SOURCE_NAMES,
        _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE,
        _INPUT_METADATA_VIEW_READ,
        _ORIG_UNTYPED_STORAGE_DATA_PTR,
        _STORAGE_ACCESSOR_FAIL_CLOSED,
        _STORAGE_ACCESSOR_MUTATOR,
        _STORAGE_ACCESSOR_NBYTES,
        _STORAGE_ACCESSOR_ORIGIN_BRIDGE,
        _STORAGE_ACCESSOR_PLACEMENT,
        _STORAGE_ACCESSOR_RAW_POINTER,
        _STORAGE_ACCESSOR_VALUE_READ,
        _TORCHLENS_ROOT,
        HOST_VALUE_ESCAPE_METHODS,
        STATE_METADATA_MIRROR,
        _CompletenessDispatchMode,
        _expand_state_alias_addresses,
        _internal_read_active,
        _nonowner_escape_observe,
        _nonowner_ptr_is_captured,
        _observe_state_metadata_read,
        _PlainScalarEscapeState,
        _record_escape_source_tensor,
        _record_input_metadata_read_at_site,
        _record_input_storage_nbytes,
        _record_state_metadata_observation,
        _record_state_metadata_read,
        _register_storage_handle_origin,
        _resolve_storage_origin,
        _WitnessState,
        internal_scalar_read,
    )

__all__ = (
    "_nonowner_storage_observe",
    "_record_state_value_escape",
    "_attribute_storage_placement",
    "_make_storage_metadata_wrapper",
    "_make_storage_property_wrapper",
    "_make_storage_raw_pointer_wrapper",
    "_completeness_census_active",
    "_make_host_value_escape_method",
    "_first_scalar_escape_source",
    "_record_bool_consumer_location",
    "_make_plain_scalar_escape_method",
    "_external_warning_stacklevel",
    "capture_scalar_escape_warning",
    "_make_host_value_predicate_module_wrapper",
    "_make_module_escape_wrapper",
    "_make_invisible_escape_property",
)


def _nonowner_storage_observe(state: _WitnessState, storage: Any) -> None:
    """Ceiling the capture when a NON-owner thread touches a CAPTURED storage handle (r67)."""

    try:
        backing = (
            storage
            if isinstance(storage, torch.UntypedStorage)
            else getattr(storage, "_untyped_storage", None)
        )
        ptr = _ORIG_UNTYPED_STORAGE_DATA_PTR(backing) if backing is not None else None
    except (RuntimeError, TypeError, NotImplementedError, AttributeError):
        ptr = None
    if isinstance(ptr, int) and ptr and _nonowner_ptr_is_captured(state, ptr):
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)


def _record_state_value_escape(trace: Any, addresses: set[str]) -> None:
    """Join a storage-spelling VALUE read into the state digest witness (r67 C3)."""

    addresses = _expand_state_alias_addresses(trace, addresses)
    state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
    if state_names is None:
        state_names = set()
        _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
    state_names |= addresses


def _attribute_storage_placement(
    state: _WitnessState,
    storage: Any,
    name: str,
    observed: bool | None,
    had_args: bool,
) -> None:
    """Attribute one storage-handle placement accessor call, tensor-spelling-identically."""

    origin = _resolve_storage_origin(state, storage)
    if origin is None:
        _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(state.trace)
        return
    origin_kind, payload = origin
    if origin_kind == "state":
        read_kind = STATE_METADATA_MIRROR[name][1]
        _record_state_metadata_observation(state.trace, set(payload), read_kind, observed)
    elif origin_kind == "input":
        if observed is not None and not had_args:
            _record_input_metadata_read_at_site(state.trace, payload, name, observed)
        else:
            # An arg-directed or raising placement query off an input is not re-checkable
            # against the raw runtime leaf: fail closed (same downgrade as a derived-view
            # metadata read).
            _INPUT_METADATA_VIEW_READ.add(state.trace)


def _make_storage_metadata_wrapper(
    original: Any, state: _WitnessState, name: str, disposition: str
) -> Any:
    """Wrap one storage-class accessor: call through ONCE, then record the real result."""

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Record one storage accessor observation before returning the real result.

        Parameters
        ----------
        self:
            Storage-like receiver.
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if _state._active_trace is not state.trace:
            return original(self, *args, **kwargs)
        if threading.get_ident() != state.owner_thread_id:
            if state.belt_armed:
                _nonowner_storage_observe(state, self)
            return original(self, *args, **kwargs)
        if not _state._logging_enabled or _internal_read_active():
            return original(self, *args, **kwargs)
        # The original runs PAUSED + marked: several storage accessors delegate through
        # tensor machinery (``UntypedStorage.is_pinned`` builds a scratch tensor and calls
        # ``Tensor.is_pinned(device)``), and that torch-internal delegation must neither be
        # captured as phantom graph ops nor double-recorded by the tensor-spelling
        # wrappers -- THIS wrapper records the single user-visible observation.
        if disposition == _STORAGE_ACCESSOR_PLACEMENT:
            had_args = bool(args or kwargs)
            try:
                with _state.pause_logging(), internal_scalar_read():
                    result = original(self, *args, **kwargs)
            except Exception:
                # The exception itself is control-flow signal; observed=None refuses.
                _attribute_storage_placement(state, self, name, None, had_args)
                raise
            observed = None if had_args else bool(result)
            _attribute_storage_placement(state, self, name, observed, had_args)
            return result
        with _state.pause_logging(), internal_scalar_read():
            result = original(self, *args, **kwargs)
        origin = _resolve_storage_origin(state, self)
        if origin is None:
            # Unattributable owner-thread storage access: fail closed (observer
            # uncertainty), never silently record nothing.
            _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(state.trace)
            return result
        origin_kind, payload = origin
        if disposition == _STORAGE_ACCESSOR_ORIGIN_BRIDGE:
            _register_storage_handle_origin(state, result, origin)
            return result
        if origin_kind == "other":
            return result
        if disposition == _STORAGE_ACCESSOR_NBYTES:
            if origin_kind == "input":
                _record_input_storage_nbytes(state.trace, payload, self)
            else:
                _record_state_metadata_read(
                    state.trace, set(payload), STATE_METADATA_MIRROR["storage_nbytes"][1]
                )
        elif disposition == _STORAGE_ACCESSOR_MUTATOR:
            _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
        elif disposition == _STORAGE_ACCESSOR_VALUE_READ:
            if name == "type" and not args and not kwargs:
                return result  # no-arg type(): a class-name string; dtype/device slot-pinned
            if origin_kind == "state":
                _record_state_value_escape(state.trace, set(payload))
            else:
                _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(state.trace)
        elif disposition == _STORAGE_ACCESSOR_FAIL_CLOSED:
            _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(state.trace)
        return result

    return wrapper


def _make_storage_property_wrapper(
    descriptor: Any, state: _WitnessState, name: str, disposition: str
) -> property:
    """Wrap a storage-class PROPERTY row (``filename`` / ``_cdata``) read-through."""

    def getter(self: Any) -> Any:
        """Read one storage property and attribute the host exposure when needed.

        Parameters
        ----------
        self:
            Storage-like receiver.

        Returns
        -------
        Any
            Value returned by ``descriptor``.
        """
        value = descriptor.__get__(self, type(self))
        if _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled and not _internal_read_active():
                    if disposition == _STORAGE_ACCESSOR_RAW_POINTER:
                        _HOST_ESCAPE_RAW_POINTER.add(state.trace)
                    else:
                        origin = _resolve_storage_origin(state, self)
                        if origin is None or origin[0] in ("input", "state"):
                            _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(state.trace)
            elif state.belt_armed:
                _nonowner_storage_observe(state, self)
        return value

    # doc= MUST be explicit (never defaulted): see DescriptorCompatProperty's
    # constructor-landmine note in _tl.py.
    replacement = DescriptorCompatProperty(getter, doc=name)
    objclass = getattr(descriptor, "__objclass__", None)
    if objclass is not None:
        replacement.__objclass__ = objclass
    replacement.__name__ = name
    return replacement


def _make_storage_raw_pointer_wrapper(original: Any, state: _WitnessState) -> Any:
    """Wrap ``UntypedStorage.data_ptr`` / ``TypedStorage.data_ptr`` to fail closed, then read through.

    ``tensor.data_ptr()`` is fail-closed by :func:`_make_invisible_escape_wrapper` (r15-H1), but the
    SAME raw pointer is reachable off the Storage HANDLE: ``tensor.untyped_storage().data_ptr()`` /
    ``tensor.storage().data_ptr()`` call ``data_ptr`` on the ``UntypedStorage`` / ``TypedStorage``
    object, NOT on ``torch.Tensor`` -- so the Tensor patch never fires and the raw pointer escapes
    unobserved (r16-C1). A ``ctypes`` READ through it bakes a stale literal and a WRITE through it
    mutates the source with no dispatch, no version bump, and no byte the forward-end watch can
    re-inspect. A genuine USER storage ``data_ptr()`` therefore fails closed to UNVERIFIABLE, exactly
    like the Tensor path. Scoped to the ``data_ptr()`` ACCESSOR only: read-only
    ``untyped_storage().nbytes()`` / ``.size()`` (pure metadata, no pointer) never trips it.
    TorchLens's own capture-internal storage-pointer reads (``aliasing._tensors_alias`` ->
    ``untyped_storage().data_ptr()``) run under the ``internal_scalar_read`` marker and are excluded.
    r43: the OWNER thread fails closed (blanket raw-pointer flag) exactly as before; a NON-OWNER
    thread ceilings ONLY when the storage belongs to a CAPTURED tensor (its raw pointer, read via
    the original accessor, is a captured input/param/activation pointer), so a foreign library's
    own ``data_ptr()`` reads never ceiling the capture.
    """

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Read one storage raw pointer while enforcing the witness downgrade.

        Parameters
        ----------
        self:
            Storage-like receiver.
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Raw pointer result from ``original``.
        """
        if _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled and not _internal_read_active():
                    _HOST_ESCAPE_RAW_POINTER.add(state.trace)
            elif state.belt_armed:
                try:
                    ptr = original(self, *args, **kwargs)
                except (RuntimeError, TypeError, NotImplementedError):
                    ptr = None
                if isinstance(ptr, int) and ptr and _nonowner_ptr_is_captured(state, ptr):
                    _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)
        return original(self, *args, **kwargs)

    return wrapper


def _completeness_census_active() -> bool:
    """Return whether the aten completeness census mode is on the active dispatch stack (r39).

    The mode-independent method/predicate belt is only NEEDED when the census is BLIND -- inside
    a ``_disable_current_modes()`` region that popped :class:`_CompletenessDispatchMode` off the
    dispatch stack (measured E6). When the census IS active it observes the escape at its aten
    dispatch, where the source tensor is fully labelled/attributed; the belt firing there too
    would record the operand PRE-dispatch (before its buffer/op label exists) and mis-route a
    legitimately-witnessed read (e.g. a registered-buffer ``if self.gate``) into the fail-closed
    unattributable gate. So the belt records ONLY when the census is not currently observing.

    The stack query routes through the ``_torch_compat`` accessor
    (``HAS_DISPATCH_MODE_STACK_QUERY``) and FAILS CLOSED (r-b4 R26-1): "cannot answer"
    reads as census-INACTIVE so the belt records the escape. The historical
    ``except Exception: return True`` failed OPEN -- a private-API rename silently
    disarmed the belt in exactly the census-blind regions it exists to cover,
    converting a belt-witnessed escape into an unwitnessed one (false VERIFIED).
    """

    from torchlens.utils._torch_compat import get_current_dispatch_mode_stack

    stack = get_current_dispatch_mode_stack()
    if stack is None:
        return False
    return any(isinstance(mode, _CompletenessDispatchMode) for mode in stack)


def _make_host_value_escape_method(original: Any, state: _WitnessState, name: str) -> Any:
    """Wrap a tensor->host VALUE method to record its tensor operand SOURCES (r39 hon2_1).

    ``item`` / ``__bool__`` / ``__int__`` / ``__float__`` / ``__index__`` / ``__complex__`` and
    the pure predicates ``equal`` / ``allclose`` / ``is_nonzero`` all read a captured tensor's
    VALUE out to the host. The aten census sees them through ``aten._local_scalar_dense`` /
    ``aten.equal`` -- EXCEPT inside torch's own ``_disable_current_modes()`` regions (tensor
    string formatting; explicit predicate guards), which pop the census TorchDispatchMode
    (measured E6). This method patch fires regardless of dispatch-mode state, feeding the SAME
    ``_record_escape_source_tensor(...)`` attribution ladder as the census, so
    the escape is witnessed by its SOURCE tensor's capture-time digest either way.

    Records ``self`` plus any tensor argument (``equal`` / ``allclose`` take a second tensor
    operand), gated to the active trace with TorchLens's own marked internal reads excluded.
    r43: fires on every thread under the owner-vs-non-owner rule -- on a NON-owner thread the
    census mode is never on that thread's dispatch stack (a ``TorchDispatchMode`` is
    thread-local), so this belt is correctly PRIMARY there and ceilings a captured-tensor touch;
    a benign own-tensor touch records nothing. Always calls the exact original unchanged
    (byte-identical values, goldens, and control flow). On the owner the census stays the
    primary observer; this is the idempotent mode-independent belt (shared source table -> no
    double count).
    """

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Record one tensor host-value method invocation when the census is blind.

        Parameters
        ----------
        self:
            Tensor receiver.
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled and not _internal_read_active():
                    if name == "__bool__":
                        _record_bool_consumer_location(state.trace, self)
                    if not _completeness_census_active():
                        _record_escape_source_tensor(state.trace, self)
                        for value in (*args, *kwargs.values()):
                            if isinstance(value, torch.Tensor):
                                _record_escape_source_tensor(state.trace, value)
            elif state.belt_armed:
                # r43: a NON-owner value escape ceilings iff its receiver OR any tensor operand
                # is a captured tensor (the census is thread-local, so the belt is PRIMARY here).
                _nonowner_escape_observe(state, self)
                for value in (*args, *kwargs.values()):
                    if isinstance(value, torch.Tensor):
                        _nonowner_escape_observe(state, value)
        return original(self, *args, **kwargs)

    return wrapper


def _first_scalar_escape_source() -> tuple[str | None, int | None]:
    """Return the first non-TorchLens frame for a scalar escape call.

    Returns
    -------
    tuple[str | None, int | None]
        Source filename and line, or ``(None, None)`` if no user frame is visible.
    """
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            filename = Path(frame.f_code.co_filename).resolve()
            try:
                filename.relative_to(_TORCHLENS_ROOT)
            except ValueError:
                return str(filename), frame.f_lineno
            frame = frame.f_back
    finally:
        del frame
    return None, None


def _record_bool_consumer_location(trace: Any, source: torch.Tensor) -> None:
    """Record where one labelled bool tensor reached Python's ``__bool__`` protocol.

    Parameters
    ----------
    trace:
        Active capture trace.
    source:
        Bool tensor consumed by Python control flow or ``bool(...)``.
    """

    if source.dtype is not torch.bool:
        # DELIBERATE, documented false negative: ``if x.sum():`` truthiness on
        # a non-bool tensor is a real bool consumption, but recording it would
        # materialize conditional arm edges whose predicate the runnable
        # witness-obligation registry cannot witness (only ``is_scalar_bool``
        # ops receive predicate witnesses), making every level="runnable" save
        # of such a model refuse at producer preflight. Lifting this gate
        # requires a truthiness predicate witness family in the runnable
        # contract first. Pinned by
        # tests/test_condbranch_hardening.py::test_float_truthiness_stays_documented_false_negative.
        return
    label = get_tensor_label(source)
    if not isinstance(label, str):
        return
    filename, line_number = _first_scalar_escape_source()
    if filename is None or line_number is None:
        return
    locations = _HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS.get(trace)
    if locations is None:
        locations = {}
        _HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS[trace] = locations
    entries = locations.setdefault(label, [])
    location = (filename, line_number)
    if location not in entries:
        entries.append(location)

    bool_sources = _HOST_ESCAPE_BOOL_SOURCE_LABELS.get(trace)
    if bool_sources is None:
        bool_sources = set()
        _HOST_ESCAPE_BOOL_SOURCE_LABELS[trace] = bool_sources
    bool_sources.add(label)


def _make_plain_scalar_escape_method(
    original: Any,
    state: _PlainScalarEscapeState,
    name: str,
) -> Any:
    """Wrap one tensor scalar protocol method for a plain capture.

    Parameters
    ----------
    original:
        Exact PyTorch method to call unchanged.
    state:
        Per-capture warning aggregate.
    name:
        Tensor scalar-protocol method name.

    Returns
    -------
    Any
        Read-through wrapper around ``original``.
    """

    @functools.wraps(original)
    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Aggregate one plain scalar escape before delegating to ``original``.

        Parameters
        ----------
        self:
            Tensor receiver.
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if (
            _state._logging_enabled
            and _state._active_trace is state.trace
            and threading.get_ident() == state.owner_thread_id
            and not _internal_read_active()
            and get_tensor_label(self) is not None
        ):
            state.count += 1
            if state.first_file is None:
                state.first_file, state.first_line = _first_scalar_escape_source()
            if name == "__bool__":
                _record_bool_consumer_location(state.trace, self)
        return original(self, *args, **kwargs)

    return wrapper


def _external_warning_stacklevel() -> int:
    """Return a warning stack level that resolves outside the TorchLens package.

    Returns
    -------
    int
        Stack level suitable for :func:`warnings.warn`.
    """
    level = 1
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            try:
                Path(frame.f_code.co_filename).resolve().relative_to(_TORCHLENS_ROOT)
            except ValueError:
                return level
            level += 1
            frame = frame.f_back
    finally:
        del frame
    return level


@contextmanager
def capture_scalar_escape_warning(trace: Any) -> Iterator[None]:
    """Warn once when a plain capture reads captured tensor data as Python scalars.

    Parameters
    ----------
    trace:
        Active plain Trace receiving the per-capture aggregate.

    Yields
    ------
    None
        The backend enters active logging inside this scoped method patch.

    Notes
    -----
    Runnable-eligible captures already install the full completeness witness
    belt and are deliberately excluded. This lightweight observer neither
    constructs witness state nor changes capture verification verdicts.
    """
    if bool(getattr(trace, "intervention_ready", False)):
        yield
        return
    if bool(getattr(trace, "structure_only", False)):
        # L7a: structure-only sessions run the ESCALATED belt
        # (structure_only_belt.structure_only_escape_belt), which raises typed
        # teaching refusals instead of aggregating a warning; installing this
        # record-and-warn belt underneath it would be dead weight. The
        # intervention_ready early-exit above is UNREACHABLE in-mode (the
        # structure_only + intervention_ready combination refuses at entry
        # with structure_only_option_conflict), pinned by the entry tests.
        yield
        return

    state = _PlainScalarEscapeState(trace=trace, owner_thread_id=threading.get_ident())
    restores: dict[str, tuple[bool, Any]] = {}

    def _restore_scalar_belt() -> None:
        """Restore every scalar-belt patch that actually landed (shadow-aware)."""
        for name, (shadowed, original) in restores.items():
            if shadowed:
                setattr(torch.Tensor, name, original)
            else:
                delattr(torch.Tensor, name)

    # R07 (the L4 unwind standard): the install loop lands process-global
    # ``torch.Tensor`` scalar-protocol patches on EVERY default capture, so a
    # BaseException escaping it (Python never calls ``__exit__`` when
    # ``__enter__`` raises) used to strand the already-installed methods for
    # the life of the process.
    try:
        for name in HOST_VALUE_ESCAPE_METHODS & {
            "item",
            "__bool__",
            "__int__",
            "__float__",
            "__index__",
            "__complex__",
        }:
            original = getattr(torch.Tensor, name, None)
            if original is None or not callable(original):
                continue
            shadowed = name in torch.Tensor.__dict__
            try:
                setattr(torch.Tensor, name, _make_plain_scalar_escape_method(original, state, name))
            except (TypeError, AttributeError):
                continue
            restores[name] = (shadowed, original)
    except BaseException:
        _restore_scalar_belt()
        raise

    def _warn_escapes() -> None:
        """Emit the one per-capture escape aggregate warning when armed."""
        if not state.count:
            return
        location = (
            f"{state.first_file}:{state.first_line}"
            if state.first_file is not None and state.first_line is not None
            else "an unknown user source location"
        )
        # SF5 (sumfam wave-0 item 5): the advisory used to EVAPORATE -- a
        # Python warning is process-transient, so a shared artifact could not
        # answer "did tensor data escape to Python scalars during capture?".
        # Record it on the trace's annotations (FieldPolicy.KEEP -> persists)
        # BEFORE warnings.warn, whose filters may raise. Only names available
        # in completeness_witness's rebound globals may be used here: locals,
        # ``state``, and builtins -- hence isinstance guards, not try/except.
        annotations = getattr(state.trace, "annotations", None)
        rows = (
            annotations.setdefault("capture_advisories", [])
            if isinstance(annotations, dict)
            else None
        )
        if isinstance(rows, list):
            rows.append(
                {
                    "kind": "scalar_escape",
                    "count": int(state.count),
                    "first_location": (location if state.first_file is not None else None),
                    "message": (
                        "tensor-to-Python scalar escape(s) observed during "
                        "capture; the dependence is not captured"
                    ),
                }
            )
        warnings.warn(
            ScalarEscapeWarning(
                "TorchLens observed "
                f"{state.count} tensor-to-Python scalar escape(s) during capture; "
                f"first at {location}. Keep it as a tensor or pass the value as an "
                "explicit input; the dependence is not captured.",
                file_path=state.first_file,
                line_no=state.first_line,
                count=state.count,
            ),
            stacklevel=_external_warning_stacklevel(),
        )

    try:
        yield
    except BaseException as inflight:
        _restore_scalar_belt()
        # b3-sol (R07 filter-hazard rollup): this advisory used to fire from an
        # unconditional ``finally``, so a warnings-as-error filter raised it
        # DURING UNWIND and replaced the real in-flight capture failure
        # (cascading into the terminal capture-failed advisory). On the
        # exception path the advisory degrades to a note; the success path
        # below keeps full filter semantics.
        # NOTE: only names present in completeness_witness's namespace may be
        # used here -- this contextmanager executes REBOUND into that module's
        # globals() (rebind_contextmanager), so a _completeness_storage-only
        # import raises NameError at call time.
        try:
            _warn_escapes()
        except Exception:
            # Bare try/except (NOT contextlib.suppress): per the rebind NOTE
            # above, this body runs with completeness_witness's globals, where
            # `suppress` is not bound -- the name lookup itself NameError'd and
            # replaced the user's in-flight exception (the exact bug this arm
            # exists to prevent).
            try:
                inflight.add_note(
                    "TorchLens scalar-escape advisory suppressed (a warnings "
                    f"filter raised it): {state.count} tensor-to-Python scalar "
                    "escape(s) were observed during this failed capture."
                )
            except Exception:
                pass
        raise
    else:
        _restore_scalar_belt()
        _warn_escapes()


def _make_host_value_predicate_module_wrapper(original: Any, state: _WitnessState) -> Any:
    """Wrap ``torch.equal`` / ``torch.allclose`` / ``torch.is_nonzero`` to record operands (r39).

    Like the Tensor-method belt, records every tensor operand ONLY when the aten census is not
    currently observing (a ``_disable_current_modes()`` region), so it complements -- never
    duplicates or pre-empts -- the census. On a non-owner thread the census mode is never on
    that thread's stack, so this belt is correctly primary there (r43 owner-vs-non-owner rule:
    a captured-tensor operand ceilings, a benign own-tensor operand records nothing). Distinct
    from :func:`_make_module_escape_wrapper` (dlpack export), which is census-INVISIBLE always
    and therefore records unconditionally.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Record module-level predicate operands when the census is inactive.

        Parameters
        ----------
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if (
                    _state._logging_enabled
                    and not _internal_read_active()
                    and not _completeness_census_active()
                ):
                    for value in (*args, *kwargs.values()):
                        if isinstance(value, torch.Tensor):
                            _record_escape_source_tensor(state.trace, value)
            elif state.belt_armed:
                for value in (*args, *kwargs.values()):
                    if isinstance(value, torch.Tensor):
                        _nonowner_escape_observe(state, value)
        return original(*args, **kwargs)

    return wrapper


def _make_module_escape_wrapper(original: Any, state: _WitnessState) -> Any:
    """Wrap a module-level tensor->host export function to record its tensor argument SOURCE.

    Used for ``torch.utils.dlpack.to_dlpack`` (and, if patchable, ``torch._C._to_dlpack``), which
    are C bindings that NEVER call the Python ``Tensor.__dlpack__`` the method patch covers. The
    wrapper records every tensor operand as an escape source under the active-forward gate
    (r41: on every thread -- in-window fail-closed, foreign positive-only), then calls through
    unchanged so the exported capsule is byte-identical.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Record module-level tensor exports before delegating.

        Parameters
        ----------
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Result from ``original``.
        """
        if _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled:
                    for value in (*args, *kwargs.values()):
                        if isinstance(value, torch.Tensor):
                            _record_escape_source_tensor(state.trace, value)
                            # r65 (F2): ``to_dlpack`` exports a zero-copy capsule pinning
                            # the operand's full layout (strides + byte offset), so a
                            # state-derived operand records the exact-layout read kinds,
                            # mirroring the ``__dlpack__`` method belt.
                            _observe_state_metadata_read(
                                state.trace, value, STATE_METADATA_MIRROR["stride"][1]
                            )
                            _observe_state_metadata_read(
                                state.trace,
                                value,
                                STATE_METADATA_MIRROR["storage_offset"][1],
                            )
            elif state.belt_armed:
                for value in (*args, *kwargs.values()):
                    if isinstance(value, torch.Tensor):
                        _nonowner_escape_observe(state, value)
        return original(*args, **kwargs)

    return wrapper


def _make_invisible_escape_property(
    descriptor: Any, state: _WitnessState, name: str = "__cuda_array_interface__"
) -> property:
    """Wrap a zero-copy buffer PROPERTY to record its SOURCE tensor, then read through.

    Used for ``__cuda_array_interface__`` (a non-callable getset descriptor the method
    patch cannot wrap). The property getter records the receiver tensor as an escape
    source under the same active-forward gate (r41: on every thread -- in-window
    fail-closed, foreign positive-only), then delegates to the original descriptor so
    the returned value is byte-identical.
    """

    def getter(self: torch.Tensor) -> Any:
        """Read the wrapped export property while attributing the tensor source.

        Parameters
        ----------
        self:
            Tensor receiver.

        Returns
        -------
        Any
            Value returned by ``descriptor``.
        """
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled:
                    _record_escape_source_tensor(state.trace, self)
                    # r65 (F2): the CUDA array interface dict carries an explicit
                    # ``strides`` key + data pointer -- a zero-copy layout export exactly
                    # like ``numpy()``/``__dlpack__`` -- so a state-derived receiver
                    # records the exact-layout read kinds.
                    _observe_state_metadata_read(
                        state.trace, self, STATE_METADATA_MIRROR["stride"][1]
                    )
                    _observe_state_metadata_read(
                        state.trace, self, STATE_METADATA_MIRROR["storage_offset"][1]
                    )
            elif state.belt_armed:
                _nonowner_escape_observe(state, self)
        return descriptor.__get__(self, torch.Tensor)

    # doc= MUST be explicit (never defaulted): see DescriptorCompatProperty's
    # constructor-landmine note in _tl.py.
    replacement = DescriptorCompatProperty(getter, doc=name)
    replacement.__objclass__ = getattr(descriptor, "__objclass__", torch.Tensor)
    replacement.__name__ = name
    return replacement

"""Scoped tensor and metadata observer patches."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state

# r-b4 R26-6c: module import so the flag is read at USE time (an import-time
# value binding never sees runtime capability flips).
from ...utils import _torch_compat
from ...utils._torch_symbols import torch_attr

if TYPE_CHECKING:
    from .completeness_witness import (
        _HOST_ESCAPE_OBSERVER_FAILED,
        _INPUT_METADATA_INT_PROPERTY_NAMES,
        _INPUT_METADATA_PRESENCE_PROPERTY_NAMES,
        _MODULE_ESCAPE_TARGETS,
        _STATE_METADATA_DIRECT_ONLY_NAMES,
        _STATE_METADATA_PLACEMENT_OBSERVED_NAMES,
        _STATE_ROUTE_READ_KIND,
        _STORAGE_RAW_POINTER_TARGETS,
        _STORAGE_WRAPPED_DISPOSITIONS,
        HOST_VALUE_ESCAPE_METHODS,
        HOST_VALUE_ESCAPE_MODULE_FUNCS,
        INPUT_METADATA_BOOL_METHODS,
        INPUT_METADATA_PREDICATE_FUNCS,
        INPUT_METADATA_PROPERTY_NAMES,
        INVISIBLE_HOST_ESCAPE_FUNCS,
        INVISIBLE_HOST_ESCAPE_PROPERTIES,
        STATE_METADATA_MIRROR,
        STORAGE_BRIDGE_ESCAPE_FUNCS,
        STORAGE_METADATA_ACCESSOR_DISPOSITIONS,
        _check_writeback_watch,
        _discharge_placement_dispatch,
        _internal_read_active,
        _make_host_value_escape_method,
        _make_host_value_predicate_module_wrapper,
        _make_invisible_escape_property,
        _make_invisible_escape_wrapper,
        _make_module_escape_wrapper,
        _make_nonowner_ops_call,
        _make_nonowner_private_c_callable,
        _make_storage_metadata_wrapper,
        _make_storage_property_wrapper,
        _make_storage_raw_pointer_wrapper,
        _nonowner_escape_observe,
        _observe_input_metadata_read,
        _observe_state_metadata_read,
        _observe_state_metadata_read_direct,
        _observe_state_placement_read,
        _observe_state_property_read,
        _placement_read_witnessed,
        _private_c_module_callables,
        _StorageOriginRegistry,
        _torch_ops_call_classes,
        _WitnessState,
    )

__all__ = (
    "_make_input_metadata_wrapper",
    "_make_input_metadata_bool_method",
    "_make_input_metadata_grad_property",
    "_observe_invisible_host_escapes",
)


def _make_input_metadata_wrapper(
    original: Any, state: _WitnessState, name: str, stride_original: Any
) -> Any:
    """Wrap a layout METHOD (``is_contiguous`` / ``stride`` / ``storage_offset``) to record a
    MODEL-INPUT layout fact, then call through.

    The wrapper computes the original result first (byte-identical behavior), then -- gated
    to the owner thread / active trace / logging-enabled window, with TorchLens's own
    marked internal reads excluded -- attributes a layout fact when the receiver is a
    model-input leaf (or downgrades on a derived view; see
    :func:`_observe_input_metadata_read`). ``stride`` records the FULL stride tuple (a
    dim-scoped ``x.stride(0)`` read is implied by it); ``is_contiguous`` with the default
    memory format records the boolean, while an explicit ``memory_format=`` probe records the
    full stride tuple instead, which determines contiguity under EVERY memory format (given
    the already-checked shape) without enumerating formats. ``storage_offset`` records the
    integer offset read from the RAW pre-clone input.
    """

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Read one input-layout accessor and record any witnessed metadata fact.

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
        result = original(self, *args, **kwargs)
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled and not _internal_read_active():
                    # r63 C1 (r65: table-driven): a layout read on REGISTERED STATE (param/
                    # buffer or a storage alias of one) attributes a state escape + a
                    # per-slot read-kind fact BEFORE the input-scoped observation (receiver
                    # sets are disjoint; a state receiver is invisible to the input nets).
                    # Recorded even when the observed value later fails to normalize (fail
                    # closed). ``is_contiguous`` probed with an explicit ``memory_format=``
                    # pins the exact stride tuple, so it resolves to the ``stride`` row.
                    if name == "is_contiguous" and (args or kwargs):
                        state_read_kind = STATE_METADATA_MIRROR["stride"][1]
                    else:
                        state_read_kind = STATE_METADATA_MIRROR[name][1]
                    _observe_state_metadata_read(state.trace, self, state_read_kind)
                    if name == "storage_offset":
                        try:
                            _observe_input_metadata_read(
                                state.trace, self, "storage_offset", int(result)
                            )
                        except (RuntimeError, TypeError, ValueError):
                            return result
                    elif name == "is_contiguous" and not args and not kwargs:
                        _observe_input_metadata_read(
                            state.trace, self, "is_contiguous", bool(result)
                        )
                    elif stride_original is not None:
                        try:
                            full_stride = tuple(int(v) for v in stride_original(self))
                        except (RuntimeError, TypeError):
                            return result
                        _observe_input_metadata_read(state.trace, self, "stride", full_stride)
            elif state.belt_armed:
                # r43: a non-owner metadata read on a CAPTURED input tensor is a captured-tensor
                # touch -> ceiling; a read on an unrelated own tensor records nothing.
                _nonowner_escape_observe(state, self)
        return result

    return wrapper


def _make_input_metadata_bool_method(original: Any, state: _WitnessState, name: str) -> Any:
    """Wrap a boolean host-value METHOD (``is_conj`` / ``is_neg`` / ``is_inference`` /
    ``is_pinned`` / ``is_shared`` / ``is_coalesced`` / ``_is_view``) to record a MODEL-INPUT
    metadata fact, then call through (r31).

    The wrapper computes the original result first (byte-identical behavior; an accessor that
    raises -- e.g. ``is_coalesced`` on a dense tensor -- propagates unchanged and records
    nothing), then -- gated to the owner thread / active trace / logging-enabled window with
    TorchLens's own marked internal reads excluded -- records ``bool(result)`` when the receiver
    is a model-input leaf or an alias of one (see :func:`_observe_input_metadata_read`). These
    accessors take no value-bearing arguments, so a call carrying args is passed through without
    recording.
    """

    is_placement = name in _STATE_METADATA_PLACEMENT_OBSERVED_NAMES

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Read one boolean metadata accessor and record any witnessed fact.

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

        def _owner_observing() -> bool:
            """Return whether the owner thread is currently recording this accessor.

            Returns
            -------
            bool
                ``True`` when the current call is an owner-thread recording observation.
            """
            return (
                isinstance(self, torch.Tensor)
                and _state._active_trace is state.trace
                and threading.get_ident() == state.owner_thread_id
                and _state._logging_enabled
                and not _internal_read_active()
            )

        try:
            result = original(self, *args, **kwargs)
        except Exception:
            # r67 C3: a RAISING placement accessor on a state receiver is control-flow
            # signal with no reproducible observed value -- record unknown (refuse).
            if is_placement and _owner_observing():
                _observe_state_placement_read(state.trace, self, name, None)
            raise
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if is_placement and (args or kwargs) and _owner_observing():
                    # Arg-directed placement query (``is_pinned(device=...)``): the return
                    # is not the slot's default-device placement -- unknown, refuse.
                    _observe_state_placement_read(state.trace, self, name, None)
                if (
                    not args
                    and not kwargs
                    and _state._logging_enabled
                    and not _internal_read_active()
                ):
                    # r63 C1 (r65: FULL mirror, table-driven -- the is_conj/is_neg-only
                    # branch is gone): every bool metadata accessor is a PHYSICAL state
                    # fact normalized by transport+staging, so a read on registered state
                    # attributes a state escape + read-kind fact. The alias-safe subset
                    # (conj/neg bits, storage/creation placement) attributes by STORAGE
                    # IDENTITY (a view's value is a pure function of the slot's storage);
                    # ``_is_view`` is autograd-family and attributes DIRECT-receiver-only;
                    # ``is_coalesced`` is structural (raises on dense strided state, and
                    # sparse layouts are refused at bind/save by the layout dim).
                    # r67 C3: ``is_shared``/``is_pinned`` carry the ACTUAL returned value
                    # into the observation ledger (observed-value read kinds).
                    state_route = STATE_METADATA_MIRROR.get(name)
                    if is_placement:
                        _observe_state_placement_read(state.trace, self, name, bool(result))
                        # r67 C3: an attributed placement dispatch (``aten.is_pinned``) is
                        # witnessed by the observation/fact ledgers -- discharge its census
                        # event so an ATTRIBUTED read no longer ceilings as an unmodeled
                        # host return; an unattributed receiver keeps the fact.
                        if _placement_read_witnessed(state.trace, self):
                            _discharge_placement_dispatch(state, f"aten.{name}")
                    elif state_route is not None and state_route[0] == _STATE_ROUTE_READ_KIND:
                        if name in _STATE_METADATA_DIRECT_ONLY_NAMES:
                            _observe_state_metadata_read_direct(state.trace, self, state_route[1])
                        else:
                            _observe_state_metadata_read(state.trace, self, state_route[1])
                    try:
                        _observe_input_metadata_read(state.trace, self, name, bool(result))
                    except (RuntimeError, TypeError, ValueError):
                        return result
            elif state.belt_armed:
                _nonowner_escape_observe(state, self)
        return result

    return wrapper


def _make_input_metadata_grad_property(
    descriptor: Any, state: _WitnessState, name: str
) -> property:
    """Wrap an autograd/leaf getset descriptor (``requires_grad`` / ``grad_fn`` / ``is_leaf``)
    to record a MODEL-INPUT autograd fact.

    Like ``_make_invisible_escape_property`` this replaces a non-callable getset descriptor
    with a recording ``property``. ``requires_grad`` is writable (``x.requires_grad = True`` /
    ``requires_grad_()`` inside a forward), so its setter MUST be delegated -- a getter-only
    property would turn that write into an ``AttributeError`` mid-capture; ``grad_fn`` /
    ``is_leaf`` are read-only, so a setter is only installed when the original descriptor
    supports ``__set__``. ``grad_fn`` is recorded as a PRESENCE boolean (the backward object
    itself is not comparable across runs); the others as their boolean value.
    """

    records_presence = name in _INPUT_METADATA_PRESENCE_PROPERTY_NAMES
    records_int = name in _INPUT_METADATA_INT_PROPERTY_NAMES

    def getter(self: torch.Tensor) -> Any:
        """Read one autograd property and record any witnessed metadata fact.

        Parameters
        ----------
        self:
            Tensor receiver.

        Returns
        -------
        Any
            Value returned by ``descriptor``.
        """
        value = descriptor.__get__(self, torch.Tensor)
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled and not _internal_read_active():
                    # r65 Cluster X: the STATE branch (the r64 gap -- this wrapper had
                    # none, so ``self.w.requires_grad`` / ``self.b._version`` reads on
                    # registered state recorded NOTHING). Dispatched through the
                    # authoritative mirror BEFORE the input-scoped observation (receiver
                    # sets are disjoint): ``requires_grad``/``grad_fn`` record a declared
                    # fact staging reproduces; the rest record escape-gated read kinds.
                    _observe_state_property_read(state.trace, self, name, value)
                    if records_presence:
                        fact: Any = value is not None
                    elif records_int:
                        try:
                            fact = int(value)
                        except (TypeError, ValueError):
                            fact = None
                    else:
                        fact = bool(value)
                    if fact is not None:
                        _observe_input_metadata_read(state.trace, self, name, fact)
            elif state.belt_armed:
                _nonowner_escape_observe(state, self)
        return value

    has_setter = hasattr(descriptor, "__set__")

    def setter(self: torch.Tensor, value: Any) -> None:
        """Delegate writes to the wrapped descriptor unchanged.

        Parameters
        ----------
        self:
            Tensor receiver.
        value:
            Value to write through to ``descriptor``.
        """
        descriptor.__set__(self, value)

    return property(getter, setter if has_setter else None)


@contextmanager
def _observe_invisible_host_escapes(state: _WitnessState) -> Iterator[None]:
    """Scoped-patch ``.tolist()`` / ``.numpy()`` / ``__array__`` to record escape sources.

    These conversions emit NO aten dispatch, so the aten census cannot see them. A
    ``TorchFunctionMode`` WOULD observe them but flips ``has_torch_function`` globally,
    which breaks TorchLens's own function-wrapping capture (an unrelated capture bug).
    Instead this temporarily replaces the exact ``torch.Tensor`` conversion methods for the
    duration of ONE runnable forward and restores them unconditionally, without installing
    any torch mode -- so ordinary capture is completely undisturbed. The record is gated to
    the active forward, so TorchLens-internal conversions do not register as user escapes.
    """

    # r3 R07 (shadow gaps): every ``torch.Tensor``/storage-class registry is
    # shadow-aware -- most of these names (``tolist``/``numpy``/``data_ptr``/
    # ``stride``/``is_contiguous``/the bool-method belt/``UntypedStorage.
    # data_ptr``) live as C slots on the BASE class and are NOT in the patched
    # class ``__dict__``, so a plain setattr restore left a permanent shadow
    # entry behind after every armed capture. Restore DELETES a shadow that was
    # not originally present and setattrs one that was.
    originals: dict[str, tuple[bool, Any]] = {}
    host_value_method_restore: dict[str, tuple[bool, Any]] = {}
    module_originals: list[tuple[Any, str, Any]] = []
    storage_originals: list[tuple[Any, bool, Any]] = []
    storage_member_restore: list[tuple[Any, str, bool, Any]] = []
    property_originals: dict[str, tuple[bool, Any]] = {}
    metadata_originals: dict[str, tuple[bool, Any]] = {}
    bool_method_originals: dict[str, tuple[bool, Any]] = {}
    grad_property_restore: dict[str, tuple[bool, Any]] = {}
    torch_ops_call_restore: list[tuple[type, Any]] = []
    private_c_call_restore: list[tuple[Any, str, Any]] = []

    def _restore_installed() -> None:
        """Disarm the belt and restore every patch installed so far.

        Shared by the install-failure unwind and the normal ``finally``:
        each registry holds exactly the patches that actually landed, so a
        partial install restores cleanly.
        """

        state.belt_armed = False
        _state._nonowner_belt_armed = False
        # r47 hon2_1: restore the ``torch._ops`` class ``__call__`` patches FIRST and only when the
        # current attr is still OUR wrapper (preserve a user mutation). A restore failure fails
        # closed. A leaked patch would corrupt ALL torch dispatch process-wide, so this must always
        # run -- it is the first action of the unconditional ``finally``.
        for _ops_cls, _ops_original in torch_ops_call_restore:
            try:
                _current_call = _ops_cls.__dict__.get("__call__")
                if getattr(_current_call, "__tl_nonowner_ops_observer__", False):
                    setattr(_ops_cls, "__call__", _ops_original)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        # r49 hon2_1: restore the private-C module free-function patches, sentinel-guarded
        # (preserve any user mutation) and fail-closed on a restore failure. A leaked patch
        # would misobserve later forwards, so this runs in the unconditional ``finally``
        # alongside the ``torch._ops`` restore.
        for _pc_module, _pc_attr, _pc_original in private_c_call_restore:
            try:
                _pc_current = getattr(_pc_module, _pc_attr, None)
                if getattr(_pc_current, "__tl_nonowner_ops_observer__", False):
                    setattr(_pc_module, _pc_attr, _pc_original)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for name, (was_shadowed, original) in originals.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, name, original)
                else:
                    delattr(torch.Tensor, name)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for module, func_name, original_func in module_originals:
            try:
                setattr(module, func_name, original_func)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for storage_cls, was_shadowed, storage_original in storage_originals:
            try:
                if was_shadowed:
                    setattr(storage_cls, "data_ptr", storage_original)
                else:
                    delattr(storage_cls, "data_ptr")
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        # r67 C3: restore the storage accessor wrappers shadow-aware (delete a shadow that
        # was not originally in the class ``__dict__``) and disarm the origin map. A restore
        # failure fails closed -- a leaked wrapper would misobserve later forwards.
        for storage_cls, member, was_shadowed, restore_value in storage_member_restore:
            try:
                if was_shadowed:
                    setattr(storage_cls, member, restore_value)
                else:
                    delattr(storage_cls, member)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        state.storage_origins = None
        state.storage_state_ptr_names = None
        for name, (was_shadowed, descriptor) in property_originals.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, name, descriptor)
                else:
                    delattr(torch.Tensor, name)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for name, (was_shadowed, original) in metadata_originals.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, name, original)
                else:
                    delattr(torch.Tensor, name)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for name, (was_shadowed, original) in bool_method_originals.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, name, original)
                else:
                    delattr(torch.Tensor, name)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for prop_name, (was_shadowed, original_descriptor) in grad_property_restore.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, prop_name, original_descriptor)
                else:
                    delattr(torch.Tensor, prop_name)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        # r39 hon2_1: restore the host-value method belt shadow-aware (delete a shadow that
        # was not originally in ``torch.Tensor.__dict__``). A restore failure fails closed.
        for name, (was_shadowed, original) in host_value_method_restore.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, name, original)
                else:
                    delattr(torch.Tensor, name)
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)

    # R07 (the L4 unwind standard, generalized): Python never calls ``__exit__``
    # when ``__enter__`` raises, so an exception escaping the install phase used
    # to strand every patch installed so far -- ~124 process-global torch
    # surfaces including ``torch._ops.*.__call__`` and the whole storage
    # accessor surface -- permanently AND monotonically (the next armed capture
    # read the leaked wrapper as its "original" and re-installed over it).
    try:
        for name in INVISIBLE_HOST_ESCAPE_FUNCS | STORAGE_BRIDGE_ESCAPE_FUNCS:
            original = getattr(torch.Tensor, name, None)
            if original is None:
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            shadowed = name in torch.Tensor.__dict__
            try:
                setattr(torch.Tensor, name, _make_invisible_escape_wrapper(original, state, name))
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            originals[name] = (shadowed, original)
        # r39 hon2_1: mode-independent belt for the aten census -- the scalar numeric protocol
        # (``item``/``__bool__``/``__int__``/``__float__``/``__index__``/``__complex__``) and the
        # pure predicates (``equal``/``allclose``/``is_nonzero``). These fire regardless of
        # dispatch-mode state, so a scalar/predicate escape inside torch's own
        # ``_disable_current_modes()`` (tensor string formatting; explicit guards) still hits a
        # Python observer. Several names are getset/slot members of the C ``TensorBase`` and NOT in
        # ``torch.Tensor.__dict__``; setting them installs a SHADOW that restore must DELETE (never
        # set back to the base slot). A required-observer install failure fails the capture closed.
        for name in HOST_VALUE_ESCAPE_METHODS:
            original = getattr(torch.Tensor, name, None)
            if original is None or not callable(original):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            shadowed = name in torch.Tensor.__dict__
            try:
                setattr(torch.Tensor, name, _make_host_value_escape_method(original, state, name))
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            host_value_method_restore[name] = (shadowed, original)
        # Module-level zero-copy export C bindings that bypass the Tensor method patch:
        # ``torch.utils.dlpack.to_dlpack`` == ``torch._C._to_dlpack`` never calls
        # ``Tensor.__dlpack__``. Patch the Python-level function (and ``torch._C._to_dlpack`` if the
        # C module permits assignment) to record the exported tensor as an escape source.
        for module, func_name in _MODULE_ESCAPE_TARGETS():
            original_func = getattr(module, func_name, None)
            if original_func is None:
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            try:
                setattr(module, func_name, _make_module_escape_wrapper(original_func, state))
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            module_originals.append((module, func_name, original_func))
        # r39 hon2_1: the ``torch.*`` MODULE predicate spellings (``torch.equal`` / ``torch.allclose``
        # / ``torch.is_nonzero``) return a raw Python bool DIRECTLY from the dispatcher and, under an
        # explicit ``_disable_current_modes()`` region, bypass the census (E6). Record every tensor
        # operand -- the same shared source table as the Tensor-method belt.
        for predicate_name in HOST_VALUE_ESCAPE_MODULE_FUNCS:
            original_predicate = torch_attr(
                predicate_name
            )  # r47 secD_1: no lazy ``torch.__getattr__``
            if original_predicate is None or not callable(original_predicate):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            try:
                setattr(
                    torch,
                    predicate_name,
                    _make_host_value_predicate_module_wrapper(original_predicate, state),
                )
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            module_originals.append((torch, predicate_name, original_predicate))
        # Storage-handle raw-pointer accessors (r16-C1): ``UntypedStorage.data_ptr`` /
        # ``TypedStorage.data_ptr`` reach the SAME raw pointer as ``Tensor.data_ptr`` but off the
        # storage object, so the Tensor patch never sees them. Fail closed on a genuine user call.
        # W1_F3 (round7 B1 class): an EMPTY storage-class scan is version-drift uncertainty, not
        # proof of absence -- the Tensor storage-bridge methods could still hand out handles of a
        # class this scan failed to enumerate, leaving every storage accessor unobserved. Same
        # fail-closed posture as the ``_torch_ops_call_classes`` / ``_private_c_module_callables``
        # empty scans below.
        if not _STORAGE_RAW_POINTER_TARGETS():
            _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for storage_cls in _STORAGE_RAW_POINTER_TARGETS():
            storage_original = storage_cls.data_ptr
            storage_shadowed = "data_ptr" in storage_cls.__dict__
            try:
                setattr(
                    storage_cls,
                    "data_ptr",
                    _make_storage_raw_pointer_wrapper(storage_original, state),
                )
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            storage_originals.append((storage_cls, storage_shadowed, storage_original))
        # r67 C3/C6: arm the capture-scoped storage-origin map and install the actual-read
        # accessor wrappers over BOTH storage classes' public surfaces, table-driven from
        # ``STORAGE_METADATA_ACCESSOR_DISPOSITIONS``. Fail CLOSED: an install failure on a
        # wrap-required row downgrades the capture to INCOMPLETE (a silent skip would be a
        # silent storage-spelling witness gap). Feature-absent members on this torch are
        # skipped (classified absent, not failed).
        state.storage_origins = _StorageOriginRegistry(
            weak_keys=_torch_compat.HAS_CACHED_UNTYPED_STORAGE_WRAPPER
        )
        for storage_cls in _STORAGE_RAW_POINTER_TARGETS():
            rows = STORAGE_METADATA_ACCESSOR_DISPOSITIONS.get(storage_cls.__name__, {})
            for member, (disposition, _why) in sorted(rows.items()):
                if disposition not in _STORAGE_WRAPPED_DISPOSITIONS or member == "data_ptr":
                    continue
                descriptor = inspect.getattr_static(storage_cls, member, None)
                if descriptor is None:
                    continue  # feature-absent on this torch build
                shadowed = member in storage_cls.__dict__
                class_attr = getattr(storage_cls, member, None)
                if callable(class_attr):
                    replacement: Any = _make_storage_metadata_wrapper(
                        class_attr, state, member, disposition
                    )
                    restore_value: Any = class_attr
                elif hasattr(descriptor, "__get__"):
                    replacement = _make_storage_property_wrapper(
                        descriptor, state, member, disposition
                    )
                    restore_value = descriptor
                else:
                    # W1_F3 (round7 B1 class): a wrap-REQUIRED row whose member EXISTS but is
                    # neither callable nor a descriptor cannot be wrapped, so its reads are
                    # unobservable -- exactly the docstring contract "can neither be wrapped nor
                    # its source recorded": fail closed, never a silent skip.
                    _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                    continue
                try:
                    setattr(storage_cls, member, replacement)
                except (TypeError, AttributeError):
                    _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                    continue
                storage_member_restore.append((storage_cls, member, shadowed, restore_value))
        for name in INVISIBLE_HOST_ESCAPE_PROPERTIES:
            descriptor = inspect.getattr_static(torch.Tensor, name, None)
            if descriptor is None or not hasattr(descriptor, "__get__"):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            shadowed = name in torch.Tensor.__dict__
            try:
                setattr(torch.Tensor, name, _make_invisible_escape_property(descriptor, state))
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            property_originals[name] = (shadowed, descriptor)
        # Model-input METADATA-PREDICATE observers (r27-H2): ``is_contiguous`` / ``stride``
        # methods and the ``requires_grad`` getset descriptor. Read-through recorders gated to
        # MODEL-INPUT receivers only; a model that never reads input layout/grad records nothing.
        stride_original = getattr(torch.Tensor, "stride", None)
        for name in INPUT_METADATA_PREDICATE_FUNCS:
            original = getattr(torch.Tensor, name, None)
            if original is None:
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            shadowed = name in torch.Tensor.__dict__
            try:
                setattr(
                    torch.Tensor,
                    name,
                    _make_input_metadata_wrapper(original, state, name, stride_original),
                )
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            metadata_originals[name] = (shadowed, original)
        # Model-input BOOLEAN metadata METHODS beyond the layout trio (r31): ``is_conj`` /
        # ``is_neg`` / ``is_inference`` / ``is_pinned`` / ``is_shared`` / ``is_coalesced`` /
        # ``_is_view``. Feature-detected (an accessor absent on the running torch is skipped) and
        # gated to model-input receivers/aliases only; a model that never reads them records nothing.
        for name in INPUT_METADATA_BOOL_METHODS:
            original = getattr(torch.Tensor, name, None)
            if original is None or not callable(original):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            shadowed = name in torch.Tensor.__dict__
            try:
                setattr(torch.Tensor, name, _make_input_metadata_bool_method(original, state, name))
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            bool_method_originals[name] = (shadowed, original)
        # ``requires_grad`` / ``grad_fn`` / ``is_leaf`` live as getset descriptors on the C BASE
        # class (``torch._C.TensorBase``), not in ``torch.Tensor.__dict__``; patching installs a
        # SHADOWING property on ``torch.Tensor`` itself, so restore must DELETE the shadow when the
        # name was not originally in ``torch.Tensor.__dict__``.
        for prop_name in INPUT_METADATA_PROPERTY_NAMES:
            prop_descriptor = inspect.getattr_static(torch.Tensor, prop_name, None)
            if prop_descriptor is None or not hasattr(prop_descriptor, "__get__"):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            shadowed = prop_name in torch.Tensor.__dict__
            try:
                setattr(
                    torch.Tensor,
                    prop_name,
                    _make_input_metadata_grad_property(prop_descriptor, state, prop_name),
                )
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            grad_property_restore[prop_name] = (shadowed, prop_descriptor)
        # r43: arm the non-owner captured-tensor belt for the whole forward window. The non-owner
        # observers gate on THIS flag, never the owner's ``pause_logging``-toggled ``_logging_enabled``.
        # r45 hon2_1: ``_state._nonowner_belt_armed`` mirrors ``belt_armed`` (SAME lifetime) so the
        # GLOBAL torch-function wrapper's non-owner fast path can short-circuit on one bool read and
        # only invoke the captured-operand observer during an armed runnable capture.
        state.belt_armed = True
        _state._nonowner_belt_armed = True
        # r47 hon2_1: install a PROCESS-WIDE class-level observer on every ``torch._ops`` class that
        # defines its own ``__call__`` (the ``torch.ops.*`` aten / higher-order / TorchBind surface,
        # which bypasses the global torch-FUNCTION wrapper and whose aten census is thread-local). This
        # is armed-lifecycle-scoped: installed for EXACTLY this forward window and restored FIRST in the
        # ``finally`` so global torch dispatch is pristine the instant the forward ends. Fail CLOSED: an
        # empty scan or an install/restore failure downgrades the capture to INCOMPLETE via
        # ``_HOST_ESCAPE_OBSERVER_FAILED`` -- never a silent "no non-owner op touch".
        _ops_call_classes = _torch_ops_call_classes()
        if not _ops_call_classes:
            _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for _ops_cls in _ops_call_classes:
            try:
                _ops_original = _ops_cls.__dict__["__call__"]
                setattr(_ops_cls, "__call__", _make_nonowner_ops_call(_ops_original))
            except (TypeError, AttributeError, KeyError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            torch_ops_call_restore.append((_ops_cls, _ops_original))
        # r49 hon2_1: extend the armed-lifecycle observer to the patchable private-C FREE-FUNCTION
        # modules (``torch._C._{nn,special,fft,linalg,sparse,nested}``), structurally enumerated
        # from the SAME curated forward-op module authority. These are a THIRD op surface: a
        # private-C free function bypasses BOTH the global torch-FUNCTION wrapper AND the
        # ``torch._ops.*`` class patch (it dispatches its inner aten op in C++), so a non-owner
        # worker consuming a captured operand through ``torch._C._nn.gelu(gate)`` was unwitnessed
        # -> false VERIFIED. Same fail-CLOSED posture: an empty scan or an install/restore failure
        # downgrades the capture to INCOMPLETE via ``_HOST_ESCAPE_OBSERVER_FAILED``.
        _private_c_callables = _private_c_module_callables()
        if not _private_c_callables:
            _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
        for _pc_module, _pc_attr, _pc_original in _private_c_callables:
            try:
                setattr(_pc_module, _pc_attr, _make_nonowner_private_c_callable(_pc_original))
            except (TypeError, AttributeError):
                _HOST_ESCAPE_OBSERVER_FAILED.add(state.trace)
                continue
            private_c_call_restore.append((_pc_module, _pc_attr, _pc_original))
    except BaseException:
        _restore_installed()
        raise
    try:
        yield
    finally:
        _restore_installed()
        _check_writeback_watch(state)

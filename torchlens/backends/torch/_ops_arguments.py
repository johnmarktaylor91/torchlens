"""Argument templates, edge uses, and tensor provenance."""

import types
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

import torch

from ..._ir.container import (
    OutputPathComponent,
)
from ..._save_budget import _retained_storage_identities
from ..._state import pause_logging
from ..._utils._internal_read import internal_scalar_read
from ..._utils._tensor_label import (
    get_live_tensor_label,
    get_param_meta,
    get_tensor_label,
    get_tensor_meta,
    mutated_parameter_label,
    session_label_storage_intact,
    session_meta_is_anchored,
)
from ..._utils._tensor_layout import CaptureLayout, layout_of
from ..._utils._tl_collective import _async_collective_elem
from ..._utils.tensor_utils import (
    safe_copy,
)
from ..._validation_argument_references import active_argument_references
from ...intervention._types import (
    ArgComponent,
    CapturedArgTemplate,
    EdgeUseRecord,
    FunctionRegistryKey,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
from ._ops_arg_slots import _peek_raw_graph_workspace, arg_slot_labels
from ._ops_saved_arg_copies import (
    _copy_saved_arg_tree,
    _copy_saved_call_args,
    _is_flat_sequence,
)
from ._param_fingerprint import param_fingerprint
from .buffer_writes import session_validated_buffer_address

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_function_registry_key",
    "_literal_value_supported",
    "_classify_arg_component",
    "_build_args_template",
    "_arg_location_to_path",
    "_build_edge_use_records",
    "_session_validated_parameter",
    "_tensor_has_known_provenance",
    # Re-exported from their home module for existing importers.
    "_copy_saved_arg_tree",
    "_copy_saved_call_args",
)


def _function_registry_key(
    func: Callable[..., Any], func_name: str | None = None, trace: "Trace | None" = None
) -> FunctionRegistryKey:
    """Build a portable registry key for a captured function.

    Parameters
    ----------
    func
        Function object being logged.
    func_name
        TorchLens-recorded function name for property getter callables whose
        underlying C descriptor does not preserve the property name.
    trace
        Active trace; its raw-graph workspace memoizes C-callable keys per capture.

    Returns
    -------
    FunctionRegistryKey
        Best-effort function identity.
    """

    if func_name in _SAFE_TENSOR_PROPERTY_NAMES:
        return FunctionRegistryKey("torch.Tensor", str(func_name), "method")
    workspace = _peek_raw_graph_workspace(trace)
    if workspace is None or not _is_permanent_c_callable(func):
        return _derive_function_registry_key(func)
    memo = workspace._function_key_memo
    hit = memo.get(id(func))
    if hit is not None and hit[0] is func:
        namespace, qualname, dispatch_kind, version, import_path = hit[1]
        return FunctionRegistryKey(namespace, qualname, dispatch_kind, version, import_path)
    key = _derive_function_registry_key(func)
    memo[id(func)] = (
        func,
        (key.namespace, key.qualname, key.dispatch_kind, key.version, key.import_path),
    )
    return key


# The capture memoizes the portable identity of permanent C callables (torch
# builtins, method and slot descriptors) in its raw-graph workspace, keyed by
# ``id`` with the callable held so the id cannot be reused: every op of the same
# function used to re-derive the same key (module and qualname reads, the
# internal-builtin probe, the runnable alias table scan). The memo lives and dies
# with one capture, so a torch namespace patched between captures is re-derived,
# and each hit returns a NEW key object, so no two ops share a key instance.
_PERMANENT_C_CALLABLE_TYPES = (
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
)


def _is_permanent_c_callable(func: Any) -> bool:
    """Return whether ``func`` is a C callable owned by a module or class, never an instance.

    A static C function (``torch.sin``: ``METH_STATIC`` on ``_VariableFunctionsClass``)
    is bound to nothing, so its ``__self__`` is ``None``.
    """

    func_type = type(func)
    if func_type in _PERMANENT_C_CALLABLE_TYPES:
        return True
    if func_type is not types.BuiltinFunctionType:
        return False
    owner = getattr(func, "__self__", None)
    return owner is None or isinstance(owner, (types.ModuleType, type))


def _derive_function_registry_key(func: Callable[..., Any]) -> FunctionRegistryKey:
    """Derive the portable registry key of ``func`` (no memo)."""

    from torchlens.intervention._resolver_registry import function_registry_key_from_callable

    return function_registry_key_from_callable(func)


def _literal_value_supported(value: Any) -> bool:
    """Return whether ``value`` is a replay-safe literal.

    Parameters
    ----------
    value
        Value to classify.

    Returns
    -------
    bool
        True when the value can be stored directly in an argument template.
    """

    # ``...``: ``x[..., :n]`` indexing (rotary halves, Gemma-class
    # attention) is a plain singleton literal; the runnable literal encoder and
    # the safe unpickler already admit it, so replay stores it directly.
    return value is Ellipsis or isinstance(
        value,
        (int, float, bool, str, bytes, type(None), torch.dtype, torch.device, slice),
    )


def _capture_layout(value: Any) -> CaptureLayout | None:
    """Record a parent argument's non-contiguous layout for faithful replay.

    ``stride()`` / ``storage_offset()`` are witnessed metadata surfaces, so this
    bookkeeping read runs under ``pause_logging`` plus ``internal_scalar_read``
    (the sanctioned pattern for TorchLens's own reads during capture).
    """

    if not isinstance(value, torch.Tensor):
        return None
    with pause_logging(), internal_scalar_read():
        return layout_of(value)


#: Exact builtin types whose instances take no attributes, so carry no TorchLens stamp:
#: their label read is always ``None`` and the template walk skips it.
_UNSTAMPABLE_TYPES = frozenset(
    {int, float, bool, complex, str, bytes, type(None), list, tuple, dict, slice, type(...)}
)


def _label_free(value: Any) -> bool:
    """Return whether ``value`` is a non-tensor the label gate would answer ``None`` for.

    Tensors go through the snapshot or a live read. Any other value has a label only
    through a stamped meta, so one meta read (which raises on a foreign stamp exactly
    as the gate does) replaces the gated label read.
    """

    if type(value) in _UNSTAMPABLE_TYPES:
        return True
    if isinstance(value, torch.Tensor):
        return False
    meta = get_tensor_meta(value)
    return meta is None or meta.label_raw is None


def _classify_arg_component(
    value: Any,
    notes: list[str],
    trace: "Trace | None" = None,
    known: dict[int, str | None] | None = None,
) -> ArgComponent:
    """Classify a function argument value for replay templating.

    Parameters
    ----------
    value
        Argument value to classify.
    notes
        Accumulator for unsupported-value notes.
    known
        The op's snapshot labels by tensor id (read once per op); tensors not in it
        are read live, and non-tensor values without a stamped label are not read.

    Returns
    -------
    ArgComponent
        Tagged replay template component.
    """

    label = None
    if known is not None and (id(value) in known or _label_free(value)):
        label = known.get(id(value))
    # A Parameter mutated in place earlier in the pass replays from its mutation op.
    elif not isinstance(value, torch.nn.Parameter) or mutated_parameter_label(value) is not None:
        if trace is None:
            label = get_tensor_label(value)
        else:
            label = get_live_tensor_label(value, trace.capture_events.live_index.by_raw_label)
    if isinstance(label, str):
        return ParentRef(label, _capture_layout=_capture_layout(value))
    if isinstance(value, torch.Tensor):
        return _unlabeled_tensor_literal(value)
    if _literal_value_supported(value):
        return LiteralValue(value)
    if isinstance(value, (list, tuple)):
        return tuple(_classify_arg_component(item, notes, trace, known) for item in value)
    if isinstance(value, dict):
        return tuple(
            (key, _classify_arg_component(item, notes, trace, known)) for key, item in value.items()
        )
    return _unsupported_component(value, notes)


def _unlabeled_tensor_literal(value: torch.Tensor) -> LiteralTensor:
    """Template a tensor argument that has no live parent label as a literal."""

    if isinstance(value, torch.nn.Parameter):
        # r75 F2: snapshot the model-prep barcode NOW (model provably alive) --
        # session cleanup strips the weak registry meta before a deferred save,
        # so the template itself must carry the capture-time identity.
        param_meta = get_param_meta(value)
        param_barcode = getattr(param_meta, "param_barcode", None)
        return LiteralTensor(
            value,
            param_barcode=str(param_barcode) if param_barcode is not None else None,
            _capture_param_fingerprint=param_fingerprint(
                value, getattr(param_meta, "param_address", None)
            ),
        )
    with pause_logging():
        return LiteralTensor(safe_copy(value))


def _unsupported_component(value: Any, notes: list[str]) -> Unsupported:
    """Template an argument value replay cannot store, noting why."""

    reason = f"unsupported argument type {type(value).__module__}.{type(value).__qualname__}"
    notes.append(reason)
    return Unsupported(reason=reason, value_type=type(value).__qualname__)


def _build_args_template(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    trace: "Trace | None" = None,
    *,
    func_name: str | None = None,
) -> CapturedArgTemplate:
    """Build a replay template from original function args and kwargs.

    Parameters
    ----------
    func
        Function object being logged.
    func_name
        TorchLens-recorded function name.
    args
        Original positional args.
    kwargs
        Original keyword args.
    trace
        Active trace used to reject stale parent labels.

    Returns
    -------
    CapturedArgTemplate
        Replay template for the function call.
    """

    notes: list[str] = []
    known = arg_slot_labels(trace, args, kwargs)
    arg_components = tuple(_classify_arg_component(arg, notes, trace, known) for arg in args)
    kwarg_components = tuple(
        (str(key), _classify_arg_component(value, notes, trace, known))
        for key, value in kwargs.items()
    )
    return CapturedArgTemplate(
        args=arg_components,
        kwargs=kwarg_components,
        func_id=_function_registry_key(func, func_name, trace),
        notes=tuple(notes),
    )


def _arg_location_to_path(location: Any) -> tuple[OutputPathComponent, ...]:
    """Convert a parent-layer arg location to the MVP edge path schema.

    Parameters
    ----------
    location
        Location key from ``parent_arg_positions``.

    Returns
    -------
    tuple[OutputPathComponent, ...]
        Path tuple mirroring the existing two-level arg-location scheme.
    """

    if isinstance(location, tuple):
        return location
    return (location,)


def _build_edge_use_records(
    self: "Trace",
    parent_arg_positions: dict[str, dict[Any, str]],
    child_label: str,
    child_func_call_id: int,
    func_name: str | None = None,
) -> list[EdgeUseRecord]:
    """Build edge provenance records from existing parent arg locations.

    ``view_or_copy`` populates from the closed
    function-semantics table (``intervention.edge_semantics``) -- the
    positional slot-0 edge of a documented always-view/always-copy call
    gets its verdict, everything else stays the honest ``unknown``.

    Parameters
    ----------
    self
        Active model log.
    parent_arg_positions
        Existing parent-location map.
    child_label
        Raw label for the child tensor output.
    child_func_call_id
        Function call id for the child operation.
    func_name
        The child call's function name (drives the storage verdict).

    Returns
    -------
    list[EdgeUseRecord]
        Edge provenance records.
    """

    from ...intervention._edge_semantics import classify_view_or_copy

    _edge_uses: list[EdgeUseRecord] = []
    for location, parent_label in parent_arg_positions["args"].items():
        arg_path = _arg_location_to_path(location)
        _edge_uses.append(
            EdgeUseRecord(
                parent_label=parent_label,
                child_label=child_label,
                arg_kind="positional",
                arg_path=arg_path,
                view_or_copy=cast(
                    Any, classify_view_or_copy(func_name, cast(tuple[Any, ...], arg_path))
                ),
                parent_func_call_id=self.capture_events.live_index.require_event(
                    parent_label
                ).func_call_id,
                child_func_call_id=child_func_call_id,
            )
        )
    for location, parent_label in parent_arg_positions["kwargs"].items():
        _edge_uses.append(
            EdgeUseRecord(
                parent_label=parent_label,
                child_label=child_label,
                arg_kind="keyword",
                arg_path=_arg_location_to_path(location),
                view_or_copy="unknown",
                parent_func_call_id=self.capture_events.live_index.require_event(
                    parent_label
                ).func_call_id,
                child_func_call_id=child_func_call_id,
            )
        )
    return _edge_uses


def _session_validated_parameter(trace: "Trace", value: torch.Tensor) -> bool:
    """Return whether ``value`` is a CURRENT-SESSION prep-stamped model Parameter.

    The param provenance rung, factored for reuse: a non-empty prep
    :class:`ParamMeta` address that resolves in THIS capture's ``param_logs``
    with EXACT object identity. A fresh in-forward ``nn.Parameter``, a foreign
    model's parameter, or a stale leaked stamp never validates here.

    Parameters
    ----------
    trace:
        Active capture Trace whose session the prep stamp must belong to.
    value:
        Parameter argument to inspect.

    Returns
    -------
    bool
        True when the prep stamp resolves session-validly for this object.
    """

    param_meta = get_param_meta(value)
    if param_meta is None or not param_meta.param_address:
        return False
    addr = param_meta.param_address
    param_logs = trace._param_logs
    if param_logs is None or addr not in param_logs:
        return False
    if getattr(param_logs[addr], "_param_ref", None) is value:
        return True
    # Offload rebind rung: accelerate offload hooks REPLACE the
    # prep-stamped meta parameter with a fresh real-valued object at every
    # forward. The offload shim (backends/torch/offload_hooks.py) registers
    # exactly that materialization -- the object it found in
    # ``module._parameters`` at the prep-recorded address, during THIS
    # session's forward -- in a session-scoped weak-valued registry. Exact
    # object identity against that registry is the same per-object belt the
    # ``_param_ref`` rung provides; anything not rebound by the shim this
    # session (foreign params, stale leaked stamps, fresh in-forward
    # Parameters) still fails closed.
    try:
        rebinds = trace._offload_param_rebinds
    except AttributeError:
        rebinds = None
    return rebinds is not None and rebinds.get(addr) is value


def _tensor_has_known_provenance(trace: "Trace", value: torch.Tensor) -> bool:
    """Return whether a tensor carries TorchLens input/op/buffer provenance.

    Parameters
    ----------
    trace:
        Active capture Trace whose session the provenance stamp must belong to.
    value:
        Tensor argument to inspect.

    Returns
    -------
    bool
        True when the tensor is a CURRENT-SESSION prep-stamped Parameter or has
        TorchLens tensor metadata.

    Notes
    -----
    the Parameter rung requires ACTUAL TorchLens provenance -- the
    prep-stamped :class:`ParamMeta` address written by model preparation
    (``_create_session_param_logs``) -- not bare ``isinstance``. A fresh
    in-forward ``torch.nn.Parameter(...)`` or a foreign model's parameter has no
    prep stamp; exempting it suppressed the ``unattributed_tensor_args`` break
    marker, so the layout ancestry-integrity rung judged the chain CLEAN and
    a layout twin replayed as false VERIFIED. The lazy op-time barcode stamp
    (``_process_parent_param_ops``, which runs BEFORE this check for the same
    call) writes ``param_address=""``, so registry presence or a barcode alone
    is NOT provenance -- only the non-empty prep address is. Unprepped
    Parameters fall through to the tensor-meta rung like any other tensor and
    leave the break marker when unlabeled.

    Session-leak defense in depth: a non-empty prep address counts ONLY when
    it resolves in THIS capture's session -- the address maps in
    ``trace._param_logs`` AND the recorded log's live object IS this value (exact
    identity). A stamp that leaked from a PRIOR session (e.g. a param popped
    from ``_parameters`` mid-forward) can
    therefore never be accepted, even if some future path escapes the
    inventory-driven cleanup again. Stale/foreign stamps fall through to the
    tensor-meta rung and leave the break marker.

    Buffer-rung parity: the tensor-meta rung requires
    CURRENT-SESSION provenance component-wise, exactly like the param rung:

    * ``label_raw`` counts only when it resolves in THIS capture's live event
      index (a stale cross-capture label -- e.g. the ``buffer_N_raw`` stamped
      onto a reassigned external by the donor's ``_record_write`` -- never
      does, so it leaves the break marker instead of silently passing);
    * ``address`` (the static buffer stamp) counts only when
      :func:`session_validated_buffer_address` proves the exact object was
      stamped THIS session AND its live storage is still the stamped storage
      (an input-``.data=``-rebound plain-attr buffer fails the storage match;
      a stale cross-capture object never resolves at all);
    * ``buffer_source`` (a promoted pre-buffer producer label) counts only
      when that producer label resolves in THIS capture's live event index.

    Label-rung parity: live-index membership is TEXT, and label text is
    deterministic per op-kind + ordinal, so an ordinary op in a later unrelated
    capture regenerates the same string and a text-only check would accept a
    foreign tensor as current-session state. Both label components
    additionally require :func:`session_meta_is_anchored` -- the object itself
    must have been stamped with that exact string during the ACTIVE session --
    giving the label rung the same per-object belt the param and buffer
    ``address`` rungs already had. Fail-closed: with no session installed,
    or with an object the session never stamped, the component is not
    provenance and the break marker stands.
    """

    if isinstance(value, torch.nn.Parameter) and _session_validated_parameter(trace, value):
        return True

    act_inner = _async_collective_elem(value)
    if act_inner is not None:
        # AsyncCollectiveTensor is a transparent async view of its inner
        # ``.elem``: the funcol boundary labeled the
        # inner tensor, and user code passes the wrapper onward. Provenance is
        # the inner tensor's -- same delegation as the get_tensor_label
        # chokepoint, same no-wait ``.elem`` attribute read.
        return _tensor_has_known_provenance(trace, act_inner)
    meta = get_tensor_meta(value)
    if meta is None:
        return False
    capture_events = getattr(trace, "capture_events", None)
    live_labels: dict[str, Any] = (
        capture_events.live_index.by_raw_label if capture_events is not None else {}
    )
    label_anchored = session_meta_is_anchored(meta)
    # Label-rung STORAGE parity: an anchored label proves current-session
    # IDENTITY; it counts as provenance only when the object's live storage
    # is ALSO still the storage it held when labeled. A state-derived
    # activation whose storage was ``.data=``/``set_``-rebound to input-derived
    # data after labeling fails the storage check and leaves the break
    # marker, instead of replaying the pre-rebind value as a false VERIFIED. An
    # in-place write into the object's OWN storage keeps the pointer and passes,
    # so honest journaled/tracked mutation is untouched. The buffer ``address``
    # rung already carries this storage belt via ``session_validated_buffer_address``
    # ; this closes the same cell on the label/``buffer_source`` rung.
    label_storage_intact = label_anchored and session_label_storage_intact(meta, value)
    if label_storage_intact and meta.label_raw is not None and meta.label_raw in live_labels:
        return True
    if meta.address is not None and session_validated_buffer_address(trace, value) is not None:
        return True
    return bool(
        label_storage_intact
        and meta.buffer_source is not None
        and meta.buffer_source in live_labels
    )


def _saved_arg_retention_devices(t_args: Any, t_kwargs: Any) -> dict[int, torch.device]:
    """Map ``id`` of each argument tensor whose retained copy moves to the device it moves to.

    Mirrors :func:`_copy_saved_arg_tree` for the retained snapshot: only a bare
    tensor argument or an item of a flat list or tuple can move. Empty outside
    validation's offloading capture.
    """

    references = active_argument_references()
    if references is None or not references.offloads:
        return {}
    devices: dict[int, torch.device] = {}
    with pause_logging():
        for arg in (*t_args, *t_kwargs.values()):
            items = arg if _is_flat_sequence(arg) else (arg,)
            for item in items:
                if isinstance(item, torch.Tensor):
                    target = references.retention_device(item)
                    if target is not None:
                        devices[id(item)] = target
    return devices


def _uncharged_parameter_aliases(*roots: Any) -> set[int]:
    """Return ``id`` of each tensor that shares storage with a live parameter of validation's copy.

    Only inside validation's own capture, on every ``output_device``: such an
    alias adds no storage, so the save budget charges it nothing. Empty
    outside validation, and on its clone-path rerun or a capture of the live
    model (no parameter is aliased there), where every argument is charged.
    """

    references = active_argument_references()
    if references is None:
        return set()
    with pause_logging():
        return {
            id(tensor)
            for tensor in _iter_tree_tensors(*roots)
            if references.shares_parameter_storage(tensor) or references.is_value_alias(tensor)
        }


def _iter_tree_storages(*roots: Any) -> Iterator[torch.UntypedStorage | torch.TypedStorage]:
    """Yield every storage leaf (a ``Tensor.set_`` storage argument) through builtin containers."""

    stack: list[Any] = list(roots)
    while stack:
        value = stack.pop()
        if isinstance(value, (torch.UntypedStorage, torch.TypedStorage)):
            yield value
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)


def _iter_tree_tensors(*roots: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor leaf reachable through builtin containers."""

    stack: list[Any] = list(roots)
    while stack:
        value = stack.pop()
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)


def _project_saved_args_bytes(t_args: Any, t_kwargs: Any) -> dict[Any, int]:
    """Project alias-deduped physical bytes per device for a saved-args snapshot."""
    # Offloaded copies (validation with an ``output_device``) are charged to
    # the device they will live on, not the device the op consumed them on;
    # parameter aliases in validation's capture add no storage and are not charged.
    moved_to = _saved_arg_retention_devices(t_args, t_kwargs)
    aliases = _uncharged_parameter_aliases(t_args, t_kwargs)
    per_device: dict[Any, int] = {}
    seen_identities: set[tuple[Any, ...]] = set()
    for tensor in _iter_tree_tensors(t_args, t_kwargs):
        if id(tensor) in aliases:
            continue
        device = moved_to.get(id(tensor), tensor.device)
        for identity, num_bytes in _retained_storage_identities(tensor):
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            per_device[device] = per_device.get(device, 0) + num_bytes
    for storage in _iter_tree_storages(t_args, t_kwargs):
        for identity, num_bytes in _retained_storage_identities(storage):
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            per_device[storage.device] = per_device.get(storage.device, 0) + num_bytes
    return per_device


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from .ops import (  # noqa: E402
    _SAFE_TENSOR_PROPERTY_NAMES,
)

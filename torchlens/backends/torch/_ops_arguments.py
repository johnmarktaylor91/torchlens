"""Argument templates, edge uses, and tensor provenance."""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, cast

import torch

from ..._ir.container import (
    OutputPathComponent,
)
from ..._save_budget import _retained_storage_identities
from ..._state import pause_logging
from ..._storage_bindings import STORAGE_CONTEXT_RECEIVER_FUNCS, copy_with_storage_context
from ..._utils._tensor_label import (
    get_live_tensor_label,
    get_param_meta,
    get_tensor_label,
    get_tensor_meta,
    mutated_parameter_label,
    session_label_storage_intact,
    session_meta_is_anchored,
)
from ..._utils._tl_collective import _async_collective_elem
from ..._utils.arg_handling import copy_arg_tree
from ..._utils.tensor_utils import (
    safe_copy,
)
from ..._validation_argument_references import active_argument_references
from ...intervention.types import (
    ArgComponent,
    CapturedArgTemplate,
    EdgeUseRecord,
    FunctionRegistryKey,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
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
)


def _function_registry_key(
    func: Callable[..., Any], func_name: str | None = None
) -> FunctionRegistryKey:
    """Build a portable registry key for a captured function.

    Parameters
    ----------
    func
        Function object being logged.
    func_name
        TorchLens-recorded function name for property getter callables whose
        underlying C descriptor does not preserve the property name.

    Returns
    -------
    FunctionRegistryKey
        Best-effort function identity.
    """

    from torchlens.intervention._resolver_registry import function_registry_key_from_callable

    if func_name in _SAFE_TENSOR_PROPERTY_NAMES:
        return FunctionRegistryKey("torch.Tensor", str(func_name), "method")
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


def _classify_arg_component(
    value: Any, notes: list[str], trace: "Trace | None" = None
) -> ArgComponent:
    """Classify a function argument value for replay templating.

    Parameters
    ----------
    value
        Argument value to classify.
    notes
        Accumulator for unsupported-value notes.

    Returns
    -------
    ArgComponent
        Tagged replay template component.
    """

    label = None
    # A Parameter mutated in place earlier in the pass replays from its mutation op.
    if not isinstance(value, torch.nn.Parameter) or mutated_parameter_label(value) is not None:
        if trace is None:
            label = get_tensor_label(value)
        else:
            label = get_live_tensor_label(value, trace.capture_events.live_index.by_raw_label)
    if isinstance(label, str):
        return ParentRef(label)
    if isinstance(value, torch.Tensor):
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
    if _literal_value_supported(value):
        return LiteralValue(value)
    if isinstance(value, (list, tuple)):
        return tuple(_classify_arg_component(item, notes, trace) for item in value)
    if isinstance(value, dict):
        return tuple(
            (key, _classify_arg_component(item, notes, trace)) for key, item in value.items()
        )

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
    arg_components = tuple(_classify_arg_component(arg, notes, trace) for arg in args)
    kwarg_components = tuple(
        (str(key), _classify_arg_component(value, notes, trace)) for key, value in kwargs.items()
    )
    return CapturedArgTemplate(
        args=arg_components,
        kwargs=kwarg_components,
        func_id=_function_registry_key(func, func_name),
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

    F10 (lovely item 8): ``view_or_copy`` populates from the closed
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

    from ...intervention.edge_semantics import classify_view_or_copy

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

    The r77/r79 param provenance rung, factored for reuse: a non-empty prep
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
    # Offload rebind rung (lane F37): accelerate offload hooks REPLACE the
    # prep-stamped meta parameter with a fresh real-valued object at every
    # forward. The offload shim (backends/torch/offload_hooks.py) registers
    # exactly that materialization -- the object it found in
    # ``module._parameters`` at the prep-recorded address, during THIS
    # session's forward -- in a session-scoped weak-valued registry. Exact
    # object identity against that registry is the same per-object belt the
    # r79 ``_param_ref`` rung provides; anything not rebound by the shim this
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
    r77 F1: the Parameter rung requires ACTUAL TorchLens provenance -- the
    prep-stamped :class:`ParamMeta` address written by model preparation
    (``_create_session_param_logs``) -- not bare ``isinstance``. A fresh
    in-forward ``torch.nn.Parameter(...)`` or a foreign model's parameter has no
    prep stamp; exempting it suppressed the ``unattributed_tensor_args`` break
    marker, so the r75 layout ancestry-integrity rung judged the chain CLEAN and
    a layout twin replayed as false VERIFIED. The lazy op-time barcode stamp
    (``_process_parent_param_ops``, which runs BEFORE this check for the same
    call) writes ``param_address=""``, so registry presence or a barcode alone
    is NOT provenance -- only the non-empty prep address is. Unprepped
    Parameters fall through to the tensor-meta rung like any other tensor and
    leave the break marker when unlabeled.

    r79 session-leak defense in depth: a non-empty prep address counts ONLY when
    it resolves in THIS capture's session -- the address maps in
    ``trace._param_logs`` AND the recorded log's live object IS this value (exact
    identity). A stamp that leaked from a PRIOR session (r78: a param popped
    from ``_parameters`` mid-forward escaped the old re-traversal cleanup) can
    therefore never be accepted, even if some future path escapes the
    inventory-driven cleanup again. Stale/foreign stamps fall through to the
    tensor-meta rung and leave the break marker.

    r81 buffer-rung parity (r80 F1+F2): the tensor-meta rung now requires
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

    r83 C1 label-rung parity: live-index membership is TEXT, and label text is
    deterministic per op-kind + ordinal, so an ordinary op in a later unrelated
    capture regenerates the same string and r81's text-only check accepted a
    foreign tensor as current-session state. Both label components now
    additionally require :func:`session_meta_is_anchored` -- the object itself
    must have been stamped with that exact string during the ACTIVE session --
    giving the label rung the same per-object belt the param (r79) and buffer
    ``address`` (r81) rungs already had. Fail-closed: with no session installed,
    or with an object the session never stamped, the component is not
    provenance and the break marker stands.
    """

    if isinstance(value, torch.nn.Parameter) and _session_validated_parameter(trace, value):
        return True

    act_inner = _async_collective_elem(value)
    if act_inner is not None:
        # AsyncCollectiveTensor is a transparent async view of its inner
        # ``.elem`` (merge-ranks C2 recording): the funcol boundary labeled the
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
    # r85 label-rung STORAGE parity: an anchored label proves current-session
    # IDENTITY (r83); it counts as provenance only when the object's live storage
    # is ALSO still the storage it held when labeled (r85). A state-derived
    # activation whose storage was ``.data=``/``set_``-rebound to input-derived
    # data after labeling (SOL-1) fails the storage check and leaves the break
    # marker, instead of replaying the pre-rebind value as a false VERIFIED. An
    # in-place write into the object's OWN storage keeps the pointer and passes,
    # so honest journaled/tracked mutation is untouched. The buffer ``address``
    # rung already carries this storage belt via ``session_validated_buffer_address``
    # (r81); this closes the same cell on the label/``buffer_source`` rung.
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


def _copy_saved_call_args(
    func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any], retained: bool = False
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Copy one call's arguments for ``save_arg_values``, receiver storage context included.

    The receiver (positional argument 0) of a
    :data:`~torchlens._storage_bindings.STORAGE_CONTEXT_RECEIVER_FUNCS` call keeps
    its whole storage; every other argument is copied as :func:`_copy_saved_arg_tree`
    does.
    """

    storage_context = func_name in STORAGE_CONTEXT_RECEIVER_FUNCS
    arg_copies = tuple(
        _copy_saved_arg_tree(arg, retained, storage_context=storage_context and index == 0)
        for index, arg in enumerate(args)
    )
    kwarg_copies = {key: _copy_saved_arg_tree(value, retained) for key, value in kwargs.items()}
    return arg_copies, kwarg_copies


def _copy_saved_arg_tree(arg: Any, retained: bool = False, *, storage_context: bool = False) -> Any:
    """Copy one ``save_arg_values`` argument tree, as the pre-call and saved snapshots do.

    Outside forward validation's own capture this is exactly
    :func:`~torchlens._utils.arg_handling.copy_arg_tree`. Inside it, a plain
    parameter argument (or a tensor on a parameter's storage), bare or in a
    flat list or tuple, is replaced by an alias of the same storage and version
    counter instead of a copy; see :mod:`torchlens._validation_argument_references`
    for the guard that reruns validation on this copy path if such a value changes.
    For the ``retained`` snapshot (``saved_args``/``saved_kwargs``), a validation
    ``output_device`` other than ``"same"`` also keeps every other such tensor's
    copy on that device, tagged with the device the op consumed it on.
    """

    references = active_argument_references()
    if references is None:
        if storage_context and isinstance(arg, torch.Tensor):
            return _copy_tensor_with_storage_context(arg)
        return copy_arg_tree(arg)
    if isinstance(arg, torch.Tensor):
        return _parameter_alias(references, arg, retained, storage_context=storage_context)
    if _is_flat_sequence(arg):
        return type(arg)(
            _parameter_alias(references, item, retained) if isinstance(item, torch.Tensor) else item
            for item in arg
        )
    return copy_arg_tree(arg)


def _is_flat_sequence(arg: Any) -> bool:
    """Whether ``arg`` is a plain list or tuple holding no nested container."""

    return type(arg) in (list, tuple) and not any(
        isinstance(item, (list, tuple, dict)) for item in arg
    )


def _parameter_alias(
    references: Any, tensor: torch.Tensor, retained: bool = False, *, storage_context: bool = False
) -> torch.Tensor:
    """Alias a parameter-storage tensor (recording its version), else copy it as before.

    The guard decides and builds the alias (see
    ``ParameterArgumentReferences.alias``) and, for a ``retained`` snapshot,
    the offloaded copy (``ParameterArgumentReferences.offload``); anything it
    declines is copied exactly as on the copy path (with its whole storage
    when ``storage_context`` is set; an alias already shares the storage).
    """

    # Paused: the guard's storage and version reads, its alias and its copy are
    # TorchLens's own calls, never captured ops (and never re-enter this copy).
    with pause_logging():
        alias = references.alias(tensor)
        if alias is None and retained and references.offloads:
            alias = references.offload(tensor)
    if alias is not None:
        return alias
    if storage_context:
        return _copy_tensor_with_storage_context(tensor)
    return cast(torch.Tensor, copy_arg_tree(tensor))


def _copy_tensor_with_storage_context(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a storage-geometry receiver with its whole storage, as TorchLens's own work."""

    with pause_logging():
        return copy_with_storage_context(tensor)


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
            if references.shares_parameter_storage(tensor)
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

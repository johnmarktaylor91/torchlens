"""Registered-buffer write capture for the torch backend."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch
from torch import nn

from ... import _state
from ...ir import BufferWriteEvent, live_record_for_label
from ...utils.tensor_utils import safe_copy
from ._tl import get_buffer_address, get_tensor_label, set_buffer_address

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


_FUSED_MUTATOR_NAMES = {
    "batch_norm",
    "native_batch_norm",
    "_native_batch_norm_legit",
    "_native_batch_norm_legit_no_training",
    "cudnn_batch_norm",
    "instance_norm",
    "native_group_norm",
}


@dataclass(frozen=True, slots=True)
class BufferSnapshot:
    """Pre-call snapshot for a tensor argument backed by registered-buffer storage."""

    address: str
    tensor: torch.Tensor
    object_id: int
    storage_key: tuple[Any, ...] | None
    version: int | None
    value: torch.Tensor


@dataclass(slots=True)
class _PatchedClass:
    """Original ``__setattr__`` and active prepared instances for one module class."""

    original_setattr: Callable[[Any, str, Any], None]
    prepared_instances: weakref.WeakSet[nn.Module]
    refcount: int = 0


class BufferWriteTracker:
    """Session-scoped registered-buffer write capture state."""

    _patched_classes: ClassVar[dict[type[nn.Module], _PatchedClass]] = {}

    def __init__(self, trace: "Trace", model: nn.Module) -> None:
        """Initialize capture state for one trace/model session.

        Parameters
        ----------
        trace:
            Active trace receiving write events.
        model:
            Root module whose registered buffers are tracked.
        """

        self.trace = trace
        self.model_ref = weakref.ref(model)
        self.address_to_tensor: dict[str, torch.Tensor] = {}
        self.address_to_snapshot: dict[str, torch.Tensor] = {}
        self.address_to_object_id: dict[str, int] = {}
        self.address_to_storage_key: dict[str, tuple[Any, ...] | None] = {}
        self.address_to_version: dict[str, int | None] = {}
        self._installed_classes: set[type[nn.Module]] = set()

    def install(self) -> None:
        """Install scoped class ``__setattr__`` patches and seed the buffer index."""

        model = self.model_ref()
        if model is None:
            return
        self.refresh_index()
        for module in model.modules():
            cls = type(module)
            patched = self._patched_classes.get(cls)
            if patched is None:
                original = cls.__setattr__
                patched = _PatchedClass(
                    original_setattr=original,
                    prepared_instances=weakref.WeakSet(),
                )
                self._patched_classes[cls] = patched
                cls.__setattr__ = _make_scoped_setattr(cls, original)  # type: ignore[assignment]
            patched.prepared_instances.add(module)
            patched.refcount += 1
            self._installed_classes.add(cls)

    def uninstall(self) -> None:
        """Restore class ``__setattr__`` methods whose session refcount reaches zero."""

        model = self.model_ref()
        modules = list(model.modules()) if model is not None else []
        for cls in list(self._installed_classes):
            patched = self._patched_classes.get(cls)
            if patched is None:
                continue
            for module in modules:
                if type(module) is cls:
                    patched.prepared_instances.discard(module)
                    patched.refcount = max(0, patched.refcount - 1)
            if patched.refcount == 0:
                cls.__setattr__ = patched.original_setattr  # type: ignore[assignment]
                del self._patched_classes[cls]
        self._installed_classes.clear()

    def refresh_index(self) -> None:
        """Refresh address, object, storage, version, and value snapshots."""

        model = self.model_ref()
        if model is None:
            return
        for module_address, module in _iter_modules_with_addresses(model):
            for name, tensor in module.named_buffers(recurse=False):
                if tensor is None or isinstance(tensor, nn.Parameter):
                    continue
                address = f"{module_address}.{name}" if module_address else name
                set_buffer_address(tensor, address)
                self.address_to_tensor[address] = tensor
                self.address_to_snapshot[address] = _copy_tensor_value(tensor)
                self.address_to_object_id[address] = id(tensor)
                self.address_to_storage_key[address] = storage_key(tensor)
                self.address_to_version[address] = _tensor_version(tensor)

    def record_reassignment(self, module: nn.Module, name: str, value: Any) -> None:
        """Record a registered-buffer reassignment performed via ``__setattr__``.

        Parameters
        ----------
        module:
            Module whose registered buffer was replaced.
        name:
            Buffer attribute name.
        value:
            Assigned value.
        """

        if not isinstance(value, torch.Tensor) or isinstance(value, nn.Parameter):
            return
        module_address = _module_address_from_meta(module)
        address = f"{module_address}.{name}" if module_address else name
        self._record_write(address, value, "reassign", get_tensor_label(value), True, None)

    def record_op_writes(
        self,
        func_name: str,
        snapshots: list[BufferSnapshot],
        producer_label_raw: str | None,
    ) -> None:
        """Record buffer writes detected by pre/post op snapshots.

        Parameters
        ----------
        func_name:
            Wrapped torch function name.
        snapshots:
            Pre-call buffer snapshots.
        producer_label_raw:
            Raw label of the operation that performed the write.
        """

        if not snapshots:
            return
        fused = _is_fused_mutator(func_name)
        fused_update = fused and _fused_update_mode(func_name)
        seen_addresses: set[str] = set()
        for snapshot in snapshots:
            if snapshot.address in seen_addresses:
                continue
            seen_addresses.add(snapshot.address)
            current = self.address_to_tensor.get(snapshot.address)
            if current is None or id(current) != snapshot.object_id:
                continue
            if storage_key(current) != snapshot.storage_key:
                continue
            current_value = _copy_tensor_value(current)
            value_changed = not _tensor_equal(snapshot.value, current_value)
            version_changed = (
                snapshot.version is not None
                and _tensor_version(current) is not None
                and snapshot.version != _tensor_version(current)
            )
            if fused:
                if fused_update:
                    self._record_write(
                        snapshot.address,
                        current,
                        "fused",
                        producer_label_raw,
                        value_changed,
                        func_name,
                    )
            elif value_changed or version_changed:
                self._record_write(
                    snapshot.address,
                    current,
                    "inplace",
                    producer_label_raw,
                    True,
                    func_name,
                )

    def reconcile(self) -> None:
        """Raise on unjournaled registered-buffer changes after forward capture."""

        model = self.model_ref()
        if model is None:
            return
        for module_address, module in _iter_modules_with_addresses(model):
            for name, tensor in module.named_buffers(recurse=False):
                if tensor is None or isinstance(tensor, nn.Parameter):
                    continue
                address = f"{module_address}.{name}" if module_address else name
                expected = self.address_to_snapshot.get(address)
                if expected is None:
                    continue
                current_value = _copy_tensor_value(tensor)
                object_changed = id(tensor) != self.address_to_object_id.get(address)
                storage_changed = storage_key(tensor) != self.address_to_storage_key.get(address)
                value_changed = not _tensor_equal(expected, current_value)
                if not (object_changed or storage_changed or value_changed):
                    continue
                producer = get_tensor_label(tensor)
                if producer is not None and not producer.startswith("buffer_"):
                    self._record_write(address, tensor, "reassign", producer, True, None)
                    continue
                raise RuntimeError(
                    "TorchLens detected an unjournaled registered-buffer change for "
                    f"'{address}'. Reassigning a buffer through '.data = tensor' is unsupported; "
                    "use 'self.buffer = tensor' or an in-place method such as copy_()."
                )

    def snapshot_buffer_args(self, tensors: list[torch.Tensor]) -> list[BufferSnapshot]:
        """Return pre-call snapshots for tensor args backed by registered buffers."""

        snapshots: list[BufferSnapshot] = []
        for tensor in tensors:
            if isinstance(tensor, nn.Parameter):
                continue
            address = _resolve_buffer_address(self, tensor)
            if address is None:
                continue
            registered = self.address_to_tensor[address]
            snapshots.append(
                BufferSnapshot(
                    address=address,
                    tensor=registered,
                    object_id=id(registered),
                    storage_key=storage_key(registered),
                    version=_tensor_version(registered),
                    value=_copy_tensor_value(registered),
                )
            )
        return snapshots

    def _record_write(
        self,
        address: str,
        value: torch.Tensor,
        kind: str,
        producer_label_raw: str | None,
        value_changed: bool,
        source_func_name: str | None,
    ) -> None:
        """Append one write event and advance the expected final snapshot."""

        copied_value = _copy_tensor_value(value)
        version_label = self._log_buffer_version_node(
            address,
            value,
            producer_label_raw,
            kind,
            value_changed,
            source_func_name,
        )
        event = BufferWriteEvent(
            address=address,
            kind=kind,  # type: ignore[arg-type]
            producer_label_raw=producer_label_raw,
            version_label_raw=version_label,
            value=copied_value,
            value_changed=value_changed,
            object_id=id(value),
            storage_key=storage_key(value),
            buffer_version=_tensor_version(value),
            source_func_name=source_func_name,
        )
        self.trace._buffer_write_events.append(event)
        self.address_to_tensor[address] = value
        self.address_to_snapshot[address] = copied_value
        self.address_to_object_id[address] = id(value)
        self.address_to_storage_key[address] = storage_key(value)
        self.address_to_version[address] = _tensor_version(value)
        set_buffer_address(value, address)

    def _log_buffer_version_node(
        self,
        address: str,
        value: torch.Tensor,
        producer_label_raw: str | None,
        kind: str,
        value_changed: bool,
        source_func_name: str | None,
    ) -> str | None:
        """Log the graph node representing one written buffer version."""

        if self.trace.capture_mode != "exhaustive":
            return None
        from .sources import log_source_tensor

        log_source_tensor(self.trace, value, "buffer", address)
        version_label = get_tensor_label(value)
        if version_label is None:
            return None
        fields = live_record_for_label(self.trace, version_label).fields
        fields["buffer_source"] = producer_label_raw
        fields["buffer_write_kind"] = kind
        fields["buffer_value_changed"] = value_changed
        fields["buffer_replay_validated"] = kind != "fused" or self.trace.save_arg_values
        fields["buffer_source_func_name"] = source_func_name
        if producer_label_raw is not None:
            _connect_live_buffer_version_parent(
                self.trace, fields, producer_label_raw, version_label
            )
        return version_label


def install_buffer_write_tracker(trace: "Trace", model: nn.Module) -> BufferWriteTracker:
    """Create and install the session buffer-write tracker."""

    tracker = BufferWriteTracker(trace, model)
    tracker.install()
    trace._buffer_write_tracker = tracker
    return tracker


def reconcile_buffer_writes(trace: "Trace") -> None:
    """Run the end-of-capture registered-buffer reconciliation diagnostic."""

    tracker = getattr(trace, "_buffer_write_tracker", None)
    if isinstance(tracker, BufferWriteTracker):
        tracker.reconcile()


def uninstall_buffer_write_tracker(trace: "Trace | None") -> None:
    """Uninstall a trace's session buffer-write tracker if present."""

    if trace is None:
        return
    tracker = getattr(trace, "_buffer_write_tracker", None)
    if isinstance(tracker, BufferWriteTracker):
        tracker.uninstall()
        trace._buffer_write_tracker = None


def snapshot_buffer_args(
    trace: "Trace",
    func_name: str,
    tensors: list[torch.Tensor],
) -> list[BufferSnapshot]:
    """Snapshot buffer-backed arguments for one wrapped torch call."""

    if trace.capture_mode != "exhaustive":
        return []
    tracker = getattr(trace, "_buffer_write_tracker", None)
    if not isinstance(tracker, BufferWriteTracker):
        return []
    if not _is_fused_mutator(func_name) and not _could_mutate(func_name):
        return []
    return tracker.snapshot_buffer_args(tensors)


def record_op_buffer_writes(
    trace: "Trace",
    func_name: str,
    snapshots: list[BufferSnapshot],
    producer_label_raw: str | None,
) -> None:
    """Record writes detected for one wrapped torch call."""

    tracker = getattr(trace, "_buffer_write_tracker", None)
    if isinstance(tracker, BufferWriteTracker):
        tracker.record_op_writes(func_name, snapshots, producer_label_raw)


def storage_key(tensor: torch.Tensor) -> tuple[Any, ...] | None:
    """Return a storage identity key guarded by object checks at use sites."""

    try:
        storage = tensor.untyped_storage()
        return (str(tensor.device), storage.data_ptr(), storage.nbytes())
    except Exception:
        return None


def _make_scoped_setattr(
    cls: type[nn.Module],
    original_setattr: Callable[[Any, str, Any], None],
) -> Callable[[Any, str, Any], None]:
    """Return a class-scoped ``__setattr__`` wrapper gated by prepared instances."""

    def scoped_setattr(self: nn.Module, name: str, value: Any) -> None:
        """Record registered-buffer replacement for prepared instances."""

        patched = BufferWriteTracker._patched_classes.get(cls)
        should_record = (
            patched is not None
            and self in patched.prepared_instances
            and _state._logging_enabled
            and _state._active_trace is not None
            and name in getattr(self, "_buffers", {})
            and isinstance(value, torch.Tensor)
        )
        original_setattr(self, name, value)
        if should_record:
            tracker = getattr(_state._active_trace, "_buffer_write_tracker", None)
            if isinstance(tracker, BufferWriteTracker):
                tracker.record_reassignment(self, name, value)

    return scoped_setattr


def _connect_live_buffer_version_parent(
    trace: "Trace",
    version_fields: dict[str, Any],
    producer_label_raw: str,
    version_label_raw: str,
) -> None:
    """Connect a live producer to a just-logged buffer-version node."""

    try:
        producer_fields = live_record_for_label(trace, producer_label_raw).fields
    except KeyError:
        return
    if producer_label_raw not in version_fields["parents"]:
        version_fields["parents"].append(producer_label_raw)
    version_fields["parent_arg_positions"]["args"][0] = producer_label_raw
    if version_label_raw not in producer_fields["children"]:
        producer_fields["children"].append(version_label_raw)
    producer_fields["has_children"] = True
    version_fields["has_input_ancestor"] = bool(producer_fields["has_input_ancestor"])
    version_fields["input_ancestors"].update(producer_fields["input_ancestors"])
    version_fields["root_ancestors"].discard(version_label_raw)
    version_fields["root_ancestors"].update(producer_fields["root_ancestors"])


def _iter_modules_with_addresses(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return model modules with TorchLens addresses."""

    modules: list[tuple[str, nn.Module]] = []
    for module in model.modules():
        modules.append((_module_address_from_meta(module), module))
    return modules


def _module_address_from_meta(module: nn.Module) -> str:
    """Return the TorchLens module address stored during model prep."""

    meta = getattr(module, "_tl", None)
    address = getattr(meta, "address", None)
    if address in {None, "self"}:
        return ""
    return str(address)


def _copy_tensor_value(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached clone suitable for write-event storage."""

    with _state.pause_logging():
        return safe_copy(tensor, detach_tensor=True)


def _tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two tensors have identical values."""

    with _state.pause_logging():
        try:
            return bool(torch.equal(left, right))
        except Exception:
            return False


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Return PyTorch's internal tensor version counter when available."""

    return getattr(tensor, "_version", None)


def _resolve_buffer_address(
    tracker: BufferWriteTracker,
    tensor: torch.Tensor,
) -> str | None:
    """Resolve a tensor/view/data tensor to a registered-buffer address."""

    direct = get_buffer_address(tensor)
    if direct in tracker.address_to_tensor:
        return direct
    key = storage_key(tensor)
    if key is None:
        return None
    tensor_start, tensor_end = _storage_range(tensor)
    for address, registered in tracker.address_to_tensor.items():
        if storage_key(registered) != key:
            continue
        reg_start, reg_end = _storage_range(registered)
        if tensor_start >= reg_start and tensor_end <= reg_end:
            return address
    return None


def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return byte range occupied by a tensor inside its storage."""

    element_size = tensor.element_size()
    start = int(tensor.storage_offset()) * element_size
    end = start + int(tensor.numel()) * element_size
    return start, end


def _could_mutate(func_name: str) -> bool:
    """Return whether a torch wrapper name can mutate tensor storage."""

    return (
        func_name.endswith("_")
        or func_name.startswith("__i")
        or func_name
        in {
            "__setitem__",
            "__delitem__",
        }
    )


def _is_fused_mutator(func_name: str) -> bool:
    """Return whether a torch function is a known fused/native buffer mutator."""

    return func_name in _FUSED_MUTATOR_NAMES


def _fused_update_mode(func_name: str) -> bool:
    """Return whether a known fused mutator should emit op-execution write events."""

    return func_name in _FUSED_MUTATOR_NAMES

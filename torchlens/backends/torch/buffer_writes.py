"""Registered-buffer write capture for the torch backend."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import torch

from ...utils._torch_compat import tensor_version_or_none
from torch import nn

from ... import _state
from ...ir import BufferWriteEvent
from ...utils.tensor_utils import safe_copy
from ._tl import (
    clear_tensor_label,
    get_buffer_address,
    get_module_meta,
    get_tensor_label,
    set_buffer_address,
)

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
class _SessionBufferStamp:
    """Current-session identity record for one buffer-stamped tensor (r81).

    The strong ``tensor`` reference keeps ``id(tensor)`` stable for the session
    and lets cleanup clear the stamp even when the object is unreachable from
    the model tree. The strong ``storage`` reference (the ``UntypedStorage``
    held at stamp time) pins the stamped storage alive so its ``data_ptr`` can
    never be recycled during the session: a live-vs-keeper pointer match is
    therefore a true storage-identity proof, not a heuristic.

    DO NOT "OPTIMISE" THE STRONG REFERENCES AWAY (r83 S2). Holding the tensor
    AND its stamp-time storage roughly DOUBLES buffer-write retention -- one
    entry per write, measured at ~385 MiB RSS delta over a 100-write stress on
    a 4 MiB buffer, on top of the O(writes x buffer_size) the write journal
    already retained. That cost is the price of CORRECTNESS, not an oversight.
    A weakref-only design would let a rebound-and-freed storage's ``data_ptr``
    be recycled by its replacement, so the pointer comparison in
    :func:`session_validated_buffer_address` would produce a FALSE identity
    match and re-open the r80 F2 launder (a plain-attr buffer ``.data``-rebound
    to input-derived data mid-forward, silently reported ``VERIFIED``). If a
    future memory pass must reduce this, the only safe direction is keeping the
    keeper for the CURRENT live registration per address and letting superseded
    objects fail validation -- strictly more conservative, at the cost of a
    possible over-trigger for an op holding a superseded buffer version. Any
    change here must re-derive the r80 F2 red first.

    Retention is bounded to the session: both inventories and this registry are
    ``FieldPolicy.DROP`` and are emptied by ``_cleanup_model_session``, verified
    empty after every capture configuration tested including a raising forward.
    """

    tensor: torch.Tensor
    address: str
    storage: Any | None


def register_session_buffer_stamp(trace: "Trace", value: torch.Tensor, address: str) -> None:
    """Stamp a buffer address and record CURRENT-SESSION identity for it (r81).

    Every path that writes a session-scoped buffer provenance stamp MUST route
    through this helper (r80 F1 root A: ``_record_write`` stamped values without
    inventorying them, so the r79 inventory cleanup could not clear them and the
    stale stamp survived into later captures). The helper:

    1. writes the ``TensorMeta.address`` stamp (``set_buffer_address``);
    2. appends the object to ``trace._session_buffer_inventory`` (once per
       object) so the inventory-driven cleanup always clears it; and
    3. records the object + address + live storage keeper in
       ``trace._session_buffer_identity`` -- the registry consulted by
       :func:`session_validated_buffer_address`, the buffer twin of the param
       rung's ``_param_ref is value`` identity belt.

    Parameters
    ----------
    trace:
        Active capture trace owning the session registries.
    value:
        Tensor receiving the buffer stamp.
    address:
        Dotted buffer address being stamped.
    """

    set_buffer_address(value, address)
    registry = getattr(trace, "_session_buffer_identity", None)
    if registry is None:
        registry = {}
        trace._session_buffer_identity = registry
    inventory = getattr(trace, "_session_buffer_inventory", None)
    if inventory is None:
        inventory = []
        trace._session_buffer_inventory = inventory
    existing = registry.get(id(value))
    if existing is None or existing.tensor is not value:
        inventory.append(value)
    storage: Any | None = None
    try:
        with _state.pause_logging():
            storage = value.untyped_storage()
    except Exception:
        storage = None
    registry[id(value)] = _SessionBufferStamp(tensor=value, address=address, storage=storage)


def session_validated_buffer_address(trace: "Trace", value: torch.Tensor) -> str | None:
    """Return a buffer stamp ONLY when it is current-session with live storage identity.

    The buffer/tensor-meta provenance belt (r81, r80 F1+F2 shared root): a
    static ``TensorMeta.address`` stamp is trusted only when ALL hold:

    1. the exact object was stamped during THIS capture session (it resolves in
       ``trace._session_buffer_identity`` by identity -- a stale cross-capture
       leftover never resolves, closing F1 even if a future tagging path
       escapes the inventory cleanup again); and
    2. the receiver's LIVE storage is still the storage held at stamp time
       (pointer/extent/device match against the pinned keeper -- a mid-forward
       ``q.data = <input-derived>`` rebind fails the match, closing F2's
       input-layout launder through a legitimately stamped plain-attr buffer).

    Storage-identity is fail-closed: if the keeper or the live storage is
    inaccessible on exactly one side, the stamp is NOT validated. Only the
    symmetric-inaccessible case (exotic tensors whose storage cannot be read at
    stamp time or now) falls back to pure object identity.

    Parameters
    ----------
    trace:
        Active capture trace owning the session registry.
    value:
        Tensor whose direct buffer stamp should be validated.

    Returns
    -------
    str | None
        The validated current-session buffer address, else ``None``.
    """

    address = get_buffer_address(value)
    if address is None:
        return None
    registry = getattr(trace, "_session_buffer_identity", None)
    if not registry:
        return None
    entry = registry.get(id(value))
    if entry is None or entry.tensor is not value or entry.address != address:
        return None
    live: Any | None = None
    try:
        with _state.pause_logging():
            live = value.untyped_storage()
    except Exception:
        live = None
    keeper = entry.storage
    if keeper is None or live is None:
        return address if (keeper is None and live is None) else None
    try:
        with _state.pause_logging():
            same = (
                live.data_ptr() == keeper.data_ptr()
                and live.nbytes() == keeper.nbytes()
                and str(live.device) == str(keeper.device)
            )
    except Exception:
        return None
    return address if same else None


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
        # r19-B: JOURNAL-EXPECTED buffer value per address. Seeded at forward START to the
        # pre-forward snapshot and advanced ONLY by a tracked/journaled in-place op (to its
        # post-op value). Unlike ``address_to_snapshot`` -- which a journaled write refreshes
        # from the buffer's ACTUAL (possibly host-contaminated) bytes -- this expected value
        # tracks exactly what the JOURNALED ops should produce from the pre-forward state. A
        # tracked in-place op whose PRE-op bytes diverge from ``expected`` reveals an untracked
        # host write that the op's version bump would otherwise MASK (r19-B): fail closed.
        self.address_to_expected_snapshot: dict[str, torch.Tensor] = {}
        self.address_to_expected_storage_snapshot: dict[str, torch.Tensor] = {}
        self.storage_key_to_addresses: dict[tuple[Any, ...], dict[str, None]] = {}
        self._storage_key_cache: dict[tuple[int, int | None], tuple[Any, ...] | None] = {}
        self._storage_range_cache: dict[tuple[int, int | None], tuple[int, int]] = {}
        self._installed_classes: set[type[nn.Module]] = set()
        # r18: named-PARAMETER whole-storage byte + version baselines, snapshotted at
        # forward START and compared at forward END. Params carry NO graph source node
        # (unlike buffers), so a host write through a pre-forward-acquired zero-copy alias
        # (``self.w.detach().numpy()[0] += 1``) is invisible to every op census and to the
        # embedded pre-forward state snapshot -- the exact buffer host-write-back tripwire,
        # mirrored onto params. address -> (whole-storage uint8 byte clone, tensor version).
        self.address_to_param_snapshot: dict[str, tuple[torch.Tensor, int | None]] = {}
        self.address_to_param_tensor: dict[str, torch.Tensor] = {}

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
        persistent_state_names = frozenset(model.state_dict())
        # r63 C1: storage-pointer -> buffer address index, the buffer twin of
        # ``_param_storage_addresses``. A registered-buffer STORAGE ALIAS (``self.b.data``,
        # a derived view) carries no buffer meta, so a metadata read routed through it
        # could not be attributed to its state slot without this index.
        buffer_storage_addresses: dict[int, str] = {}
        for module_address, module in _iter_modules_with_addresses(model):
            for name, tensor in module.named_buffers(recurse=False):
                if tensor is None or isinstance(tensor, nn.Parameter):
                    continue
                address = f"{module_address}.{name}" if module_address else name
                # r81: route the re-stamp through the session registry so the
                # identity belt stays coherent with the stamped address (e.g. a
                # double-registered alias whose last-visited name wins here).
                register_session_buffer_stamp(self.trace, tensor, address)
                try:
                    with _state.pause_logging():
                        buffer_storage_addresses[tensor.untyped_storage().data_ptr()] = address
                except (RuntimeError, TypeError, NotImplementedError):
                    pass
                pre_forward_value = _copy_tensor_value(tensor)
                self.trace._buffer_initial_values.setdefault(address, pre_forward_value)
                persistence = self.trace.__dict__.setdefault("_buffer_persistence", {})
                persistence[address] = address in persistent_state_names
                self._register_address(address, tensor, _copy_tensor_value(tensor))
                # r19-B: seed the journal-expected value to the pre-forward snapshot; only a
                # journaled in-place op advances it (see ``_advance_expected_after_journal``).
                self.address_to_expected_snapshot.setdefault(address, pre_forward_value)
                try:
                    with _state.pause_logging():
                        expected_storage = _whole_storage_uint8(tensor).clone()
                except (RuntimeError, TypeError, NotImplementedError):
                    with _state.pause_logging():
                        expected_storage = _whole_storage_uint8(pre_forward_value).clone()
                self.address_to_expected_storage_snapshot.setdefault(address, expected_storage)
        self.trace.__dict__["_buffer_storage_addresses"] = buffer_storage_addresses
        self._refresh_param_index(model)

    def _refresh_param_index(self, model: nn.Module) -> None:
        """Snapshot named-parameter whole-storage bytes + versions at forward start (r18).

        Mirrors the buffer baseline for parameters so an untracked/invisible host write into
        a param's storage during the forward is caught at forward end (F2), and builds a
        storage-pointer -> address index so a READ-ONLY param host escape resolves through the
        state-digest net exactly like the buffer twin (F3) instead of failing closed. Purely a
        diagnostic baseline: it logs no graph node, so captured goldens are byte-unchanged.
        """

        # storage data_ptr -> param address, consumed by the completeness witness to resolve a
        # read-only param host escape (``self.w.detach().numpy().sum()``) by its state slot.
        param_storage_addresses: dict[int, str] = {}
        with _state.pause_logging():
            for module_address, module in _iter_modules_with_addresses(model):
                for name, tensor in module.named_parameters(recurse=False):
                    if tensor is None:
                        continue
                    address = f"{module_address}.{name}" if module_address else name
                    if address in self.address_to_param_snapshot:
                        continue
                    try:
                        before = _whole_storage_uint8(tensor).clone()
                    except (RuntimeError, TypeError, NotImplementedError):
                        continue
                    self.address_to_param_snapshot[address] = (before, _tensor_version(tensor))
                    self.address_to_param_tensor[address] = tensor
                    try:
                        param_storage_addresses[tensor.untyped_storage().data_ptr()] = address
                    except (RuntimeError, TypeError, NotImplementedError):
                        continue
        self.trace.__dict__["_param_storage_addresses"] = param_storage_addresses

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
            if self.storage_key(current) != snapshot.storage_key:
                continue
            current_value = _copy_tensor_value(current)
            value_changed = not _tensor_equal(snapshot.value, current_value)
            version_changed = (
                snapshot.version is not None
                and _tensor_version(current) is not None
                and snapshot.version != _tensor_version(current)
            )
            journaled = False
            if fused:
                if fused_update:
                    journaled = True
                    self._record_write(
                        snapshot.address,
                        current,
                        "fused",
                        producer_label_raw,
                        value_changed,
                        func_name,
                    )
            elif value_changed or version_changed:
                journaled = True
                self._record_write(
                    snapshot.address,
                    current,
                    "inplace",
                    producer_label_raw,
                    True,
                    func_name,
                )
            if journaled:
                # r19-B: reconcile this journaled op's PRE-op bytes against the journal-expected
                # bytes before advancing expected to the post-op value. A divergence means an
                # untracked host write landed on the buffer's storage before this op and would be
                # MASKED by the op's version bump (``self.npb[0] += 10`` then ``self.b.add_(1.0)``).
                self._reconcile_journaled_buffer(snapshot, current_value)

    def _reconcile_journaled_buffer(
        self, snapshot: BufferSnapshot, current_value: torch.Tensor
    ) -> None:
        """Catch a host write masked by a journaled buffer op, then advance the expected value.

        A buffer, unlike a param, IS journaled and replayed, so a PURE journaled in-place update
        (a BatchNorm running-stat step, a plain ``self.b.add_(1.0)``) MUST stay VERIFIED. The hole
        (r19-B) is an untracked HOST write into the buffer's storage that a subsequent journaled op
        MASKS: the journaled op refreshes ``address_to_snapshot`` from the buffer's ACTUAL bytes, so
        the end-of-forward value-change check sees no discrepancy and false-VERIFIES.

        The fix reconciles the buffer's PRE-op bytes against the JOURNAL-EXPECTED bytes -- the value
        the tracked ops produce from the pre-forward snapshot. For a pure journaled update the pre-op
        bytes equal ``expected`` (no host write) -> stay VERIFIED; a host write before the op makes
        the pre-op bytes diverge from ``expected`` -> fail closed to UNVERIFIABLE. Either way expected
        is advanced to the post-op value so the NEXT journaled op reconciles against a fresh baseline.
        """

        from .completeness_witness import _HOST_ESCAPE_MUTABLE_WRITEBACK

        expected = self.address_to_expected_snapshot.get(snapshot.address)
        if expected is not None and not _tensor_equal(expected, snapshot.value):
            _HOST_ESCAPE_MUTABLE_WRITEBACK.add(self.trace)
        self.address_to_expected_snapshot[snapshot.address] = current_value
        try:
            with _state.pause_logging():
                self.address_to_expected_storage_snapshot[snapshot.address] = _whole_storage_uint8(
                    snapshot.tensor
                ).clone()
        except (RuntimeError, TypeError, NotImplementedError):
            with _state.pause_logging():
                self.address_to_expected_storage_snapshot[snapshot.address] = _whole_storage_uint8(
                    current_value
                ).clone()

    def reconcile(self) -> None:
        """Reconcile registered-buffer changes after forward capture.

        A journalled write is already accounted; an object/storage reassignment is recorded here.
        A zero-copy HOST write-back into a buffer's existing storage (r15-C2) leaves object and
        storage identity unchanged with the value changed and no journal entry: it is flagged as
        an opaque host write-back so the run reports UNVERIFIABLE, never crashing capture.
        """

        # Imported lazily to avoid an import cycle with the completeness-witness dispatcher.
        from .completeness_witness import _HOST_ESCAPE_MUTABLE_WRITEBACK

        model = self.model_ref()
        if model is None:
            return
        self._reconcile_params(model)
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
                if not object_changed and storage_changed:
                    self._record_write(address, tensor, "data_reassign", None, value_changed, None)
                    continue
                # Same object, same storage, changed value, no journal entry (r15-C2): a zero-copy
                # HOST write-back into the buffer's existing storage -- ``self.b.detach().numpy()[0]
                # = 99`` / a raw ``data_ptr`` / storage ``__setitem__`` write. The mutation bypassed
                # every aten dispatch and version bump, so the sparse DAG cannot model it and the
                # replay recomputes the pre-write value. This is an opaque host write-back exactly
                # like the activation-alias case: flag it so the run reports UNVERIFIABLE, and do
                # NOT crash capture. Recording it as ``data_reassign`` keeps the buffer journal's
                # expected-final snapshot consistent for any downstream buffer.
                _HOST_ESCAPE_MUTABLE_WRITEBACK.add(self.trace)
                self._record_write(address, tensor, "data_reassign", None, value_changed, None)
                continue

    def _reconcile_params(self, model: nn.Module) -> None:
        """Flag ANY within-forward PARAMETER storage-byte change (r18 + r19-A, fail closed).

        Compares each named parameter's whole-storage bytes against its forward-START baseline.
        ANY within-forward byte change makes the param unfaithfully replayable and is flagged so
        the run reports UNVERIFIABLE instead of a false VERIFIED.

        r19-A closes a version-mask hole: unlike a BUFFER, a PARAMETER carries NO graph source node
        and its in-place ops are NOT journaled/captured in the replayable DAG (params are excluded
        from wrapper output logging), so the embedded pre-forward param state can never reproduce a
        within-forward param mutation. A VERSION BUMP therefore does NOT mean the write replays --
        a direct in-place aten op on a param (``with torch.no_grad(): self.w.add_(1.0)``), OR a
        zero-copy host write followed by a version-bumping op, both leave the param bytes changed
        while the DAG cannot model them. So the byte diff is checked VERSION-AGNOSTICALLY: whether
        the version bumped OR stayed static, any net whole-storage change fails closed. (The pre-r19
        code EXEMPTED version-bumped params on the false premise "version bump => the write is in the
        DAG"; that premise holds for buffers, which ARE journaled, but NOT for params.) Read-only
        param access leaves the bytes unchanged and stays VERIFIED, so there is ~zero over-trigger on
        ordinary Linear/Conv/MLP/BatchNorm models (their params are untouched during the forward).
        """

        from .completeness_witness import _HOST_ESCAPE_MUTABLE_WRITEBACK

        if not self.address_to_param_snapshot:
            return
        with _state.pause_logging():
            for module_address, module in _iter_modules_with_addresses(model):
                for name, tensor in module.named_parameters(recurse=False):
                    if tensor is None:
                        continue
                    address = f"{module_address}.{name}" if module_address else name
                    baseline = self.address_to_param_snapshot.get(address)
                    if baseline is None:
                        continue
                    before, _before_version = baseline
                    # VERSION-AGNOSTIC: a param whose whole-storage bytes changed during the forward
                    # -- through ANY path, a tracked in-place aten op OR an untracked host write --
                    # is not replayable from the embedded pre-forward state, because param in-place
                    # ops are not captured in the DAG. Fail closed regardless of the version counter.
                    try:
                        after = _whole_storage_uint8(tensor)
                    except (RuntimeError, TypeError, NotImplementedError):
                        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(self.trace)
                        continue
                    try:
                        unchanged = bool(torch.equal(after, before))
                    except (RuntimeError, TypeError, NotImplementedError):
                        unchanged = False
                    if not unchanged:
                        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(self.trace)

    def snapshot_buffer_args(self, tensors: list[torch.Tensor]) -> list[BufferSnapshot]:
        """Return pre-call snapshots for tensor args backed by registered buffers."""

        self.clear_storage_metadata_cache()
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
                    storage_key=self.storage_key(registered),
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

        self.clear_storage_metadata_cache()
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
            storage_key=self.storage_key(value),
            buffer_version=_tensor_version(value),
            source_func_name=source_func_name,
        )
        self.trace._buffer_write_events.append(event)
        self._register_address(address, value, copied_value)
        # r81 (r80 F1 root A): the written/reassigned value's stamp MUST be
        # inventoried + identity-registered, not bare-stamped -- a reassigned
        # external popped from ``_buffers`` escaped every cleanup walk and its
        # surviving stamp resurrected the stale-provenance false VERIFIED.
        register_session_buffer_stamp(self.trace, value, address)
        self._refresh_overlapping_alias_snapshots(address)

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
        return version_label

    def _refresh_overlapping_alias_snapshots(self, written_address: str) -> None:
        """Refresh final snapshots for aliases changed by a journaled write."""

        written_tensor = self.address_to_tensor.get(written_address)
        if written_tensor is None:
            return
        written_key = self.storage_key(written_tensor)
        if written_key is None:
            return
        written_range = self.storage_range(written_tensor)
        for address, tensor in tuple(self.address_to_tensor.items()):
            if address == written_address:
                continue
            if tensor is None:
                continue
            if self.storage_key(tensor) != written_key:
                continue
            if not _ranges_overlap(written_range, self.storage_range(tensor)):
                continue
            clear_tensor_label(tensor)
            alias_value = _copy_tensor_value(tensor)
            self.address_to_snapshot[address] = alias_value
            self.address_to_version[address] = _tensor_version(tensor)
            # r19-B: a journaled write to an overlapping buffer also advances this alias's
            # journal-expected value, so a later journaled op on the alias reconciles against
            # the tracked post-write bytes (not a stale pre-forward baseline).
            if address in self.address_to_expected_snapshot:
                self.address_to_expected_snapshot[address] = alias_value
            if address in self.address_to_expected_storage_snapshot:
                try:
                    with _state.pause_logging():
                        self.address_to_expected_storage_snapshot[address] = _whole_storage_uint8(
                            tensor
                        ).clone()
                except (RuntimeError, TypeError, NotImplementedError):
                    with _state.pause_logging():
                        self.address_to_expected_storage_snapshot[address] = _whole_storage_uint8(
                            alias_value
                        ).clone()

    def storage_key(self, tensor: torch.Tensor) -> tuple[Any, ...] | None:
        """Return a cached storage identity key for ``tensor``.

        Parameters
        ----------
        tensor:
            Tensor whose backing storage should be identified.

        Returns
        -------
        tuple[Any, ...] | None
            Storage identity key, or ``None`` when storage access fails.
        """

        cache_key = (id(tensor), _tensor_version(tensor))
        if cache_key not in self._storage_key_cache:
            self._storage_key_cache[cache_key] = storage_key(tensor)
        return self._storage_key_cache[cache_key]

    def storage_range(self, tensor: torch.Tensor) -> tuple[int, int]:
        """Return a cached byte range for ``tensor`` within its storage.

        Parameters
        ----------
        tensor:
            Tensor whose storage span should be identified.

        Returns
        -------
        tuple[int, int]
            Half-open byte range occupied by the tensor.
        """

        cache_key = (id(tensor), _tensor_version(tensor))
        if cache_key not in self._storage_range_cache:
            self._storage_range_cache[cache_key] = _storage_range(tensor)
        return self._storage_range_cache[cache_key]

    def clear_storage_metadata_cache(self) -> None:
        """Clear cached tensor storage metadata for the current operation boundary."""

        self._storage_key_cache.clear()
        self._storage_range_cache.clear()

    def addresses_for_storage_key(self, key: tuple[Any, ...]) -> tuple[str, ...]:
        """Return registered buffer addresses backed by ``key``.

        Parameters
        ----------
        key:
            Storage identity key to look up.

        Returns
        -------
        tuple[str, ...]
            Registered addresses sharing the same storage key.
        """

        return tuple(self.storage_key_to_addresses.get(key, {}))

    def _register_address(
        self,
        address: str,
        tensor: torch.Tensor,
        snapshot: torch.Tensor,
    ) -> None:
        """Register or update one buffer address and its storage index entry.

        Parameters
        ----------
        address:
            Dotted module buffer address.
        tensor:
            Live tensor assigned to the address.
        snapshot:
            Detached value snapshot for the address.
        """

        old_key = self.address_to_storage_key.get(address)
        if old_key is not None:
            old_addresses = self.storage_key_to_addresses.get(old_key)
            if old_addresses is not None:
                old_addresses.pop(address, None)
                if not old_addresses:
                    del self.storage_key_to_addresses[old_key]
        new_key = self.storage_key(tensor)
        self.address_to_tensor[address] = tensor
        self.address_to_snapshot[address] = snapshot
        self.address_to_object_id[address] = id(tensor)
        self.address_to_storage_key[address] = new_key
        self.address_to_version[address] = _tensor_version(tensor)
        if new_key is not None:
            self.storage_key_to_addresses.setdefault(new_key, {})[address] = None


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
    kwargs: dict[str, Any],
) -> list[BufferSnapshot]:
    """Snapshot buffer-backed arguments for one wrapped torch call."""

    if trace.capture_mode != "exhaustive":
        return []
    tracker = getattr(trace, "_buffer_write_tracker", None)
    if not isinstance(tracker, BufferWriteTracker):
        return []
    if not _is_fused_mutator(func_name) and not _could_mutate(func_name, kwargs):
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


def resolve_registered_buffer_address(trace: "Trace", tensor: torch.Tensor) -> str | None:
    """Resolve an actual tensor argument to a registered-buffer address.

    Parameters
    ----------
    trace:
        Active trace whose buffer-write tracker owns the registered-buffer index.
    tensor:
        Tensor argument observed by a wrapped torch call.

    Returns
    -------
    str | None
        Registered-buffer address when the tensor aliases a tracked buffer,
        otherwise ``None``.
    """

    tracker = getattr(trace, "_buffer_write_tracker", None)
    if not isinstance(tracker, BufferWriteTracker):
        # r81: no tracker (non-exhaustive session) -- never trust the raw
        # static stamp; require current-session identity + storage identity.
        return session_validated_buffer_address(trace, tensor)
    return _resolve_buffer_address(tracker, tensor)


def storage_key(tensor: torch.Tensor) -> tuple[Any, ...] | None:
    """Return a storage identity key guarded by object checks at use sites."""

    try:
        with _state.pause_logging():
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


def _iter_modules_with_addresses(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return model modules with TorchLens addresses."""

    modules: list[tuple[str, nn.Module]] = []
    for module in model.modules():
        modules.append((_module_address_from_meta(module), module))
    return modules


def _module_address_from_meta(module: nn.Module) -> str:
    """Return the TorchLens module address stored during model prep."""

    meta = get_module_meta(module)
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
            if bool(torch.equal(left, right)):
                return True
            if (
                left.shape != right.shape
                or left.dtype != right.dtype
                or left.device != right.device
            ):
                return False
            if not (torch.is_floating_point(left) or torch.is_complex(left)):
                return False
            return bool(torch.all(torch.eq(left, right) | (torch.isnan(left) & torch.isnan(right))))
        except Exception:
            return False


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Return PyTorch's internal tensor version counter when available.

    ``_version`` is a WITNESSED input-metadata property (r33): the completeness-witness
    scoped patch records a genuine USER ``x._version`` read on a model-input leaf so a
    reversible in-place bump (identical bytes, different version) fails closed instead of
    a false VERIFIED. This buffer-tracking read is TorchLens's OWN bookkeeping on EVERY op
    tensor operand -- including input leaves -- so it must be marked internal or it records
    a spurious ``_version`` fact for every model and falsely diverges any runtime input
    whose version counter differs from capture (an over-trigger). The per-thread depth
    counter is bumped inline (not via the ``internal_scalar_read`` generator) to keep this
    hot per-op read cheap.
    """

    state = _witness_marker_state()
    if state is None:
        return tensor_version_or_none(tensor)
    depth = getattr(state, "depth", 0)
    state.depth = depth + 1
    try:
        return tensor_version_or_none(tensor)
    finally:
        state.depth = depth


_WITNESS_MARKER_STATE: Any = None


def _witness_marker_state() -> Any:
    """Lazily fetch the completeness-witness per-thread internal-read marker state.

    Imported lazily (a module-level import would cycle: ``completeness_witness`` and
    ``buffer_writes`` are mutually dependent) and cached after first resolution.
    """

    global _WITNESS_MARKER_STATE
    if _WITNESS_MARKER_STATE is None:
        from .completeness_witness import _internal_read_state as marker_state

        _WITNESS_MARKER_STATE = marker_state
    return _WITNESS_MARKER_STATE


def _whole_storage_uint8(source: torch.Tensor) -> torch.Tensor:
    """Return a ``uint8`` view over ``source``'s ENTIRE untyped storage (all bytes).

    A zero-copy host alias (``source.detach().numpy()`` / ``untyped_storage()`` / a raw
    ``data_ptr``) shares the WHOLE storage, so a host ``__setitem__`` / ``np.as_strided`` write can
    land on bytes OUTSIDE ``source``'s element extent. Comparing the whole storage (not
    ``source``'s own extent) makes ANY host write in the shared storage detectable at forward end.
    Matches ``completeness_witness._whole_storage_uint8`` for the buffer/param write tripwire.
    """

    untyped = source.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=source.device)
    view.set_(untyped, 0, (untyped.nbytes(),), (1,))
    return view


def _resolve_buffer_address(
    tracker: BufferWriteTracker,
    tensor: torch.Tensor,
) -> str | None:
    """Resolve a tensor/view/data tensor to a registered-buffer address."""

    direct = get_buffer_address(tensor)
    # r81: the direct-stamp fast path resolves ONLY for the object that IS the
    # current-session registered tensor at that address. A stale/foreign stamp
    # whose address merely collides with a registered name falls through to the
    # storage-key resolution below, which is anchored on live storage identity.
    if direct is not None and tracker.address_to_tensor.get(direct) is tensor:
        return direct
    key = tracker.storage_key(tensor)
    if key is None:
        return None
    if key not in tracker.storage_key_to_addresses:
        return None
    tensor_start, tensor_end = tracker.storage_range(tensor)
    for address in tracker.addresses_for_storage_key(key):
        registered = tracker.address_to_tensor.get(address)
        if registered is None:
            continue
        reg_start, reg_end = tracker.storage_range(registered)
        if tensor_start >= reg_start and tensor_end <= reg_end:
            return address
    return None


def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    """Return byte range occupied by a tensor inside its storage."""

    with _state.pause_logging():
        element_size = tensor.element_size()
        start = int(tensor.storage_offset()) * element_size
        end = start + int(tensor.numel()) * element_size
    return start, end


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return whether two half-open byte ranges overlap."""

    return left[0] < right[1] and right[0] < left[1]


def _could_mutate(func_name: str, kwargs: dict[str, Any] | None = None) -> bool:
    """Return whether a torch wrapper name can mutate tensor storage."""

    return (
        func_name.endswith("_")
        or func_name.startswith("__i")
        or (kwargs is not None and kwargs.get("out") is not None)
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

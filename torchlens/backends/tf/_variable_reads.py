"""TensorFlow variable reads verified as parameter sources (ruling X4).

An eager ``ReadVariableOp`` is not a function of recorded parents: its value is
the variable. Validation therefore counts such a read as a parameter source,
the way PyTorch treats parameters as leaves, only when all of these hold:

* the read's resource handle resolves to a variable of the discovered module
  tree, whose value was snapshotted before the forward;
* no op other than ``ReadVariableOp`` consumed that handle during the forward
  (any ``Assign*``, optimizer update, moving-statistics write or function call
  taking the handle counts as a possible write);
* the forward ran no graph-mode or function-call op (an opaque region);
* the read's saved payload equals the snapshot bit for bit (dtype, shape and a
  SHA-256 digest of the raw bytes).

A read meeting the first three conditions whose payload differs FAILS. Any other
read keeps the ``effectful_region`` decision it had before. Repeated reads of one
variable compare with the same snapshot. Downstream ops keep full replay and
perturbation in the validator.

The capture enters :func:`variable_read_scope` around the forward and hands the
ledger to :func:`attach_variable_reads`; the validator asks
:func:`parameter_read_decision`. A trace without a ledger (static FuncGraph
capture, a capture made without the scope) verifies no read.

A saved trace keeps, per verifiable read, the snapshot it binds to
(:func:`persisted_read`, stored in the read's replay recipe); after load
:func:`restore_variable_reads` rebuilds a ledger keyed by read label, so the
loaded payload is compared with the same snapshot and the verdict equals the
live one. A loaded trace without that record verifies no read.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._validation_shared import replay_payload
from ._tensor_probe import _tensor_ref_key
from ._tf_compat import get_op_callbacks_module

_READ_OP = "ReadVariableOp"
#: Op types that run a function body the eager callback cannot see into.
_CALL_OPS = frozenset({"StatefulPartitionedCall", "PartitionedCall"})


@dataclass(frozen=True)
class VariableSnapshot:
    """Pre-forward value of one discovered variable, by content digest.

    Parameters
    ----------
    address
        Parameter address of the variable in the discovered module tree.
    dtype
        NumPy dtype name of the value.
    shape
        Shape of the value.
    digest
        SHA-256 hex digest of the value's C-contiguous raw bytes.
    """

    address: str
    dtype: str
    shape: tuple[int, ...]
    digest: str


@dataclass
class VariableReadLedger:
    """What one eager forward did to the discovered variables.

    Parameters
    ----------
    snapshots
        Pre-forward snapshot by resource-handle reference key.
    touched
        First non-read op type that consumed each handle, by reference key.
    opaque
        The first graph-mode or function-call op seen, or ``None``.
    by_label
        For a loaded trace: the snapshot each read was verifiable against at
        capture, by the read's raw label (written by :func:`restore_variable_reads`).
    """

    snapshots: dict[object, VariableSnapshot] = field(default_factory=dict)
    touched: dict[object, str] = field(default_factory=dict)
    opaque: str | None = None
    by_label: dict[str, VariableSnapshot] = field(default_factory=dict)


_LEDGERS: weakref.WeakKeyDictionary[Any, VariableReadLedger] = weakref.WeakKeyDictionary()


def _digest(value: np.ndarray) -> str:
    """SHA-256 of an array's C-contiguous raw bytes."""

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _variable_handle(variable: Any) -> Any:
    """The resource handle of a ``tf.Variable`` or a Keras 3 variable, else ``None``."""

    for holder in (variable, getattr(variable, "_value", None)):
        if holder is None:
            continue
        try:
            handle = holder.handle
        except (AttributeError, ValueError, TypeError):
            continue
        if handle is not None:
            return handle
    return None


def _variable_value(variable: Any) -> np.ndarray | None:
    """The variable's current value as NumPy, or ``None`` when it cannot be read.

    An unreadable variable (uninitialized, or a type without a NumPy view) gets no
    snapshot, so every read of it stays ``effectful_region`` unverified.
    """

    numpy_method = getattr(variable, "numpy", None)
    if not callable(numpy_method):
        return None
    try:
        return np.asarray(numpy_method())
    except (ValueError, TypeError, RuntimeError):
        return None


def _snapshot_variables(module_tree: Any) -> dict[object, VariableSnapshot]:
    """Snapshot every discovered variable that exposes a resource handle."""

    snapshots: dict[object, VariableSnapshot] = {}
    addresses: Mapping[int, str] = getattr(module_tree, "param_address_by_id", {})
    for variable_id, variable in getattr(module_tree, "variables_by_id", {}).items():
        address = addresses.get(variable_id)
        key = _tensor_ref_key(_variable_handle(variable))
        value = _variable_value(variable)
        if address is None or key is None or value is None:
            continue
        snapshots[key] = VariableSnapshot(
            address=address, dtype=value.dtype.name, shape=value.shape, digest=_digest(value)
        )
    return snapshots


def _op_type_text(op_type: Any) -> str:
    """Callback op type as text (TensorFlow passes bytes on some paths)."""

    return op_type.decode() if isinstance(op_type, bytes) else str(op_type)


def _observe(ledger: VariableReadLedger, tf: Any, op_type: Any, inputs: Any, graph: Any) -> None:
    """Record graph-mode ops, function calls and every non-read handle consumer."""

    op_text = _op_type_text(op_type)
    opaque_op = graph is not None or op_text in _CALL_OPS or op_text.startswith("__inference_")
    if opaque_op and ledger.opaque is None:
        ledger.opaque = op_text
    if op_text == _READ_OP and graph is None:
        return
    for tensor in inputs or ():
        if getattr(tensor, "dtype", None) != tf.resource:
            continue
        key = _tensor_ref_key(tensor)
        if key is not None:
            ledger.touched.setdefault(key, op_text)


@contextmanager
def variable_read_scope(tf: Any, module_tree: Any) -> Iterator[VariableReadLedger | None]:
    """Snapshot the discovered variables, then watch handle consumers during the forward.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    module_tree
        Discovered module tree, or ``None`` (then nothing is recorded).

    Yields
    ------
    VariableReadLedger | None
        The ledger the forward fills, or ``None`` without a module tree.
    """

    if module_tree is None:
        yield None
        return
    ledger = VariableReadLedger(snapshots=_snapshot_variables(module_tree))
    callbacks = get_op_callbacks_module()

    def observer(
        op_type: Any,
        inputs: Any,
        attrs: Any,
        outputs: Any,
        op_name: Any = None,
        graph: Any = None,
    ) -> None:
        """Feed one op-callback event to the ledger; never replaces outputs."""

        del attrs, outputs, op_name
        _observe(ledger, tf, op_type, inputs, graph)

    callbacks.add_op_callback(observer)
    try:
        yield ledger
    finally:
        callbacks.remove_op_callback(observer)


def attach_variable_reads(trace: Any, ledger: VariableReadLedger | None) -> None:
    """Keep ``ledger`` with ``trace`` for validation (dies with the trace)."""

    if ledger is not None:
        _LEDGERS[trace] = ledger


def _ledger_for(trace: Any) -> VariableReadLedger | None:
    """The ledger attached to ``trace``; ``None`` also for an object that takes no weak reference."""

    try:
        return _LEDGERS.get(trace)
    except TypeError:  # not weakly referenceable, so it never had a ledger
        return None


def _verifiable_snapshot(
    ledger: VariableReadLedger | None, capture: Any
) -> VariableSnapshot | None:
    """The snapshot a read binds to, or ``None`` when the read is not verifiable."""

    if ledger is None or ledger.opaque is not None or capture is None:
        return None
    if getattr(capture, "op_type", None) != _READ_OP:
        return None
    if ledger.by_label:
        return ledger.by_label.get(str(getattr(capture, "label_raw", "")))
    handles = [item for item in getattr(capture, "inputs", ()) if item.input_index == 0]
    if len(handles) != 1:
        return None
    key = handles[0].ref_key
    if key is None or key in ledger.touched:
        return None
    return ledger.snapshots.get(key)


def _payload_matches(op: Any, snapshot: VariableSnapshot) -> bool:
    """Whether the read's saved payload equals ``snapshot`` bit for bit."""

    payload = replay_payload(op)
    if payload is None:
        return False
    numpy_method = getattr(payload, "numpy", None)
    value = np.asarray(numpy_method() if callable(numpy_method) else payload)
    return (
        value.dtype.name == snapshot.dtype
        and tuple(value.shape) == snapshot.shape
        and _digest(value) == snapshot.digest
    )


def parameter_read_decision(trace: Any, op: Any, capture: Any) -> Mapping[str, Any] | None:
    """The decision for a verifiable variable read, or ``None`` to keep it unverified.

    Parameters
    ----------
    trace
        TensorFlow trace being validated.
    op
        The materialized ``ReadVariableOp``.
    capture
        Its callback capture record.

    Returns
    -------
    Mapping[str, Any] | None
        ``validated`` with reason ``replay_match`` when the saved payload equals the
        pre-forward snapshot, ``failed`` with reason ``replay_mismatch`` when it
        differs, ``None`` when the read is not verifiable (the caller keeps the
        ``effectful_region`` decision).
    """

    if getattr(capture, "op_type", None) != _READ_OP:
        return None
    snapshot = _verifiable_snapshot(_ledger_for(trace), capture)
    if snapshot is None:
        return None
    matched = _payload_matches(op, snapshot)
    relation = "equals" if matched else "differs from"
    return {
        "op_label": str(getattr(op, "label", getattr(capture, "label_raw", ""))),
        "func_name": _READ_OP,
        "phase": "replay",
        "decision": "validated" if matched else "failed",
        "reason": "replay_match" if matched else "replay_mismatch",
        "justification": (
            f"parameter source: the read {relation} the pre-forward snapshot of "
            f"variable {snapshot.address} and no op wrote it during the forward"
        ),
    }


def read_failures(label: str, decision: Mapping[str, Any]) -> list[str]:
    """The validator's failure string for a failed read decision (empty when it passed)."""

    if decision.get("decision") != "failed":
        return []
    return [f"replay_or_perturbation_failed:{label}:{_READ_OP}:{decision.get('reason')}"]


def persisted_read(trace: Any, capture: Any) -> dict[str, Any] | None:
    """The snapshot a verifiable read binds to, as data for its replay recipe.

    Parameters
    ----------
    trace
        Live TensorFlow trace being saved.
    capture
        The read's callback capture record.

    Returns
    -------
    dict[str, Any] | None
        ``address``, ``dtype``, ``shape`` and ``digest`` of the pre-forward
        snapshot, or ``None`` when the read is not verifiable (then nothing is
        stored and the loaded read stays ``effectful_region`` unverified).
    """

    snapshot = _verifiable_snapshot(_ledger_for(trace), capture)
    if snapshot is None:
        return None
    return {
        "address": snapshot.address,
        "dtype": snapshot.dtype,
        "shape": list(snapshot.shape),
        "digest": snapshot.digest,
    }


def _snapshot_from_record(record: Any) -> VariableSnapshot | None:
    """A snapshot from one persisted record, or ``None`` when it is malformed."""

    if not isinstance(record, Mapping):
        return None
    address, dtype, shape, digest = (record.get(k) for k in ("address", "dtype", "shape", "digest"))
    if not (isinstance(address, str) and isinstance(dtype, str) and isinstance(digest, str)):
        return None
    if not isinstance(shape, list | tuple):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
        return None
    return VariableSnapshot(address=address, dtype=dtype, shape=tuple(shape), digest=digest)


def restore_variable_reads(trace: Any, records: Mapping[str, Any]) -> None:
    """Rebuild a loaded trace's ledger from the reads' persisted snapshots.

    Parameters
    ----------
    trace
        Loaded TensorFlow trace.
    records
        :func:`persisted_read` records by read raw label. A malformed record is
        skipped, so that read stays unverified; no records attach no ledger.
    """

    by_label = {
        str(label): snapshot
        for label, record in records.items()
        if (snapshot := _snapshot_from_record(record)) is not None
    }
    if by_label:
        attach_variable_reads(trace, VariableReadLedger(by_label=by_label))

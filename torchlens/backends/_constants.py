"""The shared constants rule every preview backend's finalize step applies.

During a capture a tensor argument of a recorded op is one of four things: an
input leaf, a parameter or buffer leaf of the model tree, the output of a
recorded op, or a constant. A constant is any framework tensor the forward
used that is none of the first three. The rule has two kinds:

- a *closure constant* is a tensor built outside the traced function (a
  module attribute array, a ``Tensor(ndarray)`` literal, a closed-over
  jaxpr const). The backend records it as a source op with
  ``func_name="constant"``, no parents and a saved payload, and the op that
  consumed it lists it as a parent;
- a *literal producer* is an op with no tensor inputs (``fill``, ``ones``,
  a literal ``convert_element_type``): it stays an op with zero parents and
  its output payload is saved.

Validation treats either as a leaf whose saved payload is the replay input;
a literal producer the backend can re-execute from its saved arguments is
replayed and compared, one it cannot is a leaf with reason
``constant_producer_leaf``, and perturbation skips a constant parent with
reason ``constant_parent_payload``. A constant with no readable buffer
refuses ``constant_unrealizable`` naming the op; it is never recorded as a
parentless op without a payload.

How a constant is recognised. The op record carries no dedicated field yet
(the format lanes own the field set; ``is_constant_layer`` rides their
format bump), so :func:`is_constant_op` derives the answer from fields every
record already has: ``func_name == "constant"`` for a closure constant, and
for a literal producer a parentless, non-input, non-buffer op with no
unattributed or dropped tensor argument and no RNG-producing function name.
A lost parent edge therefore never passes as a constant: the record names
its unattributed tensor arguments, and that is a capture gap validation
must still catch. When the record field lands, the two predicates below
read it and nothing else changes.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

from .._errors import CaptureGapError
from .registry import BackendUnsupportedError

__tl_layer__ = "L2"

#: ``func_name`` of a recorded closure constant (a source op with no parents).
CONSTANT_FUNC_NAME = "constant"

#: ``layer_type`` (label family) of a recorded closure constant; its label is
#: ``constant_<n>`` following the backend's label convention.
CONSTANT_LAYER_TYPE = "constant"

#: Refusal code for a constant whose buffer cannot be read (SPEC 1.3).
CONSTANT_UNREALIZABLE_CODE = "constant_unrealizable"

#: Validation reason for a constant leaf that has no function to replay.
CONSTANT_PRODUCER_LEAF_REASON = "constant_producer_leaf"

#: Validation reason for skipping perturbation of a constant parent.
CONSTANT_PARENT_PAYLOAD_REASON = "constant_parent_payload"

ConstantKind = Literal["closure", "literal"]

#: The two kinds the rule distinguishes.
CONSTANT_KINDS: tuple[ConstantKind, ...] = ("closure", "literal")

#: Name tokens of functions that draw from the framework RNG. A parentless op
#: whose ``func_name`` carries one of these tokens produces a random value,
#: not a constant: its output is not reproducible from its arguments, so the
#: constant leaf carve-out never applies to it.
RNG_PRODUCER_TOKENS: frozenset[str] = frozenset(
    {
        "rand",
        "randn",
        "randint",
        "randperm",
        "random",
        "normal",
        "uniform",
        "bernoulli",
        "multinomial",
        "poisson",
        "gumbel",
        "exponential",
        "geometric",
        "cauchy",
        "lognormal",
        "dropout",
        "shuffle",
        "categorical",
        "truncated",
    }
)

HostCopy = Callable[[Any], Any]


def constant_label(ordinal: int) -> str:
    """The raw label of the ``ordinal``-th closure constant (1-based)."""

    if ordinal < 1:
        raise ValueError(f"constant ordinals start at 1, got {ordinal}")
    return f"{CONSTANT_LAYER_TYPE}_{ordinal}"


def is_rng_producer_name(func_name: str | None) -> bool:
    """Whether ``func_name`` names a function that draws from the RNG.

    Tokens are compared whole after splitting on ``.`` and ``_`` so
    ``layer_norm`` and ``normalize`` stay clear of ``normal``.
    """

    if not func_name:
        return False
    tokens = {token for token in func_name.lower().replace(".", "_").split("_") if token}
    return bool(tokens & RNG_PRODUCER_TOKENS)


@dataclass(frozen=True)
class ConstantRecord:
    """One constant the backend recorded during a capture.

    Parameters
    ----------
    label_raw:
        Raw label of the constant op (``constant_<n>`` for a closure
        constant, the producer's own raw label for a literal producer).
    kind:
        ``"closure"`` or ``"literal"``.
    shape:
        Shape of the host copy, ``None`` when unknown.
    dtype:
        Dtype name of the host copy, ``None`` when unknown.
    consumer_label_raw:
        Raw label of the op whose argument the constant was (closure
        constants), ``None`` otherwise.
    arg_position:
        The argument position the constant occupied in the consumer's call,
        in the backend's own position vocabulary, ``None`` when not recorded.
    """

    label_raw: str
    kind: ConstantKind
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    consumer_label_raw: str | None = None
    arg_position: Any = None


class ConstantLedger:
    """Per-capture record of the constants a backend saw, keyed by raw label.

    A backend creates one ledger per capture, calls
    :meth:`register_closure` the moment it meets an unlabelled framework
    tensor among an op's arguments (before emitting the consuming op, so the
    source op precedes its consumer) and :meth:`register_literal` when it
    emits an op with no tensor inputs, then hands the ledger to
    ``finalize_single_pass_trace(constants=...)`` where
    :func:`apply_constants_rule` audits the graph against it.

    Parameters
    ----------
    backend_name:
        Registered backend name, used in refusal messages.
    """

    def __init__(self, backend_name: str) -> None:
        """Start an empty ledger for one capture."""

        self.backend_name = backend_name
        self._records: dict[str, ConstantRecord] = {}
        self._closure_count = 0

    def __len__(self) -> int:
        """Number of constants registered."""

        return len(self._records)

    def __contains__(self, label_raw: object) -> bool:
        """Whether ``label_raw`` names a registered constant."""

        return label_raw in self._records

    def __iter__(self) -> Iterator[ConstantRecord]:
        """Iterate the records in registration order."""

        return iter(self._records.values())

    def labels(self, kind: ConstantKind | None = None) -> tuple[str, ...]:
        """Raw labels registered, optionally of one kind, in registration order."""

        return tuple(record.label_raw for record in self if kind is None or record.kind == kind)

    def get(self, label_raw: str) -> ConstantRecord | None:
        """The record for ``label_raw``, else ``None``."""

        return self._records.get(label_raw)

    def register_closure(
        self,
        value: Any,
        *,
        host_copy: HostCopy,
        label_raw: str | None = None,
        consumer_label_raw: str | None = None,
        arg_position: Any = None,
    ) -> tuple[ConstantRecord, Any]:
        """Register one closure constant and return its record and host copy.

        The host copy is taken here so an unreadable buffer refuses typed
        before the consuming op is recorded (rule 4): the refusal names the
        constant's label and the consumer it was headed for.

        Parameters
        ----------
        value:
            The framework tensor the forward used.
        host_copy:
            The backend's ``host_copy`` (a detached NumPy array, shape kept).
        label_raw:
            The raw label the backend reserved for the source op in its own
            label convention (``reserve_label("constant")`` yields
            ``constant_<n>_<raw>_raw``); ``constant_<n>`` when omitted.
        consumer_label_raw:
            Raw label of the op about to consume the constant, when known.
        arg_position:
            Position the constant occupies in the consumer's arguments.

        Returns
        -------
        tuple[ConstantRecord, Any]
            The record and the host copy, which the backend stores as the
            source op's payload.

        Raises
        ------
        BackendUnsupportedError
            ``constant_unrealizable`` when the value has no readable buffer.
        """

        if label_raw is None:
            label_raw = constant_label(self._closure_count + 1)
        payload = realize_host_copy(
            value,
            host_copy=host_copy,
            backend_name=self.backend_name,
            op_label=label_raw,
            consumer_label=consumer_label_raw,
        )
        record = ConstantRecord(
            label_raw=label_raw,
            kind="closure",
            shape=_shape_of(payload),
            dtype=_dtype_name(payload),
            consumer_label_raw=consumer_label_raw,
            arg_position=arg_position,
        )
        self._closure_count += 1
        self._records[label_raw] = record
        return record, payload

    def register_literal(
        self,
        label_raw: str,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: str | None = None,
    ) -> ConstantRecord:
        """Register the op ``label_raw`` as a literal producer.

        Parameters
        ----------
        label_raw:
            Raw label of the op with no tensor inputs.
        shape:
            Output shape, when known.
        dtype:
            Output dtype name, when known.

        Returns
        -------
        ConstantRecord
            The literal record.
        """

        record = ConstantRecord(label_raw=label_raw, kind="literal", shape=shape, dtype=dtype)
        self._records[label_raw] = record
        return record


def realize_host_copy(
    value: Any,
    *,
    host_copy: HostCopy,
    backend_name: str,
    op_label: str,
    consumer_label: str | None = None,
) -> Any:
    """Take a host copy of ``value`` or refuse ``constant_unrealizable``.

    Parameters
    ----------
    value:
        Framework tensor to copy.
    host_copy:
        The backend's ``host_copy``.
    backend_name:
        Registered backend name, for the message.
    op_label:
        Label of the constant op being recorded, named by the refusal.
    consumer_label:
        Label of the consuming op, named by the refusal when known.

    Returns
    -------
    Any
        The host copy, which exposes ``shape``.

    Raises
    ------
    BackendUnsupportedError
        ``constant_unrealizable`` when ``host_copy`` raises anything but a
        typed refusal, or returns a value without a shape.
    """

    try:
        payload = host_copy(value)
    except BackendUnsupportedError:
        raise
    except Exception as exc:  # noqa: BLE001 - every framework raises its own class here
        raise _unrealizable(
            backend_name, op_label, consumer_label, cause=repr(exc), code=CONSTANT_UNREALIZABLE_CODE
        ) from exc
    if payload is None or not hasattr(payload, "shape"):
        raise _unrealizable(
            backend_name,
            op_label,
            consumer_label,
            cause=f"host copy is {type(payload).__name__}",
            code=CONSTANT_UNREALIZABLE_CODE,
        )
    return payload


def _unrealizable(
    backend_name: str, op_label: str, consumer_label: str | None, *, cause: str, code: str
) -> BackendUnsupportedError:
    """Build the typed refusal for a constant with no readable buffer.

    ``code`` is spelled at every raise site so the S-17 census reads the
    refusal as coded there (it is always :data:`CONSTANT_UNREALIZABLE_CODE`).
    """

    consumer = f" consumed by {consumer_label}" if consumer_label else ""
    return BackendUnsupportedError(
        f"{backend_name} backend cannot read the buffer of constant {op_label}{consumer}: {cause}",
        code=code,
        remedy="realize the constant on the host before the forward, or pass it as an input",
        backend=backend_name,
        op_label=op_label,
        consumer_label=consumer_label,
    )


def _shape_of(payload: Any) -> tuple[int, ...] | None:
    """``payload.shape`` as a tuple of ints, else ``None``."""

    shape = getattr(payload, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _dtype_name(payload: Any) -> str | None:
    """``payload.dtype`` as its name, else ``None``."""

    dtype = getattr(payload, "dtype", None)
    if dtype is None:
        return None
    return str(getattr(dtype, "name", dtype))


# ----------------------------------------------------------------------------
# Recognising constants on a record (the one place that flips to the record
# field ``is_constant_layer`` when the format lanes add it).


def is_closure_constant_op(op: Any) -> bool:
    """Whether ``op`` is a recorded closure constant (``func_name == "constant"``)."""

    return getattr(op, "func_name", None) == CONSTANT_FUNC_NAME


def is_literal_producer_op(op: Any) -> bool:
    """Whether ``op`` is a literal producer: an op with no tensor inputs.

    Parentless, not an input or buffer leaf, not a closure constant, with no
    unattributed or dropped tensor argument (a lost edge is a capture gap,
    never a constant) and no RNG-producing function name.
    """

    if is_closure_constant_op(op):
        return False
    if getattr(op, "is_input", False) or getattr(op, "is_buffer", False):
        return False
    if tuple(getattr(op, "parents", ()) or ()):
        return False
    if tuple(getattr(op, "unattributed_tensor_args", ()) or ()):
        return False
    if tuple(getattr(op, "dropped_edge_tensor_args", ()) or ()):
        return False
    return not is_rng_producer_name(getattr(op, "func_name", None))


def is_constant_op(op: Any) -> bool:
    """Whether ``op`` is a constant of either kind."""

    return is_closure_constant_op(op) or is_literal_producer_op(op)


def constant_kind(op: Any) -> ConstantKind | None:
    """``"closure"``, ``"literal"`` or ``None`` for a non-constant op."""

    if is_closure_constant_op(op):
        return "closure"
    if is_literal_producer_op(op):
        return "literal"
    return None


def constant_op_labels(trace: Any) -> tuple[str, ...]:
    """Labels of every constant op in a finalized trace, in graph order."""

    return tuple(op.label for op in trace.layer_list if is_constant_op(op))


# ----------------------------------------------------------------------------
# Validation helpers (rule 3). The carve-out is narrow by construction: it
# answers only for ops the predicates above recognise, and a literal producer
# whose function can be re-executed is replayed, never exempted.


def constant_leaf_reason(op: Any, *, replayable: bool) -> str | None:
    """Reason to treat ``op`` as a constant leaf in replay, else ``None``.

    Parameters
    ----------
    op:
        The op under replay.
    replayable:
        Whether the backend can re-execute this op from its saved arguments.
        A closure constant has no function and ignores the flag.

    Returns
    -------
    str | None
        ``constant_producer_leaf`` when replay reads the saved payload
        instead of re-executing; ``None`` when the op is replayed normally
        (a replayable literal producer, or any non-constant op).
    """

    if is_closure_constant_op(op):
        return CONSTANT_PRODUCER_LEAF_REASON
    if is_literal_producer_op(op) and not replayable:
        return CONSTANT_PRODUCER_LEAF_REASON
    return None


def constant_parent_skip_reason(parent_op: Any) -> str | None:
    """Reason to skip perturbing ``parent_op``, else ``None``.

    Perturbation has nothing upstream of a constant to perturb; the consuming
    op is still perturbed on its non-constant parents.
    """

    return CONSTANT_PARENT_PAYLOAD_REASON if is_constant_op(parent_op) else None


# ----------------------------------------------------------------------------
# The finalize hook.


def apply_constants_rule(
    trace: Any, *, backend_name: str, ledger: ConstantLedger | None = None
) -> tuple[str, ...]:
    """Audit the raw graph against the constants rule before finalization.

    Every recorded closure constant must be a source op (no parents, not an
    input or buffer) with payload evidence; every constant the ledger holds
    must exist in the graph. Literal producers are counted, never altered.

    Parameters
    ----------
    trace:
        Trace whose ``_raw_graph_ws.raw_layer_dict`` holds the backend's op
        logs, before per-op finalization.
    backend_name:
        Registered backend name, for messages.
    ledger:
        The capture's ledger, when the backend keeps one.

    Returns
    -------
    tuple[str, ...]
        Raw labels of the constant ops found, in graph order.

    Raises
    ------
    CaptureGapError
        A constant op with parents or leaf flags, or a ledger entry with no
        op in the graph: the backend recorded the rule inconsistently.
    BackendUnsupportedError
        ``constant_unrealizable`` for a closure constant recorded without
        payload evidence (rule 4).
    """

    raw_ops = trace._raw_graph_ws.raw_layer_dict
    found: list[str] = []
    for label_raw, op in raw_ops.items():
        if is_closure_constant_op(op):
            _audit_closure_constant(op, label_raw, backend_name=backend_name, ledger=ledger)
            found.append(label_raw)
        elif is_literal_producer_op(op):
            found.append(label_raw)
    if ledger is not None:
        missing = [label for label in ledger.labels() if label not in raw_ops]
        if missing:
            raise CaptureGapError(
                f"{backend_name} backend registered constants with no op in the graph: {missing}",
                backend=backend_name,
                missing_labels=tuple(missing),
            )
    return tuple(found)


def _audit_closure_constant(
    op: Any, label_raw: str, *, backend_name: str, ledger: ConstantLedger | None
) -> None:
    """Check one closure constant op against rules 1 and 4."""

    parents = tuple(getattr(op, "parents", ()) or ())
    if parents:
        raise CaptureGapError(
            f"{backend_name} backend recorded constant {label_raw} with parents {parents}",
            backend=backend_name,
            op_label=label_raw,
        )
    if getattr(op, "is_input", False) or getattr(op, "is_buffer", False):
        raise CaptureGapError(
            f"{backend_name} backend recorded constant {label_raw} as an input or buffer leaf",
            backend=backend_name,
            op_label=label_raw,
        )
    has_payload = getattr(op, "out", None) is not None
    has_shape = getattr(op, "shape", None) is not None
    in_ledger = ledger is not None and label_raw in ledger
    if not (has_payload or has_shape or in_ledger):
        consumer = next(
            (child for child in tuple(getattr(op, "children", ()) or ())),
            None,
        )
        raise _unrealizable(
            backend_name,
            label_raw,
            consumer,
            cause="recorded with neither payload nor shape",
            code=CONSTANT_UNREALIZABLE_CODE,
        )

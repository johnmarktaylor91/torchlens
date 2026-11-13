"""tinygrad adoption of the shared backend rules: typed refusals, named verdicts, constants.

Three small contracts the backend protocol asks of every backend,
written here once so ``backend.py`` only calls them:

- Typed refusals (SPEC 1.3): a payload that has no readable buffer refuses
  ``constant_unrealizable`` naming the op instead of surfacing tinygrad's
  ``NotImplementedError("need copyin")``; a forward that draws random
  numbers itself (lazy RNG state the forward created) refuses
  ``rng_state_in_graph`` before anything is realized. Random-init weights
  created before the forward are realized once with the capture instead
  (``_realize_once``).
- Named validation verdicts (SPEC 4.3): replay validation reports the first
  offending op and a reason from ``VALIDATION_REASONS`` instead of a bare
  boolean.
- The constants rule (SPEC 2): every parentless, non-input, non-parameter UOp
  is classified as a closure constant (``BUFFER``) or a literal producer
  (``CONST``). The classification is recorded as the annotation
  ``tinygrad_constant_kind``; the shared rule in ``backends/_constants.py``
  reads it to stamp ``Op.is_constant`` and relabel closure constants. Until
  that rule lands the recorded ``func_name`` stays the observed tinygrad name
  so the pinned op inventories hold: flip
  :data:`CLOSURE_CONSTANT_FUNC_NAME` to ``"constant"`` in one edit then.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .._errors import BackendUnsupportedError

#: Refusal code for a closure constant whose buffer tinygrad cannot read back.
CONSTANT_UNREALIZABLE_CODE = "constant_unrealizable"
#: Refusal code for a forward graph that still holds lazy RNG state.
RNG_STATE_IN_GRAPH_CODE = "rng_state_in_graph"

__tl_layer__ = "L2"

ConstantKind = Literal["closure", "literal"]

#: The one place the closure-constant relabel is switched on. ``None`` keeps
#: the observed tinygrad name (``buffer``) for ``func_name`` and ``layer_type``
#: (so labels stay ``buffer_<n>``); ``"constant"`` is SPEC 2 rule 1 and makes
#: the shared constants rule (``backends/_constants.py``) see the op as a
#: closure constant and label it ``constant_<n>``. Flip it together with the
#: pinned tinygrad inventories in ``conformance/_preview_adapters.py``.
CLOSURE_CONSTANT_FUNC_NAME: str | None = None

#: Annotation key carrying the constant classification on a tinygrad op.
CONSTANT_KIND_ANNOTATION = "tinygrad_constant_kind"

#: UOp names tinygrad's Threefry generator leaves in a graph that consumed RNG.
RNG_UOP_NAMES: frozenset[str] = frozenset({"THREEFRY"})

#: The payload policy a live capture records on a trace: every recorded value
#: realized by one ``realize_once`` call after the forward (``_realize_once``).
REALIZE_ONCE_PAYLOAD_POLICY = "realize_once"


def describe_uop(uop: Any) -> str:
    """One-line human description of a tinygrad UOp for refusal messages.

    Parameters
    ----------
    uop
        tinygrad UOp.

    Returns
    -------
    str
        ``"<OP> shape=<shape> dtype=<dtype> device=<device>"`` with each part
        present when tinygrad exposes it.
    """

    op = getattr(uop, "op", None)
    name = str(getattr(op, "name", op)).removeprefix("Ops.")
    parts = [name]
    shape = getattr(uop, "shape", None)
    if shape is not None:
        parts.append(f"shape={tuple(shape)}")
    dtype = getattr(uop, "dtype", None)
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    device = getattr(uop, "device", None)
    if device is not None:
        parts.append(f"device={device}")
    return " ".join(parts)


def refuse_constant_unrealizable(
    uop: Any, cause: BaseException, *, code: str
) -> BackendUnsupportedError:
    """Build the typed refusal for a payload tinygrad cannot read back.

    Parameters
    ----------
    uop
        The UOp whose payload has no readable buffer.
    cause
        The tinygrad error that surfaced it (``NotImplementedError("need copyin")``).
    code
        The refusal code, spelled at the raise site (:data:`CONSTANT_UNREALIZABLE_CODE`).

    Returns
    -------
    BackendUnsupportedError
        Refusal coded ``constant_unrealizable`` naming the op.
    """

    return BackendUnsupportedError(
        "the tinygrad backend cannot read the payload of op "
        f"{describe_uop(uop)}: the value has no readable buffer ({cause}). A closure "
        "constant built outside the traced callable (Tensor(ndarray) or a tinygrad.nn layer "
        "not yet realized) stays a host-side source buffer",
        code=code,
        remedy="realize the constant before capture or pass it as a traced input",
        op=describe_uop(uop),
    )


def rng_state_uops(uops: Iterable[Any]) -> tuple[Any, ...]:
    """The UOps in ``uops`` that carry tinygrad's lazy RNG generator state.

    Parameters
    ----------
    uops
        UOps of a forward graph, any order.

    Returns
    -------
    tuple[Any, ...]
        Threefry UOps, in the order met; empty when the graph consumed no RNG.
    """

    found = []
    for uop in uops:
        op = getattr(uop, "op", None)
        name = str(getattr(op, "name", op)).removeprefix("Ops.")
        if name in RNG_UOP_NAMES:
            found.append(uop)
    return tuple(found)


def refuse_rng_state_in_graph(uop: Any, *, code: str) -> BackendUnsupportedError:
    """Build the typed refusal for a forward graph holding lazy RNG state.

    Parameters
    ----------
    uop
        The first generator UOp the forward itself created.
    code
        The refusal code, spelled at the raise site (:data:`RNG_STATE_IN_GRAPH_CODE`).

    Returns
    -------
    BackendUnsupportedError
        Refusal coded ``rng_state_in_graph`` naming the op.
    """

    return BackendUnsupportedError(
        "the tinygrad backend cannot capture a forward that draws random numbers "
        f"itself (op {describe_uop(uop)}): replay validation would draw them again and get "
        "different values. Random-init weights created before the forward are captured; "
        "Tensor.rand, dropout or other RNG calls inside the traced callable are refused",
        code=code,
        remedy=(
            "draw the random values before capture and pass them in as inputs or closure "
            "tensors, or disable in-forward randomness (for example dropout) while tracing"
        ),
        op=describe_uop(uop),
    )


def constant_kind_for_uop(
    uop: Any,
    *,
    parent_labels: Sequence[str],
    is_input: bool,
    is_parameter: bool,
) -> ConstantKind | None:
    """Classify a recorded UOp under the constants rule.

    Parameters
    ----------
    uop
        The UOp being recorded.
    parent_labels
        Labels of the recorded parents the UOp consumes.
    is_input
        Whether the UOp is (or aliases) a traced input leaf.
    is_parameter
        Whether the UOp is a parameter tensor the module tree knows.

    Returns
    -------
    ConstantKind or None
        ``"closure"`` for a parentless source buffer the forward used,
        ``"literal"`` for a parentless literal producer, ``None`` otherwise.
    """

    if parent_labels or is_input or is_parameter:
        return None
    op = getattr(uop, "op", None)
    name = str(getattr(op, "name", op)).removeprefix("Ops.")
    if name == "BUFFER":
        return "closure"
    if name == "CONST":
        return "literal"
    return None


def constant_func_name(kind: ConstantKind | None, observed_name: str) -> str:
    """The ``func_name`` to record for an op of constant ``kind``.

    Parameters
    ----------
    kind
        The constant classification, or ``None`` for an ordinary op.
    observed_name
        The tinygrad name the observer handed over.

    Returns
    -------
    str
        ``observed_name`` unless the op is a closure constant and
        :data:`CLOSURE_CONSTANT_FUNC_NAME` is set.
    """

    if kind == "closure" and CLOSURE_CONSTANT_FUNC_NAME is not None:
        return CLOSURE_CONSTANT_FUNC_NAME
    return observed_name


def tinygrad_constant_ops(trace: Any) -> dict[str, ConstantKind]:
    """Constant classification of every op in a tinygrad trace, by label.

    Parameters
    ----------
    trace
        A trace captured by the tinygrad backend.

    Returns
    -------
    dict[str, ConstantKind]
        ``label -> kind`` for the ops the constants rule applies to.
    """

    kinds: dict[str, ConstantKind] = {}
    for op in trace._layer_list:
        annotations = getattr(op, "annotations", None) or {}
        kind = annotations.get(CONSTANT_KIND_ANNOTATION)
        if kind in ("closure", "literal"):
            kinds[str(op.label)] = kind
    return kinds


@dataclass(frozen=True)
class ReplayVerdict:
    """Outcome of replaying every captured tinygrad UOp against its saved payload.

    Parameters
    ----------
    passed
        Whether every op replayed to its saved payload and reacted to parent
        perturbation.
    num_replayed
        Ops replayed before the verdict was reached.
    failed_label
        Raw label of the first offending op, or ``None`` when passed.
    reason
        A member of ``VALIDATION_REASONS`` naming why the op failed, or ``None``.
    detail
        One sentence of evidence for the failure, or ``""``.
    """

    passed: bool
    num_replayed: int
    failed_label: str | None = None
    reason: str | None = None
    detail: str = ""

    @classmethod
    def ok(cls, num_replayed: int) -> ReplayVerdict:
        """A passing verdict over ``num_replayed`` ops."""

        return cls(passed=True, num_replayed=num_replayed)

    @classmethod
    def fail(cls, label: str, reason: str, detail: str, num_replayed: int) -> ReplayVerdict:
        """A failing verdict naming the first offending op and its reason."""

        return cls(
            passed=False,
            num_replayed=num_replayed,
            failed_label=label,
            reason=reason,
            detail=detail,
        )

    def decisions(self) -> tuple[Mapping[str, Any], ...]:
        """Per-op decisions for ``ValidationStatus.decisions`` (the failure only)."""

        if self.passed or self.failed_label is None:
            return ()
        return (
            {
                "op_label": self.failed_label,
                "decision": "failed",
                "reason": self.reason,
                "detail": self.detail,
            },
        )

    def message(self) -> str:
        """The status message: names the first offending op on failure."""

        if self.passed:
            return f"Replay validation passed: {self.num_replayed} tinygrad ops replayed."
        return (
            f"Replay validation failed at op {self.failed_label!r} ({self.reason}): {self.detail}"
        )

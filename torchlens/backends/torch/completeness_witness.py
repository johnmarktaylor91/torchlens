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
from ._tl import get_tensor_label, get_tensor_meta
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


HOST_ESCAPE_OPERATORS = frozenset(
    {
        "aten._local_scalar_dense",
        "aten.equal",
        "aten.allclose",
        "aten.is_nonzero",
    }
)
"""Aten operators (overload-stripped base names) that read a captured tensor's VALUE
out to the Python host.

This is a NARROW allowlist of genuine tensor->host VALUE escapes, NOT a general
"any non-tensor output" census (which mis-fires on tensor STRUCTURE/METADATA ops --
``size`` / ``sym_size`` / ``numel`` / ``dim`` / ``stride`` / ``is_contiguous`` /
``dtype`` / ``device`` / ``storage_offset`` / ... -- whose non-tensor output derives
from shape/layout, is input-VALUE-independent, and is already covered by the separate
input-shape-mismatch check; witnessing those is both wrong (it over-triggers a false
UNVERIFIABLE on an escape-free model) and a pathological per-op capture slowdown
because a real model reads shapes constantly).

* ``aten._local_scalar_dense`` -- the single dispatcher footprint of every
  tensor->Python SCALAR escape: ``.item()``, ``int()``, ``float()``, ``__index__``,
  and ``bool()`` all lower to it (single tensor operand).
* ``aten.equal`` / ``aten.allclose`` / ``aten.is_nonzero`` -- pure tensor->``bool``
  predicates that return a raw Python value DIRECTLY from the dispatcher and never emit
  ``aten._local_scalar_dense`` (two/one tensor operands).

Recording the SOURCE tensor(s) of such an escape lets the runnable descriptor witness
the escape by its producing op (keyed on the ESCAPE EVENT), never by correlating a
baked literal by value. Multi-element ``.tolist()`` / ``.numpy()`` / ``__array__`` /
``__dlpack__`` conversions do NOT emit any of these ops (they are dispatcher-invisible)
and are handled by the scoped method/property patch plus the descriptor's complementary
value-equality net.
"""

_HOST_ESCAPE_SOURCE_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = weakref.WeakKeyDictionary()
"""Per-trace raw producing-op labels of tensor->host escape sources.

Kept off the Trace ``__dict__`` (and therefore out of portable-state scrub) in a
weak-keyed side table so the runnable descriptor can read escape sources without
registering a new serialized Trace field. Entries are dropped automatically when a
Trace is garbage collected.
"""

_HOST_ESCAPE_STATE_SOURCE_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace subset of escape-source labels whose source is a registered param/buffer.

A registered buffer/parameter read on the host (``bool(self.gate)`` /
``self.threshold.item()``) is the runnable UNBOUND-STATE net's domain: it is
witnessed by its capture-time state digest, not by a tensor-op source slot. When such
a state source is read ONLY on the host it is orphan-pruned and its raw label does not
resolve to a final op -- but that is NOT a coverage gap (the unbound-state net covers
it), so the runnable producer must NOT close it as a pruned tensor-op chain. This
side set lets the producer tell an unresolved STATE label (defer to the unbound net)
from an unresolved TENSOR-OP label (a genuinely unwitnessable pruned host chain).
"""

_HOST_ESCAPE_BOOL_SOURCE_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace subset of escape-source labels whose source is a BOOL tensor.

A ``bool(...)`` truth-test steering pure-Python control flow is the control-witness /
conditional / loop / pruned-RNG net's domain, not the tensor-derived scalar net's. The
label is still recorded in ``_HOST_ESCAPE_SOURCE_LABELS`` because the pruned-RNG
control-flow detector consumes it, but the runnable producer uses this set to exclude
an unresolved (orphan-pruned) BOOL predicate from the tensor-op INCOMPLETE gate -- a
pruned bool predicate is honestly witnessed (or downgraded) by those other nets.
"""

_HOST_ESCAPE_UNATTRIBUTABLE_VALUES: "weakref.WeakKeyDictionary[Any, set[Any]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace scalar values of UNATTRIBUTABLE (unlabelled-source) non-bool escapes.

A ``.data`` alias (or any tensor whose capture label cannot be resolved) that escapes
to the host via ``aten._local_scalar_dense`` leaves no source-op label to witness by
slot. But the escape IS a scalar (``_local_scalar_dense`` extracts exactly one value),
so its numeric value is recorded here. The runnable producer treats these exactly like
baked literals: an INTERNAL-SINK op whose retained output equals an unattributable
escape value is the escaped source, so it is witnessed value-free by its capture-time
digest. This keeps the original input VERIFIED (byte-identical slot) while a changed
input recomputes different bytes -> UNVERIFIABLE, instead of a false VERIFIED.

BOOL escape sources are NOT recorded anywhere by the census: a ``bool(...)``
truth-test steering control flow is the control-witness / conditional / loop /
pruned-RNG net's domain, not the tensor-derived scalar net's. An UNATTRIBUTABLE
bool escape whose predicate is NOT covered by any of those nets is recorded in
``_HOST_ESCAPE_UNATTRIBUTABLE_BOOL`` below so the runnable producer can fail closed.
"""

INVISIBLE_HOST_ESCAPE_FUNCS = frozenset({"tolist", "numpy", "__array__", "__dlpack__"})
"""Torch-function / protocol METHOD names that hand a captured tensor's value to the
Python host WITHOUT emitting any aten dispatch.

``.tolist()`` / ``.numpy()`` / ``np.asarray(tensor)`` (``__array__``) convert a tensor
to a Python/NumPy container, and ``tensor.__dlpack__()`` (used by ``np.from_dlpack`` /
``torch.from_dlpack`` and every array library's zero-copy import) exports the tensor's
buffer as a DLPack capsule. They emit NO ``aten._local_scalar_dense`` (they are
dispatcher-invisible), so the aten census cannot see them. A scoped method patch
(``_observe_invisible_host_escapes``) records the SOURCE tensor of each such call for the
duration of one runnable forward, so an invisible escape is witnessed by the SAME
source-digest machinery as a census (``.item()``) escape -- host ARITHMETIC on the escaped
value (``sum(t.tolist())`` / ``np.from_dlpack(x)[0]``) is irrelevant because the SOURCE
tensor is what changes on a changed-input/changed-state run. A patch is used rather than a
``TorchFunctionMode`` because any active function mode flips ``has_torch_function`` globally
and breaks TorchLens's own function-wrapping capture.

Sibling zero-copy protocols audited: ``__cuda_array_interface__`` (a non-callable buffer
PROPERTY) is patched separately as a source-recording property (see
``INVISIBLE_HOST_ESCAPE_PROPERTIES``); ``__dlpack_device__`` returns only device metadata
(a ``(device_type, device_id)`` pair, no data) and ``__array_interface__`` is absent on
``torch.Tensor``, so neither is a value escape.
"""

INVISIBLE_HOST_ESCAPE_PROPERTIES = frozenset({"__cuda_array_interface__"})
"""Zero-copy buffer PROTOCOL PROPERTIES (not methods) that expose a captured tensor's
data pointer to the host.

``tensor.__cuda_array_interface__`` is the CUDA Array Interface a foreign array library
(CuPy / Numba) reads to import the tensor's device buffer zero-copy. It is a non-callable
getset descriptor, so the method patch cannot wrap it; ``_observe_invisible_host_escapes``
instead installs a source-recording ``property`` for the duration of one runnable forward.
If such a property can neither be wrapped nor its source recorded, its use must fail closed
(INCOMPLETE), never silently VERIFIED.
"""

_HOST_ESCAPE_STATE_SOURCE_NAMES: "weakref.WeakKeyDictionary[Any, set[str]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace ``state_dict`` names (addresses) of every escape source that is a registered
param/buffer, whether or not that state also feeds a traced graph op.

A registered buffer/parameter read on the host (``self.threshold.item()`` /
``self.gate.numpy()``) is witnessed by its capture-time STATE digest keyed to its state
slot -- NOT by a tensor-op source slot, and NOT only when the state is unbound. Recording
the address here lets the runnable producer witness the escape's state slot (bound OR
unbound) so a changed staged value -> UNVERIFIABLE while capture-equivalent state ->
VERIFIED. bound-ness exempts a state slot from the UNBOUND-state net, never from the escape
witness.
"""

_HOST_ESCAPE_UNATTRIBUTABLE_BOOL: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces that observed an UNATTRIBUTABLE (unlabelled-source) BOOL escape.

``bool(self.gate.data > 0.5)`` truth-tests a bool tensor produced on a ``.data`` alias;
the predicate op is orphan-pruned (input-disconnected) and the escaped source carries no
resolvable capture label. NO net (control-witness, conditional, loop, or pruned-RNG) covers
such a pruned non-RNG bool predicate, so its branch source cannot be witnessed. The runnable
producer downgrades witness completeness to keep the model honestly UNVERIFIABLE rather than
falsely VERIFIED, exactly like a pruned-RNG control escape. Membership is presence-only.
"""

_HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces that observed an UNATTRIBUTABLE census-INVISIBLE escape (``.tolist()`` /
``.numpy()`` on an unlabelled tensor, e.g. a ``.data`` alias).

A dispatcher-invisible conversion of a tensor with no resolvable capture label leaves no
source-op slot AND no reliable scalar to value-match (it may be multi-element). Its source
cannot be witnessed, so the runnable producer fails closed (UNVERIFIABLE). Presence-only.
"""


def host_escape_source_labels(trace: Any) -> frozenset[str]:
    """Return the recorded raw escape-source op labels for one trace."""

    labels = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def host_escape_state_source_names(trace: Any) -> frozenset[str]:
    """Return the ``state_dict`` names of every registered-state escape source."""

    names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
    return frozenset(names) if names else frozenset()


def host_escape_has_unattributable_bool(trace: Any) -> bool:
    """Return whether an unwitnessable (pruned, unlabelled) bool escape was seen."""

    return trace in _HOST_ESCAPE_UNATTRIBUTABLE_BOOL


def host_escape_has_unattributable_opaque(trace: Any) -> bool:
    """Return whether an unwitnessable census-invisible escape (``.tolist``/``.numpy``) was seen."""

    return trace in _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE


def host_escape_state_source_labels(trace: Any) -> frozenset[str]:
    """Return the raw escape-source labels whose source is a registered param/buffer."""

    labels = _HOST_ESCAPE_STATE_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def host_escape_bool_source_labels(trace: Any) -> frozenset[str]:
    """Return the raw escape-source labels whose source is a bool control predicate."""

    labels = _HOST_ESCAPE_BOOL_SOURCE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def host_escape_unattributable_values(trace: Any) -> frozenset[Any]:
    """Return scalar values of unattributable (unlabelled-source) non-bool escapes."""

    values = _HOST_ESCAPE_UNATTRIBUTABLE_VALUES.get(trace)
    return frozenset(values) if values else frozenset()


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


_COMPLETENESS_WITNESS_FILE = Path(__file__).resolve()
"""This module's path, skipped when locating the true invoker of an escape."""

_WRAPPERS_FILE = _COMPLETENESS_WITNESS_FILE.parent / "wrappers.py"
"""TorchLens torch-function wrapper plumbing; skipped when finding the true invoker."""

_TORCHLENS_ROOT = _COMPLETENESS_WITNESS_FILE.parents[2]
"""The ``torchlens`` package root; a true-invoker frame here marks an internal read."""

_MAX_ESCAPE_STACK_DEPTH = 60
"""Bounded stack walk when classifying an escape dispatch's origin."""


_internal_read_state = threading.local()
"""Per-thread depth counter for the explicit TorchLens internal-scalar-read marker."""


def _internal_read_active() -> bool:
    """Return whether an explicit TorchLens internal-scalar-read marker is live.

    The marker is set by construction ONLY around TorchLens's own capture-internal
    scalar/comparison reads (see :func:`internal_scalar_read`). It is a per-thread depth
    counter so nested internal reads compose correctly.
    """

    return getattr(_internal_read_state, "depth", 0) > 0


@contextmanager
def internal_scalar_read() -> Iterator[None]:
    """Mark a region as a genuine TorchLens capture-internal scalar/comparison read.

    TorchLens itself reads a scalar/bool from a tensor during capture -- extracting a
    scalar-``bool`` op-output value (``ops._log_output_tensor_info``), comparing a
    pre-call input copy against its post-call value for mutation/alias detection
    (``tensor_nanequal`` via ``detect_torch_alias_contract`` and the child-version
    snapshot), and similar bookkeeping reads. Those reads lower to
    ``aten._local_scalar_dense`` (or, for content comparisons, ``aten.equal`` /
    ``aten.allclose`` returning a Python ``bool``) and would otherwise be misrecorded as
    USER host escapes and falsely trip the fail-closed INCOMPLETE gates. This context
    manager marks them EXPLICITLY so the internal-vs-user classifier is an
    allowlist-BY-CONSTRUCTION: an escape is "internal" iff this marker is active, never
    because a stack frame's filename happens to resolve inside the ``torchlens`` package
    (which is spoofable by an ``exec``-compiled user helper carrying a torchlens
    ``co_filename``, and fails for a frameless C-callable). A genuine user escape runs
    with the marker inactive and is always recorded.
    """

    depth = getattr(_internal_read_state, "depth", 0)
    _internal_read_state.depth = depth + 1
    try:
        yield
    finally:
        _internal_read_state.depth = depth


def _escape_source_is_torchlens_internal() -> bool:
    """Return whether an escape dispatch is a marked TorchLens capture-internal read.

    Classification is allowlist-BY-CONSTRUCTION: it is ``True`` iff an explicit
    :func:`internal_scalar_read` marker is live on this thread, set only around
    TorchLens's own genuine internal scalar/comparison reads. It NEVER infers "internal"
    from a stack frame's ``co_filename`` (spoofable via an ``exec``-compiled user helper
    given a torchlens filename, and undefined for a frameless C-callable such as
    ``operator.methodcaller("item")``). A USER escape therefore can never be classified
    internal, while TorchLens's own reads never trip the fail-closed gates.
    """

    return _internal_read_active()


def _output_is_host_value(result: Any) -> bool:
    """Return whether a dispatch output is a pure Python host value carrying no tensor.

    A host value is a Python scalar (``bool`` / ``int`` / ``float`` / ``complex``) or a
    ``list`` / ``tuple`` recursively of host values. A ``torch.Tensor`` output (or any
    container carrying a tensor) is an ordinary op result, NOT a host escape.
    """

    if isinstance(result, torch.Tensor):
        return False
    if isinstance(result, (bool, int, float, complex)):
        return True
    if isinstance(result, (list, tuple)):
        return len(result) > 0 and all(_output_is_host_value(value) for value in result)
    return False


def _iter_tensor_operands(args: tuple[Any, ...]) -> Iterator[torch.Tensor]:
    """Yield each tensor operand of a dispatch, including tensors nested one list/tuple deep."""

    for value in args:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    yield item


def _record_host_escape_source(trace: Any, func: Any, args: tuple[Any, ...], result: Any) -> None:
    """Record the raw producing-op label of every tensor->host VALUE-escape SOURCE.

    NARROW value-escape rule: the dispatch's operator (overload-stripped) must be one of
    the ``HOST_ESCAPE_OPERATORS`` VALUE reads (``aten._local_scalar_dense`` from
    ``.item()`` / ``int()`` / ``float()`` / ``__index__`` / ``bool()``, or the direct
    tensor->``bool`` predicates ``aten.equal`` / ``aten.allclose`` / ``aten.is_nonzero``)
    AND its output must be a pure Python host value. That host value can be baked into a
    downstream op literal (verbatim OR after arbitrary Python arithmetic) or steer
    pure-Python control flow -- neither of which the sparse DAG can recompute.

    The rule is DELIBERATELY not a general "any non-tensor output" census: tensor
    STRUCTURE/METADATA ops (``size`` / ``sym_size`` / ``numel`` / ``dim`` / ``stride`` /
    ``is_contiguous`` / ``dtype`` / ``device`` / ``storage_offset`` / ...) also return
    non-tensor host values, but those derive from shape/layout, are input-VALUE-
    independent (already covered by the separate input-shape-mismatch check), and a real
    model reads them constantly -- witnessing them over-triggers a false UNVERIFIABLE on
    escape-free models and pathologically slows per-op capture. Restricting to the value
    allowlist keeps genuine escapes witnessed without either regression.

    Every tensor operand of a value escape is recorded as a source; its raw capture label
    lets the runnable descriptor witness that source slot and, at run time, refuse a false
    VERIFIED when the slot recomputes different bytes for a changed input.
    """

    if not args or not _is_aten_operator(func):
        return
    if _operator_base_name(func) not in HOST_ESCAPE_OPERATORS:
        return
    if not _output_is_host_value(result):
        return
    for source in _iter_tensor_operands(args):
        _record_escape_source_tensor(trace, source, invisible=False)


def _record_escape_source_tensor(trace: Any, source: torch.Tensor, *, invisible: bool) -> None:
    """Record ONE tensor->host escape source, visible or census-invisible, uniformly.

    ``invisible`` is ``True`` for a ``.tolist()`` / ``.numpy()`` / ``__array__``
    conversion (observed by the scoped method patch) and ``False`` for an
    ``aten._local_scalar_dense`` scalar escape (observed by the aten census). Both
    mechanisms feed the SAME per-trace side tables so the runnable descriptor witnesses
    every source class -- input, internal op, bound/unbound param, bound/unbound buffer --
    by its capture-time digest through one uniform pass.

    An escape dispatched from TorchLens's own op-logging internals (a metadata read of a
    freshly-produced op output) is NOT a user escape and is skipped, so the fail-closed
    INCOMPLETE gates never fire on TorchLens's own reads.
    """

    if _escape_source_is_torchlens_internal():
        return
    is_bool = source.dtype is torch.bool
    label = get_tensor_label(source)
    if not isinstance(label, str):
        # An UNATTRIBUTABLE escape: a real captured tensor whose value left to the
        # host but which carries no resolvable capture label (a ``.data`` alias --
        # its fresh Python tensor object has no TorchLens metadata, and its own
        # producing op is orphan-pruned).
        if is_bool:
            # A pruned, unlabelled bool predicate is covered by NO net -> fail closed.
            _HOST_ESCAPE_UNATTRIBUTABLE_BOOL.add(trace)
            return
        if invisible:
            # A census-invisible conversion of an unlabelled tensor leaves no source
            # slot and no reliable scalar (may be multi-element): fail closed. (Never
            # call ``.item()`` here -- it would emit a stray dispatch under the census.)
            _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(trace)
            return
        # A census scalar escape extracts exactly one value; record it so the runnable
        # producer can match it to the internal-sink op (or capture-state slot) that
        # produced it and witness that source value-free.
        try:
            escaped = source.detach().item()
        except (RuntimeError, ValueError, TypeError):
            return
        if isinstance(escaped, bool):
            return
        values = _HOST_ESCAPE_UNATTRIBUTABLE_VALUES.get(trace)
        if values is None:
            values = set()
            _HOST_ESCAPE_UNATTRIBUTABLE_VALUES[trace] = values
        values.add(escaped)
        return
    sources = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
    if sources is None:
        sources = set()
        _HOST_ESCAPE_SOURCE_LABELS[trace] = sources
    sources.add(label)
    # A BOOL predicate source stays in the main set (the pruned-RNG control-flow
    # detector consumes it) but is tracked here so the runnable producer excludes an
    # unresolved bool predicate from the tensor-op INCOMPLETE gate (it is the
    # control-witness / conditional / loop / pruned-RNG net's domain).
    if is_bool:
        bool_sources = _HOST_ESCAPE_BOOL_SOURCE_LABELS.get(trace)
        if bool_sources is None:
            bool_sources = set()
            _HOST_ESCAPE_BOOL_SOURCE_LABELS[trace] = bool_sources
        bool_sources.add(label)
    # A registered buffer/parameter source (identified by a non-None state address) is
    # witnessed by its state slot digest. Record BOTH the raw label (so the producer
    # does not close an unresolved orphan-pruned STATE label as a pruned tensor-op
    # chain) AND the state_dict name/address (so the producer witnesses the state slot
    # even when the state ALSO feeds a traced graph op -- bound-ness never exempts the
    # escape witness).
    meta = get_tensor_meta(source)
    address = getattr(meta, "address", None) if meta is not None else None
    if address is not None:
        state_sources = _HOST_ESCAPE_STATE_SOURCE_LABELS.get(trace)
        if state_sources is None:
            state_sources = set()
            _HOST_ESCAPE_STATE_SOURCE_LABELS[trace] = state_sources
        state_sources.add(label)
        state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
        if state_names is None:
            state_names = set()
            _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
        state_names.add(str(address))


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


def _operator_base_name(func: Any) -> str:
    """Return a dispatcher operator's namespace+base name with the overload stripped.

    ``_operator_name`` yields the fully-qualified ``aten.<op>.<overload>`` (e.g.
    ``aten.equal.default``); this drops the trailing ``.<overload>`` so a value-escape
    op is matched by its overload-independent base (``aten.equal``) against
    ``HOST_ESCAPE_OPERATORS``. A name with no overload segment is returned unchanged.
    """

    name = _operator_name(func)
    if name.count(".") >= 2:
        return name.rsplit(".", 1)[0]
    return name


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
        in_scope = False
        try:
            in_scope = (
                threading.get_ident() == self.state.owner_thread_id
                and _state._logging_enabled
                and _state._active_trace is self.state.trace
                and _is_aten_operator(func)
            )
            if in_scope and self.state.census:
                owner = _active_token()
                callsite = (
                    _dispatch_callsite() if owner is None or owner.func_call_id is None else None
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
        result = func(*args, **(kwargs or {}))
        # Escape recording needs the OUTPUT: a tensor->host escape is any aten dispatch
        # returning a NON-TENSOR host value from a tensor operand (equal/allclose/
        # is_nonzero/_local_scalar_dense). Recorded after redispatch so the result is
        # observable; still gated to the owner thread / active trace / logging window.
        if in_scope and self.state.record_escapes:
            escape_started = time.perf_counter_ns()
            try:
                _record_host_escape_source(self.state.trace, func, args, result)
            finally:
                self.state.callback_ns += time.perf_counter_ns() - escape_started
        return result


def _make_invisible_escape_wrapper(original: Any, state: _WitnessState) -> Any:
    """Wrap a tensor->host conversion method to record its SOURCE, then call through.

    The wrapper records the receiver tensor (the escape SOURCE) into the shared escape
    tables, gated to the owner thread / active trace / logging-enabled window so a
    TorchLens-internal conversion (run under ``pause_logging``) is never mistaken for a
    user escape. It always calls the original method unchanged, so values, goldens, and
    outputs are byte-identical.
    """

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if (
            isinstance(self, torch.Tensor)
            and threading.get_ident() == state.owner_thread_id
            and _state._logging_enabled
            and _state._active_trace is state.trace
        ):
            _record_escape_source_tensor(state.trace, self, invisible=True)
        return original(self, *args, **kwargs)

    return wrapper


def _make_invisible_escape_property(descriptor: Any, state: _WitnessState) -> property:
    """Wrap a zero-copy buffer PROPERTY to record its SOURCE tensor, then read through.

    Used for ``__cuda_array_interface__`` (a non-callable getset descriptor the method
    patch cannot wrap). The property getter records the receiver tensor as an escape
    source under the same active-forward gate, then delegates to the original descriptor
    so the returned value is byte-identical.
    """

    def getter(self: torch.Tensor) -> Any:
        if (
            isinstance(self, torch.Tensor)
            and threading.get_ident() == state.owner_thread_id
            and _state._logging_enabled
            and _state._active_trace is state.trace
        ):
            _record_escape_source_tensor(state.trace, self, invisible=True)
        return descriptor.__get__(self, torch.Tensor)

    return property(getter)


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

    originals: dict[str, Any] = {}
    for name in INVISIBLE_HOST_ESCAPE_FUNCS:
        original = getattr(torch.Tensor, name, None)
        if original is None:
            continue
        try:
            setattr(torch.Tensor, name, _make_invisible_escape_wrapper(original, state))
        except (TypeError, AttributeError):
            continue
        originals[name] = original
    property_originals: dict[str, Any] = {}
    for name in INVISIBLE_HOST_ESCAPE_PROPERTIES:
        descriptor = type(torch.Tensor).__dict__.get(name) or torch.Tensor.__dict__.get(name)
        if descriptor is None or not hasattr(descriptor, "__get__"):
            continue
        try:
            setattr(torch.Tensor, name, _make_invisible_escape_property(descriptor, state))
        except (TypeError, AttributeError):
            continue
        property_originals[name] = descriptor
    try:
        yield
    finally:
        for name, original in originals.items():
            try:
                setattr(torch.Tensor, name, original)
            except (TypeError, AttributeError):
                pass
        for name, descriptor in property_originals.items():
            try:
                setattr(torch.Tensor, name, descriptor)
            except (TypeError, AttributeError):
                pass


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
    # A runnable capture additionally observes census-INVISIBLE ``.tolist()`` /
    # ``.numpy()`` / ``__array__`` escapes via a scoped method patch so every escape
    # mechanism feeds one uniform source-witness pass. The patch is a pure observer,
    # restored unconditionally, and is skipped entirely for the non-runnable census path.
    if record_escapes:
        with _observe_invisible_host_escapes(state), mode_context:
            try:
                yield
            finally:
                if mode == "shadow":
                    _finalize_census(state)
        return
    with mode_context:
        try:
            yield
        finally:
            if mode == "shadow":
                _finalize_census(state)

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

Thread posture (r43, JMT-locked): the aten census is OWNER-thread-scoped (a
``TorchDispatchMode`` is thread-local), while the mode-independent tensor->host
escape belt (method/module/property/storage patches) fires on EVERY thread and is
the designated CROSS-thread observer. The belt collapses to ONE fail-closed
owner-vs-non-owner rule: the OWNER thread keeps the precise attribution ladder
(gated on ``_state._logging_enabled``); ANY NON-OWNER thread that TOUCHES A
CAPTURED TENSOR during the armed forward window permanently ceilings the artifact
to ``unverifiable`` (+ ``not_applicable``). Captured membership is decided by
:func:`_nonowner_touch_is_captured` (label OR registered-state OR dispatch-origin
ledger OR STORAGE IDENTITY via the true-original ``untyped_storage``/``data_ptr``
accessors that bypass every torchlens wrapper). The non-owner gate keys on the
per-capture ``belt_armed`` flag, NEVER the racy ``_logging_enabled`` toggle (which
the owner flips constantly under ``pause_logging``). A benign non-owner thread that
never touches a captured tensor -- or that only reads its OWN uncaptured tensors --
records nothing and stays ``verified``. This ONE rule subsumes the r42 hon2_1
(raw ``_thread``), hon2_2 (owner-derived alias on a worker), hon2_3 (the
``pause_logging`` toggle race), and hon2_4 (the string hook) findings.
"""

from __future__ import annotations

import functools
import inspect
import sys
import threading
import time
import types
import warnings
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch._ops as _torch_ops  # r47 hon2_1: enumerate the ``torch.ops.*`` __call__ classes
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)
from torch.utils._python_dispatch import TorchDispatchMode

from ...utils._torch_compat import tensor_version_or_none
from ...utils._torch_symbols import torch_attr
from ...utils._callable_safety import private_c_forward_op_module_names
from ... import _state
from ..._errors import TorchLensCaptureGapWarning
from ._tl import get_buffer_address, get_tensor_label, get_tensor_meta
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
    """One aten dispatcher event and its exact live wrapper owner, if any.

    r35 I2: every event carries a lifecycle ``outcome`` -- ``started`` (armed),
    then ``returned_tensor`` / ``returned_host_or_none`` / ``raised`` -- so the
    runnable ledger can discharge every observed event as an accounted call, an
    exact witness, an audited opaque boundary, or an explicit incomplete fact.
    Only safe facts are recorded for a raise: operator, owner identity, and the
    exception type module+qualname -- never exception objects/messages/tracebacks.
    """

    operator: str
    owner: ExpectedOriginalToken | None
    callsite: _DispatchCallsite | None
    in_replacement_hook: bool = False
    mutates: bool = False
    state_view_accessor: bool = False
    outcome: str = "started"
    exception_type: str | None = None
    contained_view: bool = False
    """``aten.as_strided`` whose result byte span is contained in its operand's (r37)."""


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
    ledger: bool = False
    # r43: armed for the entire forward window (SAME lifetime as
    # ``_observe_invisible_host_escapes``), cleared in its ``finally``. The
    # non-owner captured-tensor belt gates on THIS flag, never the racy global
    # ``_state._logging_enabled`` (which the owner flips under ``pause_logging``),
    # closing the hon2_3 pause-race coin-flip.
    belt_armed: bool = False
    # (source_tensor, version_at_escape, byte_snapshot) for each mutable zero-copy alias
    # (``numpy`` / ``__array__``) handed to the host this forward. Checked for host write-back
    # at forward end. Strong refs keep the aliased storage alive until the comparison.
    writeback_watch: list[tuple[torch.Tensor, int | None, torch.Tensor]] = field(
        default_factory=list
    )


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

# r37 INV-1 (hon2_2): the former ``_HOST_ESCAPE_UNATTRIBUTABLE_VALUES`` table -- scalar
# values of unlabelled escapes, discharged by value-equality against sinks/state -- is
# REMOVED. Scalar value equality is not a provenance proof (a colliding constant sink or
# threshold buffer silently blessed a changed-input run as VERIFIED). Unlabelled escape
# sources now resolve through the positive attribution ladder in
# ``_record_escape_source_tensor`` (direct state alias -> dispatch origins) or fail closed.
#
# BOOL escape sources are NOT recorded as values anywhere by the census: a ``bool(...)``
# truth-test steering control flow is the control-witness / conditional / loop /
# pruned-RNG net's domain, not the tensor-derived scalar net's. An UNATTRIBUTABLE bool
# escape whose predicate is NOT covered by any of those nets is recorded in
# ``_HOST_ESCAPE_UNATTRIBUTABLE_BOOL`` below so the runnable producer can fail closed.

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

MUTABLE_ALIAS_ESCAPE_FUNCS = frozenset({"numpy", "__array__"})
"""Census-invisible conversions that hand back a ZERO-COPY, MUTABLE host alias sharing the
source tensor's storage.

``tensor.numpy()`` (without ``force=True``) and ``np.asarray(tensor)`` / ``tensor.__array__()``
(without a dtype conversion) return a NumPy array that aliases the tensor's memory. A host WRITE
through that array (``t.numpy()[0] = 99``) mutates the source tensor's BYTES but emits NO aten
dispatch and bumps NO torch version counter, so the sparse replay recomputes the pre-write value
and would falsely VERIFY. These funcs are therefore additionally bracketed with a
before/after byte+version snapshot (see ``_observe_invisible_host_escapes``): a source whose bytes
changed with its version UNCHANGED was host-mutated through the alias -> opaque write-back ->
UNVERIFIABLE. ``.tolist()`` copies into Python lists (writes never reach the source) and
``__dlpack__`` is handled by source-witnessing, so neither needs the write-back watch. A read-only
``.numpy().sum()`` leaves the bytes unchanged and stays honestly VERIFIED.
"""

STORAGE_BRIDGE_ESCAPE_FUNCS = frozenset({"untyped_storage", "storage", "data_ptr"})
"""Census-invisible tensor methods that hand the host a ZERO-COPY handle onto the source
tensor's raw storage, through which a host WRITE can mutate the tensor's bytes with no aten
dispatch, no version bump, and no escape record (r14-H3).

``tensor.untyped_storage()`` / ``tensor.storage()`` return a Storage object aliasing the
tensor's memory (``s.fill_(0)`` / ``s[0] = 99`` mutate the source), and ``tensor.data_ptr()``
hands out the raw data pointer that ``ctypes`` / a foreign kernel writes through directly. Like
``.numpy()`` / ``.data`` these bypass the dispatcher entirely, so a write-back leaves the sparse
replay recomputing the pre-write value and would falsely VERIFY. They are therefore bracketed with
the SAME before/after byte snapshot as the mutable numpy alias (see ``_observe_invisible_host_escapes``
-> ``_check_writeback_watch``): a source whose WHOLE-STORAGE bytes changed after exposure -> opaque
host write-back -> UNVERIFIABLE. ``untyped_storage()`` / ``storage()`` are WATCH-ONLY -- they are
NOT recorded as value-escape sources, because a read-only storage identity / contiguity check
exposes NO scalar value and must stay VERIFIED (no over-trigger); a read-only exposure leaves the
bytes unchanged and stays honestly VERIFIED. ``data_ptr()`` is the exception (r15-H1): it hands out
a RAW pointer that a foreign READ (baking a stale literal) or a post-snapshot WRITE can use with no
observable trace at all, so a genuine user ``data_ptr()`` call fails closed to UNVERIFIABLE (see
``_HOST_ESCAPE_RAW_POINTER``), never relying on the byte watch alone.
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

HOST_VALUE_ESCAPE_METHODS = frozenset(
    {
        # Tensor->Python SCALAR numeric protocol: each lowers to ``aten._local_scalar_dense``
        # under the census, BUT torch's own tensor string formatting (``_tensor_str._str``,
        # backing ``__repr__``/``__str__``/``print``) runs its body under
        # ``_disable_current_modes()``, which POPS the census TorchDispatchMode. A method
        # patch fires regardless of dispatch-mode state (measured E1/E6), so it is the
        # mode-independent belt that closes the ``str()``/``repr()``/``print()`` blind spot.
        "item",
        "__bool__",
        "__int__",
        "__float__",
        "__index__",
        "__complex__",
        # Pure tensor->``bool`` predicates: ``torch.equal`` under disabled modes bypasses the
        # census entirely (measured E6), so the Tensor-method spellings are patched too.
        "equal",
        "allclose",
        "is_nonzero",
    }
)
"""Tensor **methods** that read a captured tensor's VALUE out to the Python host (r39 hon2_1).

These are the mode-independent belt for the aten census: every one is patched on
``torch.Tensor`` for the duration of one runnable forward and records its tensor operand(s)
through the SAME ``_record_escape_source_tensor(..., invisible=True)`` attribution ladder as
the census. Coupled to :data:`HOST_ESCAPE_OPERATORS` by the r39 census<->observer meta-test:
a one-sided addition (a new census op without a method/module observer, or vice versa) fails
CI. ``__repr__``/``__str__``/``__format__`` are deliberately NOT patched -- every string/format
spelling transits patched ``item``/``tolist`` under ``_disable_current_modes`` (E6), so patching
them is redundant and is pinned by regression tests instead.
"""

HOST_VALUE_ESCAPE_MODULE_FUNCS = frozenset({"equal", "allclose", "is_nonzero"})
"""``torch.*`` MODULE predicate spellings of the pure tensor->``bool`` escapes (r39 hon2_1).

The module-level ``equal`` / ``allclose`` / ``is_nonzero`` functions return a raw Python bool
DIRECTLY from the dispatcher and, under an explicit ``_disable_current_modes()`` region, bypass
the census (E6). The module functions are wrapped to record every tensor operand as an escape
source, mirroring the Tensor-method belt.
"""

INPUT_METADATA_PREDICATE_FUNCS = frozenset({"is_contiguous", "stride", "storage_offset"})
"""Tensor LAYOUT-PREDICATE **methods** whose result on a MODEL INPUT can steer unobserved
Python control flow (r27-H2, extended r29-C1). Special-cased (memory_format / full-stride
recording) by :func:`_make_input_metadata_wrapper`; the broader host-value method surface is
:data:`INPUT_METADATA_BOOL_METHODS`.

``x.is_contiguous()`` / ``x.stride()`` / ``x.storage_offset()`` return host values derived
from the input's memory LAYOUT -- which the input contract does NOT check (only shape+dtype),
so a same-shape, same-dtype runtime input with a different layout (a transposed view, or a
slice of a larger buffer whose ``storage_offset`` is non-zero) flips such a branch while the
sparse replay silently follows the CAPTURED arm: a false VERIFIED+ATTESTED. ``storage_offset``
is doubly dangerous because capture CLONES the input leaf and the clone RESETS the offset to
0, so a branch on it would be wrong even for the original-input replay; the fact is recorded
from the RAW pre-clone input the forward actually read, and re-checked against the RAW runtime
input before the executor's detach-clone.

These reads emit no aten dispatch (pure metadata, deliberately excluded from the escape
census -- see ``HOST_ESCAPE_OPERATORS``), so they are observed by the same scoped method
patch as the invisible escapes, but with a DIFFERENT recording rule: only a read whose
receiver is a MODEL-INPUT leaf tensor (or a ``.data`` / ``.detach()`` storage-alias of one,
r31) records a (site, predicate, observed value) fact, so a model that never reads input
layout records nothing and can never over-trigger. ``size``/``dim``/``numel``/``dtype``
derive from shape+dtype and stay covered by the existing input contract; they are
deliberately NOT observed (a real model reads shapes constantly -- patching them buys no
honesty and costs every capture).
"""

INPUT_METADATA_BOOL_METHODS = frozenset(
    {
        "is_conj",
        "is_neg",
        "is_inference",
        "is_pinned",
        "is_shared",
        "is_coalesced",
        "_is_view",
    }
)
"""Host-value-returning tensor-metadata **methods** (beyond the layout trio) NOT pinned by the
shape+dtype input contract (r31, capability-driven accessor table).

Round-30 confirmed false VERIFIED via control flow steered on these accessors, each of which
the shape+dtype contract does NOT pin, so a same-shape/same-dtype runtime input differing in
the property silently replays the captured arm:

* ``is_conj`` / ``is_neg`` -- the conjugate / negative dispatch bit set by ``x.conj()`` /
  ``torch._neg_view(x)``; a same-shape non-conj/non-neg twin flips the branch.
* ``is_inference`` -- whether the tensor was created under ``torch.inference_mode()``.
* ``is_pinned`` / ``is_shared`` -- pinned-memory / shared-memory STORAGE placement.
* ``is_coalesced`` -- sparse-tensor coalesced flag (raises on dense; recorded only on success).
* ``_is_view`` -- whether the receiver is itself a view (structural, treated as autograd-family
  for alias/view attribution -- ``.data`` / ``.detach()`` are always views, so it is recorded
  only for the input LEAF, never a storage-alias).

Each is a boolean method with no value-bearing args; the observer records ``bool(result)`` and
re-checks it on the RAW runtime input. Feature-detected (``getattr``) at install time; an
accessor absent on the running torch is simply skipped. Recorded ONLY for a model-input leaf
(or, for the alias-safe subset, a ``.data`` / ``.detach()`` storage-alias), so a model that
never reads them records nothing -- zero over-trigger by construction.

DELIBERATELY EXCLUDED (shape+dtype-derived or already covered, would only add noise): ``size`` /
``sym_size`` / ``numel`` / ``dim`` / ``ndimension`` / ``element_size`` / ``nelement`` /
``is_contiguous`` variants already in the layout trio / ``is_floating_point`` / ``is_complex`` /
``is_signed`` (dtype-derived). ``data_ptr`` / ``untyped_storage().data_ptr`` fail closed
(r15/r16-C1). ``dtype`` / ``device`` / ``shape`` / ``layout`` are shape+dtype/device covered.
"""

INPUT_METADATA_PROPERTY_NAMES = frozenset(
    {
        "requires_grad",
        "grad_fn",
        "is_leaf",
        "retains_grad",
        "_base",
        "grad",
        "_grad",
        "_version",
        "output_nr",
    }
)
"""Tensor autograd / structural getset PROPERTIES whose read on a MODEL INPUT is witnessed
(r27-H2 ``requires_grad``; r29-C1 adds ``grad_fn`` / ``is_leaf``; r31 adds ``retains_grad`` /
``_base``; r33 adds ``grad`` / ``_grad`` as PRESENCE facts and ``_version`` / ``output_nr`` as
INT facts).

r33 additions (each a control decision the shape+dtype contract does NOT pin, confirmed by
oracle to falsely VERIFY otherwise):

* ``grad`` / ``_grad`` -- ``if x.grad is None:`` steers on whether a gradient has been
  accumulated on the input leaf. The detach-clone erases it (a fresh clone has ``grad=None``),
  so it is witnessed as a PRESENCE boolean (the exact gradient tensor is not comparable across
  runs) and re-checked on the RAW pre-clone runtime input. ``_grad`` is the private alias.
* ``_version`` -- ``if x._version == 0:`` steers on the input leaf's in-place mutation counter.
  The detach-clone RESETS ``_version`` to 0, so like ``storage_offset`` it is read from the RAW
  pre-clone runtime input and witnessed as its INT value.
* ``output_nr`` -- the autograd output index of the tensor; witnessed as its INT value.

``x.requires_grad`` / ``x.grad_fn`` / ``x.is_leaf`` / ``x.retains_grad`` / ``x._base`` are
non-callable getset descriptors read by ``if x.requires_grad:`` / ``if x.grad_fn is not None:``
/ ``if x.is_leaf:`` / ``if x.retains_grad:`` / ``if x._base is not None:`` control flow. The
runnable executor detach-clones bound inputs, ERASING the runtime autograd state (a detached
clone has ``requires_grad=False``, ``grad_fn=None``, ``is_leaf=True``, ``retains_grad=False``,
``_base=None``), so without these facts an autograd-branching model would falsely VERIFY for a
runtime input whose autograd state differs. The scoped property patch records the observed
value only for MODEL-INPUT receivers (``grad_fn`` / ``_base`` as a PRESENCE bool -- the exact
backward object / base tensor is not comparable across runs) and must preserve descriptor SET
semantics for the writable ``requires_grad`` (``x.requires_grad = True`` inside a forward still
works); ``grad_fn`` / ``is_leaf`` / ``retains_grad`` / ``_base`` are read-only.
"""

_INPUT_METADATA_PRESENCE_PROPERTY_NAMES = frozenset({"grad_fn", "_base", "grad", "_grad"})
"""Autograd/structural PROPERTIES recorded as a PRESENCE boolean (``value is not None``): the
exact backward object (``grad_fn``), base tensor (``_base``), and accumulated gradient
(``grad`` / ``_grad``, r33) are not comparable across runs, so only their presence witnesses
the control decision."""

_INPUT_METADATA_INT_PROPERTY_NAMES = frozenset({"_version", "output_nr"})
"""PROPERTIES recorded as their INT value (r33): the in-place mutation counter (``_version``)
and the autograd output index (``output_nr``). Neither is pinned by the shape+dtype contract;
both compare exactly across runs."""

INPUT_METADATA_GRAD_PROPERTY = "requires_grad"
"""Deprecated single-name alias retained for back-compat; see ``INPUT_METADATA_PROPERTY_NAMES``."""

# --- Accessor FAMILIES governing alias/view attribution (r31) -------------------------------
#
# A metadata read whose receiver is not the input leaf OBJECT itself is attributed by the
# receiver's relationship to an input leaf, which differs per accessor family:

_INPUT_METADATA_LAYOUT_NAMES = frozenset({"is_contiguous", "stride", "storage_offset"})
"""LAYOUT accessors: value DIFFERS between a leaf and a derived view (``x.t().stride()`` !=
``x.stride()``), so a derived-view read fails closed (cannot be re-derived from the runtime
leaf); a ``.data`` / ``.detach()`` storage-alias (identical geometry) records the leaf fact."""

_INPUT_METADATA_ALIAS_SAFE_NAMES = _INPUT_METADATA_LAYOUT_NAMES | frozenset(
    {"is_conj", "is_neg", "is_inference", "is_pinned", "is_shared", "is_coalesced"}
)
"""Accessors attributed by STORAGE IDENTITY (r31, hole A). A read on a tensor sharing an input
leaf's storage with IDENTICAL geometry (``.data`` / ``.detach()``) records the leaf fact -- the
value is provably equal to a direct leaf read; a storage-alias with DIFFERENT geometry (a
derived view) fails closed. This closes the ``x.data.storage_offset()`` /
``x.detach().is_contiguous()`` class the object-identity map missed."""

_INPUT_METADATA_CONJ_NEG_NAMES = frozenset({"is_conj", "is_neg"})
"""ALIAS-SAFE accessors whose value FLIPS on a same-geometry conjugate/negative VIEW (r33 F5).
A ``.conj()`` / ``torch._neg_view()`` of an input leaf shares its storage AND geometry but sets
this dispatch bit, so a same-storage/same-geometry receiver is EQUIVALENT only if its bit also
equals the leaf's; a bit mismatch is a genuine derived view and fails closed (else a wrong leaf
fact forces a false divergence on the original complex input)."""

_INPUT_METADATA_VIEW_FAIL_AUTOGRAD_NAMES = frozenset(
    {"is_leaf", "retains_grad", "_base", "_is_view", "output_nr"}
)
"""AUTOGRAD / structural accessors whose read on a DERIVED VIEW of an input leaf fails closed
(r31 hole C). On a ``.data`` / ``.detach()`` storage-alias these are CONSTANT (detached:
``is_leaf=True``, ``retains_grad=False``, ``_base`` set, ``_is_view`` True), input-INDEPENDENT,
no hole -- IGNORED. On a DERIVED VIEW (``x.view(-1).is_leaf``, ``retains_grad`` on a non-leaf
view) the state is not re-derivable from the runtime leaf, so the read fails closed. The
framework-vs-user discriminator is the ``_base``-in-sites linkage itself: TorchLens's own
per-op capture bookkeeping NEVER reads THESE four accessors on any input-derived view
(verified empirically -- zero internal reads), so a match is a genuine USER Python view read.
Uses only the CHEAP ``_base`` attribute check -- no per-op storage-pointer cost."""

_INPUT_METADATA_LEAF_ONLY_AUTOGRAD_NAMES = frozenset(
    {"requires_grad", "grad_fn", "grad", "_grad", "_version"}
)
"""AUTOGRAD accessors witnessed ONLY on the input LEAF (object identity), never attributed to a
view or alias (r31). TorchLens's OWN per-op capture bookkeeping reads ``output.grad_fn`` and
``output.requires_grad`` on EVERY op output -- INCLUDING input-derived views (``x[i]`` /
``x.view(-1)``) -- while logging is enabled and unmarked (verified: grad_fn ~30x, requires_grad
~10x on input views for a trivial model). Those framework reads are indistinguishable at the
Python descriptor from a genuine user view read, so BOTH a fail-closed view downgrade AND a
leaf-attributed view record would over-trigger a normal model (a ``requires_grad``-oblivious
model would spuriously DIVERGE on a ``requires_grad``-changed input). The LOCKED
allowlist-by-construction principle forbids a spoofable stack-filename discriminator, and the
per-op autograd reads live outside this witness surface (in ``backend.py``), so these two
accessors stay LEAF-ONLY: a direct ``x.requires_grad`` / ``x.grad_fn`` leaf read is witnessed
and diverges correctly; a read reached ONLY through a view is a documented residual. (``grad_fn``
is not even view-invariant -- a view of a leaf has a ``ViewBackward`` grad_fn the leaf lacks --
so a leaf record would be wrong regardless.) r33 adds ``grad`` / ``_grad`` / ``_version`` here
on the same principle: ``grad`` / ``_grad`` / ``_version`` are not view-invariant (a view has a
distinct ``grad`` slot and a fresh ``_version``), so a view read is a documented leaf-only
residual, never a leaf-attributed record."""


# Per-trace map from an input leaf's BASE-storage data pointer to the list of
# ``(site, size, stride, storage_offset)`` geometries of the model-input leaves that own it.
# Kept in a weak-keyed module table (NOT ``trace.__dict__``) so it never enters the portable
# schema and needs no scrub allow-list entry (r31); dropped when the Trace is GC'd.
_RUNNABLE_INPUT_STORAGE_SITES: "weakref.WeakKeyDictionary[Any, dict[int, list[Any]]]" = (
    weakref.WeakKeyDictionary()
)

_ALIAS_EQUIVALENT = "equivalent"
"""Storage-alias classification: shares an input leaf's storage with IDENTICAL geometry
(``.data`` / ``.detach()``) -- a metadata read on it equals a direct leaf read."""

_ALIAS_DERIVED_VIEW = "derived_view"
"""Storage-alias classification: shares an input leaf's storage but with DIFFERENT geometry
(a derived view the sparse replay never re-derives) -- fails closed."""


def record_runnable_input_storage_sites(
    trace: Any, tensor_leaves: "list[tuple[torch.Tensor, Any]]"
) -> None:
    """Index model-input TENSOR leaves by BASE-storage identity for alias-read witnessing (r31).

    The object-identity map (``_runnable_input_tensor_sites``) misses a metadata read routed
    through a ``.data`` / ``.detach()`` alias (or any derived view) of an input leaf: the alias
    is a distinct Python object sharing the leaf's STORAGE but neither the leaf object nor a
    ``_base``-linked view. This companion map lets :func:`_classify_input_storage_alias`
    attribute such a read by storage identity + geometry. Storage pointers and geometry are
    read under the internal-scalar-read marker so the live ``untyped_storage`` / ``data_ptr`` /
    ``stride`` / ``storage_offset`` patches treat them as TorchLens-internal (no spurious
    escape record, no fail-closed data_ptr trip). Runs only for runnable captures; stores no
    tensors.
    """

    if not tensor_leaves:
        return
    # INV-2 annotation (r37): this map keys candidate input-leaf sites by storage
    # POINTER for attribution-only lookups (a ``.data``/view metadata read resolves
    # to its leaf). A pointer miss fails CLOSED (no attribution -> the fail-closed
    # nets keep the run honest), never proves disjointness, so identity keying is
    # sound without the absolute-interval engine.
    storage_sites: dict[int, list[Any]] = {}
    # ``pause_logging`` suppresses OP CAPTURE (``storage_offset`` / ``untyped_storage`` are
    # torch-function-wrapped and would otherwise be logged as spurious ops, shifting call ids);
    # ``internal_scalar_read`` marks the reads internal for the escape census / metadata patches.
    with _state.pause_logging(), internal_scalar_read():
        for tensor, site in tensor_leaves:
            try:
                ptr = tensor.untyped_storage().data_ptr()
                geometry = (
                    tuple(tensor.shape),
                    tuple(int(v) for v in tensor.stride()),
                    int(tensor.storage_offset()),
                )
                # r33 F5: the LEAF's conj/neg dispatch bits. A conj/neg VIEW of an input leaf
                # shares its storage AND geometry but flips these bits; recording the leaf's
                # true bits here lets the classifier reject such a same-geometry view instead
                # of misattributing its ``is_conj``/``is_neg`` read as a leaf fact.
                conj_neg = (bool(tensor.is_conj()), bool(tensor.is_neg()))
            except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
                continue
            storage_sites.setdefault(ptr, []).append((site, *geometry, *conj_neg))
    if storage_sites:
        _RUNNABLE_INPUT_STORAGE_SITES[trace] = storage_sites


def _classify_input_storage_alias(
    trace: Any, source: torch.Tensor
) -> "tuple[str | None, Any, tuple[bool, bool] | None]":
    """Classify ``source`` against the input-leaf storage map (r31, holes A/C).

    Returns ``(_ALIAS_EQUIVALENT, site, leaf_conj_neg)`` when ``source`` shares an input leaf's
    base storage with IDENTICAL geometry (a ``.data`` / ``.detach()`` alias -- a metadata read on
    it equals a direct leaf read), ``(_ALIAS_DERIVED_VIEW, site, None)`` when it shares the
    storage with DIFFERENT geometry (a derived view the replay never re-derives), or
    ``(None, None, None)`` when it does not alias any input leaf's storage (a genuine unrelated
    activation). ``leaf_conj_neg`` is the matched leaf's ``(is_conj, is_neg)`` bits so the caller
    can reject a same-geometry conj/neg view (r33 F5). Storage-pointer/geometry reads run under
    the internal marker so the live patches stay pass-through and cannot recurse back into
    observation.
    """

    storage_sites = _RUNNABLE_INPUT_STORAGE_SITES.get(trace)
    if not storage_sites:
        return (None, None, None)
    # ``pause_logging`` so the ``untyped_storage`` / ``storage_offset`` reads below are not
    # captured as spurious ops mid-forward; ``internal_scalar_read`` keeps them off the census.
    with _state.pause_logging(), internal_scalar_read():
        try:
            ptr = source.untyped_storage().data_ptr()
        except (RuntimeError, AttributeError, TypeError, NotImplementedError):
            return (None, None, None)
        candidates = storage_sites.get(ptr)
        if not candidates:
            return (None, None, None)
        try:
            geometry: Any = (
                tuple(source.shape),
                tuple(int(v) for v in source.stride()),
                int(source.storage_offset()),
            )
        except (RuntimeError, TypeError, ValueError):
            geometry = None
    for site, size, stride, offset, leaf_conj, leaf_neg in candidates:
        if geometry is not None and geometry == (size, stride, offset):
            return (_ALIAS_EQUIVALENT, site, (leaf_conj, leaf_neg))
    return (_ALIAS_DERIVED_VIEW, candidates[0][0], None)


def _input_base_tensor(source: torch.Tensor) -> "torch.Tensor | None":
    """Return ``source._base`` read under the internal marker (r31).

    ``_base`` is a witnessed getset PROPERTY replaced by a recording descriptor during a
    runnable forward; reading it here for the view-linkage check must go under the
    internal-scalar-read marker so the recording getter treats it as a TorchLens-internal read
    (no recursion back into observation).
    """

    with _state.pause_logging(), internal_scalar_read():
        try:
            base = source._base
        except (RuntimeError, AttributeError):
            return None
    return base if isinstance(base, torch.Tensor) else None


def _record_input_metadata_read_at_site(trace: Any, site: Any, name: str, value: Any) -> None:
    """Record one metadata-read fact against a resolved MODEL-INPUT leaf site.

    Facts accumulate per input site into a runtime-only Trace stash the runnable producer
    serializes as declared witness facts. A repeated read of the same predicate overwrites --
    tensor metadata is stable across one forward, so the values are identical unless an in-place
    layout change occurred, in which case the LAST observed value is the one nearest the branch.
    """

    facts = trace.__dict__.setdefault("_runnable_input_metadata_reads", {})
    site_facts = facts.setdefault(site, {})
    site_facts[name] = value


def _record_input_metadata_read(trace: Any, source: torch.Tensor, name: str, value: Any) -> None:
    """Record one metadata-read fact against the model-input leaf that IS ``source`` (by identity).

    The receiver is attributed via the object-identity map recorded at capture start
    (``_record_runnable_input_tensor_sites``); a read on any other tensor records nothing here
    (storage-alias attribution is handled by :func:`_observe_input_metadata_read`).
    """

    sites = trace.__dict__.get("_runnable_input_tensor_sites")
    if not sites:
        return
    site = sites.get(id(source))
    if site is None:
        return
    _record_input_metadata_read_at_site(trace, site, name, value)


def _observe_input_metadata_read(trace: Any, source: torch.Tensor, name: str, value: Any) -> None:
    """Attribute one metadata read to a model-input leaf, an alias, or a fail-closed view (r31).

    Cases, in order:

    * The receiver IS a model-input leaf (object identity) -> record a re-checkable
      (site, predicate, value) fact.
    * LEAF-ONLY AUTOGRAD (``requires_grad`` / ``grad_fn``): TorchLens's own per-op bookkeeping
      reads these on input-derived views while logging is enabled (verified), indistinguishable
      from a user view read, so a non-leaf read is IGNORED (leaf-only; documented residual).
    * VIEW-FAIL AUTOGRAD / structural (``is_leaf`` / ``retains_grad`` / ``_base`` / ``_is_view``):
      a read on a DERIVED VIEW of an input leaf (``retains_grad`` on a non-leaf view, r31 hole C)
      is attributed by the CHEAP ``_base``-in-sites linkage and fails closed -- the view's state
      is not re-derivable from the runtime leaf. A ``.data`` / ``.detach()`` storage-alias
      (``_base`` None) is IGNORED (CONSTANT/detached, input-independent, no hole). These four are
      NEVER read internally on an input view (verified), so a ``_base`` match is a genuine user
      Python view read -- the framework-vs-user discriminator is the linkage itself.
    * ALIAS-SAFE family (layout methods + ``is_conj`` / ``is_neg`` / ``is_inference`` /
      ``is_pinned`` / ``is_shared`` / ``is_coalesced``): attributed by STORAGE IDENTITY. A read
      on a ``.data`` / ``.detach()`` storage-alias with IDENTICAL geometry (r31 hole A) records
      the leaf fact -- its value provably equals a direct leaf read. A storage-alias with
      DIFFERENT geometry (a derived view, ``x.t().is_contiguous()``) fails closed. Only
      Python-level reads reach this patch; torch's internal C++ layout reads bypass it, so a
      layout-oblivious model records nothing.
    * Anything else (a genuinely new activation not aliasing an input) -> ignore.
    """

    sites = trace.__dict__.get("_runnable_input_tensor_sites")
    if not sites:
        return
    if id(source) in sites:
        _record_input_metadata_read(trace, source, name, value)
        return
    if name in _INPUT_METADATA_LEAF_ONLY_AUTOGRAD_NAMES:
        # ``requires_grad`` / ``grad_fn`` are read by TorchLens's own per-op bookkeeping on
        # input-derived views; witnessed on the LEAF only (see the set docstring).
        return
    if name in _INPUT_METADATA_VIEW_FAIL_AUTOGRAD_NAMES:
        base = _input_base_tensor(source)
        if base is not None and id(base) in sites:
            _INPUT_METADATA_VIEW_READ.add(trace)
        return
    if name in _INPUT_METADATA_ALIAS_SAFE_NAMES:
        kind, site, leaf_conj_neg = _classify_input_storage_alias(trace, source)
        if kind == _ALIAS_EQUIVALENT:
            # r33 F5 (over-trigger fix): a conj/neg VIEW shares an input leaf's storage AND
            # geometry but FLIPS the conj/neg dispatch bit. Geometry alone would record the
            # view's ``is_conj=True`` as a LEAF fact, which the RAW runtime leaf (``is_conj``
            # False) then contradicts -> a forced FALSE divergence on the ORIGINAL input
            # (complex models permanently diverged, r31 regression). For ``is_conj``/``is_neg``
            # the observed bit must EQUAL the leaf's; a same-geometry bit MISMATCH is a genuine
            # derived (conj/neg) view and fails closed rather than misrecording a leaf fact.
            if name in _INPUT_METADATA_CONJ_NEG_NAMES and leaf_conj_neg is not None:
                leaf_bit = leaf_conj_neg[0] if name == "is_conj" else leaf_conj_neg[1]
                if bool(value) != bool(leaf_bit):
                    _INPUT_METADATA_VIEW_READ.add(trace)
                    return
            _record_input_metadata_read_at_site(trace, site, name, value)
        elif kind == _ALIAS_DERIVED_VIEW:
            _INPUT_METADATA_VIEW_READ.add(trace)
        else:
            base = _input_base_tensor(source)
            if base is not None and id(base) in sites:
                _INPUT_METADATA_VIEW_READ.add(trace)


def _state_derived_addresses(trace: Any, source: torch.Tensor) -> set[str]:
    """Resolve ``source`` to the registered-state addresses whose storage it aliases (r63 C1).

    Positive-attribution ladder, first hit wins: the buffer meta address stamped at forward
    start (a DIRECT registered-buffer receiver; model inputs carry a label but never an
    address, so an input can never resolve here); the forward-start param storage index
    (``self.w`` and any ``.data`` / view / detach alias of it); the forward-start buffer
    storage index (the ``.data`` / view alias twin for buffers). A miss returns empty --
    an activation receiver records nothing (its geometry is recomputed by the replayed DAG).
    """

    address = get_buffer_address(source)
    if address is not None:
        return {str(address)}
    addresses = _param_derived_addresses(trace, source)
    if addresses:
        return addresses
    buffer_storage_addresses = getattr(trace, "_buffer_storage_addresses", None)
    if buffer_storage_addresses:
        ptr = _escape_storage_ptr(source)
        if ptr is not None and ptr in buffer_storage_addresses:
            return {str(buffer_storage_addresses[ptr])}
    return set()


def _observe_state_metadata_read(trace: Any, source: torch.Tensor, read_kind: str) -> None:
    """Attribute one PHYSICAL-metadata read on registered state as a state escape (r63 C1).

    Closes the four r62/r63 attribution gaps: ``is_contiguous`` / ``stride`` /
    ``storage_offset`` / ``is_conj`` (+ the ``is_neg`` lazy-bit sibling) on a registered
    param/buffer previously routed ONLY through the model-input observer, which ignores
    state -- so a model branching on ``self.weight.is_contiguous()`` produced no witness and
    the transport-normalized replay reported a false ``verified``. A resolved read now:

    * joins ``_HOST_ESCAPE_STATE_SOURCE_NAMES`` -- the slot is digest-witnessed by PASS A
      (``unbound_state_escape:<name>`` fact; changed staged state -> ``unverifiable``),
      exactly like a ``self.threshold.item()`` value read; and
    * records its READ KIND in the per-slot metadata ledger consumed by the escape-gated
      producer preflight (a read dim that was non-canonical at capture refuses the save;
      an unread non-canonical slot -- the channels-last population -- stays saveable).

    A model-input leaf receiver is the input nets' domain and is skipped; an unresolvable
    receiver (an activation) records nothing here.
    """

    sites = trace.__dict__.get("_runnable_input_tensor_sites")
    if sites and id(source) in sites:
        return
    addresses = _state_derived_addresses(trace, source)
    if not addresses:
        return
    _record_state_metadata_read(trace, addresses, read_kind)


def host_escape_state_metadata_reads(trace: Any) -> dict[str, frozenset[str]]:
    """Return the per-state-name PHYSICAL-metadata read kinds witnessed for one trace."""

    reads = _HOST_ESCAPE_STATE_METADATA_READS.get(trace)
    if not reads:
        return {}
    return {name: frozenset(kinds) for name, kinds in reads.items()}


# --- r65 Cluster X: THE authoritative state-metadata accessor mirror ------------------------
#
# The input-metadata net's authority is the union of four frozen constants
# (INPUT_METADATA_PREDICATE_FUNCS | INPUT_METADATA_BOOL_METHODS | INPUT_METADATA_PROPERTY_NAMES
# | {"storage_nbytes"}), which equals the 20-name ``_INPUT_METADATA_FACT_NAMES`` vocabulary in
# ``torchlens._io.runnable``. r63 mirrored only 5 of those 20 onto registered state, leaving an
# "Nth unwitnessed state read" class open (r64 F2/F3). This table closes the CLASS: every input
# accessor carries an EXPLICIT state disposition, and the wrappers dispatch through the table
# instead of hardcoded name tuples, so a future accessor added to any input constant without a
# state disposition is a RED parity test (T-X1), never a silent gap.

_STATE_ROUTE_READ_KIND = "read_kind"
"""Disposition: the read joins the escape-gated r63 machinery -- the slot is digest-witnessed
(``_HOST_ESCAPE_STATE_SOURCE_NAMES``) and the read KIND enters the per-slot ledger consumed by
the producer preflight, which refuses the save iff the read dim was non-canonical at capture."""

_STATE_ROUTE_DECLARED_FACT = "declared_fact"
"""Disposition: the read records a DECLARED-STATE FACT (r65 F-1 ruling) -- the observed bit is
persisted as a ``state_metadata:<name>`` witness and staging REPRODUCES it (no escape-source
join, no read-kind, no refusal except a fact staging provably cannot reproduce). Escape-gating
``requires_grad`` would refuse every frozen model and is contamination-fragile against
TorchLens's own per-op autograd bookkeeping; the fact route is immune BY CONSTRUCTION: a
spurious internally-triggered fact records the true current bit, staging reproduces exactly
that bit, and nothing ever refuses or diverges from it."""

_STATE_ROUTE_STRUCTURAL = "structural"
"""Disposition: provably covered by another gate; the wrapper records nothing for state."""

STATE_METADATA_MIRROR: "Mapping[str, tuple[str, str]]" = MappingProxyType(
    {
        # -- layout trio (r63, unchanged; ``is_contiguous`` probed with an explicit
        #    memory_format resolves to the ``stride`` row's kind at the wrapper) --
        "is_contiguous": (_STATE_ROUTE_READ_KIND, "contiguous_default"),
        "stride": (_STATE_ROUTE_READ_KIND, "stride_exact"),
        "storage_offset": (_STATE_ROUTE_READ_KIND, "storage_offset"),
        # -- lazy dispatch bits (r63, unchanged) --
        "is_conj": (_STATE_ROUTE_READ_KIND, "is_conj"),
        "is_neg": (_STATE_ROUTE_READ_KIND, "is_neg"),
        # -- storage/creation placement bits (r65): value is a pure function of the slot's
        #    storage, invariant across every view/alias, normalized by transport+staging --
        "is_shared": (_STATE_ROUTE_READ_KIND, "is_shared"),
        "is_pinned": (_STATE_ROUTE_READ_KIND, "is_pinned"),
        "is_inference": (_STATE_ROUTE_READ_KIND, "is_inference"),
        # -- base-storage geometry (r65; closes F3): recorded at storage-handle exposure --
        "storage_nbytes": (_STATE_ROUTE_READ_KIND, "storage_nbytes"),
        # -- autograd/structural family (r65): DIRECT-receiver-only attribution (see
        #    ``_STATE_METADATA_DIRECT_ONLY_NAMES``); ``_base`` presence <=> is-view --
        "_is_view": (_STATE_ROUTE_READ_KIND, "is_view"),
        "_base": (_STATE_ROUTE_READ_KIND, "is_view"),
        "is_leaf": (_STATE_ROUTE_READ_KIND, "is_leaf"),
        "retains_grad": (_STATE_ROUTE_READ_KIND, "retains_grad"),
        "output_nr": (_STATE_ROUTE_READ_KIND, "output_nr"),
        "grad": (_STATE_ROUTE_READ_KIND, "grad_presence"),
        "_grad": (_STATE_ROUTE_READ_KIND, "grad_presence"),
        # -- in-place mutation counter (r65 converged ruling): refuse-on-read of a
        #    transport-lost version -- the read kind maps to ``version_is_zero`` so a read of
        #    a NON-default captured version refuses while a version-0 read stays saveable
        #    (the staged clone reproduces version 0) --
        "_version": (_STATE_ROUTE_READ_KIND, "_version"),
        # -- declared-state facts (r65 F-1 ruling; grad_fn presence is the
        #    contamination-immune twin) --
        "requires_grad": (_STATE_ROUTE_DECLARED_FACT, "requires_grad"),
        "grad_fn": (_STATE_ROUTE_DECLARED_FACT, "grad_fn"),
        # -- sparse-only accessor: RAISES on dense strided state (pass-through, nothing to
        #    record); sparse layouts are refused at bind/save by the layout signature dim --
        "is_coalesced": (
            _STATE_ROUTE_STRUCTURAL,
            "sparse layout refused at bind/save; raises on dense strided state",
        ),
    }
)
"""ONE authoritative mirror: input-metadata accessor name -> (state route, detail).

Keys are EXACTLY the input net's accessor union (pinned by the T-X1 parity test). ``detail``
is the state read KIND for ``read_kind`` rows (the ``_STATE_METADATA_READ_REQUIRED_DIMS``
vocabulary in ``torchlens._runnable_state``), the persisted fact name for ``declared_fact``
rows (the closed ``_STATE_METADATA_FACT_NAMES`` vocabulary in ``torchlens._io.runnable``),
and a documentation pointer for ``structural`` rows.
"""

_STATE_METADATA_DIRECT_ONLY_NAMES = frozenset(
    {
        "requires_grad",
        "grad_fn",
        "is_leaf",
        "retains_grad",
        "_base",
        "_is_view",
        "output_nr",
        "grad",
        "_grad",
        "_version",
    }
)
"""AUTOGRAD/structural accessors attributed ONLY on the DIRECT registered object (the
``nn.Parameter`` / registered-buffer object itself), never through a storage alias or derived
view (r65; the state twin of the input net's leaf-only rule).

Two reasons, mirroring r31/r33: (1) CONTAMINATION -- TorchLens's own per-op bookkeeping reads
``requires_grad`` / ``grad_fn`` / ``_version`` on op outputs (including param-storage-sharing
view outputs like ``self.w[:]``) while logging is enabled; attributing a view's
``grad_fn``-present read to its slot would refuse/ceiling ordinary models (a ``ViewBackward``
on a param view is NOT a slot fact). The known DIRECT-receiver bookkeeping reads are excluded
at their source under the ``internal_scalar_read`` marker (r65). (2) NON-INVARIANCE -- unlike
the alias-safe family, a view's autograd state (``is_leaf`` False, fresh ``_version``, own
``grad`` slot) is NOT a pure function of the slot's canonical form, so a slot-attributed alias
read would be wrong regardless. An alias/view read of this family on state is the documented
residual (contract residual: the state twin of the input-derived-view autograd residual)."""

_STATE_METADATA_ALIAS_SAFE_STATE_NAMES = frozenset(
    {"is_conj", "is_neg", "is_inference", "is_pinned", "is_shared"}
)
"""BOOL-method accessors attributed on state by STORAGE IDENTITY (``_state_derived_addresses``,
r63 semantics): the value on ANY view/alias is a pure function of the slot's storage/creation
placement (a view of a pinned/shared/inference tensor is itself pinned/shared/inference), and
the replay re-derives every view from the canonical staged slot through the recorded DAG, so a
slot-attributed alias read is provably reproducible iff the slot dim was canonical."""

_STATE_METADATA_FACTS: "weakref.WeakKeyDictionary[Any, dict[str, dict[str, bool]]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace DECLARED-STATE fact ledger: state name -> {fact name -> observed bool} (r65 F-1).

Populated by the property wrapper's state branch for ``requires_grad`` (bool value) and
``grad_fn`` (presence bool) reads on DIRECT registered param/buffer receivers. These slots do
NOT join ``_HOST_ESCAPE_STATE_SOURCE_NAMES`` -- a metadata read exposes no bytes; the digest
join is the physical family's belt, not this one's. The runnable producer persists each entry
as a ``state_metadata:<name>`` SHAPE_STRUCTURE_FACT witness and staging reproduces the
recorded ``requires_grad`` bit (``grad_fn`` presence True refuses at save: no staged leaf can
carry a grad_fn). Kept weak-keyed off the schema."""


def _state_direct_address(trace: Any, source: torch.Tensor) -> "str | None":
    """Resolve ``source`` to a state address ONLY when it IS the registered object (r65).

    The DIRECT-receiver discriminator for the autograd/structural family
    (``_STATE_METADATA_DIRECT_ONLY_NAMES``): a registered buffer carries the buffer meta
    address stamped at forward start (a ``.data``/view alias carries none), and a registered
    parameter is an exact-type ``nn.Parameter`` object (op outputs and ``.data``/``detach()``
    aliases are plain ``Tensor``s -- torch ops never construct ``nn.Parameter`` results, so
    exact-type + param-storage membership identifies the registered object; an op that
    returns the parameter ITSELF, e.g. an already-contiguous ``w.contiguous()``, is the same
    object and attributes correctly). A miss returns ``None`` -- the read is the documented
    alias/view residual, never misattributed.
    """

    address = get_buffer_address(source)
    if address is not None:
        return str(address)
    if type(source) is torch.nn.Parameter:
        addresses = _param_derived_addresses(trace, source)
        if len(addresses) == 1:
            return next(iter(addresses))
    return None


def _observe_state_metadata_read_direct(trace: Any, source: torch.Tensor, read_kind: str) -> None:
    """Attribute one DIRECT-receiver-only metadata read on registered state (r65).

    The autograd/structural twin of :func:`_observe_state_metadata_read`: joins the same
    escape-source and read-kind ledgers, but ONLY when the receiver is the registered object
    itself (see :func:`_state_direct_address`). An alias/view receiver records nothing (the
    documented residual); an input-leaf receiver is the input nets' domain and is skipped.
    """

    sites = trace.__dict__.get("_runnable_input_tensor_sites")
    if sites and id(source) in sites:
        return
    address = _state_direct_address(trace, source)
    if address is None:
        return
    _record_state_metadata_read(trace, {address}, read_kind)


def _record_state_metadata_read(trace: Any, addresses: "set[str]", read_kind: str) -> None:
    """Join resolved state addresses into the escape-source + read-kind ledgers (r63/r65)."""

    state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
    if state_names is None:
        state_names = set()
        _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
    state_names |= addresses
    reads = _HOST_ESCAPE_STATE_METADATA_READS.get(trace)
    if reads is None:
        reads = {}
        _HOST_ESCAPE_STATE_METADATA_READS[trace] = reads
    for address in addresses:
        reads.setdefault(address, set()).add(read_kind)


def _observe_state_metadata_fact(
    trace: Any, source: torch.Tensor, fact_name: str, fact_value: bool
) -> None:
    """Record one DECLARED-STATE fact for a DIRECT registered param/buffer receiver (r65 F-1).

    ``requires_grad`` records its bool value; ``grad_fn`` records presence. The fact ledger is
    separate from the escape machinery by design (see ``_STATE_METADATA_FACTS``): recording is
    idempotent-by-value for a stable bit, a repeated read overwrites with the latest observed
    value, and a spurious TorchLens-internal read (should one survive the source markers)
    records the true current bit, which staging reproduces -- harmless by construction.
    """

    sites = trace.__dict__.get("_runnable_input_tensor_sites")
    if sites and id(source) in sites:
        return
    address = _state_direct_address(trace, source)
    if address is None:
        return
    facts = _STATE_METADATA_FACTS.get(trace)
    if facts is None:
        facts = {}
        _STATE_METADATA_FACTS[trace] = facts
    facts.setdefault(address, {})[fact_name] = bool(fact_value)


def host_escape_state_metadata_facts(trace: Any) -> dict[str, dict[str, bool]]:
    """Return the per-state-name DECLARED-STATE metadata facts witnessed for one trace."""

    facts = _STATE_METADATA_FACTS.get(trace)
    if not facts:
        return {}
    return {name: dict(values) for name, values in facts.items()}


def _observe_state_property_read(trace: Any, source: torch.Tensor, name: str, value: Any) -> None:
    """Dispatch one getset-PROPERTY read on a state receiver through the mirror (r65).

    The property wrapper's state branch (the r64 gap: it had NONE): ``requires_grad`` /
    ``grad_fn`` route to the declared-fact ledger; every other property routes to the
    escape-gated read-kind ledger. All property names are autograd-family, so attribution is
    DIRECT-receiver-only throughout; a non-state receiver records nothing.
    """

    route = STATE_METADATA_MIRROR.get(name)
    if route is None:
        return
    route_kind, detail = route
    if route_kind == _STATE_ROUTE_DECLARED_FACT:
        if name in _INPUT_METADATA_PRESENCE_PROPERTY_NAMES:
            fact_value = value is not None
        else:
            fact_value = bool(value)
        _observe_state_metadata_fact(trace, source, detail, fact_value)
    elif route_kind == _STATE_ROUTE_READ_KIND:
        _observe_state_metadata_read_direct(trace, source, detail)


def _maybe_record_input_storage_geometry(trace: Any, source: torch.Tensor, storage: Any) -> None:
    """Witness raw storage GEOMETRY read off a model-input leaf or an alias (r29-C1 F4; r31 A).

    ``x.untyped_storage().nbytes()`` / ``.size()`` return the input's BASE storage byte count,
    which the shape+dtype input contract does NOT pin: a same-shape input that is a slice of a
    larger buffer has a larger storage than a freshly-allocated contiguous twin, so a branch on
    storage geometry would silently replay the captured arm -- a false VERIFIED. The byte count
    is the base storage's, INVARIANT across every view/alias of the same input leaf, so it is
    recorded against the leaf site for ANY input-storage-aliasing receiver -- the leaf itself,
    a ``.data`` / ``.detach()`` alias (r31 hole A: ``x.data.untyped_storage().nbytes()``), or a
    derived view (``x[k:].untyped_storage().nbytes()``) -- and re-checked against the RAW runtime
    leaf. For a normal contiguous input the byte count is exactly ``numel * element_size``
    (already shape+dtype pinned), so recording it can only ever DIVERGE on a genuinely different
    underlying buffer -- no over-trigger for ordinary inputs or read-only identity checks.
    """

    sites = trace.__dict__.get("_runnable_input_tensor_sites")
    if not sites:
        return
    site = sites.get(id(source))
    if site is None:
        kind, aliased_site, _leaf_conj_neg = _classify_input_storage_alias(trace, source)
        if kind is None:
            return
        site = aliased_site
    try:
        nbytes = int(storage.nbytes())
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return
    _record_input_metadata_read_at_site(trace, site, "storage_nbytes", nbytes)


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

_HOST_ESCAPE_STATE_METADATA_READS: "weakref.WeakKeyDictionary[Any, dict[str, set[str]]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace ledger of METADATA reads on registered state: state name -> read kinds (r63 C1).

``self.weight.is_contiguous()`` / ``.stride()`` / ``.storage_offset()`` / ``.is_conj()`` /
``.is_neg()`` on a registered param/buffer (or a storage alias of one) returns a host value
derived from the slot's PHYSICAL form -- a fact the state byte digest is structurally blind to
(a non-contiguous tensor digests to the same logical bytes as its contiguous copy) and one that
transport NORMALIZES away (the snapshot clone compacts offset and materializes conj/neg;
safetensors re-lays stride). Recording the read KIND per state name lets the runnable producer
refuse the save exactly when a read dim was non-canonical at capture
(``producer_state_metadata``), while an UNREAD non-canonical slot (a channels-last conv weight)
stays saveable and ``verified``. Kept weak-keyed off the schema; read kinds are the
``_STATE_METADATA_READ_REQUIRED_DIMS`` vocabulary in ``torchlens._runnable_state``.
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


_INPUT_METADATA_VIEW_READ: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces that read a metadata predicate on a DERIVED VIEW of a model input (r29-C1, F5).

``x.t().is_contiguous()`` reads layout metadata on a pure view of an input leaf. The view is
an orphan-pruned intermediate the sparse replay never re-derives, so the read cannot be
re-verified against the runtime input; the producer consults this set to downgrade witness
completeness (UNVERIFIABLE) rather than falsely VERIFY a possibly-wrong replayed arm. Kept in
a weak-keyed module table (not a Trace field) so the fact survives cooking without a scrub
allow-list entry. Presence-only.
"""


def input_metadata_view_read(trace: Any) -> bool:
    """Return whether a metadata predicate was read on an input-derived view (r29-C1, F5)."""

    return trace in _INPUT_METADATA_VIEW_READ


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


_HOST_ESCAPE_LABEL_LEAF_ORIGINS: "weakref.WeakKeyDictionary[Any, dict[str, tuple[frozenset[str], frozenset[str]] | None]]" = weakref.WeakKeyDictionary()
"""Per-trace fallback witness basis for escape-source labels (r37 mechanism A).

Maps each recorded escape-source RAW label to the escape source's propagated LEAF
origins, split as ``(leaf_labels, leaf_state_names)`` -- or ``None`` when the leaf set
contained ``unknown``/``rng`` (no sound fallback exists; the producer must fail
closed). The producer consults this map ONLY for a raw label that does not resolve to
a final op (an orphan-pruned host-only chain): instead of closing INCOMPLETE, it
witnesses every leaf label's op digest (PASS B) and leaf state digest (PASS A), which
is exactly the value basis the pruned chain read from. Every leaf label must itself
resolve or the escape stays INCOMPLETE -- fallback never weakens, it substitutes an
equivalent witness basis."""


def host_escape_label_leaf_origins(
    trace: Any,
) -> Mapping[str, tuple[frozenset[str], frozenset[str]] | None]:
    """Return the per-raw-label leaf-origin fallback basis for one trace."""

    return dict(_HOST_ESCAPE_LABEL_LEAF_ORIGINS.get(trace, {}))


def _record_escape_label_fallback(trace: Any, raw_label: str, source: torch.Tensor) -> None:
    """Record the leaf-origin fallback basis for one labelled escape source.

    ``None`` (fail-closed marker) wins over any positive entry on collision: if the
    same label escapes twice and either occurrence is unresolvable, the fallback is
    unusable for that label.
    """

    with _state.pause_logging(), internal_scalar_read():
        leaf = _operand_leaf_origins(trace, source)
    if _ORIGIN_UNKNOWN in leaf or _ORIGIN_RNG in leaf:
        entry: tuple[frozenset[str], frozenset[str]] | None = None
    else:
        entry = (
            frozenset(
                origin[len(_ORIGIN_LABEL_PREFIX) :]
                for origin in leaf
                if origin.startswith(_ORIGIN_LABEL_PREFIX)
            ),
            frozenset(
                origin[len(_ORIGIN_STATE_PREFIX) :]
                for origin in leaf
                if origin.startswith(_ORIGIN_STATE_PREFIX)
            ),
        )
    table = _HOST_ESCAPE_LABEL_LEAF_ORIGINS.get(trace)
    if table is None:
        table = {}
        _HOST_ESCAPE_LABEL_LEAF_ORIGINS[trace] = table
    if raw_label in table and table[raw_label] is None:
        return  # fail-closed marker sticks
    if entry is None:
        table[raw_label] = None
        return
    previous = table.get(raw_label)
    if previous is None:
        table[raw_label] = entry
    else:
        table[raw_label] = (previous[0] | entry[0], previous[1] | entry[1])


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


_ALIAS_MUTATION_CANDIDATE_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace raw labels of genuine in-place ops whose mutation TARGET carries NO resolvable
capture label (an invisible ``.data`` / foreign alias).

``y.data.add_(5.0)`` dispatches a real ``aten.add_`` that mutates ``y``'s storage, but the
receiver (``y.data``) is a fresh, UNLABELLED Python tensor object, so TorchLens cannot connect
the mutation into the tensor graph: the op's output slot feeds nothing and it is orphan-pruned
away, silently dropping the write. Each such op's raw label is recorded here at capture. Orphan
removal then intersects these candidates with the pruned set (``_record_pruned_alias_mutation``)
to record the ones actually dropped, so the runnable producer downgrades to
UNVERIFIABLE + NOT_APPLICABLE rather than falsely VERIFYING with the mutation lost. An in-place op
on a LABELLED alias (``y.detach().add_()`` / ``clone()+add_``) has a graph-connected target, is
NOT recorded here, and is replayed normally.
"""


def alias_mutation_candidate_labels(trace: Any) -> frozenset[str]:
    """Return raw labels of in-place ops that mutate an unlabelled (invisible-alias) target."""

    labels = _ALIAS_MUTATION_CANDIDATE_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def record_alias_mutation_candidate(trace: Any, label: str) -> None:
    """Record one in-place op whose mutation target carries no resolvable capture label."""

    labels = _ALIAS_MUTATION_CANDIDATE_LABELS.get(trace)
    if labels is None:
        labels = set()
        _ALIAS_MUTATION_CANDIDATE_LABELS[trace] = labels
    labels.add(label)


_PRUNED_ALIAS_MUTATION_LABELS: "weakref.WeakKeyDictionary[Any, set[str]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace raw labels of unlabelled-alias in-place ops that were ORPHAN-PRUNED.

An alias-mutation candidate (see ``_ALIAS_MUTATION_CANDIDATE_LABELS``) whose op is dropped by
orphan removal mutated storage the sparse DAG does not model. The recorded taken forward would
replay WITHOUT the mutation (wrong output), yet nothing else witnesses the drop. Kept weak-keyed
and out of the Trace field schema like the pruned-RNG table, so the runnable producer downgrades
witness completeness to keep such a model honestly UNVERIFIABLE + NOT_APPLICABLE. A candidate op
that SURVIVES pruning is graph-represented and never recorded here.
"""


def pruned_alias_mutation_source_labels(trace: Any) -> frozenset[str]:
    """Return raw labels of unlabelled-alias in-place ops that were orphan-pruned."""

    labels = _PRUNED_ALIAS_MUTATION_LABELS.get(trace)
    return frozenset(labels) if labels else frozenset()


def record_pruned_alias_mutation_source(trace: Any, label: str) -> None:
    """Record one orphan-pruned in-place op that mutated an unlabelled (invisible) alias."""

    labels = _PRUNED_ALIAS_MUTATION_LABELS.get(trace)
    if labels is None:
        labels = set()
        _PRUNED_ALIAS_MUTATION_LABELS[trace] = labels
    labels.add(label)


_HOST_ESCAPE_MUTABLE_WRITEBACK: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces where a host WRITE-BACK through a mutable zero-copy alias was detected.

``y.detach().numpy()[0] = 99`` hands the host a NumPy array that shares ``y``'s storage; the write
mutates ``y``'s bytes with NO aten dispatch and NO version bump, so the sparse replay recomputes
the pre-write value and would falsely VERIFY. The escape observer brackets each ``numpy`` /
``__array__`` call with a byte+version snapshot (see ``_observe_invisible_host_escapes``); a source
whose bytes changed while its version stayed put was host-mutated through the alias and its trace is
recorded here so the runnable producer fails closed (UNVERIFIABLE). A read-only conversion leaves
the bytes unchanged and is never recorded, so it stays honestly VERIFIED. Presence-only.
"""


def host_escape_has_mutable_writeback(trace: Any) -> bool:
    """Return whether a host write-back through a mutable zero-copy alias was detected."""

    return trace in _HOST_ESCAPE_MUTABLE_WRITEBACK


_HOST_ESCAPE_RAW_POINTER: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces where a raw ``Tensor.data_ptr()`` pointer escaped to the host (r15-H1).

``tensor.data_ptr()`` hands out the raw integer data pointer that ``ctypes`` / a foreign kernel
reads or writes through DIRECTLY, with no aten dispatch, no version bump, and -- unlike a
``.numpy()`` alias or an ``untyped_storage()`` handle -- NO Python object whose bytes the
forward-end write-back watch can re-inspect. A raw READ through the pointer bakes a stale literal
or steers control flow (unwitnessable), and a raw WRITE may land after the watch snapshot; the
pointer is fundamentally UNOBSERVABLE. So a genuine (non-internal) user ``data_ptr()`` call on a
value-bearing tensor fails closed: the tensor's subsequent value cannot be witnessed, so the run
is honestly UNVERIFIABLE rather than a false VERIFIED. This is scoped to ``data_ptr()`` ONLY --
``untyped_storage()`` / ``storage()`` value reads are already UNVERIFIABLE by the storage-bridge
watch, and read-only ``untyped_storage().nbytes()`` / ``.size()`` metadata (no value, no pointer)
never trips it. ``data_ptr()`` is a rare low-level accessor in real models, so the fail-closed
over-triggers at ~zero cost. TorchLens's own capture-internal ``data_ptr`` reads run under the
``internal_scalar_read`` marker and are excluded. Presence-only.
"""


def host_escape_has_raw_pointer(trace: Any) -> bool:
    """Return whether a raw ``Tensor.data_ptr()`` pointer escaped to the host (r15-H1)."""

    return trace in _HOST_ESCAPE_RAW_POINTER


# --- r43 CLASS 2: non-owner captured-tensor touch (ONE fail-closed rule) --------------------
#
# JMT-locked: ANY non-owner thread that TOUCHES A CAPTURED TENSOR during the armed forward
# window permanently ceilings the artifact to UNVERIFIABLE (+ NOT_APPLICABLE). This subsumes
# the whole r41 in-window/foreign 3-class distinction (which let raw ``_thread`` and
# pre-existing workers slip through). Captured membership is decided by
# :func:`_nonowner_touch_is_captured`; the storage-identity catch-all uses the true-original
# accessors captured at import (below), which bypass every torchlens wrapper.

_HOST_ESCAPE_CROSS_THREAD_CAPTURED: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces where a NON-OWNER thread touched a CAPTURED tensor during the armed window (r43).

The single JMT-locked concurrency ceiling: a captured tensor's Python-visible value/pointer/
string/metadata escape (or a positively-known captured-derived alias) observed on any thread
other than the capture owner is outside the single-owner-thread replay model, so the runnable
producer folds it into an INCOMPLETE witness downgrade -> UNVERIFIABLE + NOT_APPLICABLE.
Presence-only. A non-owner thread that never touches a captured tensor records nothing.
"""


def host_escape_has_cross_thread_captured_tensor(trace: Any) -> bool:
    """Return whether a non-owner thread touched a captured tensor during the window (r43)."""

    return trace in _HOST_ESCAPE_CROSS_THREAD_CAPTURED


_CAPTURED_STORAGE_PTRS: "weakref.WeakKeyDictionary[Any, dict[int, tuple[weakref.ref[Any], ...]]]" = weakref.WeakKeyDictionary()
"""Per-trace ptr -> LIVE producing-tensor weakrefs for the activation storage-identity catch-all (r43).

Populated by :func:`_register_dispatch_result_origins` (owner thread) so the storage-identity
catch-all recognizes a ``.data`` / view / detach alias of a captured ACTIVATION touched off-owner.
Input-leaf pointers (``_RUNNABLE_INPUT_STORAGE_SITES``) and parameter pointers
(``_param_storage_addresses``) are held ALIVE for the whole capture, so their addresses never
churn and a FLAT int set is sound for them. Transient activation storages, by contrast, are freed
mid-forward and their addresses REUSED by unrelated (possibly worker-thread) allocations -- a flat
int set would then false-positive a benign own-tensor touch (hon2_4 over-trigger). Storing a WEAKREF
to each producing tensor makes the check LIVENESS-VERIFIED: a ptr matches only when a captured
tensor is STILL ALIVE and STILL occupies that address (i.e. the touched tensor genuinely aliases
it); a freed-then-reused address has only dead weakrefs and never matches. Meta/storageless tensors
(ptr 0 -> ``None``) are never recorded.
"""

# True-original storage accessors captured at IMPORT (before any per-forward escape patch
# replaces ``torch.Tensor.untyped_storage`` / ``torch.UntypedStorage.data_ptr``). Calling these
# bypasses ALL torchlens wrappers -- no op, no dispatch, no toggle, no observer recursion -- so a
# non-owner thread can test storage identity with zero side effects (probe:
# ``calls_through_public_patches=[]``). ``TensorBase.untyped_storage`` is NOT patched (the belt
# shadows ``torch.Tensor`` only), and ``UntypedStorage.data_ptr`` IS patched per-forward, so both
# originals must be snapshotted here.
_ORIG_TENSORBASE_UNTYPED_STORAGE = torch._C.TensorBase.untyped_storage
_ORIG_UNTYPED_STORAGE_DATA_PTR = torch.UntypedStorage.data_ptr


def _raw_storage_ptr_no_observe(tensor: Any) -> int | None:
    """Return a tensor's untyped-storage data pointer via the true originals, ptr 0 -> None (r43).

    GIL-atomic, wrapper-free, side-effect-free: it never fires an escape observer, a dispatch,
    or a logging toggle, so it is safe to call from ANY thread. ``0`` (a meta / storageless
    tensor) normalizes to ``None`` so distinct storageless tensors never alias one synthetic
    pointer.
    """

    if not isinstance(tensor, torch.Tensor):
        return None
    try:
        storage = _ORIG_TENSORBASE_UNTYPED_STORAGE(tensor)
        ptr = _ORIG_UNTYPED_STORAGE_DATA_PTR(storage)
    except (RuntimeError, TypeError, NotImplementedError, AttributeError):
        return None
    return int(ptr) if ptr else None


def _nonowner_ptr_is_captured(state: "_WitnessState", ptr: int) -> bool:
    """Return whether a storage pointer belongs to a captured input / param / activation (r43)."""

    trace = state.trace
    param_addresses = getattr(trace, "_param_storage_addresses", None)
    if param_addresses and ptr in param_addresses:
        return True
    input_sites = _RUNNABLE_INPUT_STORAGE_SITES.get(trace)
    if input_sites is not None and ptr in input_sites:
        return True
    # Activation storage identity is LIVENESS-VERIFIED: the ptr matches only when a captured
    # producing tensor is still alive AND still occupies this exact address (so the touched
    # tensor genuinely aliases it). A freed-then-reused address has only dead weakrefs -> no match
    # (no over-trigger on a benign own-tensor allocation that inherited a stale address).
    captured = _CAPTURED_STORAGE_PTRS.get(trace)
    if captured is not None:
        for producer_ref in captured.get(ptr, ()):
            producer = producer_ref()
            if producer is not None and _raw_storage_ptr_no_observe(producer) == ptr:
                return True
    return False


def _nonowner_touch_is_captured(state: "_WitnessState", tensor: Any) -> bool:
    """Return whether a non-owner thread's touched tensor is a CAPTURED tensor (r43).

    Captured membership (all reads GIL-atomic; NO torch op, NO ``pause_logging``, NO observer
    recursion): a capture label OR a registered param/buffer state address OR a dispatch-origin
    ledger hit (a previously-registered owner-derived alias) OR STORAGE IDENTITY -- the tensor's
    true-original storage pointer is a captured input-leaf, parameter, or activation pointer. A
    benign own-tensor read (no label, no state, no ledger entry, unrelated storage) returns
    ``False`` and never ceilings the capture.
    """

    if not isinstance(tensor, torch.Tensor):
        return False
    trace = state.trace
    if isinstance(get_tensor_label(tensor), str):
        return True
    meta = get_tensor_meta(tensor)
    if meta is not None and getattr(meta, "address", None) is not None:
        return True
    if get_buffer_address(tensor) is not None:
        return True
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is not None and registry.get(tensor) is not None:
        return True
    ptr = _raw_storage_ptr_no_observe(tensor)
    if ptr is not None and _nonowner_ptr_is_captured(state, ptr):
        return True
    return False


def _nonowner_escape_observe(state: "_WitnessState", tensor: Any) -> None:
    """Ceiling the capture if a non-owner thread touched a captured tensor (r43).

    The ONE non-owner belt action: NO origin resolution, NO ``pause_logging``, NO precise
    witness -- a captured-tensor touch off-owner is outside the single-owner-thread replay
    model and simply marks the cross-thread ceiling. Must be called only when the caller has
    confirmed ``not owner`` and ``state.belt_armed``.
    """

    if _nonowner_touch_is_captured(state, tensor):
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)


def observe_nonowner_operands(args: tuple[Any, ...], kwargs: dict[str, Any] | None) -> None:
    """Ceiling the runnable capture when a NON-owner thread CONSUMES a captured operand (r45 hon2_1).

    The r43 cross-thread belt patches tensor METHODS, so it only recognizes a non-owner thread's
    tensor->host escape when the escaped tensor's OBJECT IDENTITY is captured / owner-registered
    (a capture label, a registered state address, a dispatch-origin ledger hit, or captured
    storage identity). A tensor DERIVED on the worker from a captured input (``(gate * 2).sum()``,
    ``gate.clone()``, ``gate + 0``, ``gate @ w``, ``torch.cat([gate], 0)`` ...) has FRESH storage
    that the OWNER-thread-only census / dispatch-origin ledger never registered, so its later value
    escape was unwitnessed -> false ``VERIFIED`` on a changed input a fresh live run would branch
    differently on (the r44 hon2_1 finding).

    Every Python-visible torch/Tensor op flows through the GLOBAL torch-function wrapper (a
    process-wide monkeypatch, unlike the thread-local aten census / dispatch mode). This observer
    runs on the wrapper's NON-owner fast path and ceilings the artifact the FIRST time a non-owner
    thread runs ANY torch op that consumes a captured tensor as an OPERAND -- op-agnostic, so it
    covers the whole worker-derivation class by construction (no derived-product registry: ceiling
    at the first consumption makes deeper-chain and escape-time provenance moot; the derived tensor
    does not even exist yet).

    Fail-CLOSED (r45 Fork C): any operand-inspection error during an armed capture ceilings the
    trace -- an inspection failure on a non-owner op cannot be read as "no captured touch"
    (validation is a tripwire). Benign-worker-safe: a non-owner thread operating only on tensors it
    created INDEPENDENTLY of the capture matches no captured-membership signal and stays
    ``VERIFIED``. The wrapper caller has already confirmed ``_state._nonowner_belt_armed`` and
    non-owner identity, so the disarmed global hot path pays only a single bool read.

    Parameters
    ----------
    args:
        The positional operands of the wrapped torch call (``*args``).
    kwargs:
        The keyword operands of the wrapped torch call (``**kwargs``), or ``None``.
    """

    state = _ACTIVE_WITNESS_STATE
    if state is None or not state.belt_armed:
        return
    if _state._active_trace is not state.trace:
        return
    if threading.get_ident() == state.owner_thread_id:
        return
    try:
        for container in (args, kwargs):
            if container is None:
                continue
            for operand in _iter_tensors_deep(container):
                if _nonowner_touch_is_captured(state, operand):
                    _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)
                    return
    except Exception:
        # An operand-inspection failure on a non-owner op during an armed capture cannot prove
        # "no captured touch": fail closed (ceiling), never silently pass.
        _HOST_ESCAPE_CROSS_THREAD_CAPTURED.add(state.trace)


def _torch_ops_call_classes() -> tuple[type, ...]:
    """Feature-detect every ``torch._ops`` class that defines its OWN ``__call__`` (r47 hon2_1).

    The r45 hon2_1 non-owner operand observer runs on the GLOBAL torch-FUNCTION wrapper, but the
    ``torch.ops.*`` (aten / higher-order / TorchBind) surface bypasses that wrapper entirely: a
    worker thread deriving from / reading a captured tensor via ``torch.ops.aten.mul.Tensor(...)``,
    ``torch.ops.aten.sum.default(...)``, ``torch.ops.aten._local_scalar_dense(...)``, ... never
    hits the wrapper and the aten dispatch census is thread-LOCAL (a ``TorchDispatchMode`` cannot
    see a non-owner thread), so its captured-operand consumption went unwitnessed -> false
    ``VERIFIED`` on a diverging changed input (the r46 hon2_1 finding).

    Every Python-visible ``torch.ops.*`` call flows through the ``__call__`` of a small set of
    ``torch._ops`` classes (``OpOverloadPacket`` / ``OpOverload`` / ``TorchBindOpOverload`` /
    ``HigherOrderOperator`` (+ the abstract ``OperatorBase``)). This scans STRUCTURALLY -- every
    ``torch._ops`` class object defining its own callable ``__call__`` -- rather than importing the
    names, which is version-robust across the declared torch floor->ceiling (2.1 -> 2.12+):
    ``TorchBindOpOverload`` only appeared ~2.4, so a by-name import would ``ImportError`` on older
    torch. A future torch that adds a call class is auto-covered by shape; a class whose ``__call__``
    is removed silently drops out (fail-closed install downgrades the capture, never a silent hole).

    Wrapping the WHOLE set including the abstract ``OperatorBase`` never double-observes: a concrete
    subclass's ``__call__`` does NOT chain to ``super().__call__``, so a single ``aten.mul.Tensor``
    call fires the observer exactly once (probed on torch 2.8).
    """

    out: list[type] = []
    for attr in dir(_torch_ops):
        obj = getattr(_torch_ops, attr, None)
        if isinstance(obj, type) and callable(obj.__dict__.get("__call__")):
            out.append(obj)
    return tuple(out)


def _make_nonowner_ops_call(original: Any) -> Any:
    """Wrap a ``torch._ops`` class ``__call__`` with the non-owner captured-operand observer (r47).

    The patched ``__call__`` short-circuits on a SINGLE bool read (``_nonowner_belt_armed``) so the
    disarmed steady state pays ~nothing, and a real eager OWNER forward hits the Python
    ``torch.ops.*.__call__`` path ZERO times (C++ dispatch; probed), so the armed owner window adds
    ~no overhead. When the belt is armed, a runnable capture is active, and the caller is a
    NON-owner thread, it routes the operands through the SAME storage-identity captured-membership
    test (:func:`observe_nonowner_operands`, fail-closed internally) BEFORE delegating to the
    original ``__call__``. The worker op is NEVER logged into the owner trace.
    """

    @functools.wraps(original)
    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            _state._nonowner_belt_armed
            and _state._active_trace is not None
            and _state._active_owner_thread_id != threading.get_ident()
        ):
            observe_nonowner_operands(args, kwargs)
        return original(self, *args, **kwargs)

    _patched.__tl_nonowner_ops_observer__ = True  # type: ignore[attr-defined]
    return _patched


def _private_c_forward_op_modules() -> tuple[Any, ...]:
    """Resolve the patchable module-typed private-C forward-op modules (r49 hon2_1).

    Structurally enumerated from the canonical forward-op module authority
    (:func:`torchlens.utils._callable_safety.private_c_forward_op_module_names` -> the
    ``torch._C._*`` entries of ``_ALLOWED_FORWARD_OP_MODULES``), resolved on the RUNNING torch
    and filtered to ``types.ModuleType`` so:

    * a torch lacking one (``_sparse`` / ``_nested`` on an older build) degrades gracefully
      (skip-if-absent), and
    * the class-typed, read-only / non-Python-patchable holders (``_VariableFunctions`` /
      ``_TensorBase``) are EXCLUDED (their setattr raises -- accepted residual).

    On torch 2.8 this yields exactly ``{_nn, _special, _fft, _linalg, _sparse, _nested}``. A
    future private-C op module added to the curated set is auto-covered.
    """

    modules: list[Any] = []
    for name in private_c_forward_op_module_names():
        obj: Any = torch
        resolved = True
        for part in name.split(".")[1:]:  # skip the leading "torch"
            obj = getattr(obj, part, None)
            if obj is None:
                resolved = False
                break
        if resolved and isinstance(obj, types.ModuleType):
            modules.append(obj)
    return tuple(modules)


def _private_c_module_callables() -> tuple[tuple[Any, str, Any], ...]:
    """Return ``(module, attr, original)`` for every module-level callable of the patchable
    private-C forward-op modules (r49 hon2_1).

    Dunder module metadata (``__loader__`` / ``__spec__`` / ...) is skipped; every remaining
    module-level callable (the ~225 ``torch._C._{nn,special,fft,linalg,sparse,nested}`` free
    functions) is a patch target so the belt is surface-complete for the whole module, not a
    known-alias subset.
    """

    out: list[tuple[Any, str, Any]] = []
    for module in _private_c_forward_op_modules():
        for attr in dir(module):
            if attr.startswith("__"):
                continue
            value = getattr(module, attr, None)
            if callable(value):
                out.append((module, attr, value))
    return tuple(out)


def _make_nonowner_private_c_callable(original: Any) -> Any:
    """Wrap a private-C module FREE function with the non-owner captured-operand observer (r49).

    Twin of :func:`_make_nonowner_ops_call` for MODULE-level free functions (no ``self``
    receiver): private-C ops (``torch._C._nn.gelu(gate)``) are a THIRD op surface -- they bypass
    BOTH the global torch-FUNCTION wrapper (no ``__torch_function__``) AND the ``torch._ops.*``
    class patch (they dispatch their inner aten op down in C++), so a non-owner worker consuming
    a captured operand through one went unwitnessed -> false ``VERIFIED`` (the r48 hon2_1
    finding). Same three-term armed/owner short-circuit (disarmed steady state pays one bool
    read; an OWNER-thread forward never reaches ``observe_nonowner_operands``) and the same
    fail-closed operand test.
    """

    @functools.wraps(original)
    def _patched(*args: Any, **kwargs: Any) -> Any:
        if (
            _state._nonowner_belt_armed
            and _state._active_trace is not None
            and _state._active_owner_thread_id != threading.get_ident()
        ):
            observe_nonowner_operands(args, kwargs)
        return original(*args, **kwargs)

    _patched.__tl_nonowner_ops_observer__ = True  # type: ignore[attr-defined]
    return _patched


_ACTIVE_WITNESS_STATE: "_WitnessState | None" = None
"""The runnable-capture witness state currently installed, or ``None`` (r43).

Published as the LAST step of ``capture_completeness_witness`` armation and cleared on exit so
the wrappers.py string interception can classify owner vs non-owner without threading the state
through the torch-function wrapper. Captures do not nest, so a single slot suffices.
"""


def string_escape_is_owner_thread(trace: Any) -> bool:
    """Return whether the current thread is the capture owner for the string hook (r43).

    Consumed by the wrappers.py ``__repr__``/``__str__``/``_str`` interception: the OWNER
    thread keeps ``print_override`` (which formats under a global ``pause_logging``); a
    NON-OWNER thread must NEVER flip that global toggle mid-forward (the hon2_4 crash), so it
    calls the original torch string function unchanged. With no active runnable witness state
    the legacy owner behavior is preserved (``True``).
    """

    state = _ACTIVE_WITNESS_STATE
    if state is None or state.trace is not trace:
        return True
    return threading.get_ident() == state.owner_thread_id


_HOST_ESCAPE_OBSERVER_FAILED: "weakref.WeakSet[Any]" = weakref.WeakSet()
"""Traces where a REQUIRED tensor->host value observer could not be installed/restored (r39).

The mode-independent method/module observers (:data:`HOST_VALUE_ESCAPE_METHODS` /
:data:`HOST_VALUE_ESCAPE_MODULE_FUNCS`) are the belt that closes the ``_disable_current_modes``
census blind spot. If a required observer cannot be installed or its exact original cannot be
restored, coverage for that forward is unknowable -- so the capture fails closed to INCOMPLETE
rather than silently reporting no escape. Optional/absent targets on a given torch version do
NOT set this (they are classified absent by the version inventory). Presence-only.
"""


def host_escape_observer_install_failed(trace: Any) -> bool:
    """Return whether a required tensor->host value observer failed to install/restore (r39)."""

    return trace in _HOST_ESCAPE_OBSERVER_FAILED


def record_host_string_escape_source(trace: Any, tensor: Any) -> None:
    """Record a tensor->host VALUE escape via string formatting (r39 hon2_1).

    TorchLens intercepts ``__repr__`` / ``__str__`` / ``_str`` on a captured tensor and formats
    it internally under ``pause_logging()`` (``print_override`` -> ``.detach().cpu().numpy()``),
    which extracts the tensor's VALUES into the returned string -- a genuine tensor->host value
    escape the user can fold back into control flow (the string NaN guard). Because that
    extraction runs under PAUSED logging, the ordinary ``.numpy()`` / ``.item()`` escape
    observers are blind to it (they gate on ``_state._logging_enabled``), so the print
    interception records the SOURCE tensor here through the SAME attribution ladder.

    NOTE (r39): the reconciled plan's E6 measurement of str/repr transitivity was taken on RAW
    torch, where ``str()`` crosses patched ``item``/``tolist``. Inside a live capture TorchLens
    intercepts the string path itself, so the fix lives at the interception, not in a
    ``__repr__``/``__str__`` patch -- the escape still lands UNVERIFIABLE, consistently with how
    every other value-extraction spelling (``.numpy()`` / ``.tolist()``) ceilings a changed run.

    Gated by the runnable-capture escape-observation flag and skipped for TorchLens's own
    marked internal reads (``internal_scalar_read``). A no-string forward records nothing.
    """

    if not getattr(trace, "intervention_ready", False):
        return
    if not isinstance(tensor, torch.Tensor):
        return
    if _internal_read_active():
        return
    # r43 hon2_4: route the string hook through the SAME owner-vs-non-owner rule as every
    # other belt observer. The OWNER thread keeps the precise attribution ladder. A NON-OWNER
    # thread applies the captured-tensor predicate (never origin resolution -- that flips the
    # global ``pause_logging`` toggle, the hon2_4 crash path): a captured-tensor stringification
    # ceilings, a benign OWN-tensor ``str()`` records nothing (no over-trigger).
    state = _ACTIVE_WITNESS_STATE
    if (
        state is not None
        and state.trace is trace
        and threading.get_ident() != state.owner_thread_id
    ):
        if state.belt_armed:
            _nonowner_escape_observe(state, tensor)
        return
    _record_escape_source_tensor(trace, tensor, invisible=True)


_DISABLE_MODE_SITE_CATEGORIES = frozenset(
    {
        # Frozen category allowlist of torch subpackages/modules that legitimately pop
        # dispatch modes (tensor formatting, dispatch plumbing, tracing/compile stacks,
        # library/registration, subclass/ref/prim lowering). The mode-independent belt
        # covers every such transit context; a site OUTSIDE these categories is the only
        # way the audit surfaces an ``unclassified`` result (-> RED coverage meta-test).
        "_tensor_str",
        "_dispatch",
        "_dynamo",
        "_export",
        "_functorch",
        "_higher_order_ops",
        "_inductor",
        "_library",
        "_subclasses",
        "_refs",
        "_prims",
        "_prims_common",
        "_meta_registrations",
        "_decomp",
        "_ops",
        "_C",
        "overrides",
        "utils",
        "fx",
        "nn",
        "ao",
        "masked",
        "nested",
        "sparse",
        "distributed",
        "autograd",
        "func",
        "serialization",
        "onnx",
        "jit",
        "_custom_ops",
        "_guards",
        "_logging",
    }
)
"""Categories of torch modules that legitimately host ``_disable_current_modes`` sites (r39)."""


def audit_disable_current_modes_sites() -> dict[str, tuple[str, ...]]:
    """Snapshot-audit torch's ``_disable_current_modes`` sites (r39 advisory-but-armed immunizer).

    The mode-independent method/module belt (:data:`HOST_VALUE_ESCAPE_METHODS` /
    :data:`HOST_VALUE_ESCAPE_MODULE_FUNCS`) closes the census blind spot regardless of WHICH
    torch region pops the dispatch modes, so this audit is NOT load-bearing -- it is an armed
    snapshot. It enumerates the ``_disable_current_modes`` sites in the installed torch and
    classifies each by its top-level containing module against
    :data:`_DISABLE_MODE_SITE_CATEGORIES`. A site whose category is unknown is ``unclassified``,
    turning the coverage meta-test RED so a human confirms the new region introduces no
    value-escape spelling the belt misses.

    Returns
    -------
    dict[str, tuple[str, ...]]
        ``{"classified": (...), "unclassified": (...)}`` module paths (sorted).
    """

    classified: set[str] = set()
    unclassified: set[str] = set()
    try:
        torch_root = Path(torch.__file__).resolve().parent
    except Exception:  # pragma: no cover - torch always has a file
        return {"classified": (), "unclassified": ()}
    for path in torch_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError):  # pragma: no cover - unreadable file
            continue
        if "_disable_current_modes(" not in text:
            continue
        relative = path.relative_to(torch_root)
        top = relative.parts[0]
        category = top[:-3] if top.endswith(".py") else top
        (classified if category in _DISABLE_MODE_SITE_CATEGORIES else unclassified).add(
            str(relative)
        )
    return {
        "classified": tuple(sorted(classified)),
        "unclassified": tuple(sorted(unclassified)),
    }


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


def _escape_storage_ptr(source: torch.Tensor) -> int | None:
    """Return ``source``'s untyped-storage data pointer, read under the internal marker.

    The internal-read marker keeps this OWN resolution read from being mistaken for a user
    ``data_ptr()`` raw-pointer escape (the storage ``data_ptr`` accessor is patched for the forward).
    """

    try:
        with internal_scalar_read():
            return source.untyped_storage().data_ptr()
    except (RuntimeError, TypeError, NotImplementedError):
        return None


def _param_derived_addresses(trace: Any, source: torch.Tensor) -> set[str]:
    """Return the state address of the registered PARAMETER whose storage ``source`` aliases.

    DIRECT alias rung only (r18): ``source`` shares a param's storage (``self.w.detach()``,
    ``self.w[0]``, ``self.w.tolist()``, ``self.w.detach().numpy()``) -- its storage pointer
    is in the forward-start param index. Resolves for frozen params too (no autograd needed).

    r37 INV-1: the former DERIVED autograd rung (r19-C -- walk ``grad_fn`` back to
    ``AccumulateGrad`` leaves and declare purity when every leaf is a registered param) is
    REMOVED as an attribution mechanism. Measured on torch 2.8 (exp1, hon2_3): a DETACHED
    or non-differentiable-dtype operand (``x.data``, ``x.detach()``, a bool mask from
    ``x > 0``, a long index, ``where``'s condition) contributes NO autograd slot at all, so
    "every leaf is a param" never proves operand totality -- the walk blessed
    input-contaminated chains as pure-param (false VERIFIED, hon2_3). Pure param-derived
    reads are recovered ONLY through positive dispatch-origin propagation
    (:func:`_resolved_dispatch_origins`); no autograd-graph structural argument may ever
    serve as an operand-totality proof again (INV-1 banned mechanism).
    """

    param_storage_addresses = getattr(trace, "_param_storage_addresses", None)
    if not param_storage_addresses:
        return set()
    direct = _escape_storage_ptr(source)
    if direct is not None and direct in param_storage_addresses:
        return {str(param_storage_addresses[direct])}
    return set()


class _TensorOriginRegistry:
    """Identity-keyed weak map: live tensor object -> propagated origin set.

    ``WeakKeyDictionary`` is unusable for tensors (its ref-equality path invokes the
    tensor's elementwise ``__eq__``), so entries key on ``id(tensor)`` with a weakref
    finalizer removing the entry when the tensor dies, and a liveness identity check
    guarding against id reuse.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[int, tuple[Any, frozenset[str], frozenset[str]]] = {}

    def get(self, tensor: torch.Tensor) -> tuple[frozenset[str], frozenset[str]] | None:
        entry = self._entries.get(id(tensor))
        if entry is None:
            return None
        ref, display, leaf = entry
        return (display, leaf) if ref() is tensor else None

    def set(self, tensor: torch.Tensor, display: frozenset[str], leaf: frozenset[str]) -> None:
        key = id(tensor)
        entries = self._entries

        def _cleanup(dead_ref: Any, key: int = key) -> None:
            entry = entries.get(key)
            if entry is not None and entry[0] is dead_ref:
                del entries[key]

        try:
            ref = weakref.ref(tensor, _cleanup)
        except TypeError:
            return  # non-weakref-able exotic subclass: stays unregistered (-> unknown)
        entries[key] = (ref, display, leaf)


_DISPATCH_TENSOR_ORIGINS: "weakref.WeakKeyDictionary[Any, _TensorOriginRegistry]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace dispatch-origin ledger: unlabelled tensor -> propagated value origins.

r37 mechanism A (INV-1). Every in-scope aten dispatch registers each tensor RESULT with
the union of its tensor OPERANDS' origins, so a fresh unlabelled tensor (a ``.data`` /
``.detach()`` alias, or any raw-dispatch product) carries a positive record of which
witnessable sources -- capture-labeled ops/inputs (``label:<raw_label>``), registered
state (``state:<address>``), seeded torch RNG (``rng``) -- its VALUE derives from.
``unknown`` taints a result whose operand could not be positively resolved; it is never
omitted. The outer key is the Trace (weak); the inner map is weak-keyed on the live
tensor objects so entries vanish with them.
"""

_ORIGIN_UNKNOWN = "unknown"
_ORIGIN_RNG = "rng"
_ORIGIN_UNINIT = "uninit"
"""r53 hon_2: distinct uninitialized-memory origin marker.

Deliberately NOT overloading ``rng``: the report vocabulary and the torch-RNG
nets stay clean, while ``_resolved_dispatch_origins`` fails closed on BOTH --
uninit-derived escapes can no longer be attributed as a "literal-only
deterministic chain" through the empty-operand-set hole.
"""
_ORIGIN_LABEL_PREFIX = "label:"
_ORIGIN_STATE_PREFIX = "state:"

_ORIGIN_FLATTEN_DEPTH_LIMIT = 4
"""Recursion bound for flattening tensor operands/results out of dispatch containers."""


def _iter_tensors_deep(value: Any, depth: int = 0) -> Iterator[torch.Tensor]:
    """Yield every tensor in a dispatch argument/result container, bounded-depth."""

    if isinstance(value, torch.Tensor):
        yield value
        return
    if depth >= _ORIGIN_FLATTEN_DEPTH_LIMIT:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors_deep(item, depth + 1)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors_deep(item, depth + 1)


def _operand_origins(trace: Any, operand: torch.Tensor) -> frozenset[str]:
    """Resolve ONE dispatch operand to its positive value origins (or ``unknown``).

    Resolution ladder (first hit wins): capture label (a tagged input/op output is
    witnessable by its own slot digest); registered-state meta/buffer address; the
    dispatch-origin ledger (a previously registered unlabelled alias/product); direct
    registered-param storage identity. Anything unresolved is ``unknown`` -- an
    explicit taint, never an omission (INV-1).
    """

    label = get_tensor_label(operand)
    if isinstance(label, str):
        return frozenset({f"{_ORIGIN_LABEL_PREFIX}{label}"})
    meta = get_tensor_meta(operand)
    address = getattr(meta, "address", None) if meta is not None else None
    if address is not None:
        return frozenset({f"{_ORIGIN_STATE_PREFIX}{address}"})
    buffer_address = get_buffer_address(operand)
    if buffer_address is not None:
        return frozenset({f"{_ORIGIN_STATE_PREFIX}{buffer_address}"})
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is not None:
        entry = registry.get(operand)
        if entry is not None:
            return entry[0]
    param_storage_addresses = getattr(trace, "_param_storage_addresses", None)
    if param_storage_addresses:
        ptr = _escape_storage_ptr(operand)
        if ptr is not None and ptr in param_storage_addresses:
            return frozenset({f"{_ORIGIN_STATE_PREFIX}{param_storage_addresses[ptr]}"})
    return frozenset({_ORIGIN_UNKNOWN})


def _operand_leaf_origins(trace: Any, operand: torch.Tensor) -> frozenset[str]:
    """Resolve ONE operand to its TERMINAL leaf origins (state / input-or-boundary labels).

    Unlike :func:`_operand_origins` (where an interior op's own label wins -- the best,
    finest witness), leaf resolution propagates THROUGH interior labeled results down to
    a basis that survives orphan-pruning: registered state addresses and the labels of
    tensors that were never produced by an in-scope dispatch (model inputs and other
    boundary tensors). The producer consumes this basis as the fail-closed FALLBACK
    witness set for an escape whose direct source label was orphan-pruned (r37
    mechanism A: "falls back to propagated leaf origins for pruned labels").
    """

    meta = get_tensor_meta(operand)
    address = getattr(meta, "address", None) if meta is not None else None
    if address is not None:
        return frozenset({f"{_ORIGIN_STATE_PREFIX}{address}"})
    buffer_address = get_buffer_address(operand)
    if buffer_address is not None:
        return frozenset({f"{_ORIGIN_STATE_PREFIX}{buffer_address}"})
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is not None:
        entry = registry.get(operand)
        if entry is not None:
            return entry[1]
    param_storage_addresses = getattr(trace, "_param_storage_addresses", None)
    if param_storage_addresses:
        ptr = _escape_storage_ptr(operand)
        if ptr is not None and ptr in param_storage_addresses:
            return frozenset({f"{_ORIGIN_STATE_PREFIX}{param_storage_addresses[ptr]}"})
    label = get_tensor_label(operand)
    if isinstance(label, str):
        # Not a dispatch product: an input / boundary tensor whose label is terminal.
        return frozenset({f"{_ORIGIN_LABEL_PREFIX}{label}"})
    return frozenset({_ORIGIN_UNKNOWN})


def _operator_is_seeded_rng(func: Any) -> bool:
    """Return whether a dispatcher overload is torch-tagged nondeterministic-seeded."""

    try:
        return torch.Tag.nondeterministic_seeded in getattr(func, "tags", ())
    except (TypeError, RuntimeError):
        return True  # unreadable tags: treat as RNG (fail closed)


def _operator_uninit_family_tail(func: Any) -> str | None:
    """Return a dispatcher overload's base op name IF it is in the uninit family.

    r53 hon_2: the shared closed table lives in ``utils/rng.py`` (one predicate,
    three layers). At this layer the spelling is the overload-independent aten
    base name (``aten.empty_like`` -> ``empty_like``).
    """

    from ...utils.rng import _UNINIT_ALLOC_FACTORY_TAILS, _UNINIT_ALLOC_RESIZE_TAILS

    base = _operator_base_name(func)
    if not base.startswith("aten."):
        return None
    tail = base[len("aten.") :]
    if tail in _UNINIT_ALLOC_FACTORY_TAILS or tail in _UNINIT_ALLOC_RESIZE_TAILS:
        return tail
    return None


def _python_tensor_method_uninit_family_tail(
    namespace: str | None, qualname: str | None, args: tuple[Any, ...] = ()
) -> str | None:
    """Return a PYTHON-``torch.Tensor``-method spelling's uninit family tail (r55 hon_1).

    The dispatch-level origin ledger above keys the family off aten base names,
    which is complete for every spelling that REDISPATCHES an aten family op --
    including the legacy ``Tensor.new(sizes)``, whose size form redispatches
    ``aten.empty.memory_format`` (probed), so LIVE capture tainting already
    covers it transitively. This function is the DECLARED recognition of the
    family over the Python-``torch.Tensor``-method surface -- the spellings the
    load-side qualname classifier and the family-drift meta-test see (``new``
    has NO aten spelling: ``hasattr(torch.ops.aten, "new")`` is ``False``).
    It consults the single ``utils/rng.py`` table block (never re-derives):

    - a plain factory/resize tail (``empty_like``, ``resize_``) matches by
      qualname alone, exactly like the load-side classifier;
    - a SIZE-GATED tail (``new``) additionally requires the size-argument form;
      an UNDECIDABLE form (``uninit_new_call_is_size_form`` returning ``None``)
      fails closed to recognized-as-family, mirroring the grow-gate posture.

    The Python-Tensor-method drift meta-test
    (``tests/test_tlspec_runnable_r53_uninit_alloc.py``) enumerates
    ``torch.Tensor`` allocation-pattern methods against this recognition so a
    FUTURE python-only uninit factory with no aten spelling is a FAILING test,
    never a silent gap slipping both the aten drift test and this surface.
    """

    from ...utils.rng import (
        qualname_is_uninit_growth_resize,
        qualname_is_uninit_size_gated_alloc,
        qualname_is_uninitialized_alloc,
        uninit_new_call_is_size_form,
    )

    if not qualname:
        return None
    tail = qualname.rsplit(".", 1)[-1]
    if qualname_is_uninitialized_alloc(namespace, qualname) or qualname_is_uninit_growth_resize(
        namespace, qualname
    ):
        return tail
    if qualname_is_uninit_size_gated_alloc(namespace, qualname):
        if uninit_new_call_is_size_form(args) is False:
            return None  # data-form ``new([values])``/``new(tensor)``: deterministic copy
        return tail  # size form, or undecidable -> fail closed to tainted
    return None


def _operator_is_growth_resize(func: Any) -> bool:
    """Return whether a dispatcher overload is a resize spelling (grow-gated family)."""

    from ...utils.rng import _UNINIT_ALLOC_RESIZE_TAILS

    base = _operator_base_name(func)
    return base.startswith("aten.") and base[len("aten.") :] in _UNINIT_ALLOC_RESIZE_TAILS


def _operator_total_writer_destination(
    func: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> Any | None:
    """Return the tensor whose bytes this dispatch TOTALLY overwrites, or ``None``.

    Total writers per the shared r53 hon_2 sanitizer table: the ``out=`` kwarg
    destination (torch's ``out=`` convention IS a full overwrite; only an exact
    single-tensor ``out`` sanitizes) and the in-place
    ``copy_``/``zero_``/``fill_``/RNG-fill receivers. Partial or unprovable
    in-place writers return ``None`` (taint propagates, fail closed).
    """

    from ...utils.rng import _UNINIT_RNG_FILL_TAILS, _UNINIT_TOTAL_WRITER_TAILS

    if kwargs:
        out = kwargs.get("out")
        if isinstance(out, torch.Tensor):
            return out
    base = _operator_base_name(func)
    if base.startswith("aten."):
        tail = base[len("aten.") :]
        if tail in _UNINIT_TOTAL_WRITER_TAILS or tail in _UNINIT_RNG_FILL_TAILS:
            if args and isinstance(args[0], torch.Tensor):
                return args[0]
    return None


def _live_deterministic_fill_governs() -> bool:
    """Return whether the LIVE capture context proves deterministic uninit fill."""

    from ...utils._torch_compat import (
        HAS_DETERMINISTIC_ALGORITHMS_QUERY,
        read_fill_uninitialized_memory,
    )
    from ...utils.rng import deterministic_fill_governs

    deterministic = (
        bool(torch.are_deterministic_algorithms_enabled())
        if HAS_DETERMINISTIC_ALGORITHMS_QUERY
        else None
    )
    return deterministic_fill_governs(deterministic, read_fill_uninitialized_memory())


def _register_dispatch_result_origins(
    state: "_WitnessState",
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    result: Any,
    pre_dispatch_receiver_numel: int | None = None,
) -> None:
    """Propagate the union of operand origins onto every tensor result (mechanism A).

    A seeded-RNG operator additionally taints its results with the ``rng`` origin so a
    pruned host read of raw RNG output can never be attributed as a deterministic
    function of its operands. An in-place operator's mutated unlabelled receiver is
    covered because the receiver IS a result-aliasing operand: re-registering results
    updates its entry with the union (its VALUE now depends on all operands).

    r53 hon_2: an uninitialized-memory family op (``empty`` factories; a GROWING
    ``resize_``, decided against ``pre_dispatch_receiver_numel`` read at the
    interpose BEFORE the receiver was resized, failing closed to tainted when
    unavailable) additionally taints its results with the distinct ``uninit``
    origin -- closing the empty-operand-set hole where allocator garbage
    registered an EMPTY origin set and was later attributed as a "literal-only
    deterministic chain". A total writer (``out=`` destination,
    ``copy_``/``zero_``/``fill_``/RNG fill receiver) EXCLUDES the destination
    operand's prior origins from the union: the post-write value derives from
    the value-source operands only (exact value semantics -- strictly more
    precise, never less safe).
    """

    trace = state.trace
    display_union: set[str] = set()
    leaf_union: set[str] = set()
    # ``pause_logging`` + the internal marker: origin resolution reads storage
    # pointers/labels through torch-function-wrapped accessors; unpaused reads
    # would be logged as spurious ops mid-forward (shifting raw counters and
    # staling every label recorded after them) and would trip the escape census.
    # The r53 hon_2 metadata reads (``numel`` on the result, the ``out=`` kwarg
    # probe) live INSIDE the same paused scope for exactly that reason.
    with _state.pause_logging(), internal_scalar_read():
        total_write_destination = _operator_total_writer_destination(func, args, kwargs)
        for operand in _iter_tensors_deep(args):
            if operand is total_write_destination:
                continue
            display_union |= _operand_origins(trace, operand)
            leaf_union |= _operand_leaf_origins(trace, operand)
        if kwargs:
            for operand in _iter_tensors_deep(kwargs):
                if operand is total_write_destination:
                    continue
                display_union |= _operand_origins(trace, operand)
                leaf_union |= _operand_leaf_origins(trace, operand)
        exposes_uninit = False
        if _operator_uninit_family_tail(func) is not None:
            exposes_uninit = True
            result_numel = result.numel() if isinstance(result, torch.Tensor) else None
            if _operator_is_growth_resize(func):
                # Shrink/same-size preserves the element prefix (probed clean);
                # an unreadable pre-call size fails closed to tainted.
                exposes_uninit = (
                    pre_dispatch_receiver_numel is None
                    or result_numel is None
                    or result_numel > pre_dispatch_receiver_numel
                )
            elif result_numel == 0:
                exposes_uninit = False  # zero elements: no bytes to expose
            if exposes_uninit and _live_deterministic_fill_governs():
                exposes_uninit = False  # torch fills deterministically (probed NaN)
    if _operator_is_seeded_rng(func):
        display_union.add(_ORIGIN_RNG)
        leaf_union.add(_ORIGIN_RNG)
    if exposes_uninit:
        display_union.add(_ORIGIN_UNINIT)
        leaf_union.add(_ORIGIN_UNINIT)
    if _operator_base_name(func) == "aten.as_strided" and not _as_strided_result_contained(
        args, result
    ):
        # An out-of-span restride can address storage bytes outside every operand's
        # witnessed span: its value is NOT a function of the operands (fail closed).
        display_union.add(_ORIGIN_UNKNOWN)
        leaf_union.add(_ORIGIN_UNKNOWN)
    display = frozenset(display_union)
    leaf = frozenset(leaf_union)
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is None:
        registry = _TensorOriginRegistry()
        _DISPATCH_TENSOR_ORIGINS[trace] = registry
    # r43 CLASS 2: index every owner-produced activation's storage pointer so the non-owner
    # storage-identity catch-all recognizes a ``.data`` / view / detach alias of a captured
    # activation touched off-owner. The true-original accessor is wrapper-free.
    captured_ptrs = _CAPTURED_STORAGE_PTRS.get(trace)
    if captured_ptrs is None:
        captured_ptrs = {}
        _CAPTURED_STORAGE_PTRS[trace] = captured_ptrs
    for produced in _iter_tensors_deep(result):
        # Labeled results register too: the display ladder still prefers their own
        # label (finest witness), but the LEAF set must flow through them so a later
        # orphan-pruned chain can fall back to a surviving witness basis.
        registry.set(produced, display, leaf)
        produced_ptr = _raw_storage_ptr_no_observe(produced)
        if produced_ptr is None:
            continue
        try:
            produced_ref = weakref.ref(produced)
        except TypeError:
            continue  # non-weakref-able exotic subclass: storage identity not indexed
        # Copy-on-write, prune-dead on append (bounds a reused address to its LIVE aliases,
        # keeping the per-ptr tuple tiny and the worker-side read race-free against an
        # atomic dict-value reassignment).
        live = tuple(ref for ref in captured_ptrs.get(produced_ptr, ()) if ref() is not None)
        captured_ptrs[produced_ptr] = (*live, produced_ref)


def _resolved_dispatch_origins(
    trace: Any, source: torch.Tensor
) -> tuple[set[str], set[str]] | None:
    """Resolve an unlabelled escape source to positive (labels, state names), or ``None``.

    Returns ``None`` -- the caller MUST fail closed -- when the source's propagated
    origin set contains ``unknown`` (an operand the census could not attribute),
    ``rng`` (raw seeded-RNG output; the torch-RNG nets own that class, and value
    attribution through it would launder nondeterminism), or ``uninit`` (r53
    hon_2: uninitialized allocator bytes are not a function of the recorded
    computation, so attributing through them would launder nondeterminism as a
    deterministic chain). An empty origin pair is a positive result: the value
    derives from a literal-only deterministic chain that replays identically, so
    it needs no witness.
    """

    with _state.pause_logging(), internal_scalar_read():
        origins = _operand_origins(trace, source)
    if _ORIGIN_UNKNOWN in origins or _ORIGIN_RNG in origins or _ORIGIN_UNINIT in origins:
        return None
    labels = {
        origin[len(_ORIGIN_LABEL_PREFIX) :]
        for origin in origins
        if origin.startswith(_ORIGIN_LABEL_PREFIX)
    }
    states = {
        origin[len(_ORIGIN_STATE_PREFIX) :]
        for origin in origins
        if origin.startswith(_ORIGIN_STATE_PREFIX)
    }
    return labels, states


def _record_escape_source_tensor(
    trace: Any,
    source: torch.Tensor,
    *,
    invisible: bool,
    fail_closed: bool = True,
    resolve_origins: bool = True,
) -> None:
    """Record ONE tensor->host escape source, visible or census-invisible, uniformly.

    ``invisible`` is ``True`` for a ``.tolist()`` / ``.numpy()`` / ``__array__``
    conversion (observed by the scoped method patch) and ``False`` for an
    ``aten._local_scalar_dense`` scalar escape (observed by the aten census). Both
    mechanisms feed the SAME per-trace side tables so the runnable descriptor witnesses
    every source class -- input, internal op, bound/unbound param, bound/unbound buffer --
    by its capture-time digest through one uniform pass. Side tables are mutated via
    GIL-atomic ``set.add``/``dict`` writes (CPython), so cross-thread recording (r41)
    needs no extra locking.

    ``fail_closed`` (r41 hon2_1/F): ``False`` -- the FOREIGN (pre-existing) thread
    posture -- skips exactly the two unattributable rungs (the fail-closed bool/opaque
    records), so an unattributable foreign-thread read never ceilings the capture while
    every POSITIVE rung (label, registered-state alias, dispatch origin) still records.

    ``resolve_origins`` (r41): ``False`` skips the dispatch-origin resolution rung and
    the leaf-origin fallback recording, both of which take ``pause_logging`` (a GLOBAL
    toggle a non-owner thread must never flip mid-forward). An absent fallback entry is
    consumed by the producer exactly like the fail-closed ``None`` marker (an
    orphan-pruned label without a basis stays INCOMPLETE), so skipping never weakens.

    An escape dispatched from TorchLens's own op-logging internals (a metadata read of a
    freshly-produced op output) is NOT a user escape and is skipped, so the fail-closed
    INCOMPLETE gates never fire on TorchLens's own reads.
    """

    if _escape_source_is_torchlens_internal():
        return
    is_bool = source.dtype is torch.bool
    label = get_tensor_label(source)
    if not isinstance(label, str):
        # An UNLABELLED escape source (a ``.data`` alias, a raw-dispatch product):
        # r37 INV-1 single-exit attribution ladder. Every rung is a POSITIVE
        # attribution to a witnessable source; the fallthrough IS the fail-closed
        # record. Banned forever as discharge mechanisms: scalar value equality
        # (hon2_2), ``.item()`` re-extraction on unknown-arity operands (hon2_1),
        # and any autograd-graph structural purity argument (hon2_3 / exp1).
        if is_bool:
            # A pruned, unlabelled bool predicate is covered by NO net -> fail closed
            # (skipped for a foreign thread: no attribution, no ceiling).
            if fail_closed:
                _HOST_ESCAPE_UNATTRIBUTABLE_BOOL.add(trace)
            return
        # Rung 1 (r18): direct registered-param storage alias -- witnessed by the
        # param state slot (``self.w.tolist()`` directly on a param carries no label).
        param_addresses = _param_derived_addresses(trace, source)
        if param_addresses:
            state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
            if state_names is None:
                state_names = set()
                _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
            state_names |= param_addresses
            return
        # Rung 2 (r37 mechanism A): positive dispatch-origin propagation. The census
        # registered this tensor's value origins at its producing dispatch; resolve
        # them to witnessable raw labels (tensor-op/input sources -> PASS B digest)
        # and state names (param/buffer sources -> PASS A digest). Multi-element and
        # scalar sources resolve identically -- no arity assumption anywhere.
        # Owner-thread only (resolution flips the global logging toggle).
        resolved = _resolved_dispatch_origins(trace, source) if resolve_origins else None
        if resolved is not None:
            origin_labels, origin_states = resolved
            if origin_labels:
                sources = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
                if sources is None:
                    sources = set()
                    _HOST_ESCAPE_SOURCE_LABELS[trace] = sources
                sources |= origin_labels
                # An origin label can itself be an interior (later orphan-pruned)
                # op label; give each the same leaf fallback basis.
                for origin_label in origin_labels:
                    _record_escape_label_fallback(trace, origin_label, source)
            if origin_states:
                state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
                if state_names is None:
                    state_names = set()
                    _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
                state_names |= origin_states
            # Empty label+state origins: a literal-only deterministic chain whose
            # baked value replays identically -- positively attributed, witness-free.
            return
        # Fallthrough: no positive attribution -> fail closed (INCOMPLETE). This is
        # the ONLY other exit; there is no third state (INV-1). A foreign thread
        # (``fail_closed=False``) skips the record: no attribution, no ceiling.
        if fail_closed:
            _HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE.add(trace)
        return
    sources = _HOST_ESCAPE_SOURCE_LABELS.get(trace)
    if sources is None:
        sources = set()
        _HOST_ESCAPE_SOURCE_LABELS[trace] = sources
    sources.add(label)
    # r37 mechanism A: record the leaf-origin fallback basis NOW (the live tensor and
    # its propagated origins exist only during capture). Consumed by the producer only
    # if this label turns out orphan-pruned. Skipped on non-owner threads (r41,
    # ``resolve_origins=False``): an absent entry reads exactly like the fail-closed
    # marker if the label is later orphan-pruned -- never weaker.
    if resolve_origins:
        _record_escape_label_fallback(trace, label, source)
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
    # r18 + r19-C: a PARAMETER host escape resolves by state slot exactly like a buffer. A buffer
    # keeps a graph SOURCE node, so its ``.detach()`` host read survives orphan-pruning and witnesses
    # by its kept op; a parameter carries NO source node, so a param-rooted read op is orphan-pruned
    # and its raw label resolves to no final op -> the escape would fail closed (INCOMPLETE_SCALAR_
    # ESCAPE -> a spurious UNVERIFIABLE) even for a purely READ-ONLY stat log. Resolve the param
    # state slot(s) so the read-only escape is witnessed by the param's capture-time digest (value-
    # correct -- an unchanged param re-digests identically -> VERIFIED; a changed param -> UNVERIFIABLE),
    # the same honest read/write distinction the buffer path draws. ``_param_derived_addresses`` covers
    # both a DIRECT param alias (r18: ``self.w.detach()``, ``self.w[0]``) and a DERIVED pruned read
    # rooted purely in params (r19-C: ``self.w.sum()``, ``float(self.w.max())``). A genuine host WRITE
    # is caught independently by the parameter whole-storage byte tripwire
    # (``buffer_writes._reconcile_params`` -> ``_HOST_ESCAPE_MUTABLE_WRITEBACK``), so read resolution
    # never blesses a mutated param.
    state_addresses: set[str] = set()
    if address is not None:
        state_addresses.add(str(address))
    else:
        state_addresses |= _param_derived_addresses(trace, source)
    if state_addresses:
        state_sources = _HOST_ESCAPE_STATE_SOURCE_LABELS.get(trace)
        if state_sources is None:
            state_sources = set()
            _HOST_ESCAPE_STATE_SOURCE_LABELS[trace] = state_sources
        state_sources.add(label)
        state_names = _HOST_ESCAPE_STATE_SOURCE_NAMES.get(trace)
        if state_names is None:
            state_names = set()
            _HOST_ESCAPE_STATE_SOURCE_NAMES[trace] = state_names
        state_names |= state_addresses


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


# Pure-view / aliasing accessor operators emitted by the ``.data`` property getter on a
# registered buffer. Accessing ``self.b.data`` (the standard buffer-write idiom
# ``self.b.data.copy_(x)``) dispatches a raw ``aten.detach.default`` with NO python wrapper
# owner -- ``.data`` is a C-level tensor property, not a wrapped torch function, so TorchLens
# never records it as a graph op. The subsequent write (``copy_``) IS captured as a normal
# op; only this accessor detach is legitimately uncaptured. The set is deliberately narrow to
# PURE non-mutating views: crediting it in the completeness backstop cannot hide a
# value-affecting drop, and a dropped value-producing op (``aten.add`` etc.) on a buffer is
# NOT in this set and still trips the tripwire.
_BUFFER_STATE_VIEW_OPERATORS = frozenset({"aten.detach", "aten.alias"})


def _is_buffer_state_view_dispatch(
    func: Any,
    owner: "ExpectedOriginalToken | None",
    mutates: bool,
    args: tuple[Any, ...],
) -> bool:
    """Return whether an aten dispatch is a ``.data``-accessor view on a registered buffer.

    This flags the intrinsic, legitimately-uncaptured ``aten.detach`` a registered buffer's
    ``.data`` property emits during a buffer WRITE (``self.b.data.copy_(x)``). The predicate is
    intentionally strict on every axis so it can never mask a genuine untraced dispatch:

    * ``owner is None`` -- the dispatch has NO python-wrapper owner (a wrapped ``.detach()``
      call would be an accounted owner, not a gap; only the property-accessor path is unowned).
    * ``not mutates`` -- the operator writes to no argument (ground-truth schema flag). A
      value-affecting in-place drop can never be credited here.
    * ``aten.detach`` / ``aten.alias`` only -- pure aliasing views. A dropped value-producing
      op (``aten.add``/``aten.mul``/...) on the buffer is NOT in this set and stays unaccounted.
    * ``args[0]`` is a REGISTERED BUFFER (``get_buffer_address`` resolves a dotted address).

    Parameters
    ----------
    func:
        Dispatcher operator overload.
    owner:
        The live wrapper owner of the dispatch, or ``None`` when unowned.
    mutates:
        Whether the operator writes to any argument (schema ground truth).
    args:
        Positional dispatcher arguments; ``args[0]`` is the view source.

    Returns
    -------
    bool
        ``True`` only for a ``.data``-accessor view dispatch on a registered buffer.
    """

    if owner is not None or mutates:
        return False
    if _operator_base_name(func) not in _BUFFER_STATE_VIEW_OPERATORS:
        return False
    if not args:
        return False
    source = args[0]
    return isinstance(source, torch.Tensor) and get_buffer_address(source) is not None


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
        event: _DispatchEvent | None = None
        pre_dispatch_receiver_numel: int | None = None
        try:
            in_scope = (
                threading.get_ident() == self.state.owner_thread_id
                and _state._logging_enabled
                and _state._active_trace is self.state.trace
                and _is_aten_operator(func)
            )
            if in_scope and (self.state.census or self.state.ledger):
                owner = _active_token()
                # The frame-walking callsite/replacement/state-view facts are census
                # diagnostics; ledger-only events defer the replacement-hook probe to
                # OUTCOME time (only raised / host-returning events need it).
                callsite = (
                    _dispatch_callsite()
                    if self.state.census and (owner is None or owner.func_call_id is None)
                    else None
                )
                in_replacement_hook = _in_replacement_hook_frame() if self.state.census else False
                mutates = _is_mutating_operator(func)
                state_view_accessor = (
                    _is_buffer_state_view_dispatch(func, owner, mutates, args)
                    if self.state.census
                    else False
                )
                event = _DispatchEvent(
                    _operator_name(func),
                    owner,
                    callsite,
                    in_replacement_hook,
                    mutates,
                    state_view_accessor,
                )
                self.state.events.append(event)
            # Per-consumption host write-back sample (r16-H1 TOCTOU): if a mutable zero-copy alias
            # is live and THIS traced op consumes a watched source whose bytes were transiently
            # written, catch it now -- BEFORE redispatch reads the mutated input -- rather than only
            # at forward end where a byte-exact restore would have already hidden it.
            if in_scope:
                _sample_writeback_at_consumption(self.state, args, kwargs)
            # r53 hon_2: a resize-family receiver must be sized BEFORE dispatch --
            # afterwards the receiver has already been resized, so the grow fact
            # (stale-byte exposure) would be unrecoverable. ``numel`` is a
            # torch-function-wrapped accessor, so the read runs PAUSED (an
            # unpaused read would log a spurious op mid-forward and stale the
            # escape census); an unreadable size fails closed to tainted in the
            # origin registration below.
            if in_scope and self.state.record_escapes and _operator_is_growth_resize(func):
                receiver = args[0] if args else None
                if isinstance(receiver, torch.Tensor):
                    try:
                        with _state.pause_logging(), internal_scalar_read():
                            pre_dispatch_receiver_numel = int(receiver.numel())
                    except (RuntimeError, TypeError):
                        pre_dispatch_receiver_numel = None
        finally:
            self.state.callback_ns += time.perf_counter_ns() - started
        try:
            result = func(*args, **(kwargs or {}))
        except BaseException as exc:
            # r35 I2 lifecycle ledger: an op that RAISED left no captured artifact,
            # so a branch taken *because* it raised has no witness anchor. Record
            # only safe facts (type module+qualname) and re-raise unchanged.
            if event is not None:
                event.outcome = "raised"
                event.exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
                if not event.in_replacement_hook:
                    event.in_replacement_hook = _in_replacement_hook_frame()
            raise
        if event is not None:
            if _dispatch_result_holds_tensor(result):
                event.outcome = "returned_tensor"
                if not event.mutates and _operator_base_name(func) == "aten.as_strided":
                    # Owner-independent: an ``__dlpack__``-wrapper-owned interval is
                    # not a modeled call, so the audited row must still apply.
                    event.contained_view = _as_strided_result_contained(args, result)
            else:
                event.outcome = "returned_host_or_none"
                if not event.in_replacement_hook:
                    event.in_replacement_hook = _in_replacement_hook_frame()
        # Escape recording needs the OUTPUT: a tensor->host escape is any aten dispatch
        # returning a NON-TENSOR host value from a tensor operand (equal/allclose/
        # is_nonzero/_local_scalar_dense). Recorded after redispatch so the result is
        # observable; still gated to the owner thread / active trace / logging window.
        # Origin propagation (r37 mechanism A) registers every tensor RESULT with the
        # union of its operands' origins FIRST, so an escape observed later on an
        # unlabelled product of this dispatch resolves positively instead of opaquely.
        if in_scope and self.state.record_escapes:
            escape_started = time.perf_counter_ns()
            try:
                _register_dispatch_result_origins(
                    self.state,
                    func,
                    args,
                    kwargs,
                    result,
                    pre_dispatch_receiver_numel=pre_dispatch_receiver_numel,
                )
                _record_host_escape_source(self.state.trace, func, args, result)
            finally:
                self.state.callback_ns += time.perf_counter_ns() - escape_started
        return result


def _dispatch_result_holds_tensor(result: Any) -> bool:
    """Return whether an aten dispatch result contains any tensor (one level deep)."""

    if isinstance(result, torch.Tensor):
        return True
    if isinstance(result, (list, tuple)):
        return any(isinstance(item, torch.Tensor) for item in result)
    return False


_RUNNABLE_LEDGER_FACTS: "weakref.WeakKeyDictionary[Any, list[dict[str, Any]]]" = (
    weakref.WeakKeyDictionary()
)
"""Per-trace undischarged event-lifecycle facts (r35 I2, hon2_1).

Each fact is a safe, value-free record naming the site of an event the census
could not discharge: a caught in-forward raise (``caught_exception_control``),
an unmodeled successful host/``None`` return (``unmodeled_host_return``), or a
mutation-capable unknown (``opaque_side_effect``). The runnable producer maps a
non-empty fact list to an INCOMPLETE witness-completeness downgrade, so every
run of that artifact ceilings at ``unverifiable`` + ``not_applicable``.
"""


def runnable_ledger_facts(trace: Any) -> tuple[Mapping[str, Any], ...]:
    """Return the recorded undischarged lifecycle facts for one trace."""

    return tuple(_RUNNABLE_LEDGER_FACTS.get(trace, ()))


_PURE_VIEW_DISPATCH_OPERATORS = frozenset({"aten.detach", "aten.alias"})
"""Non-mutating pure-aliasing operators discharged as an audited ``returned_tensor`` row.

r37 INV-1 narrow audited row (never a blanket outcome exemption): an unowned
``aten.detach`` / ``aten.alias`` is the C-level ``.data`` property accessor (on ANY
tensor, not only registered buffers). The view itself moves no value to the host and
computes no new bytes; every hazard THROUGH it is owned by another disposition -- a
VALUE escape of the alias is attributed by the escape ladder (origin propagation
resolves the alias to its base), a MUTATION through the alias is a separate mutating
dispatch event, and a metadata read is the r31 input-metadata witness's domain. A
value-PRODUCING unowned op (``aten.add``/``aten.mul``/...) is NOT in this set and
records an incomplete fact. ``aten.as_strided`` (emitted unowned by C-level DLPack /
array-interop consumers) is discharged by the SAME argument but ONLY when its result
span is byte-contained in its operand's span (:func:`_as_strided_event_contained`);
an out-of-span restride can address storage bytes no witness covers and stays an
incomplete fact.
"""


def _tensor_abs_byte_span(value: torch.Tensor) -> tuple[int, int] | None:
    """Absolute (start, end] byte span a strided tensor's elements touch, or ``None``.

    Local minimal span math (min/max stride contributions on absolute addresses);
    the full shared relation engine lives in ``utils.tensor_utils`` -- this helper
    only answers CONTAINMENT for the as_strided audited row and fails ``None``-closed.
    """

    try:
        with internal_scalar_read():
            base = int(value.untyped_storage().data_ptr())
        esize = int(value.element_size())
        if base == 0 and value.numel() > 0:
            return None
        origin = base + int(value.storage_offset()) * esize
        if value.numel() == 0:
            return (origin, origin)
        low = 0
        high = 0
        for size, stride in zip(value.shape, value.stride()):
            contribution = (int(size) - 1) * int(stride)
            if contribution < 0:
                low += contribution
            else:
                high += contribution
        return (origin + low * esize, origin + high * esize + esize)
    except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
        return None


def _as_strided_result_contained(args: tuple[Any, ...], result: Any) -> bool:
    """Return whether an ``aten.as_strided`` result's byte span sits inside its operand's."""

    if not args or not isinstance(args[0], torch.Tensor) or not isinstance(result, torch.Tensor):
        return False
    with _state.pause_logging():
        operand_span = _tensor_abs_byte_span(args[0])
        result_span = _tensor_abs_byte_span(result)
    if operand_span is None or result_span is None:
        return False
    return operand_span[0] <= result_span[0] and result_span[1] <= operand_span[1]


def _finalize_runnable_ledger(state: _WitnessState) -> None:
    """Discharge every observed dispatch event or record an incomplete fact (r35 I2, r37 INV-1).

    EXHAUSTIVE over the outcome vocabulary: every event terminates in exactly one
    explicit disposition -- accounted modeled call, exact audited opaque boundary,
    replacement-hook construction, escape-net witness, ``.data``-accessor state view,
    audited pure-view row, or an explicit incomplete fact. Discharge rules
    (owner-accounted; no exception-type or framework-file exemptions): a subevent
    whose enclosing wrapper owner became an accounted modeled call is discharged
    (replaying the owner replays its internals); a host-return witnessed by the exact
    escape net (``HOST_ESCAPE_OPERATORS``) is discharged (post-hon2_1 the net is
    total: every operand records a positive attribution or a fail-closed flag). A
    ``returned_tensor`` event -- the corr2-1 class -- is NEVER implicitly discharged:
    an unowned mutating dispatch records ``opaque_side_effect`` and an unowned
    non-mutating value-producing dispatch records ``unmodeled_tensor_return`` (its
    product can bake into a later traced call as an unwitnessed constant). An
    unhandled outcome value is a hard internal error, never a silent pass. This
    finalize runs only when the forward COMPLETED -- an undischarged raise means the
    exception was caught before forward completion: exception-driven control flow the
    sparse replay cannot witness.
    """

    facts: list[dict[str, Any]] = []
    for event in state.events:
        owner = event.owner
        owner_accounted = owner is not None and owner.capture_accounted is True
        audited_opaque = owner is not None and _is_expected_opaque_dispatch(event.operator, owner)
        # ``_operator_name`` yields overload-qualified names (``aten.equal.default``);
        # the allowlists hold overload-stripped base names.
        base_operator = (
            event.operator.rsplit(".", 1)[0] if event.operator.count(".") >= 2 else event.operator
        )
        if event.outcome == "raised":
            if owner_accounted or audited_opaque or event.in_replacement_hook:
                continue
            facts.append(
                {
                    "kind": "caught_exception_control",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": event.exception_type,
                    "mutates": bool(event.mutates),
                }
            )
        elif event.outcome == "returned_host_or_none":
            if owner_accounted or audited_opaque or event.in_replacement_hook:
                continue
            if base_operator in HOST_ESCAPE_OPERATORS:
                # Witnessed exactly by the tensor->host escape net.
                continue
            if event.state_view_accessor:
                continue
            facts.append(
                {
                    "kind": "opaque_side_effect" if event.mutates else "unmodeled_host_return",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": None,
                    "mutates": bool(event.mutates),
                }
            )
        elif event.outcome == "returned_tensor":
            if owner_accounted or audited_opaque or event.in_replacement_hook:
                continue
            if event.state_view_accessor:
                continue
            if not event.mutates and base_operator in _PURE_VIEW_DISPATCH_OPERATORS:
                continue
            if not event.mutates and event.contained_view:
                # Audited span-contained ``as_strided`` (DLPack/array-interop restride).
                continue
            facts.append(
                {
                    "kind": "opaque_side_effect" if event.mutates else "unmodeled_tensor_return",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": None,
                    "mutates": bool(event.mutates),
                }
            )
        elif event.outcome == "started":
            # A dispatch that neither returned nor raised cannot exist on a completed
            # forward; record fail-closed rather than silently passing (INV-1).
            facts.append(
                {
                    "kind": "unclassified_event",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": None,
                    "mutates": bool(event.mutates),
                }
            )
        else:  # pragma: no cover - unreachable by construction
            raise AssertionError(
                f"Internal invariant violation: unhandled dispatch outcome {event.outcome!r}; "
                "every outcome value must have an explicit ledger disposition (INV-1)."
            )
    if facts:
        _RUNNABLE_LEDGER_FACTS.setdefault(state.trace, []).extend(facts)


def _whole_storage_uint8(source: torch.Tensor) -> torch.Tensor:
    """Return a ``uint8`` tensor viewing ``source``'s ENTIRE untyped storage (all bytes).

    A zero-copy alias (``source.numpy()`` / ``source.untyped_storage()`` / a raw ``data_ptr``)
    shares the WHOLE storage, not just ``source``'s element extent: ``np.as_strided`` and a
    storage ``__setitem__`` can write bytes OUTSIDE ``source``'s view window (r15-H2). Comparing
    the whole aliased storage -- not ``source.detach().clone()`` (its own extent only) -- makes ANY
    host write anywhere in the shared storage detectable at forward end.
    """

    untyped = source.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=source.device)
    view.set_(untyped, 0, (untyped.nbytes(),), (1,))
    return view


def _snapshot_writeback_source(state: _WitnessState, source: torch.Tensor) -> None:
    """Record a before-image of a mutable zero-copy alias source for later write-back detection.

    Snapshots ``source``'s version and a detached byte clone of its WHOLE untyped storage under
    ``pause_logging`` (so the clone is not itself captured or censused) and holds a strong ref to
    ``source`` so the shared storage stays alive until the forward-end comparison. Snapshotting the
    full aliased storage (not just ``source``'s element extent) closes the r15-H2 out-of-extent
    gap: a host write through the alias's storage handle to bytes OUTSIDE ``source``'s view window
    (storage ``__setitem__`` / ``np.as_strided``) is still caught. A source that cannot be
    snapshotted (e.g. a meta tensor with no storage) fails closed immediately.
    """

    try:
        with _state.pause_logging():
            version = tensor_version_or_none(source)
            before = _whole_storage_uint8(source).clone()
    except (RuntimeError, TypeError, NotImplementedError):
        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
        return
    state.writeback_watch.append((source, version, before))


def _iter_dispatch_tensors(
    args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> Iterator[torch.Tensor]:
    """Yield every ``torch.Tensor`` operand of a dispatch, flattening list/tuple containers.

    Aten operands are tensors, scalars, or (for ``cat`` / ``stack`` / ``_foreach_*``) lists/tuples
    of tensors; this walks those one-deep-or-more without importing a pytree so the per-consumption
    watch can see every tensor an op reads.
    """

    stack: list[Any] = list(args)
    if kwargs:
        stack.extend(kwargs.values())
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            yield item
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def _sample_writeback_at_consumption(
    state: _WitnessState, args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> None:
    """Detect a transient host write-back that is LIVE when a traced op CONSUMES a watched source.

    A mutable zero-copy alias (``numpy`` / ``__array__`` / a storage handle) can be written,
    consumed by a downstream traced op, then byte-exactly RESTORED before forward end -- so the
    single end-of-forward compare (:func:`_check_writeback_watch`) sees ``before == after`` and
    would falsely VERIFY (r16-H1 TOCTOU). Sampling the watched source's WHOLE-STORAGE bytes at each
    CONSUMPTION catches the write while it is live in a traced op's input, then rolls it back.

    Soundness (no over-trigger). Only a byte difference with the source's version UNCHANGED since
    the exposure is flagged:

    * version UNCHANGED + bytes differ -> NO tracked in-place op touched the storage, so the diff is
      an opaque host write-back that is LIVE for this traced consumer -> UNVERIFIABLE (the TOCTOU);
    * version BUMPED -> a TRACKED, replayable in-place op is responsible for the diff, so the
      per-consumption sample defers to the end-of-forward compare (which conservatively handles a
      version-bumped byte diff). Flagging it here would falsely trip a legitimate tracked-op
      sequence that later restores the bytes (e.g. ``arr=y.numpy(); y.add_(1); z=y*2; y.sub_(1)``).

    A read-only ``.numpy().sum()`` never changes the bytes, so it is never flagged. The scan runs
    only while a mutable alias is live (``writeback_watch`` non-empty) and only compares a watched
    source when THIS op actually consumes its storage, so a transient write that never reaches a
    traced consumer of the source (restored before it is read) stays honestly VERIFIED.
    """

    if not state.writeback_watch and not _has_state_toctou_watch(state.trace):
        return
    # INV-2 annotation (r37): the ``data_ptr`` matching below is ATTRIBUTION-ONLY
    # identity -- it decides which watched source a consuming op MIGHT touch, and a
    # missed match merely defers to the end-of-forward WHOLE-STORAGE content compare
    # (which needs no pointer reasoning at all). Pointer identity is never used as a
    # disjointness proof here, so the absolute-interval engine is not required.
    try:
        with _state.pause_logging():
            consumed_ptrs: set[int] = set()
            for operand in _iter_dispatch_tensors(args, kwargs):
                try:
                    consumed_ptrs.add(operand.untyped_storage().data_ptr())
                except (RuntimeError, TypeError, NotImplementedError):
                    continue
            if not consumed_ptrs:
                return
            if _sample_state_toctou_at_consumption(state, consumed_ptrs):
                return
            for source, version, before in state.writeback_watch:
                try:
                    if source.untyped_storage().data_ptr() not in consumed_ptrs:
                        continue
                    if tensor_version_or_none(source) != version:
                        continue
                    if not torch.equal(
                        _whole_storage_uint8(source), before
                    ):  # byte-exact uint8 view
                        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                        return
                except (RuntimeError, TypeError, NotImplementedError):
                    _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                    return
    except (RuntimeError, TypeError, NotImplementedError):
        return


def _has_state_toctou_watch(trace: Any) -> bool:
    """Return whether the active trace has registered state byte watches.

    Returns
    -------
    bool
        ``True`` when buffer/parameter write tracking has live state snapshots.
    """

    tracker = getattr(trace, "_buffer_write_tracker", None)
    if tracker is None:
        return False
    param_snapshots = getattr(tracker, "address_to_param_snapshot", None)
    buffer_snapshots = getattr(tracker, "address_to_expected_storage_snapshot", None)
    return bool(param_snapshots) or bool(buffer_snapshots)


def _sample_state_toctou_at_consumption(state: _WitnessState, consumed_ptrs: set[int]) -> bool:
    """Detect transient registered-state mutations when a traced op consumes them.

    Parameters
    ----------
    state:
        Active completeness-witness state.
    consumed_ptrs:
        Storage data pointers consumed by the current dispatcher operation.

    Returns
    -------
    bool
        ``True`` when an opaque state write-back was detected and recorded.
    """

    tracker = getattr(state.trace, "_buffer_write_tracker", None)
    if tracker is None:
        return False
    if _sample_param_toctou_at_consumption(state, tracker, consumed_ptrs):
        return True
    return _sample_buffer_toctou_at_consumption(state, tracker, consumed_ptrs)


def _sample_param_toctou_at_consumption(
    state: _WitnessState, tracker: Any, consumed_ptrs: set[int]
) -> bool:
    """Compare consumed parameters against their pre-forward byte snapshots.

    Parameters
    ----------
    state:
        Active completeness-witness state.
    tracker:
        Buffer/parameter write tracker attached to the active trace.
    consumed_ptrs:
        Storage data pointers consumed by the current dispatcher operation.

    Returns
    -------
    bool
        ``True`` when a consumed parameter differs from its pre-forward bytes.
    """

    tensors = getattr(tracker, "address_to_param_tensor", None)
    snapshots = getattr(tracker, "address_to_param_snapshot", None)
    if not isinstance(tensors, dict) or not isinstance(snapshots, dict):
        return False
    for address, source in tuple(tensors.items()):
        if not isinstance(source, torch.Tensor):
            continue
        try:
            if source.untyped_storage().data_ptr() not in consumed_ptrs:
                continue
        except (RuntimeError, TypeError, NotImplementedError):
            continue
        baseline = snapshots.get(address)
        if not isinstance(baseline, tuple) or not baseline:
            continue
        before = baseline[0]
        if not isinstance(before, torch.Tensor):
            continue
        try:
            if not torch.equal(_whole_storage_uint8(source), before):  # byte-exact uint8 view
                _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                return True
        except (RuntimeError, TypeError, NotImplementedError):
            _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
            return True
    return False


def _sample_buffer_toctou_at_consumption(
    state: _WitnessState, tracker: Any, consumed_ptrs: set[int]
) -> bool:
    """Compare consumed buffers against the journal-advanced expected bytes.

    Parameters
    ----------
    state:
        Active completeness-witness state.
    tracker:
        Buffer/parameter write tracker attached to the active trace.
    consumed_ptrs:
        Storage data pointers consumed by the current dispatcher operation.

    Returns
    -------
    bool
        ``True`` when a consumed buffer differs from its journal-advanced bytes.
    """

    tensors = getattr(tracker, "address_to_tensor", None)
    snapshots = getattr(tracker, "address_to_expected_storage_snapshot", None)
    if not isinstance(tensors, dict) or not isinstance(snapshots, dict):
        return False
    for address, source in tuple(tensors.items()):
        if not isinstance(source, torch.Tensor):
            continue
        try:
            if source.untyped_storage().data_ptr() not in consumed_ptrs:
                continue
        except (RuntimeError, TypeError, NotImplementedError):
            continue
        expected = snapshots.get(address)
        if not isinstance(expected, torch.Tensor):
            continue
        try:
            if not torch.equal(_whole_storage_uint8(source), expected):  # byte-exact uint8 view
                _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                return True
        except (RuntimeError, TypeError, NotImplementedError):
            _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
            return True
    return False


def _make_invisible_escape_wrapper(original: Any, state: _WitnessState, name: str) -> Any:
    """Wrap a tensor->host conversion method to record its SOURCE, then call through.

    The wrapper records the receiver tensor (the escape SOURCE) into the shared escape
    tables, gated to the active trace so a TorchLens-internal conversion (run under
    ``pause_logging``) is never mistaken for a user escape. Fires on every thread under
    the r43 owner-vs-non-owner rule: the OWNER thread (gated on ``_logging_enabled``)
    records through the full precise ladder plus the blanket flags/watches; a NON-owner
    thread (gated on ``belt_armed``) ceilings only when it touches a captured tensor and
    otherwise records nothing (a benign background thread touching its OWN tensors never
    ceilings the capture). For a
    mutable zero-copy alias conversion (``numpy`` / ``__array__``) it also records a
    before-image so a subsequent host write-back through the alias is detected at
    forward end. It always calls the original method unchanged, so values, goldens, and
    outputs are byte-identical. r43: the OWNER thread keeps the precise ladder; a NON-owner
    thread's captured-tensor touch ceilings via :func:`_nonowner_escape_observe`.
    """

    # Storage-pointer bridges (untyped_storage / storage / data_ptr) are watched for host
    # write-back but are WATCH-ONLY: a read-only pointer/identity check exposes no scalar value,
    # so recording them as value-escape sources would over-trigger and is deliberately skipped.
    is_storage_bridge = name in STORAGE_BRIDGE_ESCAPE_FUNCS
    watch_writeback = name in MUTABLE_ALIAS_ESCAPE_FUNCS or is_storage_bridge
    record_source = not is_storage_bridge
    # ``data_ptr()`` alone leaks a RAW pointer no forward-end byte watch can re-inspect (r15-H1);
    # a genuine user call fails closed to UNVERIFIABLE. ``untyped_storage()`` / ``storage()`` keep
    # the watch-only write-back treatment (their value reads are already UNVERIFIABLE).
    is_raw_pointer = name == "data_ptr"
    # ``untyped_storage()`` / ``storage()`` expose the input's BASE storage; its byte GEOMETRY
    # (``nbytes()`` / ``size()``) can steer control flow the shape+dtype contract does not pin
    # (r29-C1, F4). Record it against the model-input leaf here at exposure time.
    records_storage_geometry = name in {"untyped_storage", "storage"}
    # r65 (closes r64 F2): a zero-copy VIEW export pins the receiver's full layout with no
    # accessor call at all -- ``numpy()``/``__array__`` expose ndarray ``.strides``/``.flags``
    # and DLPack capsules carry strides + byte offset. On a STATE-derived receiver that
    # geometry is a pure function of the slot's physical form (a view-of-state receiver
    # attributes to the slot exactly as r63), so the export records the exact-layout read
    # kinds. ``tolist()`` COPIES (layout-safe) and is deliberately excluded.
    records_state_view_geometry = name in {"numpy", "__array__", "__dlpack__"}

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        result_holder: dict[str, Any] = {}
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled:
                    if record_source:
                        _record_escape_source_tensor(state.trace, self, invisible=True)
                    if records_state_view_geometry:
                        _observe_state_metadata_read(
                            state.trace, self, STATE_METADATA_MIRROR["stride"][1]
                        )
                        _observe_state_metadata_read(
                            state.trace, self, STATE_METADATA_MIRROR["storage_offset"][1]
                        )
                    if records_storage_geometry and not _internal_read_active():
                        storage = original(self, *args, **kwargs)
                        result_holder["value"] = storage
                        _maybe_record_input_storage_geometry(state.trace, self, storage)
                        # r65: a storage handle exposed off a STATE-derived receiver pins
                        # the slot's base-storage byte count (``.nbytes()``/``.size()`` are
                        # one attribute away), a fact invariant across every view/alias of
                        # the slot -- record the geometry read kind at exposure time,
                        # mirroring the input-side belt above (closes the r64 F3
                        # larger-base offset-0-contiguous class via the
                        # ``storage_nbytes_is_tight`` signature dim).
                        _observe_state_metadata_read(
                            state.trace, self, STATE_METADATA_MIRROR["storage_nbytes"][1]
                        )
                    # A raw ``data_ptr()`` pointer is unobservable; only a genuine USER call
                    # (internal marker inactive -- TorchLens's own bookkeeping ``data_ptr``
                    # reads run under it) fails closed so the tensor's subsequent value cannot
                    # be silently VERIFIED.
                    if is_raw_pointer and not _internal_read_active():
                        _HOST_ESCAPE_RAW_POINTER.add(state.trace)
                    # TorchLens's OWN capture-internal aliasing / version bookkeeping reads
                    # storage pointers (``aliasing._tensors_alias`` ->
                    # ``untyped_storage().data_ptr()``) under the explicit ``internal_scalar_read``
                    # marker. Those are NOT user exposures: snapshotting them and byte-comparing
                    # under the r14-H1 gate would falsely trip on a later legitimate TRACKED
                    # in-place op. Only watch a storage bridge when the marker is inactive -- a
                    # genuine user ``data_ptr()`` / ``storage()`` call. (The numpy / __array__
                    # mutable alias is never called internally, so it is always watched, as r13.)
                    if watch_writeback and not (is_storage_bridge and _internal_read_active()):
                        _snapshot_writeback_source(state, self)
            elif state.belt_armed:
                # r43: a non-owner touch of a captured tensor (this receiver, or a captured-
                # derived alias by storage identity) ceilings the capture. A benign OWN-tensor
                # conversion records nothing.
                _nonowner_escape_observe(state, self)
        if "value" in result_holder:
            return result_holder["value"]
        return original(self, *args, **kwargs)

    return wrapper


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
    """

    try:
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

        return any(
            isinstance(mode, _CompletenessDispatchMode)
            for mode in _get_current_dispatch_mode_stack()
        )
    except Exception:  # pragma: no cover - defensive; treat unknown as census-active (skip)
        return True


def _make_host_value_escape_method(original: Any, state: _WitnessState, name: str) -> Any:
    """Wrap a tensor->host VALUE method to record its tensor operand SOURCES (r39 hon2_1).

    ``item`` / ``__bool__`` / ``__int__`` / ``__float__`` / ``__index__`` / ``__complex__`` and
    the pure predicates ``equal`` / ``allclose`` / ``is_nonzero`` all read a captured tensor's
    VALUE out to the host. The aten census sees them through ``aten._local_scalar_dense`` /
    ``aten.equal`` -- EXCEPT inside torch's own ``_disable_current_modes()`` regions (tensor
    string formatting; explicit predicate guards), which pop the census TorchDispatchMode
    (measured E6). This method patch fires regardless of dispatch-mode state, feeding the SAME
    ``_record_escape_source_tensor(..., invisible=True)`` attribution ladder as the census, so
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

    del name  # recorded uniformly; the operand set determines the source, not the spelling

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if (
                    _state._logging_enabled
                    and not _internal_read_active()
                    and not _completeness_census_active()
                ):
                    _record_escape_source_tensor(state.trace, self, invisible=True)
                    for value in (*args, *kwargs.values()):
                        if isinstance(value, torch.Tensor):
                            _record_escape_source_tensor(state.trace, value, invisible=True)
            elif state.belt_armed:
                # r43: a NON-owner value escape ceilings iff its receiver OR any tensor operand
                # is a captured tensor (the census is thread-local, so the belt is PRIMARY here).
                _nonowner_escape_observe(state, self)
                for value in (*args, *kwargs.values()):
                    if isinstance(value, torch.Tensor):
                        _nonowner_escape_observe(state, value)
        return original(self, *args, **kwargs)

    return wrapper


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
        if _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if (
                    _state._logging_enabled
                    and not _internal_read_active()
                    and not _completeness_census_active()
                ):
                    for value in (*args, *kwargs.values()):
                        if isinstance(value, torch.Tensor):
                            _record_escape_source_tensor(state.trace, value, invisible=True)
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
        if _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled:
                    for value in (*args, *kwargs.values()):
                        if isinstance(value, torch.Tensor):
                            _record_escape_source_tensor(state.trace, value, invisible=True)
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


def _make_invisible_escape_property(descriptor: Any, state: _WitnessState) -> property:
    """Wrap a zero-copy buffer PROPERTY to record its SOURCE tensor, then read through.

    Used for ``__cuda_array_interface__`` (a non-callable getset descriptor the method
    patch cannot wrap). The property getter records the receiver tensor as an escape
    source under the same active-forward gate (r41: on every thread -- in-window
    fail-closed, foreign positive-only), then delegates to the original descriptor so
    the returned value is byte-identical.
    """

    def getter(self: torch.Tensor) -> Any:
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled:
                    _record_escape_source_tensor(state.trace, self, invisible=True)
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

    return property(getter)


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

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
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
                    state_route = STATE_METADATA_MIRROR.get(name)
                    if state_route is not None and state_route[0] == _STATE_ROUTE_READ_KIND:
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

    originals: dict[str, Any] = {}
    for name in INVISIBLE_HOST_ESCAPE_FUNCS | STORAGE_BRIDGE_ESCAPE_FUNCS:
        original = getattr(torch.Tensor, name, None)
        if original is None:
            continue
        try:
            setattr(torch.Tensor, name, _make_invisible_escape_wrapper(original, state, name))
        except (TypeError, AttributeError):
            continue
        originals[name] = original
    # r39 hon2_1: mode-independent belt for the aten census -- the scalar numeric protocol
    # (``item``/``__bool__``/``__int__``/``__float__``/``__index__``/``__complex__``) and the
    # pure predicates (``equal``/``allclose``/``is_nonzero``). These fire regardless of
    # dispatch-mode state, so a scalar/predicate escape inside torch's own
    # ``_disable_current_modes()`` (tensor string formatting; explicit guards) still hits a
    # Python observer. Several names are getset/slot members of the C ``TensorBase`` and NOT in
    # ``torch.Tensor.__dict__``; setting them installs a SHADOW that restore must DELETE (never
    # set back to the base slot). A required-observer install failure fails the capture closed.
    host_value_method_restore: dict[str, tuple[bool, Any]] = {}
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
    module_originals: list[tuple[Any, str, Any]] = []
    for module, func_name in _MODULE_ESCAPE_TARGETS():
        original_func = getattr(module, func_name, None)
        if original_func is None:
            continue
        try:
            setattr(module, func_name, _make_module_escape_wrapper(original_func, state))
        except (TypeError, AttributeError):
            continue
        module_originals.append((module, func_name, original_func))
    # r39 hon2_1: the ``torch.*`` MODULE predicate spellings (``torch.equal`` / ``torch.allclose``
    # / ``torch.is_nonzero``) return a raw Python bool DIRECTLY from the dispatcher and, under an
    # explicit ``_disable_current_modes()`` region, bypass the census (E6). Record every tensor
    # operand -- the same shared source table as the Tensor-method belt.
    for predicate_name in HOST_VALUE_ESCAPE_MODULE_FUNCS:
        original_predicate = torch_attr(predicate_name)  # r47 secD_1: no lazy ``torch.__getattr__``
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
    storage_originals: list[tuple[Any, Any]] = []
    for storage_cls in _STORAGE_RAW_POINTER_TARGETS():
        storage_original = storage_cls.data_ptr
        try:
            setattr(
                storage_cls, "data_ptr", _make_storage_raw_pointer_wrapper(storage_original, state)
            )
        except (TypeError, AttributeError):
            continue
        storage_originals.append((storage_cls, storage_original))
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
    # Model-input METADATA-PREDICATE observers (r27-H2): ``is_contiguous`` / ``stride``
    # methods and the ``requires_grad`` getset descriptor. Read-through recorders gated to
    # MODEL-INPUT receivers only; a model that never reads input layout/grad records nothing.
    metadata_originals: dict[str, Any] = {}
    stride_original = getattr(torch.Tensor, "stride", None)
    for name in INPUT_METADATA_PREDICATE_FUNCS:
        original = getattr(torch.Tensor, name, None)
        if original is None:
            continue
        try:
            setattr(
                torch.Tensor,
                name,
                _make_input_metadata_wrapper(original, state, name, stride_original),
            )
        except (TypeError, AttributeError):
            continue
        metadata_originals[name] = original
    # Model-input BOOLEAN metadata METHODS beyond the layout trio (r31): ``is_conj`` /
    # ``is_neg`` / ``is_inference`` / ``is_pinned`` / ``is_shared`` / ``is_coalesced`` /
    # ``_is_view``. Feature-detected (an accessor absent on the running torch is skipped) and
    # gated to model-input receivers/aliases only; a model that never reads them records nothing.
    bool_method_originals: dict[str, Any] = {}
    for name in INPUT_METADATA_BOOL_METHODS:
        original = getattr(torch.Tensor, name, None)
        if original is None or not callable(original):
            continue
        try:
            setattr(torch.Tensor, name, _make_input_metadata_bool_method(original, state, name))
        except (TypeError, AttributeError):
            continue
        bool_method_originals[name] = original
    # ``requires_grad`` / ``grad_fn`` / ``is_leaf`` live as getset descriptors on the C BASE
    # class (``torch._C.TensorBase``), not in ``torch.Tensor.__dict__``; patching installs a
    # SHADOWING property on ``torch.Tensor`` itself, so restore must DELETE the shadow when the
    # name was not originally in ``torch.Tensor.__dict__``.
    grad_property_restore: dict[str, tuple[bool, Any]] = {}
    for prop_name in INPUT_METADATA_PROPERTY_NAMES:
        prop_descriptor = inspect.getattr_static(torch.Tensor, prop_name, None)
        if prop_descriptor is None or not hasattr(prop_descriptor, "__get__"):
            continue
        shadowed = prop_name in torch.Tensor.__dict__
        try:
            setattr(
                torch.Tensor,
                prop_name,
                _make_input_metadata_grad_property(prop_descriptor, state, prop_name),
            )
        except (TypeError, AttributeError):
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
    torch_ops_call_restore: list[tuple[type, Any]] = []
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
    private_c_call_restore: list[tuple[Any, str, Any]] = []
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
    try:
        yield
    finally:
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
        for name, original in originals.items():
            try:
                setattr(torch.Tensor, name, original)
            except (TypeError, AttributeError):
                pass
        for module, func_name, original_func in module_originals:
            try:
                setattr(module, func_name, original_func)
            except (TypeError, AttributeError):
                pass
        for storage_cls, storage_original in storage_originals:
            try:
                setattr(storage_cls, "data_ptr", storage_original)
            except (TypeError, AttributeError):
                pass
        for name, descriptor in property_originals.items():
            try:
                setattr(torch.Tensor, name, descriptor)
            except (TypeError, AttributeError):
                pass
        for name, original in metadata_originals.items():
            try:
                setattr(torch.Tensor, name, original)
            except (TypeError, AttributeError):
                pass
        for name, original in bool_method_originals.items():
            try:
                setattr(torch.Tensor, name, original)
            except (TypeError, AttributeError):
                pass
        for prop_name, (was_shadowed, original_descriptor) in grad_property_restore.items():
            try:
                if was_shadowed:
                    setattr(torch.Tensor, prop_name, original_descriptor)
                else:
                    delattr(torch.Tensor, prop_name)
            except (TypeError, AttributeError):
                pass
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
        _check_writeback_watch(state)


def _STORAGE_RAW_POINTER_TARGETS() -> tuple[Any, ...]:
    """Return storage classes whose ``data_ptr`` accessor leaks the raw pointer (r16-C1).

    ``tensor.untyped_storage()`` yields a ``torch.UntypedStorage`` and ``tensor.storage()`` a
    ``torch.TypedStorage``; ``data_ptr()`` on either hands out the same raw pointer the r15 Tensor
    patch fails closed on. Both are Python-visible classes whose ``data_ptr`` method is patchable.
    """

    targets: list[Any] = []
    for name in ("UntypedStorage", "TypedStorage"):
        cls = torch_attr(name)  # r47 secD_1: no lazy ``torch.__getattr__``
        if cls is not None and hasattr(cls, "data_ptr"):
            targets.append(cls)
    return tuple(targets)


def _MODULE_ESCAPE_TARGETS() -> tuple[tuple[Any, str], ...]:
    """Return ``(module, attribute)`` pairs for module-level zero-copy export C bindings."""

    targets: list[tuple[Any, str]] = []
    dlpack_mod = getattr(torch.utils, "dlpack", None)
    if dlpack_mod is not None:
        targets.append((dlpack_mod, "to_dlpack"))
    c_mod = getattr(torch, "_C", None)
    if c_mod is not None and hasattr(c_mod, "_to_dlpack"):
        targets.append((c_mod, "_to_dlpack"))
    return tuple(targets)


def _check_writeback_watch(state: _WitnessState) -> None:
    """Detect a host write-back through any watched mutable zero-copy alias at forward end.

    The honest rule is keyed on the WHOLE aliased storage's BYTES, never on the version counter
    (r14-H1) and never on only the view's element extent (r15-H2). A watched source whose whole
    storage is UNCHANGED since the mutable-alias exposure was only read: it stays VERIFIED (a pure
    read-only ``.numpy().sum()`` / storage-pointer identity check is not over-triggered). A watched
    source whose storage bytes CHANGED anywhere -- INCLUDING outside the view's own window (a
    storage ``__setitem__`` / ``np.as_strided`` write) -- is UNVERIFIABLE, in BOTH sub-cases:

    * version UNCHANGED -> no tracked op touched it, so the byte diff can only be an opaque host
      write-back through the alias (no aten dispatch, no version bump) -> host write-back;
    * version BUMPED -> a tracked in-place op ALSO touched the source since the exposure, so the
      raw byte comparison is AMBIGUOUS -- the diff could be the tracked op OR an additional host
      write layered on top, and cannot prove the ABSENCE of a host write -> conservatively opaque.

    Gating detection on ``version unchanged`` (the pre-r14 behaviour) let a tracked in-place op
    that bumps the version AFTER the ``.numpy()`` / ``.data`` snapshot skip the byte compare, so a
    host write-back on the same storage went undetected and the run falsely VERIFIED. Comparing
    bytes alone closes that gate while keeping read-only exposures honestly VERIFIED.
    """

    if not state.writeback_watch:
        return
    try:
        with _state.pause_logging():
            for source, _version, before in state.writeback_watch:
                try:
                    if not torch.equal(
                        _whole_storage_uint8(source), before
                    ):  # byte-exact uint8 view
                        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                        break
                except (RuntimeError, TypeError, NotImplementedError):
                    _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                    break
    finally:
        state.writeback_watch.clear()


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
                # A ``.data``-property accessor view (``aten.detach``/``aten.alias``) on a
                # registered buffer -- the intrinsic, legitimately-uncaptured dispatch of the
                # ``self.b.data.copy_(x)`` buffer-write idiom. The completeness backstop credits
                # these apples-to-apples against the dispatch census (see
                # ``validation.core.completeness_backstop_counts``); a genuine untraced op is
                # never flagged here (unowned + non-mutating + pure-view + buffer only).
                "state_view_accessor": event.state_view_accessor,
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
        ledger=record_escapes,
    )
    mode_context = _CompletenessDispatchMode(state)
    # A runnable capture additionally observes census-INVISIBLE ``.tolist()`` /
    # ``.numpy()`` / ``__array__`` escapes via a scoped method patch so every escape
    # mechanism feeds one uniform source-witness pass. The patch is a pure observer,
    # restored unconditionally, and is skipped entirely for the non-runnable census path.
    if record_escapes:
        # r35 I2: arm wrapper ownership tokens so raised / host-returning dispatch
        # events can be attributed to their exact wrapper owner (the ledger's
        # owner-accounted discharge rule) even with both shadow modes off.
        prior_ledger_armed = _state._runnable_ledger_armed
        _state._runnable_ledger_armed = True
        # r43: publish the witness state so the wrappers.py string-hook interception can
        # classify owner vs non-owner (the ONE place a non-owner thread must not flip the
        # global ``pause_logging`` toggle). Cleared FIRST on exit.
        global _ACTIVE_WITNESS_STATE
        prior_active_state = _ACTIVE_WITNESS_STATE
        _ACTIVE_WITNESS_STATE = state
        try:
            with _observe_invisible_host_escapes(state), mode_context:
                try:
                    yield
                finally:
                    if mode == "shadow":
                        _finalize_census(state)
                    _finalize_runnable_ledger(state)
        finally:
            _ACTIVE_WITNESS_STATE = prior_active_state
            _state._runnable_ledger_armed = prior_ledger_armed
        return
    with mode_context:
        try:
            yield
        finally:
            if mode == "shadow":
                _finalize_census(state)

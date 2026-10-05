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

Thread posture (r43, locked by design): the aten census is OWNER-thread-scoped (a
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

# ruff: noqa: F401

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
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import torch
import torch._ops as _torch_ops  # r47 hon2_1: enumerate the ``torch.ops.*`` __call__ classes
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ..._errors import TorchLensCaptureGapWarning
from ..._split_rebind import (
    rebind_contextmanager as _rebind_contextmanager,
    rebind_function as _rebind_function,
)
from ...errors import ScalarEscapeWarning
from ...utils import _torch_compat
from ...utils._callable_safety import private_c_forward_op_module_names
from ...utils._torch_compat import (
    get_tensorbase_class,
    tensor_version_or_none,
)
from ...utils._torch_symbols import torch_attr
from . import (
    _completeness_boundaries as _completeness_boundaries,
    _completeness_cross_thread as _completeness_cross_thread,
    _completeness_dispatch as _completeness_dispatch,
    _completeness_dispatch_names as _completeness_dispatch_names,
    _completeness_escape_state as _completeness_escape_state,
    _completeness_finalize as _completeness_finalize,
    _completeness_metadata as _completeness_metadata,
    _completeness_origins as _completeness_origins,
    _completeness_patches as _completeness_patches,
    _completeness_storage as _completeness_storage,
)
from ._completeness_types import (
    AuditedCompletenessBoundary,
    _DispatchCallsite,
    _DispatchEvent,
    _PlainScalarEscapeState,
    _StorageOriginRegistry,
    _TensorOriginRegistry,
    _WitnessState,
)
from ._modes import _TorchLensDispatchMode
from ._tl import (
    DescriptorCompatProperty,
    get_buffer_address,
    get_tensor_label,
    get_tensor_meta,
    is_tensor_data_alias,
    session_meta_is_anchored,
)
from .buffer_writes import peek_buffer_write_tracker, session_validated_buffer_address
from .escape_detection import (
    ExpectedOriginalToken,
    _active_token,
    expected_original_call,
    mark_expected_original_accounted,
)

CompletenessWitnessMode = Literal["off", "shadow"]
"""Supported dispatcher-witness rollout modes."""

MAX_AUDITED_COMPLETENESS_BOUNDARIES = 9
"""Hard budget preventing expected-opaque wrapper scopes from growing unchecked."""

_TORCH_ROOT = Path(torch.__file__).resolve().parent
_TORCHLENS_ROOT = Path(__file__).resolve().parents[2]
_FRAMEWORK_FILENAME_VERDICTS: dict[str, bool] = {}
# Filename verdicts are keyed by ``co_filename``, which is attacker-of-bounds
# free but NOT bounded for generated-code / notebook / plugin workloads
# (``<ipython-input-N>``, exec'd templates): every unseen filename adds an
# entry for the process lifetime. FIFO-evict at the cap -- old filename
# verdicts carry no active-session value and recomputing one is a cheap
# ``Path.resolve`` pair.
_FRAMEWORK_FILENAME_VERDICTS_MAX_ENTRIES = 4096


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
    AuditedCompletenessBoundary(
        wrapper_name="autograd:grad",
        operator=None,
        reason=(
            "torch.autograd.grad executes a separately captured backward pass; its engine "
            "dispatches are not forward operations"
        ),
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

_HOST_ESCAPE_SOURCE_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()
"""Per-trace raw producing-op labels of tensor->host escape sources.

Kept off the Trace ``__dict__`` (and therefore out of portable-state scrub) in a
weak-keyed side table so the runnable descriptor can read escape sources without
registering a new serialized Trace field. Entries are dropped automatically when a
Trace is garbage collected.
"""

_HOST_ESCAPE_STATE_SOURCE_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = (
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

_HOST_ESCAPE_BOOL_SOURCE_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = (
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

_HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS: weakref.WeakKeyDictionary[
    Any, dict[str, list[tuple[str, int]]]
] = weakref.WeakKeyDictionary()
"""Per-trace user source locations where labelled bool tensors reached ``__bool__``."""

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

STORAGE_BRIDGE_ESCAPE_FUNCS = frozenset(
    {"untyped_storage", "storage", "_typed_storage", "data_ptr"}
)
"""Census-invisible tensor methods that hand the host a ZERO-COPY handle onto the source
tensor's raw storage, through which a host WRITE can mutate the tensor's bytes with no aten
dispatch, no version bump, and no escape record (r14-H3).

``tensor.untyped_storage()`` / ``tensor.storage()`` / ``tensor._typed_storage()`` (r67 C3:
the private spelling ``storage()`` itself delegates to) return a Storage object aliasing the
tensor's memory (``s.fill_(0)`` / ``s[0] = 99`` mutate the source), and ``tensor.data_ptr()``
hands out the raw data pointer that ``ctypes`` / a foreign kernel writes through directly. Like
``.numpy()`` / ``.data`` these bypass the dispatcher entirely, so a write-back leaves the sparse
replay recomputing the pre-write value and would falsely VERIFY. They are therefore bracketed with
the SAME before/after byte snapshot as the mutable numpy alias (see ``_observe_invisible_host_escapes``
-> ``_check_writeback_watch``): a source whose WHOLE-STORAGE bytes changed after exposure -> opaque
host write-back -> UNVERIFIABLE. Storage ACQUISITION is ORIGIN+WATCH-ONLY (r67 C3/C6): the
bridge registers the returned handle in the capture-scoped storage-origin map and keeps the
byte watch, but records NO read kind and NO geometry fact -- a discarded handle is not an
observation; the ACTUAL accessor call on the handle (``.nbytes()`` / ``.is_shared()`` / ...)
is what records, through ``STORAGE_METADATA_ACCESSOR_DISPOSITIONS``. ``data_ptr()`` is the
exception (r15-H1): it hands out a RAW pointer that a foreign READ (baking a stale literal) or
a post-snapshot WRITE can use with no observable trace at all, so a genuine user ``data_ptr()``
call fails closed to UNVERIFIABLE (see ``_HOST_ESCAPE_RAW_POINTER``), never relying on the byte
watch alone.
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
through the SAME ``_record_escape_source_tensor(...)`` attribution ladder as
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

# --- Accessor FAMILIES governing alias/view attribution (r31) -------------------------------
#
# A metadata read whose receiver is not the input leaf OBJECT itself is attributed by the
# receiver's relationship to an input leaf, which differs per accessor family:

_INPUT_METADATA_LAYOUT_NAMES = frozenset({"is_contiguous", "stride", "storage_offset"})
"""LAYOUT accessors: value DIFFERS between a leaf and a derived view (``x.t().stride()`` !=
``x.stride()``), so a derived-view read fails closed (cannot be re-derived from the runtime
leaf); a ``.data`` / ``.detach()`` storage-alias (identical geometry) records the leaf fact."""

INPUT_DERIVED_LAYOUT_FACT_NAME = "derived_layout_read"
"""Synthetic ``model_input_metadata`` fact name for an INPUT-DERIVED activation layout read
(r73 F1).

A layout-trio read (``is_contiguous`` / ``stride`` / ``storage_offset``) whose receiver is a
genuinely NEW activation (fresh storage -- not the input leaf, not a storage alias, not a
``_base``-linked view) previously recorded NOTHING: elementwise/conv ops PROPAGATE the input's
memory format, so ``(x * 2).is_contiguous(memory_format=torch.channels_last)`` steers a branch
on the runtime input's layout while the input contract pins only shape+dtype -- a
channels_last twin of the capture input replayed the captured arm as a false VERIFIED
(confirmed r72 hon1 F1). The traced value DAG makes the rooting attributable: the receiver's
``OpEvent.input_ancestors`` names exactly the model-input ops its VALUE derives from, so the
observer records this fact -- carrying the ROOTING LEAF's capture-time stride tuple -- on each
ancestor input site. At run time the executor compares the RAW runtime leaf's strides against
the recorded tuple and, on ANY difference, ceilings the run UNVERIFIABLE (never DIVERGED: a
changed input layout does not PROVE the intermediate's layout differs -- e.g. a ``reshape``
between input and read can canonicalize it -- so the honest verdict is "cannot verify", per
the r35 three-state rule). A same-stride runtime input compares equal and stays VERIFIED, so
honest channels_last-on-channels_last models are untouched (zero collateral). The mirrored
consumer constants live in ``torchlens._io.runnable._INPUT_METADATA_FACT_NAMES`` and
``torchlens._runnable_execution._INPUT_DERIVED_LAYOUT_FACT_NAME``.

Scope / documented residuals (future intermediate-metadata escape kinds route HERE):

* LAYOUT trio only. Non-layout bool accessors on a fresh activation (``(x * 2).is_pinned()``)
  are allocator/ambient facts, not input-propagated ones; extending coverage to a new
  input-propagated metadata kind means adding its accessor family to this same
  ancestry-attributed net, not a new mechanism.
* An UNLABELED receiver NEVER records nothing (r75 F1 -- the r73 fail-open here reopened the
  escape one ``.data`` away from the fixed spelling). It resolves through the dispatch-origin
  ledger's leaf origins, then live captured-storage identity, and otherwise FAILS CLOSED
  (``_INPUT_METADATA_VIEW_READ`` -> completeness downgrade -> UNVERIFIABLE), mirroring the
  sibling ``_HOST_ESCAPE_UNATTRIBUTABLE_*`` nets for the same receiver class. The same
  ancestry-integrity rule covers TRANSITIVE laundering: a labeled receiver whose parent chain
  passes through an op with ``unattributed_tensor_args`` (``(x * 2).data[0]``,
  ``(y.data * 1.0)``, ``torch.cat([y1, y.data])``) is re-resolved or fails closed -- an
  ancestry-orphaned empty ``input_ancestors`` is never mistaken for state-rooted. A hidden
  pre-capture attribute tensor (no label, no ledger entry, no captured storage) therefore
  ceilings honestly instead of silently verifying; an escape-laundered round trip through a
  LOGGED boundary op (``torch.from_numpy(y.numpy())``) remains the documented record-nothing
  residual -- its ancestry is genuinely internal-rooted and the ``.numpy()`` escape itself is
  digest-witnessed separately.
* STATE-rooted intermediates (``(self.w * 2).is_contiguous()``) have NO input ancestor and
  record nothing here: the state-layout twin stays contract residual (3) -- state strides are
  canonicalized at save, so no runtime comparison basis exists (unlike the input side, where
  the runtime leaf supplies the layout fresh)."""

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
_RUNNABLE_INPUT_STORAGE_SITES: weakref.WeakKeyDictionary[Any, dict[int, list[Any]]] = (
    weakref.WeakKeyDictionary()
)

_ALIAS_EQUIVALENT = "equivalent"
"""Storage-alias classification: shares an input leaf's storage with IDENTICAL geometry
(``.data`` / ``.detach()``) -- a metadata read on it equals a direct leaf read."""

_ALIAS_DERIVED_VIEW = "derived_view"
"""Storage-alias classification: shares an input leaf's storage but with DIFFERENT geometry
(a derived view the sparse replay never re-derives) -- fails closed."""


_LAYOUT_ANCESTRY_CLEAN: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()
"""Per-trace memo of raw labels whose ENTIRE traced ancestry is attribution-intact (r75 F1).

A label enters only after a full parent-chain walk found no op with
``unattributed_tensor_args`` (and no unresolvable parent label), so repeated layout reads
on deep chains stay O(1). Taint is never cached: it fails the read closed immediately and
is rare by construction. Weak-keyed off the schema; dropped with the Trace."""


_STORAGE_REBIND_BARRIER_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = (
    weakref.WeakKeyDictionary()
)
"""Per-trace raw labels of storage-SWAPPING ``.data=`` rebind ops (r28 reconcile).

A ``t.data = rhs`` whose RHS lives on a DIFFERENT storage object swaps the
receiver's storage pointer -- the exact laundering primitive the r79/r81/r85
session belts refuse to attribute verdict-steering facts across (stale/forged
stamp launders, plain-attr and registered-buffer rebind layout branches). The
capture graph legitimately threads consumers through the emitted rebind op
(round-31 M6), but ancestry rooted through one of these labels must FAIL CLOSED
for layout/witness attribution exactly as the pre-M6 unattributed break did:
:func:`_layout_ancestry_tainted` treats a barrier label like an op with
``unattributed_tensor_args``. A pointer-PRESERVING rebind (``y.data =
y.view(...)``) is never registered here, so honest same-storage siblings keep
their attribution (r85) and input-strided same-storage rebinds keep honest
divergence semantics (hon1 V6)."""


# --- r65 Cluster X: THE authoritative state-metadata accessor mirror ------------------------
#
# The input-metadata net's authority is the union of four frozen constants
# (INPUT_METADATA_PREDICATE_FUNCS | INPUT_METADATA_BOOL_METHODS | INPUT_METADATA_PROPERTY_NAMES
# | {"storage_nbytes"}), which equals the 20 ACCESSOR names of the ``_INPUT_METADATA_FACT_NAMES``
# vocabulary in ``torchlens._io.runnable`` (r73 adds the vocabulary's one SYNTHETIC,
# ancestry-attributed fact -- ``derived_layout_read`` -- which owes no mirror row; see
# ``_INPUT_METADATA_SYNTHETIC_FACT_NAMES`` there). r63 mirrored only 5 of those 20 onto
# registered state, leaving an
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

STATE_METADATA_MIRROR: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        # -- layout trio (r63, unchanged; ``is_contiguous`` probed with an explicit
        #    memory_format resolves to the ``stride`` row's kind at the wrapper) --
        "is_contiguous": (_STATE_ROUTE_READ_KIND, "contiguous_default"),
        "stride": (_STATE_ROUTE_READ_KIND, "stride_exact"),
        "storage_offset": (_STATE_ROUTE_READ_KIND, "storage_offset"),
        # -- lazy dispatch bits (r63, unchanged) --
        "is_conj": (_STATE_ROUTE_READ_KIND, "is_conj"),
        "is_neg": (_STATE_ROUTE_READ_KIND, "is_neg"),
        # -- storage/creation placement bits (r65; r67 C3 observed-value rows): value is a
        #    pure function of the slot's storage, invariant across every view/alias. The
        #    wrappers additionally carry the ACTUAL accessor return into the observation
        #    ledger (``_record_state_metadata_observation``) -- the producer validates the
        #    user's one real read against the device-defined staged predicate, never a
        #    speculative signature stamp --
        "is_shared": (_STATE_ROUTE_READ_KIND, "is_shared"),
        "is_pinned": (_STATE_ROUTE_READ_KIND, "is_pinned"),
        "is_inference": (_STATE_ROUTE_READ_KIND, "is_inference"),
        # -- base-storage geometry (r65 F3; r67 C6): recorded at the ACTUAL
        #    ``.nbytes()``/``.size()``/``__len__`` accessor call on the storage handle,
        #    never at handle acquisition (corr1-4) --
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
        # -- in-place mutation counter (r67 C4 ruling): the read kind is a
        #    ``refuse_on_any_read`` oracle-policy row -- oracle-1's default
        #    ``load_state_dict`` copy perturbs constructor-owned counters (0 -> 1 plain,
        #    1 -> 2 initialized), so NO captured version is reproducible and EVERY
        #    attributed ``_version`` read refuses the runnable save --
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

_STATE_METADATA_FACTS: weakref.WeakKeyDictionary[Any, dict[str, dict[str, bool]]] = (
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


_HOST_ESCAPE_STATE_METADATA_OBSERVATIONS: weakref.WeakKeyDictionary[
    Any, dict[str, dict[str, bool | None]]
] = weakref.WeakKeyDictionary()
"""Per-trace ledger of the ACTUAL values returned by placement accessor calls on state (r67 C3).

``is_pinned`` / ``is_shared`` are OBSERVED-VALUE read kinds: the honest producer predicate is
"the user's one actual accessor return equals the device-defined staged/oracle value", never a
speculative TorchLens re-read and never an accelerator-initialization inference. The wrapper
calls the original ONCE, then records ``{state name -> {read kind -> observed bool}}`` here;
an accessor that RAISED records ``None`` (unknown -> refuse), and two disagreeing observations
of the same kind collapse to ``None`` (mid-forward placement change -> refuse). Kept weak-keyed
off the schema like its read-kind sibling.
"""


_STATE_METADATA_PLACEMENT_OBSERVED_NAMES = frozenset({"is_shared", "is_pinned"})
"""Accessor names whose STATE reads are OBSERVED-VALUE read kinds (r67 C3): the producer
predicate compares the user's one actual return against the device-defined staged/oracle
value -- never a TorchLens speculative re-read, never an accelerator-initialization
inference (free-F4: the CUDA-init proof-by-absence stamped canonical False on genuinely
pinned XPU/MPS/externally-registered memory)."""


_HOST_ESCAPE_STATE_SOURCE_NAMES: weakref.WeakKeyDictionary[Any, set[str]] = (
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

_HOST_ESCAPE_STATE_METADATA_READS: weakref.WeakKeyDictionary[Any, dict[str, set[str]]] = (
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

_HOST_ESCAPE_UNATTRIBUTABLE_BOOL: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces that observed an UNATTRIBUTABLE (unlabelled-source) BOOL escape.

``bool(self.gate.data > 0.5)`` truth-tests a bool tensor produced on a ``.data`` alias;
the predicate op is orphan-pruned (input-disconnected) and the escaped source carries no
resolvable capture label. NO net (control-witness, conditional, loop, or pruned-RNG) covers
such a pruned non-RNG bool predicate, so its branch source cannot be witnessed. The runnable
producer downgrades witness completeness to keep the model honestly UNVERIFIABLE rather than
falsely VERIFIED, exactly like a pruned-RNG control escape. Membership is presence-only.
"""

_HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces that observed an UNATTRIBUTABLE census-INVISIBLE escape (``.tolist()`` /
``.numpy()`` on an unlabelled tensor, e.g. a ``.data`` alias).

A dispatcher-invisible conversion of a tensor with no resolvable capture label leaves no
source-op slot AND no reliable scalar to value-match (it may be multi-element). Its source
cannot be witnessed, so the runnable producer fails closed (UNVERIFIABLE). Presence-only.
"""


_INPUT_METADATA_VIEW_READ: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces that read a metadata predicate on a DERIVED VIEW of a model input (r29-C1, F5).

``x.t().is_contiguous()`` reads layout metadata on a pure view of an input leaf. The view is
an orphan-pruned intermediate the sparse replay never re-derives, so the read cannot be
re-verified against the runtime input; the producer consults this set to downgrade witness
completeness (UNVERIFIABLE) rather than falsely VERIFY a possibly-wrong replayed arm. Kept in
a weak-keyed module table (not a Trace field) so the fact survives cooking without a scrub
allow-list entry. Presence-only.
"""


_HOST_ESCAPE_LABEL_LEAF_ORIGINS: weakref.WeakKeyDictionary[
    Any, dict[str, tuple[frozenset[str], frozenset[str]] | None]
] = weakref.WeakKeyDictionary()
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


_PRUNED_RNG_CONTROL_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()
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


_ALIAS_MUTATION_CANDIDATE_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = (
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


_PRUNED_ALIAS_MUTATION_LABELS: weakref.WeakKeyDictionary[Any, set[str]] = (
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


_DATA_ALIAS_MUTATION_TRACES: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces containing a successful write through a ``Tensor.data`` alias lineage.

The ``Tensor.data`` getter is captured as a canonical detach op so read-only consumers retain
replay provenance. That graph node must not launder the descriptor's unsafe write semantics:
an in-place receiver reached directly from ``.data`` or through a storage-sharing view remains
an untracked escape surface and ceilings runnable faithfulness to ``unverifiable``.
"""


_HOST_ESCAPE_MUTABLE_WRITEBACK: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces where a host WRITE-BACK through a mutable zero-copy alias was detected.

``y.detach().numpy()[0] = 99`` hands the host a NumPy array that shares ``y``'s storage; the write
mutates ``y``'s bytes with NO aten dispatch and NO version bump, so the sparse replay recomputes
the pre-write value and would falsely VERIFY. The escape observer brackets each ``numpy`` /
``__array__`` call with a byte+version snapshot (see ``_observe_invisible_host_escapes``); a source
whose bytes changed while its version stayed put was host-mutated through the alias and its trace is
recorded here so the runnable producer fails closed (UNVERIFIABLE). A read-only conversion leaves
the bytes unchanged and is never recorded, so it stays honestly VERIFIED. Presence-only.
"""


_HOST_ESCAPE_RAW_POINTER: weakref.WeakSet[Any] = weakref.WeakSet()
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


# --- r43 CLASS 2: non-owner captured-tensor touch (ONE fail-closed rule) --------------------
#
# Locked by design: ANY non-owner thread that TOUCHES A CAPTURED TENSOR during the armed forward
# window permanently ceilings the artifact to UNVERIFIABLE (+ NOT_APPLICABLE). This subsumes
# the whole r41 in-window/foreign 3-class distinction (which let raw ``_thread`` and
# pre-existing workers slip through). Captured membership is decided by
# :func:`_nonowner_touch_is_captured`; the storage-identity catch-all uses the true-original
# accessors captured at import (below), which bypass every torchlens wrapper.

_HOST_ESCAPE_CROSS_THREAD_CAPTURED: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces where a NON-OWNER thread touched a CAPTURED tensor during the armed window (r43).

The single locked-by-design concurrency ceiling: a captured tensor's Python-visible value/pointer/
string/metadata escape (or a positively-known captured-derived alias) observed on any thread
other than the capture owner is outside the single-owner-thread replay model, so the runnable
producer folds it into an INCOMPLETE witness downgrade -> UNVERIFIABLE + NOT_APPLICABLE.
Presence-only. A non-owner thread that never touches a captured tensor records nothing.
"""


_CAPTURED_STORAGE_PTRS: weakref.WeakKeyDictionary[Any, dict[int, tuple[weakref.ref[Any], ...]]] = (
    weakref.WeakKeyDictionary()
)
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
# ``torch._C.TensorBase`` (torch >= 2.2) vs ``torch._C._TensorBase`` (torch 2.1):
# feature-detected through the ONE _torch_compat accessor (HAS_TENSORBASE_CLASS,
# r-b4 R26-2). The witness originals below are structurally REQUIRED, so absence
# raises explicitly -- a module-level ``assert`` here was stripped under
# ``python -O``, leaving ``_TENSORBASE_CLS = None`` and a delayed AttributeError.
_TENSORBASE_CLS = get_tensorbase_class()
if _TENSORBASE_CLS is None:  # pragma: no cover - torch >= 2.1 exposes the C base.
    raise RuntimeError(
        "TorchLens requires torch._C.TensorBase / _TensorBase (torch >= 2.1) for the "
        "completeness witness originals; this torch build exposes neither."
    )
_ORIG_TENSORBASE_UNTYPED_STORAGE = _TENSORBASE_CLS.untyped_storage
_ORIG_UNTYPED_STORAGE_DATA_PTR = torch.UntypedStorage.data_ptr
# r67 C3: the true-original byte-count accessor, for TorchLens's OWN base-geometry reads
# (origin resolution, input nbytes fact) -- bypasses the per-forward storage accessor
# wrappers so internal resolution never recurses into observation.
_ORIG_UNTYPED_STORAGE_NBYTES = torch.UntypedStorage.nbytes


# Authorization roster for the witness's internal-caller check: the code
# objects (held STRONGLY, so their ids can never be reused) of every function
# textually owned by the modules that legitimately call
# ``_raw_storage_ptr_no_observe``. Collected once at import time, BEFORE any
# user code runs, by walking each module's namespace (functions, methods,
# properties, and their nested code constants) and keeping only code compiled
# from that module's own source file. Membership is tested by OBJECT IDENTITY
# (``id``), never by name/path strings or code-object value equality: a frame
# authenticates only when it is executing one of TorchLens's own code objects,
# and neither ``f_globals['__name__']``, ``co_filename``, nor a byte-identical
# recompilation of the source can forge that.
_AUTHORIZED_INTERNAL_CALLER_CODE: list[types.CodeType] = []
_AUTHORIZED_INTERNAL_CALLER_CODE_IDS: set[int] = set()


_ACTIVE_WITNESS_STATE: _WitnessState | None = None
"""The runnable-capture witness state currently installed, or ``None`` (r43).

Published as the LAST step of ``capture_completeness_witness`` armation and cleared on exit so
the wrappers.py string interception can classify owner vs non-owner without threading the state
through the torch-function wrapper. Captures do not nest, so a single slot suffices.
"""


_HOST_ESCAPE_OBSERVER_FAILED: weakref.WeakSet[Any] = weakref.WeakSet()
"""Traces where a REQUIRED tensor->host value observer could not be installed/restored (r39).

The mode-independent method/module observers (:data:`HOST_VALUE_ESCAPE_METHODS` /
:data:`HOST_VALUE_ESCAPE_MODULE_FUNCS`) are the belt that closes the ``_disable_current_modes``
census blind spot. If a required observer cannot be installed or its exact original cannot be
restored, coverage for that forward is unknowable -- so the capture fails closed to INCOMPLETE
rather than silently reporting no escape. Optional/absent targets on a given torch version do
NOT set this (they are classified absent by the version inventory). Presence-only.
"""


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


_DISPATCH_TENSOR_ORIGINS: weakref.WeakKeyDictionary[Any, _TensorOriginRegistry] = (
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


class _CompletenessDispatchMode(_TorchLensDispatchMode):
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

    def _record_plane_p(self, func: Any) -> None:
        """Append one plane-P dispatch record (merge-ranks C2; never raises).

        Records EVERY dispatched op on the owner thread -- any namespace,
        paused (TorchLens-internal) windows included -- as physical evidence
        for the capture-fidelity census: ``(qualified_op,
        owner_func_call_id, paused, discharged_inside_boundary,
        has_module_context)``. The discharge flag implements v5's
        no-double-tick nesting rule below the python layer: a dispatch inside
        an open public collective boundary is that boundary's physical
        evidence, never its own. Module context is resolved lazily and only
        for records with no wrapper owner (the K4 linkage fallback).

        Parameters
        ----------
        func:
            Dispatcher operator overload being observed.
        """

        try:
            paused = not _state._logging_enabled
            from .collectives import _inside_boundary

            discharged = _inside_boundary()
            owner = _active_token()
            owner_id = None if owner is None else owner.func_call_id
            has_module_context = False
            if owner_id is None and not paused and not discharged:
                from ._aten_capture import _module_call_stack

                has_module_context = bool(_module_call_stack(self.state.trace))
            self.state.plane_p_events.append(
                (_operator_name(func), owner_id, paused, discharged, has_module_context)
            )
        except Exception:
            # Observation must never perturb or fail the capture; a dropped
            # plane-P record surfaces as a census accounting miss, never as a
            # capture failure.
            pass

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
        aten_pending: Any = None
        pre_dispatch_receiver_numel: int | None = None
        try:
            if (
                self.state.plane_p
                and threading.get_ident() == self.state.owner_thread_id
                and _state._active_trace is self.state.trace
            ):
                self._record_plane_p(func)
            phase_visible = _state._logging_enabled or (
                self.state.record_aten and self.state.capture_phase == "backward"
            )
            in_scope = (
                threading.get_ident() == self.state.owner_thread_id
                and phase_visible
                and _state._active_trace is self.state.trace
                and _is_aten_operator(func)
            )
            owner = _active_token() if in_scope else None
            if in_scope and self.state.record_aten:
                from ._aten_capture import _prepare_aten_call

                aten_pending = _prepare_aten_call(
                    self.state,
                    func,
                    (args, kwargs or {}),
                    None if owner is None else owner.func_call_id,
                    mutates=_is_mutating_operator(func),
                )
            if in_scope and (self.state.census or self.state.ledger):
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
                    _is_buffer_state_view_dispatch(self.state.trace, func, owner, mutates, args)
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
            if aten_pending is not None:
                from ._aten_capture import _finish_aten_call

                _finish_aten_call(self.state, aten_pending, exception=exc)
            raise
        if event is not None:
            if _dispatch_result_holds_tensor(result):
                event.outcome = "returned_tensor"
                event.output_refs = _dispatch_output_refs(event.owner, result)
                if not event.mutates and _operator_base_name(func) == "aten.as_strided":
                    # Owner-independent: an ``__dlpack__``-wrapper-owned interval is
                    # not a modeled call, so the audited row must still apply.
                    event.contained_view = _as_strided_result_contained(args, result)
            else:
                event.outcome = "returned_host_or_none"
                if not event.in_replacement_hook:
                    event.in_replacement_hook = _in_replacement_hook_frame()
        if aten_pending is not None:
            from ._aten_capture import _finish_aten_call

            _finish_aten_call(self.state, aten_pending, result=result)
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


_RUNNABLE_LEDGER_FACTS: weakref.WeakKeyDictionary[Any, list[dict[str, Any]]] = (
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


# PERF (w15 F1): exact classes whose ``untyped_storage``/``data_ptr`` provably resolve to the
# snapshotted true originals (`_ORIG_TENSORBASE_UNTYPED_STORAGE` / `_ORIG_UNTYPED_STORAGE_DATA_PTR`).
# The per-consumption TOCTOU scans run under ``pause_logging`` on the OWNER thread, where the
# per-forward escape patches are a pure call-through (the recording branch is gated on
# ``_state._logging_enabled``), so for these classes the raw-original read is value- AND
# side-effect-identical to the wrapped spelling. A SUBCLASS may override either accessor at the
# Python level, so anything else keeps the verbatim wrapped per-item path.
_PLAIN_TENSOR_CLS = torch.Tensor
_PLAIN_PARAM_CLS = torch.nn.Parameter


# --- r67 C3/C6: THE atomic storage-accessor disposition table -------------------------------
#
# corr1-4 root cause: handle ACQUISITION stamped ``storage_nbytes`` ("one attribute away"),
# while equivalent reads through the handle (``is_shared()``/``is_pinned()``/``nbytes()`` on
# the storage OBJECT) were unwitnessed. One table now covers the public receiver surface of
# BOTH ``UntypedStorage`` and ``TypedStorage``: acquisition records origin/watch only, and the
# ACTUAL accessor call records the real result -- the ledger claims exactly the observations
# that occurred, no more (no discarded-handle over-trigger) and no less (no storage-spelling
# escape). A reflection immunizer makes a NEW public storage member RED until classified.

_STORAGE_ACCESSOR_NBYTES = "storage_nbytes_read"
"""Disposition: the ACTUAL byte/element-count accessor call records the base-geometry fact
(input-site ``storage_nbytes`` fact / state ``storage_nbytes`` read kind)."""

_STORAGE_ACCESSOR_PLACEMENT = "placement_read"
"""Disposition: ``is_shared``/``is_pinned`` -- records the SAME observed-value read kinds as
the Tensor spelling, attributed via the origin map to the input site or full state alias
group."""

_STORAGE_ACCESSOR_RAW_POINTER = "raw_pointer"
"""Disposition: hands out the raw address (``data_ptr``/``_cdata``) -- the existing r15/r16
fail-closed belt."""

_STORAGE_ACCESSOR_MUTATOR = "captured_mutator"
"""Disposition: mutates the storage bytes/backing in place -- a captured-slot receiver joins
the host-mutation/writeback ceiling."""

_STORAGE_ACCESSOR_VALUE_READ = "value_read"
"""Disposition: copies/exposes the storage's VALUES to the host (indexing, iteration,
``tolist``, dtype/device conversions) -- a state receiver joins the digest witness; an input
receiver fails closed (base-storage bytes outside the view window are not re-bound)."""

_STORAGE_ACCESSOR_FAIL_CLOSED = "fail_closed_read"
"""Disposition: a placement-adjacent fact with NO staged reproduction recipe
(``filename``/``resizable``) -- a captured receiver fails closed (opaque ceiling)."""

_STORAGE_ACCESSOR_ORIGIN_BRIDGE = "origin_bridge"
"""Disposition: returns another handle onto the SAME bytes (``TypedStorage.untyped()``) --
registers the returned handle's origin; records nothing itself."""

_STORAGE_ACCESSOR_INERT = "inert"
"""Disposition: slot-contract-covered or value-free metadata (device/dtype/element size) --
provably reproducible from the recorded slot contract; nothing to record."""

_STORAGE_ACCESSOR_CONTAMINATED_PROPERTY = "contaminated_property_residual"
"""Disposition: a PROPERTY row TorchLens's own output/stack introspection getattr-walks on
every storage object it encounters mid-forward (measured: ``filename`` and ``_cdata`` fire
from ``_walk_output_tensors_with_paths`` / ``_search_stack_for_vars_of_type`` on an ordinary
``self.b.storage()`` call). A wrapper cannot distinguish that framework walk from a user
read (the LOCKED allowlist-by-construction principle forbids a spoofable stack-filename
discriminator), so wrapping would permanently ceiling every model that merely acquires a
storage handle. Exactly like the r31/r65 leaf-only autograd rows, these two properties are
the NAMED documented residual of the storage surface -- classified, never silently omitted."""

_STORAGE_ACCESSOR_STRUCTURAL = "structural"
"""Disposition: constructors / class-level factories producing UNRELATED storages, or
accessors covered by another gate -- nothing to record."""

_STORAGE_WRAPPED_DISPOSITIONS = frozenset(
    {
        _STORAGE_ACCESSOR_NBYTES,
        _STORAGE_ACCESSOR_PLACEMENT,
        _STORAGE_ACCESSOR_MUTATOR,
        _STORAGE_ACCESSOR_VALUE_READ,
        _STORAGE_ACCESSOR_FAIL_CLOSED,
        _STORAGE_ACCESSOR_ORIGIN_BRIDGE,
        _STORAGE_ACCESSOR_RAW_POINTER,
    }
)
"""Dispositions installed as per-forward wrappers (``data_ptr`` keeps its dedicated legacy
wrapper; ``inert``/``structural`` rows are classification-only)."""


def _storage_disposition_rows(class_name: str) -> dict[str, tuple[str, str]]:
    """Build one class's accessor->(disposition, why) rows (shared core + per-class extras)."""

    rows: dict[str, tuple[str, str]] = {}
    conversions = (
        "bfloat16",
        "bool",
        "byte",
        "char",
        "clone",
        "complex_double",
        "complex_float",
        "cpu",
        "cuda",
        "double",
        "float",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "half",
        "hpu",
        "int",
        "long",
        "pin_memory",
        "short",
        "to",
        "tolist",
        "type",
    )
    for name in conversions:
        rows[name] = (
            _STORAGE_ACCESSOR_VALUE_READ,
            "bytes leave tracking as a host-side copy/converted storage",
        )
    for name in ("copy_", "fill_", "resize_", "share_memory_", "__setitem__"):
        rows[name] = (_STORAGE_ACCESSOR_MUTATOR, "in-place byte/backing mutation")
    for name in ("nbytes", "size", "__len__"):
        rows[name] = (_STORAGE_ACCESSOR_NBYTES, "actual base-geometry read")
    rows["is_shared"] = (_STORAGE_ACCESSOR_PLACEMENT, "observed-value read, tensor-spelling parity")
    rows["is_pinned"] = (_STORAGE_ACCESSOR_PLACEMENT, "observed-value read, tensor-spelling parity")
    rows["data_ptr"] = (_STORAGE_ACCESSOR_RAW_POINTER, "raw pointer; dedicated legacy wrapper")
    rows["_cdata"] = (
        _STORAGE_ACCESSOR_CONTAMINATED_PROPERTY,
        "raw cdata address property; TL introspection getattr-walks it (named residual)",
    )
    rows["filename"] = (
        _STORAGE_ACCESSOR_CONTAMINATED_PROPERTY,
        "file backing fact; TL introspection getattr-walks it (named residual)",
    )
    rows["resizable"] = (
        _STORAGE_ACCESSOR_FAIL_CLOSED,
        "backing resizability is not staged-reproducible",
    )
    rows["untyped"] = (_STORAGE_ACCESSOR_ORIGIN_BRIDGE, "another handle onto the same bytes")
    for name in ("__getitem__", "__iter__"):
        rows[name] = (_STORAGE_ACCESSOR_VALUE_READ, "element/byte value exposure")
    for name in ("device", "element_size", "get_device", "is_cuda", "is_hpu"):
        rows[name] = (_STORAGE_ACCESSOR_INERT, "slot contract pins device/dtype")
    rows["is_sparse"] = (
        _STORAGE_ACCESSOR_INERT,
        "False constant on strided; sparse refused at bind/save",
    )
    for name in ("from_buffer", "from_file"):
        rows[name] = (_STORAGE_ACCESSOR_STRUCTURAL, "constructs an unrelated new storage")
    if class_name == "UntypedStorage":
        rows["byteswap"] = (_STORAGE_ACCESSOR_MUTATOR, "in-place byte mutation")
        rows["mps"] = (_STORAGE_ACCESSOR_VALUE_READ, "device-converted copy")
        rows["new"] = (_STORAGE_ACCESSOR_STRUCTURAL, "constructs an unrelated new storage")
        rows["is_sparse_csr"] = (
            _STORAGE_ACCESSOR_INERT,
            "False constant on strided; sparse refused at bind/save",
        )
    if class_name == "TypedStorage":
        rows["pickle_storage_type"] = (
            _STORAGE_ACCESSOR_INERT,
            "dtype-derived legacy type name; slot contract pins dtype",
        )
    return rows


STORAGE_METADATA_ACCESSOR_DISPOSITIONS: Mapping[str, Mapping[str, tuple[str, str]]] = (
    MappingProxyType(
        {
            "UntypedStorage": MappingProxyType(_storage_disposition_rows("UntypedStorage")),
            "TypedStorage": MappingProxyType(_storage_disposition_rows("TypedStorage")),
        }
    )
)
"""ONE authoritative disposition per public storage accessor, per class (r67 C3/C6).

Keys cover the public (non-underscore) ``dir()`` surface of each class on the declared torch
floor->ceiling, plus the NAMED private/dunder rows (``_cdata``, ``__len__``, ``__getitem__``,
``__setitem__``, ``__iter__``). The reflection immunizer asserts every public member has a
row, so a new torch storage accessor is RED until classified -- never a silent gap. Rows
absent on a given torch build (feature-gated members) are skipped at install.
"""


# Import-time roster collection for the witness's internal-caller
# authorization (see ``_caller_frame_is_torchlens_internal``). Runs at the
# BOTTOM of the module so every function above is already defined, and before
# any user code can run a capture. ``buffer_writes`` is the one other module
# whose functions legitimately call ``_raw_storage_ptr_no_observe`` (the
# journaled-write alias prefilter); this module imports from it at the top,
# so its namespace is guaranteed complete here.


# Private implementation slices; public names remain owned by this module.
_in_replacement_hook_frame = _rebind_function(
    _completeness_boundaries._in_replacement_hook_frame, globals()
)
_is_expected_opaque_dispatch = _rebind_function(
    _completeness_boundaries._is_expected_opaque_dispatch, globals()
)
completeness_scope_for_wrapper = _rebind_function(
    _completeness_boundaries.completeness_scope_for_wrapper, globals()
)
record_runnable_input_storage_sites = _rebind_function(
    _completeness_boundaries.record_runnable_input_storage_sites, globals()
)
_classify_input_storage_alias = _rebind_function(
    _completeness_boundaries._classify_input_storage_alias, globals()
)
_input_base_tensor = _rebind_function(_completeness_boundaries._input_base_tensor, globals())
_record_input_metadata_read_at_site = _rebind_function(
    _completeness_boundaries._record_input_metadata_read_at_site, globals()
)
_record_input_metadata_read = _rebind_function(
    _completeness_boundaries._record_input_metadata_read, globals()
)
_observe_input_metadata_read = _rebind_function(
    _completeness_boundaries._observe_input_metadata_read, globals()
)
_observe_input_derived_layout_read = _rebind_function(
    _completeness_metadata._observe_input_derived_layout_read, globals()
)
_resolve_layout_rooting_labels = _rebind_function(
    _completeness_metadata._resolve_layout_rooting_labels, globals()
)
_layout_storage_rooting_labels = _rebind_function(
    _completeness_metadata._layout_storage_rooting_labels, globals()
)
record_storage_rebind_barrier = _rebind_function(
    _completeness_metadata.record_storage_rebind_barrier, globals()
)
storage_rebind_barrier_labels = _rebind_function(
    _completeness_metadata.storage_rebind_barrier_labels, globals()
)
_layout_ancestry_tainted = _rebind_function(
    _completeness_metadata._layout_ancestry_tainted, globals()
)
_state_derived_addresses = _rebind_function(
    _completeness_metadata._state_derived_addresses, globals()
)
_observe_state_metadata_read = _rebind_function(
    _completeness_metadata._observe_state_metadata_read, globals()
)
host_escape_state_metadata_reads = _rebind_function(
    _completeness_metadata.host_escape_state_metadata_reads, globals()
)
_state_direct_address = _rebind_function(_completeness_metadata._state_direct_address, globals())
_observe_state_metadata_read_direct = _rebind_function(
    _completeness_metadata._observe_state_metadata_read_direct, globals()
)
_expand_state_alias_addresses = _rebind_function(
    _completeness_metadata._expand_state_alias_addresses, globals()
)
_record_state_metadata_read = _rebind_function(
    _completeness_metadata._record_state_metadata_read, globals()
)
_record_state_metadata_observation = _rebind_function(
    _completeness_metadata._record_state_metadata_observation, globals()
)
host_escape_state_metadata_observations = _rebind_function(
    _completeness_metadata.host_escape_state_metadata_observations, globals()
)
_observe_state_placement_read = _rebind_function(
    _completeness_metadata._observe_state_placement_read, globals()
)
_placement_read_witnessed = _rebind_function(
    _completeness_metadata._placement_read_witnessed, globals()
)
_discharge_placement_dispatch = _rebind_function(
    _completeness_metadata._discharge_placement_dispatch, globals()
)
_observe_state_metadata_fact = _rebind_function(
    _completeness_metadata._observe_state_metadata_fact, globals()
)
host_escape_state_metadata_facts = _rebind_function(
    _completeness_metadata.host_escape_state_metadata_facts, globals()
)
_observe_state_property_read = _rebind_function(
    _completeness_metadata._observe_state_property_read, globals()
)
_tensor_receiver_origin = _rebind_function(
    _completeness_metadata._tensor_receiver_origin, globals()
)
_register_storage_handle_origin = _rebind_function(
    _completeness_metadata._register_storage_handle_origin, globals()
)
_register_storage_origin = _rebind_function(
    _completeness_metadata._register_storage_origin, globals()
)
_lazy_storage_state_ptr_names = _rebind_function(
    _completeness_metadata._lazy_storage_state_ptr_names, globals()
)
_resolve_storage_origin = _rebind_function(
    _completeness_metadata._resolve_storage_origin, globals()
)
_record_input_storage_nbytes = _rebind_function(
    _completeness_metadata._record_input_storage_nbytes, globals()
)
input_metadata_view_read = _rebind_function(
    _completeness_escape_state.input_metadata_view_read, globals()
)
host_escape_source_labels = _rebind_function(
    _completeness_escape_state.host_escape_source_labels, globals()
)
host_escape_state_source_names = _rebind_function(
    _completeness_escape_state.host_escape_state_source_names, globals()
)
host_escape_has_unattributable_bool = _rebind_function(
    _completeness_escape_state.host_escape_has_unattributable_bool, globals()
)
host_escape_has_unattributable_opaque = _rebind_function(
    _completeness_escape_state.host_escape_has_unattributable_opaque, globals()
)
host_escape_state_source_labels = _rebind_function(
    _completeness_escape_state.host_escape_state_source_labels, globals()
)
host_escape_bool_source_labels = _rebind_function(
    _completeness_escape_state.host_escape_bool_source_labels, globals()
)
host_escape_bool_consumer_locations = _rebind_function(
    _completeness_escape_state.host_escape_bool_consumer_locations, globals()
)
host_escape_label_leaf_origins = _rebind_function(
    _completeness_escape_state.host_escape_label_leaf_origins, globals()
)
_record_escape_label_fallback = _rebind_function(
    _completeness_escape_state._record_escape_label_fallback, globals()
)
pruned_rng_control_source_labels = _rebind_function(
    _completeness_escape_state.pruned_rng_control_source_labels, globals()
)
record_pruned_rng_control_source = _rebind_function(
    _completeness_escape_state.record_pruned_rng_control_source, globals()
)
alias_mutation_candidate_labels = _rebind_function(
    _completeness_escape_state.alias_mutation_candidate_labels, globals()
)
record_alias_mutation_candidate = _rebind_function(
    _completeness_escape_state.record_alias_mutation_candidate, globals()
)
pruned_alias_mutation_source_labels = _rebind_function(
    _completeness_escape_state.pruned_alias_mutation_source_labels, globals()
)
record_pruned_alias_mutation_source = _rebind_function(
    _completeness_escape_state.record_pruned_alias_mutation_source, globals()
)
data_alias_mutation_detected = _rebind_function(
    _completeness_escape_state.data_alias_mutation_detected, globals()
)
record_data_alias_mutation = _rebind_function(
    _completeness_escape_state.record_data_alias_mutation, globals()
)
host_escape_has_mutable_writeback = _rebind_function(
    _completeness_escape_state.host_escape_has_mutable_writeback, globals()
)
host_escape_has_raw_pointer = _rebind_function(
    _completeness_escape_state.host_escape_has_raw_pointer, globals()
)
host_escape_has_cross_thread_captured_tensor = _rebind_function(
    _completeness_escape_state.host_escape_has_cross_thread_captured_tensor, globals()
)
_register_authorized_caller_namespace = _rebind_function(
    _completeness_escape_state._register_authorized_caller_namespace, globals()
)
_caller_frame_is_torchlens_internal = _rebind_function(
    _completeness_escape_state._caller_frame_is_torchlens_internal, globals()
)
_raw_storage_ptr_no_observe = _rebind_function(
    _completeness_escape_state._raw_storage_ptr_no_observe, globals()
)
_nonowner_ptr_is_captured = _rebind_function(
    _completeness_cross_thread._nonowner_ptr_is_captured, globals()
)
_nonowner_touch_is_captured = _rebind_function(
    _completeness_cross_thread._nonowner_touch_is_captured, globals()
)
_nonowner_escape_observe = _rebind_function(
    _completeness_cross_thread._nonowner_escape_observe, globals()
)
observe_nonowner_operands = _rebind_function(
    _completeness_cross_thread.observe_nonowner_operands, globals()
)
_torch_ops_call_classes = _rebind_function(
    _completeness_cross_thread._torch_ops_call_classes, globals()
)
_make_nonowner_ops_call = _rebind_function(
    _completeness_cross_thread._make_nonowner_ops_call, globals()
)
_private_c_forward_op_modules = _rebind_function(
    _completeness_cross_thread._private_c_forward_op_modules, globals()
)
_private_c_module_callables = _rebind_function(
    _completeness_cross_thread._private_c_module_callables, globals()
)
_make_nonowner_private_c_callable = _rebind_function(
    _completeness_cross_thread._make_nonowner_private_c_callable, globals()
)
string_escape_is_owner_thread = _rebind_function(
    _completeness_cross_thread.string_escape_is_owner_thread, globals()
)
host_escape_observer_install_failed = _rebind_function(
    _completeness_cross_thread.host_escape_observer_install_failed, globals()
)
record_host_string_escape_source = _rebind_function(
    _completeness_cross_thread.record_host_string_escape_source, globals()
)
audit_disable_current_modes_sites = _rebind_function(
    _completeness_cross_thread.audit_disable_current_modes_sites, globals()
)
_internal_read_active = _rebind_function(
    _completeness_cross_thread._internal_read_active, globals()
)
internal_scalar_read = _rebind_contextmanager(_completeness_origins.internal_scalar_read, globals())
_escape_source_is_torchlens_internal = _rebind_function(
    _completeness_origins._escape_source_is_torchlens_internal, globals()
)
_output_is_host_value = _rebind_function(_completeness_origins._output_is_host_value, globals())
_iter_tensor_operands = _rebind_function(_completeness_origins._iter_tensor_operands, globals())
_record_host_escape_source = _rebind_function(
    _completeness_origins._record_host_escape_source, globals()
)
_escape_storage_ptr = _rebind_function(_completeness_origins._escape_storage_ptr, globals())
_param_derived_addresses = _rebind_function(
    _completeness_origins._param_derived_addresses, globals()
)
_iter_tensors_deep = _rebind_function(_completeness_origins._iter_tensors_deep, globals())
_operand_origins = _rebind_function(_completeness_origins._operand_origins, globals())
_operand_leaf_origins = _rebind_function(_completeness_origins._operand_leaf_origins, globals())
_operator_is_seeded_rng = _rebind_function(_completeness_origins._operator_is_seeded_rng, globals())
_operator_uninit_family_tail = _rebind_function(
    _completeness_origins._operator_uninit_family_tail, globals()
)
_python_tensor_method_uninit_family_tail = _rebind_function(
    _completeness_origins._python_tensor_method_uninit_family_tail, globals()
)
_operator_is_growth_resize = _rebind_function(
    _completeness_origins._operator_is_growth_resize, globals()
)
_operator_total_writer_destination = _rebind_function(
    _completeness_origins._operator_total_writer_destination, globals()
)
_live_deterministic_fill_governs = _rebind_function(
    _completeness_origins._live_deterministic_fill_governs, globals()
)
_register_dispatch_result_origins = _rebind_function(
    _completeness_origins._register_dispatch_result_origins, globals()
)
_resolved_dispatch_origins = _rebind_function(
    _completeness_dispatch_names._resolved_dispatch_origins, globals()
)
_record_escape_source_tensor = _rebind_function(
    _completeness_dispatch_names._record_escape_source_tensor, globals()
)
_operator_name = _rebind_function(_completeness_dispatch_names._operator_name, globals())
_operator_base_name = _rebind_function(_completeness_dispatch_names._operator_base_name, globals())
_is_aten_operator = _rebind_function(_completeness_dispatch_names._is_aten_operator, globals())
_is_mutating_operator = _rebind_function(
    _completeness_dispatch_names._is_mutating_operator, globals()
)
_is_buffer_state_view_dispatch = _rebind_function(
    _completeness_dispatch_names._is_buffer_state_view_dispatch, globals()
)
_dispatch_callsite = _rebind_function(_completeness_dispatch_names._dispatch_callsite, globals())
record_uncaptured_owner_callsite = _rebind_function(
    _completeness_dispatch.record_uncaptured_owner_callsite, globals()
)
_dispatch_result_holds_tensor = _rebind_function(
    _completeness_dispatch._dispatch_result_holds_tensor, globals()
)
_dispatch_output_refs = _rebind_function(_completeness_dispatch._dispatch_output_refs, globals())
_event_builds_boundary_output = _rebind_function(
    _completeness_dispatch._event_builds_boundary_output, globals()
)
_event_is_capture_accounted = _rebind_function(
    _completeness_dispatch._event_is_capture_accounted, globals()
)
runnable_ledger_facts = _rebind_function(_completeness_dispatch.runnable_ledger_facts, globals())
_tensor_abs_byte_span = _rebind_function(_completeness_dispatch._tensor_abs_byte_span, globals())
_as_strided_result_contained = _rebind_function(
    _completeness_dispatch._as_strided_result_contained, globals()
)
_finalize_runnable_ledger = _rebind_function(
    _completeness_dispatch._finalize_runnable_ledger, globals()
)
_whole_storage_uint8 = _rebind_function(_completeness_dispatch._whole_storage_uint8, globals())
_snapshot_writeback_source = _rebind_function(
    _completeness_dispatch._snapshot_writeback_source, globals()
)
_iter_dispatch_tensors = _rebind_function(_completeness_dispatch._iter_dispatch_tensors, globals())
_sample_writeback_at_consumption = _rebind_function(
    _completeness_dispatch._sample_writeback_at_consumption, globals()
)
_has_state_toctou_watch = _rebind_function(
    _completeness_dispatch._has_state_toctou_watch, globals()
)
_sample_state_toctou_at_consumption = _rebind_function(
    _completeness_dispatch._sample_state_toctou_at_consumption, globals()
)
_split_consumed_state_items = _rebind_function(
    _completeness_dispatch._split_consumed_state_items, globals()
)
_sample_param_toctou_at_consumption = _rebind_function(
    _completeness_dispatch._sample_param_toctou_at_consumption, globals()
)
_param_baseline_differs = _rebind_function(
    _completeness_dispatch._param_baseline_differs, globals()
)
_sample_buffer_toctou_at_consumption = _rebind_function(
    _completeness_dispatch._sample_buffer_toctou_at_consumption, globals()
)
_buffer_expected_differs = _rebind_function(
    _completeness_dispatch._buffer_expected_differs, globals()
)
_make_invisible_escape_wrapper = _rebind_function(
    _completeness_dispatch._make_invisible_escape_wrapper, globals()
)
_nonowner_storage_observe = _rebind_function(
    _completeness_storage._nonowner_storage_observe, globals()
)
_record_state_value_escape = _rebind_function(
    _completeness_storage._record_state_value_escape, globals()
)
_attribute_storage_placement = _rebind_function(
    _completeness_storage._attribute_storage_placement, globals()
)
_make_storage_metadata_wrapper = _rebind_function(
    _completeness_storage._make_storage_metadata_wrapper, globals()
)
_make_storage_property_wrapper = _rebind_function(
    _completeness_storage._make_storage_property_wrapper, globals()
)
_make_storage_raw_pointer_wrapper = _rebind_function(
    _completeness_storage._make_storage_raw_pointer_wrapper, globals()
)
_completeness_census_active = _rebind_function(
    _completeness_storage._completeness_census_active, globals()
)
_make_host_value_escape_method = _rebind_function(
    _completeness_storage._make_host_value_escape_method, globals()
)
_first_scalar_escape_source = _rebind_function(
    _completeness_storage._first_scalar_escape_source, globals()
)
_record_bool_consumer_location = _rebind_function(
    _completeness_storage._record_bool_consumer_location, globals()
)
_make_plain_scalar_escape_method = _rebind_function(
    _completeness_storage._make_plain_scalar_escape_method, globals()
)
_external_warning_stacklevel = _rebind_function(
    _completeness_storage._external_warning_stacklevel, globals()
)
capture_scalar_escape_warning = _rebind_contextmanager(
    _completeness_storage.capture_scalar_escape_warning, globals()
)
_make_host_value_predicate_module_wrapper = _rebind_function(
    _completeness_storage._make_host_value_predicate_module_wrapper, globals()
)
_make_module_escape_wrapper = _rebind_function(
    _completeness_storage._make_module_escape_wrapper, globals()
)
_make_invisible_escape_property = _rebind_function(
    _completeness_storage._make_invisible_escape_property, globals()
)
_make_input_metadata_wrapper = _rebind_function(
    _completeness_patches._make_input_metadata_wrapper, globals()
)
_make_input_metadata_bool_method = _rebind_function(
    _completeness_patches._make_input_metadata_bool_method, globals()
)
_make_input_metadata_grad_property = _rebind_function(
    _completeness_patches._make_input_metadata_grad_property, globals()
)
_observe_invisible_host_escapes = _rebind_contextmanager(
    _completeness_patches._observe_invisible_host_escapes, globals()
)
_STORAGE_RAW_POINTER_TARGETS = _rebind_function(
    _completeness_finalize._STORAGE_RAW_POINTER_TARGETS, globals()
)
_MODULE_ESCAPE_TARGETS = _rebind_function(_completeness_finalize._MODULE_ESCAPE_TARGETS, globals())
_check_writeback_watch = _rebind_function(_completeness_finalize._check_writeback_watch, globals())
_effective_mode = _rebind_function(_completeness_finalize._effective_mode, globals())
_barcode_text = _rebind_function(_completeness_finalize._barcode_text, globals())
_finalize_census = _rebind_function(_completeness_finalize._finalize_census, globals())
_reports_include_non_input_boundary = _rebind_function(
    _completeness_finalize._reports_include_non_input_boundary, globals()
)
_finalize_input_semantics_without_census = _rebind_function(
    _completeness_finalize._finalize_input_semantics_without_census, globals()
)
_init_witness_fields = _rebind_function(_completeness_finalize._init_witness_fields, globals())
_build_witness_state = _rebind_function(_completeness_finalize._build_witness_state, globals())
_plane_p_requested = _rebind_function(_completeness_finalize._plane_p_requested, globals())
_finalize_plane_p = _rebind_function(_completeness_finalize._finalize_plane_p, globals())
capture_completeness_witness = _rebind_contextmanager(
    _completeness_finalize.capture_completeness_witness, globals()
)
_collect_authorized_internal_caller_modules = _rebind_function(
    _completeness_finalize._collect_authorized_internal_caller_modules, globals()
)

_split_namespace = {name: value for name, value in globals().items() if not name.startswith("__")}
for _split_module in (
    _completeness_boundaries,
    _completeness_metadata,
    _completeness_escape_state,
    _completeness_cross_thread,
    _completeness_origins,
    _completeness_dispatch_names,
    _completeness_dispatch,
    _completeness_storage,
    _completeness_patches,
    _completeness_finalize,
):
    _split_module.__dict__.update(_split_namespace)

_collect_authorized_internal_caller_modules()

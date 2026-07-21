"""RNG and autocast state capture/restore for reproducible forward-pass replay.

During the exhaustive logging pass, RNG states are captured *before* each
logged operation so that the validation replay can restore the exact same
random state and reproduce the operation's output.  This is critical for
ops like ``dropout`` or ``torch.randn`` that consume RNG.

**Ordering invariant**: RNG states must be captured *before*
``active_logging()`` is entered, because entering the logging context
itself may call decorated functions (e.g. tensor allocations for internal
bookkeeping) that would advance the RNG.

Three independent RNG engines are captured:
  - Python's ``random`` module
  - NumPy's ``np.random``
  - PyTorch's CPU generator (``torch.random``)
  - PyTorch's CUDA generator (if CUDA is available)

Autocast state (``torch.amp.autocast``) is captured similarly so that
mixed-precision ops can be replayed under the same dtype context.
"""

import datetime as _datetime_module
import dis as _dis_module
import gc as _gc_module
import os as _os_module
import random
import sys as _sys_module
import threading as _threading_module
import time as _time_module
import weakref as _weakref_module
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass
from types import BuiltinFunctionType, CodeType, FrameType, ModuleType, TracebackType
from typing import Any, Dict, List, TypeVar, cast

import _random as _c_random_module

import numpy as np
import torch

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]

from ._torch_compat import autocast_get_dtype, autocast_is_enabled
from .hashing import seed_barcode_rng
from .tensor_utils import _is_cuda_available

_AUTOCAST_DEVICES = ("cpu", "cuda")
_T = TypeVar("_T")

_SEEDED_RNG_NAMESPACES = frozenset({"torch", "torch.Tensor", "torch.nn.functional"})


def aten_qualname_is_seeded_rng(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable maps to a seeded ATen RNG operator.

    A "seeded" RNG operator is any ATen overload PyTorch itself tags with
    ``torch.Tag.nondeterministic_seeded`` (``rand``/``randn``/``randint``/
    ``bernoulli``/``multinomial``/``dropout`` families and their in-place
    ``*_`` spellings). Feature-detecting the maintained tag -- rather than
    hard-coding a name list -- keeps this robust across torch versions.

    Parameters
    ----------
    namespace:
        Captured callable namespace (e.g. ``"torch"``, ``"torch.Tensor"``).
    qualname:
        Captured callable qualified name (e.g. ``"rand"``, ``"Tensor.bernoulli_"``).

    Returns
    -------
    bool
        Whether any matching ATen overload carries ``nondeterministic_seeded``.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    name = qualname.rsplit(".", 1)[-1]
    candidate_names = (name, name[:-1]) if name.endswith("_") else (name,)
    for candidate_name in candidate_names:
        packet = getattr(torch.ops.aten, candidate_name, None)
        overloads = getattr(packet, "overloads", None)
        if not callable(overloads):
            continue
        for overload_name in overloads():
            overload = getattr(packet, overload_name)
            if torch.Tag.nondeterministic_seeded in getattr(overload, "tags", ()):
                return True
    return False


# --- r53 hon_2: uninitialized-memory value-source family (ONE closed table) --------
#
# The ``empty`` factory family and a GROWING ``resize_``/``resize_as_`` produce bytes
# that are not a function of the recorded computation (allocator garbage). PyTorch
# gives these ops NO distinguishing ``torch.Tag`` (verified: ``empty`` tags are only
# ``core``/``generated``/``pt2_compliant_tag``), so the family is a maintained name
# table defended by an aten-namespace drift meta-test
# (``tests/test_tlspec_runnable_r53_uninit_alloc.py``): a new ``empty*``/``resize*``
# aten name that is neither in the family table nor in the test's justified
# non-family allowlist is a FAILING test, never a silent gap.
#
# This block is the SINGLE definition consumed by all three recognition layers:
# the load-side value-source classifier (``_runnable_execution.py``), the producer
# origin ledger (``backends/torch/completeness_witness.py``), and the pruned-orphan
# control walk (``postprocess/graph_traversal.py``). No other call site may
# re-derive uninitialized-memory nondeterminism from qualnames (source-scan
# meta-test).

_UNINIT_ALLOC_FACTORY_TAILS = frozenset(
    {
        "empty",
        "empty_like",
        "empty_permuted",
        "empty_strided",
        "empty_quantized",
        "new_empty",
        "new_empty_strided",
        # Private quantized spellings: never captured through the public wrap
        # surface, tabled so the aten drift meta-test stays exhaustive.
        "_empty_affine_quantized",
        "_empty_per_channel_affine_quantized",
    }
)
"""Factory ops whose freshly allocated bytes are uninitialized."""

_UNINIT_ALLOC_SIZE_GATED_TAILS = frozenset({"new"})
"""Python-level ``torch.Tensor`` allocators whose UNINIT semantics depend on ARG FORM.

r55 hon_1: the legacy ``Tensor.new(*sizes)`` allocator returns byte-identical
uninitialized memory to ``new_empty`` (probed: distinct ``.sum()`` across
allocations; redispatches ``aten.empty.memory_format``), but the SAME name
called with DATA (``Tensor.new([values])`` / ``Tensor.new(tensor)``) is a
deterministic copy constructor (redispatches ``aten.lift_fresh``/``aten.alias``).
``new`` has NO aten spelling (``hasattr(torch.ops.aten, "new")`` is ``False``), so
the aten drift meta-test cannot see it; the Python-``torch.Tensor``-method drift
meta-test (``tests/test_tlspec_runnable_r53_uninit_alloc.py``) defends this table
instead. Membership here is NECESSARY but not SUFFICIENT for uninit taint: every
consumer must additionally gate on :func:`uninit_new_call_is_size_form` (the
size-vs-data argument-form predicate) so the data spelling is never over-ceilinged.
"""

_UNINIT_ALLOC_RESIZE_TAILS = frozenset({"resize_", "resize_as_", "resize", "resize_as"})
"""Resize spellings that expose stale allocator bytes ONLY when they GROW.

``resize_``/``resize_as_`` preserve the element prefix (probed: shrink/same-size
is byte-deterministic) but a grow beyond the pre-call element count exposes
uninitialized tail bytes (probed: 4088/4092 stale bytes recovered through a
4096-element grow). The non-underscore spellings are the deprecated
``Tensor.resize``/``resize_as`` aliases and the functionalized aten variants,
which share the grow semantics. The GROW gate is decided by the caller (shapes
are layer-local facts); an undecidable grow fails closed to tainted.
"""

_UNINIT_TOTAL_WRITER_TAILS = frozenset({"copy_", "zero_", "fill_"})
"""In-place ops whose result bytes are a TOTAL write independent of prior content."""

_UNINIT_RNG_FILL_TAILS = frozenset(
    {
        "uniform_",
        "normal_",
        "bernoulli_",
        "random_",
        "exponential_",
        "geometric_",
        "cauchy_",
        "log_normal_",
    }
)
"""In-place RNG fills: total writes that REPLACE uninit taint with the RNG source
classification (they are ``nondeterministic_seeded``-tagged, so the seeded nets
own their products)."""


def qualname_is_uninitialized_alloc(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable is an uninitialized-memory FACTORY op.

    Parameters
    ----------
    namespace:
        Captured callable namespace (e.g. ``"torch"``, ``"torch.Tensor"``).
    qualname:
        Captured callable qualified name (e.g. ``"empty_like"``,
        ``"Tensor.new_empty"``).

    Returns
    -------
    bool
        Whether the callable belongs to the ``empty`` factory family. Growing
        resizes are matched separately by :func:`qualname_is_uninit_growth_resize`
        because their taint additionally requires the grow refinement.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    return qualname.rsplit(".", 1)[-1] in _UNINIT_ALLOC_FACTORY_TAILS


def qualname_is_uninit_size_gated_alloc(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable is an ARG-FORM-GATED uninit allocator.

    r55 hon_1: matches the ``Tensor.new`` legacy allocator family
    (:data:`_UNINIT_ALLOC_SIZE_GATED_TAILS`). A ``True`` here means the call is
    uninitialized-memory-producing ONLY in its size-argument form; the caller
    OWNS the form refinement via :func:`uninit_new_call_is_size_form` and must
    fail closed to tainted on an undecidable form (the grow-gate precedent).
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    return qualname.rsplit(".", 1)[-1] in _UNINIT_ALLOC_SIZE_GATED_TAILS


def uninit_new_call_is_size_form(args: Sequence[Any]) -> bool | None:
    """Classify a ``Tensor.new(...)`` positional-argument tuple as SIZE vs DATA form.

    Probed legacy-constructor semantics (the size form allocates UNINITIALIZED
    memory; the data form is a deterministic copy):

    - ``new()`` / ``new(int...)`` / ``new(np.integer...)`` / ``new(torch.Size)``
      -> SIZE form (``aten.empty.memory_format``): returns ``True``.
    - ``new(list)`` / ``new(tensor)`` / ``new(ndarray)`` / ``new(range)``
      -> DATA form (``aten.lift_fresh`` / ``aten.alias``): returns ``False``.
    - a single plain ``tuple`` of ints -> ``None`` (UNDECIDABLE): a LIVE plain
      tuple is the data form (probed), but the portable literal grammar erases
      ``torch.Size`` to a plain tuple, so a DECODED int-tuple may have been a
      capture-time SIZE call. A caller holding live runtime types sees
      ``torch.Size`` classified ``True`` before this case can fire; a caller
      holding decoded literals must fail closed to tainted on ``None``.
    - any other spelling -> ``None`` (fail closed to tainted; an invalid
      spelling raises at execution time and never produces a value to classify).

    ``bool`` is NOT an integral size argument (``Tensor.new(True)`` raises,
    probed), so ``True``/``False`` literals fall through to ``None``.
    """

    positional = tuple(args)
    if not positional:
        return True  # zero-size uninit alloc; the zero-numel refinement is downstream
    if all(
        (isinstance(item, int) and not isinstance(item, bool)) or isinstance(item, np.integer)
        for item in positional
    ):
        return True
    if len(positional) == 1:
        head = positional[0]
        if isinstance(head, torch.Size):
            return True
        if isinstance(head, (torch.Tensor, np.ndarray, list, range)):
            return False
        if isinstance(head, tuple):
            return None  # torch.Size erased by the portable grammar: undecidable
    return None


def qualname_is_uninit_growth_resize(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable is a resize op that CAN expose stale bytes.

    The caller owns the grow refinement (``new numel > pre-call numel``); a
    matching name with an undecidable grow fact must fail closed to tainted.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    return qualname.rsplit(".", 1)[-1] in _UNINIT_ALLOC_RESIZE_TAILS


def qualname_is_uninit_total_writer(namespace: str | None, qualname: str | None) -> bool:
    """Return whether a captured callable totally overwrites its destination's bytes.

    Total writers (``copy_``/``zero_``/``fill_`` and the in-place RNG fills)
    REMOVE prior uninitialized-memory taint from their destination: the
    post-call value is independent of the destination's prior content. Partial
    or unprovable in-place writers (``index_put_``, ``masked_fill_``,
    ``scatter_``, a sliced ``copy_`` through a view's base) are deliberately
    NOT in this table -- unprovable coverage propagates taint (fail closed).
    The namespace gate keeps the sanitizer NARROW: a custom callable that
    merely shares a total-writer name never removes taint.
    """

    if namespace not in _SEEDED_RNG_NAMESPACES or not qualname:
        return False
    tail = qualname.rsplit(".", 1)[-1]
    return tail in _UNINIT_TOTAL_WRITER_TAILS or tail in _UNINIT_RNG_FILL_TAILS


def deterministic_fill_governs(
    deterministic_algorithms: bool | None,
    fill_uninitialized_memory: bool | None,
) -> bool:
    """Return whether a governing context proves deterministic uninit-memory fill.

    Under ``torch.use_deterministic_algorithms(True)`` with
    ``torch.utils.deterministic.fill_uninitialized_memory`` not ``False``
    (default ``True``; ``None`` means the runtime predates the disable knob and
    always fills under deterministic mode), torch deterministically fills the
    ``empty`` family (NaN for floating point, the max value for int dtypes;
    probed), so the family's bytes ARE a function of the recorded computation
    and carry no taint.
    """

    return deterministic_algorithms is True and fill_uninitialized_memory is not False


def set_random_seed(seed: int) -> None:
    """Set the random seed for all RNG engines simultaneously.

    Ensures deterministic behavior across Python, NumPy, and PyTorch
    (CPU + all CUDA devices).

    Parameters
    ----------
    seed:
        Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep torchlens's private barcode RNG in lockstep with the seed so a fixed
    # capture seed yields reproducible tensor barcodes (a fork replay reuses the
    # original seed; matching barcodes keep tensor/op/param cross-references
    # consistent). The barcode RNG stays a separate stream, so this does not
    # perturb the user's global ``random`` state that host-RNG honesty brackets.
    seed_barcode_rng(seed)


def execute_with_restored_rng_autocast(
    func: Callable[..., _T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rng_states: Dict[str, Any] | None,
    autocast_state: Dict[str, Any] | None,
) -> _T:
    """Execute a callable with saved RNG and autocast state in a tight scope.

    Parameters
    ----------
    func:
        Callable to execute.
    args:
        Positional arguments for ``func``.
    kwargs:
        Keyword arguments for ``func``.
    rng_states:
        RNG states captured before the original operation. ``None`` or an empty
        dict leaves the current RNG state untouched until final restoration.
    autocast_state:
        Autocast state captured before the original operation.

    Returns
    -------
    _T
        Return value from ``func``.

    Raises
    ------
    Exception
        Re-raises any exception from ``func`` after restoring caller RNG state.
    """

    current_rng_states = log_current_rng_states()
    if rng_states:
        set_rng_from_saved_states(rng_states)
    try:
        with AutocastRestore(autocast_state or {}):
            return func(*args, **kwargs)
    finally:
        set_rng_from_saved_states(current_rng_states)


def snapshot_host_rng() -> tuple[Any, Any]:
    """Snapshot the Python ``random`` and NumPy RNG states without advancing them.

    Reading ``random.getstate()`` / ``np.random.get_state()`` is side-effect free,
    so this can bracket a forward pass to detect whether user code consumed host
    (non-torch) RNG -- the signal a sparse runnable path uses to stay honest about
    Python/NumPy control-flow branches it cannot re-observe.

    Returns
    -------
    tuple[Any, Any]
        ``(python_random_state, numpy_random_state)`` snapshot pair.
    """
    return (random.getstate(), np.random.get_state())


def host_rng_advanced(before: tuple[Any, Any], after: tuple[Any, Any]) -> bool:
    """Return whether a host (Python/NumPy) RNG engine advanced between snapshots.

    Parameters
    ----------
    before:
        Snapshot from :func:`snapshot_host_rng` taken before the observed region.
    after:
        Snapshot from :func:`snapshot_host_rng` taken after the observed region.

    Returns
    -------
    bool
        ``True`` when either the Python ``random`` or NumPy engine state changed.
    """
    py_before, np_before = before
    py_after, np_after = after
    if py_before != py_after:
        return True
    return not _numpy_states_equal(np_before, np_after)


def restore_host_rng(snapshot: tuple[Any, Any]) -> None:
    """Restore Python ``random`` and NumPy engines from a :func:`snapshot_host_rng` pair."""
    py_state, np_state = snapshot
    random.setstate(py_state)
    np.random.set_state(np_state)


def _numpy_states_equal(a: Any, b: Any) -> bool:
    """Compare two ``np.random.get_state()`` tuples (array-aware) for equality."""
    try:
        if a[0] != b[0] or not np.array_equal(a[1], b[1]):
            return False
        return tuple(a[2:]) == tuple(b[2:])
    except (TypeError, IndexError, ValueError):
        return a is b


def log_current_rng_states(torch_only: bool = False) -> Dict[str, Any]:
    """Snapshot the current state of all RNG engines.

    The returned dict can be passed to :func:`set_rng_from_saved_states`
    to restore the exact same RNG position later (e.g. during validation
    replay).

    Parameters
    ----------
    torch_only:
        If True, only capture PyTorch RNG state (skip Python ``random`` and
        NumPy). This is faster and sufficient for most torch operations
        (dropout, randn, etc.).

    Returns
    -------
    dict[str, Any]
        Dict with keys ``"random"``, ``"np"``, ``"torch"``, and optionally
        ``"torch_cuda_all"``, each holding the opaque state object for that
        engine. ``"torch_cuda"`` is also populated for backward compatibility
        with older single-device snapshots.
    """
    rng_dict: Dict[str, Any] = {"torch": torch.random.get_rng_state()}
    if not torch_only:
        rng_dict["random"] = random.getstate()
        rng_dict["np"] = np.random.get_state()
    if _is_cuda_available():
        cuda_states = torch.cuda.get_rng_state_all()
        rng_dict["torch_cuda_all"] = cuda_states
        if cuda_states:
            rng_dict["torch_cuda"] = cuda_states[0]
    return rng_dict


def set_rng_from_saved_states(rng_states: Dict[str, Any]) -> None:
    """Restore RNG engines to a previously captured state.

    Parameters
    ----------
    rng_states:
        Dict produced by :func:`log_current_rng_states`. If empty (RNG capture
        was disabled), this is a no-op.
    """
    if not rng_states:
        return
    if "random" in rng_states:
        random.setstate(rng_states["random"])
    if "np" in rng_states:
        np.random.set_state(rng_states["np"])
    torch.random.set_rng_state(rng_states["torch"])
    if _is_cuda_available() and "torch_cuda_all" in rng_states:
        torch.cuda.set_rng_state_all(rng_states["torch_cuda_all"])
    elif _is_cuda_available() and "torch_cuda" in rng_states:
        torch.cuda.set_rng_state(rng_states["torch_cuda"], "cuda")


def log_current_autocast_state() -> dict[str, dict[str, Any]]:
    """Capture the current ``torch.amp.autocast`` enabled/dtype state.

    Checked for each device in :data:`_AUTOCAST_DEVICES`.  If a device
    doesn't support autocast queries, it is silently skipped.

    Returns:
        Dict mapping device name to ``{"enabled": bool, "dtype": torch.dtype}``.
    """
    state: dict[str, dict[str, Any]] = {}
    # r35 corr2_8: record the grad/inference execution mode alongside autocast in the
    # same per-op KEEP field, under a reserved non-device key. ``enabled`` stays False
    # so every autocast consumer (AutocastRestore) skips it structurally.
    state["__execution__"] = {
        "enabled": False,
        "dtype": None,
        "grad_enabled": bool(torch.is_grad_enabled()),
        "inference_mode": bool(torch.is_inference_mode_enabled()),
    }
    for device in _AUTOCAST_DEVICES:
        try:
            # Routed through the version-neutral shim so TorchLens runs on
            # torch 2.1+ (the per-device ``device_type`` argument to these
            # query helpers is torch 2.4+ only). On torch>=2.4 the shim calls
            # ``torch.is_autocast_enabled``/``torch.get_autocast_dtype``
            # directly (identical behavior); on torch 2.1-2.3 it routes to the
            # legacy per-device helpers. See utils/_torch_compat.py.
            state[device] = {
                "enabled": autocast_is_enabled(device),
                "dtype": autocast_get_dtype(device),
            }
        except (RuntimeError, TypeError):
            # Device doesn't support autocast queries (e.g. no CUDA).
            pass
    return state


class AutocastRestore:
    """Context manager that re-enters saved autocast contexts during replay.

    Only devices that were *enabled* at capture time get an autocast
    context opened.  Contexts are exited in reverse order on ``__exit__``.

    Usage::

        with AutocastRestore(saved_state):
            result = func(*args, **kwargs)
    """

    __slots__ = ("_autocast_state", "_contexts")

    def __init__(self, autocast_state: dict[str, dict[str, Any]]) -> None:
        """Store serialized autocast state for later context restoration.

        Parameters
        ----------
        autocast_state:
            Mapping from device type to captured autocast enabled/dtype state.
        """

        self._autocast_state = autocast_state
        self._contexts: List[Any] = []

    def __enter__(self) -> "AutocastRestore":
        """Enter captured autocast contexts.

        Returns
        -------
        AutocastRestore
            This context manager instance.
        """

        for device, state in self._autocast_state.items():
            if device.startswith("__"):
                # Reserved non-device entries (e.g. ``__execution__`` grad/inference
                # mode) are not autocast device records and open no context here.
                continue
            if state["enabled"]:
                autocast = cast(Any, getattr(torch.amp, "autocast"))
                ctx = autocast(device, dtype=state["dtype"])
                ctx.__enter__()
                self._contexts.append(ctx)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit restored autocast contexts in reverse nesting order.

        Parameters
        ----------
        exc_type:
            Exception type propagated by the managed block, if any.
        exc_value:
            Exception instance propagated by the managed block, if any.
        traceback:
            Traceback propagated by the managed block, if any.
        """

        # Exit in reverse order to mirror the nesting order of __enter__.
        for ctx in reversed(self._contexts):
            ctx.__exit__(exc_type, exc_value, traceback)


# ======================================================================================
# Host-nondeterminism channel monitor (r37 hon1_2; r39 CLASS A enumeration completeness).
#
# The global-engine snapshots above cover only the two REPLAYABLE engines (module
# ``random``, legacy ``np.random``). Every OTHER host entropy / clock / RNG-instance
# channel is monitored here over a data-driven FROZEN registry
# (:data:`HOST_NONDETERMINISM_REGISTRY`). A positive touch permanently ceilings the
# capture (``host_rng_consumed=True`` with NO identifiable seed -> every replay
# UNVERIFIABLE + NOT_APPLICABLE). Monitor uncertainty -- install/chain/inventory/restore
# failure, an observed-but-unclassified host event, or a live pre-existing non-owner
# Python thread whose ephemeral draws are unwitnessable on <=3.11 -- is itself recorded
# and downgrades capture completeness to INCOMPLETE; it NEVER reads as "no consumption".
#
# A negative witness (``host_rng_consumed=False``) is honest ONLY when every required
# observer over the declared channel/thread surface installed, classified, stayed
# installed, and restored exactly. This is the r39 enumeration-completeness invariant:
# "one more RNG/clock name" is a failing registry meta-test, not a false VERIFIED.
#
# Residual tail (outside any Python-visible call surface, disclaimed in contract s11):
# direct ``/dev/urandom`` file reads; ctypes / user C-extension entropy or clock reads;
# legacy ``RandomState()`` C-level construction entropy (its DRAWS stay digest-witnessed).
# ======================================================================================


@dataclass(frozen=True)
class HostNondeterminismRow:
    """One declared host-nondeterminism channel in the frozen monitoring registry.

    ``family`` groups the channel (``clock`` / ``entropy`` / ``construction`` /
    ``rng_primitive`` / ``rng_instance``); ``target`` is its human-readable identity;
    ``strategy`` is HOW it is observed (see the coverage meta-test's allowlist);
    ``thread_scope`` is WHERE a positive can be observed (``any`` -- thread-independent
    module/class patch or process inventory; ``owner`` -- capture owner thread only;
    ``hooked`` -- owner plus every thread started in-window and profile-hooked);
    ``classification`` records the semantic role used by the arg/receiver classifiers.
    """

    family: str
    target: str
    strategy: str
    thread_scope: str
    classification: str


# ---- Clock vocabulary (r39 R2/R2b) ----------------------------------------------------
#
# The current-clock readers below all read the CURRENT wall/monotonic clock; a positive
# marks on ANY covered thread. Module-attr readers fire thread-independently; the
# immutable ``datetime`` classmethods (which CANNOT be class-patched -- extension types)
# are classified by c_call identity on the owner + hooked threads.

_CLOCK_COUNTER_NAMES = (
    "time",
    "time_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
    "thread_time",
    "thread_time_ns",
    "clock_gettime",
    "clock_gettime_ns",
)
"""``time.*`` readers that always read the current clock (no explicit-time argument)."""

# Implicit-now converters: ``localtime()`` / ``strftime(fmt)`` etc. read the CURRENT
# clock when their optional explicit-time argument is absent/None, but are pure
# TRANSFORMS when given an explicit time. ``time_arg_index`` is the position of that
# optional argument (``strftime`` takes ``(format, [t])`` -> index 1; the rest index 0).
_CLOCK_IMPLICIT_NOW: tuple[tuple[str, int], ...] = (
    ("localtime", 0),
    ("gmtime", 0),
    ("asctime", 0),
    ("ctime", 0),
    ("strftime", 1),
)

# Immutable ``datetime`` current-clock readers, classified by c_call identity
# ``(receiver_type, method_name)`` because the receiver types are unpatchable C types.
_DATETIME_CLOCK_READERS: tuple[tuple[Any, str], ...] = (
    (_datetime_module.datetime, "now"),
    (_datetime_module.datetime, "utcnow"),
    (_datetime_module.datetime, "today"),
    (_datetime_module.date, "today"),
)


def _build_host_nondeterminism_registry() -> tuple[HostNondeterminismRow, ...]:
    """Assemble the frozen channel registry the runtime classifiers are built from."""

    rows: list[HostNondeterminismRow] = []
    for name in _CLOCK_COUNTER_NAMES:
        rows.append(
            HostNondeterminismRow("clock", f"time.{name}", "module_patch", "any", "current_reader")
        )
    for name, _index in _CLOCK_IMPLICIT_NOW:
        rows.append(
            HostNondeterminismRow("clock", f"time.{name}", "module_patch", "any", "current_reader")
        )
    rows.append(HostNondeterminismRow("clock", "os.times", "module_patch", "any", "current_reader"))
    rows.append(
        HostNondeterminismRow(
            "clock", "resource.getrusage", "module_patch", "any", "current_reader"
        )
    )
    for receiver, method in _DATETIME_CLOCK_READERS:
        target = f"datetime.{receiver.__name__}.{method}"
        rows.append(
            HostNondeterminismRow("clock", target, "c_call_identity", "hooked", "current_reader")
        )
    rows.append(HostNondeterminismRow("entropy", "os.urandom", "module_patch", "any", "funnel"))
    rows.append(HostNondeterminismRow("entropy", "os.getrandom", "module_patch", "any", "funnel"))
    rows.append(
        HostNondeterminismRow("entropy", "random._urandom", "module_patch", "any", "funnel")
    )
    rows.append(
        HostNondeterminismRow(
            "construction", "numpy.random.default_rng", "module_patch", "any", "construction"
        )
    )
    rows.append(
        HostNondeterminismRow(
            "construction",
            "numpy.random.bit_generator.randbits",
            "construction_entropy",
            "any",
            "construction",
        )
    )
    for cls_name in ("random.Random", "random.SystemRandom", "_random.Random"):
        for method in ("random", "getrandbits", "randbytes"):
            rows.append(
                HostNondeterminismRow(
                    "rng_primitive", f"{cls_name}.{method}", "class_patch", "any", "primitive"
                )
            )
    rows.append(
        HostNondeterminismRow(
            "rng_instance",
            "numpy.random.Generator/RandomState/BitGenerator",
            "receiver_profile",
            "hooked",
            "instance_draw",
        )
    )
    rows.append(
        HostNondeterminismRow(
            "rng_instance", "_random.Random", "receiver_profile", "hooked", "instance_draw"
        )
    )
    rows.append(
        HostNondeterminismRow(
            "rng_instance",
            "numpy.random.Generator/RandomState/BitGenerator",
            "state_inventory",
            "any",
            "instance_draw",
        )
    )
    return tuple(rows)


HOST_NONDETERMINISM_REGISTRY: tuple[HostNondeterminismRow, ...] = (
    _build_host_nondeterminism_registry()
)
"""Frozen data-driven inventory of every monitored host-nondeterminism channel.

The runtime classifiers are built FROM this registry; the coverage meta-test asserts
every row has a valid strategy + thread policy and that no numpy draw-method NAME is
enumerated anywhere (receiver typing + state digests cover new draw-method names). A
future stdlib clock/RNG endpoint is a failing meta-test until it is given a registry row.

r41 (hon1_1): every ``module_patch`` row (and the construction-entropy alias) carries a
SECOND observation layer -- the ORIGINAL builtin's identity is registered at monitor
install, before the module attribute is replaced, so a pre-window held reference
(``from time import time`` / ``from os import urandom`` at the top of a model or helper
module) is classified from the ``c_call`` profile event on the owner and every in-window
hooked thread. The held-ref layer adds NO new rows and NO new strategy vocabulary; the
r41 held-ref immunizer derives its obligations from ``strategy == "module_patch"``, so a
future module-patch row without identity registration (or without a probe recipe) is a
RED test, not a silent gap.
"""


class _NotADigestableRng(Exception):
    """Internal sentinel: the value is not a digestable numpy/`random` generator."""


class HostRngMonitorResult:
    """Outcome of one capture-scoped host-nondeterminism monitoring window."""

    __slots__ = ("channels", "uncertain", "uncertain_detail")

    def __init__(self) -> None:
        self.channels: set[str] = set()
        self.uncertain: bool = False
        # Actionable named-thread / failure detail for the INCOMPLETE ceiling so the
        # readiness diagnostic and ``tl.compat.report()`` can name the offending domain.
        self.uncertain_detail: tuple[str, ...] = ()


def _torchlens_module_globals_ids() -> frozenset[int]:
    """Identity keys of every loaded torchlens module's globals dict.

    TorchLens's own capture machinery reads clocks constantly (per-op timing); its
    reads are excluded by EXACT module ownership -- the caller frame's globals dict
    identity against this registry -- never by filename strings or frame ancestry.
    A user callback's code keeps its own module globals even when invoked from a
    TorchLens frame, so user reads are always detected.
    """

    ids = set()
    for name, module in list(_sys_module.modules.items()):
        if module is not None and (name == "torchlens" or name.startswith("torchlens.")):
            module_dict = getattr(module, "__dict__", None)
            if module_dict is not None:
                ids.add(id(module_dict))
    return frozenset(ids)


def _rng_exempt_instances() -> tuple[Any, ...]:
    """Replayable global singletons + TorchLens's private barcode RNG (identity-exempt).

    These are the load-bearing no-over-trigger gate: the legacy ``mtrand._rand``
    singleton and module ``random`` engine keep their SEEDED-reproduction semantics
    (a seeded model stays VERIFIED), and TorchLens's own barcode RNG draws during the
    forward must never ceiling the capture.
    """

    exempt: list[Any] = []
    inst = getattr(random, "_inst", None)
    if inst is not None:
        exempt.append(inst)
    np_singleton = getattr(getattr(np.random, "mtrand", None), "_rand", None)
    if np_singleton is not None:
        exempt.append(np_singleton)
    try:
        from .hashing import _BARCODE_RNG

        exempt.append(_BARCODE_RNG)
    except Exception:  # pragma: no cover - hashing is a first-party import
        pass
    # A numpy ``Generator``/``RandomState`` OWNS a distinct ``BitGenerator`` object that
    # advances on every draw and is itself GC-visible. Exempt those underlying bit
    # generators transitively so the process-wide inventory does not falsely ceiling a
    # SEEDED replayable-singleton draw (the load-bearing over-trigger gate; a seeded
    # ``np.random.random()`` model must stay VERIFIED).
    for candidate in list(exempt):
        bit_generator = None
        if isinstance(candidate, np.random.Generator):
            bit_generator = getattr(candidate, "bit_generator", None)
        elif isinstance(candidate, np.random.RandomState):
            bit_generator = getattr(candidate, "_bit_generator", None) or getattr(
                candidate, "bit_generator", None
            )
        if bit_generator is not None:
            exempt.append(bit_generator)
    return tuple(exempt)


_CUSTOM_HOLDER_SKIP_MODULES = frozenset(
    {
        # Stdlib runtime primitives + torch/numpy implementation objects the custom-holder
        # generator sweep (r42 corr2_1) must NOT recurse into: threading/asyncio/file/
        # socket handles and torch/numpy internals (a digestable RNG is snapshotted BEFORE this
        # skip check; everything else in those namespaces is impl noise). Keyed on the top-level
        # module of the value's TYPE. r45 hon1_1: ``collections`` and ``queue`` are REMOVED --
        # ``deque`` / ``ChainMap`` / ``UserList`` / ``UserDict`` / ``Counter`` and the ``queue.*``
        # containers are now reached structurally (the ``Mapping`` / non-leaf ``Collection`` descent
        # and the queue-protocol snapshot) BEFORE this skip check ever runs, so their held
        # generators are no longer silently missed.
        "builtins",
        "threading",
        "asyncio",
        "socket",
        "io",
        "_io",
        "selectors",
        "subprocess",
        "multiprocessing",
        "concurrent",
        "ctypes",
        "weakref",
        "typing",
        "functools",
        "pathlib",
        "logging",
        "numpy",
        "torch",
    }
)
"""Top-level module names whose instances the custom-holder generator sweep skips (r42 corr2_1)."""

_STDLIB_CLASS_LEAF_MODULES: frozenset[str] = frozenset(_sys_module.stdlib_module_names) - {
    "__main__"
}
"""Stdlib top-level module names whose CLASSES the class-surface walk leafs (r56 amb_1).

Structural (``sys.stdlib_module_names``), never a hand-maintained denylist -- closing the
whole "stdlib class dict carries a process-global registry" family at once (``collections.abc``
ABC ``_abc_impl`` caches were the proven member: r45 removed ``collections`` from the
INSTANCE-side skip set so containers descend, which silently re-opened the CLASS-side walk of
ABC dicts). Applies ONLY to :meth:`host_nondeterminism_monitor._is_trusted_leaf_class` -- the
instance-side container/holder descent is untouched, so a ``deque`` / ``UserList`` /
``UserDict`` subclass instance and its held generators stay fully reachable. ``__main__`` is
carved out: model classes defined in scripts/notebooks keep r53 class-attribute coverage.
"""

_INVENTORY_LEAF_TYPES: tuple[type, ...] = (
    str,
    bytes,
    bytearray,
    memoryview,
    np.ndarray,
    np.generic,
    torch.Tensor,
    torch.nn.Module,
    ModuleType,
)
"""Types the model-attribute generator sweep treats as LEAVES (r45 hon1_1).

``str`` / ``bytes`` / ``bytearray`` / ``memoryview`` / ``np.ndarray`` / a bare ``torch.Tensor``
satisfy ``collections.abc.Collection`` yet must NOT be descended element-wise: a string would
explode into per-character strings (huge / cyclic node blow-up) and a tensor / ndarray holds no
Python-level RNG. ``torch.nn.Module`` and ``ModuleType`` are likewise excluded from the
``Collection`` descent branch (this entry is inert for ``nn.Module`` -- it is not a
``collections.abc.Collection`` -- and documents intent). r51 hon1_1: an ``nn.Module`` is NO LONGER
a hard inventory leaf; it is descended via ``_is_recursable_custom_holder`` (its own
``__dict__`` / ``__slots__``), reaching a generator behind an UNREGISTERED submodule.
"""


def _derive_abc_impl_type() -> type | None:
    """The C ``_abc._abc_data`` type carried by every ``ABCMeta`` class (r56 amb_1).

    Derived from a live ABC rather than importing the private ``_abc`` module. ``None`` on
    a runtime using the pure-Python ``_py_abc`` fallback (no CPython ``_abc_impl`` slot),
    where this exclusion is inapplicable by construction.
    """

    probe = Collection.__dict__.get("_abc_impl")
    if probe is not None and type(probe).__module__ == "_abc":
        return type(probe)
    return None


_ABC_IMPL_TYPE: type | None = _derive_abc_impl_type()

_GC_EXPANSION_LEAF_TYPES: tuple[type, ...] = (
    ModuleType,
    CodeType,
    BuiltinFunctionType,
    FrameType,
    np.ndarray,
    np.generic,
    torch.Tensor,
) + ((_ABC_IMPL_TYPE,) if _ABC_IMPL_TYPE is not None else ())
"""Types the authoritative ``gc.get_referents`` fallback never expands (r55 C6).

Each exclusion is JUSTIFIED, not stylistic -- everything else reachable is walked:

- ``ModuleType``: a module is a SHARED namespace; expanding it drops the rooted
  walk into every imported framework's globals. A generator held in a shared
  module global is the explicit documented residual (contract section 11).
- ``CodeType``: code objects hold only compile-time constants (``co_consts``)
  -- a live generator cannot exist there.
- ``BuiltinFunctionType``: a C function's only referents are its ``__self__``
  module and shared runtime plumbing.
- ``FrameType``: frames chain through ``f_back`` into the ENTIRE interpreter
  call stack (TorchLens's own frames included) -- shared, unbounded state.
- ``np.ndarray`` / ``np.generic`` / ``torch.Tensor``: the r45 numeric-leaf
  posture (``_INVENTORY_LEAF_TYPES``) -- a tensor's traverse reaches autograd
  internals, and neither holds a Python-level RNG in supported usage.
- ``_abc._abc_data`` (r56 amb_1): every ``ABCMeta`` class's ``_abc_impl`` slot.
  Its ``tp_traverse`` exposes the ABC registry/cache/NEGATIVE-cache weakref sets
  -- PROCESS-GLOBAL bookkeeping accumulating a weakref to every type ever
  ``isinstance``-checked against that ABC. Expanding one bridges the rooted walk
  out of the model subgraph into the ambient object graph (every cached class's
  dict, then its attribute webs), making sweep completeness ambient-state-
  dependent: under a large ambient graph the walk digested foreign generators
  and exhausted the node cap BEFORE the model's own generator (the r55 full-run
  regression on ``test_r47_generator_container_subclass``). The caches hold only
  weakrefs to TYPES plus registered classes, so a model-held generator can never
  live inside one -- excluding them loses zero legitimate coverage.

Deliberately NOT excluded: ``torch.nn.Module`` (r51 hon1_1 -- an unregistered
submodule must stay reachable), ``type`` objects (a referent class is enqueued
and handled by the r53 class-surface branch with its trusted-leaf gate), and
stdlib wrapper instances such as ``functools._lru_cache_wrapper`` (r54 corr_4:
the ``__wrapped__`` edge must be walked; boundedness is owned by the node cap,
inertness by ``tp_traverse`` itself).
"""

_INVENTORY_NODE_CAP = 1_000_000
"""Defensive node cap for the model-attribute generator sweep (r41 hon1_2).

Realistic models sit 2-3 orders of magnitude below this (a 400-block toy holds ~7.7k
module-dict values; the cap admits ~1M visited nodes in ~300 ms measured), so no
realistic deterministic model can be ceilinged by it. Exhaustion NEVER truncates
silently: it flags ``inventory_budget_exhausted`` monitor uncertainty, downgrading
capture completeness to INCOMPLETE -- a truncated inventory never reads as
no-consumption. Read at call time so the cap-exhaustion invariant is testable at any
cap value.
"""


def _call_site_argcount(frame: Any) -> int | None:
    """Decode the positional argument count of a profile-observed ``c_call`` site.

    Reads the caller frame's bytecode at ``f_lasti``. A plain ``CALL`` instruction's
    oparg IS the exact positional argument count on the pinned interpreter (py3.11;
    the r41 unit pin goes RED at an interpreter bump so a bytecode change is caught at
    upgrade time). The monitored implicit-now converters reject keywords, so ``CALL``
    fully determines arity for every valid call.

    Parameters
    ----------
    frame:
        Caller frame supplied by the ``c_call`` profile event.

    Returns
    -------
    int | None
        Positional argument count for a plain ``CALL`` site; ``None`` for any other
        opcode (``CALL_FUNCTION_EX`` star-calls) or decode failure, so callers mark
        fail-closed (over-marking, never under-marking).
    """

    try:
        lasti = frame.f_lasti
        for instruction in _dis_module.get_instructions(frame.f_code):
            if instruction.offset == lasti:
                if instruction.opname == "CALL" and instruction.arg is not None:
                    return int(instruction.arg)
                return None
        return None
    except Exception:
        return None


_ACTIVE_MONITOR: "host_nondeterminism_monitor | None" = None
"""The capture-scoped monitor currently installed, or ``None`` (r41 hon2_1).

Published as the LAST statement of ``__enter__`` and cleared FIRST in ``__exit__`` so
readers never observe a partially-installed window. Captures do not nest
(``active_logging`` rejects nested captures), so a single slot is sufficient.
"""


def active_in_window_thread_idents() -> AbstractSet[int]:
    """Return the thread idents profile-hooked during the active capture window.

    The tensor->host escape belt (``completeness_witness``) consults this registry to
    classify a non-owner escape thread as IN-WINDOW (started during the forward and
    hooked by ``threading.setprofile`` -- registered during thread bootstrap BEFORE its
    first user statement, so even an escape-first thread is classified) versus
    PRE-EXISTING/foreign. Entries only ever come from hooked threads, so ident reuse
    cannot misclassify a foreign thread as in-window. Returns an empty set when no
    monitor window is active.
    """

    monitor = _ACTIVE_MONITOR
    if monitor is None:
        return frozenset()
    return monitor._in_window_thread_idents


class host_nondeterminism_monitor:
    """Context manager installing the registry-driven host-nondeterminism monitor.

    Mechanisms (r39 CLASS A + r41, all built FROM :data:`HOST_NONDETERMINISM_REGISTRY`):

    * **Model-attribute state digest (thread-independent belt).** A cycle-safe sweep of
      every numpy ``Generator`` / ``RandomState`` / bare ``BitGenerator`` / ``random``
      generator the MODEL itself holds -- submodule ``__dict__`` values INCLUDING builtin
      container nesting (list/tuple/set/frozenset elements and dict keys AND values, r41)
      -- never a process-wide ``gc`` scan. Before/after state digests catch a draw on ANY
      thread, including a pre-existing worker. Exhaustion of the defensive
      :data:`_INVENTORY_NODE_CAP` flags ``inventory_budget_exhausted`` (INCOMPLETE),
      never a silent truncation.
    * **Class patches (thread-independent belt).** ``random.Random`` / ``random.SystemRandom``
      / ``_random.Random`` draw primitives (the bare-``_random.Random()`` channel; measured
      E1 patchable), plus the ``os.urandom`` / ``os.getrandom`` / ``random._urandom`` entropy
      funnel and the ``numpy.random.default_rng`` factory.
    * **Construction entropy.** The writable ``numpy.random.bit_generator.randbits`` alias
      (measured E5): an UNSEEDED BitGenerator/``default_rng()`` construction on ANY thread
      marks, closing the ephemeral-unseeded-generator channel structurally.
    * **Clock family.** Module-attr wrappers for every ``time.*`` current-clock reader (the
      implicit-now converters mark only with no explicit-time argument), ``os.times`` and
      feature-detected ``resource.getrusage``; c_call identity for the immutable
      ``datetime`` current readers (E1: unpatchable extension types).
    * **Held-reference identity (r41 hon1_1).** Every module-patched original's ``id()``
      is registered BEFORE its attribute is replaced, so a pre-window held reference
      (``from time import time`` / ``from os import urandom`` in a model or helper
      module) marks by ``c_call`` identity on the owner and every in-window hooked
      thread. The implicit-now converters decode the call site's positional argcount
      from the caller frame's bytecode (:func:`_call_site_argcount`), keeping a held
      ``localtime(t)`` a pure transform; an undecodable site (star-call) marks
      fail-closed. TorchLens's own frames are exempt by exact module-globals ownership
      (its per-op clock reads route patched-attr -> wrapper -> original, emitting
      ``c_call`` for the original from the wrapper's frame).
    * **Dual chained profile hooks (belt).** ``sys.setprofile`` (owner thread) AND
      ``threading.setprofile`` (threads STARTED in-window; measured E2: a pre-existing
      worker is unreachable). Each hook chains its own exact predecessor and is
      identity-restored on success and exception. The threading hook additionally
      records each hooked thread's ident into the in-window registry consumed by the
      cross-thread escape belt (r41; see :func:`active_in_window_thread_idents`).

    Entropy / instance / construction / clock positives mark from any COVERED thread. A
    REALISTIC pre-existing-thread RNG use (a background worker drawing from a MODEL-HELD
    Generator/RandomState/BitGenerator -- held anywhere the inert-reachability walk can
    follow WITHOUT executing user code, incl. class descriptors, weakrefs, and callable
    interiors; r53 corr/F1) is witnessed thread-independently by the state digest, and an
    unseeded construction on any thread by the module/class patches. The residual is only
    an EXTERNALLY-HELD generator drawn on a pre-existing (non-hooked) thread that is
    reachable ONLY BY EXECUTING USER CODE (a property/descriptor ``__get__`` body,
    ``__getattr__``, or a callable's return value), of the same class as the adversarial
    draw+``state`` RESTORE (E4) -- a self-cleaning sequence no py<=3.11 mechanism can
    witness -- documented in contract s11, NOT a blanket ceiling: the r38 draft's
    thread-presence INCOMPLETE over-triggered every capture running alongside a benign
    background thread (DataLoader/Jupyter/pytest), so it is intentionally not applied.
    Future all-thread coverage is ``sys.monitoring`` (PEP 669, 3.12+, interpreter-wide).
    """

    def __init__(self, model: Any = None) -> None:
        # The model is swept for held numpy/`random` generators (a cheap container-aware
        # thread-independent digest belt) -- NOT a process-wide ``gc.get_objects()`` scan.
        self._model = model
        self.result = HostRngMonitorResult()
        self._restores: list[Callable[[], None]] = []
        self._owner_thread = _threading_module.get_ident()
        self._previous_sys_profile: Any = None
        self._previous_threading_profile: Any = None
        self._sys_hook: Any = None
        self._threading_hook: Any = None
        self._sys_profile_installed = False
        self._threading_profile_installed = False
        self._generator_states: list[tuple[Any, str]] = []
        self._tl_globals_ids: frozenset[int] = frozenset()
        self._exempt_ids: frozenset[int] = frozenset()
        self._clock_ccall_keys: dict[tuple[int, str], str] = {}
        # r41 hon1_1: id(original builtin) -> (channel, time_arg_index). ``None`` index
        # marks unconditionally; an int index marks only when the observed call site's
        # positional argcount leaves the explicit-time argument absent (or undecodable).
        self._held_ref_marks: dict[int, tuple[str, int | None]] = {}
        # r41 hon2_1: idents of threads hooked by the in-window threading profile hook,
        # consumed by the escape belt's 3-class thread gate via ``_ACTIVE_MONITOR``.
        self._in_window_thread_idents: set[int] = set()
        # r49 hon1_1: re-entrancy depth for monitor-INTERNAL probes. While > 0 the monitor is
        # reading through its OWN inventory probe (owner-thread, ``__enter__``-scoped, BEFORE the
        # user forward runs), so any channel a probe transitively touches must NOT be marked as a
        # model host read. Guarded at the single ``_mark`` choke point -> surface-complete over
        # every clock/entropy channel.
        self._suppress_self_marks: int = 0

    # -- helpers -----------------------------------------------------------------

    def _mark(self, channel: str) -> None:
        if self._suppress_self_marks:
            return
        self.result.channels.add(channel)

    @contextmanager
    def _monitor_internal_probe(self) -> Iterator[None]:
        """Suppress the monitor's OWN transitive channel marks during an internal probe (r49 hon1_1).

        A monitor-initiated read is NOT a model host read. The opaque-queue emptiness proof calls
        ``multiprocessing.Queue.empty()``, which reads ``time.monotonic`` through
        ``multiprocessing.connection`` -- without this guard that TorchLens-initiated probe would
        self-mark the clock channel and over-trigger a deterministic model merely holding an empty
        ``mp.Queue`` to UNVERIFIABLE (the r48 hon1_1 regression). Guarding at the single ``_mark``
        choke point is surface-complete: ANY channel a future inventory probe transitively touches
        is auto-exempt. The bracket is owner-thread and ``__enter__``-scoped (BEFORE the user
        forward runs), so no user/model/worker host read is ever inside it.
        """

        self._suppress_self_marks += 1
        try:
            yield
        finally:
            self._suppress_self_marks -= 1

    def _flag_uncertain(self, reason: str) -> None:
        self.result.uncertain = True
        if reason:
            self.result.uncertain_detail = (*self.result.uncertain_detail, reason)

    def _patch_attr(self, holder: Any, name: str, wrapper: Any) -> None:
        original = getattr(holder, name)
        setattr(holder, name, wrapper)

        def _restore(holder: Any = holder, name: str = name, original: Any = original) -> None:
            if getattr(holder, name, None) is not wrapper:
                # Someone replaced our patch mid-window: restoration cannot be
                # proven exact -> uncertainty (fail closed), restore anyway.
                self._flag_uncertain(f"patch_replaced:{getattr(holder, '__name__', holder)}.{name}")
            setattr(holder, name, original)

        self._restores.append(_restore)

    def _register_held_ref(
        self, original: Any, channel: str, time_arg_index: int | None = None
    ) -> None:
        """Register a module-patched ORIGINAL builtin for held-reference c_call marking.

        Called at each module-attr patch site BEFORE :meth:`_patch_attr` replaces the
        attribute, so a pre-window ``from module import name`` alias -- which calls the
        original object directly, bypassing the patch -- is classified by identity in
        :meth:`_classify_c_call` (r41 hon1_1). Class-patch originals are deliberately
        NOT registered: a held bound draw method hits the existing receiver-isinstance
        branch already.
        """

        # ``setdefault``: some registry channels alias ONE builtin object
        # (``random._urandom`` IS ``os.urandom``); the first-registered (canonical)
        # channel name wins for the shared identity.
        self._held_ref_marks.setdefault(id(original), (channel, time_arg_index))

    def _entropy_wrapper(self, original: Any, channel: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._mark(channel)
            return original(*args, **kwargs)

        return wrapper

    def _clock_wrapper(self, original: Any, channel: str, time_arg_index: int | None) -> Any:
        tl_ids = self._tl_globals_ids

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            explicit_time = (
                time_arg_index is not None
                and len(args) > time_arg_index
                and args[time_arg_index] is not None
            )
            if not explicit_time:
                try:
                    caller_globals_id = id(_sys_module._getframe(1).f_globals)
                except Exception:
                    caller_globals_id = -1
                if caller_globals_id not in tl_ids:
                    self._mark(channel)
            return original(*args, **kwargs)

        return wrapper

    def _instance_method_wrapper(self, original: Any, channel: str) -> Any:
        exempt_ids = self._exempt_ids

        def wrapper(self_rng: Any, *args: Any, **kwargs: Any) -> Any:
            if id(self_rng) not in exempt_ids:
                self._mark(channel)
            return original(self_rng, *args, **kwargs)

        return wrapper

    def _classify_c_call(self, frame: Any, arg: Any) -> None:
        # r41 hon1_1: held-reference identity FIRST. A pre-window ``from time import
        # time`` alias calls the ORIGINAL builtin, bypassing the module-attr patch; the
        # original was identity-registered before patching. TorchLens's own frames are
        # exempt by EXACT module-globals ownership -- its per-op clock reads route
        # patched-attr -> ``_clock_wrapper`` -> original, emitting ``c_call`` for the
        # original FROM the wrapper's frame; without the exemption every capture would
        # self-ceiling.
        mark = self._held_ref_marks.get(id(arg))
        if mark is not None:
            try:
                caller_globals_id = id(frame.f_globals)
            except Exception:
                caller_globals_id = -1
            if caller_globals_id not in self._tl_globals_ids:
                held_channel, time_arg_index = mark
                if time_arg_index is None:
                    self._mark(held_channel)
                else:
                    # Implicit-now converter: a call site providing the explicit-time
                    # argument is a pure transform. Undecodable (star-call / unknown
                    # opcode) marks fail-closed -- over-marking, never under-marking.
                    argcount = _call_site_argcount(frame)
                    if argcount is None or argcount <= time_arg_index:
                        self._mark(held_channel)
        receiver = getattr(arg, "__self__", None)
        if receiver is None:
            return
        # numpy instance draws + bare ``_random.Random`` draws (receiver typing -- no
        # draw-method NAME enumeration, so new draw methods need no detector edit).
        if id(receiver) not in self._exempt_ids and isinstance(
            receiver,
            (
                _c_random_module.Random,
                np.random.Generator,
                np.random.RandomState,
                np.random.BitGenerator,
            ),
        ):
            self._mark("c_rng_instance_draw")
            return
        # Immutable ``datetime`` current-clock readers (r42 hon1_1: subclass-safe). The base
        # readers are class methods on unpatchable C extension types, so a ``datetime.datetime``
        # / ``datetime.date`` SUBCLASS inherits them and reads the SAME wall clock -- but the
        # exact ``id(receiver)`` key misses the subclass. Classify subclass-safe, mirroring the
        # numpy instance-draw ``isinstance`` branch, while guarding a genuinely re-implemented
        # subclass method (an override that does not call the inherited reader is not attributed).
        name = getattr(arg, "__name__", None)
        if name is not None and isinstance(receiver, type):
            channel = self._datetime_clock_channel(receiver, name)
            if channel is not None:
                self._mark(channel)

    def _datetime_clock_channel(self, receiver: type, name: str) -> str | None:
        """Classify a ``datetime``/``date`` current-clock ``c_call`` receiver (r42 hon1_1)."""

        # Exact base receiver: the original registered key (fast path, unchanged).
        exact = self._clock_ccall_keys.get((id(receiver), name))
        if exact is not None:
            return exact
        # Inherited C reader on a subclass: ``now``/``utcnow``/``today`` on a ``datetime``
        # subclass, ``today`` on a (non-datetime) ``date`` subclass. Not attributed when the
        # subclass genuinely OVERRIDES the reader (its own ``__dict__`` defines the name before
        # the base in the MRO) -- an override that returns a fixed value is not a clock read; if
        # it calls the inherited reader, that inner call is caught by the exact-key path.
        if name in ("now", "utcnow", "today") and issubclass(receiver, _datetime_module.datetime):
            if not self._subclass_overrides_reader(receiver, name, _datetime_module.datetime):
                return f"datetime.datetime.{name}"
            return None
        if name == "today" and issubclass(receiver, _datetime_module.date):
            if not self._subclass_overrides_reader(receiver, name, _datetime_module.date):
                return "datetime.date.today"
        return None

    @staticmethod
    def _subclass_overrides_reader(receiver: type, name: str, base: type) -> bool:
        """Return whether ``receiver`` redefines ``name`` above ``base`` in its MRO (r42 hon1_1)."""

        for klass in receiver.__mro__:
            if klass is base:
                return False
            if name in getattr(klass, "__dict__", {}):
                return True
        return False

    def _make_profile_hook(self, predecessor: Any, *, records_thread_ident: bool = False) -> Any:
        def hook(frame: Any, event: str, arg: Any) -> Any:
            if records_thread_ident:
                # r41 hon2_1: the threading hook registers every hooked thread's ident
                # (idempotent set.add, GIL-atomic) during thread bootstrap -- BEFORE the
                # thread's first user statement -- so the escape belt's in-window
                # classification is race-free even for an escape-first thread.
                try:
                    self._in_window_thread_idents.add(_threading_module.get_ident())
                except Exception:
                    self._flag_uncertain("profile_classifier_error")
            try:
                if event == "c_call":
                    self._classify_c_call(frame, arg)
            except Exception:
                self._flag_uncertain("profile_classifier_error")
            if predecessor is not None:
                try:
                    predecessor(frame, event, arg)
                except Exception:
                    self._flag_uncertain("profile_predecessor_error")

        return hook

    @staticmethod
    def _is_recursable_custom_holder(value: Any) -> bool:
        """Return whether ``value`` is a holder to recurse into (r42 corr2_1 / r51 hon1_1).

        Recurse into plain user objects that carry an attribute surface AND (r51 hon1_1) into
        every reachable ``nn.Module`` -- its own ``__dict__`` / ``__slots__`` (which hold
        ``_parameters`` / ``_buffers`` / ``_modules`` plus arbitrary user attributes). A REGISTERED
        submodule is additionally seeded via the top ``model.modules()`` loop (and premarked in
        ``seen_container_ids`` so it is not double-walked); an UNREGISTERED submodule (held in a
        plain attribute, ``list`` / ``dict`` / nested container, or custom holder -- the "modules
        must live in an ``nn.ModuleList``" footgun) is reachable ONLY through this branch, so a
        numpy Generator behind it is finally witnessed. Skips scalars, classes (walked through
        the class-surface edge instead, r53 corr), modules, tensors, ndarrays, and any
        stdlib/torch/numpy implementation object (:data:`_CUSTOM_HOLDER_SKIP_MODULES`). A
        digestable RNG never reaches here (it is snapshotted before this branch).

        r53 corr/F1: the former blanket ``callable(value) -> False`` gate is GONE -- it made a
        user-defined CALLABLE INSTANCE (``self.op = CallableSampler()``, the idiomatic callable
        transform/sampler object) a hard leaf, leaving its held generator unwitnessed (false
        VERIFIED). r55 C6: this predicate now gates ONLY the dual-walk of container/weakref
        SUBCLASS own-attribute surfaces (the Mapping / Collection / ``weakref.ref`` branches);
        the sweep's terminal fallback expands every other node through the authoritative
        ``gc.get_referents`` enumerator (:meth:`_inert_gc_children`) instead of this
        module-gated attribute-surface check.
        """

        if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool, complex)):
            return False
        # r51 hon1_1: an nn.Module is a recursable holder. This branch MUST precede the
        # module / torch-namespace gates below because a torch built-in module (``nn.Linear``)
        # lives in the skipped ``torch`` top-level namespace -- the gate would otherwise
        # short-circuit it to a hard leaf, the hon_1 hole.
        if isinstance(value, torch.nn.Module):
            return True
        if isinstance(value, type):
            return False
        if isinstance(value, ModuleType):
            return False
        if isinstance(value, (np.ndarray, np.generic)):
            return False
        if isinstance(value, torch.Tensor):  # r51 hon1_1: was ``(torch.Tensor, torch.nn.Module)``;
            return False  # nn.Module is now recursable via the branch above.
        module = getattr(type(value), "__module__", "") or ""
        if module.split(".", 1)[0] in _CUSTOM_HOLDER_SKIP_MODULES:
            return False
        try:
            has_dict = isinstance(object.__getattribute__(value, "__dict__"), dict)
        except (AttributeError, TypeError):
            has_dict = False
        has_slots = any(
            "__slots__" in getattr(klass, "__dict__", {}) for klass in type(value).__mro__
        )
        return has_dict or has_slots

    @staticmethod
    def _custom_holder_children(value: Any) -> list[Any]:
        """Return an inert custom holder's attribute values (r42 corr2_1).

        Reads only ``__dict__`` values and ``__slots__`` slot values (via the slot member
        descriptor, never ``getattr`` -- so no property getter and no ``__getattr__`` fires).
        """

        children: list[Any] = []
        try:
            instance_dict = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            instance_dict = None
        if isinstance(instance_dict, dict):
            children.extend(instance_dict.values())
        for klass in type(value).__mro__:
            slots = klass.__dict__.get("__slots__")
            if slots is None:
                continue
            if isinstance(slots, str):
                slots = (slots,)
            try:
                slot_names = list(slots)
            except TypeError:
                continue
            for slot in slot_names:
                if not isinstance(slot, str) or slot in ("__dict__", "__weakref__"):
                    continue
                getter = getattr(klass.__dict__.get(slot), "__get__", None)
                if getter is None:
                    continue
                try:
                    children.append(getter(value, klass))
                except (AttributeError, TypeError, ValueError):
                    continue
        return children

    @staticmethod
    def _is_trusted_leaf_class(klass: type) -> bool:
        """Return whether the class-surface edge treats ``klass`` as a trusted leaf (r53 corr).

        ``object`` / ``type`` and any class whose OWN raw ``__module__`` roots in
        :data:`_CUSTOM_HOLDER_SKIP_MODULES` (torch / numpy / stdlib runtime internals) OR in
        :data:`_STDLIB_CLASS_LEAF_MODULES` (r56 amb_1: the WHOLE stdlib, structurally, not a
        hand-maintained denylist) are leaves -- their class dicts are implementation noise,
        mirroring the instance-side module gate. The r45 removal of ``collections`` from the
        INSTANCE-side skip set (so ``deque`` / ``UserList`` contents descend) must not walk
        stdlib CLASS dicts: ``collections.abc`` ABCs in a container subclass's MRO carry
        ``_abc_impl`` whose caches are process-global registries (the ambient escape).
        Instance-side descent never needed the stdlib class-dict surface; a generator
        monkeypatched ONTO a stdlib class object is shared runtime state, the same
        documented residual family as a shared module global. ``__main__`` is explicitly
        NOT a leaf even though ``sys.stdlib_module_names`` lists it -- a model class defined
        in a script/notebook must keep its r53 class-attribute coverage. ``__module__`` is
        read from the RAW class dict via the base ``type`` getset
        (``type.__dict__["__dict__"]``), so a hostile metaclass property never fires. An
        unreadable or non-string ``__module__`` fails toward WALKING the class -- the walk
        is inert, so the unprovable case errs toward MORE coverage, never code execution.
        """

        if klass is object or klass is type:
            return True
        try:
            raw_dict = type.__dict__["__dict__"].__get__(klass)
            module = raw_dict.get("__module__")
        except Exception:
            return False
        if not isinstance(module, str):
            return False
        root = module.split(".", 1)[0]
        return root in _CUSTOM_HOLDER_SKIP_MODULES or root in _STDLIB_CLASS_LEAF_MODULES

    @staticmethod
    def _shared_namespace_dict_ids() -> frozenset[int]:
        """Identity keys of every loaded module's ``__dict__`` (r55 C6 exclusion set).

        A function's ``__globals__`` / ``__builtins__`` and any other reference
        into a live module namespace are SHARED state, not model state: expanding
        one would walk every framework's globals from a single model attribute.
        Excluding them by ``__dict__`` IDENTITY (never by name heuristics) makes
        "a generator held in a shared module global, drawn on a pre-existing
        non-hooked thread" the explicit documented residual of the inert sweep.
        """

        ids = set()
        for module in list(_sys_module.modules.values()):
            module_dict = getattr(module, "__dict__", None)
            if module_dict is not None:
                ids.add(id(module_dict))
        return frozenset(ids)

    @staticmethod
    def _inert_gc_children(value: Any, shared_namespace_ids: frozenset[int]) -> list[Any]:
        """Return a node's AUTHORITATIVE inert reference edges (r55 C6, corr_2/corr_4).

        ``gc.get_referents(value)`` runs CPython ``tp_traverse`` -- pure C field
        enumeration that NEVER executes Python: no property getter, descriptor
        ``__get__``, ``__getattr__``, or user callable can fire (probed: a hostile
        ``@property``/``__getattr__`` counter stays at zero). Unlike the replaced
        hand-maintained callable-interior vocabulary (closure/defaults/kwdefaults/
        partial/property/...bound-method fields), the traverse exposes EVERY inert
        reference field an object type declares -- ``__annotations__`` (r54
        corr_2), ``functools`` wrapper ``__wrapped__`` chains (r54 corr_4),
        ``__dict__``, slots, cells -- so a new hiding field is unreachable only if
        CPython itself cannot reach it for garbage collection. This is a ROOTED
        per-object enumerator feeding the existing cycle-guarded, node-capped
        model walk -- NOT a process-wide ``gc.get_objects()`` scan.

        Dropped edges are exactly the two documented exclusion families: referents
        whose identity is a loaded module's ``__dict__`` (shared namespaces --
        ``shared_namespace_ids``) and instances of the justified
        :data:`_GC_EXPANSION_LEAF_TYPES`. Everything else is enqueued and handled
        by the sweep's typed branches (a referent dict via the Mapping protocol, a
        referent class via the trusted-leaf-gated class-surface branch, a referent
        RNG via the digest).
        """

        if isinstance(value, _GC_EXPANSION_LEAF_TYPES):
            return []
        children: list[Any] = []
        for referent in _gc_module.get_referents(value):
            if id(referent) in shared_namespace_ids:
                continue
            if isinstance(referent, _GC_EXPANSION_LEAF_TYPES):
                continue
            children.append(referent)
        return children

    @staticmethod
    def _exposes_queue_protocol(value: Any) -> bool:
        """Return whether ``value``'s TYPE implements the standard queue protocol (r45 hon1_1).

        Duck-typed on the CLASS (never the instance, so no property getter or ``__getattr__``
        side effect fires): ``get`` + ``put`` + (``qsize`` or ``empty``) as callables. This
        matches ``queue.Queue`` / ``LifoQueue`` / ``PriorityQueue`` (an inspectable ``.queue``
        deque) AND ``queue.SimpleQueue`` / ``multiprocessing.Queue`` / any future opaque queue
        (no non-mutating buffer) by construction -- NOT a concrete ``SimpleQueue`` type list.
        Ordinary ``Mapping`` / ``Collection`` holders are caught by earlier descent branches and
        never reach this predicate.
        """

        cls = type(value)
        if not (callable(getattr(cls, "get", None)) and callable(getattr(cls, "put", None))):
            return False
        return callable(getattr(cls, "qsize", None)) or callable(getattr(cls, "empty", None))

    @staticmethod
    def _opaque_queue_provably_empty(value: Any) -> bool:
        """Return whether an opaque queue is NON-MUTATINGLY provably empty (r47 hon1_2).

        ``queue.SimpleQueue`` / ``multiprocessing.Queue`` expose no non-mutating payload snapshot
        (``.get`` would DRAIN), so their contents cannot be inventoried. But ``empty()`` and
        ``qsize()`` are NON-MUTATING (probed), so a queue they report EMPTY provably holds no
        generator and can safely stay VERIFIED. True IFF ``empty()`` is exactly ``True`` (authoritative
        non-empty when it is exactly ``False``), else ``qsize()`` is integer ``0``. Any exception,
        non-bool ``empty()``, non-int/negative ``qsize()``, or unsupported value fails closed to
        ``False`` -- ``mp.Queue`` can raise/flake, so a non-empty or unknown queue is NEVER read as
        empty.
        """

        empty_fn = getattr(value, "empty", None)
        if callable(empty_fn):
            try:
                empty_result = empty_fn()
            except Exception:
                empty_result = None
            if empty_result is True:
                return True
            if empty_result is False:
                return False
        qsize_fn = getattr(value, "qsize", None)
        if callable(qsize_fn):
            try:
                size = qsize_fn()
            except Exception:
                return False
            if type(size) is int and size == 0:
                return True
        return False

    def _sweep_model_generators(self) -> list[tuple[Any, str]]:
        """Digest numpy/`random` generators reachable from the MODEL's submodule attributes.

        This is the CHEAP thread-independent belt (model attributes only), NOT a
        process-wide ``gc.get_objects()`` scan (the r39-draft GC-wide inventory cost
        ~900 ms/capture, perturbed the peak-memory bracket, and over-trigger-risked
        unrelated generators -- removed for cause and never reintroduced).

        r41 hon1_2 / r45 hon1_1: the sweep is an ITERATIVE, cycle-safe recursion by
        container PROTOCOL (not a fixed concrete-type list) -- every submodule ``__dict__``
        value, descending through every ``collections.abc.Mapping`` (KEYS and VALUES) and
        every non-leaf ``collections.abc.Collection`` (elements), so ``self.pool = [gen]``,
        ``{"g": gen}``, ``[[gen]]``, a dict-key generator, and a generator inside a ``deque``
        / ``ChainMap`` / ``UserList`` / ``UserDict`` / namedtuple / custom ``Sequence`` or
        ``Mapping`` are all digested. Descent is gated on ``Collection`` (Sized), NEVER bare
        ``Iterable``, so a one-shot generator / ``map`` / ``itertools`` attribute is never
        consumed. A safe queue (``queue.Queue`` / ``LifoQueue`` / ``PriorityQueue``) is reached
        through a non-mutating snapshot of its internal ``.queue`` deque; an opaque queue
        (``SimpleQueue`` / ``mp.Queue``) with no inspectable buffer fails closed to INCOMPLETE
        (``inventory_opaque_container``). r42 corr2_1: it ADDITIONALLY recurses into inert
        CUSTOM object holders (``self.holder.rng``) through their ``__dict__`` / ``__slots__``
        values, skipping tensors, ndarrays, callables, modules, and stdlib/torch/numpy
        implementation objects. r51 hon1_1: an ``nn.Module`` is NO LONGER a hard leaf -- every
        reachable module (REGISTERED via the top ``model.modules()`` loop AND an UNREGISTERED
        submodule held in a plain attribute / ``list`` / ``dict`` / nested container / custom
        holder) is descended through the same ``__dict__`` / ``__slots__`` protocol, so a numpy
        Generator behind an unregistered submodule is witnessed (registered ids are premarked so
        there is no double-walk). The former
        ``budget = 2000`` early return truncated SILENTLY (a generator on a late
        submodule of any >~120-module model was missed -> false VERIFIED); the
        replacement :data:`_INVENTORY_NODE_CAP` is defensive only -- exhaustion flags
        ``inventory_budget_exhausted`` (INCOMPLETE), never a silent partial snapshot.

        The realistic hon1_1/corr2_2 CROSS-THREAD draw (an externally-held numpy
        Generator drawn on an in-window helper thread) is caught by the
        ``threading.setprofile`` receiver classifier; owner-thread instance draws and
        the immutable ``datetime`` readers by the ``sys.setprofile`` classifier;
        unseeded construction and Python ``random`` by the construction/class patches.
        This digest is the thread-independent belt for a generator the model itself
        HOLDS (caught even on a pre-existing thread; r42 corr2_1 extends this to a
        generator behind a custom holder attribute; r51 hon1_1 to a generator behind an
        UNREGISTERED submodule).

        r53 corr_1/corr_2/F1: the walk's structural invariant is REACHABILITY --
        it follows EVERY reference edge that can be followed WITHOUT executing
        user-defined code. Beyond the instance/container surfaces above it walks:
        (1) class-MRO ``__dict__`` surfaces of user-defined classes (raw mappingproxy
        reads through the base ``type`` getsets -- the descriptor protocol only fires
        on getattr, NEVER on a ``values()`` read -- so a class-descriptor-held
        submodule/generator and a plain class-attribute generator are reached;
        torch/stdlib/numpy implementation classes are trusted leaves);
        (2) ``weakref.ref`` / ``WeakMethod`` referents through ONE base-C dereference
        (``weakref.ref.__call__``, immune to hostile ``__call__`` overrides; weak
        CONTAINERS descend via the ordinary Mapping/Collection protocols);
        (3) r55 C6 (corr_2/corr_4): every OTHER node's reference edges through the
        AUTHORITATIVE ``gc.get_referents`` enumerator (:meth:`_inert_gc_children`)
        -- CPython ``tp_traverse``, pure C, zero Python executed -- minus the two
        documented exclusion families (loaded-module ``__dict__`` identities and
        :data:`_GC_EXPANSION_LEAF_TYPES`). This SUBSUMES the r53 hand-maintained
        callable-interior vocabulary (closure cells, defaults, kwdefaults,
        ``partial``/``property``/``static``/``classmethod`` fields, bound-method
        ``__func__``/``__self__``, callable-instance ``__dict__``/``__slots__``)
        and closes its whole drift class: ``__annotations__`` (r54 corr_2),
        ``functools`` wrapper ``__wrapped__`` chains (r54 corr_4), and any future
        inert field a type declares are reached by construction, not by table
        maintenance. The walk never invokes a property, a descriptor ``__get__``,
        ``__getattr__``, or any user callable. Residual (contract s11): a generator
        reachable ONLY BY EXECUTING USER CODE (a property/descriptor ``__get__``
        body, ``__getattr__``, or a callable's return value) or held ONLY in a
        SHARED module-global namespace, drawn on a PRE-EXISTING (non-hooked)
        thread.
        """

        snapshots: list[tuple[Any, str]] = []
        model = self._model
        modules = getattr(model, "modules", None)
        if not callable(modules):
            return snapshots
        try:
            # r55 C6: shared-namespace exclusion set for the gc-referent fallback,
            # computed ONCE per sweep (bounded by loaded-module count).
            shared_namespace_ids = self._shared_namespace_dict_ids()
            pending: list[Any] = []
            for module in modules():
                pending.extend(getattr(module, "__dict__", {}).values())
                # r53 corr_1 (class-surface edge): the registered module's CLASS is a holder
                # surface too -- a class descriptor or a class-attribute generator on the
                # user model class is invisible to the instance-``__dict__`` seed above.
                pending.append(type(module))
            # r51 hon1_1: nn.Module is now a recursable holder (``_is_recursable_custom_holder``),
            # so a REGISTERED submodule reached during descent would be re-walked even though the
            # top loop already seeded its ``__dict__``. Premark every registered module id so the
            # ``id in seen_container_ids`` guards skip that re-walk -- a coverage-NEUTRAL dedup (a
            # generator directly on a registered module is seeded and digested from its ``__dict__``
            # before the module OBJECT is ever reached) that kills the ~2x double-walk. UNREGISTERED
            # submodules are absent from ``modules()`` and stay un-premarked, so they ARE descended.
            seen_container_ids: set[int] = {id(module) for module in modules()}
            visited_nodes = 0
            while pending:
                value = pending.pop()
                visited_nodes += 1
                if visited_nodes > _INVENTORY_NODE_CAP:
                    self._flag_uncertain("inventory_budget_exhausted")
                    return snapshots
                # r53 corr_1 (class-surface edge): a CLASS node contributes its raw
                # ``__dict__`` values -- a mappingproxy read reaches descriptor OBJECTS
                # (class-descriptor-held submodules/generators) and plain class-attribute
                # generators WITHOUT ever firing ``__get__`` (the descriptor protocol only
                # fires on getattr, never on a ``values()`` read) -- plus its remaining MRO
                # classes. Reads go through the base ``type`` getsets so a hostile
                # metaclass property on ``__dict__``/``__mro__`` never fires;
                # torch/stdlib/numpy implementation classes are trusted leaves. Per-class
                # dedup keeps this trivially bounded (distinct user classes, not modules).
                if isinstance(value, type):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    if self._is_trusted_leaf_class(value):
                        continue
                    try:
                        raw_dict = type.__dict__["__dict__"].__get__(value)
                        mro = type.__dict__["__mro__"].__get__(value)
                    except Exception:
                        continue  # inert read failed: skip posture (nothing executed)
                    pending.extend(raw_dict.values())
                    pending.extend(base for base in mro if base is not value)
                    continue
                # r53 corr_2 (weakref edge): dereference ONCE at base-C level
                # (``weakref.ref.__call__`` -- immune to a hostile subclass ``__call__``
                # override) and enqueue the live referent; a dead ref contributes nothing.
                # Covers ``WeakMethod`` (its base ref points at ``__self__``); weak
                # CONTAINERS (``WeakSet``/``WeakValueDictionary``/``WeakKeyDictionary``)
                # already descend via the Mapping/Collection protocol branches below. A
                # SUBCLASS instance additionally contributes its own inert ``__dict__`` /
                # ``__slots__`` values and its class surface. A deref failure fails closed
                # (``inventory_state_read_failed``), never reads as no-referent.
                if isinstance(value, _weakref_module.ref):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    try:
                        referent = _weakref_module.ref.__call__(value)
                    except Exception:
                        self._flag_uncertain("inventory_state_read_failed")
                        continue
                    if referent is not None:
                        pending.append(referent)
                    if type(value) is not _weakref_module.ref:
                        pending.extend(self._custom_holder_children(value))
                        pending.append(type(value))
                    continue
                # r45 hon1_1: descend by container PROTOCOL, not a fixed concrete-type list, so a
                # model-held generator inside a ``deque`` / ``ChainMap`` / ``UserList`` /
                # ``UserDict`` / namedtuple / custom ``Sequence`` / custom ``Mapping`` is reached
                # (the r44 hon1_1 gap). ``Mapping`` first (a ``Mapping`` is also a ``Collection``);
                # then any non-leaf ``Collection``. The gate is ``Collection`` (Sized+Iterable+
                # Container), NEVER bare ``Iterable`` -- a generator / ``map`` / ``zip`` /
                # ``itertools`` object is ``Iterable`` but not ``Collection``, so it is NEVER
                # iterated and a one-shot iterator is never consumed / corrupted.
                if isinstance(value, Mapping):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    pending.extend(value.keys())
                    pending.extend(value.values())
                    # r47 hon1_1: a Mapping that is ALSO a custom (non-stdlib) inert holder can
                    # carry a generator as its OWN attribute (``self.rng`` on a ``UserDict`` /
                    # custom ``Mapping`` subclass), which is NOT among its keys/values. Walk its
                    # own ``__dict__`` / ``__slots__`` too so the held generator is digest-
                    # witnessed. ``_is_recursable_custom_holder`` excludes stdlib/torch/numpy, so a
                    # plain ``dict`` is never double-walked; cycle-guarded via ``seen_container_ids``
                    # (already added above), node-capped, and never invokes a property/``__getattr__``.
                    if self._is_recursable_custom_holder(value):
                        pending.extend(self._custom_holder_children(value))
                        pending.append(type(value))  # r53 corr_1: class surface of the subclass
                    continue
                if isinstance(value, Collection) and not isinstance(value, _INVENTORY_LEAF_TYPES):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    pending.extend(value)
                    # r47 hon1_1: same dual-walk for a non-leaf Collection that is ALSO a custom
                    # inert holder -- ``self.rng`` on a ``Sequence`` / ``MutableSequence`` /
                    # ``UserList`` / ``__slots__`` collection subclass, not among its elements.
                    if self._is_recursable_custom_holder(value):
                        pending.extend(self._custom_holder_children(value))
                        pending.append(type(value))  # r53 corr_1: class surface of the subclass
                    continue
                if id(value) in self._exempt_ids:
                    continue
                try:
                    digest = self._digest_rng_instance(value)
                except _NotADigestableRng:
                    if id(value) in seen_container_ids:
                        continue
                    # r45 hon1_1: a model-held generator can sit inside a QUEUE whose buffer is a
                    # non-mutating-inspectable deque (``queue.Queue`` / ``LifoQueue`` /
                    # ``PriorityQueue`` expose ``.queue``). Snapshot that deque non-destructively.
                    # A queue with NO inspectable buffer (``SimpleQueue`` / ``mp.Queue`` / any
                    # opaque queue) cannot be inventoried without draining it -> FAIL CLOSED to
                    # INCOMPLETE (``inventory_opaque_container``) rather than reading as
                    # no-consumption. Checked BEFORE the gc-referent fallback so an opaque
                    # queue still fail-closes instead of being walked into its
                    # threading/connection internals.
                    if self._exposes_queue_protocol(value):
                        seen_container_ids.add(id(value))
                        inner = getattr(value, "queue", None)
                        if isinstance(inner, Collection) and not isinstance(
                            inner, _INVENTORY_LEAF_TYPES
                        ):
                            pending.append(inner)
                        else:
                            # r49 hon1_1: the emptiness proof is a MONITOR-INTERNAL read. Bracket
                            # ONLY the probe so a clock touched TRANSITIVELY by ``mp.Queue.empty()``
                            # (via ``multiprocessing.connection``) is self-suppressed at the
                            # ``_mark`` choke point, not mis-marked as a model host read. The
                            # ``_flag_uncertain`` branch stays OUTSIDE the bracket so the NON-EMPTY
                            # opaque-queue INCOMPLETE residual is preserved.
                            with self._monitor_internal_probe():
                                provably_empty = self._opaque_queue_provably_empty(value)
                            if provably_empty:
                                # r47 hon1_2: a non-mutatingly PROVABLY-EMPTY opaque queue
                                # (``SimpleQueue`` / ``mp.Queue``) cannot hold a generator, so it
                                # stays VERIFIED -- no over-trigger for a deterministic model that
                                # merely holds an empty queue.
                                pass
                            else:
                                # A NON-EMPTY or unknown opaque queue fails closed (a queue-held
                                # generator drawn on a pre-existing worker would otherwise be
                                # unwitnessed).
                                self._flag_uncertain("inventory_opaque_container")
                        continue
                    # r55 C6 (corr_2/corr_4): AUTHORITATIVE inert fallback. Every node that
                    # is neither a container, a weak reference, a class, a digestable RNG,
                    # nor a queue expands through ``gc.get_referents`` (CPython
                    # ``tp_traverse`` -- zero Python executed) minus the documented
                    # shared-namespace/leaf exclusions. This replaces BOTH the r42
                    # custom-holder recursion and the r53 hand-listed callable-interior
                    # vocabulary at this terminal position: ``__dict__``/``__slots__``
                    # values, closure cells, defaults, kwdefaults, ``__annotations__``,
                    # ``functools`` wrapper ``__wrapped__`` chains, bound-method
                    # ``__func__``/``__self__``, and any future inert field are enqueued by
                    # construction into the same cycle-guarded, node-capped walk (a
                    # referent dict descends via the Mapping branch; a referent class via
                    # the trusted-leaf-gated class branch; a referent generator is
                    # digested). The before/after digest then catches a draw on ANY thread
                    # once the generator is FOUND.
                    seen_container_ids.add(id(value))
                    pending.extend(self._inert_gc_children(value, shared_namespace_ids))
                    continue
                except Exception:
                    self._flag_uncertain("inventory_state_read_failed")
                    continue
                snapshots.append((value, digest))
        except Exception:
            self._flag_uncertain("inventory_scan_failed")
        return snapshots

    @staticmethod
    def _digest_rng_instance(holder: Any) -> str:
        if isinstance(holder, np.random.Generator):
            return repr(holder.bit_generator.state)
        if isinstance(holder, np.random.RandomState):
            return repr(holder.get_state())
        # r41 (Sol): a BARE model-held BitGenerator (``self.bg = PCG64(...)`` drawn
        # through a wrapping Generator) advances its own ``state``; digest it directly
        # so the registry's BitGenerator claim is digest-true.
        if isinstance(holder, np.random.BitGenerator):
            return repr(holder.state)
        if isinstance(holder, random.Random):
            try:
                state = holder.getstate()
            except NotImplementedError:
                # r55 corr_1: ``random.SystemRandom`` (and any stateless ``Random``
                # subclass following its documented protocol) INTENTIONALLY has no
                # digestible state -- ``getstate()`` raises ``NotImplementedError``
                # by design. Possession of an UNDRAWN stateless engine is not
                # nondeterminism: classify monitored-not-digestible (the sweep then
                # walks it structurally like any holder) instead of letting the
                # generic inventory error path over-trigger
                # ``inventory_state_read_failed`` on a deterministic model. Actual
                # draws stay witnessed by the class-method patches on
                # ``random.SystemRandom.{random,getrandbits,randbytes}``. Any OTHER
                # exception from ``getstate()`` (a genuinely broken state read)
                # still propagates to the fail-closed inventory error path.
                raise _NotADigestableRng from None
            return repr(state)
        raise _NotADigestableRng

    # -- context protocol ---------------------------------------------------------

    def __enter__(self) -> HostRngMonitorResult:
        try:
            self._tl_globals_ids = _torchlens_module_globals_ids()
            self._exempt_ids = frozenset(id(item) for item in _rng_exempt_instances())
            # rng_primitive: Python RNG class primitives (instances + subclasses +
            # the bare C base for ``_random.Random()``).
            for holder in (random.Random, random.SystemRandom, _c_random_module.Random):
                for method_name in ("random", "getrandbits", "randbytes"):
                    if method_name in vars(holder):
                        self._patch_attr(
                            holder,
                            method_name,
                            self._instance_method_wrapper(
                                getattr(holder, method_name),
                                f"{holder.__module__}.{holder.__qualname__}.{method_name}",
                            ),
                        )
            # entropy: OS entropy + the secrets funnel alias + uuid4's feed. Each
            # original is identity-registered BEFORE patching (r41 held-ref layer).
            self._register_held_ref(_os_module.urandom, "os.urandom")
            self._patch_attr(
                _os_module, "urandom", self._entropy_wrapper(_os_module.urandom, "os.urandom")
            )
            if hasattr(_os_module, "getrandom"):
                self._register_held_ref(_os_module.getrandom, "os.getrandom")
                self._patch_attr(
                    _os_module,
                    "getrandom",
                    self._entropy_wrapper(_os_module.getrandom, "os.getrandom"),
                )
            if hasattr(random, "_urandom"):
                self._register_held_ref(random._urandom, "random._urandom")
                self._patch_attr(
                    random,
                    "_urandom",
                    self._entropy_wrapper(random._urandom, "random._urandom"),
                )
            # construction: the modern NumPy generator factory + the writable
            # construction-entropy alias for unseeded BitGenerator construction (E5).
            self._register_held_ref(np.random.default_rng, "np.random.default_rng")
            self._patch_attr(
                np.random,
                "default_rng",
                self._entropy_wrapper(np.random.default_rng, "np.random.default_rng"),
            )
            bit_generator_module = getattr(np.random, "bit_generator", None)
            if bit_generator_module is not None and hasattr(bit_generator_module, "randbits"):
                self._register_held_ref(bit_generator_module.randbits, "np_bit_generator_randbits")
                self._patch_attr(
                    bit_generator_module,
                    "randbits",
                    self._entropy_wrapper(
                        bit_generator_module.randbits, "np_bit_generator_randbits"
                    ),
                )
            # clock: the frozen ``time.*`` readers (thread-independent module patches).
            for clock_name in _CLOCK_COUNTER_NAMES:
                if hasattr(_time_module, clock_name):
                    self._register_held_ref(getattr(_time_module, clock_name), f"time.{clock_name}")
                    self._patch_attr(
                        _time_module,
                        clock_name,
                        self._clock_wrapper(
                            getattr(_time_module, clock_name), f"time.{clock_name}", None
                        ),
                    )
            for clock_name, time_arg_index in _CLOCK_IMPLICIT_NOW:
                if hasattr(_time_module, clock_name):
                    self._register_held_ref(
                        getattr(_time_module, clock_name), f"time.{clock_name}", time_arg_index
                    )
                    self._patch_attr(
                        _time_module,
                        clock_name,
                        self._clock_wrapper(
                            getattr(_time_module, clock_name), f"time.{clock_name}", time_arg_index
                        ),
                    )
            if hasattr(_os_module, "times"):
                self._register_held_ref(_os_module.times, "os.times")
                self._patch_attr(
                    _os_module, "times", self._clock_wrapper(_os_module.times, "os.times", None)
                )
            if _resource_module is not None and hasattr(_resource_module, "getrusage"):
                self._register_held_ref(_resource_module.getrusage, "resource.getrusage")
                self._patch_attr(
                    _resource_module,
                    "getrusage",
                    self._clock_wrapper(_resource_module.getrusage, "resource.getrusage", None),
                )
            # clock: immutable ``datetime`` current readers via c_call identity.
            for receiver, method in _DATETIME_CLOCK_READERS:
                if hasattr(receiver, method):
                    self._clock_ccall_keys[(id(receiver), method)] = (
                        f"datetime.{receiver.__name__}.{method}"
                    )
            # BELT (thread-independent, CHEAP -- model attributes + builtin-container
            # nesting): digest generators the model HOLDS, so a draw on ANY thread
            # (incl. a pre-existing worker) is caught by the before/after state
            # comparison at __exit__. Deliberately NOT a process-wide
            # ``gc.get_objects()`` scan -- the r39-draft GC-wide inventory cost
            # ~900 ms/capture, perturbed the tracemalloc peak, and could over-trigger
            # on unrelated generators (removed for cause). Realistic CROSS-THREAD
            # external draws (hon1_1/corr2_2) are caught by ``threading.setprofile``
            # below; an EXTERNALLY-held generator drawn on a PRE-EXISTING (non-hooked)
            # thread is the documented residual (contract s11), same class as the
            # adversarial draw+state-restore.
            self._generator_states = self._sweep_model_generators()
            # BELT: dual chained profile hooks (owner thread + threads started in-window). These
            # are the r37/base mechanism (base runs them and is fast); the owner hook catches
            # owner-thread numpy Generator instance draws and the immutable ``datetime`` readers,
            # the threading hook catches an in-window helper-thread draw (hon1_1/corr2_2) and
            # records each hooked thread's ident for the escape belt's 3-class gate (r41).
            self._previous_sys_profile = _sys_module.getprofile()
            self._sys_hook = self._make_profile_hook(self._previous_sys_profile)
            _sys_module.setprofile(self._sys_hook)
            self._sys_profile_installed = True
            self._previous_threading_profile = (
                _threading_module.getprofile() if hasattr(_threading_module, "getprofile") else None
            )
            self._threading_hook = self._make_profile_hook(
                self._previous_threading_profile, records_thread_ident=True
            )
            _threading_module.setprofile(self._threading_hook)
            self._threading_profile_installed = True
        except Exception:
            self._flag_uncertain("monitor_install_failed")
        # r41 hon2_1: publish the in-window registry LAST so the escape belt never
        # observes a partially-installed window (cleared FIRST in __exit__).
        global _ACTIVE_MONITOR
        _ACTIVE_MONITOR = self
        return self.result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _ACTIVE_MONITOR
        _ACTIVE_MONITOR = None
        if self._threading_profile_installed:
            try:
                if (
                    hasattr(_threading_module, "getprofile")
                    and _threading_module.getprofile() is not self._threading_hook
                ):
                    self._flag_uncertain("threading_profile_replaced")
                _threading_module.setprofile(self._previous_threading_profile)
            except Exception:
                self._flag_uncertain("threading_profile_restore_failed")
        if self._sys_profile_installed:
            try:
                if _sys_module.getprofile() is not self._sys_hook:
                    self._flag_uncertain("sys_profile_replaced")
                _sys_module.setprofile(self._previous_sys_profile)
            except Exception:
                self._flag_uncertain("sys_profile_restore_failed")
        for restore in reversed(self._restores):
            try:
                restore()
            except Exception:
                self._flag_uncertain("patch_restore_failed")
        for holder, before in self._generator_states:
            try:
                if self._digest_rng_instance(holder) != before:
                    self._mark("model_attribute_generator")
            except Exception:
                self._flag_uncertain("inventory_compare_failed")

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
import os as _os_module
import random
import sys as _sys_module
import threading as _threading_module
import time as _time_module
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import TracebackType
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
    Generator/RandomState/BitGenerator) is witnessed thread-independently by the state
    digest, and an unseeded construction on any thread by the module/class patches.
    The residual is only an EXTERNALLY-HELD generator drawn on a pre-existing (non-hooked)
    thread, of the same class as the adversarial draw+``state`` RESTORE (E4) -- a
    self-cleaning sequence no py<=3.11 mechanism can witness -- documented in contract
    s11, NOT a blanket ceiling: the r38 draft's thread-presence INCOMPLETE over-triggered
    every capture running alongside a benign background thread (DataLoader/Jupyter/
    pytest), so it is intentionally not applied. Future all-thread coverage is
    ``sys.monitoring`` (PEP 669, 3.12+, interpreter-wide).
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

    # -- helpers -----------------------------------------------------------------

    def _mark(self, channel: str) -> None:
        self.result.channels.add(channel)

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
        # Immutable ``datetime`` current-clock readers, matched by (receiver, method).
        name = getattr(arg, "__name__", None)
        if name is not None:
            channel = self._clock_ccall_keys.get((id(receiver), name))
            if channel is not None:
                self._mark(channel)

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

    def _sweep_model_generators(self) -> list[tuple[Any, str]]:
        """Digest numpy/`random` generators reachable from the MODEL's submodule attributes.

        This is the CHEAP thread-independent belt (model attributes only), NOT a
        process-wide ``gc.get_objects()`` scan (the r39-draft GC-wide inventory cost
        ~900 ms/capture, perturbed the peak-memory bracket, and over-trigger-risked
        unrelated generators -- removed for cause and never reintroduced).

        r41 hon1_2: the sweep is an ITERATIVE, cycle-safe, FULL builtin-container
        recursion -- every submodule ``__dict__`` value, descending through ``list`` /
        ``tuple`` / ``set`` / ``frozenset`` elements and ``dict`` KEYS and VALUES (so
        ``self.pool = [gen]``, ``{"g": gen}``, ``[[gen]]``, and a dict-key generator are
        all digested). It does NOT walk arbitrary custom-object attributes, tensors,
        ndarray internals, or modules outside ``model.modules()``. The former
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
        HOLDS (caught even on a pre-existing thread). Residuals (contract s11): an
        EXTERNALLY-held generator drawn on a PRE-EXISTING (non-hooked) thread, and a
        generator behind a CUSTOM object attribute (``self.cfg.gen`` -- not swept;
        hooked-thread draws of it stay receiver-classified).
        """

        snapshots: list[tuple[Any, str]] = []
        model = self._model
        modules = getattr(model, "modules", None)
        if not callable(modules):
            return snapshots
        try:
            pending: list[Any] = []
            for module in modules():
                pending.extend(getattr(module, "__dict__", {}).values())
            seen_container_ids: set[int] = set()
            visited_nodes = 0
            while pending:
                value = pending.pop()
                visited_nodes += 1
                if visited_nodes > _INVENTORY_NODE_CAP:
                    self._flag_uncertain("inventory_budget_exhausted")
                    return snapshots
                if isinstance(value, (list, tuple, set, frozenset)):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    pending.extend(value)
                    continue
                if isinstance(value, dict):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    pending.extend(value.keys())
                    pending.extend(value.values())
                    continue
                if id(value) in self._exempt_ids:
                    continue
                try:
                    digest = self._digest_rng_instance(value)
                except _NotADigestableRng:
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
            return repr(holder.getstate())
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

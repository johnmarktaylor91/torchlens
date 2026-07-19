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

import os as _os_module
import random
import sys as _sys_module
import threading as _threading_module
import time as _time_module
from collections.abc import Callable
from types import TracebackType
from typing import Any, Dict, List, TypeVar, cast

import _random as _c_random_module

import numpy as np
import torch

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
# r37 hon1_2 -- host-nondeterminism channel monitor (the conservative ceiling).
#
# The global-engine snapshots above cover only the two REPLAYABLE engines (module
# ``random``, legacy ``np.random``). Every other host entropy/clock/RNG-instance
# channel is monitored here over a FROZEN vocabulary; a positive touch permanently
# ceilings the capture (``host_rng_consumed=True`` with NO identifiable seed ->
# every replay UNVERIFIABLE + NOT_APPLICABLE). Monitor uncertainty (install/chain/
# restore failure) is itself recorded and downgrades capture completeness -- it
# never reads as "no consumption". Absence of a touch claims nothing beyond the
# named vocabulary (residual tail: direct /dev/urandom file reads, user C-extension
# RNGs, ctypes -- outside any sane monitor).
# ======================================================================================


class HostRngMonitorResult:
    """Outcome of one capture-scoped host-nondeterminism monitoring window."""

    __slots__ = ("channels", "uncertain")

    def __init__(self) -> None:
        self.channels: set[str] = set()
        self.uncertain: bool = False


_CLOCK_CHANNEL_NAMES = (
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
"""Frozen monitored clock vocabulary (r37): any in-forward user read ceilings."""


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
    """Replayable global singletons + TorchLens's private barcode RNG (identity-exempt)."""

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
    return tuple(exempt)


class host_nondeterminism_monitor:
    """Context manager installing the four-layer host-RNG/entropy/clock monitor.

    Layers (agreed r37 design):

    1. replayable global engines -- NOT handled here (the snapshot bracket at the
       call site keeps its unchanged seeded-reproduction semantics);
    2. scoped patches over the frozen channel vocabulary: ``random.Random`` /
       ``random.SystemRandom`` draw primitives (class-level, so private instances
       and subclasses are caught; the module-global engine binds its methods at
       import time and bypasses them structurally), ``os.urandom`` (+
       ``os.getrandom`` where present) and the import-time ``random._urandom``
       alias (the ``secrets.token_*`` funnel), the ``np.random.default_rng``
       factory, and the full clock family;
    3. a model-attribute generator state sweep (belt for layer 4);
    4. a capture-scoped chained ``sys.setprofile`` ``c_call`` classifier marking
       any draw whose bound receiver is a NumPy ``Generator`` / ``BitGenerator`` /
       ``RandomState`` or a C-level ``_random.Random``, identity-exempting the two
       replayable global singletons and TorchLens's private barcode RNG.

    Entropy/instance channels mark from ANY thread (a forward offloading a draw to
    a helper thread is still a touch); clock channels mark only on the owner thread
    (unrelated daemon threads must not ceiling a deterministic capture). Install,
    chain, or restore failure sets ``uncertain`` -- the caller must downgrade
    capture completeness, never report no-consumption.
    """

    def __init__(self, model: Any = None) -> None:
        self.result = HostRngMonitorResult()
        self._model = model
        self._restores: list[Callable[[], None]] = []
        self._owner_thread = _threading_module.get_ident()
        self._previous_profile: Any = None
        self._profile_installed = False
        self._hook: Any = None
        self._generator_states: list[tuple[Any, Any]] = []
        self._tl_globals_ids: frozenset[int] = frozenset()
        self._exempt_ids: frozenset[int] = frozenset()

    # -- helpers -----------------------------------------------------------------

    def _mark(self, channel: str) -> None:
        self.result.channels.add(channel)

    def _patch_attr(self, holder: Any, name: str, wrapper: Any) -> None:
        original = getattr(holder, name)
        setattr(holder, name, wrapper)

        def _restore(holder: Any = holder, name: str = name, original: Any = original) -> None:
            if getattr(holder, name, None) is not wrapper:
                # Someone replaced our patch mid-window: restoration cannot be
                # proven exact -> uncertainty (fail closed), restore anyway.
                self.result.uncertain = True
            setattr(holder, name, original)

        self._restores.append(_restore)

    def _entropy_wrapper(self, original: Any, channel: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self._mark(channel)
            return original(*args, **kwargs)

        return wrapper

    def _clock_wrapper(self, original: Any, channel: str) -> Any:
        tl_ids = self._tl_globals_ids
        owner = self._owner_thread

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _threading_module.get_ident() == owner:
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

    def _profile_hook(self, frame: Any, event: str, arg: Any) -> Any:
        try:
            if event == "c_call":
                receiver = getattr(arg, "__self__", None)
                if (
                    receiver is not None
                    and id(receiver) not in self._exempt_ids
                    and isinstance(
                        receiver,
                        (
                            _c_random_module.Random,
                            np.random.Generator,
                            np.random.RandomState,
                            np.random.BitGenerator,
                        ),
                    )
                ):
                    self._mark("c_rng_instance_draw")
        except Exception:
            self.result.uncertain = True
        previous = self._previous_profile
        if previous is not None:
            try:
                previous(frame, event, arg)
            except Exception:
                self.result.uncertain = True

    def _sweep_model_generators(self) -> list[tuple[Any, Any]]:
        states: list[tuple[Any, Any]] = []
        model = self._model
        modules = getattr(model, "modules", None)
        if not callable(modules):
            return states
        try:
            budget = 1000
            for module in modules():
                for value in list(getattr(module, "__dict__", {}).values()):
                    budget -= 1
                    if budget <= 0:
                        return states
                    try:
                        if isinstance(value, np.random.Generator):
                            states.append((value, repr(value.bit_generator.state)))
                        elif isinstance(value, np.random.RandomState):
                            states.append((value, repr(value.get_state())))
                        elif isinstance(value, random.Random):
                            states.append((value, value.getstate()))
                    except Exception:
                        self.result.uncertain = True
        except Exception:
            self.result.uncertain = True
        return states

    # -- context protocol ---------------------------------------------------------

    def __enter__(self) -> HostRngMonitorResult:
        try:
            self._tl_globals_ids = _torchlens_module_globals_ids()
            self._exempt_ids = frozenset(id(item) for item in _rng_exempt_instances())
            # Layer 2a: Python RNG class primitives (instances + subclasses).
            for method_name in ("random", "getrandbits", "randbytes"):
                if hasattr(random.Random, method_name):
                    self._patch_attr(
                        random.Random,
                        method_name,
                        self._instance_method_wrapper(
                            getattr(random.Random, method_name),
                            f"random.Random.{method_name}",
                        ),
                    )
                if method_name in vars(random.SystemRandom):
                    self._patch_attr(
                        random.SystemRandom,
                        method_name,
                        self._instance_method_wrapper(
                            getattr(random.SystemRandom, method_name),
                            f"random.SystemRandom.{method_name}",
                        ),
                    )
            # Layer 2b: OS entropy + the secrets funnel alias + uuid4's feed.
            self._patch_attr(
                _os_module, "urandom", self._entropy_wrapper(_os_module.urandom, "os.urandom")
            )
            if hasattr(_os_module, "getrandom"):
                self._patch_attr(
                    _os_module,
                    "getrandom",
                    self._entropy_wrapper(_os_module.getrandom, "os.getrandom"),
                )
            if hasattr(random, "_urandom"):
                # ``random.py`` imports ``os.urandom`` at import time; SystemRandom
                # and ``secrets.token_bytes`` route through THIS alias, not the
                # patched ``os.urandom`` attribute (measured, exp3).
                self._patch_attr(
                    random,
                    "_urandom",
                    self._entropy_wrapper(random._urandom, "random._urandom"),
                )
            # Layer 2c: the modern NumPy generator factory.
            self._patch_attr(
                np.random,
                "default_rng",
                self._entropy_wrapper(np.random.default_rng, "np.random.default_rng"),
            )
            # Layer 2d: the frozen clock family (owner-thread, TL-ownership-excluded).
            for clock_name in _CLOCK_CHANNEL_NAMES:
                if hasattr(_time_module, clock_name):
                    self._patch_attr(
                        _time_module,
                        clock_name,
                        self._clock_wrapper(
                            getattr(_time_module, clock_name), f"time.{clock_name}"
                        ),
                    )
            # Layer 3: model-attribute generator state sweep (belt for layer 4).
            self._generator_states = self._sweep_model_generators()
            # Layer 4: chained c_call classifier on the owner thread. The bound
            # method is materialized ONCE so the exit identity check compares the
            # exact installed object (attribute access would rebind per access).
            self._hook = self._profile_hook
            self._previous_profile = _sys_module.getprofile()
            _sys_module.setprofile(self._hook)
            self._profile_installed = True
        except Exception:
            self.result.uncertain = True
        return self.result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._profile_installed:
            try:
                if _sys_module.getprofile() is not self._hook:
                    # The hook was replaced mid-forward: coverage between the
                    # replacement and now is unknowable.
                    self.result.uncertain = True
                _sys_module.setprofile(self._previous_profile)
            except Exception:
                self.result.uncertain = True
        for restore in reversed(self._restores):
            try:
                restore()
            except Exception:
                self.result.uncertain = True
        for holder, before in self._generator_states:
            try:
                if isinstance(holder, np.random.Generator):
                    after: Any = repr(holder.bit_generator.state)
                elif isinstance(holder, np.random.RandomState):
                    after = repr(holder.get_state())
                else:
                    after = holder.getstate()
                if after != before:
                    self._mark("model_attribute_generator")
            except Exception:
                self.result.uncertain = True

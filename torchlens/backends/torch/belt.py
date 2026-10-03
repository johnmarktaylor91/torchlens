"""The mechanical BELT: protocol-invisible stale-reference coverage.

Stage-2 safety net, part 2 (design-review verdict). A small class of wrapped torch
functions is invisible to EVERY ``TorchFunctionMode`` — their C
implementations never enter the override-protocol dispatch (measured: zero
callbacks), so a stale pre-wrap reference to one of them produces NO signal
anywhere: no mode callback, no aten event, and with the shipped-default
escape detector off the op would silently vanish from the trace. The rescue
re-run cannot fix what nothing detects, so these functions keep the targeted
stale-reference patching (module-level attributes, the measured reachable
holder class) after the broad crawler's deletion.

The belt membership is DERIVED MECHANICALLY per build, never hand-listed:

1. Candidates are every ``ORIG_TORCH_FUNCS`` entry whose ORIGINAL callable is
   absent from torch's own override registries
   (``get_overridable_functions`` + ``get_testing_overrides``) — the static
   not-mode-visible superset.
2. Each candidate with a registered probe recipe is CALLED under a counting
   ``TorchFunctionMode`` (with logging paused). A successful tensor-touching
   call with zero callbacks is a belt member; a call that fires the mode is
   excluded (this resolves build-dependent visibility such as ``from_file``
   automatically per version).
3. Candidates without a runnable probe are DISCLOSED in the report — by name
   (``unprobed_candidates``) and count — never silently classified. Probing
   itself is recipe-gated: only ``PROBE_RECIPES`` entries can be measured, so
   the mechanical derivation resolves visibility for the RECIPE-COVERED
   tensor-source family and honestly discloses the rest.

On this torch build the derived set is ``{torch.from_numpy, torch.from_dlpack,
torch.frombuffer, torch.Tensor.as_subclass, torch.Tensor._make_subclass}``
(pinned in ``tests/test_mechanical_belt.py``).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import types
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.overrides import TorchFunctionMode, get_overridable_functions, get_testing_overrides

from ... import _state

__all__ = [
    "BeltReport",
    "belt_report",
    "restore_belt_references",
    "sweep_stale_belt_references",
]


_SKIP_MODULE_PREFIXES = (
    # Same shallow-scan skip set the historical crawler used, plus torchlens
    # itself: these namespaces are either torch-owned (already decorated at
    # their public slots) or known torch-free.
    "torchlens",
    "torch.",
    "numpy.",
    "pytest",
    "pluggy",
    "setuptools",
)


class _CountingMode(TorchFunctionMode):
    """Count protocol dispatches during a probe call, passing through."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __torch_function__(
        self,
        func: Any,
        types_: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.calls += 1
        return func(*args, **(kwargs or {}))


class _ProbeSubTensor(torch.Tensor):
    """Minimal Tensor subclass for the ``as_subclass`` probe."""


def _from_file_args() -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build args/kwargs for the ``torch.from_file`` probe, backed by a temp file.

    The caller removes the temporary path after the probe call. ``shared=False``
    means the returned tensor does not require the pathname to remain present.
    """

    array = np.array([0.25, 0.5], dtype=np.float32)
    fd, path = tempfile.mkstemp()
    os.write(fd, array.tobytes())
    os.close(fd)
    return (path,), {"shared": False, "size": 2, "dtype": torch.float32}


PROBE_RECIPES: dict[tuple[str, str], Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]] = {
    # Tensor-source candidates in the statically not-mode-visible superset.
    # A recipe only synthesizes ARGUMENTS (built BEFORE the counting mode is
    # armed, so a factory call in the recipe cannot fire it); MEMBERSHIP is
    # decided by the measured callback count of the candidate call itself, so
    # a recipe for a mode-visible function (e.g. from_file on builds where it
    # dispatches) is harmless and self-excluding.
    ("torch", "from_numpy"): lambda: ((np.array([0.25, 0.5], dtype=np.float32),), {}),
    # The modern DLPack interop boundary is protocol-invisible for the same reason
    # ``from_numpy`` is: it builds a tensor from a FOREIGN buffer, so no
    # ``__torch_function__`` mode ever sees the call. It was not even a belt CANDIDATE
    # before, because candidacy is derived from ORIG_TORCH_FUNCS and the function was
    # absent from both of torch's override registries (see constants.py).
    ("torch", "from_dlpack"): lambda: ((np.array([0.25, 0.5], dtype=np.float32),), {}),
    ("torch", "frombuffer"): lambda: (
        (bytearray(np.array([0.25, 0.5], dtype=np.float32).tobytes()),),
        {"dtype": torch.float32},
    ),
    ("torch", "from_file"): _from_file_args,
    ("torch.Tensor", "as_subclass"): lambda: (
        (torch.tensor([0.25, 0.5]), _ProbeSubTensor),
        {},
    ),
    # Exact sibling of ``as_subclass``: builds a subclass VIEW from raw
    # storage below the override protocol, so a stale pre-wrap
    # ``_make_subclass`` reference loses the op with zero signal. Without a
    # recipe the pair sat disclosed-but-unprobed forever (b6-fable carried).
    ("torch.Tensor", "_make_subclass"): lambda: (
        (_ProbeSubTensor, torch.tensor([0.25, 0.5])),
        {},
    ),
    ("torch", "manual_seed"): lambda: ((7,), {}),
}


@contextmanager
def _probe_rng_bracket() -> Iterator[None]:
    """Snapshot/restore the global torch RNG around one probe evaluation.

    The probe framework executes candidate ORIGINALS with synthesized
    arguments at first wrap, inside the user's first capture. The candidate
    inventory is build-derived, so a state-mutating factory row entering it
    (``manual_seed`` already has a recipe that would call ``manual_seed(7)``;
    it is merely dead on current builds) would silently clobber the user's
    global seed. The bracket makes probe evaluation RNG-neutral by
    construction (b8-fable R56 latent-reseed hardening).
    """

    # Route through the W11-F1-guarded snapshot pair: gating on
    # ``torch.cuda.is_available()`` alone re-created the visible-but-unusable
    # CUDA abort on pure-CPU captures (the snapshot never touches CUDA until
    # the process has actually initialized it, and latches on a failed read).
    from ...utils.rng import log_current_rng_states, set_rng_from_saved_states

    saved_states = log_current_rng_states(torch_only=True)
    try:
        yield
    finally:
        set_rng_from_saved_states(saved_states)


@dataclass(frozen=True)
class BeltReport:
    """Derivation evidence for the protocol-invisible belt."""

    members: tuple[tuple[str, str], ...]
    probed_visible: tuple[tuple[str, str], ...]
    probe_failures: tuple[tuple[str, str], ...]
    unprobed_candidate_count: int
    unprobed_candidates: tuple[tuple[str, str], ...] = ()
    probe_failure_details: tuple[tuple[str, str, str], ...] = ()
    """``(namespace, func, exception repr)`` for each probe failure.

    A failed probe is a COVERAGE GAP, not a benign skip: the candidate could
    not be measured, so it is neither belt-patched nor proven mode-visible.
    A stale pre-wrap reference to it would lose ops with zero signal while
    the capture still reports ``capture_verified=True``. Consumers
    (``torchlens.utils.doctor()`` and ``torchlens.compat.report()``) surface
    these rows; the details make the failure actionable per build.
    """


_report: BeltReport | None = None
_member_map: dict[int, Any] | None = None
"""id(original callable) -> decorated wrapper, for the derived belt members."""

_swept_module_ids: dict[int, Callable[[], Any | None]] = {}
"""Module identities already swept this wrapper epoch (weak where possible)."""

_swept_ids_live: set[int] = set()
"""Ids of swept modules PROVABLY still alive, for the O(new) sweep pre-filter.

A weakref death callback discards the id the moment its module is finalized,
so a reused id is absent from this set and honestly reads as a new module.
Non-weakrefable modules enter permanently: the memo's strong closure keeps
them alive, so their id can never be reused. The set is a pure pre-filter --
membership only ever SKIPS work the per-module weakref memo would also skip;
any miss falls through to the unchanged authoritative loop."""

_ledger: list[tuple[Callable[[], Any | None], str, Any, Any]] = []
"""(module_ref, attr_name, original, replacement) reversal entries."""


def _touches_tensor(result: Any, func: Callable[..., Any]) -> bool:
    """A belt candidate must produce or consume tensors to matter."""

    if isinstance(result, torch.Tensor):
        return True
    qualname = getattr(func, "__qualname__", "")
    return qualname.startswith(("Tensor.", "TensorBase."))


def _resolve_namespace(namespace_name: str) -> Any | None:
    """Resolve a dotted ``torch.*`` namespace name, or ``None`` if any part is absent."""

    obj: Any = torch
    for part in namespace_name.replace("torch.", "").split("."):
        if part:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
    return obj


def _derive() -> tuple[BeltReport, dict[int, Any]]:
    """Measure the protocol-invisible wrapped set on the running build."""

    from ...constants import get_orig_torch_funcs

    inventory = get_orig_torch_funcs(include_torchvision=False)
    statically_visible: set[int] = set()
    for functions in get_overridable_functions().values():
        statically_visible.update(id(func) for func in functions)
    statically_visible.update(id(func) for func in get_testing_overrides())

    members: list[tuple[str, str]] = []
    probed_visible: list[tuple[str, str]] = []
    probe_failures: list[tuple[str, str]] = []
    probe_failure_details: list[tuple[str, str, str]] = []
    member_map: dict[int, Any] = {}
    unprobed_candidates: list[tuple[str, str]] = []
    seen_original_ids: set[int] = set()

    for namespace_name, func_name in inventory:
        namespace = _resolve_namespace(namespace_name)
        if namespace is None or not hasattr(namespace, func_name):
            continue
        current = getattr(namespace, func_name)
        original = _state._decorated_to_orig.get(id(current), current)
        if id(original) in seen_original_ids:
            continue
        seen_original_ids.add(id(original))
        if id(original) in statically_visible:
            continue
        recipe = PROBE_RECIPES.get((namespace_name, func_name))
        if recipe is None:
            unprobed_candidates.append((namespace_name, func_name))
            continue
        mode = _CountingMode()
        cleanup_path: str | None = None
        try:
            with _state.pause_logging(), _probe_rng_bracket():
                args, kwargs = recipe()
                if (namespace_name, func_name) == ("torch", "from_file"):
                    cleanup_path = str(args[0])
                with mode:
                    result = original(*args, **kwargs)
        except Exception as exc:
            probe_failures.append((namespace_name, func_name))
            probe_failure_details.append((namespace_name, func_name, repr(exc)))
            continue
        finally:
            if cleanup_path is not None:
                try:
                    os.unlink(cleanup_path)
                except FileNotFoundError:
                    pass
        if mode.calls:
            probed_visible.append((namespace_name, func_name))
            continue
        if not _touches_tensor(result, original):
            continue
        members.append((namespace_name, func_name))
        wrapper = _state._orig_to_decorated.get(id(original))
        if wrapper is not None:
            member_map[id(original)] = wrapper

    report = BeltReport(
        members=tuple(members),
        probed_visible=tuple(probed_visible),
        probe_failures=tuple(probe_failures),
        unprobed_candidate_count=len(unprobed_candidates),
        unprobed_candidates=tuple(unprobed_candidates),
        probe_failure_details=tuple(probe_failure_details),
    )
    return report, member_map


def _member_map_is_current() -> bool:
    """True while every cached wrapper is still the live one for its original.

    The cached derivation binds wrapper OBJECT identities. A fresh full
    decoration pass (first-wrap retry after a partial failure) mints a new
    wrapper generation, at which point sweeping the cached objects would
    patch dead wrappers into user modules.
    """

    if _member_map is None:
        return False
    return all(
        _state._orig_to_decorated.get(original_id) is wrapper
        for original_id, wrapper in _member_map.items()
    )


def belt_report() -> BeltReport | None:
    """Return the derivation report, deriving on first use after wrapping.

    The derivation is re-validated against the live wrapper registries: if
    the wrapper generation changed underneath the cache, mutations made with
    the dead generation are reversed and the belt re-derives.

    From a NON-OWNER thread while a capture is live (``doctor()`` /
    ``compat.report()`` polled during a long forward), the cached report is
    served READ-ONLY (r8 R54 belt pair): deriving would evaluate probe
    originals under the global-RNG snapshot/restore bracket, and the restore
    would REWIND any draws the capture made in between -- silently repeating
    its dropout/noise stream -- while the stale-generation reversal would
    patch module slots underneath the running capture.
    """

    global _report, _member_map
    if not _state._is_decorated:
        return _report
    if (
        (_state._logging_enabled or _state._active_trace is not None)
        and _state._active_owner_thread_id is not None
        and _state._active_owner_thread_id != threading.get_ident()
    ):
        return _report
    if _report is not None and not _member_map_is_current():
        restore_belt_references()
        _report = None
        _member_map = None
    if _report is None:
        _report, _member_map = _derive()
    return _report


def _weak_module_ref(module: types.ModuleType) -> Callable[[], Any | None]:
    """Weak reference to ``module``, degrading to a strong closure when unsupported."""

    try:
        return weakref.ref(module)
    except TypeError:
        return lambda: module


def _weak_swept_module_ref(
    module: types.ModuleType,
) -> Callable[[], Any | None]:
    """Return a weak module reference for the per-module sweep memo.

    Weakrefable modules register a death callback that evicts their id from
    ``_swept_ids_live``; the strong-closure fallback keeps the module alive,
    so its id stays valid and may remain in the live set permanently.
    """

    module_id = id(module)
    try:
        ref = weakref.ref(module, _evict_live_id(module_id))
    except TypeError:
        _swept_ids_live.add(module_id)
        return lambda: module
    _swept_ids_live.add(module_id)
    return ref


def _evict_live_id(entry_id: int) -> Callable[[Any], None]:
    """Death callback evicting ``entry_id`` from the sweep pre-filter live set."""

    def _evict(_ref: Any) -> None:
        """Discard the captured id when its referent is finalized."""
        _swept_ids_live.discard(entry_id)

    return _evict


def sweep_stale_belt_references() -> int:
    """Patch stale module-level references to belt members, with a ledger.

    Scans each live module identity ONCE per wrapper epoch (new imports are
    picked up on the next sweep). Only module ``__dict__`` slots are
    patched — the measured reachable holder class for the protocol-invisible
    functions — and every mutation is recorded for conditional reversal at
    ``unwrap_torch()``.

    Returns
    -------
    int
        Number of slots patched by this sweep.
    """

    # grind-r5 b3 R02 (opus+sol corroborated LOW, 2 rounds): NO length
    # watermark. ``len(sys.modules)`` equality is not identity -- a same-length
    # mutation (del one key + insert another) skipped the sweep and left a
    # protocol-invisible stale ``from_numpy``/``frombuffer``/``as_subclass``
    # reference unpatched, exactly the zero-signal class the belt exists to
    # close. The per-module memo below already makes every sweep O(modules)
    # dict lookups with O(new modules) real work.
    report = belt_report()
    if report is None or _member_map is None or not _member_map:
        return 0
    # O(new-modules) pre-filter: every id in ``_swept_ids_live`` is a module
    # this epoch's loop already scanned AND that is provably still the same
    # object (death callbacks evict dead ids, so a reused id reads as new).
    # ``set(map(id, ...))`` executes no Python bytecode, so it is atomic
    # under the GIL like the ``list(sys.modules.items())`` snapshot below.
    if not set(map(id, sys.modules.values())) - _swept_ids_live:
        return 0
    patched = 0
    for mod_key, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType):
            # Non-module sys.modules entries (e.g. the typing.io/typing.re
            # pseudo-module classes) are never scanned, but they must still
            # enter the live-id set under the same weakref-eviction contract
            # or their ids read as new forever and the pre-filter never
            # fires. Unweakrefable entries stay out: degraded, never wrong.
            entry_id = id(module)
            if entry_id not in _swept_ids_live:
                try:
                    entry_ref = weakref.ref(module, _evict_live_id(entry_id))
                except TypeError:
                    continue
                _swept_module_ids[entry_id] = entry_ref
                _swept_ids_live.add(entry_id)
            continue
        previous_ref = _swept_module_ids.get(id(module))
        if previous_ref is not None and previous_ref() is module:
            continue
        _swept_module_ids[id(module)] = _weak_swept_module_ref(module)
        if mod_key.startswith(_SKIP_MODULE_PREFIXES) or ".dist-info" in mod_key:
            continue
        try:
            module_dict = vars(module)
        except TypeError:
            continue
        for attr_name, attr_val in list(module_dict.items()):
            replacement = _member_map.get(id(attr_val))
            if replacement is None:
                continue
            try:
                if module_dict.get(attr_name) is not attr_val:
                    continue
                module_dict[attr_name] = replacement
            except (KeyError, TypeError):
                continue
            _ledger.append((_weak_module_ref(module), attr_name, attr_val, replacement))
            patched += 1
    return patched


def restore_belt_references() -> None:
    """Conditionally reverse belt mutations and reset the epoch sweep memo.

    A slot is restored only when it still holds the exact wrapper the belt
    installed; user reassignments made after the sweep are preserved.
    """

    for module_ref, attr_name, original, replacement in reversed(_ledger):
        module = module_ref()
        if module is None:
            continue
        try:
            module_dict = vars(module)
            if module_dict.get(attr_name) is replacement:
                module_dict[attr_name] = original
        except (KeyError, TypeError):
            continue
    _ledger.clear()
    _swept_module_ids.clear()
    _swept_ids_live.clear()

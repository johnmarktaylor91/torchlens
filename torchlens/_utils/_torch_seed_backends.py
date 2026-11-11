"""Per-backend torch seeding with ``torch.manual_seed``'s exact effect on compiled backends.

``torch.manual_seed`` seeds, in order, every CUDA device, MPS, every XPU device, every
MTIA device, the privateuse1 backend and then the CPU default generator, skipping an
accelerator whose module reports a bad fork. A CUDA, XPU or MTIA backend that is not
initialized gets a lazy-seed closure queued together with a ``traceback.format_stack()``
(three stack captures per seed on a CPU-only build, about 1 ms of every capture). A
backend NOT compiled into this torch build can never initialize, so its queued closure
can never run: skipping that one call is the only difference from ``torch.manual_seed``.
Every compiled backend gets the same call, in the same order, with the same bad-fork
guard.

The backend list is read from torch's own seeding code (the names its code object
references), never from ``torch.__version__``, by the probe in ``_torch_symbols`` (kept
there so ``_torch_compat`` publishes the flag without importing this module). When that
code names anything the probe does not know, ``HAS_PER_BACKEND_SEED`` is ``False`` and
:func:`manual_seed_compiled_backends` calls ``torch.manual_seed`` itself.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import torch

from ._torch_symbols import (  # noqa: F401 - re-exported
    _SEED_CALLS,
    HAS_PER_BACKEND_SEED,
    TORCH_SEED_PLAN as _SEED_PLAN,
)

# Compiled-in verdicts, filled on first use; tests set entries to simulate other builds.
BACKEND_COMPILED: dict[str, bool] = {}


def _backend_compiled(name: str, module: ModuleType) -> bool:
    """Return whether accelerator ``name`` is compiled into this torch build.

    MPS seeding is a cheap no-op without MPS, so only the lazy-queue backends are probed;
    a backend without an ``_is_compiled`` probe, or whose probe raises, counts as compiled
    so it keeps torch's own call (never a skipped seed on doubt).
    """

    verdict = BACKEND_COMPILED.get(name)
    if verdict is None:
        probe = getattr(module, "_is_compiled", None)
        verdict = True
        if name != "mps" and callable(probe):
            try:
                verdict = bool(probe())
            except Exception:  # noqa: BLE001 - an unreadable probe keeps torch's call
                verdict = True
        BACKEND_COMPILED[name] = verdict
    return verdict


def manual_seed_compiled_backends(seed: int) -> None:
    """Seed torch like ``torch.manual_seed``, skipping lazy seeds for uncompiled backends.

    Parameters
    ----------
    seed:
        Seed value; converted with ``int`` and range-checked by the CPU generator, as
        ``torch.manual_seed`` does.
    """

    if _SEED_PLAN is None:
        torch.manual_seed(seed)
        return
    seed = int(seed)
    accelerators, seeds_custom_device = _SEED_PLAN
    for name in accelerators:
        try:
            module = importlib.import_module(f"torch.{name}")
        except ImportError:  # torch's ``hasattr(torch, name)`` guard: nothing to seed
            continue
        if _backend_compiled(name, module) and not module._is_in_bad_fork():
            getattr(module, _SEED_CALLS[name])(seed)
    if seeds_custom_device:
        torch.random._seed_custom_device(seed)
    torch.default_generator.manual_seed(seed)

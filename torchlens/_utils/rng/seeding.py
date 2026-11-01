"""Seeding every RNG engine from one capture seed, and the seeded-op predicate.

:func:`set_random_seed` seeds Python ``random``, NumPy, torch (CPU and every accelerator)
and TorchLens's private barcode stream from one seed, re-arming the CUDA snapshot latch
in :mod:`.streams`. :func:`aten_qualname_is_seeded_rng` tells whether a captured callable
is an ATen operator PyTorch tags ``nondeterministic_seeded``.
"""

from __future__ import annotations

import random
import warnings as _warnings_module

import numpy as np
import torch

from .._uninit_alloc import _SEEDED_RNG_NAMESPACES
from ..hashing import seed_barcode_rng
from . import streams
from .monitor_state import _suppress_active_monitor_marks


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


def _seed_torch_engines(seed: int) -> None:
    """Seed torch's CPU and accelerator generators, degrading on a broken stack.

    ``torch.manual_seed`` seeds the accelerator engines (every visible CUDA
    device, MPS, XPU) BEFORE the CPU default generator, so a CUDA runtime that
    claims initialization but cannot serve its generators (first observed on
    real H200 hardware as an ``IndexError`` from
    ``torch.cuda.default_generators``) aborted a pure-CPU capture with the CPU
    engine still unseeded.  A broken accelerator must degrade a capture, never
    abort it (the :func:`_snapshot_cuda_rng_states` contract): the failure
    falls back to seeding the CPU default generator directly, with a warning;
    the later CUDA RNG snapshot surfaces and latches its own read failure.

    Parameters
    ----------
    seed:
        Seed value to set.
    """
    try:
        torch.manual_seed(seed)
    except Exception as exc:  # noqa: BLE001 - any broken-accelerator failure mode
        torch.default_generator.manual_seed(seed)
        _warnings_module.warn(
            "Could not seed torch accelerator RNG engines "
            f"({type(exc).__name__}: {exc}); the torch CPU generator was "
            "seeded directly and the capture continues. Operations that "
            "consume accelerator randomness cannot be reproduced exactly "
            "for this capture.",
            stacklevel=3,
        )


def set_random_seed(seed: int) -> None:
    """Set the random seed for all RNG engines simultaneously.

    Ensures deterministic behavior across Python, NumPy, and PyTorch
    (CPU + all CUDA devices).

    Parameters
    ----------
    seed:
        Seed value to set.
    """
    # Every capture seeds at entry, so this is the per-capture seam that
    # re-arms the CUDA RNG snapshot retry latch: a transient generator-read
    # failure (busy device, momentary OOM) degrades only the capture that hit
    # it instead of latching the whole process to "no CUDA RNG snapshots".
    streams.rearm_cuda_rng_latch()
    # r65 CLUSTER Z: TorchLens-OWNED seeding is never model host nondeterminism.
    # Normally this runs pre-forward (outside any monitor window), but the bracket
    # keeps any in-window TorchLens-initiated reseed from marking the torch RNG
    # mutation channels the r65 registry rows monitor.
    with _suppress_active_monitor_marks():
        random.seed(seed)
        np.random.seed(seed)
        _seed_torch_engines(seed)
        # Keep torchlens's private barcode RNG in lockstep with the seed so a fixed
        # capture seed yields reproducible tensor barcodes (a fork replay reuses the
        # original seed; matching barcodes keep tensor/op/param cross-references
        # consistent). The barcode RNG stays a separate stream, so this does not
        # perturb the user's global ``random`` state that host-RNG honesty brackets.
        seed_barcode_rng(seed)

"""CUDA availability and initialization probes, cached once per process.

A leaf module: it imports torch and nothing from TorchLens, so every caller
(stream snapshots, allocator cleanup, device memory bookkeeping) can read the
probes without joining the capture import cycle.
"""

from __future__ import annotations

import warnings

import torch

# Cached result of torch.cuda.is_available().  Evaluated once per process
# because CUDA availability cannot change at runtime.  Avoids repeated
# calls into the CUDA runtime (which involve driver queries).
_cuda_available: bool | None = None


def _is_cuda_available() -> bool:
    """Return True if CUDA is available (cached after first call).

    The result is cached in a module-level global because CUDA availability
    is fixed for the lifetime of the process, and ``torch.cuda.is_available()``
    involves a non-trivial driver query.

    ``torch.cuda.is_available()`` normally swallows driver failures and returns
    False, but a visible-but-unusable CUDA stack (stale driver, mismatched
    build) can make the probe itself raise.  A failed *probe* is treated as
    "no CUDA": TorchLens' CUDA uses are all opportunistic (cache release,
    device-side RNG snapshots), so a broken accelerator must degrade a CPU
    capture, never abort it.  The failure is surfaced as a warning, once per
    process, rather than silently.
    """
    global _cuda_available
    if _cuda_available is None:
        try:
            _cuda_available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001 - any driver/runtime probe failure
            # Cache BEFORE warning: under a caller's warnings-as-errors policy the
            # warn() itself raises, and the answer must still be latched so the
            # broken probe is not repeated on the next call.
            _cuda_available = False
            warnings.warn(
                "torch.cuda.is_available() raised "
                f"{type(exc).__name__}: {exc}. Treating CUDA as unavailable for "
                "this process; CPU capture continues unaffected.",
                stacklevel=2,
            )
    return _cuda_available


def _is_cuda_initialized() -> bool:
    """Return True if this process has already initialized the CUDA runtime.

    Unlike :func:`_is_cuda_available` this is a pure read of torch's own
    module-level init flag: it never queries the driver, never initializes a
    device, and is therefore NOT cached (it flips from False to True the first
    time anything in the process touches CUDA).

    Callers use it to distinguish "CUDA state exists and may have been
    consumed" from "nothing in this process has ever touched CUDA", so that
    opportunistic CUDA bookkeeping can be skipped instead of force-initializing
    every visible device.
    """
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - torch build without CUDA support
        return False

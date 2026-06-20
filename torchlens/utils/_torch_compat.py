"""Backward-compatible shims for torch APIs whose signature changed across versions.

TorchLens targets the *eager* PyTorch surface and only uses a tiny number of
version-sensitive APIs.  This module isolates those so that TorchLens can run on
**torch 2.1+** (instead of requiring torch 2.4+), while behaving *identically* on
modern torch.

The single incompatibility is autocast state introspection:

* On **torch >= 2.4**, the per-device query helpers take a ``device_type``
  argument::

      torch.is_autocast_enabled("cpu")
      torch.get_autocast_dtype("cuda")

* On **torch 2.1 - 2.3**, those functions take *no* argument (they query the
  CUDA/GPU autocast state only), and per-device queries route through the older
  device-specific helpers::

      torch.is_autocast_cpu_enabled()      # CPU enabled flag
      torch.get_autocast_cpu_dtype()       # CPU autocast dtype
      torch.is_autocast_enabled()          # CUDA/GPU enabled flag
      torch.get_autocast_gpu_dtype()       # CUDA/GPU autocast dtype

The ``device_type`` argument was added in torch 2.4.0 (see
https://github.com/pytorch/pytorch and the widely-hit downstream breakage
https://github.com/huggingface/transformers/issues/43508).  The legacy
device-specific helpers still exist on modern torch (as deprecated aliases), but
we *prefer the modern signature when available* so that we never emit deprecation
warnings on supported torch and so behavior tracks the canonical implementation.

We probe the modern signature **once** at import time (a capability probe, not a
brittle version-string parse) and bind the appropriate implementation.  This keeps
the hot path branch-free and makes the fallback robust to any future torch that
keeps the modern signature but changes its version string.
"""

from __future__ import annotations

import torch

__all__ = ["autocast_is_enabled", "autocast_get_dtype", "AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED"]


def _probe_device_type_arg_supported() -> bool:
    """Return True if ``torch.is_autocast_enabled`` accepts a ``device_type`` arg.

    True on torch >= 2.4, False on torch 2.1 - 2.3.  We call the modern signature
    inside a try/except rather than parsing ``torch.__version__`` so the result is
    derived from the actual runtime API, not a version string that could be a
    nightly/custom build.
    """
    try:
        # On torch 2.1-2.3 this raises TypeError ("takes no arguments" /
        # "takes 0 positional arguments but 1 was given").  On torch >= 2.4 it
        # returns a bool.
        torch.is_autocast_enabled("cpu")  # type: ignore[call-arg]
        return True
    except TypeError:
        return False


AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED: bool = _probe_device_type_arg_supported()


# --- Legacy (torch 2.1-2.3) per-device fallbacks -------------------------------
#
# These map a device_type string onto the old device-specific helpers.  We only
# special-case the two device types TorchLens captures ("cpu", "cuda"); any other
# device_type on legacy torch has no autocast-state query API and raises
# RuntimeError, which the caller in rng.py already swallows (it skips devices that
# can't be queried).


def _legacy_is_autocast_enabled(device_type: str) -> bool:
    if device_type == "cpu":
        return bool(torch.is_autocast_cpu_enabled())  # type: ignore[attr-defined]
    if device_type == "cuda":
        # The no-arg form queries the CUDA/GPU autocast flag on legacy torch.
        return bool(torch.is_autocast_enabled())
    raise RuntimeError(
        f"autocast state query not supported for device_type={device_type!r} "
        f"on torch {torch.__version__}"
    )


def _legacy_get_autocast_dtype(device_type: str) -> torch.dtype:
    if device_type == "cpu":
        return torch.get_autocast_cpu_dtype()  # type: ignore[attr-defined]
    if device_type == "cuda":
        return torch.get_autocast_gpu_dtype()  # type: ignore[attr-defined]
    raise RuntimeError(
        f"autocast dtype query not supported for device_type={device_type!r} "
        f"on torch {torch.__version__}"
    )


# --- Public, version-neutral entry points --------------------------------------
#
# Bound once at import time to the correct implementation.  On torch >= 2.4 these
# are exactly ``torch.is_autocast_enabled`` / ``torch.get_autocast_dtype`` (byte
# identical behavior); on torch 2.1-2.3 they route to the legacy helpers above.

if AUTOCAST_DEVICE_TYPE_ARG_SUPPORTED:

    def autocast_is_enabled(device_type: str) -> bool:
        """Return whether autocast is enabled for ``device_type`` (torch>=2.4 path)."""
        return bool(torch.is_autocast_enabled(device_type))

    def autocast_get_dtype(device_type: str) -> torch.dtype:
        """Return the autocast dtype for ``device_type`` (torch>=2.4 path)."""
        return torch.get_autocast_dtype(device_type)

else:

    def autocast_is_enabled(device_type: str) -> bool:
        """Return whether autocast is enabled for ``device_type`` (torch 2.1-2.3 path)."""
        return _legacy_is_autocast_enabled(device_type)

    def autocast_get_dtype(device_type: str) -> torch.dtype:
        """Return the autocast dtype for ``device_type`` (torch 2.1-2.3 path)."""
        return _legacy_get_autocast_dtype(device_type)

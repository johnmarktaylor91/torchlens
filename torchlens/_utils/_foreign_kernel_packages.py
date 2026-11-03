"""Probe for FlashAttention 4: is the CuTe interface package installed.

``backends/torch/_known_foreign_kernels.py`` records every ``autograd.Function`` of
``flash_attn.cute.interface`` (``flash_attn_func`` and ``flash_attn_varlen_func`` run through
``FlashAttnFunc.apply`` and ``FlashAttnVarlenFunc.apply``) as one foreign-kernel boundary
op: the forward hands its tensors to a CuTe kernel through TVM FFI, below the torch function
layer and without the numpy or dlpack export the dynamic detector sees. The
``HAS_FLASH_ATTENTION_4_INTERFACE`` flag reports whether that package is installed. It reads
the package's module search path and never imports it (importing ``flash_attn.cute`` loads
the CUTLASS DSL), so ``import torchlens`` stays as light as before; the flag and its
capability-snapshot row stay in ``torchlens._utils._torch_compat``, which imports this leaf
(the sibling of ``_script_call_slot``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

FLASH_ATTENTION_4_INTERFACE_MODULE = "flash_attn.cute.interface"
"""Module that defines the FlashAttention 4 ``autograd.Function`` entry points."""


def probe_flash_attention_4_interface() -> bool:
    """Return whether the FlashAttention 4 CuTe interface is installed.

    Returns
    -------
    bool
        True when a ``flash_attn`` package on ``sys.path`` ships ``cute/interface.py``
        (the ``flash-attn-4`` distribution; FlashAttention 2 and 3 ship no ``cute``
        package). Nothing is imported: ``find_spec`` of a top-level name only locates it.
    """

    try:
        spec = importlib.util.find_spec("flash_attn")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    return any(
        (Path(location) / "cute" / "interface.py").is_file()
        for location in spec.submodule_search_locations
    )

"""Known foreign-kernel ``autograd.Function`` classes: FlashAttention 4.

The dynamic detector in ``_function_boundary`` collapses a Function whose forward left
torch through an export it can see (numpy, dlpack). FlashAttention 4 (``flash_attn.cute``)
hands q, k, v and its fresh ``torch.empty`` output to a CuTe kernel through TVM FFI: no
export the detector sees, no ``custom_op``, no aten op for the attention itself, so the
trace used to record the allocation in place of the attention and the completeness census
reported the hole (forward validation failed on ``bfs_completeness``). Every
``autograd.Function`` defined in ``flash_attn.cute.interface`` is therefore a known foreign
kernel: its ``apply`` is recorded as one ``autogradfunction`` boundary op with the exit
``flash_attn.cute``, the same op, census token and replay the dynamic path produces, so
nothing is exempted: the boundary op owns every dispatch inside the call and replays by
calling the Function's ``apply`` again. Gated by ``HAS_FLASH_ATTENTION_4_INTERFACE``, read
at call time so a flipped flag is honoured.
"""

from __future__ import annotations

from typing import Any

from ..._utils import _torch_compat
from ..._utils._foreign_kernel_packages import FLASH_ATTENTION_4_INTERFACE_MODULE

FLASH_ATTENTION_4_EXIT = "flash_attn.cute"
"""Exit name recorded in the boundary op's ``transform_config["exits"]`` and its warning."""

_KNOWN_FOREIGN_KERNEL_MODULES: tuple[tuple[str, str, str], ...] = (
    (FLASH_ATTENTION_4_INTERFACE_MODULE, "HAS_FLASH_ATTENTION_4_INTERFACE", FLASH_ATTENTION_4_EXIT),
)
"""``(module, capability flag, exit name)`` per known foreign-kernel Function module."""


def known_foreign_kernel_exits(cls: type[Any]) -> tuple[str, ...]:
    """Return the exit names of a Function class known to leave torch below the dispatcher.

    Parameters
    ----------
    cls:
        The ``autograd.Function`` subclass being applied.

    Returns
    -------
    tuple[str, ...]
        ``("flash_attn.cute",)`` for a Function defined in the FlashAttention 4 interface
        module (or a submodule of it) while its capability flag is set; empty otherwise.
    """

    module = getattr(cls, "__module__", None)
    if not isinstance(module, str):
        return ()
    for known_module, flag, exit_name in _KNOWN_FOREIGN_KERNEL_MODULES:
        if module != known_module and not module.startswith(known_module + "."):
            continue
        if getattr(_torch_compat, flag, False):
            return (exit_name,)
    return ()

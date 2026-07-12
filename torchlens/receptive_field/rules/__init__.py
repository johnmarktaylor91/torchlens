"""Built-in receptive-field rules and idempotent pack registration."""

from __future__ import annotations

from .._rules import _install_builtin_rule_pack
from . import attention, conv_pool, elementwise, interpolation, linear, norms, sequence, transforms


_install_builtin_rule_pack()

__all__ = [
    "attention",
    "conv_pool",
    "elementwise",
    "interpolation",
    "linear",
    "norms",
    "sequence",
    "transforms",
]

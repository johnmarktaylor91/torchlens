"""Torch function wrapping and model preparation for logging."""

from .torch_funcs import (
    decorate_all_once,
    patch_detached_references,
    patch_model_instance,
    wrap_torch,
    unwrap_torch,
    wrapped,
)

__all__ = [
    "decorate_all_once",
    "patch_detached_references",
    "patch_model_instance",
    "wrap_torch",
    "unwrap_torch",
    "wrapped",
]

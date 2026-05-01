"""Torch function wrapping and model preparation for logging."""

from .torch_funcs import (
    clear_patch_detached_references_cache,
    decorate_all_once,
    patch_detached_references,
    patch_model_instance,
    wrap_torch,
    unwrap_torch,
    wrapped,
)

__all__ = [
    "decorate_all_once",
    "clear_patch_detached_references_cache",
    "patch_detached_references",
    "patch_model_instance",
    "wrap_torch",
    "unwrap_torch",
    "wrapped",
]

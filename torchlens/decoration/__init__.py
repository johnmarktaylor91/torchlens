"""Torch function wrapping and model preparation for logging."""

from .torch_funcs import (
    decorate_all_once,
    patch_detached_references,
    patch_model_instance,
    redecorate_all_globally,
    undecorate_all_globally,
)

__all__ = [
    "decorate_all_once",
    "patch_detached_references",
    "patch_model_instance",
    "redecorate_all_globally",
    "undecorate_all_globally",
]

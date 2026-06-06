"""Built-in TorchLens semantic facet recipes."""

from __future__ import annotations

from importlib import metadata
from typing import Any
import warnings

from . import attention, embedding, mlp, norm, residual
from ..facets import mark_current_registry_as_builtins

BUILTIN_FACET_CAPABILITY_INVENTORY: dict[str, dict[str, str]] = {
    "attention": {
        "q": "op_structural",
        "k": "op_structural",
        "v": "op_structural",
        "attn_out": "op_structural",
        "input": "module_input",
        "n_heads": "computed_read_only",
        "n_q_heads": "computed_read_only",
        "n_kv_heads": "computed_read_only",
        "d_head": "computed_read_only",
        "head": "computed_read_only",
        "scores": "computed_read_only",
        "pattern": "computed_read_only",
        "z": "computed_read_only",
        "result": "computed_read_only",
    },
    "mlp": {
        "gated_out": "computed_read_only",
        "up_out": "op_structural",
        "down_out": "op_structural",
        "intermediate": "computed_read_only",
        "input": "module_input",
        "output": "op_structural",
    },
    "norm": {
        "normalized": "op_structural",
        "gamma": "parameter",
        "beta": "parameter",
        "input": "module_input",
    },
    "embedding": {
        "lookup": "op_structural",
        "weight": "parameter",
        "indices": "module_input",
    },
    "residual": {
        "resid_pre": "op_structural",
        "resid_mid": "op_structural",
        "resid_post": "op_structural",
    },
}

mark_current_registry_as_builtins()


def _load_entrypoint_recipes() -> None:
    """Load setuptools entry-point recipe plugins fail-safely.

    Returns
    -------
    None
        Entry points are imported for registration side effects.
    """

    try:
        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            recipe_points = entry_points.select(group="torchlens.recipes")
        else:
            entry_point_map: Any = entry_points
            recipe_points = getattr(entry_point_map, "get")("torchlens.recipes", ())
    except Exception as exc:
        warnings.warn(
            f"Could not inspect torchlens.recipes entry points: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return
    for entry_point in recipe_points:
        try:
            loaded: Any = entry_point.load()
            if callable(loaded) and getattr(loaded, "_torchlens_recipe_autoload", False):
                loaded()
        except Exception as exc:
            warnings.warn(
                f"Skipping broken torchlens.recipes entry point {entry_point.name!r}: {exc}",
                UserWarning,
                stacklevel=2,
            )


_load_entrypoint_recipes()

__all__ = [
    "BUILTIN_FACET_CAPABILITY_INVENTORY",
    "attention",
    "embedding",
    "mlp",
    "norm",
    "residual",
]

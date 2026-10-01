"""Built-in TorchLens semantic facet recipes.

Entry points: discover everything, execute nothing (architecture memo 6.4).
The historical module-scope autoloader called ``entry_point.load()`` on every
installed ``torchlens.recipes`` provider at import time -- and because the
lazy root facade reaches this module, ``hasattr(tl, "facets")`` executed
installed third-party code (measured; the launch-blocking class). Discovery
is now METADATA-ONLY (:func:`installed_recipe_providers` parses dist-info
text and executes nothing) and activation is EXPLICIT
(:func:`activate_entrypoint_recipes`): importing the provider package or
calling the one activation operation. Never at ``import torchlens``, never at
namespace import, never inside ``__getattr__`` -- ``hasattr``, ``dir()``, IDE
sweeps, and agent surface walks cannot express consent to execute installed
code. Plugins that relied on autoload must now be activated explicitly; both
activation spellings are DOCUMENTED-UNSTABLE pending naming-session
ratification.
"""

from __future__ import annotations

import warnings
from importlib import metadata
from typing import Any

from ..facets import mark_current_registry_as_builtins
from . import attention, embedding, lm_head, mlp, norm, residual

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
    "lm_head": {
        "logits": "op_structural",
        "unembed_weight": "parameter",
        "unembed_bias": "parameter",
        "final_norm_kind": "computed_read_only",
        "final_norm_eps": "computed_read_only",
        "final_norm_gamma": "parameter",
        "final_norm_beta": "parameter",
        "final_norm_input": "op_structural",
    },
}

mark_current_registry_as_builtins()


def _installed_recipe_entry_points() -> tuple[metadata.EntryPoint, ...]:
    """Enumerate installed ``torchlens.recipes`` entry points, metadata-only.

    ``importlib.metadata.entry_points()`` parses dist-info text and executes
    nothing; no provider code runs here.

    Returns
    -------
    tuple[importlib.metadata.EntryPoint, ...]
        Installed recipe entry points, or ``()`` when enumeration fails
        (disclosed with a warning).
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
            stacklevel=3,
        )
        return ()
    return tuple(recipe_points)


def installed_recipe_providers() -> tuple[dict[str, str], ...]:
    """Return the metadata-only inventory of installed recipe providers.

    Nothing is imported and nothing executes: each row is parsed dist-info
    text, suitable for teaching errors and availability listings. Activation
    is a separate, explicit act (:func:`activate_entrypoint_recipes`).

    Returns
    -------
    tuple[dict[str, str], ...]
        One row per installed ``torchlens.recipes`` entry point:
        ``{"name", "value", "activated"}`` (``activated`` is ``"true"`` /
        ``"false"``).
    """

    return tuple(
        {
            "name": entry_point.name,
            "value": entry_point.value,
            "activated": "true" if entry_point.name in _ACTIVATED_RECIPE_PROVIDERS else "false",
        }
        for entry_point in _installed_recipe_entry_points()
    )


#: Names of entry-point providers explicitly activated in this session.
_ACTIVATED_RECIPE_PROVIDERS: set[str] = set()


def activate_entrypoint_recipes(
    names: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Explicitly load installed ``torchlens.recipes`` providers.

    THE one provider-load operation (architecture memo 6.4): each selected
    entry point is loaded (this imports and executes the provider's module),
    and a loaded callable marked ``_torchlens_recipe_autoload`` is invoked
    for registration side effects. Broken providers warn and are skipped,
    never crash the batch; already-activated providers are skipped.

    Parameters
    ----------
    names:
        Entry-point names to activate, or ``None`` for every installed one.
        Unknown requested names warn (disclosed, never silent).

    Returns
    -------
    tuple[str, ...]
        Names of the providers activated by THIS call.
    """

    requested = None if names is None else set(names)
    installed = _installed_recipe_entry_points()
    if requested is not None:
        unknown = requested - {entry_point.name for entry_point in installed}
        if unknown:
            from ...errors import TorchLensWarning

            warnings.warn(
                TorchLensWarning(
                    "Unknown torchlens.recipes entry point name(s): "
                    f"{sorted(unknown)}; installed providers: "
                    f"{sorted(entry_point.name for entry_point in installed)}. "
                    "Remedy: activate an installed provider name from "
                    "installed_recipe_providers(), or install the provider "
                    "distribution first",
                    code="recipes_unknown_provider",
                    unknown_names=sorted(unknown),
                ),
                stacklevel=2,
            )
    activated: list[str] = []
    for entry_point in installed:
        if requested is not None and entry_point.name not in requested:
            continue
        if entry_point.name in _ACTIVATED_RECIPE_PROVIDERS:
            continue
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
            continue
        _ACTIVATED_RECIPE_PROVIDERS.add(entry_point.name)
        activated.append(entry_point.name)
    return tuple(activated)


__all__ = [
    "BUILTIN_FACET_CAPABILITY_INVENTORY",
    "activate_entrypoint_recipes",
    "attention",
    "embedding",
    "installed_recipe_providers",
    "lm_head",
    "mlp",
    "norm",
    "residual",
]

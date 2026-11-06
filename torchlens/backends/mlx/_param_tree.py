"""Flattened MLX parameter-tree walkers for the backend's parameter logs.

Extracted from ``backend`` so that module stays under its size ceiling.
"""

from __future__ import annotations

from typing import Any

from .model_prep import MLXModuleTree, _flatten_parameter_tree


def _iter_trainable_parameter_tree(model: object) -> list[tuple[str, Any]]:
    """Return flattened MLX trainable parameter leaves.

    Parameters
    ----------
    model
        MLX module.

    Returns
    -------
    list[tuple[str, Any]]
        Trainable parameter addresses and arrays.
    """

    trainable_parameters = getattr(model, "trainable_parameters", None)
    if not callable(trainable_parameters):
        return []
    return list(_flatten_parameter_tree(trainable_parameters(), ""))


def _iter_parameter_tree(model: object) -> list[tuple[str, Any]]:
    """Return flattened MLX parameter leaves.

    Parameters
    ----------
    model
        MLX module.

    Returns
    -------
    list[tuple[str, Any]]
        Parameter addresses and arrays.
    """

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return []
    return list(_flatten_parameter_tree(parameters(), ""))


def _alias_to_primary(tree: MLXModuleTree) -> dict[str, str]:
    """Return module alias to primary address mapping.

    Parameters
    ----------
    tree
        Discovered MLX module tree.

    Returns
    -------
    dict[str, str]
        Alias-to-primary mapping.
    """

    aliases: dict[str, str] = {}
    for primary, metadata in tree.metadata.items():
        for alias in metadata.get("all_addresses", [primary]):
            aliases[str(alias)] = primary
    return aliases

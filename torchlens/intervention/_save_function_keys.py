"""Function registry key serialization for saved intervention specs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .._utils._record_group_table import record_member
from ..intervention._resolver_registry import (
    function_registry_key_from_callable,
)
from ._types import (
    FunctionRegistryKey,
)


def _serialize_function_registry_keys(log: Any) -> list[dict[str, Any]]:
    """Serialize function registry keys from a model log.

    Parameters
    ----------
    log:
        Model log.

    Returns
    -------
    list[dict[str, Any]]
        Function registry key entries.
    """

    entries: list[dict[str, Any]] = []
    for layer in log._layer_list:
        key = _function_key_for_layer(layer)
        if key is None:
            continue
        entries.append({"layer_label": str(layer.layer_label), "key": asdict(key)})
    return entries


def _function_key_for_layer(layer: Any) -> FunctionRegistryKey | None:
    """Return or infer a layer's function registry key.

    Parameters
    ----------
    layer:
        Layer pass log-like object.

    Returns
    -------
    FunctionRegistryKey | None
        Function registry key or ``None`` for source nodes.
    """

    template = record_member(layer, "args_template", None)
    key = getattr(template, "func_id", None)
    if isinstance(key, FunctionRegistryKey):
        return key
    func = getattr(layer, "func", None)
    if func is None:
        return None
    return function_registry_key_from_callable(func)

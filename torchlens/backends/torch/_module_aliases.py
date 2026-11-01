"""Shared-module alias addresses for prepared torch modules.

A module registered under more than one name (``self.alias = self.block``)
has one canonical address, the first registration ``named_modules()`` reports,
and alias addresses for every other registration. These helpers enumerate
those aliases and record them on the parameter logs of the affected modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from ..._data_classes.param import Param


#: Ceiling on the alias addresses one enumeration collects. Every registration
#: path is an address, so a model that shares a shared module's parent many
#: times over grows the count multiplicatively; past the ceiling the remaining
#: alias spellings go unrecorded (no door can refuse them, and a selector spelled
#: with one matches nothing), never a wrong module.
_MAX_ALIAS_ADDRESSES = 4096


def shared_module_addresses(model: nn.Module) -> dict[int, tuple[str, ...]]:
    """Return every registered address of each module registered more than once.

    ``named_modules()`` reports a module object once, under its first
    registration (the canonical address TorchLens labels its calls with). The
    other registrations (``self.alias = self.block``, and every address beneath
    them such as ``alias.fc``) are alias addresses of the same object, as
    ``named_modules(remove_duplicate=False)`` reports them. Unlike that call,
    this walk is bounded on cyclic registrations (a module registered inside
    its own subtree), which it skips.

    Parameters
    ----------
    model:
        Root module; its own address is ``""``.

    Returns
    -------
    dict[int, tuple[str, ...]]
        ``id(module)`` -> ``(canonical, alias, ...)`` for modules with at least
        one alias address; empty when no module is registered twice.
    """

    registrations = 0
    distinct: set[int] = set()
    for module in model.modules():
        for child in module._modules.values():
            if child is not None:
                registrations += 1
                distinct.add(id(child))
    if registrations == len(distinct) and id(model) not in distinct:
        return {}
    canonical = {id(module): address for address, module in model.named_modules()}
    max_depth = len(canonical)
    aliases: dict[int, list[str]] = {}
    collected = 0
    stack: list[tuple[str, nn.Module, int]] = [("", model, 0)]
    while stack and collected < _MAX_ALIAS_ADDRESSES:
        prefix, module, depth = stack.pop()
        if depth >= max_depth:
            continue  # only a cyclic registration nests deeper than the module count
        for name, child in reversed(module._modules.items()):
            if child is None or child is model:
                continue
            address = f"{prefix}.{name}" if prefix else name
            if address != canonical.get(id(child)):
                aliases.setdefault(id(child), []).append(address)
                collected += 1
            stack.append((address, child, depth + 1))
    return {
        module_id: (canonical[module_id], *dict.fromkeys(names))
        for module_id, names in aliases.items()
        if module_id in canonical
    }


def _add_shared_module_aliases(
    model: nn.Module, param_logs: dict[str, Param], param_id_to_address: dict[int, str]
) -> None:
    """Add the addresses of modules registered under a second name to their Params.

    ``model.modules()`` yields a module object once (``self.tied2 = self.tied``),
    so only the duplicate-keeping parameter walk sees the second name.
    """

    for alias_address, param in model.named_parameters(remove_duplicate=False):
        param_log = param_logs.get(param_id_to_address.get(id(param), ""))
        if param_log is None:
            continue
        if alias_address not in param_log.all_addresses:
            param_log.all_addresses.append(alias_address)
        alias_module_address = alias_address.rpartition(".")[0] or "self"
        if alias_module_address not in param_log.all_module_addresses:
            param_log.all_module_addresses.append(alias_module_address)

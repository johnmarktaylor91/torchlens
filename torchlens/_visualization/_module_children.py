"""A module's recorded children, spelled by canonical address.

A module object registered under several names (one ``nn.ReLU`` given to three blocks)
has one canonical address (``act``) and alias addresses (``blocks.0.act``). A parent's
``address_children`` lists the alias it holds, and ``trace.modules`` resolves either
spelling, but every per-module table the collapse planner builds (signals, child maps,
flow graphs) is keyed by the canonical address. Children are therefore read through
this one helper so the planner never meets an alias key.
"""

from __future__ import annotations

from typing import Any


def recorded_child_addresses(trace: Any, module: Any) -> tuple[str, ...]:
    """The canonical addresses of ``module``'s recorded children, in order, without repeats."""

    modules = trace.modules
    children = (
        str(modules[child].address)
        for child in getattr(module, "address_children", ()) or ()
        if child in modules
    )
    return tuple(dict.fromkeys(child for child in children if child != str(module.address)))

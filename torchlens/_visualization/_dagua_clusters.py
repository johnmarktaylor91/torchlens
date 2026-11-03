"""Module cluster boxes for the Dagua bridge (:mod:`torchlens._visualization._dagua_bridge`)."""

from __future__ import annotations

from typing import Any


def _module_chain(entry: Any, vis_mode: str, vis_call_depth: int) -> list[str]:
    """Return module cluster addresses retained for a rendered entry.

    Parameters
    ----------
    entry:
        Layer-pass or layer aggregate entry.
    vis_mode:
        Active visualization mode.
    vis_call_depth:
        Maximum number of nested modules to include.

    Returns
    -------
    list[str]
        Module addresses ordered from outer to inner.
    """

    modules = list(getattr(entry, "modules", None) or [])
    if vis_mode == "rolled":
        modules = [str(m).split(":")[0] for m in modules]
    if vis_call_depth > 0:
        modules = modules[:vis_call_depth]
    return modules


def _add_module_clusters(g: Any, trace: Any, entries: list[Any], vis_mode: str, depth: int) -> None:
    """Add one Dagua cluster per retained module address, nested by address prefix."""
    kept_modules: dict[str, list[str]] = {}
    for entry in entries:
        mods = _module_chain(entry, vis_mode, depth)
        if not mods:
            continue
        node_id = getattr(entry, "layer_label")
        for mod in mods:
            kept_modules.setdefault(mod, []).append(node_id)

    module_logs = {}
    if hasattr(trace, "modules"):
        try:
            module_logs = {
                m.address: m for m in trace.modules if getattr(m, "address", "self") != "self"
            }
        except Exception:
            module_logs = {}

    for mod, members in kept_modules.items():
        if not members:
            continue
        parent = None
        parts = mod.split(".")
        for depth in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:depth])
            if candidate in kept_modules:
                parent = candidate
                break
        module_log = module_logs.get(mod)
        class_name = getattr(module_log, "class_name", "") if module_log is not None else ""
        label = f"@{mod} ({class_name})" if class_name else f"@{mod}"
        g.add_cluster(mod, members, label=label, parent=parent, strict=False)

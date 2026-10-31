"""Keep ``--dist loadgroup`` honouring ``xdist_group`` marks on pytest 9.2+.

pytest-xdist (3.8.0, the newest release) groups tests by appending
``@<group>`` to each worker item's node id in its own
``pytest_collection_modifyitems``, and it does so by writing
``item._nodeid``. pytest 9.2 moved the node id into a structured ``item._id``
(``Node.nodeid`` now returns ``str(self._id)``), so that write lands on a dead
attribute: the controller sees no suffix, ``LoadGroupScheduling`` treats every
test as its own group and spreads one group's tests across workers.

This plugin repeats xdist's stamp on the attribute pytest reads. It runs
``trylast`` (after xdist's hook) and skips items whose id already ends with
the suffix, so it is a no-op on pytest versions where xdist's own write still
works and on a future xdist that fixes the write itself.
"""

from __future__ import annotations

from typing import Any

import pytest


def xdist_group_suffix(item: Any) -> str | None:
    """Return the ``@<group>`` suffix xdist would stamp on ``item``, if any.

    Parameters
    ----------
    item:
        Collected test item.

    Returns
    -------
    str | None
        ``"@" + "_".join(sorted(group names))`` (xdist's own spelling), or
        ``None`` when the item carries no ``xdist_group`` mark.
    """

    names: set[str] = set()
    for mark in item.iter_markers("xdist_group"):
        name = mark.args[0] if mark.args else mark.kwargs.get("name", "default")
        names.add(str(name))
    if not names:
        return None
    return "@" + "_".join(sorted(names))


def stamp_xdist_groups(items: list[Any]) -> int:
    """Append the group suffix to each marked item's structured node id.

    Parameters
    ----------
    items:
        Collected test items of a loadgroup worker session.

    Returns
    -------
    int
        Number of items stamped here (0 when xdist's own write already took).
    """

    stamped = 0
    for item in items:
        suffix = xdist_group_suffix(item)
        if suffix is None or item.nodeid.endswith(suffix):
            continue
        node_id = getattr(item, "_id", None)
        if node_id is None or not hasattr(type(node_id), "parse"):
            # Pre-9.2 node: the id is the plain ``_nodeid`` string xdist writes.
            continue
        # NodeId.parse keeps the exact string it was given as ``str(id)``, so
        # the worker reports and the controller schedules ``<nodeid>@<group>``.
        item._id = type(node_id).parse(item.nodeid + suffix)
        stamped += 1
    return stamped


class XdistGroupStamp:
    """Plugin object: stamp group suffixes on loadgroup worker sessions."""

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: pytest.Config, items: list[Any]) -> None:
        """Stamp the items of a ``--dist loadgroup`` worker after xdist's own hook.

        Parameters
        ----------
        config:
            Worker configuration; xdist sets ``option.loadgroup`` on workers.
        items:
            Collected items.
        """

        if getattr(config.option, "loadgroup", False):
            stamp_xdist_groups(items)


PLUGIN_NAME = "torchlens-xdist-group-stamp"


def register(config: pytest.Config) -> None:
    """Register the stamp plugin once per configuration.

    Parameters
    ----------
    config:
        Active pytest configuration.
    """

    if not config.pluginmanager.has_plugin(PLUGIN_NAME):
        config.pluginmanager.register(XdistGroupStamp(), PLUGIN_NAME)

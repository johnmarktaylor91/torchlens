"""pytest-xdist support in tests/conftest.py: stable test ids.

Under xdist every test runs in a worker process, so a test id that embeds an
object address differs per worker and xdist refuses the run; conftest rejects
such ids at collection.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _conftest_plugin(request: pytest.FixtureRequest) -> Any:
    """Return the live registered tests/conftest.py plugin module."""

    plugin = next(
        (
            plugin
            for plugin in request.config.pluginmanager.get_plugins()
            if hasattr(plugin, "_reject_process_unstable_nodeids")
        ),
        None,
    )
    assert plugin is not None, "tests/conftest.py lost its unstable-nodeid rejection"
    return plugin


def test_address_bearing_nodeids_are_rejected(request: pytest.FixtureRequest) -> None:
    plugin = _conftest_plugin(request)
    clean = [SimpleNamespace(nodeid="tests/x.py::test_a[torchlens.summary]")]
    plugin._reject_process_unstable_nodeids(clean)
    unstable = [SimpleNamespace(nodeid="tests/x.py::test_a[<function <lambda> at 0x7c3b45aa9a80>]")]
    with pytest.raises(pytest.UsageError, match="explicit ids="):
        plugin._reject_process_unstable_nodeids(unstable)

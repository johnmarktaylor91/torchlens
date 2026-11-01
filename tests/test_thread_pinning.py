"""Thread pinning in ``tests/conftest.py``.

Test processes run torch with one intra-op thread so parallel workers never
oversubscribe a runner; ``TORCHLENS_TEST_THREADS`` overrides the count and is
validated at session start.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch


def _conftest_plugin(request: pytest.FixtureRequest) -> Any:
    """Return the live registered tests/conftest.py plugin module."""

    plugin = next(
        (
            plugin
            for plugin in request.config.pluginmanager.get_plugins()
            if hasattr(plugin, "_test_thread_count")
        ),
        None,
    )
    assert plugin is not None, "tests/conftest.py lost its thread pinning"
    return plugin


def test_torch_runs_one_thread_per_test_process_by_default() -> None:
    if os.environ.get("TORCHLENS_TEST_THREADS") not in (None, "1"):
        pytest.skip("thread count overridden by TORCHLENS_TEST_THREADS")
    assert torch.get_num_threads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_thread_override_is_validated(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _conftest_plugin(request)
    monkeypatch.setenv("TORCHLENS_TEST_THREADS", "3")
    assert plugin._test_thread_count() == 3
    for bad in ("0", "-2", "many"):
        monkeypatch.setenv("TORCHLENS_TEST_THREADS", bad)
        with pytest.raises(pytest.UsageError, match="TORCHLENS_TEST_THREADS"):
            plugin._test_thread_count()

"""The duration tripwire's load measurement (tests/conftest.py).

Budgets are compute budgets in reference-host seconds. The slowdown factor is
measured with a fixed CPU-bound probe instead of ``loadavg / cpu_count``, and
torch runs one intra-op thread per test process. These tests pin both halves
of the contract: a genuinely over-budget test still fails, and a measured
slowdown (a slower core, not a longer run queue) no longer fails a test that
fits its budget at reference speed.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
import torch


def _conftest_plugin(request: pytest.FixtureRequest) -> Any:
    """Return the live registered tests/conftest.py plugin module."""

    plugin = next(
        (
            plugin
            for plugin in request.config.pluginmanager.get_plugins()
            if hasattr(plugin, "_item_budget_factor")
        ),
        None,
    )
    assert plugin is not None, "tests/conftest.py lost the measured slowdown factor"
    return plugin


@pytest.fixture
def plugin(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The conftest plugin with its session factor state isolated per test."""

    module = _conftest_plugin(request)
    monkeypatch.setattr(module, "_SESSION_LOAD_FACTOR", 1.0)
    monkeypatch.setattr(module, "_SESSION_MAX_LOAD_FACTOR", 1.0)
    return module


def _probe_at(monkeypatch: pytest.MonkeyPatch, plugin: Any, speed_ratio: float) -> None:
    """Make the probe report a core ``speed_ratio`` times slower than reference."""

    seconds = plugin.SLOWDOWN_PROBE_REFERENCE_SECONDS * speed_ratio
    monkeypatch.setattr(plugin, "_slowdown_probe_seconds", lambda: seconds)


def _is_offender(plugin: Any, base_budget: float, charged: float) -> bool:
    factor = plugin._item_budget_factor(base_budget, charged)
    return charged > base_budget * factor + plugin.DURATION_BUDGET_GRACE_SECONDS


def test_factor_maps_probe_time_onto_the_unchanged_band(plugin: Any) -> None:
    reference = plugin.SLOWDOWN_PROBE_REFERENCE_SECONDS
    assert plugin._slowdown_factor_from_probe(reference) == 1.0
    assert plugin._slowdown_factor_from_probe(reference * 0.4) == 1.0  # faster host: no shrink
    assert plugin._slowdown_factor_from_probe(reference * 2.0) == pytest.approx(2.0)
    assert plugin._slowdown_factor_from_probe(reference * 10.0) == plugin.SLOWDOWN_FACTOR_CAP
    assert plugin.SLOWDOWN_FACTOR_CAP == 4.0
    assert plugin.SMOKE_DURATION_BUDGET_SECONDS == 5.0
    assert plugin.HEAVY_DURATION_BUDGET_SECONDS == 20.0
    assert plugin.DURATION_BUDGET_GRACE_SECONDS == 2.0


def test_over_budget_test_still_fails_on_a_quiet_host(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probe_at(monkeypatch, plugin, 1.0)
    assert _is_offender(plugin, 5.0, 9.5)
    assert _is_offender(plugin, 20.0, 23.0)
    assert not _is_offender(plugin, 5.0, 6.9)


def test_measured_slowdown_no_longer_fails_a_fitting_test(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 4.5 s test read 9 s because contention halved the core's speed after
    # the session started: the re-measure sees 2x and the test passes.
    _probe_at(monkeypatch, plugin, 2.0)
    assert not _is_offender(plugin, 5.0, 9.0)
    assert pytest.approx(2.0) == plugin._SESSION_MAX_LOAD_FACTOR


def test_slowdown_excuses_only_what_was_measured(
    plugin: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _probe_at(monkeypatch, plugin, 1.5)
    assert _is_offender(plugin, 5.0, 12.0)  # budget 1.5 x 5 + 2 = 9.5 s
    _probe_at(monkeypatch, plugin, 100.0)
    assert _is_offender(plugin, 5.0, 23.0)  # capped at 4 x 5 + 2 = 22 s


def test_fitting_test_never_pays_a_re_measure(plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail() -> float:
        raise AssertionError("a test inside its budget must not trigger a probe")

    monkeypatch.setattr(plugin, "_slowdown_probe_seconds", _fail)
    assert not _is_offender(plugin, 5.0, 3.0)


def test_probe_charges_cpu_not_waiting(plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # Time spent off-core (sleeping here, queued behind other processes in
    # practice) must not read as a slower host.
    monkeypatch.setattr(plugin, "_slowdown_probe_body", lambda iterations: time.sleep(0.05))
    assert plugin._slowdown_probe_seconds() < 0.02


def test_torch_runs_one_thread_per_test_process_by_default() -> None:
    if os.environ.get("TORCHLENS_TEST_THREADS") not in (None, "1"):
        pytest.skip("thread count overridden by TORCHLENS_TEST_THREADS")
    assert torch.get_num_threads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_thread_override_is_validated(plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORCHLENS_TEST_THREADS", "3")
    assert plugin._test_thread_count() == 3
    for bad in ("0", "-2", "many"):
        monkeypatch.setenv("TORCHLENS_TEST_THREADS", bad)
        with pytest.raises(pytest.UsageError, match="TORCHLENS_TEST_THREADS"):
            plugin._test_thread_count()

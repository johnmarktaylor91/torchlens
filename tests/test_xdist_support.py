"""pytest-xdist support in tests/conftest.py: stable ids and the duration tripwire.

Under xdist every test runs in a worker process. Two things then break silently
unless conftest handles them: test ids that embed an object address differ per
worker (xdist refuses the run), and the always-on duration-budget tripwire
records offenders on worker sessions that the controller never sees.
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
            if hasattr(plugin, "_import_xdist_duration_ledger")
        ),
        None,
    )
    assert plugin is not None, "tests/conftest.py lost its xdist duration-ledger merge"
    return plugin


def _fake_session(config: object, **attrs: object) -> SimpleNamespace:
    return SimpleNamespace(config=config, exitstatus=0, **attrs)


def test_address_bearing_nodeids_are_rejected(request: pytest.FixtureRequest) -> None:
    plugin = _conftest_plugin(request)
    clean = [SimpleNamespace(nodeid="tests/x.py::test_a[torchlens.summary]")]
    plugin._reject_process_unstable_nodeids(clean)
    unstable = [SimpleNamespace(nodeid="tests/x.py::test_a[<function <lambda> at 0x7c3b45aa9a80>]")]
    with pytest.raises(pytest.UsageError, match="explicit ids="):
        plugin._reject_process_unstable_nodeids(unstable)


def test_worker_offenders_reach_the_controller_tripwire(request: pytest.FixtureRequest) -> None:
    plugin = _conftest_plugin(request)
    controller_config = SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: None))
    offender = ("tests/x.py::test_slow", "smoke", 9.0, 8.0, 7.0)
    for worker_offenders in ([offender], []):
        worker = _fake_session(
            SimpleNamespace(workeroutput={}),
            _tl_duration_budget_offenders=worker_offenders,
            _tl_smoke_family_stats={"tests/x.py::fam": (1.0, 2)},
        )
        plugin._export_worker_duration_ledger(worker)
        node = SimpleNamespace(config=controller_config, workeroutput=worker.config.workeroutput)
        plugin.pytest_testnodedown(node, None)

    controller = _fake_session(controller_config)
    plugin._import_xdist_duration_ledger(controller)
    assert controller._tl_duration_budget_offenders == [offender]
    assert controller._tl_smoke_family_stats == {"tests/x.py::fam": (2.0, 4)}
    plugin._enforce_duration_budget_at_sessionfinish(controller, 0)
    assert controller.exitstatus == 1


def test_family_budget_is_recomputed_from_merged_cells(request: pytest.FixtureRequest) -> None:
    plugin = _conftest_plugin(request)
    controller_config = SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: None))
    # 120 cells split over two workers: the merged budget is 0.1 s x 120 + 2 s grace = 14 s,
    # so 2 x 6.5 s = 13 s stays green while 2 x 7.5 s = 15 s trips.
    for per_worker_total, expected_status in ((6.5, 0), (7.5, 1)):
        controller_config.__dict__.pop("_tl_xdist_duration_ledger", None)
        for _ in range(2):
            worker = _fake_session(
                SimpleNamespace(workeroutput={}),
                _tl_smoke_family_stats={"tests/x.py::fam": (per_worker_total, 60)},
            )
            plugin._export_worker_duration_ledger(worker)
            node = SimpleNamespace(
                config=controller_config, workeroutput=worker.config.workeroutput
            )
            plugin.pytest_testnodedown(node, None)
        controller = _fake_session(controller_config)
        plugin._import_xdist_duration_ledger(controller)
        load_factor = controller_config._tl_xdist_duration_ledger["load_factor"]
        assert controller._tl_smoke_family_budgets["tests/x.py::fam"] == pytest.approx(
            load_factor * 12.0 + 2.0
        )
        plugin._enforce_duration_budget_at_sessionfinish(controller, 0)
        if load_factor == 1.0:
            assert controller.exitstatus == expected_status


def test_non_xdist_sessions_are_untouched(request: pytest.FixtureRequest) -> None:
    plugin = _conftest_plugin(request)
    session = _fake_session(SimpleNamespace(), _tl_duration_budget_offenders=[])
    plugin._export_worker_duration_ledger(session)
    plugin._import_xdist_duration_ledger(session)
    assert session._tl_duration_budget_offenders == []
    assert not hasattr(session, "_tl_smoke_family_stats")


def test_session_header_reports_the_measured_slowdown_factor(
    request: pytest.FixtureRequest,
) -> None:
    plugin = _conftest_plugin(request)
    measured = plugin._smoke_budget_load_factor()
    run_config = SimpleNamespace(option=SimpleNamespace(collectonly=False))
    (line,) = plugin.pytest_report_header(run_config)
    assert f"slowdown factor {measured:.2f}x" in line
    collect_config = SimpleNamespace(option=SimpleNamespace(collectonly=True))
    assert plugin.pytest_report_header(collect_config) == []


def test_terminal_summary_reports_session_and_worker_factors(
    request: pytest.FixtureRequest,
) -> None:
    plugin = _conftest_plugin(request)
    plugin._smoke_budget_load_factor()
    lines: list[str] = []
    reporter = SimpleNamespace(write_line=lines.append)
    config = SimpleNamespace(
        option=SimpleNamespace(collectonly=False),
        _tl_xdist_duration_ledger={"load_factor": 2.5},
    )
    plugin.pytest_terminal_summary(reporter, 0, config)
    (line,) = lines
    assert "slowdown factor" in line and "2.50x largest xdist worker" in line

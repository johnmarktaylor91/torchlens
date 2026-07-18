"""Round-17 VS3 v3 OS-enforcement composition tests."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

from menagerie.crawler.tests.conftest import RealEnvironmentFixture
from menagerie.crawler.tests.test_round19_environment_authority_composition import (
    _run_host_denial_composition,
)

VS3_LANDING_MANIFEST: dict[str, Any] = {
    "findings": (
        "SOL-R16-02",
        "SOL-R16-04",
        "SOL-R16-05",
        "SOL-R16-06",
        "Fable-F4",
    ),
    "production_symbols": {
        "authority": (
            "compile_execution_read_manifest_v3",
            "environment_read_capability",
            "derive_mode_summary",
            "derive_terminal_proof",
        ),
        "driver": (
            "CrawlerDriver._rehydrate_final_authority",
            "_execution_identity",
            "_attempts_from_supervised",
        ),
        "artifact_transactions": (
            "resolve_final_artifact_transaction",
            "rehydrate_artifact_transaction",
        ),
    },
    "real_composition_nodes": (
        "menagerie/crawler/tests/test_round17_vs3_authority_composition.py::"
        "test_manifest_v3_real_os_policy_denies_undeclared_root_member",
        "menagerie/crawler/tests/test_slice_f_driver.py::"
        "test_linux_handoff_attempts_both_deferred_statuses_and_supersedes",
        "menagerie/crawler/tests/test_round17_vs1_v3_composition.py::"
        "test_real_v3_worker_result_awards_through_driver_and_reducer",
        "menagerie/crawler/tests/test_reducer.py::"
        "test_deferred_terminal_positive_capability_probe_is_persisted_and_admitted",
    ),
    "structural_nodes": ("test_vs3_dead_deferral_producer_is_absent",),
}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux real OS policy composition")
def test_manifest_v3_real_os_policy_denies_undeclared_root_member(
    tmp_path: Path,
    real_environment_fixture: RealEnvironmentFixture,
) -> None:
    """A real v3 worker must award declared code and poison an undeclared read.

    Parameters
    ----------
    tmp_path:
        Isolated campaign root.
    real_environment_fixture:
        Strictly bound real prefix used by the shipped v3 compiler and worker.
    """

    _run_host_denial_composition(
        tmp_path,
        real_environment_fixture,
        expected_sandbox="bubblewrap",
    )


def test_vs3_dead_deferral_producer_is_absent() -> None:
    """The superseded driver-owned deferral attempt producer must stay deleted."""

    from menagerie.crawler import driver

    assert not hasattr(driver, "_driver_deferral_attempt")

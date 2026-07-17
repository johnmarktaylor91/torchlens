"""Round-17 VS3 exact-member OS enforcement composition tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from menagerie.crawler.authority import (
    RuntimeLookupDirectory,
    RuntimeMember,
    compile_execution_read_manifest_v2,
    exact_read_capability,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.policy import detect_os_sandbox
from menagerie.crawler.tests.conftest import make_worker_result_v3_mapping
from menagerie.crawler.worker_supervisor import run_isolated_subprocess


HASH = "sha256:" + "a" * 64


@pytest.mark.skipif(sys.platform != "linux", reason="Linux real OS policy composition")
@pytest.mark.parametrize("member_location", ("repository", "environment"))
def test_manifest_v2_real_os_policy_denies_undeclared_root_member(
    tmp_path: Path,
    member_location: str,
) -> None:
    """A caught read of an adjacent undeclared ``.py`` file must poison success.

    Parameters
    ----------
    tmp_path:
        Isolated repository, scratch, and result roots.
    member_location:
        Formerly semantic repository or environment root containing the undeclared member.
    """

    if detect_os_sandbox("Linux") is None:
        pytest.skip("no working Linux OS sandbox tool installed")
    repository = tmp_path / "repository"
    repository.mkdir()
    declared = repository / "declared.py"
    declared.write_text("VALUE = 'declared'\n", encoding="utf-8")
    member_root = repository if member_location == "repository" else tmp_path / "environment"
    member_root.mkdir(exist_ok=True)
    undeclared = member_root / "undeclared.py"
    undeclared.write_text("VALUE = 'undeclared'\n", encoding="utf-8")
    manifest = compile_execution_read_manifest_v2(
        stable_id="m_exact_member",
        work_id="work-exact-member",
        execution_identity=HASH,
        code_manifest_identity=HASH,
        environment_generation=HASH,
        installed_package_inventory_sha256=HASH,
        code_members=(
            RuntimeMember(
                path=declared,
                sha256=hash_bytes(declared.read_bytes()),
                kind="python-source",
                provenance="accepted-model-code",
            ),
        ),
        runtime_members=(),
        lookup_directories=(RuntimeLookupDirectory(path=repository, provenance="lookup-only"),),
    )
    capability = exact_read_capability(manifest)
    assert capability.member_paths == (declared,)
    assert undeclared not in capability.member_paths

    result_root = tmp_path / "result"
    result_root.mkdir()
    receipt_path = result_root / "receipt.json"
    receipt_path.write_text(
        json.dumps(make_worker_result_v3_mapping({}), sort_keys=True),
        encoding="utf-8",
    )
    script = (
        f"from pathlib import Path; print(Path({str(undeclared)!r}).read_text(encoding='utf-8'))"
    )
    observation = run_isolated_subprocess(
        (sys.executable, "-S", "-c", script),
        tmp_path / "scratch",
        timeout_seconds=10,
        rss_limit_bytes=1024**3,
        cwd=repository,
        additional_write_roots=(result_root,),
        execution_read_manifest=manifest,
    )

    if member_location == "repository":
        assert observation.exit_code == 0
    else:
        assert observation.exit_code == 1
        assert str(undeclared) in observation.failed_read_probe_paths
    poisoned = json.loads(receipt_path.read_text(encoding="utf-8"))["diagnostic"]
    policy = poisoned["policy_observation"]
    assert policy["checkpoint_or_weight_read_attempted"] is True
    assert str(undeclared) in policy["checkpoint_paths"]
    assert poisoned["error"]["reason_code"] == "checkpoint-read"


def test_vs3_dead_deferral_producer_is_absent() -> None:
    """The superseded driver-owned deferral attempt producer must stay deleted."""

    from menagerie.crawler import driver

    assert not hasattr(driver, "_driver_deferral_attempt")

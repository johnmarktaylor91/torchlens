"""W051-AGENT: live --fail-on gates on every record shape (AUD-CODE 3.11a),
tier-0 purity on non-directory paths (3.11e), and the corrupted-metadata
cache miss (3.11c)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import torchlens as tl
from tests.test_agent_surface_helpers import (
    RepeatedBlockNet,
    deterministic_input,
    save_clean_artifact,
    save_nan_artifact,
)
from torchlens.agent import call_tool, cli

REPO_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="module")
def clean(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The deterministic clean fixture artifact (module-scoped)."""

    return save_clean_artifact(tmp_path_factory.mktemp("w051_cli"))


@pytest.fixture(scope="module")
def nonfinite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The NaN-poisoned fixture artifact (module-scoped)."""

    return save_nan_artifact(tmp_path_factory.mktemp("w051_cli_nan"))


def _envelope(status: str = "ok", **data: object) -> dict:
    """A minimal synthetic result record."""

    return {"status": status, "data": data, "truncation": None}


@pytest.mark.smoke
def test_fail_on_predicates_read_every_record_shape() -> None:
    """Direct records and per-side compare records both gate; None never trips unverified."""

    complete = {"capture_status": "complete", "capture_verified": None, "structure_only": False}
    ceilinged = {**complete, "capture_verified": False}
    halted = {**complete, "capture_status": "halted"}
    quiet_audit = {"health": {}, "nonfinite": {"n_labels": 0}}
    nan_audit = {"health": {}, "nonfinite": {"n_labels": 2}}
    predicates = cli._FAIL_ON_PREDICATES
    assert predicates["unverified"](_envelope(capture=complete)) is False
    assert predicates["unverified"](_envelope(capture=ceilinged)) is True
    assert predicates["unverified"](
        _envelope(capture={"reference": complete, "subject": ceilinged})
    )
    assert predicates["incomplete"](_envelope(capture=complete, audit=quiet_audit)) is False
    assert predicates["incomplete"](_envelope(capture=halted)) is True
    assert predicates["incomplete"](_envelope(capture={"reference": complete, "subject": halted}))
    assert predicates["incomplete"](_envelope(status="budget_floor_exceeded", capture=complete))
    assert predicates["incomplete"](_envelope(report="no capture block")) is True
    assert predicates["nonfinite"](_envelope(audit=quiet_audit)) is False
    assert predicates["nonfinite"](
        _envelope(audit={"reference": quiet_audit, "subject": nan_audit})
    )


def test_explain_and_diff_gates_are_live(clean: Path, nonfinite: Path) -> None:
    """explain/diff carry the blocks the gates read, so the gates can trip."""

    assert (
        cli.main(["explain", str(clean), "--json", "--fail-on", "unverified,incomplete,nonfinite"])
        == cli.EXIT_OK
    )
    assert (
        cli.main(["explain", str(nonfinite), "--json", "--fail-on", "nonfinite"]) == cli.EXIT_GATE
    )
    assert (
        cli.main(
            [
                "diff",
                str(clean),
                str(clean),
                "--json",
                "--fail-on",
                "nonfinite,unverified,incomplete",
            ]
        )
        == cli.EXIT_OK
    )
    assert (
        cli.main(["diff", str(clean), str(nonfinite), "--json", "--fail-on", "nonfinite"])
        == cli.EXIT_GATE
    )
    assert (
        cli.main(["dump", str(clean), "--json", "--view", "graph", "--fail-on", "incomplete"])
        == cli.EXIT_OK
    )


def test_incomplete_gate_trips_on_a_halted_capture(tmp_path: Path) -> None:
    """A HALTED capture is not complete: --fail-on incomplete exits 1."""

    log = tl.trace(RepeatedBlockNet().eval(), deterministic_input(), halt=tl.func("relu"))
    assert log.outcome.status.value == "halted"
    path = tmp_path / "halted.tlspec"
    tl.save(log, str(path))
    envelope = call_tool("torchlens_overview", {"path": str(path)})
    assert envelope["data"]["capture"]["capture_status"] == "halted"
    assert cli.main(["overview", str(path), "--json", "--fail-on", "incomplete"]) == cli.EXIT_GATE
    assert cli.main(["explain", str(path), "--json", "--fail-on", "incomplete"]) == cli.EXIT_GATE
    assert cli.main(["overview", str(path), "--json", "--fail-on", "unverified"]) == cli.EXIT_OK


def test_tier0_info_on_a_non_directory_path_stays_torch_free(tmp_path: Path) -> None:
    """A plain file (or any non-.tlspec path) hashes with the stdlib only."""

    stray = tmp_path / "stray.tlspec"
    stray.write_bytes(b"not an artifact")
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "from torchlens.agent import cli\n"
        f"rc = cli.main(['info', {str(stray)!r}, '--json'])\n"
        "assert rc == 0, rc\n"
        "assert 'torch' not in sys.modules, 'tier-0 info imported torch'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr[-800:]


def test_local_sha256_matches_the_io_authority(clean: Path) -> None:
    """The tier-0 streaming hash cannot drift from the _io authority."""

    from torchlens._io.manifest import sha256_of_file
    from torchlens.agent._artifacts import _sha256_of_file

    target = clean / "metadata.pkl"
    assert _sha256_of_file(target) == sha256_of_file(target)


@pytest.mark.smoke
def test_corrupted_metadata_is_never_served_from_the_cache(clean: Path, tmp_path: Path) -> None:
    """Same manifest, different metadata.pkl -> cache MISS -> the loader's integrity refusal."""

    healthy = call_tool("torchlens_overview", {"path": str(clean)})  # warms the cache
    assert healthy["status"] == "ok"
    corrupt = tmp_path / "corrupt.tlspec"
    shutil.copytree(clean, corrupt)
    (corrupt / "metadata.pkl").write_bytes(b"garbage")
    with pytest.raises(Exception) as exc:
        call_tool("torchlens_overview", {"path": str(corrupt)})
    assert exc.value.fields["code"] == "bundle_metadata_integrity_refused"
    assert cli.main(["overview", str(corrupt), "--json"]) == cli.EXIT_ARTIFACT
    # The manifest-only preflight still answers (it never unpickles).
    assert (
        call_tool("torchlens_overview", {"path": str(corrupt), "mode": "manifest"})["status"]
        == "ok"
    )
    # And the healthy original still serves from the cache afterwards.
    assert call_tool("torchlens_overview", {"path": str(clean)})["status"] == "ok"

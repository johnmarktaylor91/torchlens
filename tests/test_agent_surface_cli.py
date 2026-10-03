"""F29: the tiered CLI -- tier-0 torch purity, exit codes, CI gates.

Tier-0 verbs must leave torch out of ``sys.modules`` (the memo's purity
test); exit codes are a closed set CI can branch on; ``--fail-on`` gates
READ existing result records, never recompute.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_agent_surface_helpers import (
    save_ablated_artifact,
    save_clean_artifact,
    save_nan_artifact,
)
from torchlens.agent import cli


@pytest.fixture()
def clean(tmp_path: Path) -> Path:
    """The deterministic clean fixture artifact."""

    return save_clean_artifact(tmp_path)


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a fresh subprocess from the repo checkout."""

    return subprocess.run(
        [sys.executable, "-m", "torchlens", *args],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


def test_tier0_verbs_stay_torch_free(clean: Path) -> None:
    """info/schema/guide/version/ls never import torch (the purity gate)."""

    probe = (
        "import sys\n"
        "from torchlens.agent import cli\n"
        "rc = cli.main({args!r})\n"
        "assert rc == 0, rc\n"
        "assert 'torch' not in sys.modules, 'tier-0 verb imported torch'\n"
    )
    for args in (
        ["version", "--json"],
        ["guide"],
        ["schema", "--json"],
        ["info", str(clean), "--json"],
        ["ls", str(clean.parent), "--json"],
    ):
        result = subprocess.run(
            [sys.executable, "-c", probe.format(args=args)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, (args, result.stderr[-500:])


def test_info_emits_the_manifest_envelope(clean: Path) -> None:
    """`info --json` is the manifest-mode overview envelope on stdout."""

    result = _run_cli(["info", str(clean), "--json"])
    assert result.returncode == cli.EXIT_OK
    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "torchlens.agent.overview_manifest.v1"
    assert envelope["data"]["manifest_readable"] is True
    assert envelope["data"]["declared_payload_count"] == 3


@pytest.mark.smoke
def test_missing_artifact_exits_3(tmp_path: Path) -> None:
    """Artifact unreadable -> exit 3, canonical error JSON on stderr."""

    assert cli.main(["info", str(tmp_path / "missing.tlspec")]) == cli.EXIT_ARTIFACT


def test_usage_error_exits_2() -> None:
    """Unknown verbs are usage errors (exit 2)."""

    assert cli.main(["definitely-not-a-verb"]) == cli.EXIT_USAGE


def test_diff_fail_on_mismatch_is_the_ci_wedge(clean: Path, tmp_path: Path) -> None:
    """The adoption wedge: torchlens diff base cand --fail-on mismatch."""

    ablated = save_ablated_artifact(tmp_path)
    same = cli.main(["diff", str(clean), str(clean), "--json", "--fail-on", "mismatch"])
    assert same == cli.EXIT_OK
    changed = cli.main(["diff", str(clean), str(ablated), "--json", "--fail-on", "mismatch"])
    assert changed == cli.EXIT_GATE


def test_overview_fail_on_nonfinite_gates(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--fail-on nonfinite reads the audit block of the overview record."""

    nan_artifact = save_nan_artifact(tmp_path)
    code = cli.main(["overview", str(nan_artifact), "--json", "--fail-on", "nonfinite"])
    assert code == cli.EXIT_GATE
    assert "nonfinite" in capsys.readouterr().err


def test_unknown_fail_on_check_is_a_usage_error(clean: Path) -> None:
    """A typo'd gate name exits 2, naming the closed set."""

    assert cli.main(["overview", str(clean), "--json", "--fail-on", "bogus"]) == cli.EXIT_USAGE


def test_query_verb_round_trips_the_ast(clean: Path, capsys: pytest.CaptureFixture) -> None:
    """`query --query <json>` serves the same rows as the Python transport."""

    code = cli.main(["query", str(clean), "--json", "--query", '{"op": "func", "value": "relu"}'])
    assert code == cli.EXIT_OK
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["header"]["matches_total"] == 3


def test_declared_bytes_matches_the_io_authority(clean: Path) -> None:
    """The tier-0 byte table cannot drift from the _io payload_reader authority."""

    import json as json_module

    from torchlens._io.payload_reader import _DTYPE_NBYTES as io_table, declared_payload_bytes
    from torchlens.agent._artifacts import _DTYPE_NBYTES as agent_table, declared_bytes_and_count

    assert agent_table == io_table
    manifest = json_module.loads((clean / "manifest.json").read_text())
    agent_bytes, _ = declared_bytes_and_count(manifest)
    assert agent_bytes == declared_payload_bytes(manifest)

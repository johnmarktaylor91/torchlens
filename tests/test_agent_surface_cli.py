"""F29: the tiered CLI -- tier-0 torch purity, exit codes, CI gates.

Tier-0 verbs must leave torch out of ``sys.modules`` (the memo's purity
test); exit codes are a closed set CI can branch on; ``--fail-on`` gates
READ existing result records, never recompute.
"""

from __future__ import annotations

import json
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

"""Gallery membership closure (testing memo 5.1: exact count, zero slack).

The gallery is a CLOSED enumerated set: the passed-ID manifest, the test
files on disk, and the RG01-RG20 roster must agree exactly, in every
direction. This gate runs everywhere (no venue skip): it reads source, not
models, so a renamed or vanished scenario is red even on a cold box.
The zero-skip/zero-xfail law is enforced structurally here (no skip or
xfail marker may appear in gallery source) and behaviorally by the venue
runner's exact-passed floor.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke]

GALLERY_DIR = Path(__file__).resolve().parent
REPO_ROOT = GALLERY_DIR.parents[1]
MANIFEST = GALLERY_DIR / "rg_passed_ids.txt"
ENUMERATED_RED = GALLERY_DIR / "rg_enumerated_red_hf5.tsv"

RG_IDS = tuple(f"rg{n:02d}" for n in range(1, 21))


def _manifest_nodes() -> list[str]:
    return [
        line.strip()
        for line in MANIFEST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _source_nodes() -> list[str]:
    nodes = []
    for path in sorted(GALLERY_DIR.glob("test_proofnet_rg_*.py")):
        if path.name == Path(__file__).name:
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                nodes.append(f"tests/workflow_gallery/{path.name}::{node.name}")
    return nodes


def test_manifest_and_source_agree_exactly() -> None:
    """Every source scenario is in the manifest and vice versa."""

    manifest = set(_manifest_nodes())
    source = set(_source_nodes())
    assert manifest == source, (
        f"gallery closure broken: manifest-only={sorted(manifest - source)}"
        f" source-only={sorted(source - manifest)}"
    )


def test_every_rg_id_has_exactly_one_scenario() -> None:
    """RG01-RG20, each claimed by exactly one test node."""

    claimed: dict[str, list[str]] = {rg_id: [] for rg_id in RG_IDS}
    for node in _source_nodes():
        match = re.search(r"test_(rg\d{2})_", node)
        assert match, f"gallery test without an RG id in its name: {node}"
        claimed[match.group(1)].append(node)
    problems = {rg_id: nodes for rg_id, nodes in claimed.items() if len(nodes) != 1}
    assert not problems, f"RG ids without exactly one scenario: {problems}"


def test_gallery_source_carries_no_skip_or_xfail() -> None:
    """The zero-skip/zero-xfail law, enforced on source (structural half).

    The ONE sanctioned skip is the out-of-venue directory gate in
    conftest.py; scenario files themselves may never skip or xfail.
    """

    offenders = []
    for path in sorted(GALLERY_DIR.glob("test_proofnet_rg_*.py")):
        text = path.read_text()
        for token in ("pytest.mark.skip", "pytest.mark.xfail", "pytest.skip(", "pytest.xfail("):
            if token in text and path.name != Path(__file__).name:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, (
        f"skip/xfail spellings inside gallery scenarios: {offenders} -- a"
        " blocked workflow is an enumerated-red or KNOWN-GAP row, never a"
        " skipped member"
    )


def test_enumerated_red_rows_reference_live_scenarios() -> None:
    """Every enumerated-red row names an owner and a scenario that exists;
    the manifest cannot silently quote retired ids."""

    rows = [
        line.split("\t")
        for line in ENUMERATED_RED.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("row_id\t")
    ]
    assert rows, "the enumerated-red manifest is unreadable"
    source_text = " ".join(_source_nodes())
    for row in rows:
        assert len(row) == 5, f"malformed enumerated-red row: {row}"
        row_id, scenario, signature, owner, issue = row
        assert row_id.startswith("RED-"), row
        rg_match = re.search(r"RG(\d{2})", scenario)
        assert rg_match and f"rg{rg_match.group(1)}" in source_text, (
            f"{row_id}: names scenario {scenario!r} with no live gallery node"
        )
        assert owner and issue and signature

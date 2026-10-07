"""Gate-witness manifest + except-skip scanner + EXECUTES-NOWHERE ratchet
(testing memo D4, build items C1-C3; F36 waves B-E).

Extends the three standing skip-audit layers (tests/test_skip_audit.py)
with the wave B-E obligations:

- every NAMED gate (``GATE-ID:`` declaration or venue-skip conftest) has a
  manifest row with a REQUIRED ``executing_leg`` -- per-leg floors cannot
  see "runs nowhere", the manifest can;
- ``pytest.skip`` reachable from an ``except`` handler is flagged (a broad
  except around a model load that skips is invisible forever) and BANNED
  outright in the gallery tier;
- the ``unavailable-ok`` importorskip tier is re-read as EXECUTES-NOWHERE
  with a cap ratcheted toward ZERO (a correctness test that can execute on
  no CI runner moves to the external corpus or the project provisions a
  runner -- it may not remain a permanently green skip);
- the junit gate-witness property helper is proven to land in junitxml.

Every scanner proves red-capability against a planted offender.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Marks are PER-TEST: the whole-tree except-skip walk is heavy (ast-parses
# every test_*.py; ~10s), the rest stay smoke. Marker lint forbids a module
# pytestmark that would stack smoke onto the heavy test.

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
MANIFEST_PATH = TESTS_DIR / "support" / "proofnet" / "gate_manifest.tsv"

#: FROZEN CEILING (burns DOWN only). DESIGN: 31 importorskip targets sat in
#: the unavailable-ok tier at the F36 freeze (2026-08-30) -- each is a gate
#: that EXECUTES NOWHERE in CI. Every row burned down (declared extra
#: lands, runner provisioned, or the test moves to the external corpus)
#: lowers this number in the same commit; it never rises.
#: 31 -> 32 (2026-10-02 ci-fix fast2): `torchaudio` was previously misclassified
#: TEST_EXTRA (claiming the full [test] install has it) when it is deliberately
#: NOT in that extra -- its last release (2.11.0) only loads against torch 2.11
#: and fails (`undefined symbol: torch_library_impl`) against every other
#: pinned torch, pyproject.toml confirms the exclusion is deliberate, and no
#: extra anywhere declares it, so UNAVAILABLE_OK is the textually correct tier
#: per this module's own definitions. The row already existed (reclassified,
#: not new); this is the one-time correction of a wrong classification, not a
#: new gate entering the tier.
EXECUTES_NOWHERE_CEILING = 32

GALLERY_DIR = TESTS_DIR / "workflow_gallery"


def _manifest_rows() -> list[dict[str, str]]:
    rows = []
    header: list[str] | None = None
    for line in MANIFEST_PATH.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts, strict=True)))
    assert header == [
        "gate_id",
        "kind",
        "source",
        "condition",
        "executing_leg",
        "cadence",
        "owner",
        "expiry",
    ]
    return rows


def _declared_gate_ids(root: Path) -> dict[str, str]:
    """Census: every ``GATE-ID:`` declaration in test sources under root."""

    found: dict[str, str] = {}
    for path in root.rglob("conftest.py"):
        for match in re.finditer(r"GATE-ID:\s*([A-Z0-9_]+)", path.read_text()):
            found[match.group(1)] = str(path)
    return found


def test_every_declared_gate_has_a_manifest_row_with_a_leg() -> None:
    """Census -> manifest closure; executing_leg is REQUIRED, never blank."""

    manifest = {row["gate_id"]: row for row in _manifest_rows()}
    declared = _declared_gate_ids(TESTS_DIR)
    unmanifested = {
        gate_id: source for gate_id, source in declared.items() if gate_id not in manifest
    }
    assert not unmanifested, (
        f"gates declared in source with NO manifest row: {unmanifested} -- a"
        " new gate without a leg claim is red at PR time (memo D4)"
    )
    for gate_id, row in manifest.items():
        assert row["executing_leg"].strip(), f"{gate_id}: blank executing_leg"
        assert row["owner"].strip() and row["cadence"].strip(), gate_id


def test_gate_census_is_red_capable(tmp_path: Path) -> None:
    """A planted GATE-ID declaration outside the manifest is caught."""

    planted = tmp_path / "conftest.py"
    planted.write_text('"""GATE-ID: PLANTED_UNMANIFESTED_GATE"""\n')
    found = _declared_gate_ids(tmp_path)
    manifest_ids = {row["gate_id"] for row in _manifest_rows()}
    assert "PLANTED_UNMANIFESTED_GATE" in found
    assert "PLANTED_UNMANIFESTED_GATE" not in manifest_ids


def _except_skip_sites(path: Path) -> list[str]:
    """Every ``pytest.skip(...)`` lexically inside an ``except`` handler."""

    sites: list[str] = []
    tree = ast.parse(path.read_text())

    class _Visitor(ast.NodeVisitor):
        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr in {"skip", "importorskip"}
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "pytest"
                ):
                    sites.append(f"{path.name}:{child.lineno}")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sites


#: FROZEN except-skip inventory at the F36 freeze (2026-08-30), the memo
#: D4 live-instance class measured across the merged tree: each site skips
#: invisibly on any environment where its guarded setup throws. The set is
#: MONOTONE-DOWN: fixing a site (narrow the exception, fail loudly, or move
#: behind a manifested gate) deletes its row here in the same commit; a NEW
#: except-skip anywhere is red immediately.
EXCEPT_SKIP_LEDGER: frozenset[str] = frozenset(
    {
        "test_checkpoint_live_ref_real_models.py:97",
        "test_compare_gate_qwen.py:79",
        "test_distributed_census_topologies.py:109",
        "test_distributed_honesty.py:740",
        "test_distributed_honesty.py:756",
        "test_distributed_tierb_identity.py:76",
        "test_docs_snippets.py:160",
        "test_docs_snippets.py:364",
        "test_input_coercion.py:380",
        "test_intervention_phase7.py:614",
        "test_intervention_phase7.py:621",
        "test_layer_log.py:436",
        "test_lit_bridge_real.py:53",
        "test_mlp_recipes.py:127",
        "test_output_aesthetics.py:1647",
        "test_output_aesthetics.py:1699",
        "test_punchlist_terminology_lint.py:154",
        "test_real_world_models.py:1496",
        "test_real_world_models.py:1728",
        "test_removed_spelling_lint.py:349",
        "test_tlspec_runnable_r41_crossthread_witness.py:465",
        "test_tlspec_runnable_r69_input_contract.py:779",
        "test_transforms_lib_real_models.py:290",
        "test_weightsfree_hf.py:51",
        "test_weightsfree_hf.py:131",
        "test_weightsfree_hf.py:168",
        "test_weightsfree_hf.py:205",
    }
)


@pytest.mark.heavy
def test_no_unledgered_except_skips_anywhere_in_tests() -> None:
    """A skip inside an except handler is invisible forever; the frozen
    inventory only burns down, and a new site is red immediately."""

    found = set()
    for path in TESTS_DIR.rglob("test_*.py"):
        found.update(_except_skip_sites(path))
    new_sites = found - EXCEPT_SKIP_LEDGER
    stale_rows = EXCEPT_SKIP_LEDGER - found
    assert not new_sites, (
        f"NEW pytest.skip sites reachable from except handlers: {sorted(new_sites)}"
        " -- on any cold cache these skip invisibly (memo D4); fail loudly or"
        " gate behind a manifested venue instead"
    )
    assert not stale_rows, (
        f"except-skip ledger rows whose site no longer exists: {sorted(stale_rows)}"
        " -- delete them in the same commit (monotone burn-down; line drift"
        " re-anchors here so moves are conscious)"
    )


def test_gallery_tier_bans_except_skips_outright() -> None:
    """No ledger can license an except-skip in the acceptance gallery."""

    offenders = [site for path in GALLERY_DIR.glob("*.py") for site in _except_skip_sites(path)]
    assert not offenders, f"except-skips in the gallery tier (banned): {offenders}"


def test_except_skip_scanner_is_red_capable(tmp_path: Path) -> None:
    """A planted except-skip is caught; a conditional decoy is not."""

    planted = tmp_path / "test_planted.py"
    planted.write_text(
        "import pytest\n"
        "def test_swallows():\n"
        "    try:\n"
        "        load_model()\n"
        "    except Exception:\n"
        "        pytest.skip('cold cache')\n"
        "def test_decoy():\n"
        "    if False:\n"
        "        pytest.skip('conditional, not except')\n"
    )
    sites = _except_skip_sites(planted)
    assert len(sites) == 1 and sites[0].endswith(":6")


def test_executes_nowhere_tier_only_burns_down() -> None:
    """The unavailable-ok tier IS executes-nowhere: capped, monotone down."""

    from tests.test_skip_audit import IMPORTORSKIP_LEDGER, UNAVAILABLE_OK

    rows = {
        target: note
        for target, (tier, note) in IMPORTORSKIP_LEDGER.items()
        if tier == UNAVAILABLE_OK
    }
    count = len(rows)
    assert count <= EXECUTES_NOWHERE_CEILING, (
        f"{count} executes-nowhere importorskip targets exceed the frozen"
        f" ceiling {EXECUTES_NOWHERE_CEILING}: a new gate entered the tier"
        " that no CI runner executes -- declare its extra/leg instead"
    )
    assert count == EXECUTES_NOWHERE_CEILING, (
        f"only {count} executes-nowhere rows remain but the ceiling reads"
        f" {EXECUTES_NOWHERE_CEILING}: lower it in the same commit"
    )


def test_junit_witness_property_lands_in_junitxml(tmp_path: Path) -> None:
    """Witness mechanics step 2: a guarded branch can emit its gate id as a
    junit property that survives into the report the union job reads."""

    test_file = tmp_path / "test_witness_probe.py"
    test_file.write_text(
        "def test_emits_gate_witness(record_property):\n"
        "    record_property('gate_witness', 'R1_OFFLINE_VENUE')\n"
    )
    junit = tmp_path / "out.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
            "-o",
            "junit_family=xunit1",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout[-800:]
    content = junit.read_text()
    assert 'name="gate_witness"' in content and "R1_OFFLINE_VENUE" in content, (
        "the junit witness channel is dead: record_property did not reach"
        " the report the weekly union job consumes"
    )

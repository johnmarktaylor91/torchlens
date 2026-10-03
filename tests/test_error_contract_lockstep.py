"""Error-refusal contract doc <-> code LOCKSTEP (grind B1-11 / matrix R64+R11).

``docs/reference/error_refusal_contract.md`` is the public vocabulary of stable
refusal codes, and its closing sentence makes adding or renaming a code a
documented public change. Until this module existed nothing enforced that:
the doc's ~145 rows and the ``code=`` literals in ``torchlens/`` could drift
apart silently (the exact mechanism gap behind the 2026-08 conflict-door
incident, fixed in 7f7623b9 but previously unguarded).

The scanner derives the code universe from the SOURCE (every ``code="..."``
constructor argument or ``code = "..."`` class attribute under ``torchlens/``)
and the doc universe from the contract table, then demands exact set equality
in both directions. Codes governed by OTHER contract docs (runnable, merged,
capture-outcome vocabularies) are declared through enum members or differently
named attributes, so they do not enter this scanner's universe; if one ever
does, the equality check goes red and the collision is adjudicated explicitly.

Everything here is smoke-tier: pure file scanning, no capture.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTRACT_DOC = _REPO_ROOT / "docs" / "reference" / "error_refusal_contract.md"
_PACKAGE_ROOT = _REPO_ROOT / "torchlens"

# Matches both the strict-constructor spelling ``code="..."`` and the
# class-attribute spelling ``code = "..."`` / ``code: str = "..."`` used by
# the backend registry error classes. ``\b`` keeps ``zip_code =`` style names
# out because ``_`` is a word character.
_CODE_PATTERN = re.compile(r'\bcode(?:\s*:\s*str)?\s*=\s*"([a-z0-9_]+)"')

_DOC_ROW_PATTERN = re.compile(r"^\| `([a-z0-9_]+)` \| (.*?) \| (.*?) \|$")

# Constant-spelled refusal identifiers (surfaced on ``fields["kind"]`` rather
# than ``fields["code"]``, spelled as module constants rather than inline
# literals). The inline-literal scanner cannot see them, which is exactly how
# the three collective refusal kinds shipped documented NOWHERE (grind R25-5):
# each is enrolled here as ``code -> (declaring file, constant name)`` and
# verified against the source, so a constant rename or string drift fails this
# gate exactly like an inline-code drift. New constant-spelled kinds must be
# enrolled here AND documented in the contract's constant-spelled table.
_CONSTANT_SPELLED_CODES: dict[str, tuple[str, str]] = {
    "ambiguous_group_lifetime": (
        "torchlens/distributed/_lifecycle.py",
        "AMBIGUOUS_GROUP_LIFETIME",
    ),
    "uncaptured_collective_op": (
        "torchlens/distributed/_recognizer.py",
        "UNCAPTURED_COLLECTIVE_OP",
    ),
    "wildcard_recv_unsupported": (
        "torchlens/backends/torch/collectives.py",
        "WILDCARD_RECV_UNSUPPORTED",
    ),
    "intervention_fire_results_unrecordable": (
        "torchlens/backends/torch/_ops_interventions.py",
        "INTERVENTION_FIRE_RESULTS_UNRECORDABLE",
    ),
    "intervention_fire_results_cleanup_failed": (
        "torchlens/backends/torch/_ops_interventions.py",
        "INTERVENTION_FIRE_RESULTS_CLEANUP_FAILED",
    ),
    # L7a structure-only chokepoint codes: raised only through the capability
    # rows (require_structure_only_capability reads row.refusal_code), so the
    # inline-literal scanner cannot see them; each is spelled as a module
    # constant beside the table.
    "structure_only_measurements_unsupported": (
        "torchlens/capture/structure_only.py",
        "STRUCTURE_ONLY_MEASUREMENTS_UNSUPPORTED",
    ),
    "structure_only_runnable_unsupported": (
        "torchlens/capture/structure_only.py",
        "STRUCTURE_ONLY_RUNNABLE_UNSUPPORTED",
    ),
    "structure_only_replay_unsupported": (
        "torchlens/capture/structure_only.py",
        "STRUCTURE_ONLY_REPLAY_UNSUPPORTED",
    ),
    "structure_only_validation_unsupported": (
        "torchlens/capture/structure_only.py",
        "STRUCTURE_ONLY_VALIDATION_UNSUPPORTED",
    ),
    "structure_only_backward_unsupported": (
        "torchlens/capture/structure_only.py",
        "STRUCTURE_ONLY_BACKWARD_UNSUPPORTED",
    ),
    "structure_only_episode_unsupported": (
        "torchlens/capture/structure_only.py",
        "STRUCTURE_ONLY_EPISODE_UNSUPPORTED",
    ),
}


def _documented_codes() -> dict[str, tuple[str, str]]:
    """Return the contract table as ``code -> (refusal, remedy class)``.

    Returns
    -------
    dict[str, tuple[str, str]]
        Every documented stable code with its two prose columns.
    """

    rows: dict[str, tuple[str, str]] = {}
    for line in _CONTRACT_DOC.read_text().splitlines():
        match = _DOC_ROW_PATTERN.match(line.strip())
        if match is None:
            continue
        code, refusal, remedy = match.groups()
        assert code not in rows, f"duplicate contract row for code {code!r}"
        rows[code] = (refusal, remedy)
    return rows


def _source_codes() -> dict[str, set[str]]:
    """Return every stable refusal code declared in the package source.

    Returns
    -------
    dict[str, set[str]]
        Mapping from code to the repo-relative files declaring it.
    """

    declared: dict[str, set[str]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        text = path.read_text()
        for match in _CODE_PATTERN.finditer(text):
            declared.setdefault(match.group(1), set()).add(
                str(path.relative_to(_PACKAGE_ROOT.parent))
            )
    for code, (rel_path, constant_name) in _CONSTANT_SPELLED_CODES.items():
        source_path = _PACKAGE_ROOT.parent / rel_path
        if not source_path.exists():
            # Enrollment integrity is enforced separately by
            # test_constant_spelled_codes_resolve_in_source against the real
            # tree; the red-capability test points _PACKAGE_ROOT elsewhere.
            continue
        pattern = re.compile(
            rf'^{re.escape(constant_name)}\s*(?::\s*[^=]+)?=\s*"{re.escape(code)}"', re.M
        )
        if pattern.search(source_path.read_text()):
            declared.setdefault(code, set()).add(rel_path)
    return declared


def test_contract_doc_parses_to_a_nonempty_table() -> None:
    """The scanner actually finds the contract table (anti-vacuity guard)."""

    rows = _documented_codes()
    assert len(rows) > 100, f"contract table parse collapsed: {len(rows)} rows"
    for code, (refusal, remedy) in rows.items():
        assert refusal.strip(), f"contract row {code!r} has an empty refusal column"
        assert remedy.strip(), f"contract row {code!r} has an empty remedy column"


def test_source_scanner_finds_a_nonempty_universe() -> None:
    """The source scan actually matches code declarations (anti-vacuity guard)."""

    declared = _source_codes()
    assert len(declared) > 100, f"source code scan collapsed: {len(declared)} codes"


def test_every_documented_code_exists_in_source() -> None:
    """No contract row names a code the package no longer raises."""

    stale = set(_documented_codes()) - set(_source_codes())
    assert stale == set(), (
        "contract doc rows with no matching code= declaration in torchlens/ "
        f"(renamed or deleted without a doc update): {sorted(stale)}"
    )


def test_every_source_code_is_documented() -> None:
    """No refusal code ships without a contract row."""

    declared = _source_codes()
    undocumented = set(declared) - set(_documented_codes())
    assert undocumented == set(), (
        "refusal codes declared in torchlens/ but absent from "
        "docs/reference/error_refusal_contract.md: "
        + "; ".join(
            f"{code} ({', '.join(sorted(declared[code]))})" for code in sorted(undocumented)
        )
    )


def test_constant_spelled_codes_resolve_in_source() -> None:
    """Every enrolled constant-spelled kind exists with its exact string value.

    A rename of the constant, a move to another module, or a drift of the
    string value fails HERE with the enrollment row named, rather than as an
    opaque missing-code diff in the set-equality tests.
    """

    for code, (rel_path, constant_name) in _CONSTANT_SPELLED_CODES.items():
        source_path = _PACKAGE_ROOT.parent / rel_path
        assert source_path.exists(), (
            f"constant-spelled code {code!r} enrolled against a missing file {rel_path}"
        )
        pattern = re.compile(
            rf'^{re.escape(constant_name)}\s*(?::\s*[^=]+)?=\s*"{re.escape(code)}"', re.M
        )
        assert pattern.search(source_path.read_text()), (
            f'constant {constant_name} = "{code}" not found in {rel_path}; '
            "update the enrollment table and the contract doc's constant-spelled "
            "table in the same change as the source"
        )


def test_constant_spelled_codes_are_documented() -> None:
    """Every enrolled constant-spelled kind has a contract-doc row."""

    documented = set(_documented_codes())
    missing = set(_CONSTANT_SPELLED_CODES) - documented
    assert missing == set(), f"constant-spelled kinds missing contract-doc rows: {sorted(missing)}"


def test_lockstep_scanners_are_red_capable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planted drift in either direction is reported (the gate can fail)."""

    doc = tmp_path / "contract.md"
    doc.write_text(
        "| Code | Refusal | Remedy class |\n"
        "|---|---|---|\n"
        "| `documented_only_code` | A refusal | A remedy |\n"
        "| `shared_code` | A refusal | A remedy |\n"
    )
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text(
        'X = dict(code="shared_code")\nY = dict(code="source_only_code")\n'
    )
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_CONTRACT_DOC", doc)
    monkeypatch.setattr(module, "_PACKAGE_ROOT", package)

    with pytest.raises(AssertionError, match="documented_only_code"):
        test_every_documented_code_exists_in_source()
    with pytest.raises(AssertionError, match="source_only_code"):
        test_every_source_code_is_documented()

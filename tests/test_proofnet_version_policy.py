"""Band-edge lockstep: claim -> named leg -> executable evidence (memo D3/B3).

ONE support-policy table (tests/support/proofnet/support_policy.tsv) is the
authority; this suite holds pyproject's declared bands, the per-leg
constraints files, the installed environment, and the HF_5_CANDIDATE
enumerated-red manifest in lockstep with it. An un-executed band edge gets
narrowed, not documented; an OPEN-ended band is legal only with a daily
latest canary.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

pytestmark = [pytest.mark.smoke]

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "tests" / "support" / "proofnet" / "support_policy.tsv"
CONSTRAINTS_DIR = REPO_ROOT / "tests" / "support" / "proofnet" / "constraints"
ENUMERATED_RED = REPO_ROOT / "tests" / "workflow_gallery" / "rg_enumerated_red_hf5.tsv"
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text()

#: Legs whose constraints files THIS table owns; `lowest_direct` is the
#: pre-existing nightly floor leg (documented in pyproject) and `-` means
#: no such leg for the row.
OWNED_LEGS = {"hf_floor", "hf_4_current", "hf_5_candidate", "latest_canary"}


def _policy_rows() -> list[dict[str, str]]:
    rows = []
    header: list[str] | None = None
    for line in POLICY_PATH.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts, strict=True)))
    assert header == [
        "package",
        "claimed_band",
        "open_ended",
        "floor_leg",
        "current_leg",
        "candidate_leg",
        "canary",
    ]
    return rows


def _declared_band(package: str) -> str:
    """Extract the declared band for one package from pyproject text."""

    matches = re.findall(rf'"{package}([><=~!][^"]*)"', PYPROJECT)
    if package == "torch":
        # torch's floor is python-conditioned; the lowest condition is the claim.
        conditioned = [m.split(";")[0] for m in matches]
        assert conditioned, "torch band missing from pyproject"
        return sorted(conditioned)[0]
    assert matches, f"{package}: no declared band found in pyproject"
    bare = sorted({m.split(";")[0] for m in matches})
    assert len(bare) == 1, (
        f"{package}: pyproject declares CONFLICTING bands {bare} -- one"
        " support policy, one spelling"
    )
    return bare[0]


def _leg_pins(leg: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    path = CONSTRAINTS_DIR / f"{leg}.txt"
    assert path.is_file(), f"constraints file missing for named leg {leg}"
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line.strip())
        pin = str(requirement.specifier)
        assert pin.startswith("=="), f"{leg}: {line!r} is not an exact reviewed pin"
        pins[requirement.name] = pin.removeprefix("==")
    return pins


def test_policy_table_matches_pyproject_bands() -> None:
    """Lockstep direction 1: the table's claimed bands ARE pyproject's."""

    for row in _policy_rows():
        declared = _declared_band(row["package"])
        assert row["claimed_band"] == declared, (
            f"{row['package']}: support-policy table says"
            f" {row['claimed_band']!r}, pyproject declares {declared!r} --"
            " update BOTH in one commit (one authority, two mirrors)"
        )


def test_every_named_leg_has_a_constraints_file_with_reviewed_pins() -> None:
    """Lockstep direction 2: a claimed leg is a file, not a sentence."""

    for row in _policy_rows():
        for column in ("floor_leg", "current_leg", "candidate_leg"):
            leg = row[column]
            if leg in OWNED_LEGS and leg != "latest_canary":
                pins = _leg_pins(leg)
                if row["package"] in pins:
                    band = SpecifierSet(row["claimed_band"])
                    inside = Version(pins[row["package"]]) in band
                    if column == "candidate_leg":
                        continue  # the candidate leg is outside-claim by design
                    assert inside, (
                        f"{leg}: pins {row['package']}=={pins[row['package']]}"
                        f" OUTSIDE the claimed band {row['claimed_band']}"
                    )


def test_floor_leg_pins_sit_at_the_band_floor_edge() -> None:
    """The floor leg executes the CLAIMED minimum, not a comfortable middle."""

    pins = _leg_pins("hf_floor")
    transformers_band = next(
        row["claimed_band"] for row in _policy_rows() if row["package"] == "transformers"
    )
    floor_spelling = re.search(r">=([0-9.]+)", transformers_band)
    assert floor_spelling, transformers_band
    claimed_floor = Version(floor_spelling.group(1))
    pinned = Version(pins["transformers"])
    assert pinned.release[:2] == claimed_floor.release[:2], (
        f"HF_FLOOR pins transformers {pinned}, but the claimed floor line is"
        f" {claimed_floor}.x -- the floor edge is not being executed"
    )


def test_open_ended_bands_carry_a_latest_canary() -> None:
    """No upper bound without a daily canary obligation (the D3 invariant)."""

    for row in _policy_rows():
        band = row["claimed_band"]
        has_ceiling = "<" in band or band.startswith("~=") or "==" in band
        declared_open = row["open_ended"] == "yes"
        assert declared_open == (not has_ceiling), (
            f"{row['package']}: open_ended={row['open_ended']} but band is"
            f" {band!r} -- the table lies about the ceiling"
        )
        if declared_open:
            assert row["canary"] == "latest_canary", (
                f"{row['package']}: an OPEN band with no latest canary is an unexecuted claim"
            )


def test_installed_environment_is_inside_a_named_leg() -> None:
    """The venue we actually test in must be one of the named legs' worlds:
    inside the claimed band, or exactly the candidate leg's pin."""

    from importlib import metadata

    candidate_pins = _leg_pins("hf_5_candidate")
    for row in _policy_rows():
        package = row["package"]
        try:
            installed = Version(metadata.version(package).split("+")[0])
        except metadata.PackageNotFoundError:
            continue
        inside_claim = installed in SpecifierSet(row["claimed_band"])
        is_candidate_world = package in candidate_pins and installed == Version(
            candidate_pins[package]
        )
        assert inside_claim or is_candidate_world, (
            f"{package} {installed} is outside the claimed band"
            f" {row['claimed_band']} AND is not the candidate leg's pin --"
            " this venue tests a world no leg claims"
        )


def test_candidate_leg_pin_matches_the_enumerated_red_venue() -> None:
    """The enumerated-red manifest's measured venue IS the candidate pin
    (a manifest measured on a different version proves nothing)."""

    pins = _leg_pins("hf_5_candidate")
    manifest_text = ENUMERATED_RED.read_text()
    measured = re.search(r"transformers ([0-9][0-9.]*)", manifest_text)
    assert measured, "the enumerated-red manifest does not state its venue"
    assert measured.group(1) == pins["transformers"], (
        f"enumerated-red manifest measured on transformers {measured.group(1)}"
        f" but the candidate leg pins {pins['transformers']} -- re-measure or"
        " re-pin in the same commit"
    )

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

from packaging.requirements import Requirement

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


#: The torch-FLOOR CI rows (tests.yml) pin torchvision to the exact release
#: upstream PAIRED with torch's own floor patch for that row -- torchvision
#: 0.18+ hard-requires torch>=2.3, so a floor row running torch's OWN claimed
#: floor (>=2.1) physically cannot also satisfy torchvision's independent
#: >=0.18 extras commitment (vision-shims/test/all-stretch). This is a
#: reviewed, NAMED floor-pairing leg exactly like hf_5_candidate is a named
#: pre-release leg: torchvision there is pulled in incidentally (the floor
#: rows install dev,tabular,viz, never test/vision-shims), never claimed as
#: "TorchLens supports torchvision 0.16/0.17", so it is an explicit
#: out-of-band exception, not a silent loosening of the >=0.18 commitment.
_TORCH_FLOOR_PAIRED_TORCHVISION: frozenset[str] = frozenset({"0.16.2", "0.17.2"})

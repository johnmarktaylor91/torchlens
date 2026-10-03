"""Mypy strictness ratchet: the flag set may only ever GROW.

R68 (3 consecutive hunt passes): the claimed mypy "package ratchet" was a
comment — zero strictness flags beyond ``check_untyped_defs``, nothing pinning
the config, so a lane could narrow it (or the claim could stay aspirational)
without any gate noticing. This lockstep test makes the ratchet mechanical:
every flag in ``_RATCHETED_TRUE_FLAGS`` must read ``true`` in
``[tool.mypy]``; removing or flipping one goes red here.

Each flag joined at measured ZERO cost against the pinned mypy (2026-08-15:
``mypy torchlens/`` clean under all of them). Growing the set is welcome —
add the flag here in the same change that lands it in pyproject.

Parsing is line/regex-based on purpose (tomllib is 3.11+; the suite runs a
3.10 leg — the test_order_isolation_infra precedent).
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Flags that must read ``true`` under [tool.mypy]. GROW-ONLY: never remove an
#: entry to make a gate pass — that is the silent ratchet-narrowing this test
#: exists to prevent.
_RATCHETED_TRUE_FLAGS = (
    "warn_unused_configs",
    "check_untyped_defs",
    "no_implicit_optional",
    "disallow_incomplete_defs",
    "strict_equality",
)


def _mypy_section(pyproject_text: str) -> str:
    """Return the raw ``[tool.mypy]`` section body.

    Parameters
    ----------
    pyproject_text:
        Full pyproject.toml text.

    Returns
    -------
    str
        Section text up to the next table header.
    """

    match = re.search(
        r"^\[tool\.mypy\]\n(.*?)(?=^\[)", pyproject_text, flags=re.MULTILINE | re.DOTALL
    )
    assert match, "pyproject.toml lost its [tool.mypy] section entirely"
    return match.group(1)


def _flag_values(section: str) -> dict[str, str]:
    """Parse ``name = value`` lines from a TOML section body."""

    return {
        name: value.strip()
        for name, value in re.findall(r"^([A-Za-z_]+)\s*=\s*(\S+)", section, flags=re.MULTILINE)
    }


def test_mypy_strictness_flags_never_narrow() -> None:
    """Every ratcheted strictness flag must remain ``true`` in [tool.mypy]."""

    section = _mypy_section((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    values = _flag_values(section)
    broken = [flag for flag in _RATCHETED_TRUE_FLAGS if values.get(flag) != "true"]
    assert not broken, (
        f"mypy strictness flags narrowed or vanished: {broken}. The ratchet is "
        "grow-only (R68): restore each flag; if a flag must genuinely retire, "
        "that is an explicit owner decision recorded in both files, never a "
        "quiet config edit."
    )


def test_mypy_ratchet_lock_is_red_capable() -> None:
    """A planted narrowed config is flagged (red-capability self-test)."""

    planted = 'python_version = "3.11"\ncheck_untyped_defs = false\nstrict_equality = true\n'
    values = _flag_values(planted)
    broken = [flag for flag in _RATCHETED_TRUE_FLAGS if values.get(flag) != "true"]
    assert "check_untyped_defs" in broken
    assert "no_implicit_optional" in broken  # missing entirely also trips
    assert "strict_equality" not in broken

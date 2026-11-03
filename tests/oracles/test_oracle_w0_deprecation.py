"""The bidirectional call-time deprecation gate + plants (build item 5).

Static side (zero emission sites) is owned by
scripts/checks/deprecation_inventory.py and CITED, never duplicated; this gate
owns the CALL-TIME behavior root (D6: five of seven shims at the panel's SHA
presented the implementation's metadata everywhere and revealed themselves
only when called). The registry is empty today; the plants prove both red
directions on planted registries so emptiness is never a dead channel.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ._deprecation import (
    DeprecatedDoor,
    audit_registered_door,
)

# ---------------------------------------------------------------------------
# Plants: both directions of the iff must be able to go red.
# ---------------------------------------------------------------------------


def _planted_warning_door() -> None:
    """A door that warns deprecation on call (the undocumented direction)."""

    warnings.warn("planted_old is deprecated; use planted_new", DeprecationWarning, stacklevel=2)


def _planted_silent_door() -> None:
    """A door that does NOT warn (the silently-overwritten-shim direction)."""


def test_plant_registered_nonwarning_door_goes_red(tmp_path: Path) -> None:
    """PLANT: a registered door that stays silent is caught."""

    door = DeprecatedDoor(door="planted.old", replacement="torchlens.trace", since="2026-08-26")
    findings = audit_registered_door(door, _planted_silent_door)
    assert any("silently overwritten" in finding for finding in findings), findings

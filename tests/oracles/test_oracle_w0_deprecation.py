"""The bidirectional call-time deprecation gate + plants (build item 5).

Static side (zero emission sites) is owned by
tests/test_deprecation_inventory.py and CITED, never duplicated; this gate
owns the CALL-TIME behavior root (D6: five of seven shims at the panel's SHA
presented the implementation's metadata everywhere and revealed themselves
only when called). The registry is empty today; the plants prove both red
directions on planted registries so emptiness is never a dead channel.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from ._deprecation import (
    DeprecatedDoor,
    audit_registered_door,
    call_time_deprecations,
    load_deprecated_doors,
    resolve_dotted,
)
from ._invocation_templates import SEED_TEMPLATES

pytestmark = pytest.mark.smoke


def test_registry_is_empty_and_loads() -> None:
    """The package is deprecation-free; the registry says so as DATA."""

    assert load_deprecated_doors() == (), (
        "a deprecation-door row appeared; interim-phase policy is "
        "remove-and-rename (tests/test_deprecation_inventory.py) -- a real "
        "shim needs BOTH the registry row and a conscious policy change"
    )


@pytest.mark.parametrize("template", SEED_TEMPLATES, ids=lambda t: t.door)
def test_unregistered_doors_never_warn_deprecation_at_call_time(template) -> None:
    """Direction 2 of the iff: no unregistered door warns when CALLED.

    Rooted in observed calls through the shared invocation templates (one
    asset, three consumers: purity harness, option witnesses, this gate).
    """

    registered = frozenset(row.door for row in load_deprecated_doors())
    if template.door in registered:
        pytest.skip("registered doors are audited by the direction-1 gate")
    emitted = call_time_deprecations(template.invoke)
    assert emitted == (), (
        f"{template.door} emitted DeprecationWarning at call time without a "
        f"registry row (undocumented deprecation): {[str(m.message) for m in emitted]}"
    )


def test_registered_doors_honor_the_contract() -> None:
    """Direction 1 of the iff: every registered door warns, names, resolves.

    Vacuously green on the empty registry; the plants below keep the
    machinery honest so this can never rot into a dead check.
    """

    for door in load_deprecated_doors():
        invoke = resolve_dotted(door.door)
        findings = audit_registered_door(door, invoke)
        assert findings == (), findings


def test_seed_templates_cover_seven_plus_doors_or_disclose() -> None:
    """The call-time gate's door coverage is COUNTED, never implied.

    The memo sized the gate at "7+ doors" against the panel-SHA shim set;
    P00's batch-8 deletion removed those shims, so today's honest coverage
    is the seed template set. The count is published; item 7 (Wave 1) grows
    templates to 12-18 and this disclosure keeps the gap visible.
    """

    print(
        f"call-time deprecation gate coverage: {len(SEED_TEMPLATES)} doors via "
        f"invocation templates: {[t.door for t in SEED_TEMPLATES]}; "
        "shim set deleted at P00, registry empty by policy"
    )
    assert len(SEED_TEMPLATES) >= 5


# ---------------------------------------------------------------------------
# Plants: both directions of the iff must be able to go red.
# ---------------------------------------------------------------------------


def _planted_warning_door() -> None:
    """A door that warns deprecation on call (the undocumented direction)."""

    warnings.warn("planted_old is deprecated; use planted_new", DeprecationWarning, stacklevel=2)


def _planted_silent_door() -> None:
    """A door that does NOT warn (the silently-overwritten-shim direction)."""


def test_plant_undocumented_deprecation_goes_red() -> None:
    """PLANT: an unregistered door that warns at call time is caught."""

    emitted = call_time_deprecations(_planted_warning_door)
    assert emitted, "the call-time channel missed a planted DeprecationWarning"


def test_plant_registered_nonwarning_door_goes_red(tmp_path: Path) -> None:
    """PLANT: a registered door that stays silent is caught."""

    door = DeprecatedDoor(door="planted.old", replacement="torchlens.trace", since="2026-08-26")
    findings = audit_registered_door(door, _planted_silent_door)
    assert any("silently overwritten" in finding for finding in findings), findings


def test_plant_registered_door_with_dead_replacement_goes_red() -> None:
    """PLANT: a registered replacement that does not resolve is caught."""

    door = DeprecatedDoor(
        door="planted.old", replacement="torchlens.no_such_replacement", since="2026-08-26"
    )
    findings = audit_registered_door(door, _planted_warning_door)
    assert any("does not resolve" in finding for finding in findings), findings


def test_plant_warning_that_names_no_replacement_goes_red() -> None:
    """PLANT: a warning that omits the registered replacement is caught."""

    door = DeprecatedDoor(door="planted.old", replacement="torchlens.extract", since="2026-08-26")
    findings = audit_registered_door(door, _planted_warning_door)
    assert any("does not name" in finding for finding in findings), findings


def test_plant_registry_row_parses(tmp_path: Path) -> None:
    """PLANT: a real registry row round-trips through the loader."""

    registry = tmp_path / "deprecated_doors.tsv"
    registry.write_text("door\treplacement\tsince\nplanted.old\ttorchlens.trace\t2026-08-26\n")
    rows = load_deprecated_doors(registry)
    assert rows == (
        DeprecatedDoor(door="planted.old", replacement="torchlens.trace", since="2026-08-26"),
    )

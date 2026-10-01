"""Purity/state harness skeleton gates: templates, controls, vocabularies.

Wave 0 owes the SKELETON (invocation-template schema + history-cell schema +
the D10 purity vocabulary), not the harness: build item 7 (Wave 1) wires the
snapshot union and the postconditions over these exact templates. "Templates,
not compute, are the cost" (fact 11: 10 of 12 doors refused a 0.7 s sweep for
want of templates), so the seed templates land runnable and positive-
controlled from day one.
"""

from __future__ import annotations

import pickle

import pytest

from ._invocation_templates import SEED_TEMPLATES, _fixture_model
from ._registry import PURITY_CONTRACTS, PURITY_MECHANISMS

pytestmark = pytest.mark.smoke


def test_template_ids_and_doors_are_unique() -> None:
    """Template ids and doors collide never; the set is countable data."""

    ids = [template.template_id for template in SEED_TEMPLATES]
    doors = [template.door for template in SEED_TEMPLATES]
    assert len(set(ids)) == len(ids)
    assert len(set(doors)) == len(doors)


def test_every_template_declares_a_positive_control() -> None:
    """D7: every fixture channel names its liveness check."""

    for template in SEED_TEMPLATES:
        assert callable(template.positive_control), template.template_id
        assert callable(template.invoke), template.template_id


@pytest.mark.parametrize("template", SEED_TEMPLATES, ids=lambda t: t.door)
def test_positive_control_passes_on_the_live_tree(template) -> None:
    """Every seed template's measurement channel is ALIVE.

    A dead channel exonerates real defects (the panel's function-local
    pickle fixture reported "False -> False"); the control runs BEFORE any
    wave-1 verdict is trusted.
    """

    template.positive_control()


def test_fixture_model_is_module_level_picklable() -> None:
    """CF-022: the fixture spec itself passes its own positive control.

    A function-local fixture class cannot pickle, which kills the pickle
    postcondition channel before the measurement. The shared fixture must
    be picklable BEFORE any capture so the wave-1 pickle postcondition
    measures TorchLens, not the fixture.
    """

    payload = pickle.dumps(_fixture_model())
    assert payload, "fixture model failed to pickle; the channel is dead pre-capture"


def test_state_contracts_stay_unset_pending_fork_a() -> None:
    """FORK-A is JMT's call; no template pre-empts it.

    The 2-1 PURE_OBSERVER lean is real but was never stress-tested as a
    stable majority (the two moving labs swapped sides in round 3); the
    declared cell stays UNSET until the ruling, and the harness is identical
    under both branches.
    """

    for template in SEED_TEMPLATES:
        assert template.state_contract in PURITY_CONTRACTS
        assert template.state_contract == "UNSET", (
            f"{template.template_id} pre-empted FORK-A with "
            f"{template.state_contract!r}; the fork is batched for JMT"
        )


def test_purity_vocabulary_is_the_d10_three_plus_mechanism() -> None:
    """D10: THREE public contracts plus a mechanism column, never four.

    RESTORED is not a contract (a user cannot observe "never touched" vs
    "touched and restored"); it is a mechanism value selecting required
    tests (restored rows owe the mid-call exception plant, Wave 1).
    """

    assert PURITY_CONTRACTS == (
        "PURE_OBSERVER",
        "FORWARD_EQUIVALENT",
        "DECLARED_MUTATOR",
        "UNSET",
    )
    assert set(PURITY_MECHANISMS) == {"untouched", "restored", "n/a", "unset"}


def test_plant_broken_control_goes_red() -> None:
    """PLANT: a positive control wired to a dead channel raises."""

    def dead_channel_control() -> None:
        """Simulate the function-local-fixture failure mode."""

        raise AssertionError("channel dead: fixture unpicklable before the call")

    with pytest.raises(AssertionError, match="channel dead"):
        dead_channel_control()

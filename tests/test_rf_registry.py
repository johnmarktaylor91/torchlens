"""Tests for receptive-field rule registration and invalidation."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from support.rf_isolation import preserved_rf_registry

from torchlens._capture.arg_positions import _normalize_func_name
from torchlens.receptive_field import _rules


@pytest.fixture(autouse=True)
def _isolated_rf_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry around every test.

    These tests call ``register_rf_rule`` directly; without a teardown the
    synthetic rules stayed in ``_RF_RULES`` for the rest of the session and
    bumped the epoch — the documented anti-pattern the module-level
    registry-mutation gate exists to prevent (r7 R77-2).
    """

    with preserved_rf_registry(clear=False):
        yield


def _stub_rule(context: object) -> object:
    """Provide a minimal custom rule for registry tests."""

    return context


# Kept by shrink-r4 (per-id subset rule): sole coverage of receptive_field/_rules.py.
def test_duplicate_registration_raises() -> None:
    """Duplicate rule names require explicit replacement."""

    name = "rf_registry_duplicate_case"
    _rules.register_rf_rule(name)(_stub_rule)

    with pytest.raises(ValueError, match="already registered"):
        _rules.register_rf_rule(name)(_stub_rule)


# Kept by shrink-r4 (per-id subset rule): sole coverage of receptive_field/_rules.py.
def test_custom_rule_registers_and_is_retrievable() -> None:
    """A custom rule is retrievable using TorchLens's normalized name."""

    name = "Rf_Registry_Custom_Rule"

    @_rules.register_rf_rule(name)
    def custom_rule(context: object) -> object:
        """Return the test context unchanged."""

        return context

    assert _rules.rules()[_normalize_func_name(name)] is custom_rule

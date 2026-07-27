"""Tests for receptive-field rule registration and invalidation."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from torchlens.capture.arg_positions import _normalize_func_name
from torchlens.receptive_field import _rules


def _stub_rule(context: object) -> object:
    """Provide a minimal custom rule for registry tests."""

    return context


def test_duplicate_registration_raises() -> None:
    """Duplicate rule names require explicit replacement."""

    name = "rf_registry_duplicate_case"
    _rules.register_rf_rule(name)(_stub_rule)

    with pytest.raises(ValueError, match="already registered"):
        _rules.register_rf_rule(name)(_stub_rule)


def test_replace_registration_overwrites_rule() -> None:
    """Replacement explicitly overwrites an existing registration."""

    name = "rf_registry_replace_case"

    def original_rule(context: object) -> object:
        """Return the original sentinel."""

        return context

    def replacement_rule(context: object) -> None:
        """Return the replacement sentinel."""

        return None

    _rules.register_rf_rule(name)(original_rule)
    _rules.register_rf_rule(name, replace=True)(replacement_rule)

    assert _rules.rules()[_normalize_func_name(name)] is replacement_rule


def test_rules_snapshot_is_immutable() -> None:
    """Rules exposes a read-only snapshot, not the mutable registry."""

    snapshot = _rules.rules()

    assert isinstance(snapshot, Mapping)
    with pytest.raises(TypeError):
        snapshot["rf_registry_snapshot_case"] = _stub_rule  # type: ignore[index]


def test_registry_epoch_increments_for_register_and_replace() -> None:
    """Every accepted registration invalidates prior cached solutions."""

    name = "rf_registry_epoch_case"
    before = _rules._rf_rules_epoch()
    _rules.register_rf_rule(name)(_stub_rule)
    after_register = _rules._rf_rules_epoch()
    _rules.register_rf_rule(name, replace=True)(_stub_rule)

    assert after_register == before + 1
    assert _rules._rf_rules_epoch() == after_register + 1


def test_custom_rule_registers_and_is_retrievable() -> None:
    """A custom rule is retrievable using TorchLens's normalized name."""

    name = "Rf_Registry_Custom_Rule"

    @_rules.register_rf_rule(name)
    def custom_rule(context: object) -> object:
        """Return the test context unchanged."""

        return context

    assert _rules.rules()[_normalize_func_name(name)] is custom_rule

"""Regression coverage for RF built-in rule-pack initialization and restoration."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import (
    ReceptiveFieldRule,
    ReceptiveFieldRuleContext,
    _RuleResult,
)
from torchlens.receptive_field._types import ReceptiveFieldStatus


def _restore_rule_registry(rules: Mapping[str, ReceptiveFieldRule], epoch: int) -> None:
    """Restore a complete rule-registry snapshot and its matching epoch.

    Parameters
    ----------
    rules:
        Rule registrations to restore.
    epoch:
        Registry epoch associated with ``rules``.

    Returns
    -------
    None
        The process-global test registry is restored in place.
    """

    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(rules)
    _rules._RF_RULES_EPOCH = epoch


def test_builtin_rules_are_available_to_the_first_descriptor_query() -> None:
    """Install built-ins before an ordinary descriptor query reaches the solver."""

    model = nn.Conv2d(1, 1, 3, padding=1)
    trace = tl.trace(model, torch.ones(1, 1, 5, 5))
    op = next(item for item in trace.layer_list if item.func_name == "conv2d")
    descriptor = next(iter(op.receptive_field.per_input.values()))

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert _rules.rules()
    assert "conv2d" in _rules.rules()


def test_builtin_rules_are_available_to_receptive_field_validation() -> None:
    """Run the public RF validation scope without a fixture-installed rule pack."""

    results = tl.validate(
        nn.Conv2d(1, 1, 3, padding=1),
        torch.ones(1, 1, 5, 5, requires_grad=True),
        scope="receptive_field",
    )

    assert results
    assert all(result.status.value == "pass" for result in results)


def test_registry_snapshot_restores_builtin_after_custom_override() -> None:
    """Restore the installed built-in rule after a test-local custom override."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    builtin_add = saved_rules["add"]
    try:

        @_rules.register_rf_rule("add", replace=True)
        def custom_add(context: ReceptiveFieldRuleContext) -> _RuleResult:
            """Provide a deliberately local custom rule for lifecycle coverage."""

            return context.passthrough()

        assert _rules.rules()["add"] is custom_add
    finally:
        _restore_rule_registry(saved_rules, saved_epoch)

    assert _rules.rules()["add"] is builtin_add


def test_builtin_rule_pack_reimport_is_idempotent() -> None:
    """Keep the full registry and its epoch unchanged when the pack is re-imported."""

    before_rules = dict(_rules._RF_RULES)
    before_epoch = _rules._RF_RULES_EPOCH
    pack = importlib.import_module("torchlens.receptive_field.rules")

    importlib.reload(pack)

    assert dict(_rules._RF_RULES) == before_rules
    assert _rules._RF_RULES_EPOCH == before_epoch


def test_core_then_extension_rf_files_share_one_deterministic_registry() -> None:
    """Run the adversarial core-then-extension RF order in one fresh process."""

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_rf_geometric.py",
            "tests/test_rf_ext_forward_walk.py",
            "tests/test_rf_ext_l2l_gradient.py",
            "tests/test_rf_crossval.py",
            "-q",
        ],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

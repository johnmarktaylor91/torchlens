"""Tests for menagerie dependency package mapping."""

from __future__ import annotations

import argparse
import subprocess

import pytest

from menagerie.catalog import CatalogRow
from menagerie.runtime import dependency_plan, install_dependency_plan


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture.

    Parameters
    ----------
    overrides:
        Field overrides for the default row.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    data = {
        "model_id": 1,
        "display_index": 1,
        "stable_id": "m-effdet",
        "name": "effdet_tf_efficientdet_d7",
        "variant": "",
        "family": "efficientdet_tf",
        "family_normalized": "efficientdet_tf",
        "domain": "vision/detection",
        "zoo": "effdet",
        "constructor_call": "from effdet import create_model; model=create_model('x')",
        "input_shape": "(1, 3, 1536, 1536)",
        "input_dtype": "float32",
        "era": "2026",
        "verified": True,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "recipe-a",
        "input_is_real": True,
        "verification_expectation": "forward_required",
        "quarantine": False,
    }
    data.update(overrides)
    return CatalogRow(**data)


def test_effdet_dependency_has_pip_mapping_and_does_not_no_mapping_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing but pip-installable effdet dependency is installed, not skipped."""

    plan = dependency_plan(_row())
    commands: list[tuple[str, ...]] = []
    installed = False

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Record the pip install command and return success."""

        nonlocal installed
        del check, capture_output, text, timeout
        commands.append(tuple(command))
        installed = True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "menagerie.runtime.module_importable",
        lambda module: installed if module == "effdet" else True,
    )
    monkeypatch.setattr("menagerie.runtime.subprocess.run", fake_run)
    args = argparse.Namespace(install_deps=True, pip_args=[], install_timeout=30.0)

    error = install_dependency_plan(plan, args)

    assert plan.packages == ("effdet",)
    assert error != "dependency missing with no package mapping: effdet"
    assert error is None
    assert commands
    assert commands[0][-1] == "effdet"


def test_pip_installable_dep_without_mapping_is_not_a_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmapped external import is a router gap, not a justified dependency skip."""

    plan = dependency_plan(
        _row(
            stable_id="m-unmapped",
            name="UnmappedDepNet",
            zoo="pip:definitely-pip-installable-unit-dep",
            constructor_call=(
                "from definitely_pip_installable_unit_dep import Model; model=Model()"
            ),
        )
    )
    args = argparse.Namespace(install_deps=True, pip_args=[], install_timeout=30.0)

    monkeypatch.setattr("menagerie.runtime.module_importable", lambda module: False)

    error = install_dependency_plan(plan, args)

    assert plan.top_modules == ("definitely_pip_installable_unit_dep",)
    assert plan.packages == ()
    assert error == (
        "dependency missing with no package mapping: definitely_pip_installable_unit_dep"
    )

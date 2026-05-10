"""Snapshot equality tests for module-containment metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torchlens as tl

from _module_containment_snapshot import build_snapshot
from fixtures.module_containment_models import ALL_FIXTURES, FixtureBuilder

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "module_containment"


def _unpack_fixture(result: tuple[Any, ...]) -> tuple[Any, Any, str, Any | None]:
    """Unpack fixture results with optional hook handles.

    Parameters
    ----------
    result:
        Fixture builder return tuple.

    Returns
    -------
    tuple[Any, Any, str, Any | None]
        Model, input args, fixture name, and optional hook handle.
    """

    if len(result) == 4:
        model, input_args, fixture_name, hook_handle = result
        return model, input_args, fixture_name, hook_handle
    model, input_args, fixture_name = result
    return model, input_args, fixture_name, None


def _assert_lazy_address_stable(model: Any, fixture_name: str) -> None:
    """Assert LazyLinear address survives first-call materialization.

    Parameters
    ----------
    model:
        Fixture model after tracing.
    fixture_name:
        Name of the current fixture.
    """

    if fixture_name != "lazy_linear_demo":
        return
    lazy_layer = model[0]
    assert getattr(lazy_layer, "tl_address", None) == "0"


def _assert_synthetic_replacement_present(actual: dict[str, Any], fixture_name: str) -> None:
    """Assert synthetic raw-hook fixture includes an interventionreplacement op.

    Parameters
    ----------
    actual:
        Built snapshot dictionary.
    fixture_name:
        Name of the current fixture.
    """

    if fixture_name != "raw_hook_replacement_synthetic":
        return
    func_names = {op["func_name"] for op in actual["ops"]}
    assert "interventionreplacement" in func_names


@pytest.mark.parametrize("builder", ALL_FIXTURES, ids=lambda builder: builder.__name__)
def test_module_containment_snapshot(builder: FixtureBuilder) -> None:
    """Compare module-containment snapshot for one fixture."""

    model, input_args, fixture_name, hook_handle = _unpack_fixture(builder())
    try:
        trace = tl.trace(model, input_args, vis_opt="none")
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    _assert_lazy_address_stable(model, fixture_name)
    actual = build_snapshot(trace, fixture_name)
    _assert_synthetic_replacement_present(actual, fixture_name)

    snapshot_path = SNAPSHOT_DIR / f"{fixture_name}.json"
    if not snapshot_path.exists():
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(actual, indent=2, sort_keys=True, default=str))
        pytest.skip(f"baseline snapshot generated: {snapshot_path}")

    expected = json.loads(snapshot_path.read_text())
    assert actual == expected, (
        f"snapshot drift for {fixture_name}; rerun "
        f"`rm {snapshot_path}` and re-snapshot if intentional"
    )

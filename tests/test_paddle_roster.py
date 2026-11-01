"""The Paddle conformance roster's config_built rows hold against a live capture.

Each row's declared op inventory and NumPy intermediates are independent of
the capture (computed from the row definition); this test is the per-backend
evidence until the suite's C1 cells run the roster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.registry import get_backend_spec  # noqa: E402
from torchlens.conformance._rosters import paddle as paddle_roster  # noqa: E402

pytestmark = pytest.mark.backend_paddle

_SPECS = {row.family: row for row in paddle_roster.roster(include_ecosystem=True)}
_ROW_SPECS = {spec.family: spec for spec in paddle_roster.config_built_specs()}
_CONFIG_BUILT = sorted(_ROW_SPECS)


def test_roster_shape() -> None:
    """At least eight config_built rows, two pretrained ecosystem rows with evidence."""

    assert len(_CONFIG_BUILT) >= 8
    pretrained = [row for row in _SPECS.values() if row.realism == "pretrained"]
    assert len(pretrained) >= 2
    assert all("md5" in row.checkpoint_evidence for row in pretrained)
    assert all(_SPECS[family].expected_op_kinds for family in _CONFIG_BUILT)


def _capture(family: str) -> Any:
    """Build and capture one row."""

    model, example = _ROW_SPECS[family].builder()
    args = example if isinstance(example, tuple) else (example,)
    return tl.trace(model, args, backend="paddle")


def _assert_close(captured: Any, expected: np.ndarray, where: str) -> None:
    """Captured payload equals the NumPy value within float32 tolerance."""

    value = np.asarray(captured.numpy())
    assert value.shape == expected.shape, where
    np.testing.assert_allclose(value, expected, rtol=1e-4, atol=1e-5, err_msg=where)


@pytest.mark.parametrize("family", _CONFIG_BUILT)
def test_roster_row_inventory_intermediates_and_validation(family: str) -> None:
    """Op inventory equal, intermediates match NumPy, validation passes (A9, A15, A18)."""

    spec = _ROW_SPECS[family]
    trace = _capture(family)

    assert tuple(op.func_name for op in trace._layer_list) == spec.kinds
    for kind, expected in spec.intermediates.items():
        ops = [op for op in trace._layer_list if op.func_name == kind]
        values = expected if isinstance(expected, list) else [expected]
        assert len(ops) == len(values), f"{family}:{kind}"
        for index, (op, value) in enumerate(zip(ops, values, strict=True)):
            _assert_close(op.out, np.asarray(value, dtype=np.float32), f"{family}:{kind}[{index}]")
    if spec.has_constants:
        assert any(not list(op.parents) and op.func_name != "input" for op in trace._layer_list)
    protocol_factory = get_backend_spec("paddle").protocol
    assert protocol_factory is not None
    status = protocol_factory().validate(trace, None)
    assert status.state == "passed", f"{family}: {status.message}"

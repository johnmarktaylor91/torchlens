"""Paddle structure-only capture: real forward, no payloads, typed validation refusal.

The fixture opens the Paddle ``structure_only_capture`` gate the way the
registry change for this capability does (spec flag plus implementing surface).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens._capture.structure_only import StructureOnlyCapabilityError  # noqa: E402
from torchlens.backends import registry  # noqa: E402
from torchlens.backends.paddle import _stream_capture  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]

_SHELL = CaptureOptions(structure_only=True)


@pytest.fixture(autouse=True)
def _paddle_structure_only_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the Paddle structure-only gate for the test."""

    spec = registry.get_backend_spec("paddle")
    replaced = dataclasses.replace(
        spec,
        capabilities=dataclasses.replace(spec.capabilities, structure_only_capture=True),
        capability_implementations={
            **dict(spec.capability_implementations),
            "structure_only_capture": lambda: _stream_capture,
        },
    )
    monkeypatch.setitem(registry._REGISTRY, "paddle", replaced)


class _Branchy(paddle.nn.Layer):
    """Takes a different path for wide and narrow inputs (concrete control flow)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = paddle.nn.Linear(6, 6)

    def forward(self, x: Any) -> Any:
        h = paddle.nn.functional.relu(self.fc(x))
        if x.shape[0] > 2:
            h = paddle.tanh(h) * 2.0
        return paddle.sum(h, axis=-1)


def _input(rows: int) -> Any:
    """Return a deterministic ``rows`` x 6 input."""

    return paddle.arange(rows * 6, dtype="float32").reshape([rows, 6]) / 10.0


def _structure(trace: Any) -> list[tuple[str, tuple[int, ...] | None, tuple[str, ...]]]:
    """Return each op's label, shape and parent labels."""

    return [
        (op.label, tuple(op.shape) if op.shape is not None else None, tuple(op.parents))
        for op in trace._layer_list
    ]


@pytest.mark.parametrize("rows", [2, 4], ids=["narrow", "wide"])
def test_structure_only_records_the_taken_path_without_payloads(rows: int) -> None:
    """Same graph, shapes and taken branch as a full capture; no op keeps a payload."""

    paddle.seed(0)
    model = _Branchy()
    full = tl.trace(model, _input(rows), backend="paddle")
    shell = tl.trace(model, _input(rows), backend="paddle", capture=_SHELL)

    assert _structure(shell) == _structure(full)
    assert any("tanh" in op.label for op in shell._layer_list) is (rows > 2)
    assert bool(shell._options_structure_only) is True
    computed = [op for op in shell._layer_list if op.type not in ("input", "output")]
    assert computed and all(op.out is None for op in computed)
    assert all(not op.has_saved_activation for op in computed)


def test_structure_only_refuses_validation_typed() -> None:
    """Validation refuses through the structure-only chokepoint, never reports pass or fail."""

    paddle.seed(0)
    model = _Branchy()
    shell = tl.trace(model, _input(4), backend="paddle", capture=_SHELL)

    with pytest.raises(StructureOnlyCapabilityError) as info:
        shell.validate_forward_pass(model(_input(4)))
    assert info.value.fields["code"] == "structure_only_validation_unsupported"


def test_structure_only_is_refused_while_the_gate_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the spec gate the option is still refused, as before this lane."""

    monkeypatch.undo()
    with pytest.raises(Exception, match="structure_only"):
        tl.trace(_Branchy(), _input(2), backend="paddle", capture=_SHELL)


def test_plain_capture_has_no_structure_only_marker() -> None:
    """A plain capture keeps every payload and no structure-only marker."""

    trace = tl.trace(_Branchy(), _input(2), backend="paddle")
    assert not bool(trace._options_structure_only)
    computed = [op for op in trace._layer_list if op.type not in ("input", "output")]
    assert all(op.out is not None for op in computed)

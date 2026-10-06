"""Forward validation fails a capture whose graph is missing a tensor argument's origin.

Replay re-executes each op on its saved arguments, so a tensor argument with no
recorded graph/source provenance (a closure or module-global tensor, a module-held
tensor the held-tensor scan cut off, an opaque module output adopted at a module
boundary) replays perfectly and used to validate. The ``source_provenance`` check in
``validation/_source_provenance.py`` fails the run instead, on the final trace, with
the single existing carve-out: a genuine, ledger-corroborated intervention
replacement. These tests pin both sides: every source-less class fails, and every
known source (inputs, parameters, buffers, module-held tensors, tensors created
inside forward, a genuine hook replacement) still validates.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure

_GLOBAL_TABLE = torch.randn(5)


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    """Validate forward, silencing the capture's own provenance warnings."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validate(model, x, scope="forward"))


def _failure_reasons() -> tuple[str, list[str]]:
    failure = last_validation_failure()
    assert failure is not None
    return failure.check, list(failure.extra.get("reasons", []))


class _GlobalReader(nn.Module):
    """Reads a module-global tensor that no capture source knows about."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + _GLOBAL_TABLE)


def _closure_model() -> nn.Module:
    table = torch.randn(5)

    class _ClosureReader(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(x * table)

    return _ClosureReader()


@pytest.mark.smoke
def test_global_tensor_argument_fails_forward_validation() -> None:
    torch.manual_seed(0)
    x = torch.randn(5)
    assert _validate(_GlobalReader(), x) is False
    check, reasons = _failure_reasons()
    assert check == "source_provenance"
    assert reasons == ["unattributed_tensor_args"]
    failure = last_validation_failure()
    assert failure is not None and failure.func_name == "__add__"
    assert failure.op_label is not None and failure.op_label.startswith("add")


def test_closure_tensor_argument_fails_forward_validation() -> None:
    torch.manual_seed(0)
    x = torch.randn(5)
    assert _validate(_closure_model(), x) is False
    check, reasons = _failure_reasons()
    assert check == "source_provenance"
    assert reasons == ["unattributed_tensor_args"]


def test_saved_scope_fails_the_same_source_less_capture() -> None:
    torch.manual_seed(0)
    x = torch.randn(5)
    model = _GlobalReader()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="saved") is False
    assert _failure_reasons()[0] == "source_provenance"


class _TooDeepUnread(nn.Module):
    """A plain tensor held past the scan's depth bound; the forward never reads it."""

    def __init__(self) -> None:
        super().__init__()
        self.deep = {"a": {"b": {"c": {"d": {"e": torch.zeros(1)}}}}}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def test_truncated_held_scan_fails_forward_validation_and_persists_an_advisory() -> None:
    x = torch.randn(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_TooDeepUnread(), x)
    rows = [
        row
        for row in trace.annotations.get("capture_advisories", [])
        if row["kind"] == "held_tensor_scan_truncated"
    ]
    assert len(rows) == 1
    assert "deep['a']['b']['c']['d'] (depth bound 4)" in rows[0]["message"]
    assert _validate(_TooDeepUnread(), x) is False
    assert _failure_reasons() == ("source_provenance", ["held_tensor_scan_truncated"])


class _Opaque(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.tanh.default(x)


class _OpaqueHolder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act = _Opaque()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.act(x))


def test_module_boundary_adoption_fails_forward_validation() -> None:
    x = torch.randn(2, 4)
    model = _OpaqueHolder()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(copy.deepcopy(model), x)
    kinds = [row["kind"] for row in trace.annotations.get("capture_advisories", [])]
    assert kinds == ["module_boundary_adoption"]
    assert _validate(model, x) is False
    assert _failure_reasons() == ("source_provenance", ["module_boundary_adoption"])


# --- Negative: every known source still validates ---------------------------------


class _KnownSources(nn.Module):
    """Inputs, a parameter, a registered buffer, a held plain tensor and a list item."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5))
        self.register_buffer("offset", torch.randn(5))
        self.table = torch.randn(5)
        self.items = [torch.randn(5)]
        self.cache = {"cpu": torch.randn(5)}

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = x * self.weight + self.offset + y
        return out + self.table + self.items[0] + self.cache["cpu"]


class _CreatedInForward(nn.Module):
    """Constants created inside forward by factories and ``torch.tensor``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0])
        return x * scale + torch.ones(5) + torch.arange(5.0)


@pytest.mark.parametrize(
    ("model", "inputs"),
    [
        (nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2)), (torch.randn(3, 5),)),
        (_KnownSources(), (torch.randn(5), torch.randn(5))),
        (_CreatedInForward(), (torch.randn(5),)),
    ],
    ids=["sequential", "known_sources", "created_in_forward"],
)
def test_known_sources_still_validate(model: nn.Module, inputs: tuple[Any, ...]) -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, list(inputs), scope="forward") is True
    assert last_validation_failure() is None


def test_genuine_hook_replacement_still_validates() -> None:
    """The one carve-out: a genuine opaque output-replacement hook (an intervention)."""

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(self.relu(self.fc1(x)))

    def _inject(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.mul.Tensor(output, torch.full_like(output, 0.5))

    model = _Mlp().eval()
    model.relu.register_forward_hook(_inject)
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    replaced = [op for op in trace.layer_list if op.intervention_replaced]
    assert replaced, "expected a genuine intervention replacement op"
    assert _validate(model, x) is True

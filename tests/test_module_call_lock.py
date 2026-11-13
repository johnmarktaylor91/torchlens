"""Module-call records do not depend on layer grouping.

A module call (``layer1.0.relu:2``) individuates one call of one module object, and that
record is a separate concept from the layer an op is grouped into. This test pins every
module-call fact of four models whose modules are called more than once: num_calls, the calls
and their labels, each call's ops (by op identity: creation index and function, never the
layer label), lookup of each call by ``address:N``, every op's module context (call stack,
atomic module call, module-call inputs and outputs), and the module-call boxes of the
unrolled drawing. The pinned record was taken before stateless modules stopped grouping
their calls as passes; any change to the grouping rule must leave it identical.

Regenerate only when a module-call fact changes on purpose:
``TORCHLENS_REGEN_MODULE_CALL_LOCK=1 pytest tests/test_module_call_lock.py``.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl

GOLDEN = Path(__file__).parent / "fixtures" / "module_call_lock.json"
REGEN = os.environ.get("TORCHLENS_REGEN_MODULE_CALL_LOCK") == "1"

_CLUSTER = re.compile(r"cluster_([^\"\s{]+?)_pass(\d+)")
_MARK = re.compile(r"@([A-Za-z0-9_.]+(?::[0-9][0-9,\-]*)?)")


class _Encoder(nn.Module):
    """A stateful encoder with a stateless activation module inside."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.lin(x))


class SiameseNet(nn.Module):
    """One encoder called on two lines, then a head."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = _Encoder()
        self.head = nn.Linear(8, 2)

    def forward(self, x: Tensor) -> Tensor:
        a = self.enc(x)
        b = self.enc(x * 2)
        return self.head(a - b)


class StepRNN(nn.Module):
    """An RNN cell and a stateless activation called once per step from one loop line."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.RNNCell(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        h = torch.zeros(x.shape[0], 8)
        for t in range(x.shape[1]):
            h = self.act(self.cell(x[:, t], h))
        return h


class _SharedActBlock(nn.Module):
    """A residual block that calls the activation module it was given on two lines."""

    def __init__(self, act: nn.Module) -> None:
        super().__init__()
        self.lin1 = nn.Linear(8, 8)
        self.lin2 = nn.Linear(8, 8)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        y = self.act(self.lin1(x))
        return self.act(self.lin2(y)) + x


class SharedReLUNet(nn.Module):
    """One ``nn.ReLU`` object shared by three blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.blocks = nn.ModuleList(_SharedActBlock(self.act) for _ in range(3))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _resnet18() -> tuple[nn.Module, Tensor]:
    torchvision = pytest.importorskip("torchvision")
    return torchvision.models.resnet18(weights=None).eval(), torch.randn(1, 3, 64, 64)


_MODELS: dict[str, Callable[[], tuple[nn.Module, Tensor]]] = {
    "resnet18": _resnet18,
    "siamese": lambda: (SiameseNet(), torch.randn(2, 8)),
    "step_rnn": lambda: (StepRNN(), torch.randn(2, 4, 8)),
    "shared_relu": lambda: (SharedReLUNet(), torch.randn(2, 8)),
}


def _texts(values: Any) -> list[str]:
    """Strings of a module-context list (labels or records), in order."""

    return [str(value) for value in (values or [])]


def _module_record(trace: Any, ident: Callable[[Any], str]) -> dict[str, Any]:
    """Every module's calls: count, labels, ops by identity, and the ``address:N`` lookup."""

    record: dict[str, Any] = {}
    for module in trace.modules:
        address = str(module.address)
        calls = []
        for key in module.calls:
            call = module.calls.get(key)
            entry: dict[str, Any] = {
                "label": str(call.label),
                "ops": [ident(trace.ops[label]) for label in call.ops],
                "output_ops": [ident(trace.ops[label]) for label in call.output_ops],
            }
            if address not in ("", "self"):
                looked_up = trace[str(call.label)]
                entry["lookup_is_call"] = looked_up is call
            calls.append(entry)
        record[address] = {
            "num_calls": int(module.num_calls),
            "call_labels": _texts(module.call_labels),
            "calls": calls,
        }
    return record


def _op_record(trace: Any, ident: Callable[[Any], str]) -> dict[str, Any]:
    """Every op's module context, keyed by op identity."""

    record: dict[str, Any] = {}
    for op in trace._layer_list:
        context = op.module_context
        record[ident(op)] = {
            "call_stack": _texts(context.call_stack),
            "is_atomic_module": bool(context.is_atomic_module),
            "atomic_module_call": str(context.atomic_module_call),
            "input_to_module_calls": _texts(context.input_to_module_calls),
            "output_of_module_calls": _texts(context.output_of_module_calls),
        }
    return record


def _drawing_record(trace: Any, tmp_path: Path) -> dict[str, Any]:
    """Module-call boxes and module marks of the unrolled drawing."""

    dot = trace.draw(file_format="dot", path=str(tmp_path / "lock")).dot
    return {
        "module_call_boxes": sorted({f"{name}:{n}" for name, n in _CLUSTER.findall(dot)}),
        "module_marks": sorted(set(_MARK.findall(dot))),
    }


def _record(name: str, tmp_path: Path) -> dict[str, Any]:
    """The whole module-call record of one model."""

    torch.manual_seed(0)
    model, x = _MODELS[name]()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)

    def ident(op: Any) -> str:
        return f"{op.raw_index}:{op.func_name}"

    return {
        "modules": _module_record(trace, ident),
        "ops": _op_record(trace, ident),
        "drawing": _drawing_record(trace, tmp_path),
    }


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("resnet18", marks=pytest.mark.heavy),
        "siamese",
        "step_rnn",
        "shared_relu",
    ],
)
def test_module_call_records_are_unchanged(name: str, tmp_path: Path) -> None:
    """Module-call records equal the pinned record; only layer grouping may differ."""

    if REGEN:
        # Generation is in-process: refuse to generate the pinned record on a torch
        # already wrapped by earlier tests (SF-53).
        from _oracle_env import guard_wrap_state_for_golden_update

        guard_wrap_state_for_golden_update("TORCHLENS_REGEN_MODULE_CALL_LOCK")
    record = _record(name, tmp_path)
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    if REGEN:
        golden[name] = record
        GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        pytest.skip("module-call lock regenerated")
    assert name in golden, f"no pinned module-call record for {name}"
    expected = golden[name]
    assert record["modules"] == expected["modules"]
    assert record["ops"] == expected["ops"]
    assert record["drawing"] == expected["drawing"]

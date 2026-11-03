"""Forward validation fails a capture whose graph is missing a tensor argument's origin.

Replay re-executes each op on its saved arguments, so a tensor argument with no
recorded graph/source provenance (a module-held tensor the held-tensor scan cut off,
an opaque module output adopted at a module boundary or consumed by an op) replays
perfectly and used to validate. A closure or module-global tensor that existed
before the forward is a recorded ``constant`` source op (SPEC 2), not a gap. The ``source_provenance`` check in
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

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._state import pause_logging
from torchlens.backends.torch import rescue
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
    return failure.check_name, list(failure.extra.get("reasons", []))


def _constant_ops(trace: Any) -> list[Any]:
    return [op for op in trace._layer_list if op.func_name == "constant"]


# --- Review fixes: a module RETURNING an outside tensor; orphan-pruned consumers ---


def _returns_outside_parent(sub: nn.Module) -> nn.Module:
    class _Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sub = sub

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.sub(x) + x

    return _Parent()


def _returns_closure() -> nn.Module:
    table = torch.randn(5)

    class _ReturnsClosure(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return table

    return _ReturnsClosure()


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/model_prep.py.
@pytest.mark.parametrize("make_sub", [_returns_closure], ids=["closure"])
def test_submodule_returning_an_outside_tensor_returns_a_constant(make_sub: Any) -> None:
    """A submodule returning a closure/global tensor returns a recorded ``constant`` source."""

    x = torch.randn(5)
    model = _returns_outside_parent(make_sub())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        trace = tl.trace(copy.deepcopy(model), x)
    assert len(_constant_ops(trace)) == 1
    assert not trace.annotations.get("capture_advisories")
    assert _validate(model, x) is True


def _unseen_tanh(x: torch.Tensor) -> torch.Tensor:
    """A value built where TorchLens cannot see it (logging paused): no provenance."""

    with pause_logging():
        return x.tanh()


class _UnseenItemSink(nn.Module):
    """An unseen tensor is consumed only by ``.item()`` (orphan-pruned)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + _unseen_tanh(x).sum().item()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/graph_traversal.py.
@pytest.mark.parametrize("model", [_UnseenItemSink()], ids=["item"])
def test_orphan_pruned_consumer_keeps_the_source_less_witness(
    model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = torch.randn(5)
    # Keep the primary capture: a rescue re-run could record the unseen op.
    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    rows = [
        row
        for row in trace.annotations.get("capture_advisories", [])
        if row["kind"] == "orphan_unattributed_tensor_args"
    ]
    assert len(rows) == 1 and "sum" in rows[0]["message"]
    assert _validate(model, x) is False
    assert _failure_reasons() == ("source_provenance", ["orphan_unattributed_tensor_args"])


class _ReturnsOwn(nn.Module):
    """Returns its own held plain tensor, buffer or Parameter directly."""

    def __init__(self, which: str) -> None:
        super().__init__()
        self.which = which
        self.table = torch.randn(5)
        self.register_buffer("offset", torch.randn(5))
        self.weight = nn.Parameter(torch.randn(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, self.which)


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/model_prep.py, campaign mutant kills.
@pytest.mark.parametrize(
    "model",
    [
        _returns_outside_parent(_ReturnsOwn("weight")),
    ],
    ids=[
        "returns_param",
    ],
)
def test_model_owned_returns_and_sourced_orphans_still_validate(model: nn.Module) -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, torch.randn(5), scope="forward") is True
    assert last_validation_failure() is None


# --- Negative: every known source still validates ---------------------------------


# --- Boundary review fixes: data egress (``tolist``/``numpy``/``__array__``) ---------

_GLOBAL_SCALAR = torch.tensor(2.5)


class _Egress(nn.Module):
    """Reads a module-global tensor's data into Python through one escape spelling."""

    def __init__(self, read: Any) -> None:
        super().__init__()
        self.read = read

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.read(x)


_PRUNED_OP_EGRESS = {
    "sum_tolist": lambda x: x * _GLOBAL_TABLE.sum().tolist(),
    "argsort_tolist": lambda x: x[_GLOBAL_TABLE.argsort().tolist()],
    "sum_numpy": lambda x: x * float(_GLOBAL_TABLE.sum().numpy()),
    "sum_cpu_numpy": lambda x: x * float(_GLOBAL_TABLE.sum().cpu().numpy()),
    "sum_array": lambda x: x * float(np.asarray(_GLOBAL_TABLE.sum())),
    "sum_dunder_array": lambda x: x * float(_GLOBAL_TABLE.sum().__array__()),
}

_DIRECT_EGRESS = {
    "tensor_of_tolist": lambda x: x + torch.tensor(_GLOBAL_TABLE.tolist()),
    "from_numpy": lambda x: x + torch.from_numpy(_GLOBAL_TABLE.numpy().copy()),
    "asarray": lambda x: x * float(np.asarray(_GLOBAL_SCALAR)),
    "item": lambda x: x * _GLOBAL_SCALAR.item(),
    "float": lambda x: x * float(_GLOBAL_SCALAR),
    "int": lambda x: x * int(_GLOBAL_SCALAR),
}


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_completeness_storage.py, backends/torch/_foreign_data.py, backends/torch/identity_shims.py ....
@pytest.mark.parametrize("read", list(_DIRECT_EGRESS.values()), ids=list(_DIRECT_EGRESS))
def test_direct_data_read_of_a_source_less_tensor_fails(read: Any) -> None:
    """``torch.tensor(G.tolist())`` reads a global with no op to carry a witness."""

    model = _Egress(read)
    x = torch.randn(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x)
    rows = [
        row
        for row in trace.annotations.get("capture_advisories", [])
        if row["kind"] == "orphan_unattributed_tensor_args"
    ]
    assert len(rows) == 1 and "source-less tensor" in rows[0]["message"], rows
    assert _validate(model, x) is False
    assert _failure_reasons() == ("source_provenance", ["orphan_unattributed_tensor_args"])


class _OwnEgress(nn.Module):
    """Reads its own sources' data into Python: input, held tensor, buffer, Parameter."""

    def __init__(self, which: str) -> None:
        super().__init__()
        self.which = which
        self.table = torch.randn(5)
        self.register_buffer("offset", torch.tensor(1.5))
        self.weight = nn.Parameter(torch.randn(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.which == "input_sum_tolist":
            return x * x.sum().tolist()
        if self.which == "constant_numpy":
            return x * float(torch.ones(3).sum().numpy())
        if self.which == "held_tolist":
            return x + torch.tensor(self.table.tolist())
        if self.which == "buffer_item":
            return x * self.offset.item()
        return x * float(np.asarray(self.weight.detach().sum()))


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
@pytest.mark.parametrize(
    "which",
    ["buffer_item"],
)
def test_data_egress_of_sourced_tensors_still_validates(which: str) -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(_OwnEgress(which), torch.randn(5), scope="forward") is True
    assert last_validation_failure() is None

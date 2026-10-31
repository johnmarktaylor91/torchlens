"""``tl.record(...).to_trace()`` keeps exactly the ops ``tl.trace`` keeps.

The glossary says a Recording's ``to_trace()`` materializes full graph
structure, so on the same model in the same state both doors must list the
same functions in the same capture order. Each model here exercises one kind
of op the sparse recorder has to classify the way exhaustive capture does:
tensor constructors and parameter-only ops with no input ancestry, a prepared
Parameter rebound through ``param.data = rhs`` (a source rebinding, never an
op output), and a lazy module's in-forward initialization.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


def _func_names(trace: Any) -> list[str]:
    ops = sorted(trace.ops, key=lambda op: (op.raw_index is None, op.raw_index or 0))
    return [op.func_name for op in ops]


class _ConstructorAndParamOnly(nn.Module):
    """A constructor and a parameter-only op both feed the output."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = torch.zeros(4)
        magnitude = self.scale.abs()
        return self.lin(x) + offset + magnitude


class _ParameterDataRebind(nn.Module):
    """PACT-style first-call init: a prepared Parameter rebound via ``.data``."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            self.alpha.data = x.abs().max()
            self.initialized = True
        return x * self.alpha


class _LazyInit(nn.Module):
    """A lazy module materializes its Parameters inside the captured forward."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.LazyLinear(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


CASES = {
    "constructor_and_param_only": (_ConstructorAndParamOnly, (2, 4)),
    "parameter_data_rebind": (_ParameterDataRebind, (3,)),
    "lazy_init": (_LazyInit, (2, 5)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_record_to_trace_keeps_the_traced_ops(name: str) -> None:
    model_cls, shape = CASES[name]
    torch.manual_seed(0)
    x = torch.randn(*shape)
    torch.manual_seed(1)
    traced = _func_names(tl.trace(model_cls(), x.clone()))
    torch.manual_seed(1)
    recorded = _func_names(tl.record(model_cls(), x.clone(), default_op=True).to_trace())
    assert recorded == traced


def test_record_rebind_leaves_the_parameter_rebound() -> None:
    """Classifying the rebind as a source write must not skip the write itself."""

    torch.manual_seed(0)
    x = torch.randn(3)
    model = _ParameterDataRebind()
    tl.record(model, x.clone(), default_op=True)
    assert isinstance(model.alpha, nn.Parameter)
    assert torch.equal(model.alpha.detach(), x.abs().max().reshape(model.alpha.shape))

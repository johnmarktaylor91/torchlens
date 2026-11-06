"""The one-backward read table's ``to_pandas`` names the tabular extra when pandas is absent.

It imported pandas bare, so a missing pandas surfaced as ``ModuleNotFoundError``
instead of the message every other table door gives.
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

import torchlens as tl


def _mlp_trace() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Trace a small eval-mode MLP; return the model, its input and the trace."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
    x = torch.randn(2, 4)
    return model, x, tl.trace(model, x)


def test_read_table_to_pandas_without_pandas_names_the_tabular_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-backward table refuses with the tabular-extra message when pandas is absent."""

    from torchlens._utils._torch_compat import get_gradient_edge_support
    from torchlens.attribution import _onebackward as ob

    if not get_gradient_edge_support():
        pytest.skip("one-backward reads require torch.autograd.graph.GradientEdge (2.4+)")
    _model, _x, trace = _mlp_trace()
    table = ob.read(trace, target=ob.seed("output_1", index=(0, 0)), method="grad", reduce="sum")
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=r"torchlens\[tabular\]"):
        table.to_pandas()

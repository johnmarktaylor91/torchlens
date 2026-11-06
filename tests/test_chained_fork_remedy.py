"""A chained ``trace.fork().do(...)[...]`` read names the fork in its refusal remedy.

Views never keep their Trace alive, so the fork in a one-line chain is
collected before ``.out`` runs. The refusal used to say "keep the Trace in a
variable", which the user had done; the remedy now names the fork.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._errors import RecordBindingError


def _mlp_trace() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Trace a small eval-mode MLP; return the model, its input and the trace."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
    x = torch.randn(2, 4)
    return model, x, tl.trace(model, x)


def test_chained_fork_read_names_the_fork_in_its_remedy() -> None:
    """``trace.fork().do(...)[...].out`` is the value or a refusal whose remedy names forks."""

    _model, _x, trace = _mlp_trace()
    try:
        out = trace.fork().do(tl.module("1"), tl.zero_ablate())["2"].out
    except RecordBindingError as exc:
        assert exc.fields["code"] == "trace_reference_collected"
        assert "fork" in exc.fields["remedy"], exc.fields["remedy"]
    else:
        assert out is not None

"""Messages and reference docs spell the intervention engine the way ``Trace.do`` takes it.

``Trace.do`` has no ``engine=`` keyword; the engine rides
``intervention=InterventionOptions(engine=...)``. Messages that taught
``do(..., engine='rerun')`` sent users to a ``TypeError``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import EngineDispatchError

_REPO = Path(__file__).resolve().parents[1]


_ENGINE_AS_DO_KEYWORD = re.compile(r"do\((?:\.\.\.,\s*)?engine=")


@pytest.mark.parametrize("root", ["torchlens", "docs/reference"])
def test_no_message_or_reference_spells_engine_as_a_do_keyword(root: str) -> None:
    """``Trace.do`` takes no ``engine=``; text must not teach ``do(..., engine=...)``."""

    offenders = []
    for path in sorted((_REPO / root).rglob("*")):
        if path.suffix not in {".py", ".md"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _ENGINE_AS_DO_KEYWORD.search(line):
                offenders.append(f"{path.relative_to(_REPO)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def _mlp_trace() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Trace a small eval-mode MLP; return the model, its input and the trace."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
    x = torch.randn(2, 4)
    return model, x, tl.trace(model, x)


def test_engine_dispatch_refusals_name_intervention_options() -> None:
    """Half a rerun request names the real spelling of the engine."""

    model, _x, trace = _mlp_trace()
    with pytest.raises(EngineDispatchError, match=r"InterventionOptions\(engine='replay'\)"):
        trace.fork().do(tl.module("1"), tl.zero_ablate(), model=model)
    rerun = tl.options.InterventionOptions(engine="rerun")
    with pytest.raises(EngineDispatchError, match=r"InterventionOptions\(engine='rerun'\)"):
        trace.fork().do(tl.module("1"), tl.zero_ablate(), model=model, intervention=rerun)

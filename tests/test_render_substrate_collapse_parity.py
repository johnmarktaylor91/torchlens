"""Three-way collapse pricing parity gate (lane C05; collapse memo D1).

The phantom-k class: the optimizer scored a node count that is not what
gets drawn (a selected box priced k=1 while rendering one box PER CALL plus
kept atomic ops; synthetic child graphs claiming zero parent-owned ops).
The gate pins the parity chain on PLANS, not frontier records:

    scored k == CollapsePlan.count() == emitted render units

where the emitted leg re-derives the plan through the SAME node-universe
entry point both render backends consume, and ``RepeatFold`` counts as 2
(representative + ellipsis). Until this gate is green corpus-wide, every
weight, band target, and schedule number is frozen (they are computed on k).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._visualization.collapse_optimizer import select_collapse_plan
from torchlens._visualization.collapse_plan import (
    RenderContext,
)

# Per-test smoke marks, NOT a module pytestmark: markers are additive, and
# the googlenet parity cell below is heavy -- a module-level smoke mark
# would hand it the banned smoke+heavy combination (marker-lint).


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin_a = torch.nn.Linear(8, 8)
        self.lin_b = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin_b(torch.relu(self.lin_a(x))))


class _ReusedBlockModel(torch.nn.Module):
    """One module instance called three times: a selected box renders 3 boxes."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.block(x)
        return self.head(x)


class _RootFanModel(torch.nn.Module):
    """70 parallel root-owned boundary outputs: the cache-flood cliff shape."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(x + float(index) for index in range(70))


def test_floor_fallback_result_is_typed_and_visible(tmp_path, monkeypatch) -> None:
    # F9 unit wiring: when the DP frontier empties, the result must carry
    # planner="floor_fallback" + the K_CAP diagnosis, the draw must warn,
    # and the render caption must carry a visible notice.
    import torchlens._visualization.collapse_optimizer as co
    from torchlens.errors._base import TorchLensWarning

    log = tl.trace(_RootFanModel(), torch.randn(2, 4))
    monkeypatch.setattr(co, "_select_best_decision", lambda **kwargs: None)
    result = select_collapse_plan(log, RenderContext(vis_mode="unrolled"), mode="auto")
    assert result.planner == "floor_fallback"
    assert result.k_cap_exhausted is True
    assert result.root_own_units > 64
    assert result.reason is not None and "module= focus" in result.reason
    with pytest.warns(TorchLensWarning, match="floor plan"):
        source = log.draw(
            path=str(tmp_path / "cliff"),
            file_format="svg",
            collapse="auto",
        ).dot
    # Visible notice on the render (never only a warning).
    assert "collapse fell back to the floor plan" in source

"""A selection that used to ride the removed ``module_filter`` composes into ``save=``.

``CaptureOptions.module_filter`` was a third save gate ANDed with ``save=``.
It is gone: ``save=`` is the one site selection, so the same selection is a
selector (or a predicate) passed as ``save=``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torchlens as tl


class ThreeStep(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _saved_labels(log: tl.Trace) -> list[str]:
    return [op.layer_label for op in log if getattr(op, "out", None) is not None]


def test_selection_composed_into_save_saves_the_same_sites() -> None:
    """A selector algebra and a predicate function pick exactly the same sites."""

    model, x = ThreeStep(), torch.randn(2, 4)
    by_selector = tl.trace(model, x, save=tl.func("linear") & ~tl.in_module("fc2"))
    by_predicate = tl.trace(
        model,
        x,
        save=tl.where(lambda ctx: ctx.func_name == "linear" and "fc2" not in ctx.module_stack),
    )
    saved = _saved_labels(by_selector)
    assert saved, "the composed selection should keep the fc1 payload"
    assert all(label.startswith("linear") for label in saved)
    assert saved == _saved_labels(by_predicate)
    assert [op.layer_label for op in by_selector] == [op.layer_label for op in by_predicate]

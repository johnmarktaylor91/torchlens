"""Pins for the saved-for-backward draw annotation (opt-in, DOCUMENTED-UNSTABLE).

The row makes autograd's retained-tensor memory visible per node from the
capture-time ``num_autograd_tensors`` / ``autograd_memory`` measurements.
HONESTY CONTRACT: the row appears ONLY where retention was measured positive;
an absent row makes no claim (zero saved and never-measured both stay silent),
and rolled multi-pass rows disclose that their figures are cross-pass sums.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl

ROW_RE = re.compile(r"saved for backward[^<\"]*")


class LoopMlp(nn.Module):
    """Recurrent MLP producing a multi-pass rolled layer."""

    def __init__(self) -> None:
        """Initialize the shared linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared layer three times.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output after three recurrent passes.
        """

        for _ in range(3):
            x = torch.relu(self.fc(x))
        return x


def _draw(log: tl.Trace, tmp_path: Path, **kwargs: Any) -> str:
    """Render a trace to DOT source without opening a viewer.

    Parameters
    ----------
    log:
        Completed trace.
    tmp_path:
        Scratch directory.
    **kwargs:
        Extra ``draw`` options.

    Returns
    -------
    str
        DOT source.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    return log.draw(
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / "graph"),
        **kwargs,
    )


def test_opt_in_adds_rows_and_default_stays_clean(tmp_path: Path) -> None:
    """Rows appear only under the flag and match the captured measurements."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    trace = tl.trace(model, torch.randn(2, 4))

    annotated = _draw(trace, tmp_path / "on", show_saved_for_backward=True)
    rows = ROW_RE.findall(annotated)
    measured_positive = sum(
        1 for op in trace.ops if (getattr(op, "num_autograd_tensors", None) or 0) > 0
    )
    assert len(rows) == measured_positive > 0
    assert all(re.match(r"saved for backward: \d+ tensors?, ", row) for row in rows)

    plain = _draw(trace, tmp_path / "off")
    assert "saved for backward" not in plain


def test_no_grad_capture_has_no_rows(tmp_path: Path) -> None:
    """A capture without a backward graph never claims retention."""

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    with torch.no_grad():
        trace = tl.trace(model, torch.randn(2, 4))

    dot = _draw(trace, tmp_path, show_saved_for_backward=True)

    assert "saved for backward" not in dot


def test_rolled_multipass_rows_disclose_cross_pass_totals(tmp_path: Path) -> None:
    """Rolled multi-pass rows carry the aggregation disclosure; unrolled do not."""

    torch.manual_seed(2)
    trace = tl.trace(LoopMlp(), torch.randn(2, 4))

    rolled = ROW_RE.findall(
        _draw(trace, tmp_path / "rolled", vis_mode="rolled", show_saved_for_backward=True)
    )
    unrolled = ROW_RE.findall(_draw(trace, tmp_path / "unrolled", show_saved_for_backward=True))

    assert rolled and all(row.endswith("(total across passes)") for row in rolled)
    assert unrolled and not any("(total across passes)" in row for row in unrolled)
    assert len(unrolled) > len(rolled)


def test_explicit_field_picker_is_not_injected(tmp_path: Path) -> None:
    """node_label_fields keeps full control of its rows even with the flag on."""

    torch.manual_seed(3)
    trace = tl.trace(nn.Linear(4, 2), torch.randn(2, 4))

    dot = _draw(
        trace,
        tmp_path,
        node_label_fields=["label", "shape"],
        show_saved_for_backward=True,
    )

    assert "saved for backward" not in dot

"""F10 lovely items 10-11: glance + seam adapters + echo purity.

One record, many densities: the cell formatter and the HTML fixture read
the SAME frozen record the line renders (a number may never exist only in
HTML); the echo observer never contaminates the capture it observes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.stats import (
    degrade,
    echo,
    glance,
    render_core_line,
    summary_cell,
    tensor_stats,
    treescope_card_fields,
)


def test_glance_is_the_core_grammar() -> None:
    """glance renders exactly the record's core line (one grammar)."""

    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    stats = tensor_stats(tensor, identity=f"glance:{tuple(tensor.shape)}:{tensor.dtype}")
    assert glance(tensor) == render_core_line(stats)


def test_glance_refuses_non_tensors_teaching() -> None:
    """The refusal names the captured-record spelling (teaching refusal)."""

    with pytest.raises(TypeError, match="record surfaces"):
        glance([1.0, 2.0])  # type: ignore[arg-type]


@pytest.mark.smoke
def test_glance_width_ladder_keeps_extrema_and_health() -> None:
    """D14: degradation drops bytes/n/sparkline, never extrema or health."""

    tensor = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
    tensor[0, 0] = float("nan")
    narrow = glance(tensor, max_width=60)
    assert "nan=" in narrow
    assert "[" in narrow  # the extrema bracket survives


def test_summary_cell_matches_line_renderer() -> None:
    """Seam pin: the cell IS the line renderer at the cell width."""

    stats = tensor_stats(torch.arange(100, dtype=torch.float32))
    assert summary_cell(stats, max_width=40) == render_core_line(stats, max_width=40)


@pytest.mark.smoke
def test_treescope_fields_carry_no_invented_numbers() -> None:
    """The HTML fixture is a projection of the record, both encodings."""

    stats = tensor_stats(torch.arange(1000, dtype=torch.float32))
    fields = treescope_card_fields(stats)
    assert fields["line_ascii"] == degrade(fields["line_unicode"])
    assert fields["mean"] == stats.mean and fields["sd"] == stats.sd
    assert fields["evidence"]["mean"] == "exact"
    if fields["sparkline_ascii"] is not None:
        assert fields["sparkline_ascii"] == degrade(fields["sparkline_unicode"])


class _EchoModel(nn.Module):
    """Two-op model for the echo purity pin."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


@pytest.mark.smoke
def test_echo_mounts_and_stays_pure() -> None:
    """The echo fires per site AND adds ZERO ops to the captured graph."""

    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    control = tl.trace(_EchoModel().eval(), x.clone())
    control_ops = control.num_ops
    control.cleanup()

    sink_lines: list[str] = []
    observer = echo(tl.func("relu"), sink=sink_lines.append)
    echoed = tl.trace(
        _EchoModel().eval(),
        x.clone(),
        capture=tl.options.CaptureOptions(intervention_ready=True, hooks=observer),
    )
    assert observer.lines and sink_lines == observer.lines
    assert " -> " in observer.lines[0] and "mean=" in observer.lines[0]
    # Purity: the echo's own stats math never enters the captured graph.
    assert echoed.num_ops == control_ops
    line = repr(observer)
    assert "EchoObserver(" in line and "lines echoed" in line
    assert "\n" not in line
    echoed.cleanup()

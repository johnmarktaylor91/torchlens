"""Typed-code provocation tests for the C05 render-substrate disclosures.

Three C05 warning codes are disclosure-tier (never refusals):
``layout_engine_stderr`` (render_execution), ``collapse_floor_fallback``
and ``collapse_near_uncollapsed`` (auto_collapse, collapse memo D4).
The natural floor/near-uncollapsed provocations arise on huge graphs
(K_CAP exhaustion on cached decoders -- diagnostic-tier renders), so the
disclosure body is provoked at its unit seam with a real trace and a
real ``OptimizerResult`` shaped like the fallback cases.
"""

from __future__ import annotations

import torch

from torchlens._visualization.render_execution import surface_layout_stderr


class _TwoBlockModel(torch.nn.Module):
    """Small two-block model for real-trace disclosure provocations."""

    def __init__(self) -> None:
        """Initialize the two linear blocks."""

        super().__init__()
        self.block_a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.block_b = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks in sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the second block.
        """

        return self.block_b(self.block_a(x))


def test_layout_engine_silence_stays_silent() -> None:
    """Empty or ``None`` stderr never warns and returns the empty string."""

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert surface_layout_stderr(None, engine="dot") == ""
        assert surface_layout_stderr(b"", engine="dot") == ""
        assert surface_layout_stderr("  \n", engine="dot") == ""

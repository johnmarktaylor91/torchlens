"""Phase 8 cross-architecture bundle alignment tests."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _OneHidden(nn.Module):
    """Single-hidden-layer model."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.fc2(torch.relu(self.fc1(x)))


class _TwoHidden(nn.Module):
    """Two-hidden-layer model with comparable sites."""

    def __init__(self) -> None:
        """Initialize modules."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.extra = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        hidden = torch.relu(self.fc1(x))
        return self.fc2(torch.relu(self.extra(hidden)))


def _log(model: nn.Module, x: torch.Tensor) -> tl.ModelLog:
    """Capture a model log.

    Parameters
    ----------
    model:
        Model to capture.
    x:
        Input tensor.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    return tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)


def test_aligned_pairs_returns_best_match_sites_across_architectures() -> None:
    """aligned_pairs emits one-to-one layer matches with documented heuristics."""

    torch.manual_seed(10)
    x = torch.randn(2, 4)
    bundle = tl.bundle(
        {
            "small": _log(_OneHidden(), x),
            "large": _log(_TwoHidden(), x),
        }
    )

    pairs = bundle.aligned_pairs("small", "large")
    pair_labels = [(left.layer_label, right.layer_label) for left, right in pairs]

    assert pairs
    assert len({right.layer_label for _, right in pairs}) == len(pairs)
    assert any("linear" in left.func_name and "linear" in right.func_name for left, right in pairs)
    assert pair_labels == [(left.layer_label, right.layer_label) for left, right in pairs]

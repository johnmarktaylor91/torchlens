"""Phase 8 bundle comparison primitive tests."""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import torch
from torch import nn

import torchlens as tl


class _TinyRelu(nn.Module):
    """Small model used for bundle primitive tests."""

    def __init__(self, offset: float = 0.0) -> None:
        """Initialize the model.

        Parameters
        ----------
        offset:
            Constant output offset.
        """

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.offset = offset

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

        return torch.relu(self.linear(x)) + self.offset


def _capture_pair(seed: int, offset: float) -> tl.Bundle:
    """Return a two-member bundle with deterministic captures.

    Parameters
    ----------
    seed:
        Random seed for model initialization and input.
    offset:
        Output offset for the second model.

    Returns
    -------
    tl.Bundle
        Bundle with baseline and compared traces.
    """

    torch.manual_seed(seed)
    x = torch.randn(2, 3)
    baseline_model = _TinyRelu()
    changed_model = _TinyRelu(offset=offset)
    baseline = tl.log_forward_pass(
        baseline_model,
        x,
        vis_opt="none",
        intervention_ready=True,
    )
    changed = tl.log_forward_pass(
        changed_model,
        x,
        vis_opt="none",
        intervention_ready=True,
    )
    return tl.bundle({"baseline": baseline, "changed": changed}, baseline="baseline")


def _three_model_pairs() -> Iterator[tl.Bundle]:
    """Yield the three model pairs required by Phase 8.

    Yields
    ------
    tl.Bundle
        Bundle under test.
    """

    yield _capture_pair(seed=1, offset=0.0)
    yield _capture_pair(seed=2, offset=0.25)
    yield _capture_pair(seed=3, offset=-0.5)


def test_delta_map_norm_delta_output_delta_on_three_model_pairs() -> None:
    """Bundle delta helpers return stable per-node and per-output payloads."""

    for bundle in _three_model_pairs():
        delta = bundle.delta_map("relative_l2")
        norm_delta = bundle.norm_delta()
        output_delta = bundle.output_delta("baseline")
        comparison = bundle.compare("relative_l2")

        assert delta
        assert norm_delta == delta
        assert set(output_delta) == {"baseline", "changed"}
        assert comparison["nodes"] == delta
        assert comparison["outputs"] == output_delta
        assert comparison["metric"] == "relative_l2"
        assert any("baseline" in node_values for node_values in delta.values())


def test_bundle_method_count_stays_within_phase_budget() -> None:
    """Bundle public method/property count includes Phase 11 persistence."""

    members = [
        name
        for name, value in inspect.getmembers(tl.Bundle)
        if not name.startswith("_") and (inspect.isfunction(value) or isinstance(value, property))
    ]

    assert len(members) <= 26
    assert "save" in members
    assert "supergraph" in members

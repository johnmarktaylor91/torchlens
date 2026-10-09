"""The rank layout must disclose the elements it does not draw.

The rank engine positions IR units only. Orphan islands
(``show_orphans=True``) and standalone intervention hook nodes
(``vis_intervention_mode="as_node"``) are drawn only by the Graphviz dot path,
so a rank render without them must say so instead of dropping them silently.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from example_models import TinyReluAdd
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning


class _WithOrphanIsland(nn.Module):
    """Connected linear path plus a dead-end ``randn -> mul`` island."""

    def __init__(self) -> None:
        """Initialize the connected path."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the connected path and an unreachable island.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        _dead = torch.randn(4) * 2.0
        return self.lin(x)


def _draw(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> tuple[str, list[str]]:
    """Render DOT source and collect TorchLens warning messages."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source = trace.draw(
            vis_save_only=True,
            vis_fileformat="dot",
            vis_outpath=str(tmp_path / name),
            **kwargs,
        )
    messages = [
        str(entry.message) for entry in caught if issubclass(entry.category, TorchLensWarning)
    ]
    return str(source), messages


def _omission_messages(messages: list[str]) -> list[str]:
    """Return the rank-omission disclosures among ``messages``."""

    return [message for message in messages if "rank layout does not draw" in message]


def test_rank_layout_discloses_orphans_it_does_not_draw(tmp_path: Path) -> None:
    """Explicit rank layout with ``show_orphans=True`` warns that the orphans are omitted."""

    torch.manual_seed(0)
    trace = tl.trace(
        _WithOrphanIsland(),
        torch.ones(1, 4),
        capture=tl.options.CaptureOptions(keep_orphans=True),
    )
    try:
        assert trace.orphans
        dot_source, dot_messages = _draw(
            trace, tmp_path, "dot", show_orphans=True, vis_node_placement="dot"
        )
        assert "orphan__" in dot_source
        assert not _omission_messages(dot_messages)

        rank_source, rank_messages = _draw(
            trace, tmp_path, "rank", show_orphans=True, vis_node_placement="rank"
        )
        assert "orphan__" not in rank_source
        (omission,) = _omission_messages(rank_messages)
        assert f"{len(trace.orphans)} orphan node" in omission
        assert "vis_node_placement='dot'" in omission

        _, hidden_messages = _draw(trace, tmp_path, "rank_hidden", vis_node_placement="rank")
        assert not _omission_messages(hidden_messages)
    finally:
        trace.cleanup()


def test_rank_layout_keeps_the_dropped_orphan_husk_warning(tmp_path: Path) -> None:
    """Without ``keep_orphans`` the rank path gives the same re-trace hint as dot."""

    torch.manual_seed(0)
    trace = tl.trace(_WithOrphanIsland(), torch.ones(1, 4))
    try:
        _, messages = _draw(trace, tmp_path, "rank", show_orphans=True, vis_node_placement="rank")
        assert any("re-trace with keep_orphans=True" in message for message in messages)
        assert not _omission_messages(messages)
    finally:
        trace.cleanup()


def test_rank_layout_discloses_hook_nodes_it_does_not_draw(tmp_path: Path) -> None:
    """Explicit rank layout with ``as_node`` hooks warns that the hook nodes are omitted."""

    trace = tl.trace(
        TinyReluAdd(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    try:
        with pytest.warns(tl.errors.MutateInPlaceWarning):
            trace.set(tl.func("relu"), torch.zeros(2, 3))
        dot_source, dot_messages = _draw(
            trace, tmp_path, "dot", vis_intervention_mode="as_node", vis_node_placement="dot"
        )
        assert "intervention_hook_relu" in dot_source
        assert not _omission_messages(dot_messages)

        rank_source, rank_messages = _draw(
            trace, tmp_path, "rank", vis_intervention_mode="as_node", vis_node_placement="rank"
        )
        assert "intervention_hook_" not in rank_source
        (omission,) = _omission_messages(rank_messages)
        assert "1 intervention hook node" in omission

        _, mark_messages = _draw(
            trace,
            tmp_path,
            "rank_mark",
            vis_intervention_mode="node_mark",
            vis_node_placement="rank",
        )
        assert not _omission_messages(mark_messages)
    finally:
        trace.cleanup()

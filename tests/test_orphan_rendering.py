"""Regression tests for ``draw(show_orphans=...)`` island rendering."""

from __future__ import annotations

import graphviz
import pytest
import torch
from torch import nn

import example_models
import torchlens as tl
from torchlens.validation import check_metadata_invariants
from torchlens.validation.invariants import MetadataInvariantError


class WithOrphanIsland(nn.Module):
    """Model with a dead-end computation unreachable from inputs and outputs."""

    def __init__(self) -> None:
        """Initialize a small connected path alongside the island."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a connected path plus an orphan ``randn -> mul`` island."""

        _dead = torch.randn(4) * 2.0  # unreachable from both inputs and outputs
        return self.lin(x)


def _orphan_trace() -> tl.Trace:
    """Trace the island model with orphans retained (opt-in)."""

    torch.manual_seed(0)
    return tl.trace(WithOrphanIsland(), torch.ones(1, 4), keep_orphans=True)


@pytest.mark.parametrize("show_orphans", (False, True))
@pytest.mark.parametrize("model", (WithOrphanIsland(), example_models.OrphanTensors()))
def test_orphan_models_render_and_validate_with_either_visibility(
    model: nn.Module, show_orphans: bool
) -> None:
    """Incident orphan models validate whether islands are rendered or hidden."""

    trace = tl.trace(model, torch.ones(1, 4), keep_orphans=True)
    assert trace.orphans

    dot = trace.draw(show_orphans=show_orphans, return_graph=True, vis_save_only=True)
    source = dot.source if isinstance(dot, graphviz.Digraph) else str(dot)

    assert ("cluster_orphans" in source) is show_orphans
    assert check_metadata_invariants(trace) is True


def test_show_orphans_renders_island_cluster() -> None:
    """``show_orphans=True`` emits a dashed orphan cluster with the island nodes."""

    trace = _orphan_trace()
    dot = trace.draw(show_orphans=True, return_graph=True, vis_save_only=True)
    source = dot.source if isinstance(dot, graphviz.Digraph) else str(dot)

    assert "cluster_orphans" in source
    assert "orphans (unreachable from inputs & outputs)" in source
    assert "orphan__randn_1_1" in source


def test_orphans_hidden_by_default() -> None:
    """Without ``show_orphans`` the island stays out of the rendered graph."""

    trace = _orphan_trace()
    dot = trace.draw(return_graph=True, vis_save_only=True)
    source = dot.source if isinstance(dot, graphviz.Digraph) else str(dot)

    assert "cluster_orphans" not in source
    assert "orphan__randn_1_1" not in source


def test_orphan_island_ops_have_safe_repr_and_list_printing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Completed orphan parent/child ops use the finalized, orphan-aware repr path."""

    trace = _orphan_trace()
    parent, child = tuple(trace.orphans)

    assert parent.children == [child.layer_label]
    assert child.parents == [parent.layer_label]
    assert "PASS NOT FINISHED" not in repr(parent)
    assert "PASS NOT FINISHED" not in repr(child)
    print(list(trace.orphans))
    printed = capsys.readouterr().out
    assert parent.layer_label in printed
    assert child.layer_label in printed


def test_default_orphans_warn_instead_of_rendering_empty_cluster() -> None:
    """Dropped orphan husks do not produce an empty cluster when requested for display."""

    trace = tl.trace(WithOrphanIsland(), torch.ones(1, 4))

    with pytest.warns(UserWarning, match="re-trace with keep_orphans=True"):
        dot = trace.draw(show_orphans=True, return_graph=True, vis_save_only=True)
    source = dot.source if isinstance(dot, graphviz.Digraph) else str(dot)

    assert not trace.orphans
    assert "cluster_orphans" not in source


@pytest.mark.parametrize("show_orphans", (False, True))
def test_orphans_do_not_mask_num_ops_self_consistency_failure(show_orphans: bool) -> None:
    """A bad aggregate count still fails regardless of orphan visibility."""

    trace = _orphan_trace()
    trace.draw(show_orphans=show_orphans, return_graph=True, vis_save_only=True)
    assert check_metadata_invariants(trace) is True

    trace.num_ops += 1

    with pytest.raises(MetadataInvariantError, match="trace_self_consistency"):
        check_metadata_invariants(trace)

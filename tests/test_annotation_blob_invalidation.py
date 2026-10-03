"""Annotation blobs must not survive trace derivation (fork/rerun).

``Trace._annotation_blobs`` holds render-time payloads (feature maps, RDM,
MDS, scree) derived from the CAPTURED activations of one specific run. Any
operation that derives a trace with different run-state — ``fork()`` (a
mutation target) or ``run()``/rerun (fresh activations, possibly a new input
batch) — must invalidate them, or draw-time hooks render OLD activations
captioned with the new trace's labels and alpha-blend old-batch heatmaps onto
the new stimuli.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.viz.feature_maps import feature_map_evolution


class _TinyConv(nn.Module):
    """Small deterministic conv model producing spatial activations."""

    def __init__(self) -> None:
        """Initialize the conv head."""

        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]], [[[-1.0]]]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv layer."""

        return self.conv(x)


def _input_batch(n_stimuli: int = 4, *, offset: float = 0.0) -> torch.Tensor:
    """Return deterministic image-like inputs.

    Parameters
    ----------
    n_stimuli:
        Batch size.
    offset:
        Constant added so two batches carry different values.

    Returns
    -------
    torch.Tensor
        Input tensor with shape ``[N, 1, 4, 4]``.
    """

    values = torch.arange(n_stimuli * 16, dtype=torch.float32).reshape(n_stimuli, 1, 4, 4)
    return values / values.max().clamp_min(1.0) + offset


def _annotated_trace(model: nn.Module) -> tl.Trace:
    """Capture the conv fixture and store feature-map annotation blobs."""

    trace = tl.trace(model, _input_batch(), save=tl.func("conv2d"))
    feature_map_evolution(trace, save=tl.func("conv2d"))
    assert isinstance(trace._annotation_blobs, dict) and trace._annotation_blobs
    return trace


def test_fork_drops_annotation_blobs() -> None:
    """A fork never inherits the parent's render-time annotation payloads."""

    model = _TinyConv().eval()
    trace = _annotated_trace(model)
    fork = trace.fork()
    assert fork._annotation_blobs is None
    # The parent's own blobs stay intact.
    assert isinstance(trace._annotation_blobs, dict) and trace._annotation_blobs


def test_rerun_same_shape_drops_annotation_blobs() -> None:
    """The in-place same-shape rerun refresh invalidates stale blobs."""

    model = _TinyConv().eval()
    trace = _annotated_trace(model)
    stale_maps = trace._annotation_blobs["featmap:layer:conv2d_1_1:maps"].clone()
    tl.run(trace, model, _input_batch(offset=1.0))
    assert trace._annotation_blobs is None, (
        "rerun kept stale annotation blobs; draw() would render the OLD "
        f"activations (stale mean {stale_maps.mean():.4f}) on the new batch"
    )


@pytest.mark.smoke
def test_replace_state_from_drops_annotation_blobs() -> None:
    """The atomic run-state swap takes the fresh log's (empty) blob state."""

    model = _TinyConv().eval()
    trace = _annotated_trace(model)
    new_log = tl.trace(model, _input_batch(offset=1.0), save=tl.func("conv2d"))
    trace.replace_state_from(new_log)
    assert trace._annotation_blobs is None

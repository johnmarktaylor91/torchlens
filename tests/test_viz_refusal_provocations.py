"""Cold-trigger provocations for visualization/lookup refusal codes (R25).

Each test provokes one historically UNPROVOKED_BASELINE code through its
documented public door and asserts the exact ``fields["code"]`` -- the
b6 R25 ratchet direction: a code leaves the baseline only when a test like
these lands, and a code swap at the raise site then fails a test instead
of zero tests.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


@pytest.fixture(scope="module")
def small_trace() -> Iterator[object]:
    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_direction_refuses_typed() -> None:
    from torchlens.visualization._render_utils import direction_to_rankdir

    with pytest.raises(InvalidArgumentError) as exc_info:
        direction_to_rankdir("diagonal")
    assert exc_info.value.fields["code"] == "visualization_direction_invalid"


def test_renderer_refuses_typed(small_trace) -> None:
    with pytest.raises(InvalidArgumentError) as exc_info:
        small_trace.draw(vis_renderer="bogus", vis_save_only=True)
    assert exc_info.value.fields["code"] == "visualization_renderer_invalid"


def test_vis_mode_refuses_typed(small_trace) -> None:
    # The forward draw path reached source_graph's raw ValueError while only
    # the backward/combined gates were typed; all three doors now carry the
    # one code.
    with pytest.raises(InvalidArgumentError) as exc_info:
        small_trace.draw(vis_mode="sideways", vis_save_only=True)
    assert exc_info.value.fields["code"] == "visualization_mode_invalid"


def test_backward_graph_unavailable_refuses_typed(small_trace) -> None:
    with pytest.raises(Exception) as exc_info:
        small_trace.draw_backward(vis_save_only=True)
    assert getattr(exc_info.value, "fields", {}).get("code") == "backward_graph_unavailable"


@pytest.mark.smoke
def test_graphviz_render_failure_refuses_typed(
    small_trace, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchlens.visualization import _render_utils
    from torchlens.visualization._render_common import GraphvizRenderError

    def _boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"synthetic engine crash")

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _boom)
    with pytest.raises(GraphvizRenderError) as exc_info:
        small_trace.draw(vis_outpath=str(tmp_path / "g"), vis_save_only=True)
    assert exc_info.value.fields["code"] == "graphviz_render_failed"

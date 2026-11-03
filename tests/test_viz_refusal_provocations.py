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


@pytest.fixture(scope="module")
def small_trace() -> Iterator[object]:
    trace = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_graphviz_render_failure_refuses_typed(
    small_trace, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from torchlens._visualization import _render_utils
    from torchlens._visualization._render_common import GraphvizRenderError

    def _boom(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"synthetic engine crash")

    monkeypatch.setattr(_render_utils, "run_bounded_subprocess", _boom)
    with pytest.raises(GraphvizRenderError) as exc_info:
        small_trace.draw(path=str(tmp_path / "g"))
    assert exc_info.value.fields["code"] == "graphviz_render_failed"

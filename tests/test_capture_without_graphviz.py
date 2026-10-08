"""A plain capture must not need the drawing dependencies.

graphviz and Pillow are base dependencies, but only the drawing and image
surfaces use them. A ``pip install --no-deps torchlens`` (or any environment
without them) must still be able to run a capture-only ``tl.trace`` and forward
validation; drawing then fails at call time, never at capture time. The
subprocess blocks both with ``sys.modules`` ``None`` entries, so any import of
either raises ``ModuleNotFoundError`` exactly as on a host where it is not
installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CAPTURE_WITHOUT_GRAPHVIZ = textwrap.dedent(
    """
    import sys

    sys.modules["graphviz"] = None
    sys.modules["PIL"] = None

    import torch
    from torch import nn

    import torchlens as tl

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            return torch.relu(self.fc(x)) + 1

    trace = tl.trace(Tiny().eval(), torch.ones(2, 4))
    assert len(trace.layers) > 0
    sparse = tl.trace(
        Tiny().eval(),
        torch.ones(2, 4),
        save=tl.module("fc"),
        capture=tl.options.CaptureOptions(inference_only=True),
    )
    assert sparse.find_sites(tl.module("fc")).first().out is not None
    assert tl.validate(Tiny().eval(), torch.ones(2, 4), scope="forward")
    assert tl.__file__.startswith(sys.argv[1]), tl.__file__
    print("CAPTURE_OK")
    """
)


def test_plain_capture_runs_with_drawing_dependencies_blocked() -> None:
    """Capture and forward validation succeed when graphviz and Pillow cannot be imported."""

    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _CAPTURE_WITHOUT_GRAPHVIZ, str(_REPO_ROOT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "CAPTURE_OK" in result.stdout

"""A plain capture must not need graphviz.

graphviz is a base dependency, but only the drawing surface uses it. A
``pip install --no-deps torchlens`` (or any environment without graphviz) must
still be able to run a capture-only ``tl.trace``; drawing then fails at call
time, never at capture time. The subprocess blocks graphviz with a
``sys.modules`` ``None`` entry, so any import of it raises
``ModuleNotFoundError`` exactly as on a host where it is not installed.
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
    assert tl.__file__.startswith(sys.argv[1]), tl.__file__
    print("CAPTURE_OK")
    """
)


def test_plain_capture_runs_with_graphviz_blocked() -> None:
    """A capture-only trace succeeds when graphviz cannot be imported."""

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

import subprocess
import sys
import textwrap


def test_dagua_renderer_requires_experimental_opt_in() -> None:
    """Core renderer rejects Dagua before the experimental module is imported."""

    script = textwrap.dedent(
        """
import torch
from torch import nn
import torchlens as tl

class M(nn.Module):
    def forward(self, x):
        return x + 1

log = tl.trace(M(), torch.ones(1), save=None)
try:
    try:
        log.draw(renderer="dagua", file_format="svg")
    except RuntimeError as exc:
        message = "opt in via `from torchlens.experimental import dagua` first"
        raise SystemExit(0 if message in str(exc) else 2)
    raise SystemExit(1)
finally:
    log.cleanup()
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

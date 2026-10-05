"""Torch's own lazy ``torch._dynamo`` import never crashes under installed wrappers.

On meta tensors torch runs Python decompositions as meta kernels, and several are
wrapped in ``torch._compile._disable_dynamo``, which imports ``torch._dynamo`` on
first call. That import's module bodies call ordinary torch functions. In a fresh
process (nothing has imported ``torch._dynamo`` yet) those calls reached TorchLens's
installed wrappers mid-import, re-entered a decomposition, and hit ``_disable_dynamo``
again while ``torch._dynamo`` was half initialised (``AttributeError: partially
initialized module 'torch._dynamo' has no attribute 'disable'``). A long suite hides
the bug because some earlier test usually imports ``torch._dynamo`` first, so every
case here runs in its own fresh subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_PROLOGUE = """
import sys
import torch
from torch import nn
import torchlens as tl
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.options import CaptureOptions

assert "torch._dynamo" not in sys.modules, "precondition: torch._dynamo already imported"
"""

_CASES = {
    "witness_structure_only_meta": """
wrap_torch(completeness_witness=True)
with torch.device("meta"):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
trace = tl.trace(
    model.eval(),
    torch.empty(2, 4, device="meta"),
    capture=CaptureOptions(structure_only=True),
)
assert trace.completeness_witness_verified is True, trace.completeness_witness_verified
assert trace.capture_verified is None, trace.capture_verified
""",
    "structure_only_meta": """
with torch.device("meta"):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
trace = tl.trace(
    model.eval(),
    torch.empty(2, 4, device="meta"),
    capture=CaptureOptions(structure_only=True),
)
assert len(trace.ops) > 0
""",
    "wrapped_meta_forward_outside_capture": """
wrap_torch(completeness_witness=True)
with torch.device("meta"):
    model = nn.Linear(4, 4)
out = model(torch.empty(2, 4, device="meta"))
assert out.shape == (2, 4)
""",
}


@pytest.mark.heavy
@pytest.mark.parametrize("case", sorted(_CASES))
def test_fresh_process_meta_capture_survives_lazy_dynamo_import(case: str) -> None:
    """A fresh process whose first ``torch._dynamo`` import fires inside wrapped torch."""

    script = textwrap.dedent(_PROLOGUE) + textwrap.dedent(_CASES[case]) + "print('OK')\n"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0 and "OK" in completed.stdout, (
        f"fresh-process case {case!r} failed ({completed.returncode}):\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )

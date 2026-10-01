"""The modern-spelling quickstart gate: zero warnings, in a fresh process.

Quickstart memo B1: the block a newcomer actually types (modern spellings
only -- ``tl.trace`` + ``summary()`` + indexed reads, eval-mode torchvision
resnet18) must emit ZERO warnings. Historically the only spellings that
controlled render output were deprecated ones and TorchLens's own internal
calls used deprecated kwargs, so a warning-free quickstart could not be
written; the shim-removal pass fixed the spellings and this gate keeps the
first screen warning-free permanently. A subprocess keeps the measurement
cold (no import-order or warning-registry state from the test session).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.heavy

_REPO_ROOT = Path(__file__).resolve().parent.parent

_MODERN_SPELLING_BLOCK = """
import json
import sys
import warnings

captured = []
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import torch
    import torchvision.models as models
    import torchlens as tl

    model = models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 224, 224)

    log = tl.trace(model, x)
    summary = log.summary()
    conv_shape = tuple(log["conv2d_1_1"].out.shape)
    module_shape = tuple(log["conv1"].out.shape)
    captured = [
        f"{type(w.message).__name__}: {w.message}" for w in caught
    ]

print(json.dumps({
    "warnings": captured,
    "summary_ok": "ResNet" in summary,
    "conv_shape": conv_shape,
    "module_shape": module_shape,
}))
"""


def test_modern_spelling_resnet_block_emits_zero_warnings() -> None:
    """The eval-mode resnet18 quickstart block runs warning-free, cold."""

    pytest.importorskip("torchvision")
    completed = subprocess.run(
        [sys.executable, "-c", _MODERN_SPELLING_BLOCK],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, (
        f"modern-spelling block failed ({completed.returncode}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload["warnings"] == [], (
        "the modern-spelling quickstart block must emit ZERO warnings; got:\n"
        + "\n".join(payload["warnings"])
    )
    assert payload["summary_ok"], "summary() lost the model name"
    assert payload["conv_shape"] == [1, 64, 112, 112]
    assert payload["module_shape"] == [1, 64, 112, 112]

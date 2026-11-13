"""BackwardPass reads its torch-only providers through a leaf registry.

``_data_classes/backward_pass.py`` must not import the torch backward census or
the report attribution module (that tied every backward module into the largest
import cycle); the providers register themselves when their modules import.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import torchlens
from torchlens._data_classes import _backward_providers

_BACKWARD_PASS = Path(torchlens.__file__).parent / "_data_classes" / "backward_pass.py"


def test_backward_pass_imports_no_provider_module() -> None:
    """No import of the census or attribution module, at module top or in a function."""

    tree = ast.parse(_BACKWARD_PASS.read_text(encoding="utf-8"))
    imported = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name.endswith("_backward_dispatch") for name in imported)
    assert not any(name.endswith("_backward_attribution") for name in imported)


_FRESH_CAPTURE = """
import torch
import torchlens as tl
from torchlens._data_classes import _backward_providers

model = torch.nn.Linear(2, 1)
trace = tl.trace(model, torch.randn(3, 2), capture=tl.options.CaptureOptions(backward_ready=True))
trace.backward(trace.output.sum())
for name in ("recorded_ops", "attribute_census", "executed_flops_record", "attribution_from_record"):
    assert callable(_backward_providers.backward_provider(name)), name
print("ok")
"""


@pytest.mark.heavy
def test_a_fresh_capture_and_backward_fill_every_provider() -> None:
    """In a fresh interpreter, ``tl.trace`` imports cleanly and a backward fills all four providers.

    A fresh process matters: an eager import of the report package from the
    capture stack once closed an import cycle that only a process importing
    ``torchlens.report`` first could hide.
    """

    done = subprocess.run(
        [sys.executable, "-c", _FRESH_CAPTURE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    assert done.stdout.strip().endswith("ok")


def test_the_loader_record_check_fills_the_attribution_providers() -> None:
    """The load-time executed-FLOPs record check imports the attribution module."""

    from torchlens._io import _load_validators
    from torchlens.report import _backward_attribution

    source = Path(_load_validators.__file__).read_text(encoding="utf-8")
    assert "from ..report._backward_attribution import" in source
    assert _backward_providers.backward_provider("attribution_from_record") is (
        _backward_attribution.attribution_from_record
    )

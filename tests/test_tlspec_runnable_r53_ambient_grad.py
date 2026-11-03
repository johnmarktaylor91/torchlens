"""r53 hon_1: ambient global autograd/inference mode is RECORDED and RESTORED.

A pure-Python branch on ``torch.is_grad_enabled()`` / ``is_inference_mode_enabled()``
is steered by an ambient GLOBAL every other result-affecting knob already records.
The whole-class immunizer: the mode is a REQUIRED recorded ambient field
(snapshot-coverage meta-test pins schema/snapshot lockstep), replay re-enters the
recorded mode as scoped contexts (a fresh-instance comparison under the recorded
ambient takes the SAME branch -> honest VERIFIED), and a missing field is a typed
analysis-only refusal -- never a defaulted grad mode (a default is a false
VERIFIED). A Python READ of the flag is deliberately never ceilinged: library code
reads it constantly, so the VERIFIED assertions here double as the no-over-trigger
pin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import PathFaithfulness
from torchlens.errors import RunnableTLSPECError
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)


class GradBranchModel(nn.Module):
    """The r52 hon_1 repro: a Python branch on the global autograd mode."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * (2.0 if torch.is_grad_enabled() else 3.0)


class InferenceBranchModel(nn.Module):
    """Same class, keyed on the global inference mode."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * (5.0 if torch.is_inference_mode_enabled() else 7.0)


def _save_runnable(model: nn.Module, x: torch.Tensor, path: Path, **save_kwargs: Any) -> Path:
    trace = tl.trace(model, x.clone(), capture=_CAPTURE)
    trace.save(path, level="runnable", **save_kwargs)
    return path


def _descriptor_ambient(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    return dict(manifest["run"]["ambient_context"])


# Kept: the catch of its regression fault.
def test_no_grad_capture_records_false_and_matches_no_grad_oracle(tmp_path: Path) -> None:
    """A ``no_grad()`` capture records ``grad_enabled=False`` and VERIFIED means
    faithfulness against the no_grad oracle -- under ANY caller ambient."""

    torch.manual_seed(0)
    model = GradBranchModel().eval()
    x = torch.randn(2, 3)
    with torch.no_grad():
        trace = tl.trace(model, x.clone(), capture=_CAPTURE)
        expected_no_grad = model(x.clone())
    path = tmp_path / "no-grad.tlspec"
    trace.save(path, level="runnable", include_weights=True)

    assert _descriptor_ambient(path)["grad_enabled"] is False
    result = tl.load(path).run(inputs=x.clone())  # caller is grad-ON here
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, expected_no_grad)


# Kept: sole branch coverage of _io/runnable_load.py.
@pytest.mark.parametrize("field", ("grad_enabled", "inference_mode", "fill_uninitialized_memory"))
def test_missing_ambient_mode_field_is_typed_analysis_only(field: str, tmp_path: Path) -> None:
    """A v2 descriptor missing an ambient mode field (a pre-r53 dev-window
    artifact) refuses at parse -- ``context_field_invalid`` analysis-only load,
    NEVER a defaulted mode and NEVER a hard load failure."""

    torch.manual_seed(0)
    model = GradBranchModel().eval()
    x = torch.randn(2, 3)
    # Value-free descriptor (no weights): the load surfaces the typed readiness
    # refusal instead of hard-failing on a missing parsed descriptor.
    path = _save_runnable(model, x, tmp_path / "tampered.tlspec")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run"]["ambient_context"].pop(field)
    manifest_path.write_text(json.dumps(manifest))

    loaded = tl.load(path)  # analysis-only load must succeed
    readiness = loaded.readiness
    assert readiness is not None
    assert readiness.status.value == "unavailable"
    codes = " ".join(str(d.code.value) for d in readiness.diagnostics)
    stages = " ".join(str(d.detection_stage) for d in readiness.diagnostics)
    assert "context_field_invalid" in codes, codes
    assert "context_parse_validation" in stages, stages
    with pytest.raises(RunnableTLSPECError):
        loaded.run(inputs=torch.randn(2, 3))

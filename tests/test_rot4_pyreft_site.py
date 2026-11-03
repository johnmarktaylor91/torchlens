"""``LoreftArtifact.subspace`` given a site from a trace checks the site's shape first.

The edit acts on the trailing axis, so a site whose trailing axis is not the artifact's
width (or whose feature axis is not last) refuses ``pyreft_site_invalid``, and positions
that do not cover the other axes refuse ``pyreft_positions_invalid``; each remedy runs.
A site that fits gives the same edit as its label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_bridge_pyreft import _D, _SITE, _X, _positions, _ready, _write
from torchlens._errors import InvalidArgumentError


class _Transposed(nn.Module):
    """``lin(x).transpose(1, 2)``: a ``[batch, 6, seq]`` site with the feature axis first."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(_D, _D).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).transpose(1, 2) * 2.0


def _site_of_shape(trace: Any, shape: tuple[int, ...]) -> Any:
    sites = [op for _, op in trace.ops.items() if tuple(getattr(op, "shape", ()) or ()) == shape]
    assert sites, f"no site of shape {shape}"
    return sites[0]


def _remedy_runs(error: InvalidArgumentError, artifact: Any) -> None:
    """Run the refusal's spelling on the site that fits and compare with the label spelling."""

    remedy = error.fields["remedy"]
    assert "artifact.subspace(trace[label], positions=positions)" in remedy
    trace = _ready()
    positions = torch.zeros(trace[_SITE].shape[:-1], dtype=torch.bool)
    positions[:, -1] = True
    fork = trace.fork()
    fork.do(artifact.subspace(trace[_SITE], positions=positions), artifact.edit)
    by_site = fork.push()[_SITE].out
    fork = _ready().fork()
    fork.do(artifact.subspace(_SITE, positions=positions), artifact.edit)
    assert torch.equal(by_site, fork.push()[_SITE].out)


def test_site_whose_trailing_axis_is_another_width_refuses_and_the_remedy_runs(
    tmp_path: Path,
) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    trace = _ready()
    narrow = _site_of_shape(trace, (2, 5, 3))
    with pytest.raises(InvalidArgumentError) as excinfo:
        artifact.subspace(narrow, positions=_positions())
    fields = excinfo.value.fields
    assert fields["code"] == "pyreft_site_invalid"
    assert fields["site"] == narrow.label and narrow.label in str(excinfo.value)
    assert fields["site_shape"] == (2, 5, 3) and fields["artifact_width"] == _D
    assert "3 wide" in str(excinfo.value) and f"{_D}-wide" in str(excinfo.value)
    _remedy_runs(excinfo.value, artifact)


def test_site_whose_feature_axis_is_not_last_refuses_and_the_remedy_runs(tmp_path: Path) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    trace = tl.trace(_Transposed().eval(), _X, capture=tl.options.CaptureOptions(replayable=True))
    transposed = _site_of_shape(trace, (2, _D, 5))
    with pytest.raises(InvalidArgumentError) as excinfo:
        artifact.subspace(transposed)
    assert excinfo.value.fields["code"] == "pyreft_site_invalid"
    assert excinfo.value.fields["site"] == transposed.label
    assert "feature axis is not last" in str(excinfo.value)
    _remedy_runs(excinfo.value, artifact)


def test_positions_that_miss_the_site_axes_refuse_and_the_remedy_runs(tmp_path: Path) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    trace = _ready()
    with pytest.raises(InvalidArgumentError) as excinfo:
        artifact.subspace(trace[_SITE], positions=torch.ones(2, 4, dtype=torch.bool))
    assert excinfo.value.fields["code"] == "pyreft_positions_invalid"
    assert excinfo.value.fields["site_shape"] == (2, 5, _D)
    _remedy_runs(excinfo.value, artifact)


def test_something_that_is_not_a_site_refuses_and_the_remedy_runs(tmp_path: Path) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    with pytest.raises(InvalidArgumentError) as excinfo:
        artifact.subspace(7)
    assert excinfo.value.fields["code"] == "pyreft_site_invalid"
    _remedy_runs(excinfo.value, artifact)


def test_site_that_fits_gives_the_same_edit_as_its_label(tmp_path: Path) -> None:
    artifact = tl.bridge.pyreft.read(_write(tmp_path / "artifact"))
    for positions in (None, _positions()):
        trace = _ready()
        fork = trace.fork()
        fork.do(artifact.subspace(trace[_SITE], positions=positions), artifact.edit)
        by_site = fork.push()[_SITE].out
        fork = _ready().fork()
        fork.do(artifact.subspace(_SITE, positions=positions), artifact.edit)
        assert torch.equal(by_site, fork.push()[_SITE].out)

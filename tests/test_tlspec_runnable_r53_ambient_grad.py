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

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnableTLSPECError
from torchlens.options import CaptureOptions
from torchlens.runnable import AmbientExecutionContext, PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


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


@pytest.mark.smoke
def test_grad_on_capture_records_ambient_and_replays_recorded_arm(tmp_path: Path) -> None:
    """Grad-ON capture: recorded ``grad_enabled=True``; replay under a caller's
    ``no_grad()`` still takes the RECORDED arm (the hon_1 false-VERIFIED class).
    """

    torch.manual_seed(0)
    model = GradBranchModel().eval()
    x = torch.randn(2, 3)
    path = _save_runnable(model, x, tmp_path / "grad-on.tlspec", include_weights=True)

    ambient = _descriptor_ambient(path)
    assert ambient["grad_enabled"] is True
    assert ambient["inference_mode"] is False

    expected_grad_on = model(x.clone())  # fresh oracle under the recorded ambient
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, expected_grad_on)

    # The sharp class pin: a caller replaying inside the STANDARD inference
    # context must still observe the recorded grad-ON branch, because the
    # scoped ambient restore re-enters the recorded global around the run.
    with torch.no_grad():
        no_grad_result = tl.load(path).run(inputs=x.clone())
        assert not torch.is_grad_enabled()  # caller context intact inside
    assert torch.allclose(no_grad_result.output, expected_grad_on)
    assert no_grad_result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.is_grad_enabled()  # caller ambient restored on exit


@pytest.mark.smoke
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


def test_inference_mode_capture_records_and_replays(tmp_path: Path) -> None:
    """An ambient ``inference_mode()`` capture records True and replays its arm."""

    torch.manual_seed(0)
    model = InferenceBranchModel().eval()
    x = torch.randn(2, 3)
    with torch.inference_mode():
        trace = tl.trace(model, x.clone(), capture=_CAPTURE)
        expected = model(x.clone()).clone()
    path = tmp_path / "inference.tlspec"
    trace.save(path, level="runnable", include_weights=True)

    ambient = _descriptor_ambient(path)
    assert ambient["inference_mode"] is True
    result = tl.load(path).run(inputs=x.clone())  # caller NOT in inference mode
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, expected)
    assert not torch.is_inference_mode_enabled()  # caller ambient restored


@pytest.mark.smoke
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


@pytest.mark.parametrize("field", ("grad_enabled", "inference_mode"))
def test_null_ambient_mode_is_a_typed_refusal_never_a_default(field: str, tmp_path: Path) -> None:
    """``null`` is not a legal producer value for the global mode (every
    supported torch exposes both queries): a nulled field is a typed refusal
    through EITHER the manifest schema preflight or the strict parser --
    a defaulted-mode run is unreachable both ways."""

    torch.manual_seed(0)
    model = GradBranchModel().eval()
    x = torch.randn(2, 3)
    path = _save_runnable(model, x, tmp_path / "nulled.tlspec")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run"]["ambient_context"][field] = None
    manifest_path.write_text(json.dumps(manifest))

    try:
        loaded = tl.load(path)
    except Exception as exc:  # schema preflight refusal (typed IO error)
        assert field in str(exc) or "ambient_context" in str(exc)
        return
    readiness = loaded.readiness
    assert readiness is not None
    assert readiness.status.value == "unavailable"
    with pytest.raises(RunnableTLSPECError):
        loaded.run(inputs=torch.randn(2, 3))


def test_ambient_snapshot_covers_every_recorded_field() -> None:
    """Whole-class immunizer: snapshot keys == AmbientExecutionContext fields
    (minus the derived ``attestation_ineligible_context``). A future
    result-affecting knob added to either side without the other FAILS here."""

    from torchlens.utils._torch_compat import snapshot_ambient_execution_context

    snapshot_keys = set(snapshot_ambient_execution_context())
    recorded_fields = {field.name for field in dataclasses.fields(AmbientExecutionContext)} - {
        "attestation_ineligible_context"
    }
    assert snapshot_keys == recorded_fields


def test_live_grad_read_never_ceilings_or_degrades(tmp_path: Path) -> None:
    """No-over-trigger pin: a model that merely READS ``torch.is_grad_enabled()``
    stays VERIFIED (+ATTESTED with archive) -- reads are never witnessed."""

    from torchlens.runnable import NumericAttestationStatus

    torch.manual_seed(0)
    model = GradBranchModel().eval()
    x = torch.randn(2, 3)
    path = _save_runnable(
        model,
        x,
        tmp_path / "attest.tlspec",
        include_weights=True,
        include_activations=True,
    )
    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert report.nondeterministic_sources == ()

"""r35 stage-0 regression: v2 REQUIRED-explicit context records and legacy-v1 posture.

Decision A: v2 context records are REQUIRED and EXPLICIT -- the parser REJECTS
absence, so an absent record can only mean a legacy v1 artifact, which loads
analysis-only with a typed readiness refusal (decision G). No absent context is
ever interpreted as a default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.runnable import (
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    ReadinessStatus,
    RunnableErrorCode,
)

pytestmark = pytest.mark.smoke


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _save_runnable(tmp_path: Path, **save_kwargs: Any) -> Path:
    model = _TinyModel().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, intervention_ready=True)
    bundle = tmp_path / "artifact.tlspec"
    tl.save(trace, str(bundle), level="runnable", **save_kwargs)
    return bundle


def _rewrite_manifest(bundle: Path, mutate: Any) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))


def test_v2_descriptor_round_trips_with_required_context(tmp_path: Path) -> None:
    """A fresh v2 artifact records explicit call + ambient context and runs."""

    bundle = _save_runnable(tmp_path, include_weights=True)
    loaded = tl.load(str(bundle))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert descriptor.capability == RUNNABLE_TLSPEC_SCHEMA_VERSION
    assert descriptor.ambient_context.default_dtype == "torch.float32"
    for call in descriptor.calls:
        # Disabled autocast is written affirmatively, never omitted.
        assert call.execution_context.autocast, call.call_id
        assert all(entry.enabled is False for entry in call.execution_context.autocast)
    result = loaded.run(inputs=torch.randn(2, 4))
    assert result.report.path_faithfulness.value == "verified"


def test_legacy_v1_capability_loads_analysis_only(tmp_path: Path) -> None:
    """A legacy v1 descriptor loads for analysis and refuses run() typed."""

    bundle = _save_runnable(tmp_path)

    def _to_legacy(manifest: dict[str, Any]) -> None:
        run = manifest["run"]
        run["capability"] = "sparse_recorded_taken_path_v1"
        run["call_recipe"] = "non_tensor_args_and_tensor_slots_v1"
        run["initializer_policy_version"] = "torchlens_role_init_v1"
        run["compatibility"]["descriptor_version"] = "sparse_recorded_taken_path_v1"
        run["compatibility"]["call_recipe_version"] = "non_tensor_args_and_tensor_slots_v1"
        run["compatibility"]["initializer_policy_version"] = "torchlens_role_init_v1"
        run["payload_layers"]["activations"]["schema"] = "selected_activation_v1"
        run.pop("ambient_context", None)
        # r69 A: the required-witness inventory is v2-only, exactly like the ambient
        # context -- a genuine legacy v1 artifact predates and never carries it.
        run.pop("required_witness_inventory", None)
        # r71 A: the input-boundary record, the gap ledger, and the per-call/slot
        # obligation fields are likewise v2-only.
        run.pop("input_boundary", None)
        run.pop("coverage_gaps", None)
        for call in run["calls"]:
            call.pop("execution_context", None)
            call.pop("control_obligations", None)
            call.pop("control_dependencies", None)
        for slot in run["tensor_slots"]:
            slot.pop("host_escape", None)
            slot.pop("inert_sink", None)

    _rewrite_manifest(bundle, _to_legacy)
    loaded = tl.load(str(bundle))
    readiness = loaded.__dict__["_runnable_readiness"]
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert readiness.capability == "sparse_recorded_taken_path_v1"
    codes = {diag.code for diag in readiness.diagnostics}
    assert RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE in codes
    joined = " ".join(diag.message for diag in readiness.diagnostics)
    assert "analysis-only" in joined
    assert "context" in joined
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.run(inputs=torch.randn(2, 4))


def test_v2_descriptor_missing_call_context_is_rejected(tmp_path: Path) -> None:
    """A v2 descriptor with an absent per-call context is refused, not defaulted."""

    bundle = _save_runnable(tmp_path)

    def _strip_call_context(manifest: dict[str, Any]) -> None:
        manifest["run"]["calls"][0].pop("execution_context", None)

    _rewrite_manifest(bundle, _strip_call_context)
    loaded = tl.load(str(bundle))
    readiness = loaded.__dict__["_runnable_readiness"]
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    with pytest.raises(RunCapabilityUnavailableError):
        loaded.run(inputs=torch.randn(2, 4))


def test_v2_descriptor_missing_ambient_context_is_rejected(tmp_path: Path) -> None:
    """A v2 descriptor with an absent ambient context is refused, not defaulted."""

    bundle = _save_runnable(tmp_path)

    def _strip_ambient(manifest: dict[str, Any]) -> None:
        manifest["run"].pop("ambient_context", None)

    _rewrite_manifest(bundle, _strip_ambient)
    with pytest.raises(Exception, match="ambient_context|fields mismatch"):
        tl.load(str(bundle))

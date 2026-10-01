"""F03 ledger memo items 5b + 5c.

5b: the edited-rerun divergence-detector composition row ("after 5b: no
warning, divergence_count == 0"). The panel's 3.4 defect (TorchLens's own
edit tripping the detector on resnet18, measured at f74e4a9e) does NOT
reproduce on the F03 base tip — a 4-cell mode x helper matrix on torchvision
resnet18 plus a Conv+BN model all rerun clean — so this file PINS the
healthy behavior rather than re-fixing a dead defect (the memo itself
records the evidence as a single lab's unreplicated round-3 insertion).

5c: bundle-level retained-bytes aggregation (lower-bound labelled) and the
typed BEFORE-the-first-candidate retention preflight.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle._bytes import bundle_retained_bytes, preflight_retention_projection
from torchlens.errors.episode import BundleExperimentError
from torchlens.intervention.errors import ControlFlowDivergenceWarning

pytestmark = pytest.mark.smoke


class _ConvBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x))).mean()


# ---------------------------------------------------------------------------
# 5b: edited rerun x divergence detector (the memo's composition row)
# ---------------------------------------------------------------------------


def test_edited_rerun_does_not_trip_divergence_detector() -> None:
    torch.manual_seed(0)
    model = _ConvBN().eval()
    x = torch.randn(2, 3, 8, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = log.fork()
    fork.attach_hooks(tl.when(tl.func("relu"), tl.scale(0.5)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork.run(model, x)
    divergence_warnings = [
        w for w in caught if issubclass(w.category, ControlFlowDivergenceWarning)
    ]
    assert not divergence_warnings, [str(w.message) for w in divergence_warnings]
    rerun_rows = [
        row for row in fork.state_history if isinstance(row, dict) and "divergence_count" in row
    ]
    assert rerun_rows, "rerun wrote no state-history row"
    assert all(row["divergence_count"] == 0 for row in rerun_rows)


def test_edited_rerun_strict_does_not_raise() -> None:
    torch.manual_seed(0)
    model = _ConvBN().eval()
    x = torch.randn(2, 3, 8, 8)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = log.fork()
    fork.attach_hooks(tl.when(tl.func("relu"), tl.scale(0.5)))
    fork.run(model, x, replay=tl.options.ReplayOptions(strict=True))


# ---------------------------------------------------------------------------
# 5c: retained-bytes aggregation + retention preflight
# ---------------------------------------------------------------------------


def test_bundle_retained_bytes_lower_bound_aggregation() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 3, 8, 8)
    members = {
        name: tl.trace(
            _ConvBN().eval(), x, capture=tl.options.CaptureOptions(intervention_ready=True)
        )
        for name in ("a", "b")
    }
    bundle = tl.Bundle(members)
    report = bundle_retained_bytes(bundle)
    assert report["basis"] == "lower_bound"
    assert set(report["per_member"]) == {"a", "b"}
    assert report["per_member"]["a"] > 0
    assert report["total_bytes"] == sum(report["per_member"].values())
    assert report["unmetered_members"] == []


def test_bundle_retained_bytes_names_unmetered_members() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 3, 8, 8)
    member = tl.trace(_ConvBN().eval(), x)
    member._save_budget_accountant = None  # a loaded/legacy member
    bundle = tl.Bundle({"legacy": member})
    report = bundle_retained_bytes(bundle)
    assert report["per_member"]["legacy"] == 0
    assert report["unmetered_members"] == ["legacy"]


def test_retention_preflight_refuses_typed_before_any_work() -> None:
    with pytest.raises(BundleExperimentError) as excinfo:
        preflight_retention_projection(
            per_candidate_bytes=500_000_000,
            retained_candidates=12,
            ceiling_bytes=1_000_000_000,
        )
    fields = excinfo.value.fields
    assert fields["code"] == "retention_projection_over_budget"
    assert fields["projected_bytes"] == 6_000_000_000
    assert fields["ceiling_bytes"] == 1_000_000_000
    assert fields["basis"] == "lower_bound"
    assert "BEFORE the first" in str(excinfo.value)

    record = preflight_retention_projection(
        per_candidate_bytes=10,
        retained_candidates=3,
        ceiling_bytes=1_000,
        current_bytes=100,
    )
    assert record["projected_bytes"] == 130

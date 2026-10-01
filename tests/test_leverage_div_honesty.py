"""B3 + B8 + B9: honesty batch, overlay repair, escrow bound + address preflight.

Every raise site added by lane F34 gets its provoking test here (census law):
``node_overlay_zero_match``, ``node_overlay_partial_match`` (warning),
``cone_origin_invalid``, ``site_table_empty``, ``escrow_budget_exceeded``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from test_leverage_div_fixtures import ReusedReluNet

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.capture.preflight import address_preflight
from torchlens.errors._base import TorchLensWarning

pytestmark = pytest.mark.smoke

_SAVE_ALL = {"capture": tl.options.CaptureOptions(layers_to_save="all")}


def _traced_toy():
    torch.manual_seed(0)
    model = ReusedReluNet()
    x = torch.rand(1, 2, 3, 3)
    return model, x, tl.trace(model, x, **_SAVE_ALL)


# ---------------------------------------------------------------------------
# B8: the overlay door stops lying.
# ---------------------------------------------------------------------------


def test_overlay_zero_match_refuses(tmp_path):
    """A mapping that paints nothing refuses typed — never a blank success."""

    _, _, trace = _traced_toy()
    scores = {"s1|relu|relu||1": 1.0, "some_foreign_label": 2.0}
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.draw(node_overlay=scores, vis_outpath=str(tmp_path / "g"), vis_save_only=True)
    assert excinfo.value.fields["code"] == "node_overlay_zero_match"


def test_overlay_partial_match_discloses(tmp_path):
    """A partially-matched mapping renders WITH the realized-rows disclosure."""

    _, _, trace = _traced_toy()
    scores = {"relu_1_2": 1.0, "not_a_real_label": 2.0}
    with pytest.warns(TorchLensWarning) as warning_records:
        trace.draw(node_overlay=scores, vis_outpath=str(tmp_path / "g"), vis_save_only=True)
    codes = {getattr(record.message, "fields", {}).get("code") for record in warning_records}
    assert "node_overlay_partial_match" in codes


def test_overlay_full_match_renders_silently(tmp_path, recwarn):
    """A fully-realized mapping renders without overlay warnings."""

    _, _, trace = _traced_toy()
    trace.draw(node_overlay={"relu_1_2": 1.0}, vis_outpath=str(tmp_path / "g"), vis_save_only=True)
    codes = {getattr(record.message, "fields", {}).get("code", None) for record in recwarn.list}
    assert "node_overlay_partial_match" not in codes


def test_overlay_selection_source_lowers_to_selected_counts(tmp_path):
    """Element-mask law: a Selection paints selected COUNTS, never membership."""

    from torchlens.visualization.overlays import resolve_overlay_request

    _, _, trace = _traced_toy()
    selection = tl.units("relu_1_2", [(0, 0, 0, 0), (0, 1, 1, 1)]).resolve(trace)
    scores = resolve_overlay_request(trace, selection)
    assert scores == {"relu_1_2": 2}  # 2 selected elements, not "member: yes"


# ---------------------------------------------------------------------------
# B3: honesty batch.
# ---------------------------------------------------------------------------


def test_cone_of_effect_string_origin_refuses():
    from torchlens.intervention.replay import cone_of_effect

    _, _, trace = _traced_toy()
    with pytest.raises(InvalidArgumentError) as excinfo:
        cone_of_effect(trace, "relu_1_2")
    assert excinfo.value.fields["code"] == "cone_origin_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        cone_of_effect(trace, ["relu_1_2"])
    assert excinfo.value.fields["code"] == "cone_origin_invalid"


def test_cone_of_effect_op_origins_still_work():
    from torchlens.intervention.replay import cone_of_effect

    _, _, trace = _traced_toy()
    cone = cone_of_effect(trace, trace["relu_1_2"].ops)
    assert any(op.layer_label == "relu_2_4" for op in cone)


def test_do_accepts_find_sites_output():
    """NEW-4: the refusal that recommends find_sites() accepts its output."""

    torch.manual_seed(0)
    model = ReusedReluNet()
    x = torch.rand(1, 2, 3, 3)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = trace.fork()
    sites = trace.find_sites("conv2d_1_1")
    fork.do(sites, tl.zero_ablate())
    assert fork.intervention_audit  # fire evidence, never silence


def test_do_empty_site_table_refuses():
    from torchlens.intervention.errors import SiteResolutionError
    from torchlens.intervention.resolver import SiteTable

    _, _, trace = _traced_toy()
    fork = trace.fork()
    with pytest.raises(SiteResolutionError) as excinfo:
        fork.do(SiteTable((), query="nothing"), tl.zero_ablate())
    assert excinfo.value.fields["code"] == "site_table_empty"


def test_when_refuses_raw_string_at_build():
    """NEW-16 (landed upstream, pinned here): spec-build refusal is typed."""

    with pytest.raises(Exception) as excinfo:
        tl.when("relu_1_2", tl.zero_ablate())
    assert "WHERE term must be a callable selector or predicate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# B9: escrow bound + address preflight.
# ---------------------------------------------------------------------------


def test_escrow_spill_budget_refuses_typed():
    """Crossing the declared spill bound refuses with the accounting on fields."""

    from torchlens.capture.plan import RetentionKind, RetentionProfile
    from torchlens.capture.session import CaptureSession

    profile = RetentionProfile(
        activation_kind=RetentionKind.ACTIVATION,
        activation_window=None,
        spillable=True,
        activation_ram_budget_bytes=64,
        escrow_spill_budget_bytes=128,
    )
    session = CaptureSession.__new__(CaptureSession)
    plan = type("_Plan", (), {"retention_profile": profile})()
    session.plan = plan
    session.activation_escrow = {}
    session.activation_escrow_ram_bytes = 0
    session.activation_escrow_peak_ram_bytes = 0
    session.activation_escrow_spilled_bytes = 0
    session._activation_escrow_spill_index = 0
    session._activation_spill_dir = None
    with pytest.raises(SaveBudgetExceededError) as excinfo:
        for index in range(8):
            session.escrow_candidate(index, torch.zeros(32))  # 128 B each
    assert excinfo.value.fields["code"] == "escrow_budget_exceeded"
    assert excinfo.value.fields["budget_bytes"] == 128


def test_escrow_budget_none_disables_bound(tmp_path):
    from torchlens.capture.plan import RetentionProfile

    profile = RetentionProfile(escrow_spill_budget_bytes=None)
    assert profile.escrow_spill_budget_bytes is None


def test_address_preflight_classifies_and_names_equivalent():
    """The preflight prices the address and NAMES the cheap spelling."""

    _, _, trace = _traced_toy()
    report = address_preflight(["conv2d_1_1", "conv1", -1], trace)
    verdicts = {str(v.component): v for v in report.verdicts}
    assert verdicts["conv2d_1_1"].resolution == "deferred"
    assert verdicts["conv2d_1_1"].equivalent_spelling == "conv1"
    assert verdicts["conv1"].resolution == "live"
    assert verdicts["-1"].resolution == "live"
    assert report.escrowing  # the label component forces the escrow lane
    live_only = address_preflight(["conv1", "relu"])
    assert not live_only.escrowing


def test_address_preflight_output_and_ordinal_deferred():
    report = address_preflight(["output", 5])
    assert all(v.resolution == "deferred" for v in report.verdicts)
    assert report.escrowing


class _NoSitesModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

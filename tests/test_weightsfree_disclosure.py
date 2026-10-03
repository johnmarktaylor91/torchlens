"""Evidence envelope + all-channels disclosure sweep (memo sec 5 / item 14).

ONE immutable envelope, computed once at settlement, consumed by every
surface; the banner ladder is HYPOTHESIS -> CORROBORATED -> REFUTED and no
channel may launder it. The parametrized all-channels test enumerates every
banner-bearing surface shipped on this branch so a new channel cannot
silently escape the list.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_honesty import (
    capture_honesty_facts,
    honesty_banner_lines,
)
from torchlens.options import CaptureOptions


class Toy(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _meta_capture():
    with torch.device("meta"):
        model = Toy()
    model.eval()
    return tl.trace(
        model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
    )


# ---------------------------------------------------------------------------
# The envelope itself
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_envelope_stamped_and_load_valid() -> None:
    """The settlement writer's envelope passes the C07 fail-closed validator."""

    from torchlens._io.forgery_validation import _validate_structure_evidence

    trace = _meta_capture()
    assert trace.structure_evidence is not None
    _validate_structure_evidence(trace)  # raises on any schema violation
    envelope = trace.structure_evidence
    assert envelope["substrate"] == "meta"
    assert envelope["factory_device_policy"] == "torchlens_owned"
    assert envelope["outcome"] == "complete"
    assert envelope["discharge"] == "absent"
    assert envelope["claims"]["measured_values"] == "unavailable"


def test_ordinary_capture_carries_no_envelope() -> None:
    trace = tl.trace(Toy(), torch.randn(2, 4))
    assert trace.structure_evidence is None


# ---------------------------------------------------------------------------
# The claim-status ladder on the honesty authority
# ---------------------------------------------------------------------------


def test_claim_ladder_hypothesis_then_corroborated() -> None:
    trace = _meta_capture()
    facts = capture_honesty_facts(trace)
    assert facts["claim_status"] == "hypothesis"
    assert facts["values_available"] is False
    assert facts["substrate"] == "meta"
    assert any("hypotheses" in line for line in honesty_banner_lines(trace))

    torch.manual_seed(0)
    real = Toy()
    real.eval()
    real_trace = tl.trace(real, torch.randn(2, 4))
    discharge = trace.discharge_against(real_trace)
    assert discharge.verdict.value == "corroborated"
    facts = capture_honesty_facts(trace)
    assert facts["claim_status"] == "corroborated"
    assert facts["discharge"]["comparison_vocabulary"] == "identity-v0"
    banner = "\n".join(honesty_banner_lines(trace))
    assert "CORROBORATED" in banner
    assert "no tensor payloads exist" in banner


def test_claim_ladder_refuted_is_visually_stronger() -> None:
    trace = _meta_capture()
    real = tl.trace(Toy(width=6), torch.randn(2, 4))
    discharge = trace.discharge_against(real)
    assert discharge.verdict.value == "refuted"
    banner = "\n".join(honesty_banner_lines(trace))
    assert "REFUTED" in banner
    assert banner.startswith("!!")


# ---------------------------------------------------------------------------
# All-channels sweep: every banner-bearing surface on this branch
# ---------------------------------------------------------------------------


def _channel_repr(trace, tmp_path):
    return repr(trace)


def _channel_summary(trace, tmp_path):
    return trace.summary()


def _channel_slice_summary(trace, tmp_path):
    labels = [layer.label for layer in trace.layer_list]
    return trace.between([labels[0]], [labels[-1]]).summary()


def _channel_agent_json(trace, tmp_path):
    import json as json_module

    return json_module.dumps(trace.to_agent_json())


def _channel_explain(trace, tmp_path):
    return tl.report.explain(trace)


def _channel_csv(trace, tmp_path):
    path = tl.export.csv(trace, tmp_path / "trace.csv")
    return path.read_text(encoding="utf-8")


def _channel_json(trace, tmp_path):
    path = tl.export.json(trace, tmp_path / "trace.json")
    return path.read_text(encoding="utf-8")


def _channel_svg(trace, tmp_path):
    path = tl.export.svg(trace, tmp_path / "trace.svg")
    return path.read_text(encoding="utf-8")


def _channel_html(trace, tmp_path):
    path = tl.export.html(trace, tmp_path / "trace.html")
    return path.read_text(encoding="utf-8")


_CHANNELS = {
    "repr": _channel_repr,
    "summary": _channel_summary,
    "slice_summary": _channel_slice_summary,
    "agent_json": _channel_agent_json,
    "explain": _channel_explain,
    "csv": _channel_csv,
    "json": _channel_json,
    "svg": _channel_svg,
    "html": _channel_html,
}

_MARKER_TOKENS = ("structure-only", "structure_only", "hypothes")


@pytest.mark.smoke_cells("test_every_channel_carries_the_structure_marker[slice_summary]")
@pytest.mark.parametrize("channel", sorted(_CHANNELS))
def test_every_channel_carries_the_structure_marker(channel: str, tmp_path) -> None:
    """No shipped text-bearing channel may launder the hypothesis banner."""

    trace = _meta_capture()
    text = str(_CHANNELS[channel](trace, tmp_path)).lower()
    assert any(token in text for token in _MARKER_TOKENS), (
        f"channel {channel!r} laundered the structure-only disclosure"
    )


def test_dot_render_carries_banner(tmp_path) -> None:
    """The rendered graph caption carries the banner (memo L7's named hole)."""

    trace = _meta_capture()
    trace.draw(vis_outpath=str(tmp_path / "graph"), vis_save_only=True, vis_fileformat="dot")
    candidates = list(tmp_path.glob("graph*"))
    assert candidates, "draw produced no graph artifact"
    text = "".join(path.read_text(encoding="utf-8", errors="ignore") for path in candidates).lower()
    assert any(token in text for token in _MARKER_TOKENS)


# ---------------------------------------------------------------------------
# Measurement-shaped output refuses (D15)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "export_name", ["chrome_trace", "speedscope", "flamegraph", "memory_timeline"]
)
def test_measurement_exports_refuse_typed(export_name: str, tmp_path) -> None:
    from torchlens.capture.structure_only import StructureOnlyCapabilityError

    trace = _meta_capture()
    exporter = getattr(tl.export, export_name)
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        exporter(trace, tmp_path / "out")
    assert excinfo.value.fields["code"] == "structure_only_measurements_unsupported"


def test_xarray_refuses_through_values_code(tmp_path) -> None:
    pytest.importorskip("xarray")
    from torchlens.capture.structure_only import StructureOnlyCapabilityError

    trace = _meta_capture()
    with pytest.raises(StructureOnlyCapabilityError) as excinfo:
        tl.export.xarray(trace)
    assert excinfo.value.fields["code"] == "structure_only_values_unsupported"


def test_measurement_exports_still_work_on_ordinary_captures(tmp_path) -> None:
    trace = tl.trace(Toy(), torch.randn(2, 4))
    path = tl.export.chrome_trace(trace, tmp_path / "trace.json")
    assert path.exists()

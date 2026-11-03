"""Every ``tl.debug`` finding that names an op also names the user's source line.

Covers the ``source_line`` field and column on the debug reports, the printed
``at <file>:<line>`` on their cards, the multi-line ``find_nan`` card, the
``grad_health`` capture hint and loss-scale proof, and cross-device /
cross-dtype comparison in ``compare`` and ``first_divergence``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._debug_report import display_source_line

THIS_FILE = os.path.basename(__file__)


def _line_of(fragment: str) -> int:
    """Return the 1-based line number of the first line containing ``fragment``."""

    for number, text in enumerate(Path(__file__).read_text().splitlines(), start=1):
        if fragment in text and "_line_of(" not in text:
            return number
    raise AssertionError(f"{fragment!r} not found")


class LogNet(nn.Module):
    """Linear layer followed by a log that turns negatives into NaN."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.fc(x))  # nan-site


class SigmoidMlp(nn.Module):
    """Two linear layers with sigmoid activations."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.sigmoid(self.fc1(x))  # hidden-site
        return torch.sigmoid(self.fc2(hidden))


class ReluCnn(nn.Module):
    """Convolution, ReLU and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3)
        self.fc = nn.Linear(18, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv(x))
        return self.fc(y.flatten(1))


def _nan_input() -> tuple[LogNet, torch.Tensor]:
    torch.manual_seed(0)
    return LogNet(), torch.randn(3, 4)


def _grad_trace(scale_loss: float | None = None) -> tl.Trace:
    torch.manual_seed(1)
    x = torch.randn(5, 4, requires_grad=True)
    trace = tl.trace(SigmoidMlp(), x, capture=tl.options.CaptureOptions(save_grads=True))
    out = trace[trace.output_layers[0]].out
    loss = out.sum() if scale_loss is None else out.sum() * scale_loss
    trace.backward(loss)
    return trace


def test_display_source_line_relativizes_under_cwd_only() -> None:
    """Card paths are relative under the working directory, unchanged outside it."""

    inside = os.path.join(os.getcwd(), "pkg", "model.py") + ":12"
    outside = os.path.join(os.path.dirname(os.getcwd()) or "/", "elsewhere_x", "m.py") + ":3"

    assert display_source_line(inside) == os.path.join("pkg", "model.py") + ":12"
    if os.getcwd() != os.path.dirname(os.getcwd()):
        assert display_source_line(outside) == outside
    assert display_source_line("<string>") == "<string>"


def test_find_nan_prints_a_multiline_card_with_op_and_line() -> None:
    """find_nan's str() is a card naming the op, its line, module and output."""

    model, x = _nan_input()
    report = tl.debug.find_nan(model, x)
    card = str(report)
    expected = f"{__file__}:{_line_of('# nan-site')}"

    assert report.found
    assert report.source_line == expected
    assert repr(report).startswith("FindNanReport(found=True")
    assert len(card.splitlines()) >= 3
    assert card.splitlines()[0].startswith("FindNanReport(")
    assert f"{THIS_FILE}:{_line_of('# nan-site')}" in card
    assert " at " in card
    assert "More:" in card.splitlines()[-1]


def test_nan_forensics_cards_name_the_source_line() -> None:
    """nonfinite_origin and bisect_nan print the origin op's file:line."""

    model, x = _nan_input()
    trace = tl.trace(model, x)
    site = f"{THIS_FILE}:{_line_of('# nan-site')}"

    assert site in str(tl.debug.nonfinite_origin(model, x))
    assert site in str(tl.debug.bisect_nan(trace))


def test_grad_health_hint_names_the_working_capture() -> None:
    """The empty-report hint names save_grads=True, and following it yields rows."""

    torch.manual_seed(2)
    forward_only = tl.trace(SigmoidMlp(), torch.randn(5, 4))
    report = tl.debug.grad_health(forward_only)

    assert "save_grads=True" in report.to_pandas().attrs["message"]
    assert "save_grads=True" in str(report)
    assert "backward_ready" not in str(report)
    assert len(tl.debug.grad_health(_grad_trace()).to_pandas()) > 0


def test_grad_health_proves_unscaled_plain_run() -> None:
    """A plain fp32 backward with an unscaled loss gets scale 1.0 and real verdicts."""

    report = tl.debug.grad_health(_grad_trace())
    frame = report.to_pandas()
    computed = frame[frame["verdict"] != "unavailable"]

    assert "source_line" in frame.columns
    assert set(computed["scale_provenance"]) == {"proven_unscaled"}
    assert "unknown" not in set(computed["verdict"])
    assert frame.attrs["grad_scale"] == 1.0
    assert "loss scale 1.0 (proven" in str(report)
    hidden = f"{__file__}:{_line_of('# hidden-site')}"
    assert hidden in set(frame["source_line"])
    assert f"{THIS_FILE}:" in str(report)


def test_grad_health_stays_unknown_when_loss_is_multiplied() -> None:
    """A multiply between loss and output keeps the scale unknown (AMP guard)."""

    report = tl.debug.grad_health(_grad_trace(scale_loss=0.5))
    frame = report.to_pandas()
    computed = frame[frame["verdict"] != "unavailable"]

    assert set(computed["scale_provenance"]) == {"unknown"}
    assert set(computed["verdict"]) == {"unknown"}
    assert "Mul" in frame.attrs["scale_proof"]
    assert "loss scale unknown" in str(report)


def test_grad_health_stays_unknown_under_autocast() -> None:
    """A forward under autocast keeps the scale unknown."""

    torch.manual_seed(3)
    x = torch.randn(5, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        trace = tl.trace(SigmoidMlp(), x, capture=tl.options.CaptureOptions(save_grads=True))
    trace.backward(trace[trace.output_layers[0]].out.float().sum())
    frame = tl.debug.grad_health(trace).to_pandas()

    assert "autocast" in frame.attrs["scale_proof"]
    assert set(frame[frame["verdict"] != "unavailable"]["verdict"]) == {"unknown"}


def test_grad_health_explicit_scale_is_used() -> None:
    """grad_scale= overrides the proof and is named on the card."""

    report = tl.debug.grad_health(_grad_trace(), grad_scale=1.0)

    assert "grad_scale= passed" in str(report)


def test_compare_across_dtypes_compares_in_the_wider_dtype() -> None:
    """A float32 run against a float64 run is compared, not skipped."""

    torch.manual_seed(4)
    model = SigmoidMlp()
    x = torch.randn(5, 4)
    trace_32 = tl.trace(model, x)
    trace_64 = tl.trace(_copy_double(model), x.double())
    other_64 = tl.trace(SigmoidMlp().double(), x.double())

    report = tl.debug.compare(trace_32, trace_64, rtol=1e-4, atol=1e-6)
    frame = report.to_pandas()
    aligned = frame[frame["reason"].astype(str).str.contains("compared in float64")]

    assert len(aligned) > 0
    assert frame.attrs["matched"] > 0
    assert "source_line" in frame.columns
    assert "compared" in str(report)
    assert tl.debug.first_divergence(trace_32, trace_64, rtol=1e-4, atol=1e-6) is None
    assert tl.debug.first_divergence(trace_32, other_64) is not None


def _copy_double(model: nn.Module) -> nn.Module:
    twin = SigmoidMlp()
    twin.load_state_dict(model.state_dict())
    return twin.double()


def test_first_divergence_row_carries_source_line() -> None:
    """A weight edit diverges at an op whose row names this file."""

    torch.manual_seed(5)
    model_a = SigmoidMlp()
    model_b = SigmoidMlp()
    model_b.load_state_dict(model_a.state_dict())
    with torch.no_grad():
        model_b.fc2.weight.add_(1.0)
    x = torch.randn(5, 4)

    row = tl.debug.first_divergence(tl.trace(model_a, x), tl.trace(model_b, x))

    assert row is not None
    assert THIS_FILE in str(row["source_line"])


def test_compare_shape_mismatch_is_counted_not_compared() -> None:
    """Payloads that cannot be aligned are counted on the coverage line."""

    torch.manual_seed(6)
    model = SigmoidMlp()
    report = tl.debug.compare(
        tl.trace(model, torch.randn(5, 4)), tl.trace(model, torch.randn(2, 4))
    )
    frame = report.to_pandas()

    assert "shape-mismatch" in set(frame["reason"])
    assert "not compared" in str(report)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compare_cpu_against_cuda() -> None:
    """A CPU run against a CUDA run is compared on the CPU."""

    torch.manual_seed(7)
    model = SigmoidMlp()
    x = torch.randn(5, 4)
    trace_cpu = tl.trace(model, x)
    trace_gpu = tl.trace(_copy_to_cuda(model), x.cuda())
    frame = tl.debug.compare(trace_cpu, trace_gpu).to_pandas()

    assert frame["reason"].astype(str).str.contains("compared on cpu").any()
    assert frame.attrs["matched"] > 0


def _copy_to_cuda(model: nn.Module) -> nn.Module:
    twin = SigmoidMlp()
    twin.load_state_dict(model.state_dict())
    return twin.cuda()


def test_table_reports_carry_source_line_column() -> None:
    """dead_neurons and recompute_candidates rows name the op's source line."""

    torch.manual_seed(8)
    trace = tl.trace(ReluCnn(), torch.randn(1, 1, 5, 5))

    for report in (tl.debug.dead_neurons(trace, dim=1), tl.debug.recompute_candidates(trace)):
        frame = report.to_pandas()
        assert "source_line" in frame.columns
        assert frame["source_line"].astype(str).str.contains(THIS_FILE).any()


def test_lineage_rows_name_their_columns_and_source() -> None:
    """lineage's table columns match their values and the card names lines."""

    torch.manual_seed(9)
    trace = tl.trace(SigmoidMlp(), torch.randn(5, 4))
    report = tl.debug.lineage(trace, trace.output_layers[0])
    frame = report.to_pandas()

    assert list(frame.columns) == ["label", "depth", "source_line", "shape", "dtype"]
    assert frame["source_line"].astype(str).str.contains(THIS_FILE).any()
    assert f"{THIS_FILE}:" in str(report)


def test_dtype_range_prints_its_card() -> None:
    """dtype_range's str() is the card; repr stays one line."""

    torch.manual_seed(10)
    model = SigmoidMlp().half()
    trace = tl.trace(model, torch.randn(5, 4).half() * 1e3)
    report = tl.debug.dtype_range(trace)

    assert str(report) != repr(report)
    assert "More:" in str(report).splitlines()[-1]
    assert all(
        finding.source_line is None or THIS_FILE in finding.source_line
        for finding in report.findings
    )

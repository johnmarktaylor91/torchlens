"""Phase 5a tests for taps, record spans, and report values."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.report import log_value


class MetricModel(torch.nn.Module):
    """Model that records a report value during capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ReLU and record a scalar value.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        y = torch.relu(x)
        log_value("out_sum", float(y.sum()))
        return y


def test_tap_records_without_modifying_output() -> None:
    """Tap hooks should record values and leave model output unchanged."""

    model = torch.nn.ReLU()
    x = torch.tensor([[-1.0, 2.0]])
    tap = tl.tap(tl.func("relu"))

    hooked = tl.trace(model, x, intervention_ready=True, hooks=tap)
    plain = tl.trace(model, x)

    assert tap.values()
    assert torch.equal(tap.values()[0], torch.relu(x))
    assert torch.equal(
        hooked[hooked.output_layers[0]].out,
        plain[plain.output_layers[0]].out,
    )


def test_record_span_and_log_value_metadata() -> None:
    """Record spans and logged values should land on the captured Trace."""

    with tl.record_span("phase_name"):
        log = tl.trace(MetricModel(), torch.tensor([[-1.0, 2.0]]))

    assert log.observer_spans
    assert log.observer_spans[0]["name"] == "phase_name"
    assert log.observer_spans[0]["end"] is not None
    assert log.annotations["logged_values"]["out_sum"] == 2.0
    assert not hasattr(log, "report_values")

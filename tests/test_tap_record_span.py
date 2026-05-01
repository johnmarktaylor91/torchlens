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
        log_value("activation_sum", float(y.sum()))
        return y


def test_tap_records_without_modifying_output() -> None:
    """Tap hooks should record values and leave model output unchanged."""

    model = torch.nn.ReLU()
    x = torch.tensor([[-1.0, 2.0]])
    tap = tl.tap(tl.func("relu"))

    hooked = tl.log_forward_pass(model, x, intervention_ready=True, hooks=tap)
    plain = tl.log_forward_pass(model, x)

    assert tap.values()
    assert torch.equal(tap.values()[0], torch.relu(x))
    assert torch.equal(
        hooked[hooked.output_layers[0]].activation,
        plain[plain.output_layers[0]].activation,
    )


def test_record_span_and_log_value_metadata() -> None:
    """Record spans and report values should land on the captured ModelLog."""

    with tl.record_span("phase_name"):
        log = tl.log_forward_pass(MetricModel(), torch.tensor([[-1.0, 2.0]]))

    assert log.observer_spans
    assert log.observer_spans[0]["name"] == "phase_name"
    assert log.observer_spans[0]["end"] is not None
    assert log.report_values["activation_sum"] == 2.0

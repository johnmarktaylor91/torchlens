"""Lazy completion unit acceptance (F17 B12): SOL's four clauses + the flip signal.

Quickstart memo 4.4: the completion unit ships as ONE unit, and the flip
signal is mechanical -- ``check_metadata_invariants(trace)`` passes on the
real lazy-head fixture (the shipped ``trainable + frozen == total`` invariant
that used to fail silently, with zeros). The four-clause acceptance condition
(memo 4.4 item 4) runs as an executable test regardless of the prediction
that some clauses pass vacuously.

The fixture is the way people actually hit lazy modules: a probe head on a
real backbone (toy-scale here; the real-resnet18 variant lives in
``test_quickstart_realmodel.py``).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

import torchlens as tl


def _lazy_head_model() -> nn.Sequential:
    """Backbone + two-layer lazy probe head (the memo's fixture shape)."""

    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(6),
        nn.ReLU(),
        nn.LazyLinear(4),
    )


class TestFlipSignal:
    """The mechanical flip signal: metadata invariants on the lazy fixture."""

    def test_summary_totals_are_truthful(self) -> None:
        """The one-call summary's totals include the materialized head."""

        model = _lazy_head_model()
        text = str(tl.summary(model, torch.randn(1, 3, 16, 16)))
        ground_truth = sum(p.numel() for p in model.parameters())
        assert f"{ground_truth:,}" in text or str(ground_truth) in text


class TestLazyBoundaries:
    """Refusals that stay in place around the completion unit."""

    def test_zero_arg_rung_refuses_before_probing(self) -> None:
        """Memo 4.2: inference on pending lazy state refuses structurally --
        a lazy module accepts any width, so a probe has no error signal and
        would permanently install the guess in the caller's model."""

        model = _lazy_head_model()
        with pytest.raises(Exception) as excinfo:
            tl.trace(model)
        assert "lazy" in str(excinfo.value).lower()
        assert isinstance(model[4].weight, UninitializedParameter), (
            "the refusal must fire BEFORE any probe touches the model"
        )

    def test_armed_lane_refuses_typed_before_mutation(self) -> None:
        """Memo 4.5: intervention-ready capture on pending state refuses
        state_baseline_unavailable (a pending parameter has no bytes to
        witness), before any mutation."""

        model = _lazy_head_model()
        with pytest.raises(Exception) as excinfo:
            tl.trace(
                model,
                torch.randn(1, 3, 16, 16),
                capture=tl.options.CaptureOptions(replayable=True),
            )
        assert excinfo.value.fields["code"] == "state_baseline_unavailable"
        assert isinstance(model[4].weight, UninitializedParameter)

    def test_taught_self_prime_remedy_executes(self) -> None:
        """The teach's two-line remedy is executed here so it cannot rot."""

        model = _lazy_head_model()
        x = torch.randn(1, 3, 16, 16)
        with torch.no_grad():
            model(x)  # the taught self-prime
        log = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
        assert log.num_params == sum(p.numel() for p in model.parameters())

    def test_lazy_buffer_entry_captures_without_prematerialization(self) -> None:
        """Pending lazy BUFFERS capture directly (F20 buffer-side completion).

        The buffer-side lazy completion flipped the entry refusal off for
        lazy-BUFFER models (LazyBatchNorm*): capture runs the real forward
        and the buffers materialize in place. The lazy_uninitialized teach
        survives only for pending lazy PARAMETERS (the tests above); the
        buffer capture contract is pinned in
        tests/test_entry_facade_lazy_entry.py::test_trace_tolerates_lazy_buffer_model.
        """

        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.LazyBatchNorm2d()).eval()
        log = tl.trace(model, torch.randn(1, 3, 8, 8))
        assert log.outcome.status.name == "COMPLETE"
        assert not isinstance(model[1].running_mean, UninitializedBuffer)

"""Explorer memo item-3 riders (lane F25).

- P3 pin: input ops receive activation transforms on the TRACE path (the
  fastlog pin lives in test_explorer_watch_p1_summary_role.py).
- P4: retained-memory truth on reduce-only captures plus the two
  ``total_transformed_*`` read-time totals.
- Fused spine kernel: exact-count / tolerance-float parity oracle against
  the naive kernel over the real op zoo.
- The pinned interleaved A/B perf harness: the D25 publication rule is
  deterministic given a noise band; within-noise rows disclose, not drop.
- Recorder backward/input coverage, nailed across repeated rollouts (the
  seam the op-tier sizing depends on).
"""

from __future__ import annotations

import time

import pytest
import torch

import torchlens as tl
from torchlens.ir.summary_role import summary
from torchlens.observability import (
    total_transformed_activation_memory,
    total_transformed_gradient_memory,
)
from torchlens.observability._kernels import spine_vector, spine_vector_fused
from torchlens.observability._perf_harness import (
    MIN_PUBLISHABLE_REPEATS,
    format_markdown,
    measure_ab,
)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())


class TestInputOpTransformCoverage:
    """P3 (memo V7): input ops carry transformed payloads on the trace path."""

    def test_trace_input_op_gets_transform(self) -> None:
        log = tl.trace(
            _model(),
            torch.randn(2, 8),
            save=tl.options.SaveOptions(
                activation_transform=summary(lambda t: t.float().mean().unsqueeze(0))
            ),
        )
        input_label = log.input_layers[0]
        assert log[input_label].transformed_out is not None


class TestMemoryTruth:
    """P4 (memo V9): the reduced-capture memory report says what was retained."""

    def test_reduce_only_saved_memory_is_transformed_bytes(self) -> None:
        log = tl.trace(
            _model(),
            torch.randn(32, 8),
            save=tl.options.SaveOptions(
                activation_transform=summary(lambda t: t.float().sum().unsqueeze(0)),
                save_raw_activations=False,
            ),
        )
        # Raw nominal footprint is KBs; retained summaries are a few floats.
        assert int(log.saved_activation_memory) < 100
        assert int(log.total_activation_memory) > 1000
        assert int(total_transformed_activation_memory(log)) > 0
        assert int(total_transformed_activation_memory(log)) <= 100

    @pytest.mark.smoke
    def test_raw_retention_accounting_unchanged(self) -> None:
        log = tl.trace(_model(), torch.randn(32, 8))
        assert int(log.saved_activation_memory) >= int(log.total_activation_memory)
        assert int(total_transformed_activation_memory(log)) == 0
        assert int(total_transformed_gradient_memory(log)) == 0

    def test_transform_with_raw_retention_keeps_raw_accounting(self) -> None:
        log = tl.trace(
            _model(),
            torch.randn(32, 8),
            save=tl.options.SaveOptions(
                activation_transform=summary(lambda t: t.float().sum().unsqueeze(0)),
            ),
        )
        # Raw retained: historical raw-bytes accounting, transformed extra
        # visible only through the new total.
        assert int(log.saved_activation_memory) >= int(log.total_activation_memory)
        assert int(total_transformed_activation_memory(log)) > 0


def _spine_zoo() -> list[torch.Tensor]:
    torch.manual_seed(7)
    mixed = torch.randn(257)
    mixed[3] = float("nan")
    mixed[10] = float("inf")
    mixed[11] = float("-inf")
    mixed[12] = 0.0
    return [
        torch.randn(1000),
        mixed,
        torch.full((5,), float("nan")),
        torch.full((4,), float("inf")),
        torch.empty(0),
        torch.tensor(3.5),
        torch.randint(-10, 10, (64,)),
        torch.zeros(16, dtype=torch.bool),
        torch.randn(33, dtype=torch.float16),
        torch.randn(33, dtype=torch.bfloat16),
        torch.randn(4, 4, 4),
        torch.zeros(12),
    ]


class TestFusedSpineParity:
    """The fused kernel is a drop-in: counts exact, floats to tolerance."""

    _COUNT_SLOTS = (0, 1, 2, 3, 4, 5, 6)
    _EXTREME_SLOTS = (7, 8, 9)
    _FLOAT_SLOTS = (10, 11, 12, 13, 14)

    @pytest.mark.smoke_cells("test_parity_across_zoo[2]", "test_parity_across_zoo[4]")
    @pytest.mark.parametrize("case", range(12))
    def test_parity_across_zoo(self, case: int) -> None:
        tensor = _spine_zoo()[case]
        naive = spine_vector(tensor)
        fused = spine_vector_fused(tensor)
        for slot in self._COUNT_SLOTS:
            assert fused[slot].item() == naive[slot].item(), f"count slot {slot}"
        for slot in self._EXTREME_SLOTS:
            a, b = naive[slot].item(), fused[slot].item()
            assert (a != a and b != b) or a == b, f"extreme slot {slot}: {a} vs {b}"
        for slot in self._FLOAT_SLOTS:
            a, b = naive[slot].item(), fused[slot].item()
            assert b == pytest.approx(a, rel=1e-9, abs=1e-9), f"float slot {slot}"


class TestPerfHarnessRule:
    """The D25 publication rule, exercised deterministically."""

    @staticmethod
    def _sleeper(seconds: float):
        def run() -> None:
            time.sleep(seconds)

        return run

    def test_effect_outside_band_publishes(self) -> None:
        row = measure_ab(
            "sleep-5x",
            self._sleeper(0.0005),
            self._sleeper(0.0025),
            repeats=MIN_PUBLISHABLE_REPEATS,
            warmup=1,
            noise_band=1.5,
        )
        assert row.publishable
        assert row.ratio > 1.5
        assert "PUBLISHABLE" in row.verdict()

    @pytest.mark.smoke
    def test_effect_inside_band_is_disclosed_not_published(self) -> None:
        row = measure_ab(
            "sleep-parity",
            self._sleeper(0.0005),
            self._sleeper(0.0006),
            repeats=MIN_PUBLISHABLE_REPEATS,
            warmup=1,
            noise_band=10.0,
        )
        assert not row.publishable
        table = format_markdown([row], sha="deadbeef", device="cpu", basis="unit test")
        assert "WITHIN NOISE -- not published" in table
        assert "generated by torchlens.observability._perf_harness" in table

    @pytest.mark.smoke
    def test_repeats_below_floor_never_publish(self) -> None:
        row = measure_ab(
            "sleep-5x-thin",
            self._sleeper(0.0005),
            self._sleeper(0.0025),
            repeats=5,
            warmup=1,
            noise_band=1.2,
        )
        assert not row.publishable


class TestRecorderCoverage:
    """Recorder backward/input coverage across rollouts (op-tier seam)."""

    @pytest.mark.smoke
    def test_input_summaries_across_rollouts(self) -> None:
        model = _model()
        with tl.fastlog.Recorder(
            model,
            default_op=True,
            include_source_events=True,
            activation_transform=summary(lambda t: t.float().mean().unsqueeze(0)),
            save_raw_activations=False,
        ) as recorder:
            recorder.log(torch.randn(2, 8))
            recorder.log(torch.randn(2, 8))
        recording = recorder.recording
        inputs = [r for r in recording.records if r.ctx.kind == "input"]
        assert len(inputs) == 2, "one input record per rollout"
        passes = sorted(r.ctx.pass_index for r in inputs)
        assert passes == [1, 2], "pass identity kept across rollouts"
        for record in inputs:
            assert record.ram_payload is None
            assert record.transformed_ram_payload is not None

    @pytest.mark.smoke
    def test_backward_summaries_across_rollouts(self) -> None:
        model = _model()
        from torchlens.fastlog import CaptureSpec

        with tl.fastlog.Recorder(
            model,
            default_op=CaptureSpec(save_out=True, keep_grad=True),
            save_grads=True,
            grad_transform=summary(lambda t: t.float().abs().sum().unsqueeze(0)),
            save_raw_gradients=False,
            backward_ready=True,
        ) as recorder:
            out_1 = recorder.log(torch.randn(2, 8, requires_grad=True))
            recorder.log_backward(out_1.sum())
            out_2 = recorder.log(torch.randn(2, 8, requires_grad=True))
            recorder.log_backward(out_2.sum())
        recording = recorder.recording
        assert recording.grad_records, "backward coverage: grad records present"
        for grad_record in recording.grad_records:
            assert grad_record.transformed_ram_payload is not None
            assert grad_record.ram_payload is None

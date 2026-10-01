"""Module-tier collector tests (explorer item 6; D15-D17, D21).

Observation is INERT: a collected run's losses, gradients, updates, and
model state EQUAL a seeded unobserved control, including train-mode
BatchNorm running stats and in-place ReLU models.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens.observability import (
    EventStream,
    HistoryArtifactError,
    HistoryCollector,
    StepTruth,
    WatchLifecycleError,
    WatchPlanError,
    WatchSettings,
)

pytestmark = pytest.mark.smoke


def _bn_relu_model() -> nn.Sequential:
    torch.manual_seed(11)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 5),
    )


def _mlp() -> nn.Sequential:
    torch.manual_seed(12)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


class TestDiscoveryAndPlan:
    """Discovery forward, zero-match refusal, and the D21 plan table."""

    def test_discovery_returns_the_users_result(self) -> None:
        model = _mlp()
        x = torch.randn(4, 8)
        collector = HistoryCollector(model)
        expected = model(x)
        output = collector.discover(x)
        assert torch.equal(output, expected)

    def test_zero_match_refuses_at_plan_time(self) -> None:
        collector = HistoryCollector(_mlp(), sites=("nonexistent.path",))
        with pytest.raises(WatchPlanError) as excinfo:
            collector.discover(torch.randn(2, 8))
        assert excinfo.value.fields["code"] == "watch_plan_empty"

    def test_plan_table_prints_elements_and_bytes(self) -> None:
        model = _mlp()
        collector = HistoryCollector(
            model, streams=("activation", "param", "param_delta"), cadence=2
        )
        collector.discover(torch.randn(4, 8))
        table = collector.plan.format_table()
        assert "elements/step" in table
        assert "bytes/step" in table
        assert "sampled update channel" not in table  # all params < 4096 here
        streams = {row.stream for row in collector.plan.rows}
        assert streams == {"activation", "param", "param_delta"}
        assert all(row.cadence == 2 for row in collector.plan.rows)

    def test_giant_param_is_marked_sampled(self) -> None:
        model = nn.Sequential(nn.Embedding(5000, 16))
        collector = HistoryCollector(model, streams=("param_delta",))
        collector.discover(torch.randint(0, 5000, (2, 3)))
        row = next(r for r in collector.plan.rows if r.stream == "param_delta")
        assert row.elements_per_step == 4096
        assert "sampled" in row.note

    def test_plan_before_discover_refuses(self) -> None:
        with pytest.raises(WatchLifecycleError) as excinfo:
            _ = HistoryCollector(_mlp()).plan
        assert excinfo.value.fields["code"] == "watch_lifecycle_invalid"

    def test_unknown_stream_refuses(self) -> None:
        with pytest.raises(WatchPlanError) as excinfo:
            HistoryCollector(_mlp(), streams=("activation_grad",))
        assert excinfo.value.fields["code"] == "watch_plan_invalid"
        assert "F24" in str(excinfo.value)


class TestStepSpellings:
    """Both step spellings ship; provenance is recorded (D17)."""

    def test_explicit_spelling_is_authoritative(self) -> None:
        model = _mlp()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = HistoryCollector(model, streams=("activation", "param_grad"))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        with collector.step(10, truth=StepTruth(scale=2.0, unscaled="yes", clipped="no")):
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        collector.detach()
        block = collector.ring.blocks[0]
        assert block.block.provenance == "explicit"
        assert block.block.global_step == 10
        assert block.block.scale == 2.0
        assert block.block.unscaled == "yes"

    def test_implicit_spelling_stamps_unknown_never_guesses(self) -> None:
        model = _mlp()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = HistoryCollector(model, streams=("activation", "param"))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        for _ in range(2):
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        collector.detach()
        blocks = collector.ring.blocks
        assert len(blocks) == 2
        assert all(b.block.provenance == "implicit" for b in blocks)
        assert all(b.block.unscaled == "unknown" for b in blocks)
        assert all(b.block.clipped == "unknown" for b in blocks)
        assert [b.block.global_step for b in blocks] == [0, 1]

    def test_duplicate_explicit_step_refuses_without_new_segment(self) -> None:
        model = _mlp()
        collector = HistoryCollector(model)
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach()
        with collector.step(5):
            model(x)
        with pytest.raises(Exception) as excinfo, collector.step(5):
            model(x)
        assert excinfo.value.fields["code"] == "history_step_regression"  # type: ignore[attr-defined]
        # A declared new segment (resume) starts a new axis span.
        old_segment = collector.segment_id
        with collector.step(0, new_segment=True):
            model(x)
        assert collector.segment_id != old_segment
        collector.detach()

    def test_nested_step_refuses(self) -> None:
        model = _mlp()
        collector = HistoryCollector(model)
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach()
        with collector.step(0):
            with pytest.raises(WatchLifecycleError) as excinfo, collector.step(1):
                pass
            assert excinfo.value.fields["code"] == "watch_step_conflict"
        collector.detach()

    def test_micro_batch_accumulation_folds_into_one_block(self) -> None:
        model = _mlp()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = HistoryCollector(model, streams=("activation",))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        with collector.step(0):
            for _ in range(3):  # 3 microsteps, one optimizer step
                model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        collector.detach()
        block = collector.ring.blocks[0]
        activation = next(o for o in block.observations if o.site_id == "module:0")
        assert activation.spine is not None
        # One row per optimizer step; micro-batch populations fold together.
        assert activation.spine.count_total == 3 * 4 * 16


class TestOptimizerTruth:
    """AMP/accumulation/optimizer truth: skips record NO fake update."""

    def test_skipped_step_records_skipped_and_no_update(self) -> None:
        model = _mlp()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        collector = HistoryCollector(model, streams=("activation", "param_delta"))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        with collector.step(0, truth=StepTruth(applied=False)):
            model(x).sum().backward()
            # The scaler found infs and never called optimizer.step().
            optimizer.zero_grad()
        collector.detach()
        block = collector.ring.blocks[0]
        assert block.block.optimizer_status == "skipped"
        deltas = [o for o in block.observations if o.stream == "param_delta"]
        assert deltas == []  # NO fake update, ever

    def test_update_channel_exact_for_small_params(self) -> None:
        model = _mlp()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
        collector = HistoryCollector(model, streams=("param_delta",))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        with collector.step(0):
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        collector.detach()
        block = collector.ring.blocks[0]
        for obs in block.observations:
            assert obs.stream == "param_delta"
            assert not obs.estimated  # every param here is under 4096 elements
            name = obs.site_id.removeprefix("param:")
            true_delta = (dict(model.named_parameters())[name].detach() - before[name]).to(
                torch.float64
            )
            assert obs.spine is not None
            assert obs.spine.sum_squares == pytest.approx(float((true_delta**2).sum()), rel=1e-5)

    def test_update_channel_estimated_for_giant_params(self) -> None:
        model = nn.Sequential(nn.Embedding(5000, 16))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        collector = HistoryCollector(model, streams=("param_delta",))
        x = torch.randint(0, 5000, (2, 3))
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        with collector.step(0):
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        collector.detach()
        obs = collector.ring.blocks[0].observations[0]
        assert obs.estimated
        assert obs.sample_size is not None and obs.sample_size <= 4096

    def test_param_streams_require_optimizer(self) -> None:
        model = _mlp()
        collector = HistoryCollector(model, streams=("param",))
        collector.discover(torch.randn(2, 8))
        with pytest.raises(WatchPlanError) as excinfo:
            collector.attach()
        assert excinfo.value.fields["code"] == "watch_plan_invalid"


class TestObservationInertness:
    """Observed and unobserved runs are bit-identical (explorer row 5)."""

    @staticmethod
    def _train(model: nn.Module, observed: bool) -> tuple[list[float], dict[str, torch.Tensor]]:
        torch.manual_seed(99)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        collector = None
        # The discovery forward is a REAL user forward (it returns the user's
        # actual result and has the user's own side effects, e.g. train-mode
        # BatchNorm stat updates) -- so the control runs the same forward.
        torch.manual_seed(2024)
        discovery_input = torch.randn(2, 3, 4, 4)
        if observed:
            collector = HistoryCollector(
                model, streams=("activation", "param", "param_grad", "param_delta")
            )
            collector.discover(discovery_input)
            collector.attach(optimizer=optimizer)
        else:
            model(discovery_input)
        losses = []
        torch.manual_seed(7)
        for _step in range(3):
            x = torch.randn(2, 3, 4, 4)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss))
        if collector is not None:
            collector.detach()
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        return losses, buffers

    def test_bn_running_stats_and_losses_equal_unobserved_control(self) -> None:
        observed_losses, observed_buffers = self._train(_bn_relu_model(), observed=True)
        control_losses, control_buffers = self._train(_bn_relu_model(), observed=False)
        assert observed_losses == control_losses
        for name, buffer in control_buffers.items():
            assert torch.equal(observed_buffers[name], buffer), name

    def test_inplace_relu_output_observed_safely(self) -> None:
        model = _bn_relu_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        collector = HistoryCollector(model, streams=("activation",))
        x = torch.randn(2, 3, 4, 4)
        collector.discover(x)
        collector.attach(optimizer=optimizer)
        with collector.step(0):
            model(x).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        collector.detach()
        relu_obs = next(o for o in collector.ring.blocks[0].observations if o.site_id == "module:2")
        assert relu_obs.spine is not None
        # ReLU output has no negatives; the reduction happened at module
        # return, before any later in-place mutation could rewrite it.
        assert relu_obs.spine.count_negative == 0


class TestCadence:
    """Per-stream cadence gates observation scheduling (D21)."""

    def test_cadence_skips_unscheduled_steps(self) -> None:
        model = _mlp()
        collector = HistoryCollector(model, streams=("activation",), cadence=2)
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach()
        for step in range(4):
            with collector.step(step):
                model(x)
        collector.detach()
        blocks = collector.ring.blocks
        observed_steps = [b.block.global_step for b in blocks if b.observations]
        assert observed_steps == [0, 2]

    def test_sketch_cadence_is_separate(self) -> None:
        model = _mlp()
        collector = HistoryCollector(
            model,
            streams=("activation",),
            cadence=1,
            settings=WatchSettings(sketch_cadence=2),
        )
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach()
        for step in range(3):
            with collector.step(step):
                model(x)
        collector.detach()
        blocks = collector.ring.blocks
        sketched = [
            b.block.global_step for b in blocks if any(o.sketch is not None for o in b.observations)
        ]
        assert sketched == [0, 2]


class TestChassisBinding:
    """Committed blocks publish onto the observer event stream."""

    def test_commit_events_published(self) -> None:
        stream = EventStream()
        model = _mlp()
        collector = HistoryCollector(model, settings=WatchSettings(event_stream=stream))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach()
        with collector.step(0):
            model(x)
        collector.detach()
        events = stream.snapshot()
        assert len(events) == 1
        assert events[0].key == "history.step_committed"
        assert events[0].axis_provenance == "explicit"
        assert events[0].global_step == 0

    def test_ring_refusal_fires_before_the_block(self) -> None:
        model = _mlp()
        collector = HistoryCollector(model, settings=WatchSettings(ram_capacity=2))
        x = torch.randn(4, 8)
        collector.discover(x)
        collector.attach()
        with collector.step(0):
            model(x)
        with collector.step(1):
            model(x)
        with pytest.raises(HistoryArtifactError) as excinfo, collector.step(2):
            pytest.fail("the block body must never start")
        assert excinfo.value.fields["code"] == "history_ram_budget_exceeded"
        collector.detach()

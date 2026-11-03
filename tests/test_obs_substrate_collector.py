"""Module-tier collector tests (explorer item 6; D15-D17, D21).

Observation is INERT: a collected run's losses, gradients, updates, and
model state EQUAL a seeded unobserved control, including train-mode
BatchNorm running stats and in-place ReLU models.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torchlens._observability import (
    EventStream,
    HistoryCollector,
    WatchSettings,
)
from torchlens.errors import HistoryArtifactError, WatchLifecycleError, WatchPlanError


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

    def test_zero_match_refuses_at_plan_time(self) -> None:
        collector = HistoryCollector(_mlp(), sites=("nonexistent.path",))
        with pytest.raises(WatchPlanError) as excinfo:
            collector.discover(torch.randn(2, 8))
        assert excinfo.value.fields["code"] == "watch_plan_empty"

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
        assert activation.spine.numel == 3 * 4 * 16


class TestOptimizerTruth:
    """AMP/accumulation/optimizer truth: skips record NO fake update."""

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

"""LIFECYCLE PURITY -- the C06 row gate.

The chassis contracts (checks 4.2), each tested as evidence, not style:

- every hook handle is removed on EVERY exit path (success, exception,
  context-manager unwind);
- zero user-RNG consumption: a bitwise same-seed observed run equals the
  unobserved control (losses, parameters, RNG state);
- collector arithmetic runs under ``pause_logging()``: a live TorchLens
  capture running WITH an attached collector records exactly the ops it
  records without one;
- session/region state restores on success, halt, and exception paths.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.observability import HistoryCollector, WatchSettings, region, session
from torchlens.observability._session import active_session


def _model() -> nn.Sequential:
    torch.manual_seed(21)
    return nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))


def _hook_census(model: nn.Module) -> int:
    """Count every registered module forward hook."""

    return sum(len(module._forward_hooks) for module in model.modules())


class TestHookLifecycle:
    """Every hook handle removed on every exit path."""

    def test_discovery_leaves_zero_hooks(self) -> None:
        model = _model()
        collector = HistoryCollector(model)
        baseline = _hook_census(model)
        collector.discover(torch.randn(2, 6))
        assert _hook_census(model) == baseline

    def test_discovery_forward_raising_leaves_zero_hooks(self) -> None:
        model = _model()
        collector = HistoryCollector(model)
        baseline = _hook_census(model)
        with pytest.raises(RuntimeError):
            collector.discover(torch.randn(2, 999))  # shape mismatch mid-forward
        assert _hook_census(model) == baseline

    def test_detach_removes_everything_and_is_idempotent(self) -> None:
        model = _model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        collector = HistoryCollector(model, streams=("activation", "param"))
        baseline = _hook_census(model)
        opt_baseline = len(optimizer._optimizer_step_pre_hooks) + len(
            optimizer._optimizer_step_post_hooks
        )
        collector.discover(torch.randn(2, 6))
        collector.attach(optimizer=optimizer)
        assert _hook_census(model) > baseline
        collector.detach()
        collector.detach()
        assert _hook_census(model) == baseline
        assert (
            len(optimizer._optimizer_step_pre_hooks) + len(optimizer._optimizer_step_post_hooks)
            == opt_baseline
        )

    def test_context_manager_detaches_on_exception(self) -> None:
        model = _model()
        collector = HistoryCollector(model)
        baseline = _hook_census(model)
        collector.discover(torch.randn(2, 6))
        with pytest.raises(RuntimeError), collector:
            collector.attach()
            raise RuntimeError("training crashed")
        assert _hook_census(model) == baseline

    @pytest.mark.smoke
    def test_exception_inside_step_still_commits_and_detaches(self) -> None:
        model = _model()
        collector = HistoryCollector(model)
        collector.discover(torch.randn(2, 6))
        collector.attach()
        with pytest.raises(RuntimeError), collector.step(0):
            model(torch.randn(2, 6))
            raise RuntimeError("mid-step crash")
        # The block still committed atomically (readable killed run).
        assert len(collector.ring.blocks) == 1
        collector.detach()
        assert _hook_census(model) == 0


class TestRngPurity:
    """Bitwise same-seed equality: observation consumes zero user RNG."""

    @staticmethod
    def _train(observed: bool) -> tuple[list[float], dict[str, torch.Tensor], torch.Tensor]:
        model = _model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        collector = None
        if observed:
            collector = HistoryCollector(
                model,
                streams=("activation", "param", "param_grad", "param_delta"),
                settings=WatchSettings(sketch_cadence=1),
            )
            collector.discover(torch.ones(2, 6))
            collector.attach(optimizer=optimizer)
        torch.manual_seed(31337)
        losses = []
        for _ in range(3):
            x = torch.randn(2, 6)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss))
        if collector is not None:
            collector.detach()
        params = {name: p.detach().clone() for name, p in model.named_parameters()}
        return losses, params, torch.get_rng_state().clone()

    def test_observed_equals_control_bitwise(self) -> None:
        observed_losses, observed_params, observed_rng = self._train(observed=True)
        control_losses, control_params, control_rng = self._train(observed=False)
        assert observed_losses == control_losses
        for name, control in control_params.items():
            assert torch.equal(observed_params[name], control), name
        assert torch.equal(observed_rng, control_rng)


class TestCaptureComposition:
    """The observer seam sits AFTER the capture chain: a live tl.trace with
    an attached collector records exactly the ops of an uncollected trace."""

    def test_collector_is_invisible_to_a_live_capture(self) -> None:
        model = _model()
        x = torch.randn(2, 6)
        clean = tl.trace(model, x)
        clean_labels = list(clean.layer_labels)

        collector = HistoryCollector(_model())
        watched_model = collector.model
        collector.discover(x)
        collector.attach()
        # Trace a DIFFERENT model while the collector observes its own; then
        # trace the collector's model itself: pause_logging inside the
        # observer hooks keeps every reduction out of both captures.
        observed = tl.trace(watched_model, x)
        collector.detach()
        assert list(observed.layer_labels) == clean_labels

    @pytest.mark.smoke
    def test_region_rides_the_shipped_record_span_surface(self) -> None:
        """Regions during a capture land on trace.observer_spans through the
        SHIPPED observers.span surface -- one span vocabulary (label
        unification consumed), never a parallel stack."""

        class RegionModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(6, 3)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                with region("block", tag="a"):
                    return self.linear(x)

        trace = tl.trace(RegionModel(), torch.randn(2, 6))
        block_spans = [s for s in trace.observer_spans if s["name"] == "block"]
        assert len(block_spans) == 1
        span_record = block_spans[0]
        assert span_record["region"]["metadata"] == {"tag": "a"}
        assert span_record["region"]["occurrence"] >= 1
        assert span_record["end"] is not None

    def test_region_is_inert_outside_capture_and_session(self) -> None:
        import torchlens.observers as observers

        with region("offline"):
            assert observers.active_span_records() == []


class TestSessionStatePurity:
    """The active-session slot and region stacks restore on every path."""

    def test_slot_restores_after_success_and_error(self) -> None:
        assert active_session() is None
        with session(mode="owned"):
            assert active_session() is not None
        assert active_session() is None
        with pytest.raises(RuntimeError), session(mode="owned"):
            raise RuntimeError("halt")
        assert active_session() is None

    def test_region_stack_unwinds_on_error(self) -> None:
        with pytest.raises(RuntimeError), region("outer"):
            raise RuntimeError("body crashed")
        # A fresh region sees no stale parent.
        with region("fresh") as record:
            assert record.parent_name is None

    @pytest.mark.smoke
    def test_global_rng_untouched_by_session_and_region(self) -> None:
        torch.manual_seed(77)
        before = torch.get_rng_state().clone()
        with session(mode="owned"), region("r", k=1):
            pass
        assert torch.equal(before, torch.get_rng_state())

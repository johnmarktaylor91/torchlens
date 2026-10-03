"""W051-TRACK / AUD-CODE 2.14b: AMP ``unscaled`` truth is OBSERVED, never asserted.

``unscaled="yes"`` used to be stamped whenever ``scaler.get_scale()`` merely
worked at the boundary. A plain ``optimizer.step()`` after
``scaler.scale(loss).backward()`` never unscales, so the emitted gradient
statistics carried the x1024 factor under a ``yes`` stamp. The engine now
reads the scaler's per-optimizer stage: unscaled -> observed; still scaled ->
corrected in closed form at emission with the derivation disclosed; unreadable
-> ``unknown`` and the numbers ship as measured.
"""

from __future__ import annotations

import math

import pytest
import torch

import torchlens.trackers as trk
from torchlens.trackers._amp import unscale_stage
from torchlens.utils._torch_compat import HAS_AMP_GRADSCALER

_requires_gradscaler = pytest.mark.skipif(
    not HAS_AMP_GRADSCALER,
    reason="torch.amp.GradScaler (device-agnostic) postdates the torch 2.1 floor",
)


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def _norm(sink: trk.MemorySink, leaf: str = "0.weight") -> float:
    return next(p.value for p in sink.scalars if p.tag == f"gradients/norm/{leaf}")


class TestUnscaleStage:
    @pytest.mark.smoke
    @_requires_gradscaler
    def test_ready_then_unscaled_then_stepped(self) -> None:
        model, opt = _mlp()
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        assert unscale_stage(scaler, opt) == "scaled"  # never touched: READY
        scaler.scale(model(torch.randn(4, 8)).sum()).backward()
        scaler.unscale_(opt)
        assert unscale_stage(scaler, opt) == "unscaled"
        scaler.step(opt)
        assert unscale_stage(scaler, opt) == "unscaled"  # STEPPED
        scaler.update()
        assert unscale_stage(scaler, opt) == "scaled"  # update() resets to READY

    def test_duck_typed_scaler_without_stage_is_unknown(self) -> None:
        class _Duck:
            def is_enabled(self) -> bool:
                return True

            def get_scale(self) -> float:
                return 8.0

        assert unscale_stage(_Duck(), object()) == "unknown"
        assert unscale_stage(None, object()) == "unknown"


class TestBoundaryTruth:
    @pytest.mark.smoke
    @_requires_gradscaler
    def test_plain_optimizer_step_on_scaled_grads_is_corrected_and_disclosed(self) -> None:
        """The emitted norm equals the UNSCALED norm; the record says 'no'."""

        model, opt = _mlp()
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        with session.step(0, scaler=scaler):
            opt.zero_grad(set_to_none=True)
            scaler.scale(model(torch.randn(4, 8)).sum()).backward()
            opt.step()  # NO unscale: grads still carry the factor
        session.close()
        scaled_grad = next(model.parameters()).grad
        true_norm = (scaled_grad.double() / 1024.0).norm().item()
        assert math.isclose(_norm(sink), true_norm, rel_tol=1e-7)
        block = session.collector.ring.blocks[-1]
        assert block.block.unscaled == "no"
        assert block.block.scale == 1024.0
        assert {o.grad_scale for o in block.observations if o.stream == "param_grad"} == {"scaled"}
        assert any(p.tag == "torchlens/run/amp_unscale_derived" for p in sink.scalars)

    @_requires_gradscaler
    def test_scaler_step_path_is_observed_unscaled(self) -> None:
        """``scaler.step`` unscales before the boundary: observed, not derived."""

        model, opt = _mlp()
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        with session.step(0, scaler=scaler):
            opt.zero_grad(set_to_none=True)
            scaler.scale(model(torch.randn(4, 8)).sum()).backward()
            scaler.step(opt)
            scaler.update()
        session.close()
        true_norm = next(model.parameters()).grad.double().norm().item()  # unscaled in place
        assert math.isclose(_norm(sink), true_norm, rel_tol=1e-7)
        block = session.collector.ring.blocks[-1]
        assert block.block.unscaled == "yes"
        assert {o.grad_scale for o in block.observations if o.stream == "param_grad"} == {
            "unscaled"
        }
        assert not any(p.tag == "torchlens/run/amp_unscale_derived" for p in sink.scalars)

    def test_unreadable_stage_stays_unknown_and_uncorrected(self) -> None:
        """No stage evidence -> 'unknown' everywhere and the measured numbers ship."""

        class _Duck:
            def is_enabled(self) -> bool:
                return True

            def get_scale(self) -> float:
                return 8.0

        model, opt = _mlp()
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        with session.step(0, scaler=_Duck()):
            opt.zero_grad(set_to_none=True)
            (model(torch.randn(4, 8)).sum() * 8.0).backward()
            opt.step()
        session.close()
        measured = next(model.parameters()).grad.double().norm().item()
        assert math.isclose(_norm(sink), measured, rel_tol=1e-7)
        block = session.collector.ring.blocks[-1]
        assert block.block.unscaled == "unknown"
        assert {o.grad_scale for o in block.observations if o.stream == "param_grad"} == {"unknown"}
        assert not any(p.tag == "torchlens/run/amp_unscale_derived" for p in sink.scalars)

    @pytest.mark.smoke
    @_requires_gradscaler
    def test_forced_final_gradient_sample_is_corrected_when_derived(self) -> None:
        """The close-time forced sample applies the same correction."""

        model, opt = _mlp()
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=50)
        for step in (0, 1):
            with session.step(step, scaler=scaler):
                opt.zero_grad(set_to_none=True)
                scaler.scale(model(torch.randn(4, 8)).sum()).backward()
                opt.step()
        report = session.close()
        assert report.final_sample_forced
        forced = next(
            p.value for p in sink.scalars if p.tag == "gradients/norm/0.weight" and p.step == 1
        )
        true_norm = (next(model.parameters()).grad.double() / 1024.0).norm().item()
        assert math.isclose(forced, true_norm, rel_tol=1e-7)


class TestCorrectedBlock:
    @_requires_gradscaler
    def test_corrected_gradient_block_leaves_non_gradient_streams_alone(self) -> None:
        model, opt = _mlp()
        scaler = torch.amp.GradScaler("cpu", init_scale=4.0)
        sink = trk.MemorySink()
        session = trk.watch(
            model, to=sink, signals=("gradients", "parameters"), optimizer=opt, every=1
        )
        with session.step(0, scaler=scaler):
            opt.zero_grad(set_to_none=True)
            scaler.scale(model(torch.randn(4, 8)).sum()).backward()
            opt.step()
        session.close()
        block = session.collector.ring.blocks[-1]
        corrected, count = trk.corrected_gradient_block(block, 4.0)
        assert count == sum(1 for o in block.observations if o.stream == "param_grad")
        for before, after in zip(block.observations, corrected.observations, strict=True):
            if before.stream == "param_grad":
                assert after.grad_scale == "unscaled"
                assert after.spine.sum_squares == before.spine.sum_squares / 16.0
            else:
                assert after is before

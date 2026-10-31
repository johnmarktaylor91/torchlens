"""Op-grain watching (U32.1) behind consent and attach-time calibration (U34.1).

``watch(grain="op")`` drives the Route-B op-tier collector from an ordinary
training loop through a disclosed root-forward wrap. These tests pin: the
consent refusal; per-op values that equal what the op actually produced; the
calibration disclosure (plan report, tracker rows, one warning) with the
model's RNG and buffers untouched by attach; the wrap removed at close; the
first-forward-per-step law; facet series; the implicit ``step=`` path; and the
grain vocabulary refusals.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.trackers as trk
from torchlens._observability import split_heads
from torchlens.errors import WatchConfigError


def _mlp() -> tuple[nn.Module, torch.optim.Optimizer]:
    """A small training target with one ReLU op."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    return model, torch.optim.SGD(model.parameters(), lr=0.05)


def _watch_relu(model: nn.Module, sink: trk.MemorySink, **kwargs: object) -> trk.WatchSession:
    """Attach an op-grain watch on the ReLU op, warnings swallowed."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trk.watch(
            model,
            to=sink,
            signals=("activations",),
            grain="op",
            select=tl.func("relu"),
            inputs=torch.randn(4, 8),
            allow_forward_wrap=True,
            every=1,
            **kwargs,
        )


def _norm_tags(sink: trk.MemorySink) -> dict[int, float]:
    """step -> value of the single op site's activation norm series."""

    rows = [p for p in sink.scalars if p.tag.startswith("activations/norm/")]
    assert {p.tag for p in rows}, "no op-grain activation norm series landed"
    assert len({p.tag for p in rows}) == 1, sorted({p.tag for p in rows})
    return {p.step: p.value for p in rows}


class TestConsent:
    def test_op_grain_without_consent_refuses_typed(self) -> None:
        model, _opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(
                model,
                to=trk.MemorySink(),
                signals=("activations",),
                grain="op",
                select=tl.func("relu"),
                inputs=torch.randn(4, 8),
            )
        assert info.value.fields["code"] == "watch_forward_wrap_consent_required"
        assert "allow_forward_wrap=True" in str(info.value.fields["remedy"])
        assert "forward" not in model.__dict__, "a refused attach must not wrap"

    def test_unknown_grain_and_stray_op_knobs_refuse(self) -> None:
        model, opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), optimizer=opt, grain="facet")
        assert info.value.fields["code"] == "watch_grain_invalid"
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), optimizer=opt, allow_forward_wrap=True)
        assert info.value.fields["code"] == "watch_signals_invalid"

    def test_op_grain_is_activations_only(self) -> None:
        model, opt = _mlp()
        with pytest.raises(WatchConfigError) as info:
            trk.watch(
                model,
                to=trk.MemorySink(),
                signals=("gradients",),
                optimizer=opt,
                grain="op",
                allow_forward_wrap=True,
            )
        assert info.value.fields["code"] == "watch_signals_invalid"

    def test_module_grain_default_installs_no_wrap(self) -> None:
        """Explorer sign-off 2: module grain stays the default; op grain is explicit."""

        model, opt = _mlp()
        session = trk.watch(model, to=trk.MemorySink(), optimizer=opt)
        assert "forward" not in model.__dict__
        assert "WRAPPED" not in session.plan_report()
        session.close(unwinding=True)


class TestOpValues:
    def test_records_per_op_values_equal_to_the_op_output(self) -> None:
        """Acceptance U32.1: the series value is the ReLU op output's L2 norm."""

        model, opt = _mlp()
        seen: dict[int, float] = {}
        current = {"step": -1}

        def relu_hook(_module: nn.Module, _args: object, output: torch.Tensor) -> None:
            seen.setdefault(current["step"], float(output.detach().double().norm()))

        sink = trk.MemorySink()
        session = _watch_relu(model, sink)
        handle = model[1].register_forward_hook(relu_hook)
        try:
            for step in (100, 101, 102):
                current["step"] = step
                with session.step(step):
                    loss = model(torch.randn(4, 8)).square().mean()
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
        finally:
            handle.remove()
        report = session.close()
        values = _norm_tags(sink)
        assert sorted(values) == [100, 101, 102]
        for step, value in values.items():
            assert value == pytest.approx(seen[step], rel=1e-6)
        assert report.matched_sites == 1
        assert report.sampled_steps == (100, 101, 102)

    def test_training_is_unchanged_by_the_wrap(self) -> None:
        """The observed forward returns the live output: same loss, same update."""

        torch.manual_seed(0)
        inputs = [torch.randn(4, 8) for _ in range(3)]
        reference, ref_opt = _mlp()
        watched, opt = _mlp()
        session = _watch_relu(watched, trk.MemorySink())
        for step, x in enumerate(inputs):
            ref_loss = reference(x).square().mean()
            ref_loss.backward()
            ref_opt.step()
            ref_opt.zero_grad()
            with session.step(step):
                loss = watched(x).square().mean()
                loss.backward()
                opt.step()
                opt.zero_grad()
            assert torch.equal(loss.detach(), ref_loss.detach())
        session.close()
        for ref_param, param in zip(reference.parameters(), watched.parameters(), strict=True):
            assert torch.equal(ref_param, param)

    def test_only_sampled_steps_pay_the_capture(self) -> None:
        model, opt = _mlp()
        sink = trk.MemorySink()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session = trk.watch(
                model,
                to=sink,
                signals=("activations",),
                grain="op",
                select=tl.func("relu"),
                inputs=torch.randn(4, 8),
                allow_forward_wrap=True,
                every=2,
            )
        for step in range(4):
            with session.step(step):
                model(torch.randn(4, 8)).sum().backward()
                opt.step()
                opt.zero_grad()
        session.close()
        assert sorted(_norm_tags(sink)) == [0, 2]

    def test_second_forward_in_a_step_is_a_named_skip(self) -> None:
        model, opt = _mlp()
        sink = trk.MemorySink()
        session = _watch_relu(model, sink)
        with session.step(0):
            first = model(torch.randn(4, 8))
            second = model(torch.randn(4, 8))
            (first.sum() + second.sum()).backward()
            opt.step()
        report = session.close()
        assert sorted(_norm_tags(sink)) == [0]
        assert any(skip.startswith("op_grain_extra_forward") for skip in report.named_skips)

    def test_implicit_step_source_opens_one_block_per_step(self) -> None:
        model, opt = _mlp()
        sink = trk.MemorySink()
        counter = {"step": 10}
        session = _watch_relu(model, sink, step=lambda: counter["step"])
        for step in (10, 11, 12):
            counter["step"] = step
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        report = session.close()
        assert sorted(_norm_tags(sink)) == [10, 11, 12]
        assert report.first_step == 10 and report.last_step == 12


class TestCalibrationDisclosure:
    def test_attach_calibrates_and_discloses_the_cost(self) -> None:
        """Acceptance U34.1: consent -> calibration -> disclosed cost."""

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.5))
        model.train()
        example = torch.randn(4, 8)
        buffers_before = {name: buf.clone() for name, buf in model.named_buffers()}
        rng_before = torch.random.get_rng_state()
        sink = trk.MemorySink()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            session = trk.watch(
                model,
                to=sink,
                signals=("activations",),
                grain="op",
                select=tl.func("relu"),
                inputs=example,
                allow_forward_wrap=True,
            )
        codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
        assert codes.count("watch_op_grain_cost_disclosed") == 1
        for name, buf in model.named_buffers():
            assert torch.equal(buf, buffers_before[name]), f"attach changed buffer {name}"
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        measurement = session.collector.calibration
        assert measurement is not None and measurement.reps >= 3
        assert measurement.median_ratio is not None and measurement.median_ratio > 0
        plan = session.plan_report()
        assert "WRAPPED" in plan and "calibration at attach" in plan
        with session.step(0):
            model(torch.randn(4, 8)).sum().backward()
        session.close()
        assert any(t.tag == "torchlens/meta/op_grain_calibration" for t in sink.texts)
        overhead = [p for p in sink.scalars if p.tag == "torchlens/run/op_grain_overhead_median_x"]
        assert overhead and overhead[0].value == pytest.approx(measurement.median_ratio)

    def test_close_removes_the_wrap(self) -> None:
        model, _opt = _mlp()
        x = torch.randn(4, 8)
        expected = model(x)
        session = _watch_relu(model, trk.MemorySink())
        assert "forward" in model.__dict__
        with session.step(0):
            model(x)
        session.close()
        assert "forward" not in model.__dict__
        assert torch.equal(model(x), expected)


class _Heads(nn.Module):
    """An attention-shaped op: [B, H, S, D] scores from one matmul."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.proj(x).view(x.shape[0], 3, 2, 4).transpose(1, 2)
        return torch.matmul(q, q.transpose(-1, -2))


def test_facets_land_as_per_head_series() -> None:
    """Explorer sign-off 2: the facet spelling is explicit and per-head."""

    torch.manual_seed(0)
    model = _Heads()
    x = torch.randn(2, 3, 4)
    discovery = trk.MemorySink()
    probe = _watch_relu_like(model, discovery, x, facets=None)
    label = next(site.display_label for site in probe.collector._sites_by_id.values())
    probe.close(unwinding=True)
    sink = trk.MemorySink()
    session = _watch_relu_like(model, sink, x, facets={label: split_heads(2, dim=1)})
    with session.step(0):
        model(x).sum().backward()
    session.close()
    tags = {p.tag for p in sink.scalars if p.tag.startswith("activations/norm/")}
    assert f"activations/norm/{label}/head0" in tags
    assert f"activations/norm/{label}/head1" in tags


def _watch_relu_like(
    model: nn.Module, sink: trk.MemorySink, x: torch.Tensor, facets: object
) -> trk.WatchSession:
    """Op-grain watch on the model's matmul op."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trk.watch(
            model,
            to=sink,
            signals=("activations",),
            grain="op",
            select=tl.func("matmul"),
            inputs=x,
            allow_forward_wrap=True,
            facets=facets,  # type: ignore[arg-type]
            every=1,
        )

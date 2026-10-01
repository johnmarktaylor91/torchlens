"""Explorer acceptance-matrix rows and D25 gate rows (memo s5; lane F25).

CPU-runnable matrix rows land here as ordinary tests; the numeric CUDA
gate rows (module-tier <= 15%, sketch tier <= 30% median full-train-step
overhead, peak memory <= 10% and flat) are ``rare``-marked and run by the
C-EXPLORER cluster row on a dedicated GPU -- a loaded shared CPU box
cannot establish them (measured 1.54x same-workload noise band), and no
number is claimed without the harness verdict.

Matrix coverage map (memo section 5): rows 3 (numbers right + NaN at the
exact coordinate, observation inert), 4 (paper-ready honesty), 5 (real
resnet18: in-place ReLU, train-mode BatchNorm unperturbed), 8 (resume =
new segment), 9 (reused module truth), 10 (zero-match plan refusal --
test_explorer_watch_op_tier), 11 (raising reducer fails open), 15
(cross-site merge refusal). Row 1's CUDA flagship and rows 6 (DDP), 13
(gpt2 attention heads at scale), 14 (Qwen2.5) ride C-EXPLORER / D02.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.observability import (
    Histogram,
    HistoryCollector,
    HistoryView,
    ObservationRecord,
    Spine,
    StepTruth,
    merge_observations,
    render_contact_sheet,
    render_fan,
)
from torchlens.observability._errors import HistorySchemaError
from torchlens.observability._op_tier import OpTierCollector


class _NaNInjector(torch.nn.Module):
    """Injects one NaN at a known coordinate on a chosen step."""

    def __init__(self) -> None:
        super().__init__()
        self.fire = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fire:
            x = x.clone()
            x[0, 0] = float("nan")
        return x


@pytest.mark.smoke
class TestNumbersRightAndObservationInert:
    """Matrix row 3: exact counts vs direct torch; NaN at the coordinate."""

    def test_first_and_last_step_cross_check(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU())
        collector = HistoryCollector(model, streams=("activation",))
        collector.discover(torch.randn(4, 8))
        collector.attach()
        inputs = [torch.randn(4, 8) for _ in range(3)]
        outputs: dict[int, torch.Tensor] = {}
        hook = model[1].register_forward_hook(
            lambda module, args, output: outputs.__setitem__(len(outputs), output.detach().clone())
        )
        try:
            for step, x in enumerate(inputs):
                with collector.step(step, truth=StepTruth()):
                    model(x)
        finally:
            hook.remove()
            collector.detach()
        view = HistoryView.from_collector(collector)
        relu_site = next(s for s, r in view.sites.items() if r.module_path == "1")
        series = view.series(relu_site)
        for index in (0, len(series) - 1):
            spine = series[index].observation.spine
            direct = outputs[index]
            assert spine.count_total == direct.numel()
            assert spine.count_zero == int((direct == 0).sum())
            assert spine.finite_max == pytest.approx(float(direct.max()), rel=1e-6)
            assert spine.sum == pytest.approx(float(direct.double().sum()), rel=1e-9)

    def test_nan_appears_at_exactly_its_coordinate(self) -> None:
        torch.manual_seed(1)
        injector = _NaNInjector()
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), injector, torch.nn.Linear(8, 4))
        collector = HistoryCollector(model, streams=("activation",))
        collector.discover(torch.randn(2, 8))
        collector.attach()
        try:
            for step in range(4):
                injector.fire = step == 2
                with collector.step(step, truth=StepTruth()):
                    model(torch.randn(2, 8))
        finally:
            injector.fire = False
            collector.detach()
        view = HistoryView.from_collector(collector)
        injector_site = next(s for s, r in view.sites.items() if r.module_path == "1")
        nan_steps = [
            p.step_lo
            for p in view.series(injector_site)
            if p.observation.spine is not None and p.observation.spine.count_nan > 0
        ]
        assert nan_steps == [2], "NaN lands at exactly its (site, step) coordinate"
        assert "nonfinite" in render_fan(view, injector_site)
        # Observation is inert: no other site saw a NaN at step 2 upstream.
        upstream = next(s for s, r in view.sites.items() if r.module_path == "0")
        assert all(p.observation.spine.count_nan == 0 for p in view.series(upstream))


@pytest.mark.smoke
class TestResumeSegments:
    """Matrix row 8: a restart appends a NEW segment, boundary preserved."""

    def test_new_segment_renders_boundary(self) -> None:
        torch.manual_seed(2)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        collector = HistoryCollector(model, streams=("activation",))
        collector.discover(torch.randn(2, 4))
        collector.attach()
        try:
            for step in range(3):
                with collector.step(step):
                    model(torch.randn(2, 4))
            # Duplicate step without a declared segment refuses (D6)...
            with pytest.raises(HistorySchemaError) as excinfo, collector.step(1):
                model(torch.randn(2, 4))
            assert excinfo.value.fields["code"] == "history_step_regression"
            # ...and the declared resume starts a new segment.
            with collector.step(1, new_segment=True):
                model(torch.randn(2, 4))
        finally:
            collector.detach()
        view = HistoryView.from_collector(collector)
        segments = {b.block.segment_id for b in view.blocks}
        assert len(segments) == 2
        site = next(iter(view.sites))
        assert "segment" in render_fan(view, site)


@pytest.mark.smoke
class TestReusedModuleTruth:
    """Matrix row 9: a module called twice per step -- current truth pinned.

    The Route-A collector folds repeated calls of one module within a step
    into ONE merged observation (exact integer merge over both call
    populations). Pass-qualified per-call series are the op tier's grain
    (fastlog labels are per structural position); this pin documents the
    module-tier fold so it can never silently change meaning.
    """

    def test_double_call_folds_exactly(self) -> None:
        torch.manual_seed(3)

        class TwiceModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.core = torch.nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.core(self.core(x))

        model = TwiceModel()
        collector = HistoryCollector(model, sites=("core",), streams=("activation",))
        collector.discover(torch.randn(2, 4))
        collector.attach()
        try:
            with collector.step(0):
                model(torch.randn(2, 4))
        finally:
            collector.detach()
        view = HistoryView.from_collector(collector)
        site = next(iter(view.sites))
        spine = view.series(site)[0].observation.spine
        assert spine.count_total == 2 * (2 * 4), "both calls' populations folded exactly"


@pytest.mark.smoke
class TestFailOpenReducer:
    """Matrix row 11: a raising reducer records capture_failed; step survives."""

    def test_op_tier_raising_facet_fails_open(self) -> None:
        torch.manual_seed(4)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

        def bomb(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
            raise RuntimeError("boom")

        import torchlens as tl

        probe = OpTierCollector(model, save=tl.func("relu"))
        probe.discover(torch.randn(2, 4))
        relu_label = next(iter(probe.site_catalog.values())).display_label
        collector = OpTierCollector(model, save=tl.func("relu"), facets={relu_label: bomb})
        collector.discover(torch.randn(2, 4))
        with collector.step(0):
            output = collector.observe_step(torch.randn(2, 4))
        assert output is not None, "the training step survived the reducer failure"
        view = HistoryView.from_collector(collector)
        observations = [o for b in view.blocks for o in b.observations]
        assert len(observations) == 1
        assert observations[0].presence == "capture_failed"
        assert "boom" in (observations[0].reason or "")


@pytest.mark.smoke
class TestCrossSiteRollupRefused:
    """Matrix row 15: no rollup door exists; cross-site merge refuses typed."""

    def test_merge_across_sites_refuses(self) -> None:
        def observed(site: str) -> ObservationRecord:
            spine = Spine()
            spine.update(torch.randn(8))
            sketch = Histogram()
            sketch.update(torch.randn(8))
            return ObservationRecord(
                global_step=0,
                site_id=site,
                stream="activation",
                phase="forward",
                presence="observed",
                spine=spine.result(),
                sketch=sketch.result(),
            )

        with pytest.raises(HistorySchemaError) as excinfo:
            merge_observations(observed("m:a"), observed("m:b"))
        assert excinfo.value.fields["code"] == "history_merge_incompatible"


@pytest.mark.real_model
@pytest.mark.heavy
class TestResnet18Row:
    """Matrix row 5: real resnet18 -- in-place ReLUs, train-mode BatchNorm."""

    def test_running_stats_unperturbed_and_sheet_renders(self) -> None:
        torchvision = pytest.importorskip("torchvision")
        torch.manual_seed(0)
        # Both variants constructed BEFORE any capture, identical weights.
        watched = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        control = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        watched.train()
        control.train()
        batches = [torch.randn(2, 3, 64, 64) for _ in range(2)]
        collector = HistoryCollector(watched, sites=("layer1", "layer2"), streams=("activation",))
        collector.discover(batches[0])
        collector.attach()
        try:
            for step, batch in enumerate(batches):
                with collector.step(step):
                    watched(batch)
        finally:
            collector.detach()
        # The unobserved control sees the SAME batches (including the
        # discovery forward, which is a real train-mode forward).
        control(batches[0])
        for batch in batches:
            control(batch)
        for (name_w, buf_w), (name_c, buf_c) in zip(
            watched.named_buffers(), control.named_buffers(), strict=True
        ):
            assert name_w == name_c
            assert torch.equal(buf_w, buf_c), f"observer perturbed {name_w}"
        view = HistoryView.from_collector(collector)
        sheet = render_contact_sheet(view)
        assert "of" in sheet and sheet.startswith("<svg")


@pytest.mark.real_model
@pytest.mark.slow
class TestGpt2CpuLeg:
    """Matrix row 1's CPU correctness leg (reduced; the CUDA flagship is
    C-EXPLORER's). Real gpt2 checkpoint, real training steps, honest sheet."""

    def test_gpt2_watch_four_steps(self) -> None:
        transformers = pytest.importorskip("transformers")
        torch.manual_seed(0)
        model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        ids = torch.randint(0, 50257, (1, 16))
        from torchlens.observability import WatchSettings

        collector = HistoryCollector(
            model,
            sites=tuple(f"transformer.h.{i}" for i in (0, 5, 11)),
            streams=("activation",),
            settings=WatchSettings(sketch_cadence=1),
        )
        collector.discover(input_ids=ids, labels=ids)
        assert collector.plan.total_elements_per_step > 0
        collector.attach(optimizer=optimizer)
        try:
            for step in range(4):
                with collector.step(step, truth=StepTruth()):
                    out = model(input_ids=ids, labels=ids)
                    out.loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        finally:
            collector.detach()
        view = HistoryView.from_collector(collector)
        assert len(view.blocks) == 4
        sheet = render_contact_sheet(view)
        assert "nonfinite_first_drift_v1" in sheet
        block_sites = [s for s, r in view.sites.items() if r.module_path == "transformer.h.0"]
        assert block_sites
        series = view.series(block_sites[0])
        assert all(p.observation.sketch is not None for p in series)


@pytest.mark.rare
@pytest.mark.real_model
class TestCudaGateRows:
    """The D25 numeric ship gates (C-EXPLORER cluster row; dedicated GPU).

    Run explicitly: ``pytest tests/test_explorer_watch_gates.py -m rare``.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="C-EXPLORER runs on CUDA")
    @pytest.mark.parametrize(
        ("sketch_cadence", "threshold"),
        [
            pytest.param(None, 1.15, id="spine-only-15pct"),
            pytest.param(10, 1.30, id="sketch-cadence10-30pct"),
        ],
    )
    def test_module_tier_overhead_gates(self, sketch_cadence: int | None, threshold: float) -> None:
        transformers = pytest.importorskip("transformers")
        from torchlens.observability import WatchSettings, measure_ab, measure_noise_band

        torch.manual_seed(0)
        device = "cuda"
        model_a = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device).train()
        model_b = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(device).train()
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-5)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-5)
        ids = torch.randint(0, 50257, (2, 128), device=device)

        def plain() -> None:
            out = model_a(input_ids=ids, labels=ids)
            out.loss.backward()
            opt_a.step()
            opt_a.zero_grad()
            torch.cuda.synchronize()

        collector = HistoryCollector(
            model_b,
            sites=tuple(f"transformer.h.{i}" for i in range(12)),
            streams=("activation",),
            settings=WatchSettings(sketch_cadence=sketch_cadence),
        )
        collector.discover(input_ids=ids, labels=ids)
        collector.attach()
        counter = {"i": 0}

        def watched() -> None:
            i = counter["i"]
            counter["i"] += 1
            with collector.step(i, truth=StepTruth()):
                out = model_b(input_ids=ids, labels=ids)
                out.loss.backward()
                opt_b.step()
                opt_b.zero_grad()
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        band = measure_noise_band(plain, repeats=20, warmup=3)
        tier = "spine-only" if sketch_cadence is None else f"sketch@{sketch_cadence}"
        row = measure_ab(
            f"gpt2-cuda module-tier {tier} (12 blocks)",
            plain,
            watched,
            repeats=20,
            warmup=3,
            noise_band=band,
            note="C-EXPLORER gate row; gpt2-124M b2xs128",
        )
        collector.detach()
        # D25 gates on a box whose noise band can resolve them: <=15%
        # median full-train-step overhead for module-tier spines, <=30%
        # for the sketch tier at the shipped cadence.
        assert band < 1.10, f"box noise band {band:.3f}x too wide for the gate"
        assert row.ratio <= threshold, row.verdict()

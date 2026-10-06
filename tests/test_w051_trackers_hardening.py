"""W051-TRACK / AUD-CODE 2.14 "also" items: the silently-weak spines around the engine.

Torn JSONL rows, a flush() raise skipping close(), run-health rows masking an
empty run, tag-safety refusing inside a ``finally``, a second watch() on one
model duplicating every point, sparse gradients crashing untyped inside the
optimizer step, config-class sink refusals swallowed into the runtime latch,
attach rows hardcoded to step 0, and the HF step axis off by one.
"""

from __future__ import annotations

import json
import sys
import types

import pytest
import torch

import torchlens.trackers as trk
from torchlens.trackers._errors import (
    SinkDeliveryError,
    TagGrammarError,
    TrackersError,
    WatchConfigError,
    WatchRuntimeError,
)
from torchlens.utils._torch_compat import HAS_AMP_GRADSCALER

_requires_gradscaler = pytest.mark.skipif(
    not HAS_AMP_GRADSCALER,
    reason="torch.amp.GradScaler (device-agnostic) postdates the torch 2.1 floor",
)


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def _step(session, model, opt, step: int) -> None:  # noqa: ANN001
    with session.step(step):
        opt.zero_grad(set_to_none=True)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()


class TestJSONLTornTail:
    @pytest.mark.smoke
    def test_partial_write_failure_leaves_only_whole_rows(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "run.jsonl"
        sink = trk.JSONLSink(path)
        real = sink._handle

        class _TornHandle:
            """Writes a prefix of the first row, then reports ENOSPC."""

            def __init__(self) -> None:
                self.calls = 0

            def write(self, data: bytes) -> int:
                self.calls += 1
                if self.calls == 1:
                    real.write(data[:7])
                    raise OSError(28, "No space left on device")
                return real.write(data)

            def fileno(self) -> int:
                return real.fileno()

            def flush(self) -> None:
                real.flush()

            def close(self) -> None:
                real.close()

        sink._handle = _TornHandle()  # type: ignore[assignment]
        with pytest.raises(SinkDeliveryError) as info:
            sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 3, 1.5))
        assert info.value.fields["code"] == "tracker_sink_delivery_failed"
        sink.close()
        lines = path.read_text().splitlines()
        # Header only: the torn row was cut back, every remaining line parses.
        assert len(lines) == 1
        assert json.loads(lines[0])["format"] == trk.JSONL_FORMAT
        assert '"closed"' not in path.read_text()

    def test_every_row_is_durable_before_the_next_is_accepted(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "run.jsonl"
        sink = trk.JSONLSink(path)
        sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", 1, 2.0))
        # Unbuffered: visible on disk without flush/close.
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[-1] == {"kind": "scalar", "tag": "gradients/norm/w", "step": 1, "value": 2.0}
        sink.close()


class TestCloseSpine:
    @pytest.mark.smoke
    def test_flush_raise_still_closes_the_sink(self) -> None:
        class _FlushFails(trk.MemorySink):
            def flush(self) -> None:
                raise OSError("flush died")

        model, opt = _mlp()
        sink = _FlushFails()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        _step(session, model, opt, 0)
        report = session.close()
        assert sink.closed is True
        row = report.sink_rows[0]
        assert row["failed"] is True
        assert row["failure"].startswith("flush:")

    def test_run_health_rows_never_mask_an_empty_run(self) -> None:
        """Data points decide emptiness; heartbeat/manifest rows do not."""

        class _DropsData(trk.MemorySink):
            def emit_scalar(self, point):  # noqa: ANN001
                if self.__class__.grammar_is_data(point.tag):
                    raise OSError("data rows refused")
                super().emit_scalar(point)

            @staticmethod
            def grammar_is_data(tag: str) -> bool:
                return trk.TagGrammar().is_data_tag(tag)

        model, opt = _mlp()
        sink = _DropsData()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        _step(session, model, opt, 0)
        with pytest.raises(WatchRuntimeError) as info:
            session.close()
        assert info.value.fields["code"] == "watch_close_empty"

    @pytest.mark.smoke
    @_requires_gradscaler
    def test_all_steps_explained_by_named_skips_does_not_raise(self) -> None:
        """A run whose every step is AMP-skipped is explained, not empty."""

        model, opt = _mlp()
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("updates",), optimizer=opt, every=1)
        with session.step(0, scaler=scaler):
            opt.zero_grad(set_to_none=True)
            scaler.scale(model(torch.randn(4, 8)).sum()).backward()
            next(model.parameters()).grad[0, 0] = float("inf")
            scaler.step(opt)
            scaler.update()
        report = session.close()
        assert any(skip.startswith("amp_skipped") for skip in report.named_skips)


class TestAttachPreflights:
    def test_unsafe_leaf_refuses_at_attach_not_in_the_first_drain(self) -> None:
        model, opt = _mlp()
        model.add_module("bad?name", torch.nn.Linear(2, 2))
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sink = trk.MemorySink()
        with pytest.raises(TagGrammarError) as info:
            trk.watch(model, to=sink, optimizer=opt)
        assert info.value.fields["code"] == "tracker_tag_unsafe"
        assert sink.scalars == [] and sink.texts == []

    @pytest.mark.smoke
    def test_second_session_same_grammar_refuses_distinct_name_allowed(self) -> None:
        model, opt = _mlp()
        sink = trk.MemorySink()
        first = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        assert info.value.fields["code"] == "tracker_namespace_collision"
        second = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1, name="b")
        second.close(unwinding=True)
        first.close(unwinding=True)
        # Closed sessions release the model for a fresh watch.
        again = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        again.close(unwinding=True)

    @pytest.mark.smoke
    def test_sparse_gradient_modules_refuse_at_attach(self) -> None:
        class _Emb(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.e = torch.nn.Embedding(10, 4, sparse=True)
                self.l = torch.nn.Linear(4, 1)

            def forward(self, x):  # noqa: ANN001
                return self.l(self.e(x))

        model = _Emb()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with pytest.raises(WatchConfigError) as info:
            trk.watch(model, to=trk.MemorySink(), signals=("gradients",), optimizer=opt)
        assert info.value.fields["code"] == "watch_sparse_grad_unsupported"
        assert info.value.fields["parameters"] == ("e.weight",)
        # Excluding the sparse site (or not asking for gradients) is admitted.
        narrowed = trk.watch(
            model, to=trk.MemorySink(), signals=("gradients",), optimizer=opt, select=("l",)
        )
        narrowed.close(unwinding=True)
        params_only = trk.watch(model, to=trk.MemorySink(), signals=("parameters",), optimizer=opt)
        params_only.close(unwinding=True)

    @pytest.mark.smoke
    def test_wandb_bucket_cap_refuses_at_attach_before_any_emission(self) -> None:
        """An EXPLICIT over-cap grid still refuses at attach, before any row."""

        from torchlens.observability import HistogramDescriptor, WatchSettings
        from torchlens.observability._kernels import DEFAULT_DESCRIPTOR

        class _Run:
            def __init__(self) -> None:
                self.logged: list = []

            def log(self, payload, step=None):  # noqa: ANN001
                self.logged.append((payload, step))

        model, opt = _mlp()
        run = _Run()
        with pytest.raises(TrackersError) as info:
            trk.watch(
                model,
                to=trk.WandbSink(run),
                optimizer=opt,
                hist_every=1,
                descriptor=DEFAULT_DESCRIPTOR,
            )
        assert info.value.fields["code"] == "tracker_histogram_bucket_cap"
        with pytest.raises(TrackersError) as info:
            trk.watch(
                model,
                to=trk.WandbSink(run),
                optimizer=opt,
                hist_every=1,
                settings=WatchSettings(descriptor=HistogramDescriptor()),
            )
        assert info.value.fields["code"] == "tracker_histogram_bucket_cap"
        assert run.logged == []
        safe = trk.watch(
            model,
            to=trk.WandbSink(run),
            optimizer=opt,
            hist_every=1,
            descriptor=trk.WANDB_SAFE_DESCRIPTOR,
        )
        safe.close(unwinding=True)

    @pytest.mark.smoke
    def test_wandb_sink_supplies_its_safe_grid_when_none_is_chosen(self) -> None:
        """No descriptor and no settings: the sink's cap-safe grid attaches."""

        from torchlens.observability import WatchSettings
        from torchlens.observability._kernels import DEFAULT_DESCRIPTOR

        class _Run:
            def log(self, payload, step=None):  # noqa: ANN001
                del payload, step

        model, opt = _mlp()
        sink = trk.WandbSink(_Run())
        assert sink.default_histogram_descriptor() is trk.WANDB_SAFE_DESCRIPTOR
        session = trk.watch(model, to=sink, optimizer=opt, hist_every=1)
        try:
            assert session.collector.settings.descriptor is trk.WANDB_SAFE_DESCRIPTOR
        finally:
            session.close(unwinding=True)
        # Any settings object is a caller choice, even one that leaves the C06
        # grid untouched: it is kept, so the over-cap default refuses exactly
        # like descriptor=DEFAULT_DESCRIPTOR does.
        for kwargs in ({"settings": WatchSettings()}, {"descriptor": DEFAULT_DESCRIPTOR}):
            with pytest.raises(TrackersError) as info:
                trk.watch(model, to=trk.WandbSink(_Run()), optimizer=opt, hist_every=1, **kwargs)
            assert info.value.fields["code"] == "tracker_histogram_bucket_cap"
        # Several offering sinks: the first offer applies to the whole session.
        session = trk.watch(
            model, to=(trk.WandbSink(_Run()), trk.WandbSink(_Run())), optimizer=opt, hist_every=1
        )
        try:
            assert session.collector.settings.descriptor is trk.WANDB_SAFE_DESCRIPTOR
        finally:
            session.close(unwinding=True)
        # Sinks without the hook keep the C06 default grid.
        session = trk.watch(model, to=trk.MemorySink(), optimizer=opt, hist_every=1)
        try:
            assert session.collector.settings.descriptor is not trk.WANDB_SAFE_DESCRIPTOR
        finally:
            session.close(unwinding=True)

    @pytest.mark.smoke
    def test_tensorboard_relay_refuses_histograms_at_attach(self, monkeypatch) -> None:  # noqa: ANN001
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.patched = {"tensorboard": [("torch.utils.tensorboard", "SummaryWriter")]}  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        class _Writer:
            def __init__(self) -> None:
                self.calls: list = []

            def add_scalar(self, *args, **kwargs):  # noqa: ANN001
                self.calls.append(("scalar", args))

            def add_histogram_raw(self, *args, **kwargs):  # noqa: ANN001
                self.calls.append(("hist", args))

            def add_text(self, *args, **kwargs):  # noqa: ANN001
                self.calls.append(("text", args))

            def flush(self) -> None:
                pass

        model, opt = _mlp()
        writer = _Writer()
        with pytest.raises(TrackersError) as info:
            trk.watch(model, to=trk.TensorBoardSink(writer), optimizer=opt, hist_every=1)
        assert info.value.fields["code"] == "tracker_relay_histogram_unsupported"
        assert writer.calls == []


class TestAttachRows:
    def test_manifest_and_heartbeat_ride_the_first_seen_step(self) -> None:
        model, opt = _mlp()
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt, every=1)
        assert sink.texts == [] and sink.scalars == []  # nothing at step 0 on attach
        _step(session, model, opt, 500)
        manifest = next(p for p in sink.texts if p.tag == "torchlens/meta/manifest")
        heartbeat = next(p for p in sink.scalars if p.tag == "torchlens/run/attached")
        assert manifest.step == 500 and heartbeat.step == 500
        # Manifest precedes the first data row in emission order.
        first_data = next(p for p in sink.scalars if p.tag.startswith("gradients/"))
        assert sink.scalars.index(heartbeat) < sink.scalars.index(first_data)
        session.close()
        assert sum(1 for p in sink.texts if p.tag == "torchlens/meta/manifest") == 1

    def test_attach_rows_still_land_when_no_step_ever_ran(self) -> None:
        model, opt = _mlp()
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=opt)
        session.close()
        assert [p.tag for p in sink.texts] == ["torchlens/meta/manifest"]
        assert [p.tag for p in sink.scalars] == ["torchlens/run/attached"]


class TestHFStepAxis:
    @pytest.mark.smoke
    def test_rows_land_on_the_trainer_log_axis(self) -> None:
        pytest.importorskip("transformers")
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sink = trk.MemorySink()
        callback = trk.HFTrainerWatchCallback(to=sink, signals=("gradients",), every=1)

        class _Args:
            logging_steps = 1
            max_steps = 3

        class _State:
            global_step = 0

        state = _State()
        callback.on_train_begin(_Args(), state, None, model=model, optimizer=optimizer)
        for completed in range(3):
            # HF: global_step == number of COMPLETED updates during optimizer.step().
            state.global_step = completed
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
        callback.on_train_end(_Args(), state, None)
        steps = sorted({p.step for p in sink.scalars if p.tag.startswith("gradients/")})
        assert steps == [1, 2, 3]

"""F26: framework callbacks + the exporter unification drift pin."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
import torchlens.trackers as trk

pytestmark = pytest.mark.smoke


class TestDerivedCadence:
    """Memo 3.8: callbacks derive cadence from the framework's own signals."""

    def test_logging_steps_wins(self) -> None:
        assert trk.derived_cadence(logging_steps=25) == 25

    def test_run_length_targets_100_points(self) -> None:
        assert trk.derived_cadence(max_steps=5000) == 50
        assert trk.derived_cadence(max_steps=50) == 1
        assert trk.derived_cadence(max_steps=1_000_000) == 500

    def test_default_when_nothing_visible(self) -> None:
        assert trk.derived_cadence() == trk.DEFAULT_EVERY


class TestHFTrainerCallback:
    """Driven through the TrainerCallback protocol with real HF classes."""

    def test_full_callback_cycle(self) -> None:
        transformers = pytest.importorskip("transformers")
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sink = trk.MemorySink()
        callback = trk.HFTrainerWatchCallback(to=sink, signals=("gradients",))
        assert isinstance(callback._base, type(transformers.TrainerCallback))

        class _Args:
            logging_steps = 2
            max_steps = 10

        class _State:
            global_step = 0

        state = _State()
        callback.on_train_begin(_Args(), state, None, model=model, optimizer=optimizer)
        assert callback.session is not None
        assert callback.session.every == 2
        for step in range(3):
            state.global_step = step
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
        callback.on_train_end(_Args(), state, None)
        assert callback.session is None
        steps = sorted({p.step for p in sink.scalars if p.tag.startswith("gradients/")})
        assert steps and steps[0] == 0

    def test_derived_cadence_from_trainer_args(self) -> None:
        pytest.importorskip("transformers")
        callback = trk.HFTrainerWatchCallback(to=trk.MemorySink(), every=7)
        assert callback.every == 7


class TestLightningCallback:
    """Driven through the Lightning callback protocol."""

    def test_attach_and_close(self) -> None:
        pytest.importorskip("lightning")
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        sink = trk.MemorySink()
        callback = trk.LightningWatchCallback(to=sink, signals=("gradients",))

        class _Trainer:
            optimizers = [optimizer]
            log_every_n_steps = 5
            max_steps = 100
            global_step = 0

        trainer = _Trainer()
        callback.on_train_start(trainer, model)
        assert callback.session is not None
        assert callback.session.every == 5
        for step in range(2):
            trainer.global_step = step
            optimizer.zero_grad(set_to_none=True)
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
        callback.on_train_end(trainer, model)
        assert callback.session is None
        assert any(p.tag.startswith("gradients/") for p in sink.scalars)


@pytest.fixture(scope="module")
def export_trace():
    """One real captured trace shared by every exporter."""

    trace = tl.trace(torch.nn.Sequential(torch.nn.Linear(2, 2)), torch.randn(1, 2))
    try:
        yield trace
    finally:
        trace.cleanup()


class TestExporterUnification:
    """The four one-shot exporters share ONE summary converter (drift pin)."""

    def test_all_four_emit_the_same_summary_set(self, export_trace, monkeypatch) -> None:  # noqa: ANN001
        """tensorboard/wandb/mlflow/aim summary keys are identical."""

        import sys
        import types

        class _Writer:
            def __init__(self) -> None:
                self.scalars: dict[str, float] = {}

            def add_scalar(self, tag, value, step):  # noqa: ANN001
                self.scalars[tag.split("/", 1)[1]] = value

            def add_text(self, tag, text, step):  # noqa: ANN001
                pass

            def flush(self) -> None:
                pass

        class _Client:
            def __init__(self) -> None:
                self.metrics: dict[str, float] = {}

            def log_metric(self, key, value):  # noqa: ANN001
                self.metrics[key.split(".", 1)[1]] = value

        class _Run:
            def __init__(self) -> None:
                self.tracked: dict[str, float] = {}

            def track(self, value, name):  # noqa: ANN001
                self.tracked[name.split(".", 1)[1]] = value

        fake_wandb = types.ModuleType("wandb")

        class _Table:
            def __init__(self, dataframe=None):  # noqa: ANN001
                self.dataframe = dataframe

        fake_wandb.Table = _Table
        fake_wandb.run = None
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        writer, client, run = _Writer(), _Client(), _Run()
        tl.export.tensorboard(export_trace, writer, step=1)
        mlflow_result = tl.export.mlflow(export_trace, client=client)
        aim_result = tl.export.aim(export_trace, run=run)
        wandb_result = tl.export.wandb(export_trace)

        summary_keys = set(client.metrics)
        assert set(writer.scalars) == summary_keys
        assert set(run.tracked) == summary_keys
        assert summary_keys <= set(wandb_result)
        assert summary_keys <= set(mlflow_result)
        assert summary_keys <= set(aim_result)
        assert "num_layers" in summary_keys

    def test_step_is_required_keyword(self, export_trace) -> None:  # noqa: ANN001
        """The step=0 default is REMOVED: every emission carries a step."""

        class _Writer:
            def add_scalar(self, *a):  # noqa: ANN002
                pass

        with pytest.raises(TypeError, match="step"):
            tl.export.tensorboard(export_trace, _Writer())  # type: ignore[call-arg]

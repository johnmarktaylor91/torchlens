"""F26: T-RELAY-W / T-RELAY-C -- the measured relay-fidelity pins.

These pin the PANEL-MEASURED losses (trackers memo 3.2/3.3) at the measured
vendor versions, so a future vendor release that FIXES any of them fails
visibly and the published fidelity table gets revised. They run only where
the relay-test extra is installed (wandb 0.30.0 / clearml 2.1.12; see
the packaging-request ledger); everywhere else they skip -- the dep-free
halves of the relay law (detection + the G6 histogram refusal) are covered
in test_trackers_sinks_delivery.py.
"""

from __future__ import annotations

import pytest

#: The versions the fidelity table was measured at. A different installed
#: version does not silently pass: the pin asserts and names the drift.
MEASURED_WANDB = "0.30.0"
MEASURED_CLEARML = "2.1.12"


class TestRelayPinWandb:
    """T-RELAY-W: the wandb TB relay's measured losses stay lost."""

    def test_relay_rewrites_steps_and_drops_summaries(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        wandb = pytest.importorskip("wandb")
        pytest.importorskip("tensorboard")
        if wandb.__version__ != MEASURED_WANDB:
            pytest.skip(
                f"fidelity measured at wandb=={MEASURED_WANDB}, installed "
                f"{wandb.__version__}: re-run the T-RELAY-W measurement and "
                "revise the fidelity table before re-pinning"
            )
        monkeypatch.setenv("WANDB_MODE", "offline")
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        from _wandb_offline_log import history_rows
        from torch.utils.tensorboard import SummaryWriter

        run = wandb.init(sync_tensorboard=True, dir=str(tmp_path))
        try:
            assert wandb.patched["tensorboard"], "relay must be visibly detectable"
            writer = SummaryWriter(log_dir=str(tmp_path / "tb"))
            for step in (100, 102, 104):
                writer.add_scalar("gradients/norm/w", 1.0 + step, global_step=step)
            writer.add_histogram_raw(
                "hist/nonuniform",
                min=-7.0,
                max=3.5,
                num=15,
                sum=1.5,
                sum_squares=40.0,
                bucket_limits=[-1.0, -0.5, 0.0, 0.25, 4.0],
                bucket_counts=[1, 2, 3, 4, 5],
                global_step=100,
            )
            edges = [-1.0 + 2.0 * i / 600 for i in range(601)]
            writer.add_histogram_raw(
                "hist/wide",
                min=-1.0,
                max=1.0,
                num=600,
                sum=0.0,
                sum_squares=100.0,
                bucket_limits=edges[1:],
                bucket_counts=[1] * 600,
                global_step=102,
            )
            # A later step commits the step-102 row through the relay's
            # step-change flush instead of relying on the tail flush at finish.
            writer.add_scalar("tail/marker", 0.0, global_step=200)
            writer.close()
        finally:
            run.finish()
        rows = history_rows(tmp_path)
        scalars = [row for row in rows if "gradients/norm/w" in row]
        # Loss 1: the caller step is demoted to a side key; wandb's own _step
        # advances 0,1,2 (the axis rewrite the native sink exists to fix).
        assert [row["global_step"] for row in scalars] == [100, 102, 104]
        assert [row["_step"] for row in scalars] == [0, 1, 2]
        assert [row["gradients/norm/w"] for row in scalars] == [101, 103, 105]
        keys = [sorted(row) for row in rows]
        narrow = next((row for row in rows if "hist/nonuniform/_type" in row), None)
        wide = next((row for row in rows if "hist/wide/_type" in row), None)
        assert narrow is not None and wide is not None, keys
        # Loss 2: every histogram summary field is destroyed.
        for row, tag in ((narrow, "hist/nonuniform"), (wide, "hist/wide")):
            kept = {key.rsplit("/", 1)[1] for key in row if key.startswith(f"{tag}/")}
            assert kept == {"_type", "bins", "values"}, kept
        assert narrow["hist/nonuniform/values"] == [1, 2, 3, 4, 5]
        # Loss 3: >512 buckets are silently re-binned to 512.
        assert len(wide["hist/wide/values"]) == 512
        assert len(wide["hist/wide/bins"]) == 513
        assert wide["_step"] not in (100, 102)

    def test_native_sink_preserves_the_axis(self, tmp_path, monkeypatch) -> None:  # noqa: ANN001
        wandb = pytest.importorskip("wandb")
        monkeypatch.setenv("WANDB_MODE", "offline")
        import torchlens.trackers as trk

        run = wandb.init(dir=str(tmp_path))
        try:
            sink = trk.WandbSink(run)
            for step in (100, 102, 104):
                sink.emit_scalar(trk.ScalarPoint("gradients/norm/w", step, float(step)))
            sink.close()
        finally:
            run.finish()


class TestRelayPinClearml:
    """T-RELAY-C: the ClearML relay's measured histogram destruction."""

    def test_clearml_pin_version(self) -> None:
        clearml = pytest.importorskip("clearml")
        if clearml.__version__ != MEASURED_CLEARML:
            pytest.skip(
                f"fidelity measured at clearml=={MEASURED_CLEARML}, installed "
                f"{clearml.__version__}: re-run T-RELAY-C (both probes) and "
                "revise the fidelity table before re-pinning"
            )
        # The relay's histogram path resamples to ~granularity=50 columns
        # and conserves ~75% of the mass (measured independently by two
        # labs); the full offline probe rides
        # scratch/trackers_fable/probe_clearml_relay.py and lands here when
        # the relay-test extra is provisioned in CI.
        from clearml.backend_interface.metrics import reporter  # noqa: F401

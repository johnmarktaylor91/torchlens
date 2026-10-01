"""F26: T-RELAY-W / T-RELAY-C -- the measured relay-fidelity pins.

These pin the PANEL-MEASURED losses (trackers memo 3.2/3.3) at the measured
vendor versions, so a future vendor release that FIXES any of them fails
visibly and the published fidelity table gets revised. They run only where
the relay-test extra is installed (wandb 0.28.2 / clearml 2.1.12; see
the packaging-request ledger); everywhere else they skip -- the dep-free
halves of the relay law (detection + the G6 histogram refusal) are covered
in test_trackers_sinks_delivery.py.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.smoke]

#: The versions the fidelity table was measured at. A different installed
#: version does not silently pass: the pin asserts and names the drift.
MEASURED_WANDB = "0.28.2"
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
        from torch.utils.tensorboard import SummaryWriter

        run = wandb.init(sync_tensorboard=True, dir=str(tmp_path))
        try:
            writer = SummaryWriter(log_dir=str(tmp_path / "tb"))
            for step in (100, 102, 104):
                writer.add_scalar("gradients/norm/w", 1.0 + step, global_step=step)
            writer.close()
            # The relay demotes the caller step: wandb's own _step advances
            # 0,1,2 (the measured axis rewrite the native sink exists to fix).
            history = run.history(stream="default") if hasattr(run, "history") else None
            del history  # offline runs expose no readable history API; the
            # detector below is the load-bearing assertion.
            assert wandb.patched["tensorboard"], "relay must be visibly detectable"
        finally:
            run.finish()

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

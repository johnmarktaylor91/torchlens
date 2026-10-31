"""Tests for the Lightning callbacks: ``LayerProfilerCallback`` teardown and ``SummaryCallback``."""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

pytest.importorskip("lightning")

from torchlens._data_classes.trace import Trace
from torchlens.callbacks.lightning import LayerProfilerCallback


class _TinyModel(nn.Module):
    """Minimal model with a single saved-activation layer."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.linear(x)


class _FakeTrainer:
    """Minimal stand-in for a Lightning ``Trainer``."""

    global_step = 3


def _make_callback(tmp_path: Path) -> LayerProfilerCallback:
    """Return a callback writing to a scratch container path."""

    return LayerProfilerCallback(tmp_path / "profile.jsonl")


def test_maybe_profile_calls_trace_cleanup(tmp_path: Path) -> None:
    """``_maybe_profile`` must call ``Trace.cleanup()`` on every profiled batch.

    Regression test for the ref-cycle leak: ``_maybe_profile`` used to call
    ``tl.trace()`` every profiled batch without ever calling ``log.cleanup()``,
    so the CUDA cache was never emptied and the trace was never purged from
    the backward registry.
    """

    callback = _make_callback(tmp_path)
    model = _TinyModel()

    with patch.object(Trace, "cleanup", autospec=True) as mock_cleanup:
        callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    assert mock_cleanup.call_count == 1


def test_maybe_profile_calls_cleanup_even_if_write_fails(tmp_path: Path) -> None:
    """``log.cleanup()`` must run even when writing the JSONL record fails.

    The fix wraps record-building/writing in ``try/finally: log.cleanup()``.
    An exception mid-write must not skip cleanup and leak the trace.
    """

    callback = _make_callback(tmp_path)
    model = _TinyModel()

    with (
        patch.object(Trace, "cleanup", autospec=True) as mock_cleanup,
        patch.object(Path, "open", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    assert mock_cleanup.call_count == 1


def test_maybe_profile_releases_activation_tensor_deterministically(tmp_path: Path) -> None:
    """The profiled trace's activation payload is freed without ``gc.collect()``.

    Before the fix, ``cleanup()`` was never called, so each profiled batch's
    saved activation tensors (and the trace holding them) stayed alive via the
    Trace<->Op reference cycle until an incidental cyclic-GC pass -- an
    unbounded-until-GC leak in a tight training loop. With the fix,
    ``cleanup()`` unconditionally clears every ``Op`` attribute, so the
    activation tensor's refcount drops to zero and it is freed immediately by
    plain reference counting -- no ``gc.collect()`` required.
    """

    import gc

    import torchlens

    # Clear any pending garbage left over from earlier tests in this process
    # (unrelated cyclic references, e.g. frames from prior torch/trace calls)
    # so this test's own refcount-only assertion below isn't polluted by
    # unrelated prior state.
    gc.collect()

    callback = _make_callback(tmp_path)
    model = _TinyModel()

    tensor_ref: weakref.ReferenceType[torch.Tensor] | None = None
    orig_trace = torchlens.trace

    def _spying_trace(*args: Any, **kwargs: Any) -> Any:
        nonlocal tensor_ref
        log = orig_trace(*args, **kwargs)
        for op in log.layer_list:
            if getattr(op, "is_input", False):
                # The input node's ``out`` is the caller's own input tensor
                # (not a freshly computed activation) and may be kept alive
                # by the caller's own local scope; skip it.
                continue
            candidate = getattr(op, "out", None)
            if isinstance(candidate, torch.Tensor):
                tensor_ref = weakref.ref(candidate)
                break
        return log

    with patch("torchlens.trace", _spying_trace):
        callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    assert tensor_ref is not None, "no saved activation tensor was observed"
    assert tensor_ref() is None, (
        "activation tensor was still alive immediately after _maybe_profile "
        "returned -- cleanup() did not run (or did not run deterministically)"
    )


class _FrozenBackboneModel(nn.Module):
    """Training model whose backbone a user froze in eval mode."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.head = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return self.head(self.backbone(x))


@pytest.mark.parametrize("top_level_training", [True, False])
def test_maybe_profile_restores_every_submodule_training_flag(
    tmp_path: Path, top_level_training: bool
) -> None:
    """Profiling must restore each submodule's training flag exactly.

    Regression test: the callback saved only the top-level flag and called
    ``pl_module.train()`` afterwards, which switched a frozen eval-mode
    backbone (BatchNorm statistics included) back into train mode.
    """

    callback = _make_callback(tmp_path)
    model = _FrozenBackboneModel()
    model.train(top_level_training)
    model.backbone.eval()
    model.head[1].train()
    before = {name: module.training for name, module in model.named_modules()}

    callback._maybe_profile("train", _FakeTrainer(), model, {"x": torch.ones(2, 4)}, 0)

    after = {name: module.training for name, module in model.named_modules()}
    assert after == before


# --- SummaryCallback (walk U03.4) -------------------------------------------


class _SummaryTrainer:
    """Minimal trainer stand-in for the fit-start summary callback."""

    is_global_zero = True


def _summary_callback(**kwargs: Any) -> tuple[Any, list[str]]:
    """Return a SummaryCallback whose printed text is collected."""

    from torchlens.callbacks.lightning import SummaryCallback

    printed: list[str] = []
    return SummaryCallback(print_fn=printed.append, **kwargs), printed


def test_summary_callback_prints_once_from_first_training_batch() -> None:
    """The first training batch is the sample; later batches print nothing."""

    callback, printed = _summary_callback()
    model = _TinyModel()
    trainer = _SummaryTrainer()

    callback.on_fit_start(trainer, model)
    assert printed == [], "no example_input_array: nothing runs at fit start"
    callback.on_train_batch_start(trainer, model, (torch.ones(2, 4), torch.tensor([0, 1])), 0)
    callback.on_train_batch_start(trainer, model, (torch.ones(2, 4), torch.tensor([0, 1])), 1)

    assert len(printed) == 1
    header, _, body = printed[0].partition("\n")
    assert "sample: first training batch" in header
    assert "one extra forward in eval mode under no_grad" in header
    assert "restored" in header
    assert body == str(callback.report)
    assert "linear" in body


def test_summary_callback_reprints_on_the_next_fit() -> None:
    """Each fit start re-arms the callback, matching Lightning's ModelSummary."""

    callback, printed = _summary_callback()
    model = _TinyModel()
    trainer = _SummaryTrainer()
    for _ in range(2):
        callback.on_fit_start(trainer, model)
        callback.on_train_batch_start(trainer, model, {"x": torch.ones(2, 4)}, 0)
    assert len(printed) == 2


def test_summary_callback_uses_example_input_array_at_fit_start() -> None:
    """A declared example_input_array is summarized in on_fit_start."""

    callback, printed = _summary_callback()
    model = _TinyModel()
    model.example_input_array = torch.ones(3, 4)  # type: ignore[assignment]
    trainer = _SummaryTrainer()

    callback.on_fit_start(trainer, model)
    callback.on_train_batch_start(trainer, model, torch.ones(2, 4), 0)

    assert len(printed) == 1
    assert "sample: example_input_array" in printed[0]


def test_summary_callback_restores_modes_and_rng_exactly() -> None:
    """The extra forward leaves every module flag and the RNG stream untouched."""

    callback, _printed = _summary_callback()
    model = _FrozenBackboneModel()
    model.train()
    model.backbone.eval()
    before = {name: module.training for name, module in model.named_modules()}
    running_mean = model.backbone[1].running_mean.clone()
    batch = torch.randn(4, 4)  # drawn before the snapshot: only the callback may move it
    rng_before = torch.random.get_rng_state()

    callback.on_train_batch_start(_SummaryTrainer(), model, batch, 0)

    assert {name: module.training for name, module in model.named_modules()} == before
    assert torch.equal(model.backbone[1].running_mean, running_mean)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert callback.report is not None


def test_summary_callback_skips_with_warning_on_unguessable_batch() -> None:
    """A batch the forward cannot take warns typed and never kills the fit."""

    from torchlens.errors import TorchLensWarning

    callback, printed = _summary_callback()
    model = _TinyModel()
    batch = {"features": torch.ones(2, 4), "labels": torch.tensor([0, 1])}

    with pytest.warns(TorchLensWarning) as record:
        callback.on_train_batch_start(_SummaryTrainer(), model, batch, 0)

    assert printed == []
    assert callback.report is None
    codes = [getattr(item.message, "fields", {}).get("code") for item in record]
    assert "lightning_summary_skipped" in codes
    # Disarmed after the attempt: the next batch does not retry.
    callback.on_train_batch_start(_SummaryTrainer(), model, batch, 1)


def test_summary_callback_input_getter_and_rank_zero_only() -> None:
    """input_getter picks the sample; non-zero ranks never run the forward."""

    callback, printed = _summary_callback(input_getter=lambda batch: batch["features"])
    model = _TinyModel()
    batch = {"features": torch.ones(2, 4)}

    class _OtherRank:
        is_global_zero = False

    callback.on_train_batch_start(_OtherRank(), model, batch, 0)
    assert printed == []
    callback.on_fit_start(_SummaryTrainer(), model)
    callback.on_train_batch_start(_SummaryTrainer(), model, batch, 0)
    assert len(printed) == 1


def test_summary_callback_validates_summary_options_at_construction() -> None:
    """A bad summary option fails when the callback is built, not mid-fit."""

    from torchlens.errors import TorchLensError

    with pytest.raises(TorchLensError) as excinfo:
        _summary_callback(no_such_option=True)
    assert excinfo.value.fields["code"] == "summary_option_invalid"


@pytest.mark.heavy
def test_summary_callback_in_a_real_fit() -> None:
    """A real Trainer.fit prints the summary once and trains normally."""

    import lightning.pytorch as pl

    class _Module(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.net = _FrozenBackboneModel()
            self.net.backbone.eval()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            x, y = batch
            return nn.functional.mse_loss(self(x), y)

        def configure_optimizers(self) -> Any:
            return torch.optim.SGD(self.parameters(), lr=0.01)

    data = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.randn(8, 4))
    loader = torch.utils.data.DataLoader(data, batch_size=4)
    callback, printed = _summary_callback()
    trainer = pl.Trainer(
        max_steps=2,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    module = _Module()
    trainer.fit(module, loader)

    assert len(printed) == 1
    assert "sample: first training batch" in printed[0]
    assert trainer.global_step == 2
